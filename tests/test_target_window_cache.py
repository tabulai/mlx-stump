"""The z-normalized refinement's cache of normalized target windows.

``_refine_znorm`` may normalize every target window once, in a first pass on
the refinement threads, instead of once per (row, neighbor) pair:

- the cached and per-column routes give the same bits, with NaN/inf and
  constant windows, user constant flags, extreme scales, rows and columns
  without a neighbor, 1-D indices, AB- and self-joins, on 1 and 8 threads;
- the cache is charged to the refinement budget: the float64 work of one call
  stays within ``_REFINE_MEM_BUDGET`` even for an AB-join with partially
  filled rows (a cache built on top of the lanes' budget reached ~5 of its 4
  window blocks), and where the 65536-row cap keeps the four blocks far
  below that budget (small ``m``) the lanes keep full chunks, within
  ``refine_peak_bytes(m)``, the estimate's refinement term;
- it is built only where it saves normalizations: not for an AB-join of
  equal lengths at k = 1, nor for a self-join whose rows mostly have no
  neighbor, nor beyond two budgeted blocks of target windows;
- a failure in either pass raises in the caller, stops the other threads and
  never starts the second pass early; ``_map_refine_chunks`` runs both passes
  in one pool, the second after every chunk of the first.
"""

from __future__ import annotations

import threading
import tracemalloc
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import mlx_stump._engine as eng
import mlx_stump._stump as st
from mlx_stump._preprocess import preprocess_series


def _walk(n, seed):
    return np.random.default_rng(seed).standard_normal(n).cumsum()


def _neighbors(rng, l_q, l_t, k, holes=0.0, empty_every=0):
    I = rng.integers(0, l_t, size=(l_q, k))
    I[rng.random(I.shape) < holes] = -1
    if empty_every:
        I[::empty_every] = -1
    return I


def _budget(monkeypatch, rows, m, workers=8):
    monkeypatch.setattr(eng, "_REFINE_MEM_BUDGET", rows * m * 8 * 4)
    assert eng.refine_chunk_rows(m) == rows
    monkeypatch.setattr(st, "_REFINE_MAX_WORKERS", workers)
    monkeypatch.setattr(st, "_cpu_count", lambda: 8)


def _spy_cache(monkeypatch):
    used = []
    real = st._cache_target_windows

    def spy(*args):
        used.append(real(*args))
        return used[-1]

    monkeypatch.setattr(st, "_cache_target_windows", spy)
    return used


def _case(name):
    """(query, target, I) with target.l in the cache range of a 600-row budget."""
    m = 16
    rng = np.random.default_rng(len(name))
    T = _walk(1000 + m - 1, 11)
    if name == "self_clean":
        A = preprocess_series(T, m)
        return A, A, _neighbors(rng, A.l, A.l, 3)
    if name == "self_special":
        T[100:140] = 2.5  # constant run
        T[300:303] = np.nan
        T[700] = np.inf
        A = preprocess_series(T, m)
        return A, A, _neighbors(rng, A.l, A.l, 3, holes=0.05, empty_every=17)
    if name == "self_user_flags":
        flags = np.zeros(len(T) - m + 1, dtype=bool)
        flags[::9] = True
        A = preprocess_series(T, m, isconstant=flags)
        return A, A, _neighbors(rng, A.l, A.l, 2, holes=0.02)
    if name == "self_extreme_scales":
        T[:500] *= 1e-300
        T[500:] = T[500:] * 1e250 + 1e300
        A = preprocess_series(T, m)
        return A, A, _neighbors(rng, A.l, A.l, 2)
    if name == "ab_special":
        Q = _walk(4000, 12)
        Q[1000:1010] = np.nan
        Q[2000:2200] = -3.0
        T[400:430] = 7.0
        T[800] = -np.inf
        A, B = preprocess_series(Q, m), preprocess_series(T, m)
        return A, B, _neighbors(rng, A.l, B.l, 2, holes=0.05, empty_every=13)
    if name == "ab_offset":
        # >= 3 query rows per target window: the cache pays off at k = 1 too
        A = preprocess_series(_walk(4000, 13) * 1e-3 + 1e8, m)
        B = preprocess_series(T * 1e-3 + 1e8, m)
        return A, B, _neighbors(rng, A.l, B.l, 4)
    raise AssertionError(name)


@pytest.mark.parametrize("workers", [1, 8])
@pytest.mark.parametrize(
    "name",
    [
        "self_clean",
        "self_special",
        "self_user_flags",
        "self_extreme_scales",
        "ab_special",
        "ab_offset",
    ],
)
def test_cached_and_per_column_refinement_are_bit_identical(monkeypatch, name, workers):
    A, B, I = _case(name)
    _budget(monkeypatch, 600, A.m, workers)
    with np.errstate(all="ignore"):
        for idx in (I, I[:, 0]):
            with monkeypatch.context() as mp:
                used = _spy_cache(mp)
                cached = st._refine_znorm(A, B, idx)
            assert used == [True]
            with monkeypatch.context() as mp:
                mp.setattr(st, "_cache_target_windows", lambda *a: False)
                direct = st._refine_znorm(A, B, idx)
            np.testing.assert_array_equal(cached.view(np.uint64), direct.view(np.uint64))
    assert np.isfinite(cached).any()


@pytest.mark.parametrize("join", ["self", "ab"])
def test_cache_stays_within_the_refinement_budget(monkeypatch, join):
    m, rows = 128, 1024  # four 1 MiB window blocks
    _budget(monkeypatch, rows, m)
    rng = np.random.default_rng(5)
    B = preprocess_series(_walk(2000 + m - 1, 21), m)  # ~1.95 blocks of cache
    A = B if join == "self" else preprocess_series(_walk(8000 + m - 1, 22), m)
    # holes in every column, and empty rows: the masked-copy paths
    I = _neighbors(rng, A.l, B.l, 2, holes=0.03, empty_every=97)
    out = np.empty(I.shape)
    used = _spy_cache(monkeypatch)
    tracemalloc.start()
    try:
        st._refine_znorm(A, B, I, out=out)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert used == [True]
    assert peak <= eng._REFINE_MEM_BUDGET, (peak / 2**20, eng._REFINE_MEM_BUDGET / 2**20)
    assert eng.refine_peak_bytes(m) == eng._REFINE_MEM_BUDGET


def test_small_windows_keep_full_chunks_with_the_cache(monkeypatch):
    m = 8  # four blocks of the 65536-row cap are 16 MiB of the 256 MiB budget
    monkeypatch.setattr(st, "_cpu_count", lambda: 8)
    rows = eng.refine_chunk_rows(m)
    assert rows * m * 8 * 4 < eng._REFINE_MEM_BUDGET
    B = preprocess_series(_walk(2 * rows + m - 1, 23), m)  # the largest cache
    A = preprocess_series(_walk(140_000 + m - 1, 24), m)
    # an AB-join with holes and empty rows: the lanes' worst case
    I = _neighbors(np.random.default_rng(6), A.l, B.l, 2, holes=0.03, empty_every=97)
    chunks = []
    real = st._map_refine_chunks

    def spy(job, l, m_, **kwargs):
        chunks.append(kwargs["rows"])
        return real(job, l, m_, **kwargs)

    monkeypatch.setattr(st, "_map_refine_chunks", spy)
    out = np.empty(I.shape)
    tracemalloc.start()
    try:
        st._refine_znorm(A, B, I, out=out)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert chunks == [rows]  # not shrunk below the uncached chunk
    assert rows * m * 8 * 4 < peak <= eng.refine_peak_bytes(m) <= eng._REFINE_MEM_BUDGET


def test_cache_is_built_only_where_it_saves_normalizations():
    rng = np.random.default_rng(3)
    budget_rows, m = 65536, 100
    l = 10_000

    def decide(I, target_l, shared):
        return st._cache_target_windows(I, target_l, shared, budget_rows, m)

    full_k1 = _neighbors(rng, l, l, 1)
    assert decide(full_k1, l, True)  # self-join k=1: half the normalizations
    assert not decide(full_k1, l, False)  # AB of equal lengths at k=1: none saved
    assert decide(_neighbors(rng, l, l, 4), l, False)
    assert decide(_neighbors(rng, 65_536, 2_000, 1), 2_000, False)  # narrow target
    mostly_empty = _neighbors(rng, l, l, 1)
    mostly_empty[rng.random(l) < 0.9] = -1
    assert not decide(mostly_empty, l, True)
    assert not decide(_neighbors(rng, l, 511, 4), 511, False)  # below the floor
    big = 2 * budget_rows + 1  # beyond two budgeted blocks
    assert not decide(_neighbors(rng, l, big, 16), big, False)


@pytest.mark.parametrize("failing_pass", ["prepare", "refine"])
def test_failure_in_either_pass_stops_every_thread(monkeypatch, failing_pass):
    A, B, I = _case("ab_special")
    _budget(monkeypatch, 600, A.m)
    real = st._znorm_rows
    lock = threading.Lock()
    seen = {"prepare": 0, "refine": 0}

    def flaky(w, isconstant, m):
        # the first pass normalizes slices of the cache, the second copies
        which = "prepare" if w.base is not None else "refine"
        with lock:
            seen[which] += 1
            n = seen[which]
        if which == failing_pass and n == 2:
            raise MemoryError("synthetic")
        return real(w, isconstant, m)

    monkeypatch.setattr(st, "_znorm_rows", flaky)
    threads = threading.active_count()
    with np.errstate(all="ignore"), pytest.raises(MemoryError, match="synthetic"):
        st._refine_znorm(A, B, I)
    assert threading.active_count() == threads
    if failing_pass == "prepare":
        assert seen["refine"] == 0  # the barrier held the second pass back


class _CountingPool(ThreadPoolExecutor):
    made = 0

    def __init__(self, *args, **kwargs):
        type(self).made += 1
        super().__init__(*args, **kwargs)


def test_map_refine_chunks_runs_both_passes_in_one_pool(monkeypatch):
    m = 10
    _budget(monkeypatch, 64, m)
    monkeypatch.setattr(st, "ThreadPoolExecutor", _CountingPool)
    _CountingPool.made = 0
    lock = threading.Lock()
    events = []
    live = [0, 0]

    def tracked(tag):
        def job(s, e):
            with lock:
                events.append((tag, "start", s, e))
                live[0] += e - s
                live[1] = max(live[1], live[0])
            np.sqrt(np.arange(2000.0))  # release the GIL briefly
            with lock:
                live[0] -= e - s
                events.append((tag, "end", s, e))

        return job

    st._map_refine_chunks(tracked("job"), 301, m, rows=16, prepare=(tracked("prep"), 203))
    assert _CountingPool.made == 1
    last_prep = max(i for i, ev in enumerate(events) if ev[0] == "prep")
    first_job = min(i for i, ev in enumerate(events) if ev[0] == "job")
    assert last_prep < first_job
    for tag, n in (("prep", 203), ("job", 301)):
        spans = sorted((s, e) for t, kind, s, e in events if t == tag and kind == "start")
        assert spans == [(s, min(s + 2, n)) for s in range(0, n, 2)]
    assert live[1] <= 16  # the rows override bounds the rows in flight
