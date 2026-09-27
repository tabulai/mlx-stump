"""Dense near-zero repair reuses exact float64 work without changing results."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import mlx_stump._engine as engine
import mlx_stump._match as match
import mlx_stump._stump as stump


@pytest.mark.parametrize("m", [3, 64, 257])
def test_normalized_window_cache_preserves_affine_and_one_ulp_distances(m):
    n = 4 * m + 80
    T = np.random.default_rng(m).integers(-100, 101, n).astype(np.float64)
    q_idx, exact_idx, near_idx, flagged_idx = 2, m + 12, 2 * m + 24, 3 * m + 36
    Q = np.arange(1, m + 1, dtype=np.float64)
    T[q_idx : q_idx + m] = Q
    T[exact_idx : exact_idx + m] = 3 * Q + 5
    T[near_idx : near_idx + m] = 3 * Q + 5
    T[near_idx + m // 2] = np.nextafter(T[near_idx + m // 2], np.inf)
    flags = np.zeros(n - m + 1, dtype=bool)
    flags[flagged_idx] = True
    js = np.array([exact_idx, near_idx, 0, flagged_idx], dtype=np.int64)
    cache = match._NormalizedWindowCache(T, m, flags)

    # Exercise both an implicit self-join query, which can reuse its own
    # normalized row, and an independent AB query, which cannot.
    for query in (T[q_idx : q_idx + m], Q.copy()):
        ordinary = match._refine_candidates(query, T, js, True, False, flags)
        cached = match._refine_candidates(
            query, T, js, True, False, flags, normalized_cache=cache
        )
        np.testing.assert_array_equal(cached, ordinary)
        assert cached[0] == 0.0
        assert cached[1] > 0.0


def test_normalized_cache_build_obeys_limit_and_strict_error_policy(monkeypatch):
    base = np.float64(1e250)
    ulp = np.spacing(base)
    T = base + ulp * np.resize(np.arange(32, dtype=np.float64), 320)
    m = 32
    flags = np.zeros(T.size - m + 1, dtype=bool)
    with np.errstate(all="raise"):
        cache = match._NormalizedWindowCache(T, m, flags)
    assert np.all(np.isfinite(cache.U))

    monkeypatch.setattr(match, "_REPAIR_NORMALIZED_CACHE_MAX_BYTES", 100)
    with pytest.raises(ValueError, match="memory limit"):
        match._NormalizedWindowCache(T, m, flags)


def _near_periodic(n=512, m=32):
    rng = np.random.default_rng(123)
    return np.resize(rng.normal(size=m), n) + rng.normal(scale=1e-7, size=n), m


def test_dense_self_join_uses_cache_and_preserves_full_profile(monkeypatch):
    T, m = _near_periodic()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        monkeypatch.setattr(engine, "_REPAIR_NORMALIZED_CACHE_MAX_BYTES", 0)
        ordinary = stump.stump(T, m)
        monkeypatch.setattr(engine, "_REPAIR_NORMALIZED_CACHE_MAX_BYTES", 64 << 20)
        builds = []
        real = match._NormalizedWindowCache

        class Spy(real):
            def __init__(self, *args):
                builds.append(1)
                super().__init__(*args)

        monkeypatch.setattr(match, "_NormalizedWindowCache", Spy)
        cached = stump.stump(T, m)
    assert builds == [1]
    np.testing.assert_array_equal(cached.P_.view(np.uint64), ordinary.P_.view(np.uint64))
    for name in ("I_", "left_I_", "right_I_"):
        np.testing.assert_array_equal(getattr(cached, name), getattr(ordinary, name))


def test_repair_refines_each_seed_query_once_before_rescan(monkeypatch):
    T, m = _near_periodic()
    rescanning = False
    seed_queries = []
    real_blocks = engine.MassEngine.target_blocks
    real_refine = match._refine_candidates

    def blocks(self):
        nonlocal rescanning
        rescanning = True
        yield from real_blocks(self)

    def refine(Q, *args, **kwargs):
        if not rescanning:
            seed_queries.append(Q.ctypes.data)
        return real_refine(Q, *args, **kwargs)

    monkeypatch.setattr(engine.MassEngine, "target_blocks", blocks)
    monkeypatch.setattr(match, "_refine_candidates", refine)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        stump.stump(T, m)
    assert seed_queries
    assert len(seed_queries) == len(set(seed_queries))


@pytest.mark.parametrize("mode", ["raw", "topk", "ab", "random"])
def test_normalized_target_cache_stays_off_outside_dense_k1_self_join(monkeypatch, mode):
    T, m = _near_periodic()

    def unexpected(*args):
        raise AssertionError("normalized target cache should stay off")

    monkeypatch.setattr(match, "_NormalizedWindowCache", unexpected)
    kwargs = {}
    if mode == "raw":
        kwargs["normalize"] = False
    elif mode == "topk":
        kwargs["k"] = 2
    elif mode == "ab":
        kwargs.update(T_B=T[-2 * m :].copy(), ignore_trivial=False)
    else:
        T = np.random.default_rng(8).normal(size=T.size)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        profile = stump.stump(T, m, **kwargs)
    assert profile.shape[0] == T.size - m + 1
