"""Regression tests for the sixth review round: the GPU sweep engine.

1. The compiled ``ReduceStep`` closed over ``m``, ``1/m`` and ``excl`` as
   Python constants. ``mx.compile`` writes those into the generated Metal
   source (with ~7 significant digits), so every new window size cost a
   fresh JIT compile (0.2-2.5 s measured) and 1/m was 1-4 float32 ulp off
   (m = 3, 7, 43, 700). They are 0-d array inputs now, with ``float32(1/m)``
   computed once on the host, and no shape-derived literal enters the graph.
2. Exact ties in self-joins went to the lowest column. STUMPY's diagonal
   traversal keeps the nearest-in-time candidate, and the left one on an
   equal offset, so on flatlines ``I_`` pointed thousands of samples back
   and ``left_I_`` at index 0. Every selection (left, right, combined, top-k,
   and the tiled host merges) is now lexicographic in ``(d2, key)`` with
   ``key = 2*|j - i| + (j > i)`` for self-joins and ``key = j`` for AB-joins;
   the compiled fallback selects top-k on one uint64 ``(d2, key)`` word per
   cell, in two stages (per 1024-column chunk, then overall) for wide rows.
3. The compiled reduce step wrote and re-read several ``(B, l)``
   intermediates per batch. One-pass Metal kernels (``_kernels.py``) now
   compute the distances and the ``k == 1`` minima or ``k <= 16`` top-k lists
   in registers; they are bit-identical to the compiled fallback, which stays
   the reference for the CPU device and ``k > 16``.
4. With the fused kernels only QT (4 B per cell) is live, so batches are
   sized with that cell whenever — and only when — the one dispatch
   predicate ``_fused_reducer(k)`` selects them; the fallback keeps its
   re-measured, k-dependent cells. The tiled top-k host merge is charged
   per row.
5. Each batch's query windows are built on the CPU while the previous batch
   runs on the GPU, keeping exactly one live set of device intermediates,
   and the dense loop no longer binds QT to a name through the eval.
6. (Review of 1-5.) The prefetching loops first materialized
   ``list(_batches(...))``: ~170 B per batch of host memory that
   ``estimated_peak_bytes`` does not model (160 MiB at ``l_q = 10**6``
   with ``chunk_size=1``). Two lazy generators, one a step ahead, replace
   it, and the batch bounds are Python ints: a NumPy-integer ``chunk_size``
   made the kernel's parameter array and ``mx.arange`` raise.
7. (Review of 3.) The ``k == 1`` kernel launches 1024 threads per row, and
   MLX rejects a threadgroup wider than the pipeline allows (a limit that
   depends on the GPU and the kernel's register use). The kernels are now
   probed once per process and ``k``: the width is halved until all four
   variants launch, and if none does (or they fail to build) the dispatch
   predicate falls back, with a warning, to the compiled step, which the
   batch sizing then follows.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

import mlx_stump
import mlx_stump._engine as eng
import mlx_stump._kernels as kern
import mlx_stump._stump as stump_mod
from mlx_stump._preprocess import preprocess_series

from .conftest import assert_indices_tie_tolerant, assert_profile_close, tie_tolerance

stumpy = pytest.importorskip("stumpy")

# constant and one-window inputs trigger STUMPY's own advisories
pytestmark = [
    pytest.mark.filterwarnings("ignore:A large number of values in `P`:UserWarning"),
    pytest.mark.filterwarnings("ignore:The window size:UserWarning"),
]

MIB = 1 << 20
FIELDS = ("P_", "I_", "left_I_", "right_I_")
needs_metal = pytest.mark.skipif(not mx.metal.is_available(), reason="requires a Metal GPU")


def _force_fallback(monkeypatch):
    monkeypatch.setattr(eng, "_fused_reducer", lambda k: False)


def _force_tiles(monkeypatch, m, columns):
    monkeypatch.setattr(eng, "_MATMUL_WINDOW_BYTES", 0)
    monkeypatch.setattr(eng, "_TILE_WINDOW_BYTES", columns * m * 4)


def _run(monkeypatch, T, m, T_B=None, *, fused, tiled=False, columns=150, **kwargs):
    """stump() on the fused kernels or the forced compiled fallback; with
    ``fused`` it checks that stump built the fused reducer, so a bit-identity
    test cannot silently compare the fallback with itself."""
    built = []
    with monkeypatch.context() as mp:
        if fused:
            real = stump_mod.make_reducer

            def spy(*a, **kw):
                built.append(kw["fused"])
                return real(*a, **kw)

            mp.setattr(stump_mod, "make_reducer", spy)
        else:
            _force_fallback(mp)
        if tiled:
            _force_tiles(mp, m, columns)
        out = mlx_stump.stump(T, m, T_B, **kwargs)
    assert not fused or (built and all(built)), "fused=True ran the compiled fallback"
    return out


def _assert_same(a, b, label=""):
    for f in FIELDS:
        x = np.asarray(getattr(a, f), dtype=np.float64)
        y = np.asarray(getattr(b, f), dtype=np.float64)
        assert np.array_equal(x, y, equal_nan=True), f"{label} {f} differs"


# ------------------------------------------------------------------ datasets
def _nan_inf_const(n, seed):
    rng = np.random.default_rng(seed)
    T = rng.standard_normal(n).cumsum()
    T[n // 6 : n // 6 + 12] = np.nan
    T[n // 3 : n // 3 + 40] = 3.0
    T[n // 2] = np.inf
    T[(2 * n) // 3 : (2 * n) // 3 + 25] = T[(2 * n) // 3 - 1]  # held value
    return T


def _quantized(n, seed):
    # a coarse integer walk: many exactly tied distances between
    # non-constant windows, plus a flat stretch
    rng = np.random.default_rng(seed)
    T = np.round(rng.standard_normal(n).cumsum() / 2)
    T[n // 4 : n // 4 + 30] = T[n // 4 - 1]
    return T


def _periodic(n, seed):
    # exact repeats of an integer pattern
    rng = np.random.default_rng(seed)
    return np.tile(rng.integers(-5, 6, size=37).astype(float), -(-n // 37))[:n]


BIT_DATASETS = {"nan_inf_const": _nan_inf_const, "quantized": _quantized, "periodic": _periodic}


def _tie_walk(hold: bool):
    """Dyadic random walk with NaN, inf and two constant segments (held
    values when ``hold``: a flatline-dropout series). Multiples of 1/8 keep
    STUMPY's float64 aamp recurrence exact, so its ties are true ties."""
    rng = np.random.default_rng(11)
    T = rng.standard_normal(600).cumsum()
    T[100:110] = np.nan
    if hold:
        T[300:340] = T[299]
        T[450:470] = T[449]
    else:
        T[300:340] = 100.0
        T[450:470] = -100.0
        T[520] = np.inf
    return np.round(T * 8) / 8, 15


def _sine_flatline():
    """Noisy sine with two flatline dropouts that hold the last value."""
    rng = np.random.default_rng(2)
    t = np.arange(2000)
    T = np.sin(2 * np.pi * t / 40) + 0.1 * rng.standard_normal(2000)
    for s in (400, 1300):
        T[s : s + 80] = T[s - 1]
    return T, 20


TIE_DATASETS = {
    "walk_nan_const": lambda: _tie_walk(hold=False),
    "flatline_hold": lambda: _tie_walk(hold=True),
    "all_constant": lambda: (np.full(200, 2.0), 10),
    "sine_flatline": _sine_flatline,
}


# ------------------------------------------------ 1: constants as array inputs
@pytest.mark.parametrize("m", [3, 7, 43])
def test_compiled_step_uses_exact_float32_inverse_m(m):
    """The compiled step's P2 equals the eager distance matrix's row minimum
    bit for bit; with 1/m baked in as a 7-digit literal it differed in
    nearly every row at m=7 and m=43."""
    T = np.random.default_rng(1).standard_normal(600).cumsum()
    A = preprocess_series(T, m)
    engine = eng.MassEngine(A)
    Q = eng.query_windows(A, 0, A.l, normalize=True)
    QT = mx.matmul(Q, engine.W_T)
    ref = mx.min(
        engine.znorm_sq_distances(QT, A.sig_inv_mx, A.isconstant_mx, A.isfinite_mx), axis=1
    )
    consts = eng.reduce_consts(m)
    assert consts[1] == np.float32(1.0 / m)
    step = eng.ReduceStep(A, A, normalize=True, self_join=False, excl=0, k=1, consts=consts)
    assert all(c.ndim == 0 for c in step._consts)
    got = step.full(QT, 0)[1]
    np.testing.assert_array_equal(np.array(got), np.array(ref))
    if mx.metal.is_available():
        fused = kern.FusedReduce(A, A, normalize=True, self_join=False, excl=0, k=1, consts=consts)
        np.testing.assert_array_equal(np.array(fused.full(QT, 0)[1]), np.array(ref))


def test_one_trace_serves_every_window_size():
    """m, 1/m and excl are inputs: re-running one compiled step with another
    window size's constants gives that window size's answer."""
    T = np.random.default_rng(2).standard_normal(400).cumsum()
    preps = {m: preprocess_series(T, m) for m in (9, 13)}
    engines = {m: eng.MassEngine(p) for m, p in preps.items()}
    step = eng.ReduceStep(
        preps[9], preps[9], normalize=True, self_join=True, excl=3, k=1,
        consts=eng.reduce_consts(9),
    )
    for m in (9, 13):
        A, engine = preps[m], engines[m]
        excl = int(np.ceil(m / 4))
        own = eng.ReduceStep(
            A, A, normalize=True, self_join=True, excl=excl, k=1, consts=eng.reduce_consts(m)
        )
        step._q, step._t, step._j_full, step._consts = own._q, own._t, own._j_full, own._consts
        QT = mx.matmul(eng.query_windows(A, 0, A.l, normalize=True), engine.W_T)
        for x, y in zip(step.full(QT, 0), own.full(QT, 0), strict=True):
            np.testing.assert_array_equal(np.array(x), np.array(y))


# ---------------------------------------------------- 2: STUMPY's tie rule
def _golden_ties(monkeypatch, name, normalize, tiled, fused, k=1):
    T, m = TIE_DATASETS[name]()
    ref = stumpy.stump(T, m, k=k) if normalize else stumpy.aamp(T, m, k=k)
    mp = _run(monkeypatch, T, m, fused=fused, tiled=tiled, columns=37, k=k, normalize=normalize)
    return T, m, ref, mp


@pytest.mark.parametrize("fused", [True, False])
@pytest.mark.parametrize("tiled", [False, True])
@pytest.mark.parametrize(
    "name,normalize",
    [
        ("walk_nan_const", True),
        ("walk_nan_const", False),
        ("flatline_hold", True),
        ("flatline_hold", False),
        ("all_constant", True),
        ("all_constant", False),
        ("sine_flatline", True),
    ],
)
def test_self_join_exact_ties_match_stumpy(monkeypatch, name, normalize, tiled, fused):
    """Exact ties resolve like STUMPY: nearest in time, left on an equal
    offset. (A periodic raw-mode series is left out: its float32 near-ties
    are not exact ties for STUMPY either.)"""
    T, m, ref, mp = _golden_ties(monkeypatch, name, normalize, tiled, fused)
    for f in ("I_", "left_I_", "right_I_"):
        np.testing.assert_array_equal(getattr(mp, f), getattr(ref, f), err_msg=f)
    assert_profile_close(mp.P_, ref.P_, m=m, tie_atol=tie_tolerance(m))


@pytest.mark.parametrize("fused", [True, False])
@pytest.mark.parametrize("tiled", [False, True])
@pytest.mark.parametrize("name", ["walk_nan_const", "all_constant", "sine_flatline"])
def test_topk_exact_ties_match_stumpy(monkeypatch, name, tiled, fused):
    """k=3: STUMPY's normalized top-k keeps its first-visited candidates in
    visit order on exact ties, i.e. lexicographic (distance, key)."""
    T, m, ref, mp = _golden_ties(monkeypatch, name, True, tiled, fused, k=3)
    for f in ("I_", "left_I_", "right_I_"):
        np.testing.assert_array_equal(getattr(mp, f), getattr(ref, f), err_msg=f)


@pytest.mark.parametrize("name", ["walk_nan_const", "all_constant"])
def test_raw_topk_ties_select_stumpys_set(monkeypatch, name):
    """stumpy.aamp inserts an exactly tied candidate in front of the equal
    entries already held (reverse traversal order); mlx-stump keeps its one
    nearest-first order. The neighbour sets and left/right indices agree."""
    T, m, ref, mp = _golden_ties(monkeypatch, name, False, False, True, k=3)
    assert all(
        set(a) == set(b) for a, b in zip(np.asarray(mp.I_), np.asarray(ref.I_), strict=True)
    )
    for f in ("left_I_", "right_I_"):
        np.testing.assert_array_equal(getattr(mp, f), getattr(ref, f), err_msg=f)


def test_topk_column_zero_equals_top1_on_exact_ties():
    T, m = _sine_flatline()
    top1 = mlx_stump.stump(T, m)
    top3 = mlx_stump.stump(T, m, k=3)
    np.testing.assert_array_equal(top3.I_[:, 0], top1.I_)


def test_merge_topk_is_lexicographic():
    """The tiled host merge orders by (value, key): offset-first for
    self-joins, lowest column for AB-joins."""
    rng = np.random.default_rng(5)
    rows = np.arange(40, 48)
    k = 4
    vals = rng.integers(0, 3, size=(8, 2 * k)).astype(np.float32)
    idxs = np.stack([rng.choice(np.arange(0, 100), 2 * k, replace=False) for _ in rows])
    for self_join in (True, False):
        key = stump_mod._tie_keys(idxs, rows, self_join)
        order = np.lexsort((key, vals), axis=1)
        # split into two sorted halves, as the sweep delivers them
        a, b = order[:, 0::2], order[:, 1::2]
        va, ia = np.take_along_axis(vals, a, 1), np.take_along_axis(idxs, a, 1)
        vb, ib = np.take_along_axis(vals, b, 1), np.take_along_axis(idxs, b, 1)
        v, i = stump_mod._merge_topk(va, ia, vb, ib, k, rows if self_join else None)
        np.testing.assert_array_equal(i, np.take_along_axis(idxs, order[:, :k], 1))
        np.testing.assert_array_equal(v, np.take_along_axis(vals, order[:, :k], 1))
    off = np.array([[-3, 3, -2, 2]]) + 10
    np.testing.assert_array_equal(
        stump_mod._tie_keys(off, np.array([10]), True), [[6, 7, 4, 5]]
    )


# ------------------------------------ 3: fused kernels == compiled fallback
@needs_metal
@pytest.mark.parametrize("m", [3, 4, 7, 17, 64])
@pytest.mark.parametrize("dataset", sorted(BIT_DATASETS))
def test_fused_kernels_bit_identical_to_fallback(monkeypatch, dataset, m):
    """Full stump output, bit for bit, across both distance modes, both join
    types, k in {1, 2, 5, 16} (dense) and k in {1, 5} (tiled)."""
    assert eng._fused_reducer(1) and eng._fused_reducer(16)
    T = BIT_DATASETS[dataset](500, 1)
    T_B = BIT_DATASETS[dataset](420, 2)
    for normalize in (True, False):
        for tb in (None, T_B):
            for k in (1, 2, 5, 16):
                for tiled in (False, True) if k in (1, 5) else (False,):
                    kw = dict(k=k, normalize=normalize, tiled=tiled, ignore_trivial=tb is None)
                    a = _run(monkeypatch, T, m, tb, fused=True, **kw)
                    b = _run(monkeypatch, T, m, tb, fused=False, **kw)
                    _assert_same(a, b, f"normalize={normalize} ab={tb is not None} {kw}")


@needs_metal
@pytest.mark.parametrize("tiled", [False, True])
def test_fused_bit_identical_large_window(monkeypatch, tiled):
    T = _nan_inf_const(1500, 3)
    for normalize in (True, False):
        for k in (1, 5):
            kw = dict(k=k, normalize=normalize, tiled=tiled, columns=300)
            _assert_same(
                _run(monkeypatch, T, 333, fused=True, **kw),
                _run(monkeypatch, T, 333, fused=False, **kw),
                f"m=333 {kw}",
            )


@needs_metal
def test_fused_bit_identical_with_user_and_asymmetric_flags(monkeypatch):
    rng = np.random.default_rng(4)
    T = rng.standard_normal(400).cumsum()
    T_B = rng.standard_normal(350).cumsum()
    m = 12
    fa = np.zeros(len(T) - m + 1, bool)
    fa[::7] = True
    fb = np.zeros(len(T_B) - m + 1, bool)
    fb[3::5] = True
    fs = np.zeros(len(T) - m + 1, bool)
    fs[1::9] = True
    cases = [
        (T, T_B, dict(ignore_trivial=False, T_A_subseq_isconstant=fa, T_B_subseq_isconstant=fb)),
        (T, None, dict(T_A_subseq_isconstant=fa)),
        # self-join through an explicit T_B with different flags per side
        (T, T.copy(), dict(T_A_subseq_isconstant=fa, T_B_subseq_isconstant=fs)),
    ]
    for ta, tb, kw in cases:
        for k in (1, 3):
            for tiled in (False, True):
                args = dict(kw, k=k, tiled=tiled, columns=64)
                _assert_same(
                    _run(monkeypatch, ta, m, tb, fused=True, **args),
                    _run(monkeypatch, ta, m, tb, fused=False, **args),
                    f"{sorted(kw)} k={k} tiled={tiled}",
                )


@needs_metal
@pytest.mark.parametrize("normalize", [True, False])
def test_fused_single_window_inputs(monkeypatch, normalize):
    """l == 1 on either side: size-1 device inputs are bound in Metal's
    constant address space and must still compile and match."""
    rng = np.random.default_rng(6)
    m = 8
    one = rng.standard_normal(m)
    longer = rng.standard_normal(40).cumsum()
    cases = [(one, None), (one, longer), (longer, one)]
    for ta, tb in cases:
        for k in (1, 2):
            kw = dict(k=k, normalize=normalize, ignore_trivial=tb is None)
            a = _run(monkeypatch, ta, m, tb, fused=True, **kw)
            b = _run(monkeypatch, ta, m, tb, fused=False, **kw)
            _assert_same(a, b, f"lA={len(ta)} lB={None if tb is None else len(tb)} k={k}")


@needs_metal
@pytest.mark.parametrize("normalize", [True, False])
def test_fused_bit_identical_on_rows_wider_than_a_chunk(monkeypatch, normalize):
    """Targets wider than ``_TOPK_CHUNK`` take the fallback's two-stage
    top-k selection; it still matches the kernel bit for bit."""
    T = _nan_inf_const(3000, 12)
    T_B = _quantized(2600, 13)
    for tb in (None, T_B):
        for k in (2, 5, 16):
            kw = dict(k=k, normalize=normalize, ignore_trivial=tb is None)
            _assert_same(
                _run(monkeypatch, T, 20, tb, fused=True, **kw),
                _run(monkeypatch, T, 20, tb, fused=False, **kw),
                f"ab={tb is not None} k={k}",
            )


@pytest.mark.parametrize("tiled", [False, True])
def test_chunked_topk_selection_is_exact(monkeypatch, tiled):
    """The two-stage selection returns exactly the single-stage one, ties
    included (quantized data), for k up to the fallback's range."""
    _force_fallback(monkeypatch)
    T = _quantized(3000, 14)
    T_B = _nan_inf_const(2500, 15)
    for normalize in (True, False):
        for tb in (None, T_B):
            for k in (5, 17, 100):
                kw = dict(k=k, normalize=normalize, tiled=tiled, columns=1500,
                          ignore_trivial=tb is None)
                chunked = _run(monkeypatch, T, 16, tb, fused=False, **kw)
                with monkeypatch.context() as mp:
                    mp.setattr(eng, "_TOPK_CHUNK", 1 << 30)
                    single = _run(monkeypatch, T, 16, tb, fused=False, **kw)
                _assert_same(chunked, single, f"normalize={normalize} ab={tb is not None} k={k}")


@pytest.mark.parametrize("k", [17, 100, 128, 200])
def test_large_k_runs_on_the_fallback(k):
    """k beyond the top-k kernel's range (and beyond the 32 KiB threadgroup
    limit the prototype hit at k=128) uses the compiled fallback."""
    assert not eng._fused_reducer(k)
    T = np.random.default_rng(7).standard_normal(400).cumsum()
    m = 10
    mp = mlx_stump.stump(T, m, k=k)
    ref = stumpy.stump(T, m, k=k)
    assert mp.shape == (len(T) - m + 1, 2 * k + 2)
    tie = tie_tolerance(m)
    np.testing.assert_allclose(
        np.asarray(mp.P_, dtype=float), np.asarray(ref.P_, dtype=float), atol=tie
    )
    assert_indices_tie_tolerant(mp.left_I_, ref.left_I_, T, T, m, tie_atol=tie)


def test_cpu_device_uses_the_fallback():
    T = np.random.default_rng(8).standard_normal(300).cumsum()
    m = 12
    with mx.stream(mx.cpu):
        assert not eng._fused_reducer(1)
        mp = mlx_stump.stump(T, m)
    ref = stumpy.stump(T, m)
    tie = tie_tolerance(m)
    assert_indices_tie_tolerant(mp.I_, ref.I_, T, T, m, tie_atol=tie)
    assert_profile_close(mp.P_, ref.P_, m=m, tie_atol=tie)


def test_topk_kernel_fits_threadgroup_memory():
    for k in range(2, kern.FUSED_TOPK_MAX + 1):
        assert kern.topk_threadgroup_bytes(k) <= 32 * 1024
    T = np.random.default_rng(9).standard_normal(64).cumsum()
    A = preprocess_series(T, 8)
    with pytest.raises(ValueError, match="1 <= k"):
        kern.FusedReduce(
            A, A, normalize=True, self_join=True, excl=2, k=kern.FUSED_TOPK_MAX + 1,
            consts=eng.reduce_consts(8),
        )


# ------------------------------------------------ 4: batches follow dispatch
class _FakeEngine:
    def __init__(self, l, m, tile_rows=None):
        self.l, self.m = l, m
        self.tile_rows = l if tile_rows is None else tile_rows


def test_sizing_follows_the_dispatch_predicate(monkeypatch):
    engine = _FakeEngine(131_023, 50)
    for k in (1, 5, 17):
        fused = eng._fused_reducer(k)
        assert eng.default_chunk_size(engine, engine.l, k, True) == eng.default_chunk_size(
            engine, engine.l, k, True, fused=fused
        )
        assert eng.estimated_peak_bytes(engine.l, 50, k) == eng.estimated_peak_bytes(
            engine.l, 50, k, fused=fused
        )
    # the fused batch is 4x (k=1) / 14x (top-k) the fallback's here
    assert eng.default_chunk_size(engine, engine.l, 1, True, fused=True) == 766
    assert eng.default_chunk_size(engine, engine.l, 1, True, fused=False) == 191
    assert eng.default_chunk_size(engine, engine.l, 5, True, fused=False) == 54
    # a monkeypatched predicate reaches the sizing too
    monkeypatch.setattr(eng, "_fused_reducer", lambda k: False)
    assert eng.default_chunk_size(engine, engine.l, 1, True) == 191
    with pytest.raises(ValueError, match="`fused`"):
        eng.estimated_peak_bytes(100, 10, fused="yes")


def test_tiled_topk_charges_host_merge_per_row():
    engine = _FakeEngine(10_000_000, 50, tile_rows=_tile_rows(50))
    k = 16
    b = eng.tiled_chunk_size(engine, engine.l, k, False, fused=True)
    per_row = engine.tile_rows * 4 + (8 * k + 16) + 50 * 24 + eng._CENTER_ROW_BYTES
    merge = k * eng._TILED_MERGE_CELL
    assert eng._TILED_MERGE_CELL >= 85  # measured merge peak at k=2
    assert b * (per_row + merge) <= eng._CHUNK_MEM_BUDGET
    assert (b + 1) * (per_row + merge) > eng._CHUNK_MEM_BUDGET


def _tile_rows(m):
    return max(4, eng._TILE_WINDOW_BYTES // (4 * m))


def _peak(fn):
    mx.synchronize()
    mx.clear_cache()
    mx.reset_peak_memory()
    fn()
    return mx.get_peak_memory()


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize(
    "n,k,normalize,fused",
    [
        (131_072, 1, True, True),  # fused k=1: the budget binds (766 rows)
        (65_536, 5, True, True),  # fused top-k
        (32_768, 1, False, False),  # fallback k=1, raw
        (32_768, 5, True, False),  # fallback top-k (two-stage selection)
        (32_768, 5, False, False),  # fallback top-k, raw
        (32_768, 300, True, False),  # fallback top-k (single-stage selection)
    ],
)
def test_measured_sweep_peak_within_budget(monkeypatch, n, k, normalize, fused):
    """The automatic batch keeps MLX's peak at or below the resident block
    plus the intermediates budget in every dispatch case. A fused-path
    (4 B/cell) batch leaking into the fallback peaked at 2.6 GiB. The k == 1
    sweeps fill the budget exactly, so the slack covers the O(l) per-series
    device arrays that come on top (<= 0.75 MiB here) and allocator
    variance."""
    m = 50
    T = np.random.default_rng(10).standard_normal(n).cumsum()
    l = n - m + 1
    if not fused:
        _force_fallback(monkeypatch)
    assert eng._fused_reducer(k) == fused
    peak = _peak(lambda: mlx_stump.stump(T, m, k=k, normalize=normalize))
    limit = eng.resident_block_bytes(l, m) + eng._CHUNK_MEM_BUDGET + 16 * MIB
    assert peak <= limit, f"peak {peak / MIB:.0f} MiB > {limit / MIB:.0f} MiB"


# --------------------------------------------- 5: prefetch, one live set
@pytest.mark.gpu
@pytest.mark.parametrize("tiled", [False, True])
def test_prefetch_keeps_one_live_batch(monkeypatch, tiled):
    """Building the next query batch during the current one must not start
    the next batch's device work: the peak stays one QT batch."""
    n, m, B = 32_768, 50, 512
    T = np.random.default_rng(11).standard_normal(n).cumsum()
    l = n - m + 1
    if tiled:
        _force_tiles(monkeypatch, m, 8_192)
    assert eng._fused_reducer(1)
    width = eng.resident_block_bytes(l, m) // (m * 4) if not tiled else 8_192
    block = width * m * 4
    peak = _peak(lambda: mlx_stump.stump(T, m, chunk_size=B))
    one_set = B * width * 4
    assert peak <= block + one_set + 16 * MIB, f"peak {peak / MIB:.0f} MiB"


# ------------------------------------------- 6: lazy, integer batch bounds
def test_batches_are_lazy_python_ints():
    gen = stump_mod._batches(np.int64(10), np.int64(4))
    assert iter(gen) is gen  # a generator, not a list
    batches = list(gen)
    assert batches == [(0, 0, 4), (4, 4, 8), (6, 8, 10)]
    assert all(type(x) is int for b in batches for x in b)


@pytest.mark.parametrize("tiled", [False, True])
def test_sweep_pulls_batches_lazily(monkeypatch, tiled):
    """At every reducer call the sweep has taken at most two batches (its
    own and the prefetched query's) per reduction so far: nothing is
    materialized ahead."""
    events = {"pulled": 0, "reduced": 0, "ahead": 0}
    real_batches = stump_mod._batches
    real_make = stump_mod.make_reducer

    def counting(l_q, B):
        for b in real_batches(l_q, B):
            events["pulled"] += 1
            yield b

    def counted(reduce):
        def wrapped(*a):
            events["reduced"] += 1
            events["ahead"] = max(events["ahead"], events["pulled"] - 2 * events["reduced"])
            return reduce(*a)

        return wrapped

    def make(*args, **kwargs):
        red = real_make(*args, **kwargs)
        red.full, red.block = counted(red.full), counted(red.block)
        return red

    monkeypatch.setattr(stump_mod, "_batches", counting)
    monkeypatch.setattr(stump_mod, "make_reducer", make)
    T = np.random.default_rng(12).standard_normal(400).cumsum()
    m = 10
    if tiled:
        _force_tiles(monkeypatch, m, 100)
    for k in (1, 3):
        mlx_stump.stump(T, m, k=k, chunk_size=7)
    assert events["reduced"] >= 2 * 56
    assert events["ahead"] <= 0


def test_numpy_integer_arguments_run(monkeypatch):
    """NumPy-integer m / k / chunk_size give the Python-int result on the
    fused kernels (k <= 16) and on the compiled fallback."""
    T = np.random.default_rng(13).standard_normal(500).cumsum()
    for k in (1, 5, 20):
        for fused in (True, False) if k <= kern.FUSED_TOPK_MAX else (False,):
            with monkeypatch.context() as mp:
                if not fused:
                    _force_fallback(mp)
                a = mlx_stump.stump(T, np.int64(20), k=np.int64(k), chunk_size=np.int64(33))
                b = mlx_stump.stump(T, 20, k=k, chunk_size=33)
            _assert_same(a, b, f"k={k} fused={fused}")


# ----------------------------------------- 7: the kernels are launch-probed
@needs_metal
def test_every_fused_k_launches_here():
    """On this GPU the probe accepts every fused k (otherwise the fused vs
    fallback bit-identity tests would compare the fallback with itself)."""
    for k in range(1, kern.FUSED_TOPK_MAX + 1):
        assert eng._fused_reducer(k)
        tg = kern.launch_threadgroup(k)
        preferred = kern._ARGMIN_TG if k == 1 else kern.topk_threadgroup(k)
        assert tg is not None and tg <= preferred and preferred % tg == 0


@needs_metal
def test_threadgroup_wider_than_the_pipeline_is_halved(monkeypatch):
    """A preferred width no Apple GPU pipeline accepts (4096 threads) is
    halved until the kernels launch; the output is unchanged."""
    monkeypatch.setattr(kern, "_LAUNCH", {})
    monkeypatch.setattr(kern, "_ARGMIN_TG", 4096)
    tg = kern.launch_threadgroup(1)
    assert tg is not None and tg < 4096 and 4096 % tg == 0
    T = _quantized(600, 16)
    for normalize in (True, False):
        for tiled in (False, True):
            kw = dict(normalize=normalize, tiled=tiled, columns=200)
            _assert_same(
                _run(monkeypatch, T, 12, fused=True, **kw),
                _run(monkeypatch, T, 12, fused=False, **kw),
                f"normalize={normalize} tiled={tiled}",
            )


@needs_metal
def test_kernels_that_cannot_launch_fall_back(monkeypatch):
    """If no width launches (or the kernels fail to build), the predicate
    turns False with a warning; dispatch and batch sizing both follow it."""
    monkeypatch.setattr(kern, "_LAUNCH", {})

    def broken(k, tg):
        raise RuntimeError("simulated pipeline failure")

    monkeypatch.setattr(kern, "_probe", broken)
    with pytest.warns(RuntimeWarning, match="compiled reduction") as rec:
        assert not eng._fused_reducer(5)
    # attributed to this caller, not to a line inside the package
    assert {w.filename for w in rec if "compiled reduction" in str(w.message)} == {__file__}
    assert not eng._fused_reducer(5)  # cached: one warning per process and k
    engine = _FakeEngine(131_023, 50)
    assert eng.default_chunk_size(engine, engine.l, 5, True) == eng.default_chunk_size(
        engine, engine.l, 5, True, fused=False
    )
    A = preprocess_series(np.random.default_rng(14).standard_normal(64).cumsum(), 8)
    with pytest.raises(RuntimeError, match="cannot run"):
        kern.FusedReduce(
            A, A, normalize=True, self_join=True, excl=2, k=5, consts=eng.reduce_consts(8)
        )
    T = _nan_inf_const(500, 17)
    got = mlx_stump.stump(T, 15, k=5)
    ref = _run(monkeypatch, T, 15, fused=False, k=5)
    _assert_same(got, ref, "probe failure")
