"""Regression tests for defects found in the second (external) review.

The review's headline findings, each reproduced before fixing:
1. float64 refinement used `dot - m*mu_q*mu_t`, which cancels catastrophically
   for near-constant windows at an offset (reported distances of 800+ where
   the truth was 0, above the theoretical z-norm maximum of 2*sqrt(m));
2. the large-input FFT fallback skipped target-window centering and was
   badly wrong on near-constant data (3.3% index agreement) — replaced by
   the tiled matmul engine, tested in test_stump_golden.py;
3. `match` mishandled max_distance=inf with NaN windows, resolved its default
   threshold on the unrefined float32 profile, dropped isconstant overrides
   during refinement, and accepted negative query_idx;
4. the documented per-chunk memory budget was not enforced (16-row floor,
   no top-k accounting).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import stumpy

import mlx_stump
from mlx_stump._engine import _CHUNK_MEM_BUDGET, default_chunk_size, tiled_chunk_size

from .conftest import tie_tolerance


# ------------------------------------------------- 1: refinement cancellation
def test_refine_no_cancellation_on_offset_flatlines():
    """Duplicate near-constant segments at a large offset: the true z-norm
    distance between corresponding windows is exactly 0, and no z-norm
    distance can exceed 2*sqrt(m). The old single-pass covariance
    (`dot - m*mu_q*mu_t`) reported values in the hundreds here."""
    rng = np.random.default_rng(0)
    T = rng.standard_normal(3000)
    jit = 1e-4 * rng.standard_normal(120)
    T[500:620] = 1e6 + jit
    T[2000:2120] = 1e6 + jit  # exact duplicate of the first flat segment
    m = 64
    mp = mlx_stump.stump(T, m)

    finite = np.isfinite(mp.P_)
    assert np.all(mp.P_[finite] <= 2.0 * np.sqrt(m) + 1e-6)

    # windows fully inside the first segment have an exact duplicate 1500
    # later, so their true profile value is 0; the float32 search may settle
    # on an equally-flat near-tie neighbor ~1e-7 away (the old cancellation
    # bug reported >1100 here)
    rows = np.arange(500, 620 - m + 1)
    assert np.all(mp.P_[rows] <= 1e-5)
    exact = mp.I_[rows] == rows + 1500
    np.testing.assert_array_equal(mp.P_[rows][exact], 0.0)
    assert exact.mean() > 0.3


def test_refine_matches_bruteforce_on_near_duplicate_flatlines():
    """Near- (not exactly-) duplicate flat segments: reported P must equal a
    doubly-centered float64 brute force at the chosen index."""
    rng = np.random.default_rng(1)
    T = rng.standard_normal(3000)
    # sigma/mu of 1e-8 leaves ~7.5 float64 digits of jitter (larger offsets
    # push the data itself to the representation limit, where "truth" is
    # undefined at this tolerance) while still making the old formula's
    # cancellation error O(1) in rho
    T[500:620] = 1e3 + 1e-5 * rng.standard_normal(120)
    T[2000:2120] = 1e3 + 1e-5 * rng.standard_normal(120)
    m = 64
    mp = mlx_stump.stump(T, m)

    for i in (500, 530, 556):
        j = mp.I_[i]
        a = T[i : i + m]
        b = T[j : j + m]
        ac, bc = a - a.mean(), b - b.mean()
        rho = (ac @ bc) / (m * a.std() * b.std())
        truth2 = max(2.0 * m * (1.0 - rho), 0.0)
        # compare squares: near d=0 formula-order noise (raw frame here vs
        # the standardized frame inside the refinement) sqrt-amplifies in d
        assert abs(mp.P_[i] ** 2 - truth2) <= 1e-3, f"row {i}: {mp.P_[i] ** 2} vs truth {truth2}"
        assert mp.P_[i] <= 2.0 * np.sqrt(m) + 1e-6


# --------------------------------------------------------- 3: match semantics
def test_match_inf_max_distance_with_nan_window():
    """max_distance=inf must not let inf profile entries into refinement: a
    single NaN window used to poison the greedy argmin and return nothing."""
    rng = np.random.default_rng(7)
    T = rng.standard_normal(200)
    Q = T[20:30].copy()
    T[100] = np.nan
    M = mlx_stump.match(Q, T, max_distance=float("inf"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Mr = stumpy.match(Q, T, max_distance=float("inf"))
    assert M.shape == Mr.shape
    np.testing.assert_array_equal(np.sort(M[:, 1].astype(int)), np.sort(Mr[:, 1].astype(int)))
    np.testing.assert_allclose(
        np.sort(M[:, 0].astype(float)), np.sort(Mr[:, 0].astype(float)), atol=1e-12
    )


@pytest.mark.parametrize("seed", [1, 2, 3, 11, 29])
def test_match_default_threshold_recomputed_after_refinement(seed):
    """STUMPY derives the default threshold from an exact float64 profile; the
    float32 estimate must be recomputed after refinement or the nearest match
    can fall just above it and [] comes back (reproduced at m=3)."""
    rng = np.random.default_rng(seed)
    T = rng.standard_normal(200)
    Q = rng.standard_normal(3)
    M = mlx_stump.match(Q, T)
    Mr = stumpy.match(Q, T)
    assert M.shape == Mr.shape
    if len(Mr):
        np.testing.assert_array_equal(M[:, 1].astype(int), Mr[:, 1].astype(int))
        np.testing.assert_allclose(
            M[:, 0].astype(float), Mr[:, 0].astype(float), atol=tie_tolerance(3)
        )


def test_match_isconstant_overrides_survive_refinement():
    """Explicit Q/T_subseq_isconstant flags must reach the float64 refinement
    (it used to re-detect constants from raw values and drop the overrides)."""
    rng = np.random.default_rng(42)
    T = rng.standard_normal(64)
    m = 8
    Q = T[30:38].copy()

    flag = np.zeros(T.shape[0] - m + 1, dtype=bool)
    flag[10] = True  # window 10 is NOT actually constant
    M = mlx_stump.match(Q, T, max_distance=10.0, T_subseq_isconstant=flag)
    Mr = stumpy.match(Q, T, max_distance=10.0, T_subseq_isconstant=flag)
    d = {int(i): float(v) for v, i in M}
    dr = {int(i): float(v) for v, i in Mr}
    assert d.keys() == dr.keys()
    assert d[10] == pytest.approx(np.sqrt(m), abs=1e-12)  # exactly-one-constant rule
    for i in d:
        assert d[i] == pytest.approx(dr[i], abs=1e-6)

    M = mlx_stump.match(Q, T, max_distance=10.0, Q_subseq_isconstant=np.array([True]))
    Mr = stumpy.match(Q, T, max_distance=10.0, Q_subseq_isconstant=np.array([True]))
    assert M.shape == Mr.shape
    np.testing.assert_allclose(M[:, 0].astype(float), np.sqrt(m), atol=1e-12)


def test_query_idx_out_of_range_raises():
    """Negative query_idx used to wrap via numpy indexing and fabricate
    results like [[0.0, -1]]. STUMPY itself fabricates [(0.0, -5)] for
    query_idx <= -m; rejecting every out-of-range value is a deliberate,
    stricter divergence."""
    rng = np.random.default_rng(5)
    T = rng.standard_normal(20)
    Q = T[5:9].copy()
    l = T.shape[0] - Q.shape[0] + 1
    for bad in (-1, -5, l, l + 3):
        with pytest.raises(ValueError, match="query_idx"):
            mlx_stump.mass(Q, T, query_idx=bad)
        with pytest.raises(ValueError, match="query_idx"):
            mlx_stump.match(Q, T, query_idx=bad)
    # boundary values are legal
    assert mlx_stump.mass(Q, T, query_idx=0).shape == (l,)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Q != T[l-1:] identity advisory
        assert mlx_stump.mass(Q, T, query_idx=l - 1)[l - 1] == 0.0


# ----------------------------------------------------------- 4: memory budget
class _FakeEngine:
    def __init__(self, l, tile_rows=1):
        self.l = l
        self.tile_rows = tile_rows


@pytest.mark.parametrize("l", [65_337, 2_000_000, 10_000_000])
@pytest.mark.parametrize("k,self_join", [(1, True), (2, True), (4, False)])
def test_default_chunk_size_respects_budget(l, k, self_join):
    """The budget is a real ceiling now: no 16-row floor. A single row may
    exceed the budget only when even chunk_size=1 does (b == 1)."""
    eng = _FakeEngine(l)
    b = default_chunk_size(eng, l_q=l, k=k, self_join=self_join)
    per_row = l * (16 if k == 1 else (48 if self_join else 40))
    assert b >= 1
    assert b * per_row <= _CHUNK_MEM_BUDGET or b == 1


def test_tiled_chunk_size_respects_budget():
    eng = _FakeEngine(10_000_000, tile_rows=327_680)
    for k, self_join, cell in [(1, True, 16), (2, True, 48), (2, False, 40)]:
        b = tiled_chunk_size(eng, l_q=10_000_000, k=k, self_join=self_join)
        assert b >= 1
        assert b * eng.tile_rows * cell <= _CHUNK_MEM_BUDGET or b == 1


@pytest.mark.gpu
@pytest.mark.slow
def test_topk_peak_memory_bounded():
    """k=2 at n=65536/m=200 used to peak at ~1.18 GiB (3x the budget); the
    top-k intermediates are accounted for now."""
    import mlx.core as mx

    rng = np.random.default_rng(3)
    T = rng.standard_normal(65_536).cumsum()
    mx.reset_peak_memory()
    mlx_stump.stump(T, 200, k=2)
    peak = mx.get_peak_memory()
    assert peak < 700 * (1 << 20), f"peak {peak / (1 << 20):.0f} MiB"
