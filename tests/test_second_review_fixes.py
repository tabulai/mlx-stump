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
    def __init__(self, l, m=200, tile_rows=1):
        self.l = l
        self.m = m
        self.tile_rows = tile_rows


@pytest.mark.parametrize(
    "l,m", [(65_337, 200), (2_000_000, 200), (10_000_000, 200), (16_385, 16_384)]
)
@pytest.mark.parametrize("k,self_join", [(1, True), (2, True), (4, False)])
def test_default_chunk_size_respects_budget(l, m, k, self_join):
    """The budget is a real ceiling now: no 16-row floor, and the per-row
    query-window batch (which grows with m) is accounted for. A single row
    may exceed the budget only when even chunk_size=1 does (b == 1)."""
    eng = _FakeEngine(l, m)
    b = default_chunk_size(eng, l_q=l, k=k, self_join=self_join)
    per_row = l * (16 if k == 1 else (48 if self_join else 40)) + m * 24
    assert b >= 1
    assert b * per_row <= _CHUNK_MEM_BUDGET or b == 1


def test_tiled_chunk_size_respects_budget():
    for l, m, tile_rows in [(10_000_000, 200, 327_680), (65_537, 32_768, 2_048)]:
        eng = _FakeEngine(l, m, tile_rows=tile_rows)
        for k, self_join, cell in [(1, True, 16), (2, True, 48), (2, False, 40)]:
            b = tiled_chunk_size(eng, l_q=l, k=k, self_join=self_join)
            assert b >= 1
            assert b * (eng.tile_rows * cell + m * 24) <= _CHUNK_MEM_BUDGET or b == 1


# --------------------------------------- adversarial-verification round two
def _fsum_znorm(T, m, i, j):
    import math

    a, b = T[i : i + m].astype(float), T[j : j + m].astype(float)
    ca = a - math.fsum(a) / m
    cb = b - math.fsum(b) / m
    sa = math.sqrt(math.fsum(ca * ca) / m)
    sb = math.sqrt(math.fsum(cb * cb) / m)
    if sa == 0.0 or sb == 0.0:
        return 0.0 if sa == sb else math.sqrt(m)
    return math.sqrt(math.fsum(((ca / sa) - (cb / sb)) ** 2))


def _assert_exact_and_no_bad_neighbors(T, m, mp, ref, p_tol, gap_tol=0.08):
    """Reported P must equal the fsum ground truth at its own chosen index,
    and every index disagreement with STUMPY must be a true near-tie."""
    worst_p = 0.0
    for i in range(len(mp.P_)):
        if not np.isfinite(ref.P_[i]):
            continue
        worst_p = max(worst_p, abs(mp.P_[i] - _fsum_znorm(T, m, i, mp.I_[i])))
        if mp.I_[i] != ref.I_[i]:
            gap = _fsum_znorm(T, m, i, mp.I_[i]) - _fsum_znorm(T, m, i, ref.I_[i])
            assert gap <= gap_tol, f"row {i}: our neighbor is {gap:.4f} worse"
    assert worst_p <= p_tol, f"worst |P - truth at chosen index| = {worst_p:.3g}"


def test_sigma_repair_headroom_offset_segment():
    """A huge constant segment crushes ordinary windows' variance toward the
    cumsum noise floor; without relative headroom on the sigma-repair bound,
    windows just above it kept ~percent-level sigma error (205 rows chose
    neighbors up to 0.37 worse than STUMPY's here, and reported P was off by
    up to 0.045 at its own chosen pair)."""
    rng = np.random.default_rng(1)
    T = rng.standard_normal(2000)
    T[100:300] = 1e5
    m = 3
    _assert_exact_and_no_bad_neighbors(T, m, mlx_stump.stump(T, m), stumpy.stump(T, m), 1e-9)


@pytest.mark.slow
def test_refine_exact_at_extreme_amplitude():
    """An extreme-amplitude segment used to leave the refinement consuming
    ~1e-2-relative sigma (max profile error 9.4e-3, 29 rows with neighbors
    up to 0.21 worse); refinement now recomputes exact two-pass stats."""
    rng = np.random.default_rng(2)
    T = rng.standard_normal(1000).cumsum()
    T[800:900] = 1e7 * np.sin(np.linspace(0, 3, 100))
    m = 50
    _assert_exact_and_no_bad_neighbors(T, m, mlx_stump.stump(T, m), stumpy.stump(T, m), 1e-8)


@pytest.mark.slow
def test_refine_exact_large_walk():
    """Reported P at the chosen index was drifting linearly with n through
    the cumsum rolling stats (2.4e-8 at n=131072); with self-contained
    two-pass refinement stats it sits at machine epsilon."""
    rng = np.random.default_rng(5)
    T = rng.standard_normal(32768).cumsum()
    m = 200
    mp = mlx_stump.stump(T, m)
    rows = np.linspace(0, len(mp.P_) - 1, 200).astype(int)
    worst = max(
        abs(mp.P_[i] - _fsum_znorm(T, m, i, mp.I_[i])) for i in rows if np.isfinite(mp.P_[i])
    )
    assert worst <= 1e-10, f"worst sampled |P - truth| = {worst:.3g}"


def test_aamp_mixed_scale_centered():
    """normalize=False used to compute ssq_q + ssq_t - 2*QT uncentered in
    float32: on a series with a huge segment, 83% of rows chose neighbors
    >0.2 raw units worse than STUMPY's (worst 4x). The centered form
    (||qc - tc||^2 + m*(mu_q - mu_t)^2) removes the cancellation."""
    import math

    rng = np.random.default_rng(3)
    T = rng.standard_normal(1500)
    T[100:350] = 1e5
    m = 7
    mp = mlx_stump.stump(T, m, normalize=False)
    ref = stumpy.aamp(T, m)

    def dist(i, j):
        d = T[i : i + m] - T[j : j + m]
        return math.sqrt(math.fsum(d * d))

    worst_p = worst_gap = 0.0
    for i in range(len(mp.P_)):
        if not np.isfinite(ref.P_[i]):
            continue
        worst_p = max(worst_p, abs(mp.P_[i] - dist(i, mp.I_[i])))
        if mp.I_[i] != ref.I_[i]:
            worst_gap = max(worst_gap, dist(i, mp.I_[i]) - dist(i, ref.I_[i]))
    assert worst_p <= 1e-8
    assert worst_gap <= 0.01, f"worst raw-unit gap vs STUMPY's neighbor: {worst_gap:.4f}"

    # planted occurrences must survive a raw-unit max_distance like STUMPY's
    Q = T[500:520].copy()
    M = mlx_stump.match(Q, T, normalize=False, max_distance=5.0)
    Mr = stumpy.match(Q, T, normalize=False, max_distance=5.0)
    assert sorted(int(i) for _, i in M) == sorted(int(i) for _, i in Mr)


def test_match_refinement_is_chunked():
    """max_distance=inf selects every finite entry for refinement; that must
    stream in byte-budgeted chunks (it used to materialize all l*m float64
    windows at once — ~GiBs for long series), with identical results."""
    import mlx_stump._stump as st

    rng = np.random.default_rng(11)
    T = rng.standard_normal(3000)
    Q = rng.standard_normal(21)
    expected = mlx_stump.match(Q, T, max_distance=float("inf"))
    orig = st._REFINE_MEM_BUDGET
    st._REFINE_MEM_BUDGET = 21 * 8 * 4 * 7  # ~7 rows per chunk
    try:
        got = mlx_stump.match(Q, T, max_distance=float("inf"))
    finally:
        st._REFINE_MEM_BUDGET = orig
    np.testing.assert_array_equal(expected.astype(float), got.astype(float))


def test_match_reports_true_near_duplicate_distances():
    """STUMPY's mass/match do not apply the stump P-norm zero-snap; matches a
    hair above zero must report their true distances (they used to snap to
    0.0 and inflate the returned match set)."""
    rng = np.random.default_rng(4)
    T = rng.standard_normal(400)
    pat = rng.standard_normal(12)
    for p in (50, 150, 300):
        T[p : p + 12] = pat * (1 + 1e-7 * rng.standard_normal(12))
    M = mlx_stump.match(pat.copy(), T)
    Mr = stumpy.match(pat.copy(), T)
    assert M.shape == Mr.shape
    near = M[:, 0].astype(float)[:3]
    assert np.all(near > 0.0) and np.all(near < 1e-5)


def test_target_blocks_evenly_split(monkeypatch):
    """Tiled column blocks are split evenly: a narrow (especially 1-column)
    trailing block would hit a different matmul kernel whose accumulation
    can flip float32 near-ties to a different neighbor."""
    import mlx_stump._engine as eng
    from mlx_stump._preprocess import preprocess_series

    rng = np.random.default_rng(6)
    m = 64
    monkeypatch.setattr(eng, "_MATMUL_WINDOW_BYTES", 0)
    for n, tile_bytes in [(1093, 4 * m * 256), (2500, 4 * m * 337), (801, 4 * m * 736)]:
        monkeypatch.setattr(eng, "_TILE_WINDOW_BYTES", tile_bytes)
        engine = eng.MassEngine(preprocess_series(rng.standard_normal(n), m))
        widths = [j1 - j0 for j0, j1, _ in engine.target_blocks()]
        assert sum(widths) == engine.l
        assert max(widths) <= engine.tile_rows
        assert max(widths) - min(widths) <= 1
        assert min(widths) >= 2


# ------------------------------------- adversarial-verification round three
def test_refine_raw_frame_exact():
    """Refinement runs on the RAW series: the standardized copy re-rounds
    every value by eps64 * scale, so a huge segment elsewhere in the series
    used to cost ordinary windows ~5 digits of reported-P accuracy
    (2.7e-5 error on perfectly representable windows)."""
    rng = np.random.default_rng(0)
    T = np.concatenate([np.full(2000, 2e9), 1.0 + 0.01 * rng.standard_normal(2000)])
    m = 50
    mp = mlx_stump.stump(T, m)
    rows = range(2100, 3951, 25)
    worst = max(
        abs(mp.P_[i] - _fsum_znorm(T, m, i, mp.I_[i])) for i in rows if np.isfinite(mp.P_[i])
    )
    assert worst <= 1e-9, f"worst |P - truth| = {worst:.3g}"


@pytest.mark.slow
def test_aamp_extreme_spike_match_not_dropped():
    """normalize=False float32 noise scales with each window's own energy:
    a 1e6-amplitude pattern in unit noise used to read ~4700 at its own
    exact occurrence, past the global-scale refinement cutoff, silently
    dropping the match. The cutoff is per-window now."""
    rng = np.random.default_rng(31337)
    T = rng.standard_normal(400_000)
    pat = rng.standard_normal(100) * 1e6
    T[50_000:50_100] = pat
    T[300_000:300_100] = pat + 1e-3 * rng.standard_normal(100)
    Q = T[50_000:50_100].copy()
    M = mlx_stump.match(Q, T, normalize=False, max_distance=10.0)
    Mr = stumpy.match(Q, T, normalize=False, max_distance=10.0)
    assert sorted(int(i) for _, i in M) == sorted(int(i) for _, i in Mr) == [50_000, 300_000]


def test_match_precomputed_stats_are_compatibility_metadata():
    """M_T/Σ_T are validated compatibility metadata — the float32 search and
    float64 refinement both use raw-window local normalization, so there is no
    hybrid profile (the refinement once recomputed exact stats while the search
    ranked by the supplied ones). STUMPY's own compute_mean_std
    output reproduces STUMPY's ranking; a deliberately doubled Σ_T changes
    STUMPY's result and not this one (the documented divergence)."""
    rng = np.random.default_rng(12)
    T = rng.standard_normal(400)
    Q = rng.standard_normal(21)
    M_T, Σ_T = stumpy.core.compute_mean_std(T, 21)
    kw = dict(max_distance=float("inf"), max_matches=5)
    M = mlx_stump.match(Q, T, M_T=M_T, Σ_T=Σ_T, **kw)
    Mr = stumpy.match(Q, T, M_T=M_T, Σ_T=Σ_T, **kw)
    np.testing.assert_array_equal(M[:, 1].astype(int), Mr[:, 1].astype(int))
    np.testing.assert_allclose(M[:, 0].astype(float), Mr[:, 0].astype(float), atol=1e-9)
    M2 = mlx_stump.match(Q, T, M_T=M_T, Σ_T=Σ_T * 2.0, **kw)
    M0 = mlx_stump.match(Q, T, **kw)
    np.testing.assert_array_equal(M2.astype(float), M0.astype(float))
    Mr2 = stumpy.match(Q, T, M_T=M_T, Σ_T=Σ_T * 2.0, **kw)
    assert not np.allclose(Mr2[:, 0].astype(float), Mr[:, 0].astype(float))


def test_dynamic_range_warning():
    """Local z-normalization removes the global-frame precision limit.

    Raw-distance search still uses one shared affine frame and warns at the
    float64 dynamic-range boundary; ordinary flatline-with-jitter data does
    not warn in either mode.
    """
    rng = np.random.default_rng(7)
    T = np.concatenate([rng.standard_normal(2048) * 1e18, rng.standard_normal(4096)])
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        mlx_stump.stump(T, 64)
    with pytest.warns(UserWarning, match="dynamic range"):
        mlx_stump.stump(T, 64, normalize=False)
    T2 = rng.standard_normal(2000)
    T2[900:1100] = 5.0 + 1e-9 * rng.standard_normal(200)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        mlx_stump.stump(T2, 64)


def test_tiny_tiles_floor_at_four_rows(monkeypatch):
    """tile_rows floors at 4: 1-2-wide blocks would hit GEMV-style kernels
    whose accumulation flips float32 ties (verified bit-equal at width>=3 on
    a tie-dense pure sine)."""
    import mlx_stump._engine as eng
    from mlx_stump._preprocess import preprocess_series

    T = np.sin(0.11 * np.arange(3000))
    m = 100
    ref = mlx_stump.stump(T, m)
    monkeypatch.setattr(eng, "_MATMUL_WINDOW_BYTES", 0)
    monkeypatch.setattr(eng, "_TILE_WINDOW_BYTES", 8 * m)  # would be tile_rows=2
    engine = eng.MassEngine(preprocess_series(T, m))
    widths = [j1 - j0 for j0, j1, _ in engine.target_blocks()]
    assert engine.tile_rows == 4
    assert min(widths) >= 3
    mp = mlx_stump.stump(T, m)
    np.testing.assert_array_equal(mp.I_, ref.I_)
    np.testing.assert_allclose(mp.P_, ref.P_, atol=0, rtol=0)
    np.testing.assert_array_equal(mp.right_I_, ref.right_I_)


def test_refine_exact_at_large_common_offset():
    """Raw-frame refinement must shift each window by its own first element:
    a large common offset (unix-ms timestamps, say) otherwise rounds the
    two-pass window mean at eps64*offset and cost ~9 digits of reported P
    (7e-7 error at offset 1e12; the exact truth costs nothing)."""
    rng = np.random.default_rng(42)
    base = rng.standard_normal(3000)
    m = 50
    for offset in (1e9, 1.7e12):
        T = offset + base
        # ground truth from T itself, shifted exactly (all values share one
        # binade, so T - T[0] is Sterbenz-exact); comparing against `base`
        # would ignore that T's own representation quantized the data
        Tshift = T - T[0]
        mp = mlx_stump.stump(T, m)
        rows = range(0, len(mp.P_), 37)
        worst = max(
            abs(mp.P_[i] - _fsum_znorm(Tshift, m, i, mp.I_[i]))
            for i in rows
            if np.isfinite(mp.P_[i])
        )
        assert worst <= 1e-10, f"offset {offset}: worst |P - truth| = {worst:.3g}"
    # match's refinement shares the mechanism
    T = 1.7e12 + base
    Q = T[100:121].copy()
    M = mlx_stump.match(Q, T, max_distance=1e-3)
    assert int(M[0, 1]) == 100 and float(M[0, 0]) == 0.0


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
