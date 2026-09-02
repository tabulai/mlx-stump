"""Regression tests for defects found in the third (external) review.

1. `match`/`mass` with precomputed ``M_T``/``Σ_T`` went back to STUMPY's
   ``QT - m*mu_q*M_T`` form in the float64 refinement, which cancels
   catastrophically on offset data (self-distances far above the
   ``2*sqrt(m)`` maximum) and dropped genuine matches;
2. the documented memory budget covered the per-batch GPU intermediates
   only: the tiled ``mass`` path pinned every lazy block until
   concatenation, the block-centering step and the sigma repair used
   row-based chunks whose float64 temporaries grew with ``m``, and a
   constant series "repaired" every window it has;
3. release tags were not tied to the package version and the release job
   tested the source tree rather than the wheel (see test_release_guard.py);
4. smaller gaps: the data-dependent ``match`` threshold was not recomputed
   after the last refinement round, ``query_idx`` overrode an explicit
   non-finite window in non-normalized ``mass``, and non-boolean
   ``Q_subseq_isconstant`` values were coerced through ``bool()``.
"""

from __future__ import annotations

import math
import subprocess
import sys
import textwrap
import warnings

import numpy as np
import pytest
import stumpy

import mlx_stump
import mlx_stump._engine as eng
import mlx_stump._preprocess as prep_mod
from mlx_stump._engine import (
    _CENTER_BYTES,
    _CENTER_ROW_BYTES,
    _CHUNK_MEM_BUDGET,
    _center_rows,
    estimated_peak_bytes,
    resident_block_bytes,
)
from mlx_stump._preprocess import preprocess_series, rolling_mean_sigma

MIB = 1 << 20


# ------------------------------------------------ 1: precomputed statistics
def _offset_flatlines(seed=0):
    rng = np.random.default_rng(seed)
    T = rng.standard_normal(3000)
    T[500:620] = 1e3 + 1e-5 * rng.standard_normal(120)
    T[2000:2120] = T[500:620]  # exact duplicate of the first flat segment
    return T


def test_match_precomputed_stats_no_cancellation():
    """With STUMPY's own sliding stats on a flatline at an offset, the two
    exact occurrences must come first at zero and nothing may exceed the
    z-normalized maximum (the cancelling form reported 8+ at unrelated
    indices and ranked them first)."""
    T = _offset_flatlines()
    m = 64
    Q = T[520 : 520 + m].copy()
    M_T, Σ_T = stumpy.core.compute_mean_std(T, m)
    M = mlx_stump.match(Q, T, M_T=M_T, Σ_T=Σ_T, max_distance=float("inf"), max_matches=3)
    assert [int(i) for _, i in M[:2]] == [520, 2020]
    # Cached statistics are compatibility metadata, so their rounding cannot
    # leak into either exact occurrence.
    assert all(float(d) == 0.0 for d, _ in M[:2])
    assert all(float(d) <= 2.0 * np.sqrt(m) + 1e-9 for d, _ in M)
    M0 = mlx_stump.match(Q, T, max_distance=float("inf"), max_matches=3)
    np.testing.assert_array_equal(M[:, 1].astype(int), M0[:, 1].astype(int))
    # The default (data-dependent) threshold path is identical with or without
    # the compatibility metadata.
    M = mlx_stump.match(Q, T, M_T=M_T, Σ_T=Σ_T)
    M0 = mlx_stump.match(Q, T)
    assert [int(i) for _, i in M[:2]] == [520, 2020]
    np.testing.assert_array_equal(M[:, 1].astype(int), M0[:, 1].astype(int))
    np.testing.assert_allclose(M[:, 0].astype(float), M0[:, 0].astype(float), atol=1e-5)


@pytest.mark.parametrize("offset", [0.0, 1e3, 1e6, 1e9])
def test_mass_precomputed_stats_offset_data(offset):
    """The GPU profile under user stats must equal the no-stats profile at
    every offset. STUMPY's formula itself degrades here
    (0.08 for a self-match at 1e6, total collapse at 1e9), so parity with
    STUMPY is asserted only where its formula is well-conditioned."""
    rng = np.random.default_rng(1)
    T = offset + rng.standard_normal(4000)
    m = 32
    Q = T[1000 : 1000 + m].copy()
    M_T, Σ_T = stumpy.core.compute_mean_std(T, m)
    D = mlx_stump.mass(Q, T, M_T=M_T, Σ_T=Σ_T)
    D0 = mlx_stump.mass(Q, T)
    assert D[1000] < 1e-2
    np.testing.assert_array_equal(D, D0)
    assert np.all(D <= 2.0 * np.sqrt(m) + 1e-6)
    if offset <= 1e3:
        Dr = stumpy.mass(Q, T, M_T=M_T, Σ_T=Σ_T)
        np.testing.assert_allclose(D**2, Dr**2, atol=3e-4 * m, rtol=1e-3)


def test_mass_stats_validation_and_nonfinite_windows():
    rng = np.random.default_rng(2)
    T = rng.standard_normal(300)
    m = 16
    Q = T[10 : 10 + m].copy()
    M_T, Σ_T = stumpy.core.compute_mean_std(T, m)
    with pytest.raises(ValueError, match="shape"):
        mlx_stump.mass(Q, T, M_T=M_T[:-1], Σ_T=Σ_T)
    with pytest.raises(ValueError, match="shape"):
        mlx_stump.match(Q, T, M_T=M_T, Σ_T=Σ_T[:-1])
    # STUMPY's convention: a window whose mean is inf is reported as inf
    M2 = M_T.copy()
    M2[100] = np.inf
    D = mlx_stump.mass(Q, T, M_T=M2, Σ_T=Σ_T)
    Dr = stumpy.mass(Q, T, M_T=M2, Σ_T=Σ_T)
    assert np.isinf(D[100]) and np.isinf(Dr[100])
    assert np.isfinite(D[99]) and np.isfinite(D[101])
    M = mlx_stump.match(Q, T, M_T=M2, Σ_T=Σ_T, max_distance=float("inf"))
    assert 100 not in {int(i) for _, i in M}
    # The contract: finite metadata never enters the arithmetic, so the
    # profile equals the no-stats call exactly.
    D0 = mlx_stump.mass(Q, T)
    np.testing.assert_array_equal(mlx_stump.mass(Q, T, M_T=M_T + 0.5, Σ_T=Σ_T), D0)
    np.testing.assert_array_equal(mlx_stump.mass(Q, T, M_T=M_T, Σ_T=Σ_T * 2.0), D0)
    np.testing.assert_array_equal(mlx_stump.mass(Q, T, M_T=M_T, Σ_T=Σ_T), D0)


# ------------------------------------------------------------- 2: memory
def _run_isolated(code: str) -> tuple[float, float, float]:
    """Run ``code`` in a fresh interpreter; return (rss_before_mib,
    rss_peak_mib, mlx_peak_mib). ``code`` runs after the imports, with
    ``np`` and ``mx`` bound; the RSS baseline is taken after the imports."""
    src = textwrap.dedent(
        """
        import resource, sys
        import numpy as np
        import mlx.core as mx
        import mlx_stump
        from mlx_stump._engine import MassEngine
        from mlx_stump._preprocess import preprocess_series
        _unit = 1 if sys.platform == "darwin" else 1024  # ru_maxrss: bytes vs KiB
        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * _unit
        mx.reset_peak_memory()
        """
        + code
        + """
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * _unit
        print("RESULT", before / 2**20, peak / 2**20, mx.get_peak_memory() / 2**20)
        """
    )
    out = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True, check=True)
    line = [ln for ln in out.stdout.splitlines() if ln.startswith("RESULT")][-1]
    before, peak, mlx_peak = (float(x) for x in line.split()[1:])
    return before, peak, mlx_peak


@pytest.mark.gpu
@pytest.mark.slow
def test_mass_is_blockwise():
    """mass(n=65536, m=2200) is a tiled target (557 MB / 531 MiB window
    matrix); it used to pin all lazy blocks at once (533 MiB with the old
    256 MiB tiles) — now one block at a time."""
    n, m = 65_536, 2_200
    l = n - m + 1
    block = resident_block_bytes(l, m)
    assert block < l * m * 4  # the case is tiled
    _, rss, mlx_peak = _run_isolated(
        f"""
        rng = np.random.default_rng(0)
        T = rng.standard_normal({n}).cumsum(); Q = T[100:100 + {m}].copy()
        mlx_stump.mass(Q, T)
        mlx_stump.match(Q, T, max_matches=3)
        """
    )
    assert mlx_peak * MIB < block + 64 * MIB, f"MLX peak {mlx_peak:.0f} MiB"
    assert rss * MIB < estimated_peak_bytes(l, m) + 128 * MIB, f"RSS peak {rss:.0f} MiB"


@pytest.mark.slow
def test_engine_build_is_byte_budgeted():
    """A dense window matrix used to be centered in one float64 temporary of
    the same row count (a 500 MiB matrix cost 2 GB RSS); the centering step
    is byte-budgeted now, so the build costs 2*block + 64 MiB. (m=1000 keeps
    the 246 MiB matrix under the dense cap; a tiled target builds nothing
    at construction and would make this test vacuous.)"""
    n, m = 65_536, 1_000
    l = n - m + 1
    assert resident_block_bytes(l, m) == l * m * 4  # dense
    before, rss, _ = _run_isolated(
        f"""
        rng = np.random.default_rng(0)
        prep = preprocess_series(rng.standard_normal({n}).cumsum(), {m})
        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * _unit
        eng = MassEngine(prep)
        """
    )
    assert (rss - before) * MIB <= estimated_peak_bytes(l, m) + 128 * MIB, (
        f"engine build grew RSS by {rss - before:.0f} MiB"
    )


@pytest.mark.slow
def test_constant_series_preprocessing_is_cheap():
    """Every window of a constant series is a sigma-repair suspect; they used
    to be re-read in 288 MB float64 chunks (1.4 GB RSS for stump). Known
    constant windows are written directly now, and what remains streams in
    32 MiB chunks."""
    n, m = 65_536, 2_200
    l = n - m + 1
    before, rss, _ = _run_isolated(
        f"""
        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * _unit
        preprocess_series(np.full({n}, 7.0), {m}, normalize=False)
        """
    )
    assert rss - before < 96, f"constant-series preprocessing grew RSS by {rss - before:.0f} MiB"
    before, rss, mlx_peak = _run_isolated(
        f"""
        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * _unit
        mlx_stump.stump(np.full({n}, 7.0), {m})
        """
    )
    assert (rss - before) * MIB <= estimated_peak_bytes(l, m) + 128 * MIB, (
        f"constant-series stump grew RSS by {rss - before:.0f} MiB"
    )
    assert mlx_peak * MIB <= resident_block_bytes(l, m) + _CHUNK_MEM_BUDGET + 16 * MIB


@pytest.mark.gpu
@pytest.mark.slow
def test_tiled_stump_peak_within_ceiling():
    n, m = 65_536, 2_200
    l = n - m + 1
    before, rss, mlx_peak = _run_isolated(
        f"""
        rng = np.random.default_rng(0)
        T = rng.standard_normal({n}).cumsum()
        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * _unit
        mlx_stump.stump(T, {m})
        """
    )
    assert mlx_peak * MIB <= resident_block_bytes(l, m) + _CHUNK_MEM_BUDGET + 16 * MIB
    assert (rss - before) * MIB <= estimated_peak_bytes(l, m) + 128 * MIB


def test_center_rows_respects_byte_budget():
    for m in (3, 200, 2_200, 16_384, 1 << 20):
        rows = _center_rows(m)
        assert rows >= 1
        assert rows * (m * 8 + _CENTER_ROW_BYTES) <= _CENTER_BYTES or rows == 1


def test_small_center_budget_is_bit_identical(monkeypatch):
    rng = np.random.default_rng(3)
    T = rng.standard_normal(3000)
    T[1000:1200] = 5.0 + 1e-9 * rng.standard_normal(200)
    m = 64
    ref = mlx_stump.stump(T, m)
    monkeypatch.setattr(
        eng, "_CENTER_BYTES", (64 * 8 + _CENTER_ROW_BYTES) * 37
    )  # 37 rows per step
    mp = mlx_stump.stump(T, m)
    np.testing.assert_array_equal(mp.I_, ref.I_)
    np.testing.assert_allclose(mp.P_, ref.P_, atol=0, rtol=0)


def test_sigma_repair_known_constant_shortcut(monkeypatch):
    rng = np.random.default_rng(4)
    a = rng.standard_normal(2000)
    a[300:700] = 3.0
    a[1500:1600] = -2.0
    w = 25
    T_nan = a.copy()
    detected = prep_mod.rolling_isconstant(T_nan, w)
    mu_ref, sig_ref = rolling_mean_sigma(a, w)
    mu, sig = rolling_mean_sigma(a, w, known_constant=detected)
    np.testing.assert_array_equal(sig[detected], 0.0)
    np.testing.assert_array_equal(mu[detected], a[np.nonzero(detected)[0]])
    np.testing.assert_allclose(mu, mu_ref, atol=1e-12)
    np.testing.assert_allclose(sig, sig_ref, atol=1e-12)
    # tiny repair chunks give the same numbers
    monkeypatch.setattr(prep_mod, "_SIGMA_REPAIR_BYTES", w * 8 * 5)
    mu2, sig2 = rolling_mean_sigma(a, w, known_constant=detected)
    np.testing.assert_array_equal(mu2, mu)
    np.testing.assert_array_equal(sig2, sig)
    # a constant window a user flags non-constant still gets sigma 0
    # (the shortcut uses detected constancy, not the user's flags)
    flags = np.zeros(a.shape[0] - w + 1, dtype=bool)
    p = preprocess_series(a, w, normalize=False, isconstant=flags)
    assert np.all(p.sig_inv[detected] == 0.0)


def test_tiled_blocks_are_released(monkeypatch):
    """Both sweeps drop each block before the next one is built: the MLX peak
    stays near one block, not the sum of all of them."""
    import mlx.core as mx

    rng = np.random.default_rng(5)
    T = rng.standard_normal(6000)
    m = 64
    Q = T[100 : 100 + m].copy()
    monkeypatch.setattr(eng, "_MATMUL_WINDOW_BYTES", 0)
    monkeypatch.setattr(eng, "_TILE_WINDOW_BYTES", 4 * m * 512)  # 128 KiB blocks, ~12 of them
    ref = mlx_stump.mass(Q, T)
    mx.reset_peak_memory()
    base = mx.get_active_memory()
    D = mlx_stump.mass(Q, T)
    peak = mx.get_peak_memory() - base
    l = T.shape[0] - m + 1
    assert peak < 3 * 4 * m * 512 + 4 * l * 8, f"mass peak {peak / 1024:.0f} KiB"
    np.testing.assert_array_equal(D, ref)


def test_peak_estimate_is_consistent():
    for l, m in [(1000, 8), (65_337, 200), (63_337, 2_200), (10_000_000, 200)]:
        block = resident_block_bytes(l, m)
        assert 0 < block <= max(eng._MATMUL_WINDOW_BYTES, eng._TILE_WINDOW_BYTES + 4 * m * 4)
        assert estimated_peak_bytes(l, m) >= block + _CHUNK_MEM_BUDGET


# --------------------------------------------- 4: threshold / query_idx / flags
@pytest.mark.parametrize("seed", range(0, 60, 4))
def test_match_default_threshold_parity(seed):
    """Parity of the default threshold with STUMPY. (This passes on the old
    two-round code as well — the review's empty-result scenario is
    theoretical — so the mechanism itself is guarded by
    test_match_callable_threshold_reaches_fixed_point below.)"""
    rng = np.random.default_rng(seed)
    n = 200 + 50 * (seed % 7)
    m = [3, 5, 8, 16, 32][seed % 5]
    kind = seed % 3
    if kind == 0:
        T = rng.standard_normal(n)
    elif kind == 1:
        T = rng.standard_normal(n).cumsum()
    else:
        T = np.sin(np.linspace(0, 20 * np.pi, n)) + 0.3 * rng.standard_normal(n)
    Q = rng.standard_normal(m) if seed % 2 else T[n // 3 : n // 3 + m].copy()
    M = mlx_stump.match(Q, T)
    Mr = stumpy.match(Q, T)
    assert M.shape == Mr.shape
    if len(Mr):
        np.testing.assert_array_equal(M[:, 1].astype(int), Mr[:, 1].astype(int))


def test_match_callable_threshold_reaches_fixed_point():
    """A data-dependent threshold is re-evaluated on each refined profile
    until a cutoff exposes no new candidates; the threshold that selects the
    matches is the one computed on the final changed profile, and every
    reported distance is the float64 value that profile holds. The callable
    is not invoked redundantly on an unchanged profile."""
    rng = np.random.default_rng(6)
    T = rng.standard_normal(1500)
    m = 20
    Q = T[700 : 700 + m].copy()
    for p in (100, 400, 1200):
        T[p : p + m] = Q + 1e-3 * rng.standard_normal(m)
    seen = []

    def thr(D):
        seen.append(D.copy())
        return float(np.nanmin(D)) + 0.5

    M = mlx_stump.match(Q, T, max_distance=thr)
    assert len(seen) >= 2
    final = seen[-1]
    md = float(np.nanmin(final)) + 0.5
    for d, i in M:
        assert float(d) == final[int(i)]
        assert float(d) <= md + 1e-8
    # The following refinement attempt found no new rows, so `final` remained
    # unchanged without invoking a side-effecting callback a redundant time.
    assert {int(i) for _, i in M} >= {700, 100, 400, 1200}
    Mr = stumpy.match(Q, T, max_distance=thr)
    np.testing.assert_array_equal(M[:, 1].astype(int), Mr[:, 1].astype(int))


def test_match_nan_threshold_reports_refined_distances():
    """A NaN max_distance selects nothing by comparison, but STUMPY's greedy
    loop then returns every finite entry; the same selection here must carry
    float64 distances (it used to skip refinement entirely)."""
    rng = np.random.default_rng(13)
    T = rng.standard_normal(400)
    Q = T[250:262].copy()
    for bad in (float("nan"), np.nan, lambda D: float("nan")):
        M = mlx_stump.match(Q, T, max_distance=bad)
        Mr = stumpy.match(Q, T, max_distance=bad)
        np.testing.assert_array_equal(M[:, 1].astype(int), Mr[:, 1].astype(int))
        assert float(M[0, 0]) < 1e-6 and int(M[0, 1]) == 250
        np.testing.assert_allclose(M[:, 0].astype(float), Mr[:, 0].astype(float), atol=1e-6)


def test_mass_query_sigma_exact_at_large_offset():
    """The query's own sigma is formed from the first-element-shifted window:
    with numpy's unshifted std, a flat-jitter query at offset 1e12 read up
    to ~1.3 at its own exact occurrence instead of the float32 floor."""
    for offset, seed in ((1e11, 2), (1e12, 1), (1e12, 2)):
        rng = np.random.default_rng(seed)
        T = offset + 1e-3 * rng.standard_normal(5000)
        m = 100
        T[3000 : 3000 + m] = T[1000 : 1000 + m]
        Q = T[1000 : 1000 + m].copy()
        D = mlx_stump.mass(Q, T)
        assert D[1000] < 5e-3 and D[3000] < 5e-3, (offset, seed, D[1000], D[3000])
        M = mlx_stump.match(Q, T, max_distance=1e-3)
        assert sorted(int(i) for _, i in M) == [1000, 3000]
        assert all(float(d) == 0.0 for d, _ in M)


def test_gpu_scratch_released_after_calls():
    """stump releases the window matrix and MLX's cached batch buffers before
    its float64 refinement; mass on return. Nothing stays cached."""
    import mlx.core as mx

    rng = np.random.default_rng(14)
    T = rng.standard_normal(20_000).cumsum()
    Q = T[100:300].copy()
    for call in (
        lambda: mlx_stump.stump(T, 200),
        lambda: mlx_stump.stump(T, 200, k=3),
        lambda: mlx_stump.mass(Q, T),
        lambda: mlx_stump.match(Q, T, max_matches=3),
    ):
        call()
        assert mx.get_cache_memory() < MIB, f"{mx.get_cache_memory() / MIB:.1f} MiB left cached"


def test_ragged_chunks_are_full_width_and_identical():
    """The trailing batch is recomputed at full width (rows [l-B, l)), so a
    chunk size that does not divide l gives identical results and allocates
    no second, differently sized set of intermediates."""
    import mlx.core as mx

    rng = np.random.default_rng(15)
    T = rng.standard_normal(4000)
    m = 32
    ref = mlx_stump.stump(T, m, k=2)
    l = T.shape[0] - m + 1
    for B in (7, 97, 1000, l - 1):
        mp = mlx_stump.stump(T, m, k=2, chunk_size=B)
        np.testing.assert_array_equal(
            np.asarray(mp[:, :4], dtype=float), np.asarray(ref[:, :4], dtype=float)
        )
        np.testing.assert_array_equal(mp.left_I_, ref.left_I_)
        np.testing.assert_array_equal(mp.right_I_, ref.right_I_)
    # peak of a ragged run equals that of an exact divisor of the same l
    # (l = 3969 = 9 * 441): a 9-row trailing batch used to add a second set
    # of buffers, now it is recomputed as a full 440-row batch
    assert l == 9 * 441
    mx.reset_peak_memory()
    mlx_stump.stump(T, m, chunk_size=440)
    peak_ragged = mx.get_peak_memory()
    mx.reset_peak_memory()
    mlx_stump.stump(T, m, chunk_size=441)
    peak_exact = mx.get_peak_memory()
    assert peak_ragged <= peak_exact + MIB, (peak_ragged / MIB, peak_exact / MIB)


def test_match_accepts_integer_threshold():
    rng = np.random.default_rng(7)
    T = rng.standard_normal(300)
    Q = T[20:30].copy()
    M = mlx_stump.match(Q, T, max_distance=2)
    Mf = mlx_stump.match(Q, T, max_distance=2.0)
    np.testing.assert_array_equal(M.astype(float), Mf.astype(float))


def test_query_idx_respects_isfinite_override_when_absolute():
    """stumpy.mass_absolute applies T_subseq_isfinite after the query_idx
    zero, so an explicitly non-finite self window stays inf (and match then
    returns nothing, exactly like STUMPY); z-normalized mass keeps STUMPY's
    unconditional zero."""
    rng = np.random.default_rng(8)
    T = rng.standard_normal(100)
    m = 10
    Q = T[20:30].copy()
    l = T.shape[0] - m + 1
    flag = np.ones(l, dtype=bool)
    flag[20] = False
    D = mlx_stump.mass(Q, T, normalize=False, T_subseq_isfinite=flag, query_idx=20)
    Dr = stumpy.mass(Q, T, normalize=False, T_subseq_isfinite=flag, query_idx=20)
    assert np.isinf(D[20]) and np.isinf(Dr[20])
    kw = dict(normalize=False, T_subseq_isfinite=flag, query_idx=20, max_distance=10.0)
    assert mlx_stump.match(Q, T, **kw).shape == stumpy.match(Q, T, **kw).shape == (0,)
    T2 = T.copy()
    T2[25] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        D = mlx_stump.mass(Q, T2, query_idx=20)
        Dr = stumpy.mass(Q, T2, query_idx=20)
    assert D[20] == Dr[20] == 0.0


@pytest.mark.parametrize(
    "bad", ["False", np.array(["False"]), 0, 1, np.array([0]), np.array([1.0]), [1]]
)
def test_q_isconstant_rejects_non_boolean(bad):
    rng = np.random.default_rng(9)
    T = rng.standard_normal(100)
    Q = T[20:30].copy()
    with pytest.raises(ValueError, match="Q_subseq_isconstant"):
        mlx_stump.mass(Q, T, Q_subseq_isconstant=bad)
    with pytest.raises(ValueError, match="Q_subseq_isconstant"):
        mlx_stump.match(Q, T, Q_subseq_isconstant=bad)


@pytest.mark.parametrize("good", [True, np.True_, np.array([True]), np.array(True)])
def test_q_isconstant_accepts_booleans(good):
    rng = np.random.default_rng(10)
    T = rng.standard_normal(100)
    Q = T[20:30].copy()
    D = mlx_stump.mass(Q, T, Q_subseq_isconstant=good)
    Dr = stumpy.mass(Q, T, Q_subseq_isconstant=np.array([True]))
    np.testing.assert_allclose(D, Dr, atol=1e-6)


# --------------------------------------------------- aamp mixed-scale claim
def test_aamp_mixed_scale_disagreements_are_float32_near_ties():
    """Index agreement with STUMPY's aamp drops on mixed-scale data (83% at
    m=7 with a 1e5 segment). The README does not claim equality; what it
    claims is that every disagreement is a float32 near-tie relative to the
    distance itself, and the reported P is float64-exact at its own index."""
    cases = []
    rng = np.random.default_rng(3)
    T = rng.standard_normal(1500)
    T[100:350] = 1e5
    cases.append((T, 7))
    rng = np.random.default_rng(5)
    T = rng.standard_normal(3000).cumsum()
    T[1000:1200] += 1e6
    cases.append((T, 50))
    for T, m in cases:
        mp = mlx_stump.stump(T, m, normalize=False)
        ref = stumpy.aamp(T, m)

        def dist(i, j, T=T, m=m):
            d = T[i : i + m] - T[j : j + m]
            return math.sqrt(math.fsum(d * d))

        worst_rel = worst_p = 0.0
        for i in range(len(mp.P_)):
            if not np.isfinite(ref.P_[i]):
                continue
            worst_p = max(worst_p, abs(mp.P_[i] - dist(i, mp.I_[i])))
            if mp.I_[i] != ref.I_[i]:
                d_ref = dist(i, ref.I_[i])
                worst_rel = max(worst_rel, (dist(i, mp.I_[i]) - d_ref) / max(d_ref, 1e-300))
        assert worst_p <= 1e-8
        assert worst_rel <= 4.0 * np.finfo(np.float32).eps, f"m={m}: relative gap {worst_rel:.3g}"
