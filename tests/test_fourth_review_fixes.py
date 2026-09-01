"""Regression tests for the fourth external review.

1. With a user-supplied ``Σ_T`` the ``2m(1 - rho)`` form turns ``Σ_T``'s own
   rounding into a distance floor of ``sqrt(2m*delta)`` on perfect matches
   (8e-4 at offset 1e12, 0.06 at 1e14 with STUMPY's ``compute_mean_std``), so
   ``max_distance=0.01`` missed exact occurrences; bitwise-identical windows
   now read exactly 0.
2. ``mx.clear_cache()`` ran while the per-series device arrays were still
   alive, so they entered MLX's cache when the call returned (17 MiB after
   ``mass(n=1e6)``, 86 MiB at 5e6) despite the README's "nothing stays
   cached".
3. ``estimated_peak_bytes`` was presented as a ceiling; it now includes the
   host outputs and the one-row minimum and is documented as an estimate.
4. The release workflow accepted any matching ``v*`` tag; it now refuses a
   tag whose commit is not on ``main``.
5. A non-finite ``Σ_T`` entry was reported as inf; STUMPY applies the
   constant-window rules first and treats ``inf`` as a zero sigma.
"""

from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pytest
import stumpy

import mlx_stump
from mlx_stump._engine import _CHUNK_MEM_BUDGET, estimated_peak_bytes, resident_block_bytes

MIB = 1 << 20
_REPO = pathlib.Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("offset,jitter", [(1e12, 1e-3), (1e14, 0.1), (1e15, 1.0)])
def test_user_stats_bitwise_duplicates_read_zero(offset, jitter):
    """(The jitter must exceed the float64 ulp at the offset — 1.2e-4 at
    1e12, 1.6e-2 at 1e14 — or the whole series quantizes to a constant.)"""
    rng = np.random.default_rng(3)
    T = offset + jitter * rng.standard_normal(5000)
    assert np.ptp(T) > 0
    m = 100
    T[3000 : 3000 + m] = T[1000 : 1000 + m]
    Q = T[1000 : 1000 + m].copy()
    M_T, Σ_T = stumpy.core.compute_mean_std(T, m)
    for kw in ({"max_distance": 0.01}, {"max_distance": 0.0}, {}):
        M = mlx_stump.match(Q, T, M_T=M_T, Σ_T=Σ_T, **kw)
        assert sorted(int(i) for _, i in M[:2]) == [1000, 3000], (offset, kw)
        assert all(float(d) == 0.0 for d, _ in M[:2])
    M0 = mlx_stump.match(Q, T, max_distance=0.01)
    assert sorted(int(i) for _, i in M0) == [1000, 3000]
    # a perturbed copy is not bitwise-identical: under user stats its
    # squared distance differs from the exact one by at most the documented
    # 2m*delta, delta being Σ_T's own relative deviation from the exact sigma
    T2 = T.copy()
    T2[3000 : 3000 + m] = Q + 3.0 * np.spacing(offset) * rng.standard_normal(m)
    assert not np.array_equal(T2[3000 : 3000 + m], Q)
    M2, S2 = stumpy.core.compute_mean_std(T2, m)
    D = mlx_stump.match(Q, T2, M_T=M2, Σ_T=S2, max_distance=float("inf"), max_matches=2)
    D0 = mlx_stump.match(Q, T2, max_distance=float("inf"), max_matches=2)
    d_user = {int(i): float(d) for d, i in D}[3000]
    d_exact = {int(i): float(d) for d, i in D0}[3000]
    w = T2[3000 : 3000 + m] - T2[3000]
    sig = float(np.sqrt(np.mean((w - w.mean()) ** 2)))
    delta = abs(1.0 - sig / S2[3000])
    assert abs(d_user**2 - d_exact**2) <= 1.5 * 2 * m * delta + 1e-6, (d_user, d_exact, delta)


@pytest.mark.slow
def test_nothing_cached_after_large_calls():
    import mlx.core as mx

    rng = np.random.default_rng(4)
    T = rng.standard_normal(1_000_000).cumsum()
    Q = T[100:200].copy()
    mlx_stump.mass(Q, T)
    del T, Q
    assert mx.get_cache_memory() < MIB, f"{mx.get_cache_memory() / MIB:.1f} MiB cached after mass"
    T = rng.standard_normal(200_000).cumsum()
    mlx_stump.stump(T, 50)
    del T
    assert mx.get_cache_memory() < MIB, f"{mx.get_cache_memory() / MIB:.1f} MiB cached after stump"
    assert mx.get_active_memory() < MIB


def test_nonfinite_stats_follow_stumpy_order():
    rng = np.random.default_rng(5)
    T = rng.standard_normal(300)
    T[100:140] = 2.0  # windows 100..124 are constant at m=16
    m = 16
    Q = T[10 : 10 + m].copy()
    M_T, Σ_T = stumpy.core.compute_mean_std(T, m)
    S = Σ_T.copy()
    S[110] = np.nan  # constant window: STUMPY's constant rule precedes the stats
    S[200] = np.inf  # ordinary window: STUMPY's denominator -> rho = 0
    Mb = M_T.copy()
    Mb[50] = np.inf  # STUMPY: inf mean -> inf distance, ahead of everything
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        D = mlx_stump.mass(Q, T, M_T=Mb, Σ_T=S)
        Dr = stumpy.mass(Q, T, M_T=Mb, Σ_T=S)
    assert D[110] == Dr[110] == np.sqrt(m)
    assert D[200] == pytest.approx(np.sqrt(2 * m), abs=1e-3) and Dr[200] == np.sqrt(2 * m)
    assert np.isinf(D[50]) and np.isinf(Dr[50])
    # the float64 refinement applies the same rules
    from mlx_stump._match import _refine_candidates
    from mlx_stump._preprocess import rolling_isconstant

    t_const = rolling_isconstant(T, m)
    d = _refine_candidates(Q, T, np.array([110, 200]), True, False, t_const, S)
    assert d[0] == np.sqrt(m) and d[1] == np.sqrt(2 * m)
    M = mlx_stump.match(Q, T, M_T=Mb, Σ_T=S, max_distance=float("inf"))
    assert 50 not in {int(i) for _, i in M}


def test_peak_estimate_terms():
    l, m = 63_337, 2_200
    base = estimated_peak_bytes(l, m)
    assert base >= resident_block_bytes(l, m) + _CHUNK_MEM_BUDGET + l * 32
    assert estimated_peak_bytes(l, m, k=3) == base + l * 32
    # a single batch row that exceeds the budget is what gets allocated
    huge_l = 100_000_000
    assert estimated_peak_bytes(huge_l, 3) > resident_block_bytes(huge_l, 3) + _CHUNK_MEM_BUDGET


def test_release_workflow_requires_main():
    text = (_REPO / ".github/workflows/release.yml").read_text()
    assert "fetch-depth: 0" in text
    assert "merge-base --is-ancestor" in text
    assert text.index("is on main") < text.index("check_tag_version.py")
