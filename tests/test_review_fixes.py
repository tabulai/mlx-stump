"""Regression tests for defects found in the adversarial review."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import stumpy

import mlx_stump

from .conftest import (
    DATASETS,
    assert_indices_tie_tolerant,
    tie_tolerance,
)


def _true_row_distances(T, m, i, excl):
    """Brute-force float64 distance profile row via two-pass covariance."""
    l = T.shape[0] - m + 1
    a = T[i : i + m]
    ac = a - a.mean()
    sa = a.std()
    d = np.empty(l)
    for j in range(l):
        b = T[j : j + m]
        bc = b - b.mean()
        sb = b.std()
        rho = (ac @ bc) / (m * sa * sb)
        d[j] = np.sqrt(max(2.0 * m * (1.0 - rho), 0.0))
    d[max(0, i - excl) : i + excl + 1] = np.inf
    return d


def test_flatlined_sensor_sigma_not_zeroed():
    """Near-constant windows (flatline + tiny jitter) must keep their true
    variation: deriving sigma from one-pass cumsums would cancel below the
    float64 noise floor and poison the profile with sqrt(2m) distances.

    STUMPY itself is NOT the oracle for the flatline rows: its denominator
    clamp (max(sigma*sigma*m, 1e-14)) plus the rho<=1 cap turn flat-vs-flat
    pairs into spurious zero-distance matches. mlx-stump locally centers and
    RMS-normalizes every raw window in a bounded float64 frame before upload,
    so flatline rows are checked against a brute-force float64 ground truth,
    and every other row against STUMPY as usual.
    """
    rng = np.random.default_rng(0)
    T = rng.standard_normal(2000)
    T[900:1100] = 5.0 + 1e-9 * rng.standard_normal(200)
    m = 64
    excl = int(np.ceil(m / 4))
    mp = mlx_stump.stump(T, m)
    ref = stumpy.stump(T, m)
    tie = tie_tolerance(m)

    degenerate = np.zeros(mp.shape[0], dtype=bool)
    degenerate[900 - m + 1 : 1100] = True

    normal = ~degenerate
    assert_indices_tie_tolerant(mp.I_[normal], ref.I_[normal], T, T, m, tie_atol=tie)
    # neighbors at the flat boundary sit at the float64 representation limit
    # of the jitter (5.0 + 1e-9 keeps ~7 significant digits), so sigma taken
    # over standardized vs raw values legitimately differs ~5e-8 relative,
    # moving those distances by ~1e-4
    agree = normal & (mp.I_ == ref.I_)
    np.testing.assert_allclose(mp.P_[agree], ref.P_[agree], atol=5e-4)

    for i in (900, 1000, 1036):
        truth = _true_row_distances(T, m, i, excl)
        assert abs(mp.P_[i] - truth.min()) <= tie, (
            f"row {i}: ours {mp.P_[i]:.4f} truth {truth.min():.4f}"
        )
        assert abs(truth[mp.I_[i]] - truth.min()) <= tie


def test_explicit_equal_t_b_isconstant_honored():
    """T_B_subseq_isconstant applies to the target side even when T_B equals
    T_A (previously it was silently dropped).

    Scope: rows ABOVE the flagged window index. mlx-stump applies the flag
    row-wise everywhere; STUMPY's diagonal traversal computes each self-join
    pair (i < j) once — A-stats on i, B-stats on j — and mirrors it, so with
    asymmetric flags its rows BELOW a flagged target window inherit the
    unflagged distance (stumpy.stump row 70 reports d(70,30)=1.25 while
    stumpy.mass with the same flag says sqrt(m)=2.83). We keep the
    self-consistent row-wise semantics for that corner.
    """
    rng = np.random.default_rng(1)
    T = rng.standard_normal(80)
    l = 73
    flag = np.zeros(l, dtype=bool)
    flag[30] = True
    mp = mlx_stump.stump(T, 8, T.copy(), ignore_trivial=True, T_B_subseq_isconstant=flag)
    ref = stumpy.stump(T, 8, T.copy(), ignore_trivial=True, T_B_subseq_isconstant=flag)
    plain = mlx_stump.stump(T, 8)
    # the flag changed the result at all (it used to be ignored)
    assert not np.array_equal(mp.I_, plain.I_)
    # in the region where both semantics agree (queries before the flagged
    # window), match stumpy exactly
    sl = slice(0, 30)
    tie = tie_tolerance(8)
    assert_indices_tie_tolerant(mp.I_[sl], ref.I_[sl], T, T, 8, tie_atol=tie)
    np.testing.assert_allclose(
        mp.P_[sl][mp.I_[sl] == ref.I_[sl]],
        ref.P_[sl][mp.I_[sl] == ref.I_[sl]],
        atol=1e-6,
    )


def test_self_join_ignore_trivial_false_warns():
    T = DATASETS["white_noise"](200, seed=2)
    with pytest.warns(UserWarning, match="cannot be `False` for a self-join"):
        mp = mlx_stump.stump(T, 8, ignore_trivial=False)
    ref = mlx_stump.stump(T, 8)
    np.testing.assert_array_equal(mp.I_, ref.I_)


def test_window_too_large_warns():
    T = DATASETS["white_noise"](12, seed=3)
    with pytest.warns(UserWarning, match="may be too large"):
        mlx_stump.stump(T, 8)


def test_isconstant_nan_conflict_warns():
    T = DATASETS["white_noise"](60, seed=4)
    T[2] = np.nan
    flag = np.zeros(53, dtype=bool)
    flag[0] = True  # window 0 contains the NaN at index 2
    with pytest.warns(UserWarning, match="automatically switched"):
        mlx_stump.stump(T, 8, T_A_subseq_isconstant=flag)


def test_chunk_size_validation():
    T = DATASETS["white_noise"](50, seed=5)
    for bad in (-5, 0, 2.5):
        with pytest.raises(ValueError, match="chunk_size"):
            mlx_stump.stump(T, 8, chunk_size=bad)


def test_match_tight_max_distance_finds_exact_occurrence():
    """A planted exact occurrence must survive a tight max_distance: the
    float32 profile reads ~1e-3 there, so match re-evaluates candidates in
    float64 before thresholding."""
    rng = np.random.default_rng(7)
    T = rng.standard_normal(200)
    pat = T[50:60].copy()
    M = mlx_stump.match(pat, T, max_distance=1e-3)
    Mr = stumpy.match(pat, T, max_distance=1e-3)
    assert M.shape == Mr.shape
    assert int(M[0, 1]) == int(Mr[0, 1]) == 50
    assert float(M[0, 0]) < 1e-6


def test_mass_column_query_flattened():
    T = DATASETS["white_noise"](100, seed=8)
    Q = T[10:20].copy()
    with pytest.warns(UserWarning, match="flattened"):
        D = mlx_stump.mass(Q.reshape(-1, 1), T)
    np.testing.assert_array_equal(D, mlx_stump.mass(Q, T))


def test_mass_isfinite_override_ignored_when_normalized():
    """STUMPY documents T_subseq_isfinite as ignored for normalize=True."""
    rng = np.random.default_rng(9)
    T = rng.standard_normal(60)
    Q = T[10:15].copy()
    flag = np.zeros(56, dtype=bool)
    D = mlx_stump.mass(Q, T, T_subseq_isfinite=flag)
    Dr = stumpy.mass(Q, T, T_subseq_isfinite=flag)
    assert np.all(np.isfinite(D))
    np.testing.assert_allclose(D, Dr, atol=1e-3)


def test_ab_join_no_window_warning():
    # the too-large advisory is a self-join concept; AB-joins stay silent
    T_A = DATASETS["white_noise"](12, seed=10)
    T_B = DATASETS["white_noise"](40, seed=11)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        mlx_stump.stump(T_A, 8, T_B, ignore_trivial=False)
