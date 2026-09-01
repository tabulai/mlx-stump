"""Golden tests: top-k profiles (k > 1) and the non-normalized path."""

from __future__ import annotations

import numpy as np
import pytest
import stumpy

import mlx_stump

from .conftest import (
    DATASETS,
    assert_indices_tie_tolerant,
    assert_profile_close,
    tie_tolerance,
)


@pytest.mark.parametrize("k", [2, 3])
def test_topk_self_join(k):
    T = DATASETS["random_walk"](1500, seed=20)
    m = 32
    mp = mlx_stump.stump(T, m, k=k)
    ref = stumpy.stump(T, m, k=k)
    tie = tie_tolerance(m)
    assert mp.shape == ref.shape == (T.shape[0] - m + 1, 2 * k + 2)
    assert mp.P_.shape == (T.shape[0] - m + 1, k)
    for j in range(k):
        # k-th neighbor sets are tie-sensitive; values must still agree
        assert_profile_close(
            mp.P_[:, j],
            ref.P_[:, j],
            m=m,
            exact_mask=(mp.I_[:, j] == ref.I_[:, j]),
            tie_atol=tie,
        )
        assert np.mean(mp.I_[:, j] == ref.I_[:, j]) >= 0.90
    # per-row values must be sorted ascending (inf-inf pairs diff to nan)
    diffs = np.diff(np.asarray(mp.P_, dtype=np.float64), axis=1)
    assert np.all(diffs[~np.isnan(diffs)] >= -1e-12)
    assert_indices_tie_tolerant(mp.left_I_, ref.left_I_, T, T, m, tie_atol=tie)
    assert_indices_tie_tolerant(mp.right_I_, ref.right_I_, T, T, m, tie_atol=tie)


def test_topk_exceeds_candidates():
    # l = 5 with a wide exclusion zone: fewer than k valid neighbors per row
    T = DATASETS["white_noise"](36, seed=21)
    m = 32
    k = 4
    mp = mlx_stump.stump(T, m, k=k)
    ref = stumpy.stump(T, m, k=k)
    assert mp.shape == ref.shape
    np.testing.assert_array_equal(np.isinf(mp.P_), np.isinf(ref.P_))
    np.testing.assert_array_equal(mp.I_ == -1, ref.I_ == -1)


@pytest.mark.parametrize("name", ["random_walk", "with_nans", "large_offset"])
def test_aamp_self_join(name):
    T = DATASETS[name](1500, seed=22)
    m = 40
    mp = mlx_stump.stump(T, m, normalize=False)
    ref = stumpy.stump(T, m, normalize=False)
    tie = tie_tolerance(m) * max(1.0, np.nanstd(T[np.isfinite(T)]))
    assert_indices_tie_tolerant(mp.I_, ref.I_, T, T, m, tie_atol=tie, normalize=False)
    assert_profile_close(mp.P_, ref.P_, m=m, exact_mask=(mp.I_ == ref.I_), tie_atol=tie)
    assert_indices_tie_tolerant(mp.left_I_, ref.left_I_, T, T, m, tie_atol=tie, normalize=False)
    assert_indices_tie_tolerant(mp.right_I_, ref.right_I_, T, T, m, tie_atol=tie, normalize=False)


def test_aamp_ab_join():
    import warnings

    T_A = DATASETS["sine_noise"](800, seed=23)
    T_B = DATASETS["sine_noise"](1000, seed=24)
    m = 25
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mp = mlx_stump.stump(T_A, m, T_B, ignore_trivial=False, normalize=False)
        ref = stumpy.stump(T_A, m, T_B, ignore_trivial=False, normalize=False)
    tie = tie_tolerance(m)
    assert_indices_tie_tolerant(mp.I_, ref.I_, T_A, T_B, m, tie_atol=tie, normalize=False)
    assert_profile_close(mp.P_, ref.P_, m=m, exact_mask=(mp.I_ == ref.I_), tie_atol=tie)
    assert np.all(mp.left_I_ == -1) and np.all(mp.right_I_ == -1)


def test_aamp_p_not_2_raises():
    T = DATASETS["white_noise"](200, seed=25)
    with pytest.raises(NotImplementedError, match="p=2.0"):
        mlx_stump.stump(T, 8, normalize=False, p=1.0)
