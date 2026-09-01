"""Golden tests: AB-joins (matrix profile of T_A relative to T_B)."""

from __future__ import annotations

import warnings

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


def golden_ab_join(T_A, T_B, m, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mp = mlx_stump.stump(T_A, m, T_B, ignore_trivial=False, **kwargs)
        ref = stumpy.stump(T_A, m, T_B, ignore_trivial=False, **kwargs)
    tie = tie_tolerance(m)
    assert mp.shape == ref.shape
    assert_indices_tie_tolerant(
        mp.I_, ref.I_, T_A, T_B, m, tie_atol=tie, normalize=kwargs.get("normalize", True)
    )
    assert_profile_close(mp.P_, ref.P_, m=m, exact_mask=(mp.I_ == ref.I_), tie_atol=tie)
    # STUMPY convention: AB-joins carry no left/right profiles
    np.testing.assert_array_equal(mp.left_I_, ref.left_I_)
    np.testing.assert_array_equal(mp.right_I_, ref.right_I_)
    assert np.all(mp.left_I_ == -1) and np.all(mp.right_I_ == -1)
    return mp, ref


@pytest.mark.parametrize("m", [8, 50])
def test_ab_join_basic(m):
    T_A = DATASETS["random_walk"](1200, seed=10)
    T_B = DATASETS["random_walk"](1700, seed=11)
    golden_ab_join(T_A, T_B, m)


def test_ab_join_with_nans():
    T_A = DATASETS["with_nans"](900, seed=12)
    T_B = DATASETS["with_nans"](1100, seed=13)
    golden_ab_join(T_A, T_B, 16)


def test_ab_join_shorter_b():
    T_A = DATASETS["sine_noise"](1500, seed=14)
    T_B = DATASETS["sine_noise"](400, seed=15)
    golden_ab_join(T_A, T_B, 24)


def test_ab_join_warns_and_overrides_ignore_trivial():
    T_A = DATASETS["white_noise"](300, seed=16)
    T_B = DATASETS["white_noise"](300, seed=17)
    with pytest.warns(UserWarning, match="AB-join"):
        mp = mlx_stump.stump(T_A, 8, T_B, ignore_trivial=True)
    assert np.all(mp.left_I_ == -1)


def test_equal_arrays_ignore_trivial_false_stays_ab_join():
    # STUMPY only warns here and runs the degenerate AB-join (every row
    # trivially matches itself); replicate exactly
    T = DATASETS["white_noise"](300, seed=18)
    with pytest.warns(UserWarning, match="Try setting"):
        mp = mlx_stump.stump(T, 8, T.copy(), ignore_trivial=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = stumpy.stump(T, 8, T.copy(), ignore_trivial=False)
    np.testing.assert_array_equal(mp.I_, ref.I_)
    np.testing.assert_allclose(mp.P_, ref.P_, atol=1e-6)
    assert np.all(mp.left_I_ == -1) and np.all(mp.right_I_ == -1)
