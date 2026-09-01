"""Golden tests: mass and match vs STUMPY."""

from __future__ import annotations

import numpy as np
import pytest
import stumpy

import mlx_stump

from .conftest import DATASETS, assert_dist_profiles_close


@pytest.mark.parametrize("name", ["random_walk", "sine_noise", "large_offset"])
def test_mass_golden(name):
    T = DATASETS[name](3000, seed=30)
    m = 64
    Q = T[500 : 500 + m].copy()
    D = mlx_stump.mass(Q, T)
    Dr = stumpy.mass(Q, T)
    assert D.dtype == np.float64 and D.shape == Dr.shape
    assert_dist_profiles_close(D, Dr, m=m)


def test_mass_constant_query():
    T = DATASETS["with_constants"](1000, seed=31)
    m = 20
    Q = np.full(m, 2.5)
    D = mlx_stump.mass(Q, T)
    Dr = stumpy.mass(Q, T)
    # constant-vs-constant -> 0, constant-vs-not -> sqrt(m): exact semantics
    np.testing.assert_allclose(D, Dr, atol=1e-5, rtol=0)


def test_mass_nan_in_T():
    T = DATASETS["with_nans"](1000, seed=32)
    m = 16
    Q = T[100 : 100 + m].copy()
    assert np.all(np.isfinite(Q))
    D = mlx_stump.mass(Q, T)
    Dr = stumpy.mass(Q, T)
    assert_dist_profiles_close(D, Dr, m=m)


def test_mass_nan_in_Q():
    T = DATASETS["white_noise"](500, seed=33)
    Q = T[10:26].copy()
    Q[3] = np.nan
    D = mlx_stump.mass(Q, T)
    Dr = stumpy.mass(Q, T)
    assert np.all(np.isinf(D))
    np.testing.assert_array_equal(D, Dr)


def test_mass_query_idx():
    T = DATASETS["random_walk"](800, seed=34)
    m = 32
    Q = T[200 : 200 + m].copy()
    D = mlx_stump.mass(Q, T, query_idx=200)
    assert D[200] == 0.0
    with pytest.warns(UserWarning, match="query_idx"):
        mlx_stump.mass(Q, T, query_idx=100)


def test_mass_precomputed_stats():
    T = DATASETS["sine_noise"](600, seed=35)
    m = 24
    Q = T[50 : 50 + m].copy()
    M_T, Σ_T = stumpy.core.compute_mean_std(T, m)
    D = mlx_stump.mass(Q, T, M_T=M_T, Σ_T=Σ_T)
    Dr = stumpy.mass(Q, T, M_T=M_T, Σ_T=Σ_T)
    assert_dist_profiles_close(D, Dr, m=m)


def test_mass_absolute():
    T = DATASETS["random_walk"](1000, seed=36)
    m = 32
    Q = T[300 : 300 + m].copy()
    D = mlx_stump.mass(Q, T, normalize=False)
    Dr = stumpy.mass(Q, T, normalize=False)
    assert_dist_profiles_close(D, Dr, m=m, scale=float(np.std(T)))
    with pytest.raises(NotImplementedError):
        mlx_stump.mass(Q, T, normalize=False, p=3.0)


def test_match_golden():
    T = DATASETS["sine_noise"](2000, seed=37)
    m = 50
    Q = T[400 : 400 + m].copy()
    M = mlx_stump.match(Q, T)
    Mr = stumpy.match(Q, T)
    assert M.dtype == object
    assert M.shape == Mr.shape
    np.testing.assert_array_equal(M[:, 1].astype(np.int64), Mr[:, 1].astype(np.int64))
    assert_dist_profiles_close(M[:, 0].astype(np.float64), Mr[:, 0].astype(np.float64), m=m)


def test_match_max_matches_and_distance():
    T = DATASETS["sine_noise"](2000, seed=38)
    m = 50
    Q = T[400 : 400 + m].copy()
    # a deterministic threshold: the data-dependent default is fp32-boundary
    # sensitive (covered by test_match_golden), the cap semantics are not
    M = mlx_stump.match(Q, T, max_matches=3, max_distance=float("inf"))
    Mr = stumpy.match(Q, T, max_matches=3, max_distance=float("inf"))
    assert M.shape == Mr.shape == (3, 2)
    np.testing.assert_array_equal(M[:, 1].astype(np.int64), Mr[:, 1].astype(np.int64))

    M2 = mlx_stump.match(Q, T, max_distance=10.0)
    Mr2 = stumpy.match(Q, T, max_distance=10.0)
    # boundary subsequences may flip in float32; the leading matches agree
    n_common = min(M2.shape[0], Mr2.shape[0])
    assert abs(M2.shape[0] - Mr2.shape[0]) <= 2
    np.testing.assert_array_equal(
        M2[:n_common, 1].astype(np.int64), Mr2[:n_common, 1].astype(np.int64)
    )

    M3 = mlx_stump.match(Q, T, max_distance=lambda D: float(np.nanmin(D)) + 0.5)
    Mr3 = stumpy.match(Q, T, max_distance=lambda D: float(np.nanmin(D)) + 0.5)
    assert abs(M3.shape[0] - Mr3.shape[0]) <= 2


def test_match_nan_query_raises():
    T = DATASETS["white_noise"](300, seed=39)
    Q = T[10:26].copy()
    Q[0] = np.nan
    with pytest.raises(ValueError, match="illegal"):
        mlx_stump.match(Q, T)
