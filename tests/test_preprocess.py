"""Unit tests for the float64 CPU preprocessing layer."""

from __future__ import annotations

import numpy as np
import pytest
import stumpy

from mlx_stump._preprocess import (
    check_series,
    check_window_size,
    preprocess_series,
    rolling_isconstant,
    rolling_isfinite,
    rolling_mean_sigma,
)

from .conftest import DATASETS


@pytest.mark.parametrize("name", sorted(DATASETS))
@pytest.mark.parametrize("m", [3, 7, 50])
def test_rolling_isconstant_matches_stumpy(name, m):
    T = DATASETS[name](500, seed=40)
    T_nan = np.where(np.isinf(T), np.nan, T)
    ours = rolling_isconstant(T_nan, m)
    ref = stumpy.core.rolling_isconstant(T_nan, m)
    np.testing.assert_array_equal(ours, ref)


@pytest.mark.parametrize("m", [3, 16, 128])
def test_rolling_mean_sigma_exact(m):
    T = DATASETS["random_walk"](800, seed=41)
    mu, sigma = rolling_mean_sigma(T, m)
    W = np.lib.stride_tricks.sliding_window_view(T, m)
    np.testing.assert_allclose(mu, W.mean(axis=1), atol=1e-9, rtol=1e-12)
    np.testing.assert_allclose(sigma, W.std(axis=1), atol=1e-9, rtol=1e-9)


def test_rolling_isfinite():
    T = DATASETS["with_nans"](300, seed=42)
    m = 10
    finite_pt = np.isfinite(T)
    ours = rolling_isfinite(finite_pt, m)
    W = np.lib.stride_tricks.sliding_window_view(finite_pt, m)
    np.testing.assert_array_equal(ours, W.all(axis=1))


def test_raw_preprocess_standardization_and_masks():
    T = DATASETS["with_constants"](600, seed=43)
    m = 12
    prep = preprocess_series(T, m, normalize=False)
    assert prep.l == T.shape[0] - m + 1
    # standardized series has ~zero mean / unit variance over finite values
    assert abs(prep.Ts.mean()) < 1e-9
    assert abs(prep.Ts.std() - 1.0) < 1e-9
    # constant windows get sigma_inv == 0
    assert np.all(prep.sig_inv[prep.isconstant] == 0.0)
    # original series is untouched
    np.testing.assert_array_equal(prep.T, T)


def test_normalized_preprocess_keeps_only_locally_consumed_arrays():
    """Local window normalization must not retain obsolete global stats."""
    T = DATASETS["with_constants"](600, seed=43)
    prep = preprocess_series(T, 12)

    assert prep.Ts is None
    assert prep.mu is None
    assert prep.sig_inv is None
    assert prep.ssq is None
    assert prep.mu_mx is None
    assert prep.sig_inv_mx is not None
    np.testing.assert_array_equal(prep.T, T)


def test_preprocess_nonfinite_masks_match_stumpy():
    T = DATASETS["with_nans"](400, seed=44)
    m = 9
    prep = preprocess_series(T, m)
    ref_finite = np.all(np.isfinite(np.lib.stride_tricks.sliding_window_view(T, m)), axis=1)
    np.testing.assert_array_equal(prep.isfinite, ref_finite)
    # NaN windows are never constant
    assert not np.any(prep.isconstant & ~prep.isfinite)


def test_check_series_validation():
    with pytest.raises(TypeError, match="float64"):
        check_series(np.zeros(10, dtype=np.float32), "T")
    with pytest.raises(TypeError, match="float64"):
        check_series(np.zeros(10, dtype=np.int64), "T")
    with pytest.raises(ValueError, match="1-dimensional"):
        check_series(np.zeros((5, 2)), "T")
    out = check_series(np.zeros(10), "T")
    assert out.dtype == np.float64


def test_check_window_size():
    with pytest.raises(ValueError, match="three"):
        check_window_size(2)
    with pytest.raises(ValueError, match="less than or equal"):
        check_window_size(20, 10)
    with pytest.raises(TypeError):
        check_window_size(8.0)
    assert check_window_size(3, 3) == 3
