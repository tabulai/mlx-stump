"""Regression tests for scale-safe preprocessing and refinement."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import mlx_stump


def _scaled_fixture():
    rng = np.random.default_rng(919)
    T = rng.integers(-100, 101, 400).astype(np.float64)
    m = 23
    Q = T[31 : 31 + m].copy()
    T[250 : 250 + m] = Q
    return Q, T, m


@pytest.mark.parametrize("exponent", [-1_000, -700, 700, 1_000])
def test_normalized_public_api_is_invariant_to_absolute_power_of_two_scale(exponent):
    """Changing only finite units must not underflow/overflow window stats."""
    Q, T, m = _scaled_fixture()
    D_ref = mlx_stump.mass(Q, T)
    mp_ref = mlx_stump.stump(T, m)
    matches_ref = mlx_stump.match(Q, T, max_distance=3.0)

    factor = np.ldexp(1.0, exponent)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        D = mlx_stump.mass(Q * factor, T * factor)
        mp = mlx_stump.stump(T * factor, m)
        matches = mlx_stump.match(Q * factor, T * factor, max_distance=3.0)

    np.testing.assert_array_equal(D, D_ref)
    np.testing.assert_array_equal(mp.P_, mp_ref.P_)
    np.testing.assert_array_equal(mp.I_, mp_ref.I_)
    np.testing.assert_array_equal(mp.left_I_, mp_ref.left_I_)
    np.testing.assert_array_equal(mp.right_I_, mp_ref.right_I_)
    np.testing.assert_array_equal(matches.astype(np.float64), matches_ref.astype(np.float64))
    assert D[31] == D[250] == 0.0


def test_normalized_scale_invariance_fuzzes_float64_exponent_range():
    rng = np.random.default_rng(20260901)
    T = rng.standard_normal(240)
    m = 17
    Q = T[20 : 20 + m].copy()
    T[170 : 170 + m] = Q
    D_ref = mlx_stump.mass(Q, T)
    mp_ref = mlx_stump.stump(T, m)

    for _ in range(24):
        exponent = int(rng.integers(-1_000, 1_001))
        factor = np.ldexp(float(rng.uniform(0.5, 1.0)), exponent)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            D = mlx_stump.mass(Q * factor, T * factor)
            mp = mlx_stump.stump(T * factor, m)
            matches = mlx_stump.match(
                Q * factor, T * factor, max_distance=0.0, atol=0.0
            )
        np.testing.assert_array_equal(D, D_ref)
        np.testing.assert_array_equal(mp.I_, mp_ref.I_)
        np.testing.assert_allclose(mp.P_, mp_ref.P_, atol=2e-14, rtol=0.0)
        np.testing.assert_array_equal(
            matches.astype(np.float64), np.array([[0.0, 20.0], [0.0, 170.0]])
        )


def test_normalized_ab_join_allows_independent_extreme_units():
    rng = np.random.default_rng(44)
    T_A = rng.standard_normal(180)
    T_B = rng.standard_normal(230)
    m = 17
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # explicit AB-join advisory parity
        ref = mlx_stump.stump(T_A, m, T_B, ignore_trivial=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = mlx_stump.stump(
            T_A * np.ldexp(1.0, -1_000),
            m,
            T_B * np.ldexp(1.0, 1_000),
            ignore_trivial=False,
        )
    np.testing.assert_array_equal(got.I_, ref.I_)
    np.testing.assert_array_equal(got.P_, ref.P_)


def test_user_constant_override_does_not_emit_dynamic_range_warning():
    """An intentional zero sigma is not evidence of precision loss."""
    rng = np.random.default_rng(184)
    T = rng.standard_normal(120)
    m = 9
    flags = np.zeros(T.size - m + 1, dtype=bool)
    flags[20] = True

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mlx_stump.stump(T, m, T_A_subseq_isconstant=flags)


def test_normalized_apis_handle_opposite_sign_near_float64_max():
    """The final affine subtraction must not overflow on finite endpoints."""
    m = 8
    low, high = -1e308, 1e308
    Q = np.array([low, low, high, low, low, high, low, low], dtype=np.float64)
    T = np.full(80, low, dtype=np.float64)
    T[10 : 10 + m] = Q
    T[50 : 50 + m] = Q

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.filterwarnings(
            "ignore", message="A large number of values in `P`", category=UserWarning
        )
        D = mlx_stump.mass(Q, T)
        matches = mlx_stump.match(Q, T, max_distance=0.0, atol=0.0)
        mp = mlx_stump.stump(T, m)

    assert np.all(np.isfinite(D))
    assert D[10] == D[50] == 0.0
    np.testing.assert_array_equal(
        matches.astype(np.float64), np.array([[0.0, 10.0], [0.0, 50.0]])
    )
    assert np.all(np.isfinite(mp.P_))


def test_raw_mass_respects_theoretical_normalized_distance_maximum():
    Q = np.array(
        [
            0.3398753611927038,
            0.31604787064428624,
            0.40982845732869266,
            0.6161346680022383,
            -2.107953341777443,
            -0.3644382522644612,
            -2.180210064489474,
        ],
        dtype=np.float64,
    )
    T = -0.03393451028289679 * Q - 0.0035504234359361063
    got = mlx_stump.mass(Q, T)[0]
    assert got == 2.0 * np.sqrt(Q.size)


def test_subnormal_units_remain_well_conditioned():
    Q, T, m = _scaled_fixture()
    factor = np.nextafter(0.0, 1.0)
    D_ref = mlx_stump.mass(Q, T)
    mp_ref = mlx_stump.stump(T, m)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        D = mlx_stump.mass(Q * factor, T * factor)
        mp = mlx_stump.stump(T * factor, m)
        matches = mlx_stump.match(
            Q * factor, T * factor, max_distance=0.0, atol=0.0
        )
    # The global center/scale themselves live on the coarse subnormal lattice,
    # so raw float32 MASS can move by its normal ~1e-3 floor. The refined APIs
    # and neighbor ranking remain scale-invariant.
    np.testing.assert_allclose(D, D_ref, atol=2e-3, rtol=0.0)
    np.testing.assert_array_equal(mp.I_, mp_ref.I_)
    np.testing.assert_allclose(mp.P_, mp_ref.P_, atol=2e-14, rtol=0.0)
    np.testing.assert_array_equal(
        matches.astype(np.float64), np.array([[0.0, 31.0], [0.0, 250.0]])
    )


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_point_does_not_poison_later_huge_offset_windows(bad):
    """The standardized sentinel is zero even when raw zero is enormous."""
    rng = np.random.default_rng(5)
    m = 16
    clean = 1e15 + 0.25 * rng.integers(-20, 21, 800).astype(np.float64)
    Q = clean[300 : 300 + m].copy()
    clean[600 : 600 + m] = Q
    T = clean.copy()
    bad_idx = 80
    T[bad_idx] = bad

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        D = mlx_stump.mass(Q, T)
        D_clean = mlx_stump.mass(Q, clean)
        matches = mlx_stump.match(Q, T, max_distance=0.0, atol=0.0)
        mp = mlx_stump.stump(T, m)

    affected = np.zeros(D.size, dtype=bool)
    affected[max(0, bad_idx - m + 1) : bad_idx + 1] = True
    assert np.all(np.isinf(D[affected]))
    assert np.all(np.isfinite(D[~affected]))
    # Removing one finite point from the global affine frame may change the
    # float32 profile's last bits, but not its well-conditioned distances.
    np.testing.assert_allclose(D[~affected], D_clean[~affected], atol=2e-3, rtol=0.0)
    np.testing.assert_array_equal(
        matches.astype(np.float64), np.array([[0.0, 300.0], [0.0, 600.0]])
    )
    assert np.count_nonzero(np.isfinite(mp.P_)) == D.size - np.count_nonzero(affected)


def test_absolute_nonfinite_override_preserves_raw_zero_fill_contract():
    rng = np.random.default_rng(0)
    T = 100.0 + rng.standard_normal(40)
    m = 5
    Q = T[20 : 20 + m].copy()
    bad_idx = 10
    T[bad_idx] = np.nan
    flags = np.ones(T.size - m + 1, dtype=bool)

    D = mlx_stump.mass(
        Q, T, normalize=False, T_subseq_isfinite=flags
    )
    T_zero = np.where(np.isfinite(T), T, 0.0)
    windows = np.lib.stride_tricks.sliding_window_view(T_zero, m)
    affected = np.arange(bad_idx - m + 1, bad_idx + 1)
    truth = np.sqrt(np.sum((windows[affected] - Q) ** 2, axis=1))
    np.testing.assert_allclose(D[affected], truth, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize("exponent", [-700, 700])
def test_non_normalized_mass_and_match_remain_in_raw_units(exponent):
    Q, T, _ = _scaled_fixture()
    D_ref = mlx_stump.mass(Q, T, normalize=False)
    mp_ref = mlx_stump.stump(T, Q.size, normalize=False)
    matches_ref = mlx_stump.match(
        Q, T, normalize=False, max_distance=100.0, atol=0.0
    )
    factor = np.ldexp(1.0, exponent)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.filterwarnings(
            "ignore", message="A large number of values in `P`", category=UserWarning
        )
        D = mlx_stump.mass(Q * factor, T * factor, normalize=False)
        mp = mlx_stump.stump(T * factor, Q.size, normalize=False)
        matches = mlx_stump.match(
            Q * factor,
            T * factor,
            normalize=False,
            max_distance=100.0 * factor,
            atol=0.0,
        )

    np.testing.assert_array_equal(D / factor, D_ref)
    np.testing.assert_array_equal(mp.I_, mp_ref.I_)
    np.testing.assert_array_equal(mp.left_I_, mp_ref.left_I_)
    np.testing.assert_array_equal(mp.right_I_, mp_ref.right_I_)
    np.testing.assert_array_equal(mp.P_ / factor, mp_ref.P_)
    np.testing.assert_array_equal(matches[:, 1], matches_ref[:, 1])
    np.testing.assert_array_equal(
        matches[:, 0].astype(np.float64) / factor,
        matches_ref[:, 0].astype(np.float64),
    )


@pytest.mark.parametrize("exponent", [-700, 700])
def test_raw_default_match_threshold_is_scale_safe(exponent):
    rng = np.random.default_rng(0)
    T = rng.standard_normal(200).astype(np.float64)
    Q = T[:10].copy()
    # `atol` is an absolute user-unit tolerance in the STUMPY API, so hold it
    # at zero while isolating the data-derived default threshold's invariance.
    reference = mlx_stump.match(Q, T, normalize=False, atol=0.0)
    factor = np.ldexp(1.0, exponent)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = mlx_stump.match(Q * factor, T * factor, normalize=False, atol=0.0)

    np.testing.assert_array_equal(got[:, 1], reference[:, 1])
    np.testing.assert_allclose(
        got[:, 0].astype(np.float64) / factor,
        reference[:, 0].astype(np.float64),
        rtol=2e-15,
        atol=0.0,
    )
