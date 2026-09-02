"""Focused regression tests for float64 affine-match refinement."""

import numpy as np
import pytest
import stumpy

import mlx_stump
from mlx_stump._match import (
    _exact_positive_affine_rows,
    _exact_translation_rows,
    _refine_candidates,
)


@pytest.mark.parametrize(
    "m,seed,scale,offset,denom_pow",
    [
        (3, 834, 17, 65_536, 0),
        (50, 204, 17, 65_536, 4),
        (256, 41, 17, 65_536, 12),
        (512, 137, 17, 65_536, 0),
        (1_024, 14, 17, 65_536, 20),
    ],
)
def test_exact_representable_positive_affine_rows_refine_to_zero(
    m, seed, scale, offset, denom_pow
):
    """Independent normalization may differ by ulps, but not semantically.

    Integer numerators certify that both ``Q`` and ``scale * Q + offset``
    are represented exactly as binary64 values.  The larger power-of-two
    window sizes exercise NumPy's different reduction kernels; these cases
    left residues as large as 2.2e-13 before the pairwise norm and narrow
    roundoff snap were introduced.
    """
    qi = np.random.default_rng(seed).integers(-4_096, 4_097, size=m, dtype=np.int64)
    Q = np.ldexp(qi.astype(np.float64), -denom_pow)
    numer = scale * qi + offset * (1 << denom_pow)
    assert np.max(np.abs(numer)) < 2**53
    exact_W = np.ldexp(numer.astype(np.float64), -denom_pow)
    W = scale * Q + offset
    np.testing.assert_array_equal(W, exact_W)

    got = _refine_candidates(Q, W, [0], True, False, np.array([False]))
    np.testing.assert_array_equal(got, np.array([0.0]))


def test_strict_match_snaps_only_roundoff_not_near_or_negative_affine_rows():
    """A zero threshold admits the true positive-affine occurrence only."""
    m = 64
    Q = np.random.default_rng(0).integers(-256, 257, size=m).astype(np.float64)
    T = np.random.default_rng(99).integers(-500, 501, size=900).astype(np.float64)
    exact_idx, near_idx, negative_idx = 80, 380, 680

    T[exact_idx : exact_idx + m] = 3.0 * Q + 7.0
    T[near_idx : near_idx + m] = 3.0 * Q + 7.0
    T[near_idx + m // 2] += 1e-3
    T[negative_idx : negative_idx + m] = -3.0 * Q + 7.0

    refined = _refine_candidates(
        Q,
        T,
        [exact_idx, near_idx, negative_idx],
        True,
        False,
        np.zeros(T.size - m + 1, dtype=bool),
    )
    assert refined[0] == 0.0
    assert 0.0 < refined[1] < 1e-5
    assert refined[2] == pytest.approx(2.0 * np.sqrt(m), abs=1e-14)

    kwargs = {"max_distance": 0.0, "atol": 0.0}
    got = mlx_stump.match(Q, T, **kwargs)
    reference = stumpy.match(Q, T, **kwargs)
    np.testing.assert_array_equal(got.astype(np.float64), reference.astype(np.float64))
    np.testing.assert_array_equal(got.astype(np.float64), np.array([[0.0, exact_idx]]))


def test_local_normalized_search_preserves_tiny_exact_window_in_large_range():
    """Global standardization must not re-round away a tiny exact match."""
    u = np.spacing(1.0)
    Q = np.array([0.0, u, -u])
    background = np.array(
        [
            1.0757937597341474,
            0.5921040871780917,
            -1.2174679563370514,
            1.4268670549102678,
            0.7834191059710589,
            0.8581929158308614,
            -1.1156591019733624,
            -0.1488421863132986,
            -0.3876059273022563,
            1.2802949665458054,
            0.43159536024199363,
            0.9682848398124899,
        ]
    )
    exact_idx = 6
    for exact in (Q, Q + 1.0):
        T = np.concatenate((background[:exact_idx], exact, background[exact_idx:]))
        # `mass` exposes the float32 search profile, whose identical-vector
        # self-dot may land an ulp below m. It must stay inside the ordinary
        # admission margin; the public refined APIs below guarantee exact 0.
        assert mlx_stump.mass(exact, T)[exact_idx] < 0.05
        matches = mlx_stump.match(exact, T, max_distance=0.0, atol=0.0)
        np.testing.assert_array_equal(
            matches.astype(np.float64), np.array([[0.0, exact_idx]])
        )


def test_local_normalized_stump_does_not_discard_tiny_exact_neighbor():
    """The GPU top-k search must retain the exact row before refinement."""
    u = np.spacing(1.0)
    Q = np.array([0.0, u, -u])
    decoy = np.array([0.0, 2.0 * u, -3.0 * u])
    background = np.array(
        [
            1.0757937597341474,
            0.5921040871780917,
            -1.2174679563370514,
            1.4268670549102678,
            0.7834191059710589,
            0.8581929158308614,
            -1.1156591019733624,
            -0.1488421863132986,
            -0.3876059273022563,
            1.2802949665458054,
            0.43159536024199363,
            0.9682848398124899,
        ]
    )
    T = np.concatenate((background[:6], Q, [-0.5], decoy, background[6:]))
    with pytest.warns(UserWarning, match="large number of values"):
        profile = mlx_stump.stump(Q, 3, T_B=T, ignore_trivial=False)
    assert float(profile[0, 0]) == 0.0
    assert int(profile[0, 1]) == 6


def test_decimal_subnormal_result_ignores_caller_underflow_policy():
    """A positive Decimal result below binary64 range stays non-zero, not an error."""
    rng = np.random.default_rng(8888)
    for _ in range(5):
        m = int(rng.choice([3, 4, 8, 17, 33]))
        n = int(rng.integers(2 * m + 10, 150))
        with np.errstate(all="ignore"):
            T = np.ldexp(
                rng.uniform(-1.0, 1.0, n), rng.integers(-1074, 1024, size=n)
            )
        T = np.where(np.isfinite(T), T, np.sign(T) * np.finfo(float).max)
        query_idx = int(rng.integers(0, n - m + 1))
    assert (m, n, query_idx) == (3, 134, 93)
    Q = T[query_idx : query_idx + m].copy()

    previous = np.seterr(all="raise")
    try:
        matches = mlx_stump.match(Q, T, max_distance=np.inf, max_matches=3)
    finally:
        np.seterr(**previous)
    assert matches.shape == (3, 2)
    distances = np.asarray(matches[:, 0], dtype=np.float64)
    assert distances[0] == 0.0
    np.testing.assert_array_equal(distances[1:], np.full(2, np.nextafter(0.0, 1.0)))


@pytest.mark.parametrize("m", [3, 8, 64, 1_024])
def test_one_ulp_non_affine_row_is_not_zero(m):
    """The roundoff envelope alone must never certify an affine match.

    Changing one interior value by one ULP breaks affinity: at least two
    unchanged, distinct points force ``a=1`` and ``b=0``.  Its true distance
    is tiny but non-zero, so a strict zero threshold must reject it.
    """
    Q = np.arange(m, dtype=np.float64)
    W = Q.copy()
    W[m // 2] = np.nextafter(W[m // 2], np.inf)

    np.testing.assert_array_equal(_exact_positive_affine_rows(Q, W), np.array([False]))
    refined = _refine_candidates(Q, W, [0], True, False, np.array([False]))
    assert 0.0 < refined[0] < 1e-13


def test_one_ulp_perturbation_survives_binary64_normalization_collapse():
    """The high-precision fallback also covers a literal d2 == 0 collapse."""
    m = 64
    Q = np.random.default_rng(2).standard_normal(m)
    W = Q.copy()
    W[2] = np.nextafter(W[2], np.inf)

    # Demonstrate why merely withholding the affine zero-snap is not enough:
    # ordinary binary64 normalization makes the two normalized rows bitwise
    # identical and would still report a false zero.
    q = Q - Q[0]
    q -= q.mean()
    q /= np.sqrt(np.sum(q * q) / m)
    w = W - W[0]
    w -= w.mean()
    w /= np.sqrt(np.sum(w * w) / m)
    assert np.sum((q - w) ** 2) == 0.0

    refined = _refine_candidates(Q, W, [0], True, False, np.array([False]))
    assert 0.0 < refined[0] < 1e-13


def test_positive_tiny_residual_does_not_pay_decimal_fallback(monkeypatch):
    """High precision is reserved for a literal binary64 zero collapse."""
    m = 1_024
    Q = np.arange(m, dtype=np.float64)
    W = Q + 10_000.0
    W[m // 2] = np.nextafter(W[m // 2], np.inf)

    def unexpected_decimal(*args, **kwargs):
        raise AssertionError("a positive float64 residual already proves non-zero")

    monkeypatch.setattr("mlx_stump._match._high_precision_znorm_rows", unexpected_decimal)
    refined = _refine_candidates(Q, W, [0], True, False, np.array([False]))
    assert refined[0] > 0.0


def test_exact_translation_fast_path_rejects_one_ulp_and_avoids_bigints(monkeypatch):
    """Many shifted windows use bounded vectorized exact arithmetic."""
    m = 257
    Q = np.arange(m, dtype=np.float64)
    offsets = np.arange(1, 513, dtype=np.float64)
    W = Q[None, :] + offsets[:, None]
    perturbed = W[0].copy()
    perturbed[m // 2] = np.nextafter(perturbed[m // 2], np.inf)
    assert np.all(_exact_translation_rows(Q, W))
    assert not _exact_translation_rows(Q, perturbed)[0]

    def unexpected_streaming(*args, **kwargs):
        raise AssertionError("exact translations should not reach streaming fallback")

    monkeypatch.setattr(
        "mlx_stump._match._exact_positive_affine_streaming_row",
        unexpected_streaming,
    )
    assert np.all(_exact_positive_affine_rows(Q, W))


def test_streaming_affine_fallback_is_exact_for_scaled_rows(monkeypatch):
    """The constant-memory fallback accepts scale/shift but rejects one ULP."""
    m = 4_097
    Q = np.arange(m, dtype=np.float64)
    exact = 17.0 * Q + 65_536.0
    perturbed = exact.copy()
    perturbed[m // 2] = np.nextafter(perturbed[m // 2], np.inf)

    # Force this test through the general certificate even if the bounded
    # vectorized fast path is broadened in the future.
    monkeypatch.setattr(
        "mlx_stump._match._exact_translation_rows",
        lambda _Q, rows: np.zeros(np.atleast_2d(rows).shape[0], dtype=bool),
    )
    np.testing.assert_array_equal(
        _exact_positive_affine_rows(Q, np.vstack((exact, perturbed))),
        np.array([True, False]),
    )


def test_exact_translation_fast_path_tiles_very_long_rows(monkeypatch):
    """Long translations stay on bounded vectorized tiles, without a cliff."""
    m = 200_001
    Q = np.arange(m, dtype=np.float64)
    offsets = np.array([1.0, 17.0, 65_536.0, -23.0])
    rows = Q[None, :] + offsets[:, None]
    perturbed = rows.copy()
    perturbed[0, m // 2] = np.nextafter(perturbed[0, m // 2], np.inf)
    assert not _exact_translation_rows(Q, perturbed)[0]
    assert np.all(_exact_translation_rows(Q, rows))

    def unexpected_streaming(*args, **kwargs):
        raise AssertionError("long exact translations should remain on bounded tiles")

    monkeypatch.setattr(
        "mlx_stump._match._exact_positive_affine_streaming_row",
        unexpected_streaming,
    )
    assert np.all(_exact_positive_affine_rows(Q, rows))


def test_normalized_match_includes_theoretical_maximum_boundary():
    """A one-ulp refinement overshoot must not cross the inclusive cutoff."""
    m = 1_025
    Q = np.random.default_rng(m).standard_normal(m)
    T = -3.0 * Q + 7.0
    limit = float(2.0 * np.sqrt(m))
    got = mlx_stump.match(Q, T, max_distance=limit, atol=0.0)
    reference = stumpy.match(Q, T, max_distance=limit, atol=0.0)
    np.testing.assert_array_equal(got.astype(np.float64), reference.astype(np.float64))


def test_strict_public_match_rejects_one_ulp_non_affine_row():
    m = 64
    Q = np.arange(m, dtype=np.float64)
    W = Q.copy()
    W[m // 2] = np.nextafter(W[m // 2], np.inf)
    T = np.random.default_rng(123).standard_normal(4 * m)
    exact_idx, perturbed_idx = 10, 2 * m + 20
    T[exact_idx : exact_idx + m] = Q
    T[perturbed_idx : perturbed_idx + m] = W

    got = mlx_stump.match(Q, T, max_distance=0.0, atol=0.0)
    np.testing.assert_array_equal(got.astype(np.float64), np.array([[0.0, exact_idx]]))


def test_data_dependent_threshold_is_not_recalled_when_profile_did_not_change():
    Q = np.arange(8, dtype=np.float64)
    T = np.arange(40, dtype=np.float64)
    calls = []

    def reject_all(D):
        calls.append(D.copy())
        return -np.inf

    got = mlx_stump.match(Q, T, max_distance=reject_all)
    assert got.size == 0
    assert len(calls) == 1


def test_raw_negative_infinite_threshold_is_warning_free_with_invalid_windows():
    Q = np.arange(8, dtype=np.float64)
    T = np.arange(40, dtype=np.float64)
    T[20] = np.nan
    with np.errstate(all="raise"):
        got = mlx_stump.match(Q, T, normalize=False, max_distance=-np.inf)
    assert got.size == 0
