"""Shared datasets and golden-comparison helpers.

The golden harness compares every code path against float64 STUMPY. Index
comparisons are tie-tolerant: the GPU searches in float32, so near-tied
neighbors may legitimately resolve to a different index — the assertion is
then that both chosen neighbors are equally good (their exact float64
distances agree within a small tolerance), which is what downstream motif or
discord analysis actually depends on.
"""

from __future__ import annotations

import numpy as np
import pytest

stumpy = pytest.importorskip("stumpy")


def pytest_collection_modifyitems(config, items):
    """gpu-marked tests measure Metal behavior (e.g. peak GPU memory) and are
    skipped on CPU-only runners, as the marker description promises."""
    import mlx.core as mx

    if mx.metal.is_available():
        return
    skip_gpu = pytest.mark.skip(reason="requires a Metal GPU")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


# ---------------------------------------------------------------- datasets
def random_walk(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n).cumsum()


def sine_noise(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 24 * np.pi, n)
    return np.sin(t) + 0.3 * rng.standard_normal(n)


def white_noise(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n)


def with_nans(n: int, seed: int = 0) -> np.ndarray:
    T = random_walk(n, seed)
    T[n // 5] = np.nan
    T[n // 2] = np.inf
    T[(3 * n) // 4] = -np.inf
    return T


def with_constants(n: int, seed: int = 0) -> np.ndarray:
    T = white_noise(n, seed)
    T[n // 4 : n // 4 + n // 10] = 3.0
    T[(2 * n) // 3 : (2 * n) // 3 + n // 10] = -1.5
    return T


def large_offset(n: int, seed: int = 0) -> np.ndarray:
    # exercises the global-standardization layer of the precision strategy
    return random_walk(n, seed) + 1.0e6


DATASETS = {
    "random_walk": random_walk,
    "sine_noise": sine_noise,
    "white_noise": white_noise,
    "with_nans": with_nans,
    "with_constants": with_constants,
    "large_offset": large_offset,
}


# ------------------------------------------------------- golden comparison
def _znorm_dist(T_A, T_B, m, i, j):
    """Exact float64 z-normalized distance between two subsequences."""
    a = T_A[i : i + m].astype(np.float64)
    b = T_B[j : j + m].astype(np.float64)
    sa, sb = a.std(), b.std()
    if sa == 0.0 and sb == 0.0:
        return 0.0
    if sa == 0.0 or sb == 0.0:
        return np.sqrt(m)
    rho = ((a * b).mean() - a.mean() * b.mean()) / (sa * sb)
    return np.sqrt(max(2.0 * m * (1.0 - rho), 0.0))


def _abs_dist(T_A, T_B, m, i, j):
    diff = T_A[i : i + m] - T_B[j : j + m]
    return float(np.sqrt((diff * diff).sum()))


def assert_indices_tie_tolerant(I_ours, I_ref, T_A, T_B, m, *, normalize=True, tie_atol):
    """Every disagreeing index must point at an equally good neighbor.

    Stronger than an agreement-rate floor: each mismatch is verified
    individually in float64, and heavily tied data (e.g. constant regions,
    tiny windows) may legitimately disagree on many rows.
    """
    I_ours = np.asarray(I_ours, dtype=np.int64)
    I_ref = np.asarray(I_ref, dtype=np.int64)
    assert I_ours.shape == I_ref.shape
    # -1 (no neighbor) patterns must match exactly
    np.testing.assert_array_equal(I_ours == -1, I_ref == -1)
    valid = I_ref >= 0
    if valid.sum() == 0:
        return
    agree = I_ours[valid] == I_ref[valid]
    dist = _znorm_dist if normalize else _abs_dist
    for i in np.nonzero(valid)[0][~agree]:
        d_ours = dist(T_A, T_B, m, i, I_ours[i])
        d_ref = dist(T_A, T_B, m, i, I_ref[i])
        assert abs(d_ours - d_ref) <= tie_atol, (
            f"row {i}: ours -> {I_ours[i]} (d={d_ours:.6f}), ref -> {I_ref[i]} (d={d_ref:.6f})"
        )


def assert_profile_close(P_ours, P_ref, *, m, exact_mask=None, tie_atol):
    """Profile values: exact float64 agreement where indices agree, tie
    tolerance elsewhere; inf patterns must match exactly."""
    P_ours = np.asarray(P_ours, dtype=np.float64)
    P_ref = np.asarray(P_ref, dtype=np.float64)
    np.testing.assert_array_equal(np.isinf(P_ours), np.isinf(P_ref))
    finite = np.isfinite(P_ref)
    if exact_mask is not None:
        # where indices agree both sides are float64 evaluations of the same
        # pair; compare squares because a near-zero distance sqrt-amplifies
        # even fp64 formula-order differences
        strict = finite & exact_mask
        np.testing.assert_allclose(
            P_ours[strict] ** 2, P_ref[strict] ** 2, atol=1e-7 * m, rtol=1e-6
        )
    np.testing.assert_allclose(P_ours[finite], P_ref[finite], atol=tie_atol, rtol=0)


def tie_tolerance(m: int) -> float:
    """Float32-search tie tolerance in distance units.

    Near a perfect match, an fp32 error e in the squared distance shows up as
    sqrt(e) in the distance, so the tolerance scales with sqrt(sqrt(m)).
    Tiny windows on smooth series sit on huge tie plateaus (measured worst
    disagreement at m=3 is ~0.04; zero at m>=8), hence the small-m floor.
    """
    base = 0.05 * (m / 50.0) ** 0.25
    return max(base, 0.08) if m < 6 else base


def assert_dist_profiles_close(D, Dr, *, m, scale=1.0):
    """Compare distance profiles via their squares: near d=0 an fp32 error in
    d**2 is sqrt-amplified in d, so d-space atol would be misleadingly strict
    there and misleadingly loose elsewhere."""
    D = np.asarray(D, dtype=np.float64)
    Dr = np.asarray(Dr, dtype=np.float64)
    np.testing.assert_array_equal(np.isinf(D), np.isinf(Dr))
    finite = np.isfinite(Dr)
    np.testing.assert_allclose(
        D[finite] ** 2,
        Dr[finite] ** 2,
        atol=3e-4 * m * scale**2,
        rtol=1e-3,
    )
