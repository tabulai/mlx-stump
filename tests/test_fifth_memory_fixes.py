"""Regression tests for fifth-round top-k host-memory accounting."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

import mlx_stump
from mlx_stump._engine import (
    _CENTER_ROW_BYTES,
    _CHUNK_MEM_BUDGET,
    _TILE_WINDOW_BYTES,
    estimated_peak_bytes,
    resident_block_bytes,
)


def test_topk_estimate_uses_allocator_footprint_and_ab_lengths():
    """Boxed scalar allocation classes, not logical sizes, drive RSS."""
    l, m, k = 49_951, 50, 100
    # Each neighbor has two object-array pointers and two 32-byte CPython
    # allocations. Left/right cells add another two pointer/object pairs.
    object_output = l * (2 * k + 2) * (8 + 32)
    numeric_output = l * (16 * k + 16)
    assert estimated_peak_bytes(l, m, k=k) >= object_output + numeric_output

    # The materialized window block follows the target, while output
    # assembly follows the query. Keep both dimensions effective.
    assert estimated_peak_bytes(1_000, m, k=k, l_q=50_000) > estimated_peak_bytes(
        1_000, m, k=k, l_q=1_000
    )
    assert estimated_peak_bytes(50_000, m, k=k, l_q=1_000) > estimated_peak_bytes(
        1_000, m, k=k, l_q=1_000
    )


def test_estimate_models_explicit_chunk_size_override():
    """A caller-selected batch may deliberately exceed the automatic cap."""
    l, m, k = 4_951, 50, 2
    automatic = estimated_peak_bytes(l, m, k=k)
    explicit = estimated_peak_bytes(l, m, k=k, chunk_size=l)

    one_row = l * 48 + m * 24 + _CENTER_ROW_BYTES  # plus local-normalization scratch
    numeric = l * (16 * k + 16)
    assert explicit >= resident_block_bytes(l, m) + l * one_row + numeric
    assert explicit > 2 * automatic
    for bad in (0, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            estimated_peak_bytes(l, m, k=k, chunk_size=bad)


def test_estimate_includes_one_row_centering_floor():
    """A single enormous window can exceed the nominal 64 MiB CPU budget."""
    l, m = 5, 10_000_000
    block = resident_block_bytes(l, m)
    actual_upload_floor = 2 * block + max(1 << 26, m * 8 + _CENTER_ROW_BYTES)
    assert estimated_peak_bytes(l, m, chunk_size=1) >= actual_upload_floor


def test_explicit_small_m_batch_estimate_includes_query_row_scratch():
    """A huge explicit batch retains local-normalization vectors per row."""
    l, m, l_q = 1_000, 3, 1_000_000
    one_query_row = m * 24 + _CENTER_ROW_BYTES
    expected = resident_block_bytes(l, m) + l_q * (l * 16 + one_query_row)
    assert estimated_peak_bytes(l, m, l_q=l_q, chunk_size=l_q) >= expected


@pytest.mark.parametrize(
    "kwargs,fragment",
    [
        ({"l": 0, "m": 50}, "`l`"),
        ({"l": 100, "m": 0}, "`m`"),
        ({"l": 100, "m": 50, "k": -1}, "`k`"),
        ({"l": 100, "m": 50, "l_q": 0}, "`l_q`"),
        ({"l": 100, "m": 50, "self_join": "yes"}, "`self_join`"),
        ({"l": 100, "m": 50, "chunk_size": True}, "`chunk_size`"),
    ],
)
def test_estimate_rejects_invalid_geometry(kwargs, fragment):
    with pytest.raises(ValueError, match=fragment):
        estimated_peak_bytes(**kwargs)


@pytest.mark.parametrize("name,value", [("k", True), ("chunk_size", True)])
def test_stump_rejects_boolean_integer_controls(name, value):
    T = np.arange(40, dtype=np.float64)
    with pytest.raises(ValueError, match="positive integer"):
        mlx_stump.stump(T, 8, **{name: value})


def test_tiled_topk_estimate_includes_host_merge_workspace():
    """Concatenate/sort/gather runs on the host while the block is live."""
    l, m, k, l_q = 20_000, 4_000, 2_000, 1_509
    block = resident_block_bytes(l, m)
    assert block < l * m * 4  # real tiled geometry from the RSS reproducer

    # tiled_chunk_size sizes against the nominal tile_rows upper bound even
    # though the balanced resident block can be narrower.
    tile_rows = max(4, _TILE_WINDOW_BYTES // (4 * m))
    per_row = tile_rows * 40 + m * 24 + _CENTER_ROW_BYTES  # AB top-k + query scratch
    batch = min(4_096, max(1, _CHUNK_MEM_BUDGET // per_row), l_q)
    numeric = l_q * (16 * k + 16)
    accum = l_q * 12 * k
    merge_workspace = batch * k * 80
    sweep = block + _CHUNK_MEM_BUDGET + numeric + accum + merge_workspace

    assert estimated_peak_bytes(l, m, k=k, self_join=False, l_q=l_q) >= sweep


def test_incomplete_compatibility_stats_are_ignored():
    """A lone precomputed array follows STUMPY's recompute-both path."""
    rng = np.random.default_rng(22)
    T = rng.standard_normal(128).astype(np.float64)
    Q = T[20:36].copy()
    l = T.size - Q.size + 1
    ref = mlx_stump.mass(Q, T)
    np.testing.assert_array_equal(mlx_stump.mass(Q, T, M_T=np.zeros(l)), ref)
    np.testing.assert_array_equal(mlx_stump.mass(Q, T, Σ_T=np.ones(l)), ref)


def test_raw_compatibility_stats_are_validated_then_fully_ignored():
    rng = np.random.default_rng(23)
    T = rng.standard_normal(128).astype(np.float64)
    Q = T[20:36].copy()
    l = T.size - Q.size + 1
    ref = mlx_stump.mass(Q, T, normalize=False)
    M = np.zeros(l)
    S = np.ones(l)
    M[0] = np.inf
    S[1] = np.nan
    np.testing.assert_array_equal(
        mlx_stump.mass(Q, T, M_T=M, Σ_T=S, normalize=False), ref
    )


def test_nonfinite_query_returns_inf_before_optional_metadata_validation():
    T = np.arange(40, dtype=np.float64)
    Q = np.arange(8, dtype=np.float64)
    Q[0] = np.nan
    expected = np.full(T.size - Q.size + 1, np.inf)
    np.testing.assert_array_equal(
        mlx_stump.mass(Q, T, M_T=np.ones(2), Σ_T=np.ones(3)), expected
    )
    np.testing.assert_array_equal(
        mlx_stump.mass(Q, T, Q_subseq_isconstant="not-a-bool"), expected
    )


def test_raw_constant_flags_are_consistently_validated_then_ignored():
    """All three raw APIs accept valid metadata and reject malformed flags."""
    Q = np.arange(5.0)
    T = np.arange(30.0)
    l = T.size - Q.size + 1
    target_flag = np.zeros(l, dtype=bool)
    series_flag = np.zeros(l, dtype=bool)

    mass_ref = mlx_stump.mass(Q, T, normalize=False)
    np.testing.assert_array_equal(
        mlx_stump.mass(
            Q,
            T,
            normalize=False,
            Q_subseq_isconstant=False,
            T_subseq_isconstant=target_flag,
        ),
        mass_ref,
    )
    match_ref = mlx_stump.match(Q, T, normalize=False, max_distance=0.0)
    np.testing.assert_array_equal(
        mlx_stump.match(
            Q,
            T,
            normalize=False,
            max_distance=0.0,
            Q_subseq_isconstant=False,
            T_subseq_isconstant=target_flag,
        ),
        match_ref,
    )
    stump_ref = mlx_stump.stump(T, Q.size, normalize=False)
    got = mlx_stump.stump(
        T, Q.size, normalize=False, T_A_subseq_isconstant=series_flag
    )
    np.testing.assert_array_equal(got, stump_ref)

    with pytest.raises(ValueError, match="T_subseq_isconstant"):
        mlx_stump.mass(Q, T, normalize=False, T_subseq_isconstant="bad")
    with pytest.raises(ValueError, match="T_subseq_isconstant"):
        mlx_stump.match(Q, T, normalize=False, T_subseq_isconstant="bad")
    with pytest.raises(ValueError, match="T_A_subseq_isconstant"):
        mlx_stump.stump(T, Q.size, normalize=False, T_A_subseq_isconstant="bad")


def test_exact_affine_certificate_has_bounded_auxiliary_memory():
    """A legal million-sample scaled duplicate must not build bigint lists."""
    source = r"""
import resource
import sys
import numpy as np
from mlx_stump._match import _exact_positive_affine_rows

unit = 1 if sys.platform == "darwin" else 1024
m = 1_000_000
Q = np.arange(m, dtype=np.float64)
W = 2.0 * Q + 1.0
before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * unit
assert _exact_positive_affine_rows(Q, W)[0]
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * unit
print((peak - before) / 2**20)
"""
    result = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, check=True
    )
    growth_mib = float(result.stdout.strip().splitlines()[-1])
    assert growth_mib < 64.0, f"affine certificate grew RSS by {growth_mib:.1f} MiB"


@pytest.mark.slow
def test_high_precision_non_affine_fallback_has_bounded_memory():
    """The rare Decimal path must stream rather than retain four huge lists."""
    source = r"""
import resource
import sys
import numpy as np
from mlx_stump._match import _refine_candidates

unit = 1 if sys.platform == "darwin" else 1024
m = 1_000_000
Q = np.random.default_rng(1).standard_normal(m)
W = Q.copy()
W[2] = np.nextafter(W[2], np.inf)
before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * unit
distance = _refine_candidates(Q, W, [0], True, False, np.array([False]))[0]
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * unit
assert distance > 0.0
print((peak - before) / 2**20)
"""
    result = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, check=True
    )
    growth_mib = float(result.stdout.strip().splitlines()[-1])
    assert growth_mib < 64.0, f"Decimal refinement grew RSS by {growth_mib:.1f} MiB"


def test_default_threshold_has_one_linear_scratch_array():
    """Computing the default cutoff must not retain three profile copies."""
    source = r"""
import resource
import sys
import numpy as np
from mlx_stump._match import _default_max_distance

unit = 1 if sys.platform == "darwin" else 1024
D = np.linspace(0.0, 1.0, 4_000_000, dtype=np.float64)
before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * unit
value = _default_max_distance(D)
assert np.isfinite(value)
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * unit
print((peak - before) / 2**20)
"""
    result = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, check=True
    )
    growth_mib = float(result.stdout.strip().splitlines()[-1])
    assert growth_mib < 56.0, f"default threshold grew RSS by {growth_mib:.1f} MiB"


def test_default_threshold_inverts_rounded_adjacent_float_frame_exactly():
    """A midpoint rounded to an endpoint must not shift the cutoff by one ULP."""
    from mlx_stump._match import _default_max_distance, _find_matches

    lo = 1.0
    hi = np.nextafter(lo, np.inf)
    distances = np.concatenate(([lo], np.full(9, hi)))
    expected = max(
        float(distances.mean() - 2.0 * distances.std()), float(distances.min())
    )

    assert expected == lo
    assert _default_max_distance(distances) == expected
    got = _find_matches(distances, excl_zone=0, atol=0.0)
    np.testing.assert_array_equal(got.astype(np.float64), np.array([[lo, 0.0]]))


def test_default_threshold_never_rounds_below_observed_minimum():
    """The robust inverse preserves max(..., min(D)) at ordinary scales too."""
    from mlx_stump._match import _default_max_distance, _find_matches

    rng = np.random.default_rng(801)
    distances = np.abs(rng.normal(size=int(rng.integers(1, 60))))
    minimum = float(distances.min())

    assert _default_max_distance(distances) == minimum
    got = _find_matches(distances, excl_zone=0, atol=0.0)
    np.testing.assert_array_equal(
        got.astype(np.float64), np.array([[minimum, float(np.argmin(distances))]])
    )


def test_default_threshold_ignores_nonfinite_entries_in_the_same_frame():
    """Reusing the compact finite copy must still apply the affine offset."""
    from mlx_stump._match import _default_max_distance

    finite = np.array([1.0, 2.0, 3.0])
    expected = max(float(finite.mean() - 2.0 * finite.std()), float(finite.min()))
    for sentinel in (np.inf, np.nan):
        distances = np.concatenate((finite, [sentinel]))
        assert _default_max_distance(distances) == expected
