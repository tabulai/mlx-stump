"""Checks for the benchmark-only triangular self-join prototype."""

import numpy as np
import pytest
from bench import bench_symmetric as symmetric

from mlx_stump import _engine, _stump, stump


def test_balanced_tiles_bound_memory_and_avoid_single_block():
    assert symmetric.balanced_tiles(7993, 8192) == []
    tiles = symmetric.balanced_tiles(16185, 8192)
    assert len(tiles) == 2
    assert tiles[0][0] == 0 and tiles[-1][1] == 16185
    assert max(stop - start for start, stop in tiles) <= 8192
    assert abs((tiles[0][1] - tiles[0][0]) - (tiles[1][1] - tiles[1][0])) <= 1


@pytest.mark.parametrize(
    ("left", "expected"),
    [(True, [3, 2, 5]), (False, [2, 1, 5])],
)
def test_side_merge_uses_nearest_index_on_equal_score(left, expected):
    values = np.array([1.0, 1.0, np.inf], dtype=np.float32)
    indices = np.array([2, 2, -1], dtype=np.int64)
    symmetric._merge_side(
        values,
        indices,
        0,
        np.array([1.0, 1.0, 2.0], dtype=np.float32),
        np.array([3, 1, 5], dtype=np.int64),
        left=left,
    )
    np.testing.assert_array_equal(indices, expected)


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("normalize", "dataset", "n", "m", "cap"),
    [
        (True, "walk", 512, 32, 256),
        (True, "repeat", 1024, 64, 512),
        (False, "near", 1024, 64, 512),
    ],
)
def test_metal_prototype_matches_public_output_on_small_examples(
    monkeypatch, normalize, dataset, n, m, cap
):
    if not _engine._fused_reducer(1):
        pytest.skip("Metal k=1 reducer unavailable")
    series = symmetric._example_series(n, m, dataset, 145)
    expected = stump(series, m, normalize=normalize)

    def triangle(*args, **kwargs):
        return symmetric.triangular_profile(*args, **kwargs, cap=cap)

    monkeypatch.setattr(_stump, "_compute_profile", triangle)
    actual = stump(series, m, normalize=normalize)
    for field in ("P_", "I_", "left_I_", "right_I_"):
        np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
