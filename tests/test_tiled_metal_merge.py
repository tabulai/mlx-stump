"""The Metal tiled merge must preserve the CPU merge's exact ordering."""

import mlx.core as mx
import numpy as np
import pytest

import mlx_stump
from mlx_stump import _engine, _stump
from mlx_stump._merge_kernels import TiledMerge

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires a Metal GPU")


@pytest.mark.parametrize("self_join", [False, True])
@pytest.mark.parametrize("k", [1, 5])
def test_metal_merge_sorted_lists_and_side_ties(self_join, k):
    n = 8
    rows = np.arange(n)
    prior_v = np.tile(np.arange(k, dtype=np.float32), (n, 1))
    block_v = prior_v.copy()
    prior_i = np.tile(np.arange(20, 20 + k, dtype=np.int32), (n, 1))
    block_i = np.tile(np.arange(k, dtype=np.int32), (n, 1))
    prior_v[0] = np.inf
    prior_i[0] = -1
    block_v[1] = np.inf
    block_i[1] = -1
    if self_join:
        # The same d2 can favor an earlier or later block depending on
        # distance from the query row.
        expected_v, expected_i = _stump._merge_topk(
            prior_v, prior_i, block_v, block_i + 30, k, rows
        )
    else:
        expected_v, expected_i = _stump._merge_topk(
            prior_v, prior_i, block_v, block_i + 30, k
        )
    shape = (n,) if k == 1 else (n, k)
    run = TiledMerge(n, k, self_join)
    run._run = (
        mx.array(prior_v.reshape(shape)),
        mx.array(prior_i.reshape(shape)),
    )
    block = (mx.array(block_v.reshape(shape)), mx.array(block_i.reshape(shape)))
    if self_join:
        run._run += (
            mx.full((n,), 25, dtype=mx.int32),
            mx.full((n,), 2.0, dtype=mx.float32),
            mx.full((n,), 40, dtype=mx.int32),
            mx.full((n,), 2.0, dtype=mx.float32),
        )
        block += (
            mx.full((n,), 0, dtype=mx.int32),
            mx.full((n,), 2.0, dtype=mx.float32),
            mx.full((n,), 1, dtype=mx.int32),
            mx.full((n,), 2.0, dtype=mx.float32),
        )
    reducer_order = (
        (block[1], block[0], *block[2:]) if k == 1 else block
    )
    run.add(reducer_order, 30)
    got = run.result()
    np.testing.assert_array_equal(np.array(got[0]).reshape(n, k), expected_v)
    # Infinite entries have no neighbor and their ordering is immaterial.
    finite = np.isfinite(expected_v)
    np.testing.assert_array_equal(np.array(got[1]).reshape(n, k)[finite], expected_i[finite])
    assert np.all(np.array(got[1]).reshape(n, k)[~finite] == -1)
    if self_join:
        np.testing.assert_array_equal(np.array(got[2]), np.full(n, 30))
        np.testing.assert_array_equal(np.array(got[4]), np.full(n, 40))


@pytest.mark.parametrize("self_join", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("k", [1, 5, 16])
def test_tiled_metal_merge_matches_host_path(monkeypatch, self_join, normalize, k):
    rng = np.random.default_rng(8100 + k)
    a = rng.standard_normal(360).cumsum()
    b = rng.standard_normal(320).cumsum()
    a[70:83] = a[170:183]
    a[200] = np.nan
    monkeypatch.setattr(_engine, "_MATMUL_WINDOW_BYTES", 0)
    monkeypatch.setattr(_engine, "_TILE_WINDOW_BYTES", 2048)

    original = _stump._compute_profile_tiled

    def host_merge(*args, **kwargs):
        return original(*args, **kwargs, metal_merge=False)

    monkeypatch.setattr(_stump, "_compute_profile_tiled", host_merge)
    args = (a, 32) if self_join else (a, 32, b)
    kwargs = {"k": k, "normalize": normalize, "chunk_size": 57}
    if not self_join:
        kwargs["ignore_trivial"] = False
    reference = mlx_stump.stump(*args, **kwargs)
    monkeypatch.setattr(_stump, "_compute_profile_tiled", original)
    actual = mlx_stump.stump(*args, **kwargs)
    np.testing.assert_array_equal(actual.I_, reference.I_)
    np.testing.assert_array_equal(actual.P_, reference.P_)
    np.testing.assert_array_equal(actual.left_I_, reference.left_I_)
    np.testing.assert_array_equal(actual.right_I_, reference.right_I_)


def test_tiled_metal_merge_one_query_row(monkeypatch):
    rng = np.random.default_rng(37)
    query = rng.standard_normal(32)
    target = rng.standard_normal(97)
    monkeypatch.setattr(_engine, "_MATMUL_WINDOW_BYTES", 0)
    monkeypatch.setattr(_engine, "_TILE_WINDOW_BYTES", 32 * 4 * 4)
    original = _stump._compute_profile_tiled
    kwargs = {"ignore_trivial": False, "k": 16, "chunk_size": 1}
    actual = mlx_stump.stump(query, 32, target, **kwargs)

    def host_merge(*args, **options):
        return original(*args, **options, metal_merge=False)

    monkeypatch.setattr(_stump, "_compute_profile_tiled", host_merge)
    reference = mlx_stump.stump(query, 32, target, **kwargs)
    np.testing.assert_array_equal(actual.I_, reference.I_)
    np.testing.assert_array_equal(actual.P_, reference.P_)
