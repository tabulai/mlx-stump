"""Larger narrow-target AB batches retain the same packed rows and GEMM scores."""

import mlx.core as mx
import numpy as np
import pytest

import mlx_stump
from mlx_stump import _engine
from mlx_stump._preprocess import finite_center_scale, preprocess_series


def test_narrow_ab_batch_cap_keeps_memory_budget_and_other_paths():
    target = type("Target", (), {"l": 128, "m": 50})()
    assert _engine.default_chunk_size(target, 100_000, 1, False, fused=True) == 8192
    assert _engine.default_chunk_size(target, 100_000, 1, True, fused=True) == 1024
    assert _engine.default_chunk_size(target, 100_000, 1, False, fused=False) == 1024
    target.l = 3
    assert _engine.default_chunk_size(target, 100_000, 1, False, fused=True) == 1024
    target.l = 1025
    assert _engine.default_chunk_size(target, 100_000, 1, False, fused=True) == 1024
    for width in (4, 128, 1024):
        target.l = width
        target.m = 512
        assert _engine.default_chunk_size(target, 100_000, 1, False, fused=True) == 8192
        for m in (1024, 2048):
            target.m = m
            assert _engine.default_chunk_size(target, 100_000, 1, False, fused=True) == 1024
    target.l = 128
    target.m = 50
    for k in (1, 5, 16):
        batch = _engine.default_chunk_size(target, 100_000, k, False, fused=True)
        assert batch * _engine._batch_row_bytes(target.l, target.m, k, False, True) <= (
            _engine._CHUNK_MEM_BUDGET
        )


needs_metal = pytest.mark.skipif(
    not mx.metal.is_available() or mx.default_device() != mx.gpu,
    reason="requires a Metal GPU",
)


@needs_metal
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("m,width", [(3, 4), (50, 128), (200, 1024)])
def test_narrow_ab_qt_is_bit_identical_across_batch_sizes(normalize, m, width):
    rng = np.random.default_rng(1300 + m + width)
    a = rng.standard_normal(10_000).cumsum()
    b = rng.standard_normal(width + m - 1).cumsum()
    center, scale = finite_center_scale(a, b) if not normalize else (None, None)
    kwargs = {} if normalize else {"center": center, "scale": scale}
    query = preprocess_series(a, m, normalize=normalize, **kwargs)
    target = preprocess_series(b, m, normalize=normalize, **kwargs)
    engine = _engine.MassEngine(target, normalize=normalize)
    small = _engine.query_windows(query, 0, 1024, normalize=normalize)
    large = _engine.query_windows(query, 0, 8192, normalize=normalize)
    small_qt = np.array(mx.matmul(small, engine.W_T))
    large_qt = np.array(mx.matmul(large, engine.W_T))[:1024]
    np.testing.assert_array_equal(small_qt.view(np.uint32), large_qt.view(np.uint32))


@needs_metal
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("k", [1, 5])
@pytest.mark.parametrize("near_tie", [False, True])
def test_narrow_ab_default_matches_small_batch(normalize, k, near_tie):
    rng = np.random.default_rng(1441)
    m = 50
    a = rng.standard_normal(10_000).cumsum()
    if near_tie:
        q = rng.standard_normal(m)
        for row in (0, 2048, 7000, 9000):
            a[row : row + m] = q
        near = q.copy()
        near[0] += 1e-5
        b = np.r_[near, np.full(m, 10.0), q, rng.standard_normal(27)]
    else:
        b = rng.standard_normal(177).cumsum()
    options = dict(ignore_trivial=False, normalize=normalize, k=k)
    baseline = mlx_stump.stump(a, m, b, chunk_size=1024, **options)
    widened = mlx_stump.stump(a, m, b, **options)
    for field in ("P_", "I_", "left_I_", "right_I_"):
        np.testing.assert_array_equal(getattr(widened, field), getattr(baseline, field))
    if near_tie:
        nearest = widened.I_ if k == 1 else widened.I_[:, 0]
        np.testing.assert_array_equal(nearest[[0, 2048, 7000, 9000]], 100)
