"""Tiled joins reuse one bounded packing of their query windows."""

from __future__ import annotations

import gc

import mlx.core as mx
import numpy as np
import pytest

import mlx_stump
from mlx_stump import _engine, _kernels, _stump


def _plan(**overrides):
    args = dict(
        target_rows=120_000,
        window=50,
        query_rows=8_000,
        k=1,
        self_join=False,
        fused=True,
        tile_rows=30_000,
        chunk_size=None,
    )
    args.update(overrides)
    return _engine.tiled_query_cache_plan(**args)


def test_tiled_query_cache_plan_reserves_batch_budget(monkeypatch):
    batch, reserve = _plan()
    assert batch == min(8_000, _engine._tiled_batch(30_000, 50, 1, False, True))
    assert 0 < reserve <= _engine._TILED_QUERY_CACHE_MAX_BYTES
    per_row = _engine._batch_row_bytes(30_000, 50, 1, False, True)
    # The packed rows have a separate capped allowance so the original
    # batch shape, and thus GEMM's float32 reduction, stays unchanged.
    assert batch * per_row <= _engine._CHUNK_MEM_BUDGET
    assert reserve == (
        (8_000 + batch) * 50 * 4
        + (-(-8_000 // batch)) * _engine._TILED_QUERY_CACHE_BATCH_OVERHEAD
    )
    monkeypatch.setattr(_engine, "_MATMUL_WINDOW_BYTES", 0)
    monkeypatch.setattr(_engine, "_TILE_WINDOW_BYTES", 30_000 * 50 * 4)
    estimate = _engine.estimated_peak_bytes(
        120_000, 50, self_join=False, l_q=8_000, fused=True
    )
    monkeypatch.setattr(_engine, "_TILED_QUERY_CACHE_MAX_BYTES", 0)
    no_cache_estimate = _engine.estimated_peak_bytes(
        120_000, 50, self_join=False, l_q=8_000, fused=True
    )
    assert estimate >= no_cache_estimate + reserve
    monkeypatch.setattr(_engine, "_TILED_QUERY_CACHE_MAX_BYTES", 32 << 20)

    # A three-block sweep does not amortize an extra resident packing.
    _, reserve = _plan(target_rows=90_000)
    assert reserve == 0
    # Explicit batches are never silently shrunk to accommodate a cache.
    explicit_batch, reserve = _plan(chunk_size=4096)
    assert explicit_batch == 4096
    assert reserve == 0
    # Extremely small batches are not retained as thousands of MLX arrays.
    _, reserve = _plan(window=3, tile_rows=256, query_rows=5000, chunk_size=1)
    assert reserve == 0
    tiny_batch, reserve = _plan(window=3, tile_rows=256, query_rows=4000, chunk_size=1)
    assert tiny_batch == 1
    assert reserve >= 4000 * _engine._TILED_QUERY_CACHE_BATCH_OVERHEAD
    monkeypatch.setattr(_engine, "_TILED_QUERY_CACHE_MAX_BYTES", 0)
    _, reserve = _plan()
    assert reserve == 0


def _series(self_join: bool):
    rng = np.random.default_rng(915)
    a = rng.standard_normal(260).cumsum()
    a[120:152] = a[15:47]
    a[62:103] = 17.0
    a[182] = np.nan
    if self_join:
        return (a, 32), {}
    b = rng.standard_normal(385).cumsum()
    b[80:112] = a[15:47]
    b[225:257] = a[15:47]
    b[305] = np.inf
    return (a, 32, b), {"ignore_trivial": False}


def _force_many_tiles(monkeypatch):
    monkeypatch.setattr(_engine, "_MATMUL_WINDOW_BYTES", 0)
    monkeypatch.setattr(_engine, "_TILE_WINDOW_BYTES", 4096)


def _fields(profile):
    return tuple(np.asarray(getattr(profile, key)).copy() for key in (
        "P_", "I_", "left_I_", "right_I_"
    ))


@pytest.mark.parametrize("self_join", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("k", [1, 5, 16, 17])
def test_cached_tiled_queries_preserve_full_profile(monkeypatch, self_join, normalize, k):
    _force_many_tiles(monkeypatch)
    call, options = _series(self_join)
    options.update(normalize=normalize, k=k, chunk_size=37)
    enabled_cap = _engine._TILED_QUERY_CACHE_MAX_BYTES
    monkeypatch.setattr(_engine, "_TILED_QUERY_CACHE_MAX_BYTES", 0)
    fresh = _fields(mlx_stump.stump(*call, **options))
    monkeypatch.setattr(_engine, "_TILED_QUERY_CACHE_MAX_BYTES", enabled_cap)
    cached = _fields(mlx_stump.stump(*call, **options))
    for got, expected in zip(cached, fresh, strict=True):
        np.testing.assert_array_equal(got, expected)


def test_automatic_batch_keeps_exact_tiled_profile(monkeypatch):
    _force_many_tiles(monkeypatch)
    call, options = _series(False)
    options.update(normalize=True, k=1)
    enabled_cap = _engine._TILED_QUERY_CACHE_MAX_BYTES
    monkeypatch.setattr(_engine, "_TILED_QUERY_CACHE_MAX_BYTES", 0)
    fresh = _fields(mlx_stump.stump(*call, **options))
    monkeypatch.setattr(_engine, "_TILED_QUERY_CACHE_MAX_BYTES", enabled_cap)
    cached = _fields(mlx_stump.stump(*call, **options))
    for got, expected in zip(cached, fresh, strict=True):
        np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("self_join", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
def test_query_pack_count_is_one_pass_with_cache(monkeypatch, self_join, normalize):
    _force_many_tiles(monkeypatch)
    call, options = _series(self_join)
    options.update(normalize=normalize, k=1, chunk_size=37)
    original = _stump.query_windows
    calls = []

    def count(*args, **kwargs):
        calls.append((args[1], args[2]))
        return original(*args, **kwargs)

    monkeypatch.setattr(_stump, "query_windows", count)
    enabled_cap = _engine._TILED_QUERY_CACHE_MAX_BYTES
    monkeypatch.setattr(_engine, "_TILED_QUERY_CACHE_MAX_BYTES", 0)
    mlx_stump.stump(*call, **options)
    uncached_count = len(calls)
    calls.clear()
    monkeypatch.setattr(_engine, "_TILED_QUERY_CACHE_MAX_BYTES", enabled_cap)
    mlx_stump.stump(*call, **options)
    cached_count = len(calls)

    l_q = len(call[0]) - call[1] + 1
    l_t = len(call[0] if self_join else call[2]) - call[1] + 1
    tile_rows = _engine._TILE_WINDOW_BYTES // (call[1] * 4)
    nblocks = -(-l_t // tile_rows)
    nbatches = -(-l_q // options["chunk_size"])
    assert nblocks >= 2
    assert uncached_count == nblocks * nbatches
    # A whole-matrix cache may pack via MassEngine rather than query_windows.
    assert cached_count <= nbatches


def test_cached_tiled_peak_accounts_for_packed_rows(monkeypatch):
    _force_many_tiles(monkeypatch)
    call, options = _series(False)
    options.update(normalize=True, k=1, chunk_size=37)
    cache_cap = _engine._TILED_QUERY_CACHE_MAX_BYTES
    l_q = len(call[0]) - call[1] + 1
    l_t = len(call[2]) - call[1] + 1
    tile_rows = _engine._TILE_WINDOW_BYTES // (4 * call[1])
    _, reserve = _engine.tiled_query_cache_plan(
        target_rows=l_t, window=call[1], query_rows=l_q, k=1,
        self_join=False, fused=True, tile_rows=tile_rows, chunk_size=37,
    )
    assert reserve > 0

    def peak(enabled):
        monkeypatch.setattr(
            _engine, "_TILED_QUERY_CACHE_MAX_BYTES", cache_cap if enabled else 0
        )
        gc.collect()
        mx.synchronize()
        mx.clear_cache()
        mx.reset_peak_memory()
        result = mlx_stump.stump(*call, **options)
        del result
        return mx.get_peak_memory()

    peak(False)  # warm the same kernel paths before comparing allocator peaks
    peak(True)
    uncached = peak(False)
    cached = peak(True)
    assert cached <= uncached + reserve + (1 << 20)


def test_cached_tiled_query_is_released_after_failure(monkeypatch):
    _force_many_tiles(monkeypatch)
    call, options = _series(False)
    options.update(normalize=True, k=1, chunk_size=37)
    _engine._fused_reducer(1)  # finish the kernel probe before injecting a failure
    count = 0

    def failing_block(original):
        def fail_on_third(self, *args, **kwargs):
            nonlocal count
            count += 1
            if count == 3:
                raise MemoryError("injected reducer failure")
            return original(self, *args, **kwargs)

        return fail_on_third

    for cls in (_kernels.FusedReduce, _engine.ReduceStep):
        monkeypatch.setattr(cls, "block", failing_block(cls.block))
    gc.collect()
    mx.synchronize()
    mx.clear_cache()
    base = mx.get_active_memory()
    with pytest.raises(MemoryError) as excinfo:
        mlx_stump.stump(*call, **options)
    assert count == 3
    assert mx.get_cache_memory() == 0
    assert mx.get_active_memory() <= base + (1 << 20)
    del excinfo
    gc.collect()
    # A tiny MLX allocator residue also appears with the cache disabled
    # after pytest drops the exception traceback (12,090 B on this host).
    assert mx.get_cache_memory() < (1 << 20)
