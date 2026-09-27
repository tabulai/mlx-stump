"""Fused row reductions may use fewer threads for a short target row."""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

import mlx_stump
import mlx_stump._engine as engine
import mlx_stump._kernels as kernels


@pytest.mark.parametrize("width", [1, 7, 32, 129, 512, 1024, 4096, 16384, 65536])
def test_argmin_group_width_is_valid_and_wide_rows_keep_device_limit(width):
    for max_tg in (32, 128, 256, 512, 1024):
        tg = kernels.argmin_threadgroup(width, max_tg)
        assert 32 <= tg <= max_tg
        assert tg & (tg - 1) == 0
        if width > 8192:
            assert tg == max_tg


@pytest.mark.parametrize("width", [1, 32, 128, 256, 512, 1024, 4096, 8192])
def test_topk_group_width_is_valid_and_wide_rows_keep_device_limit(width):
    for max_tg in (32, 64, 128, 256):
        tg = kernels.topk_row_threadgroup(width, max_tg)
        assert 32 <= tg <= max_tg
        assert tg & (tg - 1) == 0
        if width > 4096:
            assert tg == max_tg


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires a Metal GPU")
@pytest.mark.parametrize("width", [7, 129, 513, 4096])
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("self_join", [True, False])
@pytest.mark.parametrize("k", [1, 2, 5, 16])
def test_adaptive_reducer_is_bit_identical_to_fixed_group(width, normalize, self_join, k):
    rng = np.random.default_rng(width)
    batch = 8

    def series(length):
        flags = np.zeros(length, dtype=bool)
        flags[::5] = True
        finite = np.ones(length, dtype=bool)
        finite[::11] = False
        return SimpleNamespace(
            l=length,
            sig_inv_mx=mx.ones((length,), dtype=mx.float32),
            isconstant_mx=mx.array(flags),
            isfinite_mx=mx.array(finite),
            ssq_mx=mx.ones((length,), dtype=mx.float32),
            mu_mx=mx.zeros((length, 2), dtype=mx.float32),
        )

    query, target = series(batch), series(width)
    qt = mx.array(rng.random((batch, width), dtype=np.float32))
    consts = np.array([16, 1 / 16, 32, 64], dtype=np.float32)
    kwargs = dict(normalize=normalize, self_join=self_join, excl=1, k=k, consts=consts)
    adaptive = kernels.FusedReduce(query, target, **kwargs).full(qt, 0)
    fixed = kernels.FusedReduce(
        query, target, **kwargs, threadgroup=kernels.launch_threadgroup(k)
    ).full(qt, 0)
    for a, b in zip(adaptive, fixed, strict=True):
        np.testing.assert_array_equal(np.array(a), np.array(b))


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires a Metal GPU")
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("self_join", [True, False])
@pytest.mark.parametrize("tiled", [True, False])
@pytest.mark.parametrize("k", [1, 2, 5, 16])
def test_adaptive_reducer_matches_fixed_group(monkeypatch, normalize, self_join, tiled, k):
    rng = np.random.default_rng(501)
    ta = rng.standard_normal(360).cumsum()
    ta[100:130] = 0.25  # ties and constant windows
    tb = None if self_join else rng.standard_normal(280).cumsum()
    if tiled:
        monkeypatch.setattr(engine, "_MATMUL_WINDOW_BYTES", 0)
        monkeypatch.setattr(engine, "_TILE_WINDOW_BYTES", 80 * 16 * 4)

    options = dict(k=k, normalize=normalize, ignore_trivial=self_join)
    adaptive = mlx_stump.stump(ta, 16, tb, **options)
    choose = "argmin_threadgroup" if k == 1 else "topk_row_threadgroup"
    monkeypatch.setattr(kernels, choose, lambda width, max_tg: max_tg)
    fixed = mlx_stump.stump(ta, 16, tb, **options)

    for field in ("P_", "I_", "left_I_", "right_I_"):
        np.testing.assert_array_equal(getattr(adaptive, field), getattr(fixed, field))
