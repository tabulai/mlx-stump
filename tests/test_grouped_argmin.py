"""The grouped k=1 AB-join kernel: four narrow rows per 128-thread group.

- It is dispatched exactly for adaptive k=1 AB-join batches of at least four
  rows whose block has at most 128 columns, and its (index, distance) bits
  equal the per-row argmin kernel's and the compiled fallback's, including
  ties, constant and non-finite windows, and partial last groups.
- Self-joins, wider rows, batches of fewer than four rows and fixed-width
  reducers keep the per-row kernel.
- MLX builds a kernel at its first dispatch, so the kernel is probed with a
  dispatch once per process; when the probe fails, narrow rows keep the
  per-row kernel (same result, no warning) instead of failing stump().
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

import mlx_stump
import mlx_stump._engine as eng
import mlx_stump._kernels as kernels

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires a Metal GPU")


def _series(length, rng, *, constant_every=5, nonfinite_every=11):
    flags = np.zeros(length, dtype=bool)
    flags[::constant_every] = True
    finite = np.ones(length, dtype=bool)
    finite[3::nonfinite_every] = False
    return SimpleNamespace(
        l=length,
        sig_inv_mx=mx.array(rng.random(length, dtype=np.float32) + 0.5),
        isconstant_mx=mx.array(flags),
        isfinite_mx=mx.array(finite),
        ssq_mx=mx.array(rng.random(length, dtype=np.float32) * 64),
        mu_mx=mx.array(rng.random((length, 2), dtype=np.float32)),
    )


def _count_grouped(monkeypatch, probe=True):
    """Record the batch shapes the grouped kernel is dispatched with (after
    this process's one-time probe, unless ``probe`` is False)."""
    if probe:
        assert kernels.grouped_argmin_launches()
    calls = []
    real = kernels._grouped_argmin

    def spy(kern, QT, *args):
        calls.append(QT.shape)
        return real(kern, QT, *args)

    monkeypatch.setattr(kernels, "_grouped_argmin", spy)
    return calls


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("width", [1, 7, 31, 32, 33, 100, 127, 128])
@pytest.mark.parametrize("batch", [4, 5, 9, 37])
def test_grouped_kernel_is_dispatched_and_bit_identical(monkeypatch, normalize, width, batch):
    rng = np.random.default_rng(1000 * width + batch)
    query, target = _series(batch + 3, rng), _series(width + 2, rng)
    # quantized QT forces exact (d2, column) ties within and across simdgroups
    qt = mx.array(rng.integers(0, 6, size=(batch, width)).astype(np.float32) / 4)
    consts = np.array([16, 1 / 16, 32, 64], dtype=np.float32)
    kwargs = dict(normalize=normalize, self_join=False, excl=0, k=1, consts=consts)
    calls = _count_grouped(monkeypatch)

    grouped = kernels.FusedReduce(query, target, **kwargs).block(qt, 3, 2, 2 + width)
    assert calls == [(batch, width)]
    tg = kernels.argmin_threadgroup(width, kernels.launch_threadgroup(1))
    per_row = kernels.FusedReduce(query, target, **kwargs, threadgroup=tg).block(
        qt, 3, 2, 2 + width
    )
    fallback = eng.ReduceStep(query, target, **kwargs).block(qt, 3, 2, 2 + width)
    assert len(calls) == 1  # the fixed-width reducer kept the per-row kernel

    gi, gp = (np.array(x) for x in grouped)
    for other in (per_row, fallback):
        oi, op = (np.array(x) for x in other[:2])
        np.testing.assert_array_equal(gi, oi.astype(gi.dtype))
        np.testing.assert_array_equal(gp.view(np.uint32), op.astype(np.float32).view(np.uint32))


@pytest.mark.parametrize(
    "self_join,width,batch",
    [(True, 64, 8), (False, 129, 8), (False, 64, 3), (False, 64, 1)],
)
def test_other_shapes_keep_the_per_row_kernel(monkeypatch, self_join, width, batch):
    rng = np.random.default_rng(7)
    query, target = _series(max(batch, width) + 2, rng), _series(width + 2, rng)
    qt = mx.array(rng.random((batch, width), dtype=np.float32))
    consts = np.array([16, 1 / 16, 32, 64], dtype=np.float32)
    calls = _count_grouped(monkeypatch)
    red = kernels.FusedReduce(
        query, target, normalize=True, self_join=self_join, excl=1, k=1, consts=consts
    )
    mx.eval(*red.full(qt, 0))
    assert calls == []


def test_narrow_ab_join_matches_per_row_kernel_end_to_end(monkeypatch):
    rng = np.random.default_rng(31)
    ta = rng.standard_normal(3000).cumsum()
    ta[400:460] = 2.0  # constant windows
    ta[900:905] = np.nan
    tb = rng.standard_normal(140).cumsum()  # 111 target windows at m=30
    tb[50:52] = np.inf
    for normalize in (True, False):
        with monkeypatch.context() as mp:
            calls = _count_grouped(mp)
            grouped = mlx_stump.stump(ta, 30, tb, ignore_trivial=False, normalize=normalize)
        assert calls
        with monkeypatch.context() as mp:
            mp.setattr(kernels, "_GROUPED_LAUNCH", False)
            per_row = mlx_stump.stump(ta, 30, tb, ignore_trivial=False, normalize=normalize)
        np.testing.assert_array_equal(grouped.P_.view(np.uint64), per_row.P_.view(np.uint64))
        np.testing.assert_array_equal(grouped.I_, per_row.I_)


def test_failed_probe_keeps_the_per_row_kernel(monkeypatch):
    rng = np.random.default_rng(32)
    ta = rng.standard_normal(2000).cumsum()
    tb = rng.standard_normal(120).cumsum()
    expected = mlx_stump.stump(ta, 30, tb, ignore_trivial=False)

    def broken(*args, **kwargs):
        raise RuntimeError("synthetic build failure")

    monkeypatch.setattr(kernels, "_GROUPED_LAUNCH", None)
    monkeypatch.setattr(kernels, "_grouped_argmin", broken)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the fallback is silent
        assert kernels.grouped_argmin_launches() is False
        got = mlx_stump.stump(ta, 30, tb, ignore_trivial=False)
    assert kernels._GROUPED_LAUNCH is False  # probed once, not per call
    np.testing.assert_array_equal(got.P_.view(np.uint64), expected.P_.view(np.uint64))
    np.testing.assert_array_equal(got.I_, expected.I_)


def test_probe_runs_once_per_process(monkeypatch):
    monkeypatch.setattr(kernels, "_GROUPED_LAUNCH", None)
    calls = _count_grouped(monkeypatch, probe=False)
    assert kernels.grouped_argmin_launches() is True
    assert len(calls) == 2  # z-normalized and raw, one partial group each
    assert kernels.grouped_argmin_launches() is True
    assert len(calls) == 2
