"""Explicit GPU entry points must never use MLX's CPU fallback."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

import mlx_stump


def _series() -> np.ndarray:
    return np.random.default_rng(2026).standard_normal(80).cumsum()


def test_gpu_entry_points_reject_active_cpu_device() -> None:
    T = _series()
    with mx.stream(mx.cpu):
        assert mx.default_device() == mx.cpu
        for name, call in (
            ("gpu_stump", lambda: mlx_stump.gpu_stump(T, 8)),
            ("gpu_aamp", lambda: mlx_stump.gpu_aamp(T, 8)),
            ("gpu_stimp", lambda: mlx_stump.gpu_stimp(T, 8, 12)),
        ):
            with pytest.raises(RuntimeError, match=f"{name}.*Metal GPU"):
                call()

        # The APIs without an explicit GPU promise still support MLX's CPU path.
        assert np.isfinite(mlx_stump.stump(T, 8).P_).any()
        assert np.isfinite(mlx_stump.aamp(T, 8).P_).any()
        pan = mlx_stump.stimp(T, 8, 12)
        pan.update()
        assert pan._n_processed == 1


def test_invalid_device_id_still_precedes_gpu_check() -> None:
    T = _series()
    with mx.stream(mx.cpu):
        for call in (
            lambda: mlx_stump.gpu_stump(T, 8, device_id=-1),
            lambda: mlx_stump.gpu_aamp(T, 8, device_id=-1),
            lambda: mlx_stump.gpu_stimp(T, 8, 12, device_id=-1),
        ):
            with pytest.raises(ValueError, match="device_id"):
                call()


def test_gpu_entry_points_reject_unavailable_metal(monkeypatch) -> None:
    T = _series()
    monkeypatch.setattr(mx.metal, "is_available", lambda: False)
    for call in (
        lambda: mlx_stump.gpu_stump(T, 8),
        lambda: mlx_stump.gpu_aamp(T, 8),
        lambda: mlx_stump.gpu_stimp(T, 8, 12),
    ):
        with pytest.raises(RuntimeError, match="metal_available=False"):
            call()


@pytest.mark.gpu
def test_gpu_stimp_rechecks_device_before_each_update() -> None:
    T = _series()
    pan = mlx_stump.gpu_stimp(T, 8, 12)
    with mx.stream(mx.cpu):
        with pytest.raises(RuntimeError, match="gpu_stimp.update.*Metal GPU"):
            pan.update()
    assert pan._n_processed == 0
    assert np.all(np.isinf(pan._PAN))

    pan.update()
    assert pan._n_processed == 1
    while pan._n_processed < len(pan.M_):
        pan.update()
    with mx.stream(mx.cpu):
        pan.update()  # exhausted updates remain no-ops


@pytest.mark.gpu
def test_gpu_api_allows_compiled_reducer_on_metal() -> None:
    # k > 16 uses the compiled reducer even though the computation stays on GPU.
    profile = mlx_stump.gpu_stump(_series(), 8, k=17)
    assert profile.P_.shape == (73, 17)
    assert np.isfinite(profile.P_).all()
