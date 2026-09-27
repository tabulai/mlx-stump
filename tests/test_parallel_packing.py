"""Parallel normalized target packing preserves the serial window matrix."""

from __future__ import annotations

import threading
import time
import traceback

import mlx.core as mx
import numpy as np
import pytest

import mlx_stump
from mlx_stump import _engine
from mlx_stump._preprocess import preprocess_series


def _target(n: int = 4_600) -> np.ndarray:
    rng = np.random.default_rng(2101)
    target = rng.standard_normal(n).cumsum()
    target[64:600] = 1e12 + rng.standard_normal(536) * 0.003
    target[1300:1320] = np.nan
    target[2000:2600] = 4.5
    target[3900:3910] = np.inf
    return target


def _packed(prep) -> np.ndarray:
    engine = _engine.MassEngine(prep)
    return np.array(engine.W_T)


@pytest.mark.parametrize("n,m", [(7_000, 200), (4_600, 512), (4_600, 1024)])
def test_parallel_target_packing_is_bit_exact_and_uses_workers(monkeypatch, n, m):
    prep = preprocess_series(_target(n), m)
    monkeypatch.setattr(_engine.os, "cpu_count", lambda: 1)
    serial = _packed(prep)

    real_pool = _engine.ThreadPoolExecutor
    launches = []

    def counted_pool(*args, **kwargs):
        launches.append(kwargs["max_workers"])
        return real_pool(*args, **kwargs)

    monkeypatch.setattr(_engine, "ThreadPoolExecutor", counted_pool)
    monkeypatch.setattr(_engine.os, "cpu_count", lambda: 8)
    parallel = _packed(prep)
    np.testing.assert_array_equal(parallel.view(np.uint32), serial.view(np.uint32))
    assert launches == [8]
    assert parallel.shape == (m, len(prep.T) - m + 1)

    # A host that does not export MLX's buffer must retain the staged path.
    monkeypatch.setattr(_engine, "_writable_host_view", lambda _: None)
    staged = _packed(prep)
    np.testing.assert_array_equal(staged.view(np.uint32), serial.view(np.uint32))
    assert launches == [8]
    mx.clear_cache()


def test_parallel_steps_keep_aggregate_float64_work_bounded(monkeypatch):
    m = 512
    prep = preprocess_series(_target(), m)
    monkeypatch.setattr(_engine.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(_engine, "_CENTER_BYTES", 3 << 20)
    original = _engine.center_rows_stable
    lock = threading.Lock()
    live = 0
    peak = 0
    active_tasks = 0
    peak_tasks = 0
    calls = 0

    def tracked(work):
        nonlocal live, peak, active_tasks, peak_tasks, calls
        charge = work.shape[0] * (m * 8 + _engine._CENTER_ROW_BYTES)
        with lock:
            live += charge
            peak = max(peak, live)
            active_tasks += 1
            peak_tasks = max(peak_tasks, active_tasks)
            calls += 1
        try:
            time.sleep(0.002)  # make overlapping worker tasks observable
            return original(work)
        finally:
            with lock:
                live -= charge
                active_tasks -= 1

    monkeypatch.setattr(_engine, "center_rows_stable", tracked)
    _packed(prep)
    assert calls > 8
    assert live == 0
    assert active_tasks == 0
    assert peak_tasks > 1
    assert peak <= _engine._CENTER_BYTES
    mx.clear_cache()


def test_parallel_and_serial_stump_profiles_are_exact(monkeypatch):
    m = 512
    target = _target(3_100)
    monkeypatch.setattr(_engine.os, "cpu_count", lambda: 1)
    serial = mlx_stump.stump(target, m)
    monkeypatch.setattr(_engine.os, "cpu_count", lambda: 8)
    parallel = mlx_stump.stump(target, m)
    for name in ("P_", "I_", "left_I_", "right_I_"):
        left = getattr(parallel, name)
        right = getattr(serial, name)
        if name == "P_":
            np.testing.assert_array_equal(left.view(np.uint64), right.view(np.uint64))
        else:
            np.testing.assert_array_equal(left, right)
    mx.clear_cache()


def test_small_target_skips_thread_pool(monkeypatch):
    m = 64
    prep = preprocess_series(np.random.default_rng(19).standard_normal(350).cumsum(), m)
    monkeypatch.setattr(_engine.os, "cpu_count", lambda: 8)

    def unexpected_pool(*args, **kwargs):
        raise AssertionError("small packing unexpectedly launched threads")

    monkeypatch.setattr(_engine, "ThreadPoolExecutor", unexpected_pool)
    _packed(prep)
    mx.clear_cache()


@pytest.mark.parametrize("tiled", [False, True])
def test_worker_failure_releases_packed_block_before_cache_clear(monkeypatch, tiled):
    target = _target(6_400 if tiled else 4_600)
    if tiled:
        monkeypatch.setattr(_engine, "_MATMUL_WINDOW_BYTES", 0)
        monkeypatch.setattr(_engine, "_TILE_WINDOW_BYTES", 3_000 * 512 * 4)
    original = _engine.center_rows_stable
    monkeypatch.setattr(_engine.os, "cpu_count", lambda: 8)
    mx.synchronize()
    mx.clear_cache()
    baseline_active = mx.get_active_memory()
    baseline_cache = mx.get_cache_memory()

    def fail_worker(work):
        if threading.current_thread() is not threading.main_thread():
            raise MemoryError("injected parallel packing failure")
        return original(work)

    monkeypatch.setattr(_engine, "center_rows_stable", fail_worker)
    with pytest.raises(MemoryError, match="injected parallel packing failure") as caught:
        mlx_stump.stump(target, 512)
    frames = traceback.extract_tb(caught.value.__traceback__)
    assert any(frame.name == "pack_rows" for frame in frames)
    assert mx.get_active_memory() <= baseline_active + (1 << 20)
    assert mx.get_cache_memory() <= baseline_cache + (1 << 20)
