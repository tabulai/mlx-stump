"""Benchmark production row-parallel float64 target-window packing.

Run with ``PYTHONPATH=src python bench/bench_parallel_packing.py`` on a Metal
host. The serial comparison uses the same production arithmetic with one
reported CPU; the parallel path reports eight. Both use the same GPU sweep.
"""

from __future__ import annotations

import gc
import statistics
import time

import mlx.core as mx
import numpy as np

import mlx_stump
from mlx_stump import _engine

_ORIGINAL_CPU_COUNT = _engine.os.cpu_count


def measure(T: np.ndarray, m: int, variant: str):
    _engine.os.cpu_count = (lambda: 1) if variant == "serial" else (lambda: 8)
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()
    start = time.perf_counter()
    result = mlx_stump.stump(T, m)
    elapsed = time.perf_counter() - start
    return elapsed, mx.get_peak_memory(), result


def main():
    rng = np.random.default_rng(29)
    try:
        for n, m in ((8192, 1024), (16384, 200), (4096, 200)):
            T = np.cumsum(rng.normal(size=n)).astype(np.float64)
            samples = {"serial": [], "parallel": []}
            peaks = {"serial": [], "parallel": []}
            baseline = None
            for variant in ("serial", "parallel") * 5:
                elapsed, peak, result = measure(T, m, variant)
                samples[variant].append(elapsed)
                peaks[variant].append(peak)
                if baseline is None:
                    baseline = result
                else:
                    assert np.array_equal(baseline.P_.view(np.uint64), result.P_.view(np.uint64))
                    assert np.array_equal(baseline.I_, result.I_)
                    assert np.array_equal(baseline.left_I_, result.left_I_)
                    assert np.array_equal(baseline.right_I_, result.right_I_)
            print(
                f"n={n} m={m}",
                "median_ms",
                {k: round(statistics.median(v) * 1000, 2) for k, v in samples.items()},
                "peak_MiB",
                {k: round(max(v) / 2**20, 2) for k, v in peaks.items()},
                "exact_outputs=True",
                flush=True,
            )
    finally:
        _engine.os.cpu_count = _ORIGINAL_CPU_COUNT


if __name__ == "__main__":
    main()
