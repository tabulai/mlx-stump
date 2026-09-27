"""Benchmark production normalized target-row reuse in near-zero repair.

Run with ``PYTHONPATH=src python bench/bench_repair_candidates.py`` on a Metal
host. The two variants run the same production repair algorithm; setting its
cache budget to zero for one run isolates the row-cache benefit. Seed reuse
and the direct k=1 merge remain enabled in both variants.
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time
import warnings

import mlx.core as mx
import numpy as np

import mlx_stump
from mlx_stump import _engine


def measure(T: np.ndarray, m: int, cache_bytes: int):
    _engine._REPAIR_NORMALIZED_CACHE_MAX_BYTES = cache_bytes
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()
    start = time.perf_counter()
    result = mlx_stump.stump(T, m)
    elapsed = time.perf_counter() - start
    return elapsed, mx.get_peak_memory(), result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--noise", type=float, default=1e-7)
    parser.add_argument("--reps", type=int, default=4)
    args = parser.parse_args()
    if args.n < args.m or args.m < 3 or args.reps < 1:
        parser.error("require n >= m >= 3 and reps >= 1")

    rng = np.random.default_rng(123)
    motif = rng.normal(size=args.m)
    T = np.resize(motif, args.n) + rng.normal(scale=args.noise, size=args.n)
    original_cap = _engine._REPAIR_NORMALIZED_CACHE_MAX_BYTES
    samples = {"uncached": [], "cached": []}
    peaks = {"uncached": [], "cached": []}
    baseline = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for rep in range(args.reps):
                order = ("uncached", "cached") if rep % 2 == 0 else ("cached", "uncached")
                for variant in order:
                    cap = 0 if variant == "uncached" else original_cap
                    elapsed, peak, result = measure(T, args.m, cap)
                    samples[variant].append(elapsed * 1000)
                    peaks[variant].append(peak / (1 << 20))
                    if baseline is None:
                        baseline = result
                    else:
                        assert np.array_equal(
                            baseline.P_.view(np.uint64), result.P_.view(np.uint64)
                        )
                        for name in ("I_", "left_I_", "right_I_"):
                            assert np.array_equal(getattr(baseline, name), getattr(result, name))
        print(f"n={args.n}, m={args.m}, noise={args.noise}, exact_outputs=True")
        for variant in samples:
            print(
                f"{variant}: median {statistics.median(samples[variant]):.2f} ms, "
                f"peak MLX {max(peaks[variant]):.2f} MiB"
            )
    finally:
        _engine._REPAIR_NORMALIZED_CACHE_MAX_BYTES = original_cap


if __name__ == "__main__":
    main()
