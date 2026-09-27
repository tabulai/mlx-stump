"""Benchmark repeated ordinary and prepared-target MASS/match queries.

The ordinary path uses ``mass(Q, T)`` or ``match(Q, T)`` for every query.
The prepared path builds an owned target snapshot once per timed batch, then
times only the queries against it. Preparation time is reported separately,
so callers can judge whether reuse pays for their query count. The prepared
object is closed before the next ordinary batch; neither path is measured
while the other's resident target matrix is live. Every output must match
exactly, including the float64-refined match distances and indices.

Usage:
    PYTHONPATH=src python bench/bench_prepared_target.py --n 65536 --m 256 \
        --query-count 8 --repeat 5
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from bench_stump import provenance

import mlx_stump


def _run_queries(kind: str, T: np.ndarray, queries: list[np.ndarray], target=None):
    if kind == "mass":
        if target is None:
            return [mlx_stump.mass(Q, T) for Q in queries]
        return [target.mass(Q) for Q in queries]
    if target is None:
        return [mlx_stump.match(Q, T, max_distance=np.inf, max_matches=3) for Q in queries]
    return [target.match(Q, max_distance=np.inf, max_matches=3) for Q in queries]


def _benchmark(kind: str, T: np.ndarray, m: int, queries: list[np.ndarray], repeat: int):
    # Warm the ordinary path, prepared packing, Metal kernels and refinement
    # outside the clock. Close the prepared object before baseline timing.
    _run_queries(kind, T, queries[:1])
    with mlx_stump.prepare_target(T, m) as target:
        _run_queries(kind, T, queries[:1], target)

    timings = {"ordinary": [], "prepared": []}
    prepare_times = []
    for rep in range(repeat):
        outputs = {}
        order = ("ordinary", "prepared") if rep % 2 == 0 else ("prepared", "ordinary")
        for mode in order:
            if mode == "ordinary":
                start = time.perf_counter()
                outputs[mode] = _run_queries(kind, T, queries)
                timings[mode].append(time.perf_counter() - start)
            else:
                start = time.perf_counter()
                target = mlx_stump.prepare_target(T, m)
                prepare_times.append(time.perf_counter() - start)
                try:
                    start = time.perf_counter()
                    outputs[mode] = _run_queries(kind, T, queries, target)
                    timings[mode].append(time.perf_counter() - start)
                finally:
                    target.close()
        exact = all(
            np.array_equal(a, b)
            for a, b in zip(outputs["ordinary"], outputs["prepared"], strict=True)
        )
        if not exact:
            raise SystemExit(f"{kind} prepared path changed the result on repeat {rep + 1}")

    ordinary = float(np.median(timings["ordinary"])) / len(queries)
    prepared = float(np.median(timings["prepared"])) / len(queries)
    build = float(np.median(prepare_times))
    return ordinary, prepared, build


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=65536, help="target series length")
    parser.add_argument("--m", type=int, default=256, help="window length")
    parser.add_argument("--query-count", "--queries", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--seed", type=int, default=64)
    args = parser.parse_args()
    if args.n < args.m or args.m < 3:
        parser.error("require n >= m >= 3")
    if args.query_count < 1 or args.repeat < 1:
        parser.error("query-count and repeat must be positive")

    rng = np.random.default_rng(args.seed)
    T = rng.standard_normal(args.n).cumsum().astype(np.float64)
    starts = rng.integers(0, args.n - args.m + 1, size=args.query_count)
    queries = [T[i : i + args.m].copy() for i in starts]
    packed_mib = (args.n - args.m + 1) * args.m * 4 / (1 << 20)

    print(provenance())
    print(
        f"prepared target: n={args.n} m={args.m} query_count={args.query_count} "
        f"repeat={args.repeat} packed_float32={packed_mib:.1f} MiB; "
        "random-walk windows; median of full-query batches"
    )
    print("| operation | ordinary/query | prepared/query | speedup | prepare once | exact |")
    print("|---|---:|---:|---:|---:|---:|")
    for kind in ("mass", "match"):
        ordinary, prepared, build = _benchmark(kind, T, args.m, queries, args.repeat)
        print(
            f"| {kind} | {ordinary * 1000:.2f} ms | {prepared * 1000:.2f} ms "
            f"| {ordinary / prepared:.2f}x | {build * 1000:.2f} ms | yes |"
        )


if __name__ == "__main__":
    main()
