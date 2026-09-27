"""Compare the current sweep optimizations with their exact prior paths.

The tiled comparison forces a target block size so it runs quickly on modest
series. Both versions use the same fused Metal distance reducer; only the
block-result merge differs. The dense comparison changes only query packing.

Usage:
    PYTHONPATH=src python bench/bench_fusion.py --mode tiled --n 8192 --m 200 --k 16
    PYTHONPATH=src python bench/bench_fusion.py --mode dense --n 4096 --m 2048
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from bench_stump import provenance

import mlx_stump
from mlx_stump import _engine, _stump


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("tiled", "dense"), default="tiled")
    ap.add_argument("--n", type=int, default=8192)
    ap.add_argument("--m", type=int, default=200)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--tile-kib", type=int, default=256)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--ab", action="store_true", help="AB join (tiled mode only)")
    ap.add_argument("--raw", action="store_true", help="non-normalized distances")
    args = ap.parse_args()
    if not (args.n >= args.m >= 3 and args.k >= 1 and args.batch >= 1):
        ap.error("require n >= m >= 3, k >= 1 and batch >= 1")
    if args.repeat < 1 or args.tile_kib < 1:
        ap.error("repeat and tile-kib must be positive")
    if args.mode == "dense" and args.ab:
        ap.error("dense query reuse applies to self joins only")

    rng = np.random.default_rng(145)
    a = rng.standard_normal(args.n).cumsum()
    b = rng.standard_normal(args.n).cumsum() if args.ab else None
    call = (a, args.m, b) if args.ab else (a, args.m)
    options = dict(k=args.k, normalize=not args.raw, chunk_size=args.batch)
    if args.ab:
        options["ignore_trivial"] = False

    original_tiled = _stump._compute_profile_tiled
    original_queries = _stump._query_args
    original_cap = _engine._MATMUL_WINDOW_BYTES
    original_tile = _engine._TILE_WINDOW_BYTES
    if args.mode == "tiled":
        _engine._MATMUL_WINDOW_BYTES = 0
        _engine._TILE_WINDOW_BYTES = args.tile_kib * 1024

    def host_merge(*pos, **kw):
        return original_tiled(*pos, **kw, metal_merge=False)

    def fresh_queries(batches, query, normalize, packed_target=None):
        return original_queries(batches, query, normalize)

    def run(which):
        _stump._compute_profile_tiled = host_merge if which == "old" else original_tiled
        _stump._query_args = fresh_queries if which == "old" else original_queries
        return mlx_stump.stump(*call, **options)

    try:
        # Compile/warm both kernel paths before timing.
        run("old")
        run("new")
        timings = {"old": [], "new": []}
        results = {}
        for r in range(args.repeat):
            order = ("old", "new") if r % 2 == 0 else ("new", "old")
            for which in order:
                start = time.perf_counter()
                result = run(which)
                timings[which].append(time.perf_counter() - start)
                results[which] = tuple(
                    np.array(getattr(result, field), copy=True)
                    for field in ("P_", "I_", "left_I_", "right_I_")
                )
                del result
        old, new = results["old"], results["new"]
        equal = all(
            np.array_equal(a, b) for a, b in zip(old, new, strict=True)
        )
        old_t = float(np.median(timings["old"]))
        new_t = float(np.median(timings["new"]))
        print(provenance())
        print(
            f"{args.mode} n={args.n} m={args.m} k={args.k} "
            f"join={'AB' if args.ab else 'self'} normalize={not args.raw} "
            f"repeat={args.repeat}"
        )
        print(f"old={old_t:.4f}s new={new_t:.4f}s speedup={old_t / new_t:.2f}x exact={equal}")
        if not equal:
            raise SystemExit("optimized path changed the result")
    finally:
        _stump._compute_profile_tiled = original_tiled
        _stump._query_args = original_queries
        _engine._MATMUL_WINDOW_BYTES = original_cap
        _engine._TILE_WINDOW_BYTES = original_tile


if __name__ == "__main__":
    main()
