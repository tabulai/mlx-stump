"""End-to-end tiled-query cache benchmark with exact profile comparison.

The target block cap is reduced to exercise several tiles at a practical
benchmark size. The same `stump` call runs with cache disabled and enabled.

Usage:
    PYTHONPATH=src python bench/bench_tiled_query_cache.py --n 4096 --qn 1536 \
        --m 128 --tile-kib 64 --repeat 3
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from bench_stump import provenance

import mlx_stump
from mlx_stump import _engine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=4096, help="target series length")
    parser.add_argument("--qn", type=int, default=1536, help="query series length")
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--tile-kib", type=int, default=64)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--self-join", action="store_true")
    parser.add_argument("--raw", action="store_true")
    args = parser.parse_args()
    if not (args.n >= args.m >= 3 and args.qn >= args.m and args.k >= 1):
        parser.error("require n and qn >= m >= 3, k >= 1")
    if args.batch < 1 or args.tile_kib < 1 or args.repeat < 1:
        parser.error("batch, tile-kib and repeat must be positive")

    rng = np.random.default_rng(916)
    a = rng.standard_normal(args.qn).cumsum()
    b = rng.standard_normal(args.n).cumsum()
    call = (b, args.m) if args.self_join else (a, args.m, b)
    options = dict(k=args.k, normalize=not args.raw, chunk_size=args.batch)
    if not args.self_join:
        options["ignore_trivial"] = False

    original_dense = _engine._MATMUL_WINDOW_BYTES
    original_tile = _engine._TILE_WINDOW_BYTES
    original_cache = _engine._TILED_QUERY_CACHE_MAX_BYTES
    _engine._MATMUL_WINDOW_BYTES = 0
    _engine._TILE_WINDOW_BYTES = args.tile_kib * 1024

    def run(enabled):
        _engine._TILED_QUERY_CACHE_MAX_BYTES = original_cache if enabled else 0
        return mlx_stump.stump(*call, **options)

    try:
        run(False)
        run(True)
        times = {False: [], True: []}
        outputs = {}
        for rep in range(args.repeat):
            for enabled in ((False, True) if rep % 2 == 0 else (True, False)):
                start = time.perf_counter()
                profile = run(enabled)
                times[enabled].append(time.perf_counter() - start)
                outputs[enabled] = tuple(
                    np.asarray(getattr(profile, field)).copy()
                    for field in ("P_", "I_", "left_I_", "right_I_")
                )
                del profile
        exact = all(
            np.array_equal(a, b)
            for a, b in zip(outputs[False], outputs[True], strict=True)
        )
        old, new = (float(np.median(times[enabled])) for enabled in (False, True))
        target_windows = len(call[0] if args.self_join else call[2]) - args.m + 1
        tile_rows = max(4, _engine._TILE_WINDOW_BYTES // (4 * args.m))
        blocks = -(-target_windows // tile_rows)
        print(provenance())
        print(
            f"tiled-query-cache target_n={args.n} query_n={args.qn} m={args.m} "
            f"k={args.k} blocks={blocks} batch={args.batch} "
            f"join={'self' if args.self_join else 'AB'} normalize={not args.raw}"
        )
        print(f"uncached={old:.4f}s cached={new:.4f}s speedup={old/new:.2f}x exact={exact}")
        if not exact:
            raise SystemExit("cached path changed the result")
    finally:
        _engine._MATMUL_WINDOW_BYTES = original_dense
        _engine._TILE_WINDOW_BYTES = original_tile
        _engine._TILED_QUERY_CACHE_MAX_BYTES = original_cache


if __name__ == "__main__":
    main()
