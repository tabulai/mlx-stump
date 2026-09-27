"""Prototype: keep the two tiled top-k merge heads in Metal registers.

This is a reducer-only microbenchmark. It compares the production merge
kernel to a version that loads a list head only when its cursor advances.
Both use the same thread geometry, tie rule, outputs, and synchronization.

Usage:
    PYTHONPATH=src python bench/bench_merge_heads.py --rows 65536 --k 16
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx
import numpy as np
from bench_stump import provenance

import mlx_stump
from mlx_stump import _engine, _merge_kernels

_BODY = r"""
    const uint row = thread_position_in_grid.x;
    if (row >= (uint)runV_shape[0]) return;
    const int j0 = par[0];
    const uint base = row * KK;
    int a = 0, b = 0;
    float av = runV[base], bv = blkV[base];
    int ai = metal::isfinite(av) ? runI[base] : -1;
    int raw_bi = blkI[base];
    int bi = metal::isfinite(bv) && raw_bi >= 0 ? raw_bi + j0 : -1;
    uint ak = tie_key(ai, (int)row), bk = tie_key(bi, (int)row);
    for (int t = 0; t < KK; t++) {
        if (before(bv, bk, av, ak)) {
            outV[base + t] = bv;
            outI[base + t] = bi;
            b++;
            if (b < KK) {
                bv = blkV[base + b];
                raw_bi = blkI[base + b];
                bi = metal::isfinite(bv) && raw_bi >= 0 ? raw_bi + j0 : -1;
                bk = tie_key(bi, (int)row);
            } else {
                bv = INFINITY;
                bi = -1;
                bk = 0xFFFFFFFFu;
            }
        } else {
            outV[base + t] = av;
            outI[base + t] = ai;
            a++;
            if (a < KK) {
                av = runV[base + a];
                ai = metal::isfinite(av) ? runI[base + a] : -1;
                ak = tie_key(ai, (int)row);
            } else {
                av = INFINITY;
                ai = -1;
                ak = 0xFFFFFFFFu;
            }
        }
    }
"""


def _inputs(rows: int, k: int, self_join: bool, seed: int):
    rng = np.random.default_rng(seed)
    # Quantized values force many exact ties, including ties across the two
    # sorted lists. An infinity suffix tests the missing-candidate sentinel.
    run_v = rng.integers(0, 32, size=(rows, k)).astype(np.float32)
    blk_v = rng.integers(0, 32, size=(rows, k)).astype(np.float32)
    run_i = rng.integers(0, rows, size=(rows, k), dtype=np.int32)
    blk_i = rng.integers(0, rows, size=(rows, k), dtype=np.int32)
    run_v[:, -1] = np.inf
    blk_v[:, -1] = np.inf
    run_i[:, -1] = -1
    blk_i[:, -1] = -1
    row = np.arange(rows, dtype=np.int32)[:, None]
    for val, idx, shift in ((run_v, run_i, 0), (blk_v, blk_i, rows + 100)):
        full = idx + shift
        key = 2 * np.abs(full - row) + (full > row) if self_join else full
        key[idx < 0] = np.iinfo(np.int32).max
        order = np.lexsort((key, val), axis=1)
        val[:] = np.take_along_axis(val, order, axis=1)
        idx[:] = np.take_along_axis(idx, order, axis=1)
    inputs = [mx.array(x) for x in (run_v, run_i, blk_v, blk_i)]
    if self_join:
        side_v = rng.integers(0, 32, size=rows).astype(np.float32)
        side_i = rng.integers(0, rows, size=rows, dtype=np.int32)
        inputs += [
            mx.array(side_i),
            mx.array(side_v),
            mx.array(side_i),
            mx.array(side_v),
            mx.array(side_i),
            mx.array(side_v),
            mx.array(side_i),
            mx.array(side_v),
        ]
    inputs.append(mx.array([rows + 100], dtype=mx.int32))
    mx.eval(*inputs)
    return inputs


def prototype_kernel(self_join: bool):
    side_names = (
        ["runIl", "runPl", "runIr", "runPr", "blkIl", "blkPl", "blkIr", "blkPr"]
        if self_join
        else []
    )
    side_out = ["outIl", "outPl", "outIr", "outPr"] if self_join else []
    header = f"#define SELF_JOIN {int(self_join)}\n" + _merge_kernels._HEADER
    return mx.fast.metal_kernel(
        name=f"mlx_stump_merge_heads_{int(self_join)}",
        input_names=["runV", "runI", "blkV", "blkI", *side_names, "par"],
        output_names=["outV", "outI", *side_out],
        source=_BODY + (_merge_kernels._SIDES if self_join else ""),
        header=header,
    )


def benchmark_profile(args) -> None:
    rng = np.random.default_rng(447)
    a = rng.standard_normal(args.profile_n).cumsum()
    b = None if args.self_join else rng.standard_normal(args.profile_n).cumsum()
    call = (a, args.m) if args.self_join else (a, args.m, b)
    options = dict(k=args.k, chunk_size=args.batch, normalize=not args.raw)
    if not args.self_join:
        options["ignore_trivial"] = False
    old_dense, old_tile = _engine._MATMUL_WINDOW_BYTES, _engine._TILE_WINDOW_BYTES
    old_kernel = _merge_kernels._KERNELS.get(args.self_join)
    _engine._MATMUL_WINDOW_BYTES = 0
    _engine._TILE_WINDOW_BYTES = args.tile_kib * 1024

    def run(variant):
        if variant == "heads":
            _merge_kernels._KERNELS[args.self_join] = prototype
        elif old_kernel is None:
            _merge_kernels._KERNELS.pop(args.self_join, None)
        else:
            _merge_kernels._KERNELS[args.self_join] = old_kernel
        return mlx_stump.stump(*call, **options)

    try:
        baseline = _merge_kernels._kernel(args.self_join)
        prototype = prototype_kernel(args.self_join)
        old_kernel = baseline
        run("baseline")
        run("heads")
        timings = {"baseline": [], "heads": []}
        results = {}
        for rep in range(args.repeat):
            order = ("baseline", "heads") if rep % 2 == 0 else ("heads", "baseline")
            for variant in order:
                start = time.perf_counter()
                result = run(variant)
                timings[variant].append(time.perf_counter() - start)
                results[variant] = tuple(
                    np.asarray(getattr(result, field)).copy()
                    for field in ("P_", "I_", "left_I_", "right_I_")
                )
                del result
        exact = all(
            np.array_equal(a, b) for a, b in zip(results["baseline"], results["heads"], strict=True)
        )
        baseline_t, heads_t = (float(np.median(timings[key])) for key in ("baseline", "heads"))
        print(provenance())
        print(
            f"stump n={args.profile_n} m={args.m} k={args.k} batch={args.batch} "
            f"tile_kib={args.tile_kib} self_join={args.self_join} normalize={not args.raw}"
        )
        print(
            f"baseline={baseline_t:.4f}s heads={heads_t:.4f}s "
            f"speedup={baseline_t / heads_t:.2f}x exact={exact}"
        )
        if not exact:
            raise SystemExit("register-head prototype changed the matrix profile")
    finally:
        _engine._MATMUL_WINDOW_BYTES, _engine._TILE_WINDOW_BYTES = old_dense, old_tile
        if old_kernel is None:
            _merge_kernels._KERNELS.pop(args.self_join, None)
        else:
            _merge_kernels._KERNELS[args.self_join] = old_kernel


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=65536)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--repeat", type=int, default=15)
    ap.add_argument("--self-join", action="store_true")
    ap.add_argument("--profile-n", type=int, default=0, help="benchmark an end-to-end tiled stump")
    ap.add_argument("--m", type=int, default=200)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--tile-kib", type=int, default=256)
    ap.add_argument("--raw", action="store_true")
    args = ap.parse_args()
    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        ap.error("a Metal GPU must be the active MLX device")
    if not (args.rows >= 1 and 1 <= args.k <= 16 and args.repeat >= 1):
        ap.error("rows and repeat must be positive; k must be 1–16")
    if args.profile_n:
        if args.profile_n < args.m or args.m < 3 or args.batch < 1 or args.tile_kib < 1:
            ap.error("profile-n must be >= m >= 3; batch and tile-kib must be positive")
        benchmark_profile(args)
        return
    inp = _inputs(args.rows, args.k, args.self_join, 447)
    prototype = prototype_kernel(args.self_join)
    baseline = _merge_kernels._kernel(args.self_join)
    shape = (args.rows, args.k)
    shapes = [shape, shape]
    dtypes = [mx.float32, mx.int32]
    if args.self_join:
        shapes += [(args.rows,)] * 4
        dtypes += [mx.int32, mx.float32, mx.int32, mx.float32]

    def run(kernel):
        outs = kernel(
            inputs=inp,
            template=[("KK", args.k)],
            grid=(args.rows, 1, 1),
            threadgroup=(min(256, args.rows), 1, 1),
            output_shapes=shapes,
            output_dtypes=dtypes,
        )
        mx.eval(*outs)
        mx.synchronize()
        return outs

    old, new = run(baseline), run(prototype)
    equal = all(np.array_equal(np.array(a), np.array(b)) for a, b in zip(old, new, strict=True))
    if not equal:
        raise SystemExit("register-head prototype changed the merge result")
    timings = {"baseline": [], "heads": []}
    kernels = {"baseline": baseline, "heads": prototype}
    for rep in range(args.repeat):
        order = ("baseline", "heads") if rep % 2 == 0 else ("heads", "baseline")
        for name in order:
            start = time.perf_counter()
            run(kernels[name])
            timings[name].append(time.perf_counter() - start)
    base, heads = (float(np.median(timings[name])) for name in ("baseline", "heads"))
    print(provenance())
    print(f"rows={args.rows} k={args.k} self_join={args.self_join} repeat={args.repeat}")
    print(
        f"baseline={base * 1e3:.3f}ms heads={heads * 1e3:.3f}ms "
        f"speedup={base / heads:.2f}x exact={equal}"
    )


if __name__ == "__main__":
    main()
