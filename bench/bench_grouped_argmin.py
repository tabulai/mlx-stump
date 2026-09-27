"""Prototype: one SIMD group per narrow AB-join query row.

The current reducer launches one threadgroup per row, even when a row has
only 32–128 target columns. This benchmark compares it with four independent
SIMD groups in one 128-thread group. Both evaluate the same distance expression
and exact (distance, column) tie rule. Only `k=1` AB joins are modeled.

Usage:
    PYTHONPATH=src python bench/bench_grouped_argmin.py --rows 8192 --width 128
"""

from __future__ import annotations

import argparse
import time
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
from bench_stump import provenance

from mlx_stump import _kernels


def grouped_kernel(normalize: bool, rows_per_group: int):
    pre, d2 = _kernels._dist(normalize)
    body = (
        r"""
    const uint row = ROWS_PER_GROUP * threadgroup_position_in_grid.y
        + simdgroup_index_in_threadgroup;
    if (row >= (uint)QT_shape[0]) return;
    const uint lane = thread_index_in_simdgroup;
    const int W = QT_shape[1];
    const int i = par[0] + (int)row;
    const int J0 = par[1];
    const size_t qbase = (size_t)row * (size_t)W;
    const float c_m = consts[0];
    const float c_inv_m = consts[1];
    const float c_2m = consts[2];
    const float c_4m = consts[3];
    const bool q_fin = qf[i];
"""
        + pre
        + r"""
    float bb = INFINITY;
    uint kb = 0xFFFFFFFFu;
    for (int j = (int)lane; j < W; j += 32) {
        const int jg = J0 + j;
"""
        + d2
        + r"""
        lex_min(bb, kb, d2, (uint)j);
    }
    for (ushort off = 16; off > 0; off >>= 1) {
        float ov = simd_shuffle_down(bb, off);
        uint ok = simd_shuffle_down(kb, off);
        lex_min(bb, kb, ov, ok);
    }
    if (lane == 0) {
        outI[row] = kb == 0xFFFFFFFFu ? -1 : (int)kb;
        outP[row] = bb;
    }
"""
    )
    return mx.fast.metal_kernel(
        name=f"mlx_stump_grouped_argmin_{'z' if normalize else 'a'}_{rows_per_group}",
        input_names=["QT", "qa", "qb", "qf", "ta", "tb", "tf", "par", "consts"],
        output_names=["outI", "outP"],
        source=body,
        header=f"#define ROWS_PER_GROUP {rows_per_group}\n" + _kernels._HEADER,
    )


def _series(length: int, normalize: bool, rng):
    flag = rng.random(length) < 0.1
    finite = rng.random(length) > 0.05
    if normalize:
        return SimpleNamespace(
            l=length,
            sig_inv_mx=mx.array(rng.random(length, dtype=np.float32) + 0.5),
            isconstant_mx=mx.array(flag),
            isfinite_mx=mx.array(finite),
        )
    return SimpleNamespace(
        l=length,
        ssq_mx=mx.array(rng.random(length, dtype=np.float32) * 64),
        mu_mx=mx.array(rng.random((length, 2), dtype=np.float32)),
        isfinite_mx=mx.array(finite),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=8192)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--repeat", type=int, default=25)
    ap.add_argument("--raw", action="store_true")
    args = ap.parse_args()
    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        ap.error("a Metal GPU must be the active MLX device")
    if args.rows < 1 or args.width < 1 or args.repeat < 1:
        ap.error("rows, width and repeat must be positive")
    normalize = not args.raw
    rng = np.random.default_rng(448)
    q = _series(args.rows, normalize, rng)
    t = _series(args.width, normalize, rng)
    qt = mx.array(rng.random((args.rows, args.width), dtype=np.float32) * 32)
    consts_np = np.array([64, 1 / 64, 128, 256], dtype=np.float32)
    old = _kernels.FusedReduce(
        q, t, normalize=normalize, self_join=False, excl=0, k=1, consts=consts_np
    )
    kernels = {r: grouped_kernel(normalize, r) for r in (1, 4)}
    inp = [
        qt,
        *old._q,
        *old._t,
        mx.array([0, 0, 0, 0], dtype=mx.int32),
        mx.array(consts_np),
    ]
    mx.eval(*inp)

    def baseline():
        out = old.full(qt, 0)
        mx.eval(*out)
        mx.synchronize()
        return out

    def grouped(rows_per_group):
        out = kernels[rows_per_group](
            inputs=inp,
            grid=(32 * rows_per_group, -(-args.rows // rows_per_group), 1),
            threadgroup=(32 * rows_per_group, 1, 1),
            output_shapes=[(args.rows,), (args.rows,)],
            output_dtypes=[mx.int32, mx.float32],
        )
        mx.eval(*out)
        mx.synchronize()
        return out

    a, b, c = baseline(), grouped(1), grouped(4)
    exact = all(
        np.array_equal(np.array(x), np.array(y))
        for other in (b, c)
        for x, y in zip(a, other, strict=True)
    )
    if not exact:
        raise SystemExit("grouped reducer changed the reduction result")
    timings = {"baseline": [], "one_warp": [], "four_warps": []}
    variants = {
        "baseline": baseline,
        "one_warp": lambda: grouped(1),
        "four_warps": lambda: grouped(4),
    }
    for rep in range(args.repeat):
        order = ("baseline", "one_warp", "four_warps")
        if rep % 2:
            order = tuple(reversed(order))
        for name in order:
            start = time.perf_counter()
            variants[name]()
            timings[name].append(time.perf_counter() - start)
    base, one, four = (
        float(np.median(timings[name])) for name in ("baseline", "one_warp", "four_warps")
    )
    print(provenance())
    print(
        f"rows={args.rows} width={args.width} normalize={normalize} repeat={args.repeat} "
        f"baseline_tg={_kernels.argmin_threadgroup(args.width, old._tg)}"
    )
    print(
        f"baseline={base * 1e3:.3f}ms one_warp={one * 1e3:.3f}ms "
        f"four_warps={four * 1e3:.3f}ms speedup={base / four:.2f}x exact={exact}"
    )


if __name__ == "__main__":
    main()
