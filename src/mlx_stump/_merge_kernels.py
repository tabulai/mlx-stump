"""Metal merge of per-block results in a tiled matrix-profile sweep."""

from __future__ import annotations

import mlx.core as mx

from ._kernels import FUSED_TOPK_MAX

_KERNELS: dict[bool, object] = {}

_HEADER = r"""
inline uint tie_key(int j, int i) {
    if (j < 0) return 0xFFFFFFFFu;
    if (SELF_JOIN) {
        const int delta = j - i;
        return 2u * (uint)abs(delta) + (delta > 0 ? 1u : 0u);
    }
    return (uint)j;
}
inline bool before(float av, uint ak, float bv, uint bk) {
    return av < bv || (av == bv && ak < bk);
}
"""

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

_SIDES = r"""
    const float pl = blkPl[row];
    if (pl <= runPl[row]) {
        outPl[row] = pl;
        outIl[row] = metal::isfinite(pl) && blkIl[row] >= 0 ? blkIl[row] + j0 : -1;
    } else {
        outPl[row] = runPl[row];
        outIl[row] = runIl[row];
    }
    const float pr = blkPr[row];
    if (pr < runPr[row]) {
        outPr[row] = pr;
        outIr[row] = metal::isfinite(pr) && blkIr[row] >= 0 ? blkIr[row] + j0 : -1;
    } else {
        outPr[row] = runPr[row];
        outIr[row] = runIr[row];
    }
"""


def _kernel(self_join: bool):
    if self_join not in _KERNELS:
        side_in = (
            ["runIl", "runPl", "runIr", "runPr", "blkIl", "blkPl", "blkIr", "blkPr"]
            if self_join else []
        )
        side_out = ["outIl", "outPl", "outIr", "outPr"] if self_join else []
        _KERNELS[self_join] = mx.fast.metal_kernel(
            name=f"mlx_stump_tiled_merge_{'self' if self_join else 'ab'}",
            input_names=["runV", "runI", "blkV", "blkI", *side_in, "par"],
            output_names=["outV", "outI", *side_out],
            source=_BODY + (_SIDES if self_join else ""),
            header=f"#define SELF_JOIN {int(self_join)}\n" + _HEADER,
        )
    return _KERNELS[self_join]


class TiledMerge:
    """Accumulate sorted block results on the GPU for k up to sixteen."""

    def __init__(self, l_q: int, k: int, self_join: bool):
        if not 1 <= k <= FUSED_TOPK_MAX:
            raise ValueError(f"Metal tiled merge supports 1 <= k <= {FUSED_TOPK_MAX}")
        self.l_q = l_q
        self.k = k
        self.self_join = self_join
        shape = (l_q,) if k == 1 else (l_q, k)
        self._run = (
            mx.full(shape, float("inf"), dtype=mx.float32),
            mx.full(shape, -1, dtype=mx.int32),
        )
        if self_join:
            self._run += (
                mx.full((l_q,), -1, dtype=mx.int32),
                mx.full((l_q,), float("inf"), dtype=mx.float32),
                mx.full((l_q,), -1, dtype=mx.int32),
                mx.full((l_q,), float("inf"), dtype=mx.float32),
            )
        self._kern = _kernel(self_join)

    def add(self, block: tuple[mx.array, ...], j0: int) -> None:
        """Merge a complete target block; block indices are block-local."""
        shape = (self.l_q,) if self.k == 1 else (self.l_q, self.k)
        shapes = [shape, shape]
        dtypes = [mx.float32, mx.int32]
        if self.self_join:
            shapes += [(self.l_q,)] * 4
            dtypes += [mx.int32, mx.float32, mx.int32, mx.float32]
        # The reducer returns (I, P) for k=1 and (P, I) otherwise.
        blk = (block[1], block[0], *block[2:]) if self.k == 1 else block
        par = mx.array([int(j0)], dtype=mx.int32)
        self._run = tuple(
            self._kern(
                inputs=[*self._run[:2], *blk[:2], *self._run[2:], *blk[2:], par],
                template=[("KK", self.k)],
                grid=(self.l_q, 1, 1),
                threadgroup=(min(256, self.l_q), 1, 1),
                output_shapes=shapes,
                output_dtypes=dtypes,
            )
        )
        mx.eval(*self._run)
        mx.synchronize()

    def result(self) -> tuple[mx.array, ...]:
        return self._run
