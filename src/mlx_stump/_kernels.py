"""One-pass Metal row reductions for the stump sweep (GPU only).

``FusedReduce`` is the fast path behind ``_engine.make_reducer``: one
threadgroup per query row reads each ``QT`` element exactly once, evaluates
the same float32 distance expressions as ``_engine._znorm_sq`` /
``_engine._abs_sq`` in registers, and keeps running minima (``k == 1``) or a
sorted top-k list (``2 <= k <= FUSED_TOPK_MAX``) that a simdgroup/threadgroup
reduction combines. No ``(B, width)`` intermediate besides ``QT`` itself is
materialized, so the per-batch device footprint is 4 bytes per ``QT`` cell
plus O(k) bytes per row.

Bit-identity with the compiled fallback (``_engine.ReduceStep``, the
reference) is by construction:

- no FMA contraction: the Metal compiler fuses ``x + (m*dmu)*dmu`` (raw
  mode) into an FMA, which MLX's own kernels do not do. ``#pragma METAL fp
  contract(off)`` prevents it where the compiler honours the pragma, but
  the Metal compiler of macOS 14 ignores it (measured on GitHub's macos-14
  runners), so the raw product is also passed through a bitwise OR with a
  runtime zero (``par[3]``) that no compiler can fold: the product is then
  rounded on its own before the addition, on every macOS;
- the same float32 constants: the kernels read ``m``, the host-computed
  ``float32(1/m)``, ``2m`` and ``4m`` from one input buffer; the compiled
  step receives ``m`` and the same ``float32(1/m)`` as 0-d arrays and forms
  ``2m`` and ``4m`` on the device, which is exact in float32 for integer
  ``m``;
- MLX's NaN-propagating ``maximum``/``minimum`` (``isnan(x) ? x : ...``);
- the same lexicographic ``(d2, key)`` selection order (see ``_engine``):
  self-joins use ``key = 2*|j - i| + (j > i)``, AB-joins ``key = j``.

The row offset, the block's first global column, the exclusion zone and the
block width are runtime inputs (``par`` and ``QT_shape``), not template
constants, so a new series length or window size never triggers a JIT
compile; only ``k`` (the register list length) and the threadgroup size are
template parameters. ``QT`` is indexed directly: MLX binds size-1 inputs in
the ``constant`` address space, so aliasing it as a ``device`` pointer would
not compile for a one-window series.

Before first use, ``launch_threadgroup(k)`` dispatches every variant of the
``k`` kernel once on a tiny input: MLX rejects a threadgroup wider than the
pipeline's ``maxTotalThreadsPerThreadgroup``, which depends on the GPU and
the kernel's register use, so the preferred width is halved until all
variants launch (the lexicographic reduction does not depend on it). If
none does, or the kernel fails to build, the dispatch predicate
(``_engine._fused_reducer``) sends ``k`` to the compiled fallback, whose
batches it also sizes.
"""

from __future__ import annotations

import os
import sys
import warnings
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

# largest k served by the top-k kernel; larger k use the compiled fallback.
# In review measurements a per-thread-list kernel beat the compiled step
# 8x per batch at k=16 but only 1.5-1.9x at k=32-48, broke even at 64,
# lost at 100 (0.7x), and at k >= 128 exceeded the 32 KiB of threadgroup
# memory Apple GPUs provide.
FUSED_TOPK_MAX = 16
_TG_MEM_LIMIT = 32 * 1024
_ARGMIN_TG = 1024

_HEADER = r"""
#pragma METAL fp contract(off)

// lexicographic (value, key) order: the tie rule of the whole sweep
inline bool lex_lt(float a, uint ka, float b, uint kb) {
    return a < b || (a == b && ka < kb);
}

inline void lex_min(thread float &bv, thread uint &bk, float v, uint k) {
    if (lex_lt(v, k, bv, bk)) { bv = v; bk = k; }
}

// MLX's float maximum/minimum: a NaN first operand propagates
inline float mlx_max(float x, float y) { return metal::isnan(x) ? x : (x > y ? x : y); }
inline float mlx_min(float x, float y) { return metal::isnan(x) ? x : (x < y ? x : y); }
"""

# per-row prologue shared by every kernel; ``i`` is the global query row,
# ``J0`` the global column of QT[:, 0]
_PROLOGUE = r"""
    const uint row = threadgroup_position_in_grid.y;
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const int W = QT_shape[1];
    const int i = par[0] + (int)row;
    const int J0 = par[1];
    const int EXCL = par[2];
    const size_t qbase = (size_t)row * (size_t)W;
    const float c_m = consts[0];
    const float c_inv_m = consts[1];
    const float c_2m = consts[2];
    const float c_4m = consts[3];
    const bool q_fin = qf[i];
"""

# _engine._znorm_sq, operation for operation
_ZNORM_PRE = r"""
    const float q_a = qa[i];
    const bool q_c = qb[i];
"""
_ZNORM_D2 = r"""
        float rho = QT[qbase + j] * ((q_a * ta[jg]) * c_inv_m);
        float d2 = mlx_min(mlx_max(c_2m * (1.0f - rho), 0.0f), c_4m);
        const bool t_c = tb[jg];
        d2 = (q_c && t_c) ? 0.0f : ((q_c || t_c) ? c_m : d2);
        d2 = (q_fin && tf[jg]) ? d2 : INFINITY;
"""
# _engine._abs_sq, operation for operation; mu is a (hi, lo) float32 pair
_ABS_PRE = r"""
    const float q_s = qa[i];
    const float q_mu0 = qb[2 * i];
    const float q_mu1 = qb[2 * i + 1];
    const uint zbits = (uint)par[3];  // always 0, unknown to the compiler
"""
_ABS_D2 = r"""
        float dmu = (q_mu0 - tb[2 * jg]) + (q_mu1 - tb[2 * jg + 1]);
        float x = (q_s + ta[jg]) - 2.0f * QT[qbase + j];
        // round the product on its own (see the module docstring): OR-ing
        // its bits with a runtime zero keeps any compiler from fusing the
        // sum below into an FMA, even one that ignores the contract pragma
        float p = (c_m * dmu) * dmu;
        p = as_type<float>(as_type<uint>(p) | zbits);
        float d2 = mlx_max(x, 0.0f) + p;
        d2 = (q_fin && tf[jg]) ? d2 : INFINITY;
"""

# self-join candidate key 2*|jg - i| + (jg > i) and the left/right minima;
# the trivial-match zone is excluded from both and from the top-k list
_SELF_LR = r"""
        const int dj = jg - i;
        const uint key = 2u * (uint)abs(dj) + (dj > 0 ? 1u : 0u);
        if (dj <= -(EXCL + 1)) {
            lex_min(bl, kl, d2, key);
        } else if (dj >= EXCL + 1) {
            lex_min(br, kr, d2, key);
        } else {
            d2 = INFINITY;
        }
"""


def _lex_reduce(pairs: list[tuple[str, str]]) -> str:
    """Reduce per-thread lexicographic minima across the threadgroup (``TG``
    threads, a multiple of 32); afterwards simdgroup 0, lane 0 holds the
    row's minima in the same variables."""
    shuffle = "".join(
        f"""
        {{ float ov = simd_shuffle_down({v}, off); uint ok = simd_shuffle_down({k}, off);
          lex_min({v}, {k}, ov, ok); }}"""
        for v, k in pairs
    )
    decl = "".join(
        f"    threadgroup float tg_{v}[TG / 32]; threadgroup uint tg_{k}[TG / 32];\n"
        for v, k in pairs
    )
    store = "".join(f" tg_{v}[sg] = {v}; tg_{k}[sg] = {k};" for v, k in pairs)
    load = "".join(
        f"""
        {v} = lane < (uint)(TG / 32) ? tg_{v}[lane] : INFINITY;
        {k} = lane < (uint)(TG / 32) ? tg_{k}[lane] : 0xFFFFFFFFu;"""
        for v, k in pairs
    )
    return (
        f"""
    for (ushort off = 16; off > 0; off >>= 1) {{{shuffle}
    }}
"""
        + decl
        + f"""    if (lane == 0) {{{store} }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {{{load}
        for (ushort off = 16; off > 0; off >>= 1) {{{shuffle}
        }}
    }}
"""
    )


# block-local column of a self-join key (-1 for the empty sentinel)
_SELF_COL = r"""
inline int self_col(uint key, int i, int J0) {
    if (key == 0xFFFFFFFFu) return -1;
    const int d = (int)(key >> 1);
    return ((key & 1u) ? i + d : i - d) - J0;
}
"""


def _dist(normalize: bool) -> tuple[str, str]:
    return (_ZNORM_PRE, _ZNORM_D2) if normalize else (_ABS_PRE, _ABS_D2)


def _argmin_source(normalize: bool, self_join: bool) -> str:
    pre, d2 = _dist(normalize)
    if self_join:
        body = (
            r"""
    float bl = INFINITY, br = INFINITY;
    uint kl = 0xFFFFFFFFu, kr = 0xFFFFFFFFu;
    for (int j = (int)tid; j < W; j += TG) {
        const int jg = J0 + j;
"""
            + d2
            + _SELF_LR
            + "    }\n"
            + _lex_reduce([("bl", "kl"), ("br", "kr")])
            + r"""
    if (sg == 0 && lane == 0) {
        const int jl = self_col(kl, i, J0);
        const int jr = self_col(kr, i, J0);
        const bool left = lex_lt(bl, kl, br, kr);
        outI[row] = left ? jl : jr;
        outP[row] = left ? bl : br;
        outIl[row] = jl;
        outPl[row] = bl;
        outIr[row] = jr;
        outPr[row] = br;
    }
"""
        )
    else:
        body = (
            r"""
    float bb = INFINITY;
    uint kb = 0xFFFFFFFFu;
    for (int j = (int)tid; j < W; j += TG) {
        const int jg = J0 + j;
"""
            + d2
            + "        lex_min(bb, kb, d2, (uint)j);\n    }\n"
            + _lex_reduce([("bb", "kb")])
            + r"""
    if (sg == 0 && lane == 0) {
        outI[row] = kb == 0xFFFFFFFFu ? -1 : (int)kb;
        outP[row] = bb;
    }
"""
        )
    return _PROLOGUE + pre + body


def _topk_source(normalize: bool, self_join: bool) -> str:
    pre, d2 = _dist(normalize)
    select = _SELF_LR if self_join else "        const uint key = (uint)j;\n"
    body = (
        r"""
    float v[KK];
    uint kv[KK];
    for (int t = 0; t < KK; t++) { v[t] = INFINITY; kv[t] = 0xFFFFFFFFu; }
    float bl = INFINITY, br = INFINITY;
    uint kl = 0xFFFFFFFFu, kr = 0xFFFFFFFFu;
    for (int j = (int)tid; j < W; j += TG) {
        const int jg = J0 + j;
"""
        + d2
        + select
        + r"""
        if (lex_lt(d2, key, v[KK - 1], kv[KK - 1])) {
            int p = KK - 1;
            while (p > 0 && lex_lt(d2, key, v[p - 1], kv[p - 1])) {
                v[p] = v[p - 1];
                kv[p] = kv[p - 1];
                p--;
            }
            v[p] = d2;
            kv[p] = key;
        }
    }
    // pairwise tree merge of the per-thread sorted lists
    threadgroup float sv[TG * KK];
    threadgroup uint sk[TG * KK];
    for (int t = 0; t < KK; t++) { sv[tid * KK + t] = v[t]; sk[tid * KK + t] = kv[t]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 1; s < (uint)TG; s <<= 1) {
        if ((tid % (2 * s)) == 0) {
            const uint a = tid * KK, b = (tid + s) * KK;
            int pa = 0, pb = 0;
            for (int t = 0; t < KK; t++) {
                if (lex_lt(sv[b + pb], sk[b + pb], sv[a + pa], sk[a + pa])) {
                    v[t] = sv[b + pb]; kv[t] = sk[b + pb]; pb++;
                } else {
                    v[t] = sv[a + pa]; kv[t] = sk[a + pa]; pa++;
                }
            }
            for (int t = 0; t < KK; t++) { sv[a + t] = v[t]; sk[a + t] = kv[t]; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int t = (int)tid; t < KK; t += TG) {
        outV[row * KK + t] = sv[t];
"""
        + (
            "        outI[row * KK + t] = self_col(sk[t], i, J0);\n"
            if self_join
            else "        outI[row * KK + t] = sk[t] == 0xFFFFFFFFu ? -1 : (int)sk[t];\n"
        )
        + "    }\n"
    )
    if self_join:
        body += _lex_reduce([("bl", "kl"), ("br", "kr")]) + r"""
    if (sg == 0 && lane == 0) {
        outIl[row] = self_col(kl, i, J0);
        outPl[row] = bl;
        outIr[row] = self_col(kr, i, J0);
        outPr[row] = br;
    }
"""
    return _PROLOGUE + pre + body


def topk_threadgroup(k: int) -> int:
    """Preferred threads per row for the top-k kernel: the per-thread lists
    live in threadgroup memory during the merge, so wide lists get fewer
    threads."""
    tg = 256
    while tg > 32 and tg * k * 8 > 16 * 1024:
        tg //= 2
    return tg


def topk_threadgroup_bytes(k: int, tg: int | None = None) -> int:
    """Threadgroup memory of the top-k kernel at ``tg`` threads (default: the
    preferred width, the largest it launches with): the value/key lists plus
    the left/right reduction scratch."""
    tg = topk_threadgroup(k) if tg is None else tg
    return tg * k * 8 + 2 * 8 * max(1, tg // 32)


_KERNELS: dict = {}
# k -> the threadgroup width all four variants of the k kernel launch with
# on this GPU, or None when they cannot run here (see launch_threadgroup)
_LAUNCH: dict = {}
_PKG_PREFIX = os.path.dirname(__file__) + os.sep


def _warn_outside_package(message: str, category: type[Warning]) -> None:
    """Warn at the first frame outside this package: the depth of the
    user's call differs per entry point (``estimated_peak_bytes``, ``stump``,
    the tiled sweep, ``stimp``), so no fixed ``stacklevel`` is right."""
    if sys.version_info >= (3, 12):
        warnings.warn(message, category, skip_file_prefixes=(_PKG_PREFIX,))
        return
    f, level = sys._getframe(1), 1
    while f is not None and f.f_code.co_filename.startswith(_PKG_PREFIX):
        f, level = f.f_back, level + 1
    warnings.warn(message, category, stacklevel=level + 1)


def _kernel(kind: str, normalize: bool, self_join: bool):
    key = (kind, normalize, self_join)
    if key not in _KERNELS:
        lr = ["outIl", "outPl", "outIr", "outPr"] if self_join else []
        if kind == "argmin":
            source = _argmin_source(normalize, self_join)
            outs = ["outI", "outP", *lr]
        else:
            source = _topk_source(normalize, self_join)
            outs = ["outV", "outI", *lr]
        _KERNELS[key] = mx.fast.metal_kernel(
            name=f"mlx_stump_{kind}_{'z' if normalize else 'a'}{'s' if self_join else 'ab'}",
            input_names=["QT", "qa", "qb", "qf", "ta", "tb", "tf", "par", "consts"],
            output_names=outs,
            source=source,
            header=_HEADER + _SELF_COL,
        )
    return _KERNELS[key]


class FusedReduce:
    """The fused sweep reduction for ``k <= FUSED_TOPK_MAX``.

    Same call convention and outputs as ``_engine.ReduceStep``: ``full(QT,
    s0)`` / ``block(QT, s0, j0, j1)`` for the batch whose first query row is
    ``s0``; indices are block-local int32. A top-k list is always ``k`` wide:
    columns beyond the block's width hold ``(inf, -1)``. ``threadgroup``
    overrides the probed width (``launch_threadgroup``; the probe passes it).
    """

    def __init__(
        self,
        query,
        target,
        *,
        normalize: bool,
        self_join: bool,
        excl: int,
        k: int,
        consts: np.ndarray,
        threadgroup: int | None = None,
    ):
        if not 1 <= k <= FUSED_TOPK_MAX:
            raise ValueError(f"fused reduction supports 1 <= k <= {FUSED_TOPK_MAX}, got {k}")
        tg = launch_threadgroup(k) if threadgroup is None else threadgroup
        if tg is None:
            raise RuntimeError(f"the fused k={k} kernels cannot run on this device")
        self.k = k
        self.self_join = self_join
        # as in ReduceStep: a zone of l columns excludes every self-join
        # candidate, and the clamp keeps EXCL + 1 and the int32 parameter
        # buffer from overflowing (an extreme denominator raised bad_cast)
        self.excl = min(int(excl), int(target.l))
        if normalize:
            self._q = (query.sig_inv_mx, query.isconstant_mx, query.isfinite_mx)
            self._t = (target.sig_inv_mx, target.isconstant_mx, target.isfinite_mx)
        else:
            self._q = (query.ssq_mx, query.mu_mx, query.isfinite_mx)
            self._t = (target.ssq_mx, target.mu_mx, target.isfinite_mx)
        self._consts = mx.array(consts)
        self._tg = tg
        if k == 1:
            self._kern = _kernel("argmin", normalize, self_join)
            self._template = [("TG", tg)]
        else:
            self._kern = _kernel("topk", normalize, self_join)
            tg_bytes = topk_threadgroup_bytes(k, tg)
            assert tg_bytes <= _TG_MEM_LIMIT, f"top-k kernel needs {tg_bytes} B threadgroup memory"
            self._template = [("TG", tg), ("KK", k)]

    def _run(self, QT, s0: int, j0: int):
        B = QT.shape[0]
        par = mx.array([int(s0), int(j0), self.excl, 0], dtype=mx.int32)
        lr_shapes = [(B,)] * 4 if self.self_join else []
        lr_dtypes = [mx.int32, mx.float32] * 2 if self.self_join else []
        if self.k == 1:
            shapes = [(B,), (B,), *lr_shapes]
            dtypes = [mx.int32, mx.float32, *lr_dtypes]
        else:
            shapes = [(B, self.k), (B, self.k), *lr_shapes]
            dtypes = [mx.float32, mx.int32, *lr_dtypes]
        return self._kern(
            inputs=[QT, *self._q, *self._t, par, self._consts],
            template=self._template,
            grid=(self._tg, B, 1),
            threadgroup=(self._tg, 1, 1),
            output_shapes=shapes,
            output_dtypes=dtypes,
        )

    def full(self, QT, s0: int):
        return self._run(QT, s0, 0)

    def block(self, QT, s0: int, j0: int, j1: int):
        return self._run(QT, s0, j0)


def _probe(k: int, tg: int) -> None:
    """Build and dispatch all four variants (z-normalized/raw x self/AB-join)
    of the ``k`` kernel at ``tg`` threads on a tiny input with the real
    input dtypes; raises what MLX raises."""
    n = 64
    stats = mx.zeros((n,), dtype=mx.float32)
    flags = mx.zeros((n,), dtype=mx.bool_)
    series = SimpleNamespace(
        l=n,
        sig_inv_mx=stats,
        isconstant_mx=flags,
        isfinite_mx=mx.ones((n,), dtype=mx.bool_),
        ssq_mx=stats,
        mu_mx=mx.zeros((n, 2), dtype=mx.float32),
    )
    consts = np.array([8.0, 1.0 / 8.0, 16.0, 32.0], dtype=np.float32)
    QT = mx.zeros((2, n), dtype=mx.float32)
    outs = []
    for normalize in (True, False):
        for self_join in (True, False):
            red = FusedReduce(
                series,
                series,
                normalize=normalize,
                self_join=self_join,
                excl=1,
                k=k,
                consts=consts,
                threadgroup=tg,
            )
            outs.extend(red.full(QT, 0))
    mx.eval(*outs)


def launch_threadgroup(k: int) -> int | None:
    """Threads per row the ``k`` kernels launch with on this GPU, or None
    when they cannot run here; probed once per process and ``k`` (see the
    module docstring). Call only with a Metal GPU as the default device."""
    if k not in _LAUNCH:
        tg = _ARGMIN_TG if k == 1 else topk_threadgroup(k)
        err = None
        while tg >= 32:
            try:
                _probe(k, tg)
                break
            except Exception as exc:  # any failure: not at this width
                err = exc
                tg //= 2
        else:
            _warn_outside_package(
                f"mlx_stump: the fused Metal kernels for k={k} cannot run on this GPU "
                f"({err}); using the slower compiled reduction.",
                RuntimeWarning,
            )
            tg = None
        _LAUNCH[k] = tg
    return _LAUNCH[k]
