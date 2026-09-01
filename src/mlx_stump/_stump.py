"""`stump`: the matrix profile, computed on the Metal GPU via batched MASS."""

from __future__ import annotations

import warnings

import mlx.core as mx
import numpy as np

from ._engine import (
    MassEngine,
    default_chunk_size,
    make_reduce_step,
    query_windows,
    tiled_chunk_size,
)
from ._mparray import mparray
from ._preprocess import (
    EXCL_ZONE_DENOM,
    PreprocessedSeries,
    check_series,
    check_window_size,
    preprocess_series,
)

_INF = float("inf")
# byte budget for the float64 window copies held live by one refinement chunk
# (two fancy-indexed window blocks plus their centered copies)
_REFINE_MEM_BUDGET = 1 << 28  # ~256 MiB
# matches stumpy.config.STUMPY_P_NORM_THRESHOLD: squared distances below this
# snap to exactly 0.0, so exact-duplicate subsequences report P == 0.0
P_NORM_THRESHOLD = 1e-14


def _refine_chunk_rows(m: int) -> int:
    return max(1, min(1 << 16, _REFINE_MEM_BUDGET // (m * 8 * 4)))


def _refine_znorm(
    query: PreprocessedSeries, target: PreprocessedSeries, I: np.ndarray
) -> np.ndarray:
    """Recompute z-normalized distances at chosen indices in float64.

    The GPU search runs in float32; re-evaluating d(i, I[i]) on the CPU in
    float64 removes the sqrt-cancellation noise from the reported profile
    values at O(l * m) cost. The squared distance is computed as the sum of
    squared differences of the two z-normalized windows (each centered and
    scaled by *exact two-pass* float64 stats): unlike `dot - m*mu_q*mu_t` —
    which cancels catastrophically for near-constant windows at an offset
    even in float64 — and unlike `2m(1-rho)` — whose ~2m*eps noise floor
    keeps exact duplicates from snapping to 0 — a sum of squares has no
    cancellation at all and is relatively accurate down to 0. Stats are
    recomputed here rather than reusing the O(n) cumsum rolling stats: those
    are only ~1e-6-relative after the suspect repair, fine for the float32
    search but visible in a float64-exact reported value.
    """
    m = query.m
    out = np.full(I.shape, np.inf)
    valid = np.nonzero(I >= 0)[0]
    if valid.size == 0:
        return out
    WQ = np.lib.stride_tricks.sliding_window_view(query.Ts, m)
    WT = np.lib.stride_tricks.sliding_window_view(target.Ts, m)
    dmax = 2.0 * np.sqrt(m)  # rho >= -1; sqrt rounding can overshoot 1 ulp
    chunk = _refine_chunk_rows(m)
    for s in range(0, valid.size, chunk):
        qi = valid[s : s + chunk]
        tj = I[qi]
        qc = query.isconstant[qi]
        tc = target.isconstant[tj]
        qw = WQ[qi]  # fancy indexing copies, so everything can be in place
        qw -= qw.mean(axis=1)[:, None]
        sq = np.sqrt(np.einsum("ij,ij->i", qw, qw) / m)
        tw = WT[tj]
        tw -= tw.mean(axis=1)[:, None]
        st = np.sqrt(np.einsum("ij,ij->i", tw, tw) / m)
        sq_inv = np.where((sq > 0.0) & ~qc, 1.0 / np.where(sq > 0.0, sq, 1.0), 0.0)
        st_inv = np.where((st > 0.0) & ~tc, 1.0 / np.where(st > 0.0, st, 1.0), 0.0)
        qw *= sq_inv[:, None]
        tw *= st_inv[:, None]
        qw -= tw
        d2 = np.einsum("ij,ij->i", qw, qw)
        # a zero sig_inv on either side (constant flag, or a truly flat
        # window flagged non-constant) means rho == 0 on the GPU; mirror
        # that, and let the constant-flag rules below overwrite as needed
        d2 = np.where((sq_inv == 0.0) | (st_inv == 0.0), 2.0 * m, d2)
        d2[d2 < P_NORM_THRESHOLD] = 0.0
        d = np.minimum(np.sqrt(d2), dmax)
        d = np.where(qc & tc, 0.0, np.where(qc ^ tc, np.sqrt(m), d))
        d[~(query.isfinite[qi] & target.isfinite[tj])] = np.inf
        out[qi] = d
    return out


def _refine_absolute(
    query: PreprocessedSeries, target: PreprocessedSeries, I: np.ndarray
) -> np.ndarray:
    """Recompute non-normalized (p=2) distances at chosen indices in float64."""
    m = query.m
    out = np.full(I.shape, np.inf)
    valid = np.nonzero(I >= 0)[0]
    if valid.size == 0:
        return out
    WQ = np.lib.stride_tricks.sliding_window_view(query.T, m)
    WT = np.lib.stride_tricks.sliding_window_view(target.T, m)
    chunk = _refine_chunk_rows(m)
    for s in range(0, valid.size, chunk):
        qi = valid[s : s + chunk]
        tj = I[qi]
        diff = WQ[qi] - WT[tj]
        d2 = np.einsum("ij,ij->i", diff, diff)
        d2[d2 < P_NORM_THRESHOLD] = 0.0
        d = np.sqrt(d2)
        d[~(query.isfinite[qi] & target.isfinite[tj])] = np.inf
        out[qi] = d
    return out


def _topk_chunk(D: mx.array, k: int, l: int):
    """Per-row k smallest distances (ascending) and their column indices."""
    kk = min(k, l)
    part = mx.argpartition(D, kth=kk - 1, axis=1)[:, :kk]
    vals = mx.take_along_axis(D, part, axis=1)
    order = mx.argsort(vals, axis=1)
    vals = mx.take_along_axis(vals, order, axis=1)
    idxs = mx.take_along_axis(part, order, axis=1)
    return vals, idxs


def _merge_topk(run_vals, run_idxs, blk_vals, blk_idxs, k):
    """Row-wise merge of two ascending top-k sets; earlier (lower-index)
    candidates win ties because the running set sorts first under stable
    argsort and blocks arrive in ascending column order."""
    allv = np.concatenate([run_vals, blk_vals], axis=1)
    alli = np.concatenate([run_idxs, blk_idxs], axis=1)
    order = np.argsort(allv, axis=1, kind="stable")[:, :k]
    return np.take_along_axis(allv, order, axis=1), np.take_along_axis(alli, order, axis=1)


def _compute_profile_tiled(
    query: PreprocessedSeries,
    engine: MassEngine,
    *,
    self_join: bool,
    normalize: bool,
    k: int,
    chunk_size: int | None,
):
    """Chunked sweep for targets too large to materialize in one piece.

    The target window matrix is streamed as column blocks (each block
    doubly-centered exactly like the single-block path), and per-row minima /
    top-k sets are merged across blocks on the CPU. Blocks arrive in
    ascending column order and merges use strict ``<``, so on exact ties the
    lowest column index wins — the same first-minimum semantics as a full-row
    argmin.
    """
    m = query.m
    l_q = query.l
    excl = int(np.ceil(m / EXCL_ZONE_DENOM))
    B = chunk_size or tiled_chunk_size(engine, l_q, k, self_join)

    IL = np.full(l_q, -1, dtype=np.int64)
    IR = np.full(l_q, -1, dtype=np.int64)
    if self_join:
        rPl2 = np.full(l_q, np.inf, dtype=np.float32)
        rIl = np.full(l_q, -1, dtype=np.int64)
        rPr2 = np.full(l_q, np.inf, dtype=np.float32)
        rIr = np.full(l_q, -1, dtype=np.int64)
    else:
        rP2 = np.full(l_q, np.inf, dtype=np.float32)
        rI = np.full(l_q, -1, dtype=np.int64)
    if k > 1:
        rPk2 = np.full((l_q, k), np.inf, dtype=np.float32)
        rIk = np.full((l_q, k), -1, dtype=np.int64)

    for j0, j1, W in engine.target_blocks():
        j_row = mx.arange(j0, j1)[None, :]
        for s in range(0, l_q, B):
            e = min(s + B, l_q)
            Q = query_windows(query, s, e)
            QT = mx.matmul(Q, W)
            if normalize:
                D2 = engine.znorm_sq_distances(
                    QT,
                    query.sig_inv_mx[s:e],
                    query.isconstant_mx[s:e],
                    query.isfinite_mx[s:e],
                    j0,
                    j1,
                )
            else:
                D2 = engine.absolute_sq_distances(
                    QT, query.ssq_mx[s:e], query.mu_mx[s:e], query.isfinite_mx[s:e], j0, j1
                )
            i_col = mx.arange(s, e)[:, None]

            outs = []
            if self_join:
                dl = mx.where(j_row <= i_col - (excl + 1), D2, _INF)
                pl2 = mx.min(dl, axis=1)
                il = mx.argmin(dl, axis=1)
                dr = mx.where(j_row >= i_col + (excl + 1), D2, _INF)
                pr2 = mx.min(dr, axis=1)
                ir = mx.argmin(dr, axis=1)
                outs += [pl2, il, pr2, ir]
            else:
                p2 = mx.min(D2, axis=1)
                ii = mx.argmin(D2, axis=1)
                outs += [p2, ii]
            if k > 1:
                Dm = mx.where(mx.abs(i_col - j_row) <= excl, _INF, D2) if self_join else D2
                vals2, idxs = _topk_chunk(Dm, k, j1 - j0)
                outs += [vals2, idxs]
            mx.eval(*outs)

            if self_join:
                pl2 = np.array(outs[0])
                il = np.array(outs[1], dtype=np.int64) + j0
                upd = pl2 < rPl2[s:e]
                rPl2[s:e][upd] = pl2[upd]
                rIl[s:e][upd] = il[upd]
                pr2 = np.array(outs[2])
                ir = np.array(outs[3], dtype=np.int64) + j0
                upd = pr2 < rPr2[s:e]
                rPr2[s:e][upd] = pr2[upd]
                rIr[s:e][upd] = ir[upd]
            else:
                p2 = np.array(outs[0])
                ii = np.array(outs[1], dtype=np.int64) + j0
                upd = p2 < rP2[s:e]
                rP2[s:e][upd] = p2[upd]
                rI[s:e][upd] = ii[upd]
            if k > 1:
                v = np.array(outs[-2])
                ix = np.array(outs[-1], dtype=np.int64) + j0
                ix[~np.isfinite(v)] = -1
                if v.shape[1] < k:
                    pad = k - v.shape[1]
                    v = np.pad(v, ((0, 0), (0, pad)), constant_values=np.inf)
                    ix = np.pad(ix, ((0, 0), (0, pad)), constant_values=-1)
                rPk2[s:e], rIk[s:e] = _merge_topk(rPk2[s:e], rIk[s:e], v, ix, k)

    if self_join:
        pl2 = rPl2.astype(np.float64)
        IL = np.where(np.isfinite(pl2), rIl, -1)
        pr2 = rPr2.astype(np.float64)
        IR = np.where(np.isfinite(pr2), rIr, -1)
        # combined left/right minimum IS the global minimum; ties go left
        left_better = pl2 <= pr2
        p2_min = np.where(left_better, pl2, pr2)
        I_min = np.where(left_better, rIl, rIr)
    else:
        p2_min = rP2.astype(np.float64)
        I_min = rI

    P = np.empty((l_q, k), dtype=np.float64)
    I = np.empty((l_q, k), dtype=np.int64)
    if k == 1:
        P[:, 0] = np.sqrt(p2_min)
        I[:, 0] = np.where(np.isfinite(p2_min), I_min, -1)
    else:
        P[:, :] = np.sqrt(rPk2.astype(np.float64))
        I[:, :] = np.where(np.isfinite(rPk2), rIk, -1)
    return P, I, IL, IR


def _compute_profile(
    query: PreprocessedSeries,
    engine: MassEngine,
    *,
    self_join: bool,
    normalize: bool,
    k: int,
    chunk_size: int | None,
):
    """Chunked GPU sweep: returns (P (l,k) f64-in-f32, I, IL, IR) numpy arrays."""
    if engine.tiled:
        return _compute_profile_tiled(
            query,
            engine,
            self_join=self_join,
            normalize=normalize,
            k=k,
            chunk_size=chunk_size,
        )
    m = query.m
    l_q = query.l
    l_t = engine.l
    excl = int(np.ceil(m / EXCL_ZONE_DENOM))
    B = chunk_size or default_chunk_size(engine, l_q, k, self_join)

    P = np.empty((l_q, k), dtype=np.float64)
    I = np.empty((l_q, k), dtype=np.int64)
    IL = np.full(l_q, -1, dtype=np.int64)
    IR = np.full(l_q, -1, dtype=np.int64)

    if k == 1:
        # fused compiled path: squared distances + argmin in one graph
        step = make_reduce_step(engine, normalize=normalize, self_join=self_join, excl=excl)
        for s in range(0, l_q, B):
            e = min(s + B, l_q)
            Q = query_windows(query, s, e)
            QT = engine.sliding_dot_products(Q)
            if normalize:
                a, b = query.sig_inv_mx[s:e], query.isconstant_mx[s:e]
            else:
                a, b = query.ssq_mx[s:e], query.mu_mx[s:e]
            outs = step(
                QT,
                a,
                b,
                query.isfinite_mx[s:e],
                mx.arange(s, e)[:, None],
            )
            mx.eval(*outs)
            p2 = np.array(outs[1], dtype=np.float64)
            P[s:e, 0] = np.sqrt(p2)
            I[s:e, 0] = np.where(np.isfinite(p2), np.array(outs[0], dtype=np.int64), -1)
            if self_join:
                pl2 = np.array(outs[3], dtype=np.float64)
                IL[s:e] = np.where(np.isfinite(pl2), np.array(outs[2], dtype=np.int64), -1)
                pr2 = np.array(outs[5], dtype=np.float64)
                IR[s:e] = np.where(np.isfinite(pr2), np.array(outs[4], dtype=np.int64), -1)
        return P, I, IL, IR

    j_row = mx.arange(l_t)[None, :]
    for s in range(0, l_q, B):
        e = min(s + B, l_q)
        Q = query_windows(query, s, e)
        QT = engine.sliding_dot_products(Q)
        if normalize:
            D2 = engine.znorm_sq_distances(
                QT,
                query.sig_inv_mx[s:e],
                query.isconstant_mx[s:e],
                query.isfinite_mx[s:e],
            )
        else:
            D2 = engine.absolute_sq_distances(
                QT, query.ssq_mx[s:e], query.mu_mx[s:e], query.isfinite_mx[s:e]
            )

        i_col = mx.arange(s, e)[:, None]
        if self_join:
            D2 = mx.where(mx.abs(i_col - j_row) <= excl, _INF, D2)
            Dl = mx.where(j_row <= i_col - (excl + 1), D2, _INF)
            Pl2 = mx.min(Dl, axis=1)
            Il = mx.argmin(Dl, axis=1)
            Dr = mx.where(j_row >= i_col + (excl + 1), D2, _INF)
            Pr2 = mx.min(Dr, axis=1)
            Ir = mx.argmin(Dr, axis=1)

        vals2, idxs = _topk_chunk(D2, k, l_t)
        outs = [vals2, idxs] + ([Pl2, Il, Pr2, Ir] if self_join else [])
        mx.eval(*outs)

        v = np.sqrt(np.array(vals2, dtype=np.float64))
        ix = np.array(idxs, dtype=np.int64)
        ix[~np.isfinite(v)] = -1
        kk = v.shape[1]
        P[s:e, :kk] = v
        I[s:e, :kk] = ix
        if kk < k:
            P[s:e, kk:] = np.inf
            I[s:e, kk:] = -1
        if self_join:
            pl2 = np.array(Pl2, dtype=np.float64)
            IL[s:e] = np.where(np.isfinite(pl2), np.array(Il, dtype=np.int64), -1)
            pr2 = np.array(Pr2, dtype=np.float64)
            IR[s:e] = np.where(np.isfinite(pr2), np.array(Ir, dtype=np.int64), -1)

    return P, I, IL, IR


def stump(
    T_A,
    m,
    T_B=None,
    ignore_trivial=True,
    normalize=True,
    p=2.0,
    k=1,
    T_A_subseq_isconstant=None,
    T_B_subseq_isconstant=None,
    *,
    chunk_size=None,
):
    """Compute the (top-k) matrix profile of ``T_A`` (optionally joined to ``T_B``).

    Drop-in for ``stumpy.stump``: identical signature and output layout —
    an object array whose columns are the profile values, profile indices,
    left indices, and right indices (``mparray`` with ``P_``, ``I_``,
    ``left_I_``, ``right_I_`` accessors). AB-joins return -1 left/right
    indices, exactly like STUMPY.

    ``normalize=False`` computes the non-normalized (aamp-style) profile and
    supports ``p=2.0`` only. ``chunk_size`` is the number of distance
    profiles evaluated per GPU batch; when omitted it is chosen so the live
    per-batch intermediates stay under a ~384 MiB budget.
    """
    T_A = check_series(T_A, "T_A")
    if not (isinstance(k, (int, np.integer)) and k >= 1):
        raise ValueError(f"`k` must be a positive integer but found {k}.")
    k = int(k)
    if chunk_size is not None and not (
        isinstance(chunk_size, (int, np.integer)) and chunk_size >= 1
    ):
        raise ValueError(f"`chunk_size` must be a positive integer but found {chunk_size}.")

    # join disambiguation, replicating STUMPY's warnings exactly: a T_B equal
    # to T_A with ignore_trivial=False stays an AB-join (warn only)
    share_b_prep = False
    if T_B is None:
        if not ignore_trivial:
            warnings.warn(
                "`ignore_trivial` cannot be `False` for a self-join and "
                "has been automatically overridden and set to `True`.",
                stacklevel=2,
            )
        T_B = T_A
        ignore_trivial = True
        T_B_subseq_isconstant = T_A_subseq_isconstant
        share_b_prep = True
    else:
        T_B = check_series(T_B, "T_B")
        equal = np.array_equal(T_A, T_B, equal_nan=True)
        if not ignore_trivial and equal:
            warnings.warn(
                "Arrays T_A, T_B are equal, which implies a self-join. "
                "Try setting `ignore_trivial = True`.",
                stacklevel=2,
            )
        if ignore_trivial and not equal:
            warnings.warn(
                "Arrays T_A, T_B are not equal, which implies an AB-join. "
                "`ignore_trivial` has been automatically set to `False`.",
                stacklevel=2,
            )
            ignore_trivial = False
    self_join = ignore_trivial

    m = check_window_size(
        m,
        min(T_A.shape[0], T_B.shape[0]),
        warn_n=T_A.shape[0] if self_join else None,
    )

    if not normalize and p != 2.0:
        raise NotImplementedError(
            "mlx-stump supports p=2.0 only when normalize=False; "
            f"found p={p}. Use stumpy.aamp for other p-norms."
        )

    if normalize:
        A = preprocess_series(
            T_A,
            m,
            isconstant=T_A_subseq_isconstant,
            isconstant_name="T_A_subseq_isconstant",
        )
        if share_b_prep:
            Bs = A
        else:
            Bs = preprocess_series(
                T_B,
                m,
                isconstant=T_B_subseq_isconstant,
                isconstant_name="T_B_subseq_isconstant",
            )
    else:
        # shared affine frame keeps cross distances exactly invariant
        finite = np.concatenate([T_A[np.isfinite(T_A)], T_B[np.isfinite(T_B)]])
        center = float(finite.mean()) if finite.size else 0.0
        scale = float(finite.std()) if finite.size else 1.0
        if not np.isfinite(scale) or scale == 0.0:
            scale = 1.0
        A = preprocess_series(T_A, m, normalize=False, center=center, scale=scale)
        Bs = (
            A
            if share_b_prep
            else preprocess_series(T_B, m, normalize=False, center=center, scale=scale)
        )

    engine = MassEngine(Bs, normalize=normalize)
    P32, I, IL, IR = _compute_profile(
        A, engine, self_join=self_join, normalize=normalize, k=k, chunk_size=chunk_size
    )

    # float64 re-evaluation of the profile values at the chosen indices
    refine = _refine_znorm if normalize else _refine_absolute
    P = np.empty_like(P32)
    for j in range(k):
        P[:, j] = refine(A, Bs, I[:, j])
    if k > 1:
        # near-ties can reorder under the exact values; keep columns ascending
        order = np.argsort(P, axis=1, kind="stable")
        P = np.take_along_axis(P, order, axis=1)
        I = np.take_along_axis(I, order, axis=1)

    out = np.empty((A.l, 2 * k + 2), dtype=object)
    for j in range(k):
        out[:, j] = P[:, j]
        out[:, k + j] = I[:, j]
    out[:, 2 * k] = IL
    out[:, 2 * k + 1] = IR
    return mparray(out, m, k, EXCL_ZONE_DENOM)
