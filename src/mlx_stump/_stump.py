"""`stump`: the matrix profile, computed via batched MASS on MLX's active device."""

from __future__ import annotations

import os
import threading
import warnings
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
import numpy as np
import numpy.typing as npt

from . import _engine
from ._engine import (
    MassEngine,
    default_chunk_size,
    free_gpu_after_error,
    make_reducer,
    query_windows,
    query_windows_at,
    refine_chunk_rows,
    tiled_chunk_size,
)
from ._mparray import mparray
from ._preprocess import (
    IsConstantSpec,
    PreprocessedSeries,
    center_rows_stable,
    check_series,
    check_window_size,
    excl_zone_denom,
    exclusion_zone,
    finite_center_scale,
    preprocess_series,
    process_isconstant,
    rowwise_l2_inplace,
)

# Matches stumpy.config.STUMPY_P_NORM_THRESHOLD for the normalized profile:
# tiny z-distance residuals come only from normalization roundoff.  It must
# not be applied to raw-unit AAMP distances, where any fixed absolute cutoff
# would make the result depend on the user's choice of units.
P_NORM_THRESHOLD = 1e-14

# The GPU ranks float32 squared distances. Around zero, two distinct raw
# windows can both round to the same score (often exactly zero), so retaining
# only the winning index can discard an exact match. The bound covers input
# rounding and a sequential m-term float32 dot product with generous
# headroom. It is applied only to a row whose kth or side winner is this
# close to zero; ordinary rows do not need a second search.
_ZERO_SCORE_ULPS = 128


def _zero_score_bound(m: int) -> float:
    eps = np.finfo(np.float32).eps
    # The sequential-dot bound becomes uninformative for very long windows.
    # Keep this a near-zero check: a larger GPU error needs a different
    # algorithm, not an all-pairs float64 replay.
    return min(m / 8.0, _ZERO_SCORE_ULPS * eps * m * (m + 2))


def _raw_score_factor(m: int) -> float:
    # Raw centered-covariance cancellation is proportional to the *pair's*
    # energy, not m alone. The cap keeps the test local to near-duplicates.
    return min(1.0 / 8.0, _ZERO_SCORE_ULPS * np.finfo(np.float32).eps * (m + 2))


def _ambiguous_scores(
    query: PreprocessedSeries,
    target: PreprocessedSeries,
    rows: np.ndarray,
    indices: np.ndarray,
    scores: np.ndarray,
    normalize: bool,
) -> np.ndarray:
    if normalize:
        return scores <= _zero_score_bound(query.m)
    valid = indices >= 0
    safe = np.maximum(indices, 0)
    energy = query.ssq[rows] + target.ssq[safe]
    return valid & (scores <= _raw_score_factor(query.m) * energy)


# Upper bound on the threads of one refinement call. Every refinement step
# is row-wise and NumPy releases the GIL in the gather, ufunc and reduction
# loops that dominate, so row chunks run in parallel with bit-identical
# results. The threads split one serial chunk between them, so the float64
# windows live at once stay within the same _REFINE_MEM_BUDGET.
_REFINE_MAX_WORKERS = 8


def _cpu_count() -> int:
    """CPUs this process may use: ``os.process_cpu_count`` (Python 3.13+,
    which honours CPU affinity and ``PYTHON_CPU_COUNT``) where available."""
    return getattr(os, "process_cpu_count", os.cpu_count)() or 1


def _map_refine_chunks(job, l: int, m: int) -> None:
    """Run ``job(s, e)`` over row chunks ``[s, e)`` that cover ``[0, l)``.

    One serial refinement chunk, ``refine_chunk_rows(m)`` rows, is divided
    among up to ``_REFINE_MAX_WORKERS`` threads of a per-call pool, so at
    most that many rows are in flight at once. Chunks write disjoint rows,
    so the result does not depend on the split or the scheduling. Each job
    must enter its own ``np.errstate`` (it is per thread); a single chunk
    runs in the calling thread. A failing job, or an interrupted caller
    (Ctrl-C), stops every thread after the chunk it holds; the pool is
    joined before this returns or raises.
    """
    rows = refine_chunk_rows(m)
    workers = max(1, min(_REFINE_MAX_WORKERS, _cpu_count(), rows))
    step = rows // workers
    starts = range(0, l, step)
    lanes = min(workers, len(starts))
    if lanes <= 1:
        for s in starts:
            job(s, min(s + step, l))
        return
    # Each lane pulls the next chunk from one shared cursor, so the pool
    # holds one task per thread rather than one per chunk (at large m there
    # are ~l*m/2**20 chunks) while a slow thread still takes fewer chunks.
    cursor = iter(starts)
    lock = threading.Lock()
    failed = threading.Event()

    def lane():
        while not failed.is_set():
            with lock:
                s = next(cursor, None)
            if s is None:
                return
            try:
                job(s, min(s + step, l))
            except BaseException:
                failed.set()  # the other lanes stop at their next chunk
                raise

    with ThreadPoolExecutor(max_workers=lanes) as pool:
        try:
            tasks = [pool.submit(lane) for _ in range(lanes)]
            for task in tasks:
                task.result()  # re-raises a worker exception here
        except BaseException:
            # a worker failed, or the caller was interrupted (Ctrl-C) while
            # submitting or waiting: stop every lane at its next chunk before
            # the pool joins, instead of refining to the end in the background
            failed.set()
            raise


def _znorm_rows(w: np.ndarray, isconstant: np.ndarray, m: int) -> np.ndarray:
    """Z-normalize gathered float64 windows in place; return their 1/RMS.

    The inverse is 0 for flagged-constant rows and for rows that are flat
    after centering.
    """
    center_rows_stable(w)
    rms = np.sqrt(np.sum(w * w, axis=1) / m)
    inv = np.where((rms > 0.0) & ~isconstant, 1.0 / np.where(rms > 0.0, rms, 1.0), 0.0)
    w *= inv[:, None]
    return inv


def _refine_znorm(
    query: PreprocessedSeries,
    target: PreprocessedSeries,
    I: np.ndarray,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Recompute z-normalized distances at chosen indices in float64.

    ``I`` holds the neighbor indices, shape ``(l,)`` or ``(l, k)``, with -1
    for "no neighbor" (reported as ``inf``). The distances are written into
    ``out`` (same shape, float64; allocated when omitted), which is returned.

    The GPU search runs in float32; re-evaluating d(i, I[i, j]) on the CPU in
    float64 removes the sqrt-cancellation noise from the reported profile
    values at O(l * k * m) cost, a sizeable share of the runtime for large
    ``k``. Each row chunk normalizes its query windows once and reuses them
    for all ``k`` neighbor columns, and the chunks run on a few threads
    (:func:`_map_refine_chunks`). The squared distance is computed as the sum of
    squared differences of the two z-normalized windows (each centered and
    scaled by freshly recomputed two-pass float64 stats): unlike `dot - m*mu_q*mu_t` —
    which cancels catastrophically for near-constant windows at an offset
    even in float64 — and unlike `2m(1-rho)` — whose ~2m*eps noise floor
    keeps exact duplicates from snapping to 0 — a sum of squares avoids
    those cancellation-dominated formulas and remains accurate near 0. The
    local stats are recomputed here rather than derived from the float32 search
    windows: their cast is sufficient for ranking but visible in a directly
    recomputed reported value. The windows come from the RAW series
    (z-normalized distance is affine-invariant): the
    standardized copy quietly re-rounds every value by eps64 * scale, which
    a window whose own sigma is far below the global scale cannot afford.
    Each window is first mapped into its own midpoint/max-deviation frame and
    only then mean-centered. This keeps every value bounded before squaring,
    so neither a large common offset nor uniformly tiny/huge units can poison
    its statistics through cancellation, underflow, or overflow.
    """
    m = query.m
    if out is None:
        out = np.empty(I.shape)
    I2 = I[:, None] if I.ndim == 1 else I
    out2 = out[:, None] if out.ndim == 1 else out  # a view: writes land in out
    WQ = np.lib.stride_tricks.sliding_window_view(query.T, m)
    WT = np.lib.stride_tricks.sliding_window_view(target.T, m)
    dmax = 2.0 * np.sqrt(m)  # rho >= -1; sqrt rounding can overshoot 1 ulp

    def job(s, e):
        Ic = I2[s:e]
        out2[s:e] = np.inf
        rows = np.nonzero((Ic >= 0).any(axis=1))[0]
        if rows.size == 0:
            return
        qi_all = rows + s
        qc_all = query.isconstant[qi_all]
        # raw windows may hold NaN/inf; those rows are overwritten with inf
        # below, so let the intermediate arithmetic run silently
        with np.errstate(invalid="ignore", over="ignore", under="ignore", divide="ignore"):
            qw_all = WQ[qi_all]  # fancy indexing copies, so everything can be in place
            sq_inv_all = _znorm_rows(qw_all, qc_all, m)
            for j in range(Ic.shape[1]):
                tj = Ic[rows, j]
                has = tj >= 0
                if not has.any():
                    continue
                # usually every row has this neighbor; a full slice then
                # selects by view, so the query block is not copied per column
                sel = slice(None) if has.all() else has
                tj = tj[sel]
                qi = qi_all[sel]
                qc = qc_all[sel]
                tc = target.isconstant[tj]
                tw = WT[tj]
                st_inv = _znorm_rows(tw, tc, m)
                sq_inv = sq_inv_all[sel]
                # the target copy doubles as the difference buffer
                np.subtract(qw_all[sel], tw, out=tw)
                d2 = np.einsum("ij,ij->i", tw, tw)
                del tw
                # a zero sig_inv on either side (constant flag, or a truly flat
                # window flagged non-constant) means rho == 0 on the GPU; mirror
                # that, and let the constant-flag rules below overwrite as needed
                d2 = np.where((sq_inv == 0.0) | (st_inv == 0.0), 2.0 * m, d2)
                d2[~np.isfinite(d2)] = 2.0 * m  # NaN/inf windows: masked below
                d2[d2 < P_NORM_THRESHOLD] = 0.0
                d = np.minimum(np.sqrt(d2), dmax)
                d = np.where(qc & tc, 0.0, np.where(qc ^ tc, np.sqrt(m), d))
                d[~(query.isfinite[qi] & target.isfinite[tj])] = np.inf
                out2[qi, j] = d

    _map_refine_chunks(job, I2.shape[0], m)
    return out


def _refine_absolute(
    query: PreprocessedSeries,
    target: PreprocessedSeries,
    I: np.ndarray,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Recompute non-normalized (p=2) distances at chosen indices in float64.

    ``I``, ``out`` and the chunking are as in :func:`_refine_znorm`. There is
    no query normalization to share between columns here. Keeping the query
    block across columns would hold a third window copy next to the
    difference and the norm's scratch, so each column gathers its own.
    """
    m = query.m
    if out is None:
        out = np.empty(I.shape)
    I2 = I[:, None] if I.ndim == 1 else I
    out2 = out[:, None] if out.ndim == 1 else out
    WQ = np.lib.stride_tricks.sliding_window_view(query.T, m)
    WT = np.lib.stride_tricks.sliding_window_view(target.T, m)

    def job(s, e):
        out2[s:e] = np.inf
        for j in range(I2.shape[1]):
            qi = np.nonzero(I2[s:e, j] >= 0)[0] + s
            if qi.size == 0:
                continue
            tj = I2[qi, j]
            diff = WQ[qi]  # fancy indexing copies, so the difference can be in place
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                np.subtract(diff, WT[tj], out=diff)
            d = rowwise_l2_inplace(diff)
            del diff  # not alive while the next column gathers
            d[~(query.isfinite[qi] & target.isfinite[tj])] = np.inf
            out2[qi, j] = d

    _map_refine_chunks(job, I2.shape[0], m)
    return out


def _batches(l_q: int, B: int):
    """Yield ``(s0, s, e)``: rows ``[s, e)`` are the batch's output rows, computed
    over ``[s0, e)`` with ``s0 <= s`` so that every batch has the same width
    ``B`` (whenever ``l_q >= B``). A narrower trailing batch would allocate a
    second, differently sized set of intermediates that MLX's buffer cache
    cannot reuse (up to another full budget retained after the call), and at
    width 1 would dispatch a GEMV-shaped kernel whose float32 accumulation
    order differs from the batched GEMM. Recomputing ``s - s0`` rows once
    costs less than one batch. The bounds are Python ints (the device calls
    reject NumPy integers) and the generator is lazy: a materialized list
    costs ~170 B per batch, 160 MiB at ``l_q = 10**6`` with ``chunk_size=1``."""
    l_q, B = int(l_q), int(B)
    for s in range(0, l_q, B):
        e = min(s + B, l_q)
        yield max(0, e - B), s, e


def _tie_keys(idxs, rows, self_join: bool):
    """The sweep's exact-tie key for global columns ``idxs`` of query
    ``rows``: ``2*|j - i| + (j > i)`` for self-joins (nearest in time, then
    left, as STUMPY's diagonal traversal), the column itself for AB-joins."""
    if not self_join:
        return idxs
    off = idxs - rows[:, None]
    key = np.abs(off)
    key *= 2
    key += off > 0
    return key


def _merge_topk(run_vals, run_idxs, blk_vals, blk_idxs, k, rows=None):
    """Row-wise merge of two top-k sets, each ascending in ``(value, key)``,
    keeping the k smallest in that lexicographic order. ``rows`` (the global
    query row of each line) selects the self-join key; without it the key is
    the column (AB-joins). Entries with an infinite value carry index -1 and
    order arbitrarily among themselves; they are reported as (inf, -1)."""
    allv = np.concatenate([run_vals, blk_vals], axis=1)
    alli = np.concatenate([run_idxs, blk_idxs], axis=1)
    keys = _tie_keys(alli, rows, rows is not None)
    order = np.lexsort((keys, allv), axis=1)[:, :k]
    del keys
    return np.take_along_axis(allv, order, axis=1), np.take_along_axis(alli, order, axis=1)


def _query_args(batches, query, normalize):
    """Yield each batch's float32 query windows, built one batch ahead: the
    caller asks for batch i+1 while batch i runs on the GPU. ``batches`` is
    a second ``_batches`` generator running one step ahead of the sweep's."""
    for s0, _, e in batches:
        yield query_windows(query, s0, e, normalize=normalize)


def _compute_profile_tiled(
    query: PreprocessedSeries,
    engine: MassEngine,
    *,
    self_join: bool,
    normalize: bool,
    k: int,
    chunk_size: int | None,
    excl: int,
    uncertain_out: np.ndarray | None = None,
):
    """Chunked sweep for targets too large to materialize in one piece.

    The target window matrix is streamed as column blocks (each block
    doubly-centered exactly like the single-block path), and per-row minima /
    top-k sets are merged across blocks on the CPU in the sweep's
    lexicographic ``(d2, key)`` order (see ``_engine.ReduceStep``). Blocks
    arrive in ascending column order: on the right (and in AB-joins) an
    earlier block's equal minimum is the nearer / lower column and a strict
    ``<`` keeps it, while on the left a later block's is nearer and ``<=``
    takes it. The result is identical to the dense sweep's.
    """
    l_q = query.l
    fused = _engine._fused_reducer(k)
    B = chunk_size or tiled_chunk_size(engine, l_q, k, self_join, fused=fused)

    IL = np.full(l_q, -1, dtype=np.int64)
    IR = np.full(l_q, -1, dtype=np.int64)
    if self_join:
        rPl2 = np.full(l_q, np.inf, dtype=np.float32)
        rIl = np.full(l_q, -1, dtype=np.int64)
        rPr2 = np.full(l_q, np.inf, dtype=np.float32)
        rIr = np.full(l_q, -1, dtype=np.int64)
    elif k == 1:
        rP2 = np.full(l_q, np.inf, dtype=np.float32)
        rI = np.full(l_q, -1, dtype=np.int64)
    if k > 1:
        rPk2 = np.full((l_q, k), np.inf, dtype=np.float32)
        rIk = np.full((l_q, k), -1, dtype=np.int64)

    red = make_reducer(
        query, engine, normalize=normalize, self_join=self_join, excl=excl, k=k, fused=fused
    )
    for j0, j1, W in engine.target_blocks():
        Qs = _query_args(_batches(l_q, B), query, normalize)
        Q = next(Qs)
        for s0, s, e in _batches(l_q, B):
            off = s - s0
            outs = red.block(mx.matmul(Q, W), s0, j0, j1)
            mx.async_eval(*outs)
            # build the next batch's windows on the CPU while this one runs
            Q = next(Qs, None)
            mx.eval(*outs)
            # wait for the command buffer's completion handler as well: it is
            # what returns this batch's buffers to the allocator, and without
            # it the next batch can allocate a second set (2x the budget)
            mx.synchronize()

            if self_join:
                # k == 1: (I, P2, Il, Pl2, Ir, Pr2); k > 1: (vals2, idxs, Il, Pl2, Ir, Pr2)
                pl2 = np.array(outs[3])[off:]
                il = np.array(outs[2], dtype=np.int64)[off:] + j0
                upd = pl2 <= rPl2[s:e]  # a later block's equal left minimum is nearer
                rPl2[s:e][upd] = pl2[upd]
                rIl[s:e][upd] = il[upd]
                pr2 = np.array(outs[5])[off:]
                ir = np.array(outs[4], dtype=np.int64)[off:] + j0
                upd = pr2 < rPr2[s:e]
                rPr2[s:e][upd] = pr2[upd]
                rIr[s:e][upd] = ir[upd]
            elif k == 1:
                p2 = np.array(outs[1])[off:]
                ii = np.array(outs[0], dtype=np.int64)[off:] + j0
                upd = p2 < rP2[s:e]
                rP2[s:e][upd] = p2[upd]
                rI[s:e][upd] = ii[upd]
            if k > 1:
                v = np.array(outs[0])[off:]
                ix = np.array(outs[1], dtype=np.int64)[off:] + j0
                ix[~np.isfinite(v)] = -1
                if v.shape[1] < k:
                    pad = k - v.shape[1]
                    v = np.pad(v, ((0, 0), (0, pad)), constant_values=np.inf)
                    ix = np.pad(ix, ((0, 0), (0, pad)), constant_values=-1)
                rows = np.arange(s, e) if self_join else None
                rPk2[s:e], rIk[s:e] = _merge_topk(rPk2[s:e], rIk[s:e], v, ix, k, rows)
            del outs
        del W  # release this block before the generator builds the next one

    # Only the neighbours leave the sweep: stump() recomputes every reported
    # value in float64, so the float32 minima serve only to mark rows without
    # a finite candidate (-1) and, for k == 1, to combine the two sides.
    # (Comparing and testing the float32 values is exact: no float64 copy.)
    if self_join:
        IL = np.where(np.isfinite(rPl2), rIl, -1)
        IR = np.where(np.isfinite(rPr2), rIr, -1)
        if k == 1:
            # the combined left/right minimum IS the global one; exact ties
            # go to the nearer side, and to the left on an equal offset
            rows = np.arange(l_q)
            left_better = (rPl2 < rPr2) | ((rPl2 == rPr2) & ((rows - rIl) <= (rIr - rows)))
            rP2 = np.where(left_better, rPl2, rPr2)
            rI = np.where(left_better, rIl, rIr)
            del rows, left_better
    rows = np.arange(l_q)
    boundary = rP2 if k == 1 else rPk2[:, -1]
    boundary_i = rI if k == 1 else rIk[:, -1]
    uncertain = _ambiguous_scores(
        query, engine.target, rows, boundary_i, boundary, normalize
    )
    if self_join:
        uncertain |= _ambiguous_scores(
            query, engine.target, rows, rIl, rPl2, normalize
        ) | _ambiguous_scores(query, engine.target, rows, rIr, rPr2, normalize)
    if uncertain_out is not None:
        uncertain_out[:] = uncertain
    if k == 1:
        rI[~np.isfinite(rP2)] = -1
        return rI.reshape(l_q, 1), IL, IR
    rIk[~np.isfinite(rPk2)] = -1
    return rIk, IL, IR


def _compute_profile(
    query: PreprocessedSeries,
    engine: MassEngine,
    *,
    self_join: bool,
    normalize: bool,
    k: int,
    chunk_size: int | None,
    excl: int,
    uncertain_out: np.ndarray | None = None,
):
    """Chunked GPU sweep: returns the neighbour indices ``(I (l, k), IL, IR)``.

    Rows without a finite candidate get -1. No profile values are returned:
    :func:`stump` recomputes every one of them in float64 at these indices.
    ``excl`` is the self-join exclusion-zone half width, ``ceil(m / denom)``.
    The reduction is the fused Metal kernel or the compiled fallback, per
    ``_engine._fused_reducer(k)``, and the automatic batch is sized for the
    one that runs. Exact ties follow STUMPY (see ``_engine.ReduceStep``).
    """
    if engine.tiled:
        return _compute_profile_tiled(
            query,
            engine,
            self_join=self_join,
            normalize=normalize,
            k=k,
            chunk_size=chunk_size,
            excl=excl,
            uncertain_out=uncertain_out,
        )
    l_q = query.l
    fused = _engine._fused_reducer(k)
    B = chunk_size or default_chunk_size(engine, l_q, k, self_join, fused=fused)

    I = np.empty((l_q, k), dtype=np.int64)
    IL = np.full(l_q, -1, dtype=np.int64)
    IR = np.full(l_q, -1, dtype=np.int64)
    uncertain = np.zeros(l_q, dtype=bool)

    red = make_reducer(
        query, engine, normalize=normalize, self_join=self_join, excl=excl, k=k, fused=fused
    )
    Qs = _query_args(_batches(l_q, B), query, normalize)
    Q = next(Qs)
    for s0, s, e in _batches(l_q, B):
        off = s - s0
        # QT is passed inline, never bound to a name: a live reference would
        # keep the B*l*4-byte product alive through the eval below
        outs = red.full(mx.matmul(Q, engine.W_T), s0)
        mx.async_eval(*outs)
        # build the next batch's windows on the CPU while this one runs; still
        # exactly one live set of device intermediates
        Q = next(Qs, None)
        mx.eval(*outs)
        mx.synchronize()  # see _compute_profile_tiled: releases this batch's buffers
        # the float32 squared distances only mark rows without a finite
        # candidate; stump() recomputes every reported value in float64
        if k == 1:
            p2 = np.array(outs[1])[off:]
            I[s:e, 0] = np.where(np.isfinite(p2), np.array(outs[0], dtype=np.int64)[off:], -1)
            uncertain[s:e] |= _ambiguous_scores(
                query, engine.target, np.arange(s, e), I[s:e, 0], p2, normalize
            )
        else:
            vals = np.array(outs[0])[off:]
            ix = np.array(outs[1], dtype=np.int64)[off:]
            ix[~np.isfinite(vals)] = -1
            kk = ix.shape[1]
            I[s:e, :kk] = ix
            if kk < k:
                I[s:e, kk:] = -1
            else:
                uncertain[s:e] |= _ambiguous_scores(
                    query, engine.target, np.arange(s, e), I[s:e, -1], vals[:, -1], normalize
                )
        if self_join:
            pl2 = np.array(outs[3])[off:]
            IL[s:e] = np.where(np.isfinite(pl2), np.array(outs[2], dtype=np.int64)[off:], -1)
            pr2 = np.array(outs[5])[off:]
            IR[s:e] = np.where(np.isfinite(pr2), np.array(outs[4], dtype=np.int64)[off:], -1)
            rows = np.arange(s, e)
            uncertain[s:e] |= _ambiguous_scores(
                query, engine.target, rows, IL[s:e], pl2, normalize
            ) | _ambiguous_scores(query, engine.target, rows, IR[s:e], pr2, normalize)
        del outs
    if uncertain_out is not None:
        uncertain_out[:] = uncertain
    return I, IL, IR


def _repair_zero_neighbors(
    query: PreprocessedSeries,
    target: PreprocessedSeries,
    engine: MassEngine,
    I: np.ndarray,
    IL: np.ndarray,
    IR: np.ndarray,
    uncertain: np.ndarray,
    *,
    normalize: bool,
    self_join: bool,
    excl: int,
) -> None:
    """Recover exact matches hidden by float32 near-zero ties.

    The first sweep keeps only k indices. A near copy can have the same
    float32 score as an exact copy and win by the search tie key, so refining
    those k indices alone cannot restore the true minimum. Only rows whose
    kth or side score is inside the zero-score error bound are replayed here.
    Each target block and small query batch is materialized once; candidates
    in the band are refined immediately and reduced to k plus left/right
    winners, keeping host storage bounded even on repeated series.
    """
    rows = np.flatnonzero(uncertain)
    if rows.size == 0:
        return

    # A repeated/constant series can put every row in the zero band. When
    # every reported neighbor is already an exact raw copy (or both windows
    # carry the normalized constant flag), no near copy has displaced it.
    # Avoid replaying an O(l^2) plateau merely to rediscover those zeros.
    Q_windows = np.lib.stride_tricks.sliding_window_view(query.T, query.m)
    T_windows = np.lib.stride_tricks.sliding_window_view(target.T, target.m)

    def selected_exact(row: int, j: int) -> bool:
        return j >= 0 and (
            (normalize and query.isconstant[row] and target.isconstant[j])
            or (
                (not normalize or query.isconstant[row] == target.isconstant[j])
                and np.array_equal(Q_windows[row], T_windows[j])
            )
        )

    def already_exact(row: int) -> bool:
        if not all(selected_exact(row, int(j)) for j in I[row]):
            return False
        if not self_join:
            return True
        if IL[row] >= 0:
            if not selected_exact(row, int(IL[row])):
                return False
        elif row - excl - 1 >= 0:
            return False
        if IR[row] >= 0:
            return selected_exact(row, int(IR[row]))
        return row + excl + 1 >= target.l

    keep = np.fromiter((not already_exact(int(row)) for row in rows), dtype=bool, count=rows.size)
    rows = rows[keep]
    del keep
    if rows.size == 0:
        return

    # The first sweep's large batch buffers are no longer used. Keep its
    # resident window block, but release cached buffers before interleaving
    # bounded GPU rescans with float64 candidate verification.
    mx.clear_cache()
    from ._match import _refine_candidates

    k = I.shape[1]
    bound = _zero_score_bound(query.m) if normalize else None
    raw_factor = _raw_score_factor(query.m) if not normalize else None
    refine_rows = max(1, (8 << 20) // (query.m * 8 * 4))
    # Shifted/scaled copies are also exact normalized matches. Certify the
    # currently selected neighbors before replaying a whole affine plateau;
    # _refine_candidates proves zero without stump's 1e-14 output snap.
    def refine_seeds(row: int) -> tuple[np.ndarray, np.ndarray]:
        seeds = I[row, I[row] >= 0]
        if self_join:
            seeds = np.concatenate((seeds, [IL[row], IR[row]]))
            seeds = seeds[seeds >= 0]
        seeds = np.unique(seeds)
        d = _refine_candidates(
            query.T[row : row + query.m], target.T, seeds, normalize,
            bool(query.isconstant[row]), target.isconstant,
            max_chunk_rows=refine_rows,
        )
        return seeds, d

    keep = np.ones(rows.size, dtype=bool)
    for pos, row in enumerate(rows):
        _, d = refine_seeds(int(row))
        complete = np.all(I[row] >= 0)
        if self_join:
            complete &= (IL[row] >= 0 or row - excl - 1 < 0) and (
                IR[row] >= 0 or row + excl + 1 >= target.l
            )
        if complete and d.size and np.all(d == 0.0):
            keep[pos] = False
    rows = rows[keep]
    del keep
    if rows.size == 0:
        return

    best_i = np.full((rows.size, k), -1, dtype=np.int64)
    best_d = np.full((rows.size, k), np.inf, dtype=np.float64)
    left_i = np.full(rows.size, -1, dtype=np.int64)
    right_i = np.full(rows.size, -1, dtype=np.int64)
    left_d = np.full(rows.size, np.inf, dtype=np.float64)
    right_d = np.full(rows.size, np.inf, dtype=np.float64)

    def keys(row: int, js: np.ndarray) -> np.ndarray:
        if not self_join:
            return js
        return 2 * np.abs(js - row) + (js > row)

    def add(pos: int, js: np.ndarray, distances: np.ndarray) -> None:
        row = int(rows[pos])
        valid = np.isfinite(distances)
        js, distances = js[valid], distances[valid]
        if js.size == 0:
            return
        old = best_i[pos]
        old_valid = old >= 0
        fresh = ~np.isin(js, old[old_valid])
        joined_i = np.concatenate((old[old_valid], js[fresh]))
        joined_d = np.concatenate((best_d[pos, old_valid], distances[fresh]))
        order = np.lexsort((keys(row, joined_i), joined_d))[:k]
        best_i[pos] = -1
        best_d[pos] = np.inf
        best_i[pos, : order.size] = joined_i[order]
        best_d[pos, : order.size] = joined_d[order]
        if self_join:
            for side, mask in (("left", js < row), ("right", js > row)):
                side_js, side_d = js[mask], distances[mask]
                if side_js.size == 0:
                    continue
                winner = np.lexsort((keys(row, side_js), side_d))[0]
                index, distance = int(side_js[winner]), float(side_d[winner])
                current_d = left_d[pos] if side == "left" else right_d[pos]
                current_i = left_i[pos] if side == "left" else right_i[pos]
                if distance < current_d or (
                    distance == current_d
                    and (current_i < 0 or keys(row, np.array([index]))[0]
                         < keys(row, np.array([current_i]))[0])
                ):
                    if side == "left":
                        left_i[pos], left_d[pos] = index, distance
                    else:
                        right_i[pos], right_d[pos] = index, distance

    # Include the original GPU winners. A candidate inside the float32 band
    # is not necessarily truly better than one just outside it.
    for pos, row in enumerate(rows):
        seeds, d = refine_seeds(int(row))
        add(pos, seeds, d)

    for j0, j1, W in engine.target_blocks():
        # Bound both the score matrix and locally normalized query windows.
        # A single very long row can exceed the nominal 8 MiB query budget.
        batch = max(
            1,
            min(
                64,
                (8 << 20) // (4 * (j1 - j0)),
                (8 << 20) // (24 * query.m + 128),
            ),
        )
        for start in range(0, rows.size, batch):
            stop = min(start + batch, rows.size)
            selected = rows[start:stop]
            index = mx.array(selected.astype(np.int32))
            Q = query_windows_at(query, selected, normalize=normalize)
            QT = mx.matmul(Q, W)
            if normalize:
                d2 = engine.znorm_sq_distances(
                    QT,
                    mx.take(query.sig_inv_mx, index, axis=0),
                    mx.take(query.isconstant_mx, index, axis=0),
                    mx.take(query.isfinite_mx, index, axis=0),
                    j0, j1,
                )
            else:
                d2 = engine.absolute_sq_distances(
                    QT,
                    mx.take(query.ssq_mx, index, axis=0),
                    mx.take(query.mu_mx, index, axis=0),
                    mx.take(query.isfinite_mx, index, axis=0),
                    j0, j1,
                )
            mx.eval(d2)
            mx.synchronize()
            scores = np.array(d2)
            del index, Q, QT, d2
            for local, row in enumerate(selected):
                if normalize:
                    candidates = scores[local] <= bound
                else:
                    pair_energy = query.ssq[row] + target.ssq[j0:j1]
                    candidates = scores[local] <= raw_factor * pair_energy
                js = np.flatnonzero(candidates).astype(np.int64) + j0
                if self_join:
                    js = js[np.abs(js - row) > excl]
                if js.size == 0:
                    continue
                d = _refine_candidates(
                    query.T[row : row + query.m], target.T, js, normalize,
                    bool(query.isconstant[row]), target.isconstant,
                    max_chunk_rows=refine_rows,
                )
                add(start + local, js, d)
            del scores
        del W

    I[rows] = best_i
    if self_join:
        IL[rows] = left_i
        IR[rows] = right_i


def stump(
    T_A: npt.ArrayLike,
    m: int,
    T_B: npt.ArrayLike | None = None,
    ignore_trivial: bool = True,
    normalize: bool = True,
    p: float = 2.0,
    k: int = 1,
    T_A_subseq_isconstant: IsConstantSpec = None,
    T_B_subseq_isconstant: IsConstantSpec = None,
    *,
    chunk_size: int | None = None,
) -> mparray:
    """Compute the (top-k) matrix profile of ``T_A`` (optionally joined to ``T_B``).

    Drop-in for ``stumpy.stump``: the same upstream parameters and output
    layout, plus the keyword-only ``chunk_size`` memory/performance control.
    The result is an object array whose columns are the profile values,
    profile indices, left indices, and right indices (``mparray`` with
    ``P_``, ``I_``, ``left_I_``, ``right_I_`` accessors). AB-joins return
    -1 left/right indices, exactly like STUMPY.

    A self-join ignores neighbors within the exclusion zone
    ``ceil(m / stumpy.config.STUMPY_EXCL_ZONE_DENOM)`` (default denominator
    4) of each subsequence. Like STUMPY, the denominator is read at call
    time (whenever STUMPY has been imported). With an explicit ``T_B`` and
    ``ignore_trivial=True``, the join is a self-join only if ``T_B`` equals
    ``T_A``. As in STUMPY, which non-finite marker (NaN, inf or -inf) marks
    a missing sample does not matter. STUMPY's zero fill also equates a
    missing sample with a real 0.0 (or -0.0) at the same position; here
    such a pair makes the series different.

    ``normalize=False`` computes the non-normalized (aamp-style) profile and
    supports ``p=2.0`` only. Its refined profile remains in raw input units,
    including below ``1e-7``; STUMPY's fixed raw-unit P-norm threshold snaps
    those small-but-nonzero distances to zero. ``chunk_size`` is the number
    of distance profiles evaluated per GPU batch; when omitted it is chosen
    so the live per-batch intermediates stay under a ~384 MiB budget. Results
    do not depend on it, except that ``chunk_size=1`` dispatches a
    matrix-vector kernel whose float32 accumulation order differs from the
    batched one and can resolve near-ties to a different, equally close
    neighbor.
    """
    return _stump(
        T_A,
        m,
        T_B=T_B,
        ignore_trivial=ignore_trivial,
        normalize=normalize,
        p=p,
        k=k,
        T_A_subseq_isconstant=T_A_subseq_isconstant,
        T_B_subseq_isconstant=T_B_subseq_isconstant,
        chunk_size=chunk_size,
        stacklevel=3,
    )


def _stump(
    T_A,
    m,
    *,
    T_B,
    ignore_trivial,
    normalize,
    p,
    k,
    T_A_subseq_isconstant,
    T_B_subseq_isconstant,
    chunk_size,
    stacklevel: int,
) -> mparray:
    """The implementation behind :func:`stump` and its STUMPY-named wrappers.

    ``stacklevel`` locates this frame's warnings at the user's call. It is 3
    when a public function calls this directly. The helpers one frame deeper
    get one more.
    """
    T_A = check_series(T_A, "T_A")
    if not (
        isinstance(k, (int, np.integer))
        and not isinstance(k, (bool, np.bool_))
        and k >= 1
    ):
        raise ValueError(f"`k` must be a positive integer but found {k}.")
    k = int(k)
    if chunk_size is not None and not (
        isinstance(chunk_size, (int, np.integer))
        and not isinstance(chunk_size, (bool, np.bool_))
        and chunk_size >= 1
    ):
        raise ValueError(f"`chunk_size` must be a positive integer but found {chunk_size}.")
    if chunk_size is not None:
        # like k: a numpy integer would reach mx.arange (which rejects it),
        # and a small unsigned one would wrap in the batch arithmetic
        chunk_size = int(chunk_size)

    # join disambiguation, replicating STUMPY's warnings exactly: a T_B equal
    # to T_A with ignore_trivial=False stays an AB-join (warn only)
    share_b_prep = False
    if T_B is None:
        if not ignore_trivial:
            warnings.warn(
                "`ignore_trivial` cannot be `False` for a self-join and "
                "has been automatically overridden and set to `True`.",
                stacklevel=stacklevel,
            )
        T_B = T_A
        ignore_trivial = True
        T_B_subseq_isconstant = T_A_subseq_isconstant
        share_b_prep = True
    else:
        T_B = check_series(T_B, "T_B")
        # STUMPY compares the series after its preprocessing has replaced
        # every non-finite value, so the marker of a missing sample (NaN,
        # inf, -inf) does not matter. Its zero fill would also equate a
        # missing sample with a real 0.0; that is not copied: those series
        # differ, and the mirrored rows STUMPY reports for them are wrong.
        equal = T_A.shape == T_B.shape and bool(
            np.all((T_A == T_B) | (~np.isfinite(T_A) & ~np.isfinite(T_B)))
        )
        if not ignore_trivial and equal:
            warnings.warn(
                "Arrays T_A, T_B are equal, which implies a self-join. "
                "Try setting `ignore_trivial = True`.",
                stacklevel=stacklevel,
            )
        if ignore_trivial and not equal:
            warnings.warn(
                "Arrays T_A, T_B are not equal, which implies an AB-join. "
                "`ignore_trivial` has been automatically set to `False`.",
                stacklevel=stacklevel,
            )
            ignore_trivial = False
    self_join = ignore_trivial

    # read once per call, like STUMPY: the zone used for the search, the
    # advisory, and the returned mparray's metadata must all agree
    denom = excl_zone_denom()
    m = check_window_size(
        m,
        min(T_A.shape[0], T_B.shape[0]),
        warn_n=T_A.shape[0] if self_join else None,
        denom=denom,
        stacklevel=stacklevel + 1,
    )

    if not normalize and p != 2.0:
        raise NotImplementedError(
            "mlx-stump supports p=2.0 only when normalize=False; "
            f"found p={p}. Use stumpy.aamp for other p-norms."
        )

    # One handler covers preprocessing too: a failure while preparing T_B
    # (e.g. a bad or raising T_B_subseq_isconstant) must not leave T_A's
    # uploaded window statistics in MLX's cache any more than a failed sweep.
    A = Bs = engine = None
    try:
        if normalize:
            A = preprocess_series(
                T_A,
                m,
                isconstant=T_A_subseq_isconstant,
                isconstant_name="T_A_subseq_isconstant",
                stacklevel=stacklevel + 1,
            )
            if share_b_prep:
                Bs = A
            else:
                Bs = preprocess_series(
                    T_B,
                    m,
                    isconstant=T_B_subseq_isconstant,
                    isconstant_name="T_B_subseq_isconstant",
                    stacklevel=stacklevel + 1,
                )
        else:
            # Constant-window flags do not affect raw Euclidean distances, but
            # validate them consistently with mass/match instead of silently
            # accepting malformed controls in this one API.
            # process_isconstant reads only the length for an array spec and
            # hands a callable its own inf->NaN copy, so no series copy is needed.
            if T_A_subseq_isconstant is not None:
                process_isconstant(T_A, m, T_A_subseq_isconstant, "T_A_subseq_isconstant")
            if not share_b_prep and T_B_subseq_isconstant is not None:
                process_isconstant(T_B, m, T_B_subseq_isconstant, "T_B_subseq_isconstant")
            # shared affine frame keeps cross distances exactly invariant (both
            # series even for a self-join, so the numerics never change), built
            # without compacted-length copies
            center, scale = finite_center_scale(T_A, T_B)
            A = preprocess_series(
                T_A, m, normalize=False, center=center, scale=scale, stacklevel=stacklevel + 1
            )
            Bs = (
                A
                if share_b_prep
                else preprocess_series(
                    T_B, m, normalize=False, center=center, scale=scale, stacklevel=stacklevel + 1
                )
            )

        engine = MassEngine(Bs, normalize=normalize)
        excl = exclusion_zone(m, denom)
        uncertain = np.zeros(A.l, dtype=bool)
        I, IL, IR = _compute_profile(
            A,
            engine,
            self_join=self_join,
            normalize=normalize,
            k=k,
            chunk_size=chunk_size,
            excl=excl,
            uncertain_out=uncertain,
        )
        _repair_zero_neighbors(
            A, Bs, engine, I, IL, IR, uncertain,
            normalize=normalize, self_join=self_join, excl=excl,
        )
    except BaseException as exc:
        # an error or Ctrl-C mid-sweep: give the window block and the batch
        # buffers back to the system as a normal return does, rather than
        # leaving them in MLX's cache (hundreds of MiB) until the next call
        engine = None
        for prep in (A, Bs):  # either may not exist yet; idempotent
            if prep is not None:
                prep.release_device()
        free_gpu_after_error(exc)
        raise
    # the sweep is over: drop the window matrix and return the batch buffers
    # MLX cached for it to the system before the CPU refinement allocates its
    # float64 chunks, so the documented ceiling holds one phase at a time and
    # nothing stays cached after the call
    del engine
    A.release_device()
    Bs.release_device()  # the same object for self-joins; idempotent
    mx.clear_cache()
    A.release_search_arrays()
    Bs.release_search_arrays()  # likewise idempotent for self-joins

    # float64 re-evaluation of the profile values at the chosen indices: the
    # sweep keeps only the neighbours, and every reported value is computed
    # here (all k columns in one pass; rows without a neighbour get inf)
    refine = _refine_znorm if normalize else _refine_absolute
    P = refine(A, Bs, I)
    if k > 1:
        # near-ties can reorder under the refined values; keep columns ascending
        order = np.argsort(P, axis=1, kind="stable")
        P = np.take_along_axis(P, order, axis=1)
        I = np.take_along_axis(I, order, axis=1)
        # Object boxing is the next (and often largest) host-memory phase.
        # Release the l-by-k permutation as soon as both numeric arrays have
        # been reordered instead of retaining it while `out` is populated.
        del order

    # Mirror STUMPY's post-profile diagnostic. A profile dominated by tiny
    # distances often means a self-join was requested without its exclusion
    # zone; downstream users rely on this warning when checking join setup.
    warning_threshold = 1e-6
    first_profile = P[:, 0]
    # a subnormal-scale raw profile underflows in the mean; that must not
    # trip a caller's np.seterr(under="raise") after all the work is done
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        profile_too_small = first_profile.mean() < warning_threshold or np.all(
            first_profile < warning_threshold
        )
    if profile_too_small:
        warnings.warn(
            f"A large number of values in `P` are smaller than {warning_threshold}.\n"
            "For a self-join, try setting `ignore_trivial=True`.",
            stacklevel=stacklevel,
        )

    # whole-block assignment boxes the same Python floats/ints as a column
    # loop but walks both arrays in memory order (measured 2.9-4x faster
    # from k=16, the same at small k); IL/IR stay separate columns so no
    # (l, k+2) index temporary is built
    out = np.empty((A.l, 2 * k + 2), dtype=object)
    out[:, :k] = P
    out[:, k : 2 * k] = I
    out[:, 2 * k] = IL
    out[:, 2 * k + 1] = IR
    return mparray(out, m, k, denom)


def _check_device_id(device_id) -> None:
    """Validate STUMPY's CUDA ``device_id`` loosely: an int or a list of them.

    It selects nothing here. The GPU wrappers check MLX's active device
    separately, so the ids are only checked for shape and then ignored.
    """
    ids = [device_id] if isinstance(device_id, (int, np.integer)) else device_id
    try:
        ids = list(ids)
    except TypeError:
        ids = []
    if not ids or not all(
        isinstance(i, (int, np.integer)) and not isinstance(i, (bool, np.bool_)) and i >= 0
        for i in ids
    ):
        raise ValueError(
            "`device_id` must be a non-negative integer or a non-empty list of them "
            f"but found {device_id!r}."
        )


def _require_metal_gpu(entry: str) -> None:
    """Reject CPU execution through an explicitly GPU-named public API."""
    metal = mx.metal.is_available()
    device = mx.default_device()
    if not metal or device != mx.gpu:
        raise RuntimeError(
            f"`{entry}` requires an available Metal GPU as MLX's active device "
            f"(metal_available={metal}, default_device={device}). "
            "Select `mx.gpu` or use the unsuffixed API for CPU execution."
        )


def aamp(
    T_A: npt.ArrayLike,
    m: int,
    T_B: npt.ArrayLike | None = None,
    ignore_trivial: bool = True,
    p: float = 2.0,
    k: int = 1,
    *,
    chunk_size: int | None = None,
) -> mparray:
    """Non-normalized (top-k) matrix profile; drop-in for ``stumpy.aamp``.

    This is ``stump(..., normalize=False)`` under ``stumpy.aamp``'s own
    positional signature, whose fifth argument is ``p`` rather than
    ``normalize``. Only ``p=2.0`` is supported. ``chunk_size`` is as in
    :func:`stump`.
    """
    return _stump(
        T_A,
        m,
        T_B=T_B,
        ignore_trivial=ignore_trivial,
        normalize=False,
        p=p,
        k=k,
        T_A_subseq_isconstant=None,
        T_B_subseq_isconstant=None,
        chunk_size=chunk_size,
        stacklevel=3,
    )


def gpu_stump(
    T_A: npt.ArrayLike,
    m: int,
    T_B: npt.ArrayLike | None = None,
    ignore_trivial: bool = True,
    device_id: int | Sequence[int] = 0,
    normalize: bool = True,
    p: float = 2.0,
    k: int = 1,
    T_A_subseq_isconstant: IsConstantSpec = None,
    T_B_subseq_isconstant: IsConstantSpec = None,
    *,
    chunk_size: int | None = None,
) -> mparray:
    """:func:`stump` under ``stumpy.gpu_stump``'s positional signature.

    ``device_id`` (an int or a list of ints, as in STUMPY) is validated and
    then ignored. An available Metal GPU must be MLX's active device. This
    wrapper exists because a bare ``gpu_stump = stump`` alias would silently bind a
    positional ``device_id`` to ``normalize``.
    """
    _check_device_id(device_id)
    _require_metal_gpu("gpu_stump")
    return _stump(
        T_A,
        m,
        T_B=T_B,
        ignore_trivial=ignore_trivial,
        normalize=normalize,
        p=p,
        k=k,
        T_A_subseq_isconstant=T_A_subseq_isconstant,
        T_B_subseq_isconstant=T_B_subseq_isconstant,
        chunk_size=chunk_size,
        stacklevel=3,
    )


def gpu_aamp(
    T_A: npt.ArrayLike,
    m: int,
    T_B: npt.ArrayLike | None = None,
    ignore_trivial: bool = True,
    device_id: int | Sequence[int] = 0,
    p: float = 2.0,
    k: int = 1,
    *,
    chunk_size: int | None = None,
) -> mparray:
    """:func:`aamp` under ``stumpy.gpu_aamp``'s positional signature.

    ``device_id`` is validated and ignored as in :func:`gpu_stump`.
    An available Metal GPU must be MLX's active device.
    """
    _check_device_id(device_id)
    _require_metal_gpu("gpu_aamp")
    return _stump(
        T_A,
        m,
        T_B=T_B,
        ignore_trivial=ignore_trivial,
        normalize=False,
        p=p,
        k=k,
        T_A_subseq_isconstant=None,
        T_B_subseq_isconstant=None,
        chunk_size=chunk_size,
        stacklevel=3,
    )
