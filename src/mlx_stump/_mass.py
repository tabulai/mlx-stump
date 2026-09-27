"""`mass`: distance profile of one query subsequence against a series."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import mlx.core as mx
import numpy as np
from numpy.typing import ArrayLike

from ._engine import MassEngine, refine_chunk_rows
from ._preprocess import (
    IsConstantSpec,
    apply_affine_frame,
    call_isconstant,
    center_rows_stable,
    check_series,
    check_window_size,
    finite_center_scale,
    preprocess_series,
    process_isconstant,
    rowwise_l2_inplace,
    split_float32,
)


def _free_gpu_after_error(exc: BaseException) -> None:
    """Return a failed call's GPU buffers to the system.

    The success path releases its device arrays and clears MLX's cache
    itself. Call this from an inline ``except`` handler (a decorator would
    add a frame and shift every warning's ``stacklevel``) after dropping
    the handler frame's own device references. The finished frames below
    the handler (the block generator, the distance helpers) are kept alive
    by the traceback with their locals, e.g. the window block: clear those
    of mlx-stump's own frames (user frames stay intact for post-mortem
    debugging), wait for work still in flight, then clear the cache.
    (Private copy of the engine's helper, to be deduplicated.)
    """
    tb = exc.__traceback__
    tb = tb.tb_next if tb is not None else None  # the handler's frame is executing
    while tb is not None:
        if tb.tb_frame.f_globals.get("__name__", "").startswith("mlx_stump."):
            try:
                tb.tb_frame.clear()
            except RuntimeError:  # a still-executing or suspended frame
                pass
        tb = tb.tb_next
    mx.synchronize()  # a batch still in flight holds its buffers until it completes
    mx.clear_cache()


def _as_flag(value, name: str, Q: np.ndarray | None = None) -> bool | None:
    """Normalize a ``Q_subseq_isconstant`` spec to a bool (or ``None``).

    Accepts a Python/NumPy boolean or a boolean array of shape ``()`` or
    ``(1,)``. Anything else is rejected, as STUMPY does: coercing ``"False"``,
    ``0`` or ``[1.0]`` through ``bool()`` would silently flip the semantics.
    A callable is evaluated once on the (validated) query ``Q``, as
    ``f(Q, len(Q))`` with inf replaced by NaN, and must return a boolean of
    size 1 for the query's single window (STUMPY indexes the result, so a
    ``(1, 1)`` array is accepted there too).
    """
    if value is None:
        return None
    if callable(value):
        value = np.asarray(call_isconstant(value, Q, Q.shape[0], name))
        if value.dtype == np.bool_ and value.size == 1:
            value = value.reshape(())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    arr = np.asarray(value)
    if arr.dtype != np.bool_:
        raise ValueError(
            f"`{name}` must be a boolean (dtype `np.bool_`) but found dtype {arr.dtype}."
        )
    if arr.shape not in ((), (1,)):
        raise ValueError(f"`{name}` must be a single boolean value.")
    return bool(arr.reshape(-1)[0])


def _check_isfinite_override(T_subseq_isfinite, l: int) -> np.ndarray | None:
    if T_subseq_isfinite is None:
        return None
    arr = np.asarray(T_subseq_isfinite)
    if arr.dtype != np.bool_ or arr.shape != (l,):
        raise ValueError(f"`T_subseq_isfinite` must be a boolean array of shape ({l},).")
    return arr.copy()


def _check_stats(M_T, Σ_T, l: int) -> tuple[np.ndarray, np.ndarray]:
    """Validate precomputed sliding stats; returns float64 copies of shape ``(l,)``."""
    M = np.array(M_T, dtype=np.float64, copy=True)
    S = np.array(Σ_T, dtype=np.float64, copy=True)
    if M.shape != (l,) or S.shape != (l,):
        raise ValueError(
            f"`M_T` and `Σ_T` must both have shape ({l},) but found {M.shape} and {S.shape}."
        )
    return M, S


def _raw_window_distances(Q: np.ndarray, T: np.ndarray, js: np.ndarray) -> np.ndarray:
    """Float64 Euclidean distances from ``Q`` to the windows ``js`` of ``T``.

    Non-finite points of ``T`` count as raw zero, like the engine's
    zero-filled series. They are zeroed in each gathered chunk, not in a
    zero-filled copy of the whole series, so a handful of rows costs O(m)
    rather than O(n). Rows are processed in bounded chunks.
    """
    js = np.asarray(js, dtype=np.int64)
    m = Q.shape[0]
    out = np.empty(js.size, dtype=np.float64)
    Wfull = np.lib.stride_tricks.sliding_window_view(T, m)
    chunk = refine_chunk_rows(m)
    for start in range(0, js.size, chunk):
        W = Wfull[js[start : start + chunk]]  # fancy indexing: a float64 copy
        finite = np.isfinite(W)
        if not finite.all():
            W[~finite] = 0.0
        del finite
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            W -= Q[None, :]
        out[start : start + chunk] = rowwise_l2_inplace(W)
    return out


@dataclass
class _MassInfo:
    """What ``match`` reuses from the ``mass`` call it wraps."""

    q_const: bool | None = None  # resolved query constant flag
    isconstant: np.ndarray | None = None  # (l,) resolved target flags (normalized)
    isfinite: np.ndarray | None = None  # (l,) finite windows of T, before any override
    sigma: np.ndarray | None = None  # (l,) rolling sigma in the shared frame (raw)
    center: float = 0.0  # the shared affine frame (raw)
    scale: float = 1.0


def mass(
    Q: ArrayLike,
    T: ArrayLike,
    M_T: ArrayLike | None = None,
    Σ_T: ArrayLike | None = None,
    normalize: bool = True,
    p: float = 2.0,
    T_subseq_isfinite: ArrayLike | None = None,
    T_subseq_isconstant: IsConstantSpec = None,
    Q_subseq_isconstant: IsConstantSpec = None,
    query_idx: int | None = None,
) -> np.ndarray:
    """Compute the distance profile of query ``Q`` against series ``T``.

    Drop-in for ``stumpy.mass``. Returns a float64 array of length
    ``len(T) - len(Q) + 1``. A query containing NaN/inf yields an all-inf
    profile. ``query_idx`` must lie in ``[0, len(T) - len(Q)]`` and forces an
    exact zero at the query's own position (self-join convention; with
    ``normalize=False`` a window that ``T_subseq_isfinite`` marks non-finite
    stays inf, exactly like ``stumpy.mass_absolute``).
    ``normalize=False`` supports ``p=2.0`` only.
    ``T_subseq_isfinite`` is ignored when ``normalize=True``, like STUMPY.
    ``T_subseq_isconstant``/``Q_subseq_isconstant`` may be boolean arrays or
    STUMPY-style callables ``f(a, w)``, evaluated once on a copy of the
    series (or query) with inf replaced by NaN.

    With ``normalize=True``, precomputed ``M_T``/``Σ_T`` (when both are
    supplied, each with shape ``(l,)``) are accepted as STUMPY-compatibility
    metadata, not as a computational cache or an input to the arithmetic:
    they are validated,
    an infinite ``M_T`` marks its window non-finite (STUMPY's convention for
    windows containing NaN), and otherwise every raw window is locally
    centered and RMS-normalized in float64 exactly as in the no-stats call,
    so both profiles are identical. This is a deliberate choice
    of mathematical semantics over STUMPY's literal use of the supplied
    values, whose rounding STUMPY lets into the distance: its
    ``QT - m·μ_Q·M_T`` amplifies ``M_T``'s rounding by ``(μ/σ)²`` and
    collapses on offset data, and its ``1/(σ_Q·Σ_T)`` leaves a
    ``sqrt(2m·δ)`` floor on perfect matches from ``Σ_T``'s own relative
    rounding ``δ``. A deliberately scaled or biased ``M_T``/``Σ_T``
    therefore changes STUMPY's result but not this one; passing
    ``compute_mean_std``'s output reproduces STUMPY's ranking to within its
    own rounding. With ``normalize=False``, the pair is shape-validated and
    otherwise ignored, including non-finite entries.

    The target window matrix is streamed in bounded column blocks, so the
    profile costs one block of GPU memory at a time regardless of ``n·m``.

    Note the unrefined profile is the float32 GPU result: near-perfect matches read
    ~1e-3 rather than ~1e-8 (``stump`` and ``match`` re-evaluate their
    reported distances in float64; ``mass`` itself is not refined in either mode).
    """
    return _mass(
        Q,
        T,
        M_T,
        Σ_T,
        normalize,
        p,
        T_subseq_isfinite,
        T_subseq_isconstant,
        Q_subseq_isconstant,
        query_idx,
        stacklevel=3,
    )[0]


def mass_absolute(
    Q: ArrayLike,
    T: ArrayLike,
    T_subseq_isfinite: ArrayLike | None = None,
    p: float = 2.0,
    query_idx: int | None = None,
) -> np.ndarray:
    """Non-normalized distance profile with ``stumpy.core.mass_absolute``'s signature.

    The same computation as ``mass(Q, T, normalize=False, p=p,
    T_subseq_isfinite=T_subseq_isfinite, query_idx=query_idx)``, so a
    positional call ported from STUMPY binds the same parameters. Supports
    ``p=2.0`` only.
    """
    return _mass(
        Q,
        T,
        normalize=False,
        p=p,
        T_subseq_isfinite=T_subseq_isfinite,
        query_idx=query_idx,
        stacklevel=3,
    )[0]


def _mass(
    Q,
    T,
    M_T=None,
    Σ_T=None,
    normalize=True,
    p=2.0,
    T_subseq_isfinite=None,
    T_subseq_isconstant=None,
    Q_subseq_isconstant=None,
    query_idx=None,
    *,
    zero_query: bool = True,
    keep_sigma: bool = False,
    copy_series: bool = True,
    stacklevel: int,
) -> tuple[np.ndarray, _MassInfo]:
    """``mass``, plus the resolved flags and statistics ``match`` reuses.

    ``stacklevel`` (required) is that of a warning issued directly in this
    function; the public wrappers pass the depth of the user's call. With
    ``zero_query=False``, ``query_idx`` is range-checked but its entry is
    neither compared with ``Q`` nor zeroed (``stumpy.aamp_match`` keeps the
    true distance there). ``keep_sigma`` (raw mode) also returns the
    target's rolling sigma in the shared frame. ``copy_series=False`` is for
    a caller whose ``T`` is already a private validated copy (``match``
    after flattening an ``(n, 1)`` series or converting a byte-swapped one):
    it is used without a second copy. Nothing here writes to ``T``.
    """
    Q = np.asarray(Q)
    if Q.ndim == 2 and Q.shape[1] == 1:
        warnings.warn(
            "`Q` must be 1-dimensional and was automatically flattened", stacklevel=stacklevel
        )
        Q = Q.flatten()
    T = np.asarray(T)
    if T.ndim == 2 and T.shape[1] == 1:
        T = T.flatten()
    Q = check_series(Q, "Q")
    T = check_series(T, "T", copy=copy_series)
    m = check_window_size(int(Q.shape[0]), T.shape[0])
    l = T.shape[0] - m + 1
    info = _MassInfo()

    if not normalize and p != 2.0:
        raise NotImplementedError(
            "mlx-stump supports p=2.0 only when normalize=False; "
            f"found p={p}. Use stumpy.mass_absolute for other p-norms."
        )

    if query_idx is not None:
        query_idx = int(query_idx)
        if not 0 <= query_idx < l:
            # negative values would silently wrap via numpy indexing and
            # fabricate a zero-distance "match" at a bogus position
            raise ValueError(
                f"`query_idx` must be an integer in [0, {l - 1}] but found {query_idx}."
            )
        if not zero_query:
            query_idx = None
    if query_idx is not None:
        Q_isfinite_pt = np.isfinite(Q)
        T_win = T[query_idx : query_idx + m]
        T_isfinite_pt = np.isfinite(T_win)
        if not np.array_equal(Q_isfinite_pt, T_isfinite_pt) or not np.allclose(
            Q[Q_isfinite_pt], T_win[T_isfinite_pt]
        ):
            warnings.warn(
                "Subsequences `Q` and `T[query_idx:query_idx+m]` are different but "
                "were expected to be identical. Please verify that `query_idx` "
                "is correct.",
                stacklevel=stacklevel,
            )

    # STUMPY returns immediately for an invalid query before consulting
    # optional cached statistics or constant flags. Preserve the simple
    # documented contract even when those otherwise-validated inputs are bad.
    if not np.all(np.isfinite(Q)):
        return np.full(l, np.inf), info

    user_stats = M_T is not None and Σ_T is not None
    if user_stats:
        M_T, Σ_T = _check_stats(M_T, Σ_T, l)
    else:
        # Like STUMPY, an incomplete pair means "recompute both". Drop a
        # lone temporary array now instead of pinning it through the GPU call.
        del M_T, Σ_T
    q_const = _as_flag(Q_subseq_isconstant, "Q_subseq_isconstant", Q)

    if not normalize and T_subseq_isconstant is not None:
        # Constant flags have no role in Euclidean distance, but accepting a
        # malformed value in only some raw APIs is hazardous. Validate the
        # compatibility control consistently, then deliberately ignore it.
        process_isconstant(T, m, T_subseq_isconstant, "T_subseq_isconstant")

    if q_const is None:
        q_const = bool(np.min(Q) == np.max(Q))
    info.q_const = q_const

    D2 = np.empty(l, dtype=np.float64)
    forced_zero_fill = None
    prep = None
    try:
        if normalize:
            # T_subseq_isfinite is deliberately NOT applied here: STUMPY documents
            # it as ignored when normalize=True (it only feeds mass_absolute)
            prep = preprocess_series(
                T, m, isconstant=T_subseq_isconstant, stacklevel=stacklevel + 1
            )
            info.isconstant, info.isfinite = prep.isconstant, prep.isfinite
            if user_stats:
                # The supplied arrays are compatibility metadata: every raw window
                # still goes through the same local float64 centering and RMS
                # normalization, so the profile equals the no-stats call exactly.
                # STUMPY's literal
                # use of the supplied values (`QT - m*mu_Q*M_T`, `1/(sigma_Q*Σ_T)`)
                # lets their rounding into the distance — amplified by
                # (mu/sigma)^2 for M_T, and as a sqrt(2m*delta) floor on perfect
                # matches for Σ_T. The one convention kept is STUMPY's marker
                # for windows containing NaN: an infinite M_T reports inf.
                bad_mean = np.isinf(M_T)
                if bad_mean.any():
                    prep.isfinite = prep.isfinite & ~bad_mean
                    prep.isfinite_mx = mx.array(prep.isfinite)
                del bad_mean, M_T, Σ_T

            # Put Q in a bounded midpoint/range frame before its two-pass stats:
            # shifting by the first element alone avoids large common-offset
            # cancellation but can still overflow for opposite-sign extremes,
            # while squaring raw tiny values can underflow to zero.
            Qc = Q.copy()[None, :]
            center_rows_stable(Qc)
            Qc = Qc[0]
            sigma_q = float(np.sqrt(Qc @ Qc / m))
            s = sigma_q if (np.isfinite(sigma_q) and sigma_q > 0.0) else 1.0
            Qs = Qc / s
            del Qc

            engine = MassEngine(prep)
            Qb = mx.array(Qs.astype(np.float32))[None, :]
            del Qs
            sig_inv_q = mx.array([0.0 if q_const else 1.0], dtype=mx.float32)
            isconst_q = mx.array([q_const])
            isfinite_q = mx.array([True])
            for j0, j1, W in engine.target_blocks():
                d2 = engine.znorm_sq_distances(
                    mx.matmul(Qb, W), sig_inv_q, isconst_q, isfinite_q, j0, j1
                )
                mx.eval(d2)
                mx.synchronize()  # completion handler returns the block's buffers
                D2[j0:j1] = np.array(d2[0], dtype=np.float64)
                del W, d2  # release this block before the next one is built
            del Qb, sig_inv_q, isconst_q, isfinite_q
        else:
            if user_stats:
                # normalize=False does not consult compatibility statistics.
                del M_T, Σ_T
            # the shared frame of the query and T's finite values, without
            # compacted-length copies (Q is finite here)
            center, scale = finite_center_scale(Q, T)
            prep = preprocess_series(
                T,
                m,
                normalize=False,
                center=center,
                scale=scale,
                keep_sigma=keep_sigma,
                stacklevel=stacklevel + 1,
            )
            info.isfinite, info.sigma = prep.isfinite, prep.sigma
            info.center, info.scale = prep.center, prep.scale
            override = _check_isfinite_override(T_subseq_isfinite, l)
            if override is not None:
                forced_zero_fill = override & ~prep.isfinite
                prep.isfinite = override
                prep.isfinite_mx = mx.array(override)
            del override
            Qs = apply_affine_frame(Q, center, scale)
            mu_q = float(Qs.mean())
            Qsc = Qs - mu_q
            del Qs
            engine = MassEngine(prep, normalize=False)
            Qb = mx.array(Qsc.astype(np.float32))[None, :]
            ssq_q = mx.array([float(np.sum(Qsc * Qsc))], dtype=mx.float32)
            del Qsc
            mu_q_mx = mx.array(split_float32(np.array([mu_q])))
            isfinite_q = mx.array([True])
            for j0, j1, W in engine.target_blocks():
                d2 = engine.absolute_sq_distances(
                    mx.matmul(Qb, W), ssq_q, mu_q_mx, isfinite_q, j0, j1
                )
                mx.eval(d2)
                mx.synchronize()
                D2[j0:j1] = np.array(d2[0], dtype=np.float64)
                del W, d2
            del Qb, ssq_q, mu_q_mx, isfinite_q
    except BaseException as exc:
        # an error or interrupt (Ctrl-C) mid-search: drop this frame's device
        # references (the block loop's locals) and prep's arrays, then what
        # the traceback still holds below this frame, so nothing stays
        # cached in MLX after the call raises
        engine = W = d2 = Qb = None
        sig_inv_q = isconst_q = isfinite_q = ssq_q = mu_q_mx = None
        if prep is not None:
            prep.release_device()
        _free_gpu_after_error(exc)
        raise

    # nothing of the GPU phase is needed any more: drop every device array
    # (window block, query batch, per-window stats) BEFORE clearing MLX's
    # buffer cache, or they would enter it when this function returns
    del engine
    prep.release_device()
    mx.clear_cache()
    # Reuse the profile buffer: allocating a second l-wide float64 array here
    # needlessly overlaps the final GPU/cache teardown phase.
    np.sqrt(D2, out=D2)
    profile = D2
    if not normalize:
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            profile *= prep.scale
        if forced_zero_fill is not None and np.any(forced_zero_fill):
            # Preprocessing uses the affine center (standardized zero) as a
            # harmless sentinel so one NaN cannot poison later rolling stats.
            # Preserve the historical explicit-override contract separately:
            # a window the user forces finite is measured against raw zero at
            # its non-finite points. Recompute only those exceptional rows in
            # bounded CPU chunks, before match applies any threshold.
            js = np.nonzero(forced_zero_fill)[0]
            profile[js] = _raw_window_distances(Q, T, js)
    if query_idx is not None and (normalize or prep.isfinite[query_idx]):
        # STUMPY zeroes the self-match unconditionally when z-normalized, but
        # mass_absolute re-applies its finite mask afterwards, so a window an
        # explicit T_subseq_isfinite marks non-finite stays inf there
        profile[query_idx] = 0.0
    return profile, info
