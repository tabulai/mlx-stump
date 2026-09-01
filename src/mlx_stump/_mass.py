"""`mass`: distance profile of one query subsequence against a series."""

from __future__ import annotations

import warnings

import mlx.core as mx
import numpy as np

from ._engine import MassEngine
from ._preprocess import (
    check_series,
    check_window_size,
    preprocess_series,
    split_float32,
)


def _as_flag(value, name: str) -> bool | None:
    """Normalize a ``Q_subseq_isconstant`` spec to a bool (or ``None``).

    Accepts a Python/NumPy boolean or a boolean array of shape ``()`` or
    ``(1,)``. Anything else is rejected, as STUMPY does: coercing ``"False"``,
    ``0`` or ``[1.0]`` through ``bool()`` would silently flip the semantics.
    """
    if value is None:
        return None
    if callable(value):
        raise NotImplementedError(
            f"Callable `{name}` is not supported yet; pass a boolean instead."
        )
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


def mass(
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
):
    """Compute the distance profile of query ``Q`` against series ``T``.

    Drop-in for ``stumpy.mass``. Returns a float64 array of length
    ``len(T) - len(Q) + 1``. A query containing NaN/inf yields an all-inf
    profile. ``query_idx`` must lie in ``[0, len(T) - len(Q)]`` and forces an
    exact zero at the query's own position (self-join convention; with
    ``normalize=False`` a window that ``T_subseq_isfinite`` marks non-finite
    stays inf, exactly like ``stumpy.mass_absolute``).
    ``normalize=False`` supports ``p=2.0`` only.
    ``T_subseq_isfinite`` is ignored when ``normalize=True``, like STUMPY.

    Precomputed ``M_T``/``Σ_T`` (both required, shape ``(l,)``) are a
    *cache*, not an input to the arithmetic: they are validated, an
    infinite ``M_T`` marks its window non-finite (STUMPY's convention for
    windows containing NaN), and otherwise every window is centered by its
    own exact float64 mean and scaled by its own exact sigma, so the
    profile equals the no-stats call exactly. This is a deliberate choice
    of mathematical semantics over STUMPY's literal use of the supplied
    values, whose rounding STUMPY lets into the distance: its
    ``QT - m·μ_Q·M_T`` amplifies ``M_T``'s rounding by ``(μ/σ)²`` and
    collapses on offset data, and its ``1/(σ_Q·Σ_T)`` leaves a
    ``sqrt(2m·δ)`` floor on perfect matches from ``Σ_T``'s own relative
    rounding ``δ``. A deliberately scaled or biased ``M_T``/``Σ_T``
    therefore changes STUMPY's result but not this one; passing
    ``compute_mean_std``'s output reproduces STUMPY's ranking to within
    its own rounding.

    The target window matrix is streamed in bounded column blocks, so the
    profile costs one block of GPU memory at a time regardless of ``n·m``.

    Note the profile is the float32 GPU result: near-perfect matches read
    ~1e-3 rather than ~1e-8 (``stump`` and ``match`` re-evaluate their
    reported distances in float64; a raw ``mass`` profile is not refined).
    """
    Q = np.asarray(Q)
    if Q.ndim == 2 and Q.shape[1] == 1:
        warnings.warn("`Q` must be 1-dimensional and was automatically flattened", stacklevel=2)
        Q = Q.flatten()
    T = np.asarray(T)
    if T.ndim == 2 and T.shape[1] == 1:
        T = T.flatten()
    Q = check_series(Q, "Q")
    T = check_series(T, "T")
    m = check_window_size(int(Q.shape[0]), T.shape[0])
    l = T.shape[0] - m + 1

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
                stacklevel=2,
            )

    # validate eagerly (STUMPY would fail on a shape mismatch too), even for
    # the all-inf early return below
    user_stats = M_T is not None and Σ_T is not None
    if user_stats:
        M_T, Σ_T = _check_stats(M_T, Σ_T, l)
    q_const = _as_flag(Q_subseq_isconstant, "Q_subseq_isconstant")

    if not np.all(np.isfinite(Q)):
        return np.full(l, np.inf)

    if q_const is None:
        q_const = bool(np.ptp(Q) == 0.0)

    D2 = np.empty(l, dtype=np.float64)
    if normalize:
        # T_subseq_isfinite is deliberately NOT applied here: STUMPY documents
        # it as ignored when normalize=True (it only feeds mass_absolute)
        prep = preprocess_series(T, m, isconstant=T_subseq_isconstant)
        if user_stats:
            # cached statistics are a cache: every window is centered by its
            # own exact float64 mean and scaled by its own exact sigma, so
            # the profile equals the no-stats call exactly. STUMPY's literal
            # use of the supplied values (`QT - m*mu_Q*M_T`, `1/(sigma_Q*Σ_T)`)
            # lets their rounding into the distance — amplified by
            # (mu/sigma)^2 for M_T, and as a sqrt(2m*delta) floor on perfect
            # matches for Σ_T. The one convention kept is STUMPY's marker
            # for windows containing NaN: an infinite M_T reports inf.
            bad_mean = np.isinf(M_T)
            if bad_mean.any():
                prep.isfinite = prep.isfinite & ~bad_mean
                prep.isfinite_mx = mx.array(prep.isfinite)

        # standardize Q by its own moments (the window then has mean 0, sigma
        # 1), shifted by its first element first: `x - x0` errs with the
        # window's SPREAD, so a large common offset cannot bias the mean and,
        # through it, sigma_q (an unshifted Q.std() left an exact self-match
        # reading ~1 instead of ~1e-3 for flat-jitter queries at offset 1e12)
        Qs0 = Q - Q[0]
        Qc = Qs0 - Qs0.mean()
        sigma_q = float(np.sqrt(Qc @ Qc / m))
        s = sigma_q if (np.isfinite(sigma_q) and sigma_q > 0.0) else 1.0
        Qs = Qc / s

        engine = MassEngine(prep)
        Qb = mx.array(Qs.astype(np.float32))[None, :]
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
        finite = np.concatenate([Q, T[np.isfinite(T)]])
        center = float(finite.mean())
        scale = float(finite.std())
        if not np.isfinite(scale) or scale == 0.0:
            scale = 1.0
        prep = preprocess_series(T, m, normalize=False, center=center, scale=scale)
        override = _check_isfinite_override(T_subseq_isfinite, l)
        if override is not None:
            prep.isfinite = override
            prep.isfinite_mx = mx.array(override)
        Qs = (Q - center) / scale
        mu_q = float(Qs.mean())
        Qsc = Qs - mu_q
        engine = MassEngine(prep, normalize=False)
        Qb = mx.array(Qsc.astype(np.float32))[None, :]
        ssq_q = mx.array([float(np.sum(Qsc * Qsc))], dtype=mx.float32)
        mu_q_mx = mx.array(split_float32(np.array([mu_q])))
        isfinite_q = mx.array([True])
        for j0, j1, W in engine.target_blocks():
            d2 = engine.absolute_sq_distances(mx.matmul(Qb, W), ssq_q, mu_q_mx, isfinite_q, j0, j1)
            mx.eval(d2)
            mx.synchronize()
            D2[j0:j1] = np.array(d2[0], dtype=np.float64)
            del W, d2
        del Qb, ssq_q, mu_q_mx, isfinite_q

    # nothing of the GPU phase is needed any more: drop every device array
    # (window block, query batch, per-window stats) BEFORE clearing MLX's
    # buffer cache, or they would enter it when this function returns
    del engine
    prep.release_device()
    mx.clear_cache()
    profile = np.sqrt(D2)
    if not normalize:
        profile *= prep.scale
    if query_idx is not None and (normalize or prep.isfinite[query_idx]):
        # STUMPY zeroes the self-match unconditionally when z-normalized, but
        # mass_absolute re-applies its finite mask afterwards, so a window an
        # explicit T_subseq_isfinite marks non-finite stays inf there
        profile[query_idx] = 0.0
    return profile
