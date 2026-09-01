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
)


def _as_flag(value, name: str) -> bool | None:
    """Normalize a Q_subseq_isconstant spec (None, bool, or shape-(1,) array)."""
    if value is None:
        return None
    if callable(value):
        raise NotImplementedError(
            f"Callable `{name}` is not supported yet; pass a boolean instead."
        )
    arr = np.asarray(value)
    if arr.shape not in ((), (1,)):
        raise ValueError(f"`{name}` must be a single boolean value.")
    return bool(arr.reshape(-1)[0]) if arr.shape == (1,) else bool(arr)


def _check_isfinite_override(T_subseq_isfinite, l: int) -> np.ndarray | None:
    if T_subseq_isfinite is None:
        return None
    arr = np.asarray(T_subseq_isfinite)
    if arr.dtype != np.bool_ or arr.shape != (l,):
        raise ValueError(f"`T_subseq_isfinite` must be a boolean array of shape ({l},).")
    return arr.copy()


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
    profile. ``query_idx`` must lie in ``[0, len(T) - len(Q)]`` and forces an exact
    zero at the query's own position (self-join convention).
    ``normalize=False`` supports ``p=2.0`` only.
    ``T_subseq_isfinite`` is ignored when ``normalize=True``, like STUMPY.

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

    if not np.all(np.isfinite(Q)):
        return np.full(l, np.inf)

    q_const = _as_flag(Q_subseq_isconstant, "Q_subseq_isconstant")
    if q_const is None:
        q_const = bool(np.ptp(Q) == 0.0)

    if normalize:
        # T_subseq_isfinite is deliberately NOT applied here: STUMPY documents
        # it as ignored when normalize=True (it only feeds mass_absolute)
        prep = preprocess_series(T, m, isconstant=T_subseq_isconstant)
        if M_T is not None and Σ_T is not None:
            # trusted precomputed raw-frame stats, mapped into the standardized frame
            mu = (np.asarray(M_T, dtype=np.float64) - prep.center) / prep.scale
            sigma = np.asarray(Σ_T, dtype=np.float64) / prep.scale
            sigma = sigma.copy()
            sigma[prep.isconstant] = 0.0
            with np.errstate(divide="ignore"):
                sig_inv = np.where(sigma > 0.0, 1.0 / sigma, 0.0)
            prep.mu, prep.sig_inv = mu, sig_inv
            prep.sig_inv_mx = mx.array(sig_inv.astype(np.float32))

        # standardize Q by its own moments: the window then has mean 0, sigma 1
        mu_q = float(Q.mean())
        sigma_q = float(Q.std())
        s = sigma_q if (np.isfinite(sigma_q) and sigma_q > 0.0) else 1.0
        Qs = (Q - mu_q) / s

        engine = MassEngine(prep)
        QT = engine.sliding_dot_products(mx.array(Qs.astype(np.float32))[None, :])
        D2 = engine.znorm_sq_distances(
            QT,
            mx.array([0.0 if q_const else 1.0], dtype=mx.float32),
            mx.array([q_const]),
            mx.array([True]),
        )
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
        engine = MassEngine(prep, normalize=False)
        QT = engine.sliding_dot_products(mx.array(Qs.astype(np.float32))[None, :])
        D2 = engine.absolute_sq_distances(
            QT,
            mx.array([float(np.sum(Qs * Qs))], dtype=mx.float32),
            mx.array([True]),
        )

    mx.eval(D2)
    profile = np.sqrt(np.array(D2[0], dtype=np.float64))
    if not normalize:
        profile *= prep.scale
    if query_idx is not None:
        profile[query_idx] = 0.0
    return profile
