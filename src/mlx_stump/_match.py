"""`match`: all occurrences of a query in a series, nearest first."""

from __future__ import annotations

import warnings

import numpy as np

from ._mass import _as_flag, mass
from ._preprocess import EXCL_ZONE_DENOM, process_isconstant, rolling_isfinite
from ._stump import _refine_chunk_rows


def _refine_candidates(Q, T, D, cutoff, normalize, q_const, t_const):
    """Float64 re-evaluation of finite profile entries at or below ``cutoff``.

    The GPU profile is float32, so a perfect occurrence reads ~1e-3 instead
    of ~0 and a tight ``max_distance`` would silently drop it; re-evaluating
    every candidate near the threshold restores STUMPY's behavior.

    Only finite entries are candidates (non-finite windows can never match,
    and would poison the greedy argmin with NaNs); candidates are processed
    in byte-budgeted chunks (``max_distance=inf`` selects the whole profile,
    which must not materialize l*m float64 windows at once). The squared
    distance is a sum of squared differences of the two z-normalized windows
    — the `W@Q - m*mu_t*mu_q` form cancels catastrophically for
    near-constant windows at an offset, even in float64. ``q_const``/
    ``t_const`` are the resolved constant flags (including any user
    overrides), applied exactly like the GPU path applies them. Unlike
    ``stump``, no P-norm zero-snap is applied: STUMPY's mass/match report
    raw float64 distances (exact duplicates still read 0.0 here because the
    sum of squares is exactly 0 for identical normalized windows).
    """
    js = np.nonzero((D <= cutoff) & np.isfinite(D))[0]
    if js.size == 0:
        return D
    m = Q.shape[0]
    D = D.copy()
    chunk = _refine_chunk_rows(m)
    if normalize:
        Wfull = np.lib.stride_tricks.sliding_window_view(T, m)
        Qc = Q - Q.mean()
        sig_q = float(np.sqrt(Qc @ Qc / m))
        sig_inv_q = 0.0 if (q_const or sig_q == 0.0) else 1.0 / sig_q
        u = Qc * sig_inv_q
        for s in range(0, js.size, chunk):
            idx = js[s : s + chunk]
            W = Wfull[idx].astype(np.float64)
            W -= W.mean(axis=1)[:, None]
            sig_t = np.sqrt(np.einsum("ij,ij->i", W, W) / m)
            tc = t_const[idx]
            pos = (sig_t > 0.0) & ~tc
            sig_inv_t = np.where(pos, 1.0 / np.where(sig_t > 0.0, sig_t, 1.0), 0.0)
            W *= sig_inv_t[:, None]
            W -= u[None, :]
            d2 = np.einsum("ij,ij->i", W, W)
            # zero sig_inv on either side means rho == 0 on the GPU; mirror
            # it, then let the constant-flag rules overwrite as needed
            d2 = np.where((sig_inv_t == 0.0) | (sig_inv_q == 0.0), 2.0 * m, d2)
            d = np.sqrt(d2)
            D[idx] = np.where(q_const & tc, 0.0, np.where(q_const ^ tc, np.sqrt(m), d))
    else:
        # the engine computes on the zero-filled series; mirror it so a user
        # T_subseq_isfinite override cannot inject NaN into the profile
        Tf = np.where(np.isfinite(T), T, 0.0)
        Wfull = np.lib.stride_tricks.sliding_window_view(Tf, m)
        for s in range(0, js.size, chunk):
            idx = js[s : s + chunk]
            diff = Wfull[idx].astype(np.float64) - Q[None, :]
            D[idx] = np.sqrt(np.einsum("ij,ij->i", diff, diff))
    return D


def _apply_exclusion_zone(a: np.ndarray, idx: int, excl_zone: int, val: float) -> None:
    zone_start = max(0, idx - excl_zone)
    zone_stop = min(a.shape[-1], idx + excl_zone)
    a[zone_start : zone_stop + 1] = val


def _find_matches(
    D: np.ndarray,
    excl_zone: int,
    max_distance=None,
    max_matches=None,
    query_idx=None,
    atol=1e-8,
) -> np.ndarray:
    """Greedy nearest-first selection with exclusion zones (STUMPY port)."""
    D = D.copy().astype(np.float64)
    if max_distance is None:

        def max_distance(D):
            D_copy = D.copy()
            D_copy[np.isinf(D_copy)] = np.nan
            return np.nanmax([np.nanmean(D_copy) - 2.0 * np.nanstd(D_copy), np.nanmin(D_copy)])

    if not isinstance(max_distance, float):
        max_distance = max_distance(D)

    if max_matches is None:
        max_matches = np.inf

    if query_idx is not None:
        candidate_idx = query_idx
    else:
        candidate_idx = np.argmin(D)

    matches = []
    for _ in range(len(D)):
        if (
            D[candidate_idx] > atol + max_distance
            or ~np.isfinite(D[candidate_idx])
            or len(matches) >= max_matches
        ):
            break
        matches.append([D[candidate_idx], int(candidate_idx)])
        _apply_exclusion_zone(D, candidate_idx, excl_zone, np.inf)
        candidate_idx = np.argmin(D)

    return np.array(matches, dtype=object)


def match(
    Q,
    T,
    M_T=None,
    Σ_T=None,
    max_distance=None,
    max_matches=None,
    atol=1e-8,
    query_idx=None,
    normalize=True,
    p=2.0,
    T_subseq_isfinite=None,
    T_subseq_isconstant=None,
    Q_subseq_isconstant=None,
):
    """Find all subsequences of ``T`` matching query ``Q``, nearest first.

    Drop-in for ``stumpy.match``: returns an object array of
    ``[distance, index]`` rows sorted by distance, using STUMPY's default
    ``max_distance`` (``max(mean(D) - 2*std(D), min(D))``) and exclusion-zone
    semantics. ``max_distance`` may also be a callable receiving the distance
    profile (called on the float32 profile and once more on the float64-refined
    one; STUMPY calls it once on its exact profile). ``normalize=False``
    supports ``p=2.0`` only.
    """
    Q = np.asarray(Q)
    if Q.ndim == 2 and Q.shape[1] == 1:
        Q = Q.flatten()
    T = np.asarray(T)
    if T.ndim == 2 and T.shape[1] == 1:
        T = T.flatten()
    if np.any(np.isnan(Q)) or np.any(np.isinf(Q)):
        raise ValueError("Q contains illegal values (NaN or inf)")

    m = Q.shape[-1]
    excl_zone = int(np.ceil(m / EXCL_ZONE_DENOM))

    D = mass(
        Q,
        T,
        M_T=M_T,
        Σ_T=Σ_T,
        normalize=normalize,
        p=p,
        T_subseq_isfinite=T_subseq_isfinite,
        T_subseq_isconstant=T_subseq_isconstant,
        Q_subseq_isconstant=Q_subseq_isconstant,
        query_idx=query_idx,
    )

    Qf = np.asarray(Q, dtype=np.float64)
    Tf = np.asarray(T, dtype=np.float64)
    q_const = _as_flag(Q_subseq_isconstant, "Q_subseq_isconstant")
    if q_const is None:
        q_const = bool(np.ptp(Qf) == 0.0)
    T_nan = np.where(np.isinf(Tf), np.nan, Tf)
    with warnings.catch_warnings():
        # the mass() call above already resolved these flags and warned
        warnings.simplefilter("ignore")
        t_const = process_isconstant(T_nan, m, T_subseq_isconstant, "T_subseq_isconstant")
    t_const &= rolling_isfinite(np.isfinite(Tf), m)

    def _threshold(D):
        if max_distance is None:
            D_copy = D.copy()
            D_copy[np.isinf(D_copy)] = np.nan
            return float(
                np.nanmax([np.nanmean(D_copy) - 2.0 * np.nanstd(D_copy), np.nanmin(D_copy)])
            )
        if not isinstance(max_distance, float):
            return float(max_distance(D))
        return max_distance

    # estimate the threshold on the float32 profile, re-evaluate every
    # near-threshold candidate in float64, then recompute the threshold on
    # the refined profile (STUMPY derives it from an exact float64 profile —
    # in particular min(D) shifts when refinement corrects the best match)
    # and refine once more in case it rose
    margin = 0.1 * (m / 50.0) ** 0.25  # float32 search noise bound
    if not normalize:
        # absolute distances (and their float32 noise) scale with the data's
        # units; use the same shared frame mass() standardized with
        finite = np.concatenate([Qf, Tf[np.isfinite(Tf)]])
        scale = float(finite.std()) if finite.size else 1.0
        if not np.isfinite(scale) or scale == 0.0:
            scale = 1.0
        margin *= scale
    md = _threshold(D)
    D = _refine_candidates(Qf, Tf, D, md + atol + margin, normalize, q_const, t_const)
    if not isinstance(max_distance, float):
        md = _threshold(D)
        D = _refine_candidates(Qf, Tf, D, md + atol + margin, normalize, q_const, t_const)
    if query_idx is not None:
        D[query_idx] = 0.0

    return _find_matches(
        D,
        excl_zone,
        max_distance=md,
        max_matches=max_matches,
        query_idx=query_idx,
        atol=atol,
    )
