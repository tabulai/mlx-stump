"""`match`: all occurrences of a query in a series, nearest first."""

from __future__ import annotations

import warnings

import numpy as np

from ._mass import _as_flag, mass
from ._preprocess import (
    EXCL_ZONE_DENOM,
    process_isconstant,
    rolling_isfinite,
    rolling_mean_sigma,
)
from ._stump import _refine_chunk_rows

# refinement/threshold rounds for a data-dependent max_distance (the loop
# converges as soon as a round refines nothing new, typically in 2-3 rounds)
_MAX_REFINE_ROUNDS = 8


def _refine_candidates(Q, T, js, normalize, q_const, t_const):
    """Float64 re-evaluation of the finite profile entries ``js``.

    The GPU profile is float32, so a perfect occurrence reads ~1e-3 instead
    of ~0 and a tight ``max_distance`` would silently drop it; re-evaluating
    every candidate near the threshold restores STUMPY's behavior. Returns
    the refined distances for ``js`` (same order).

    Candidates are processed in byte-budgeted chunks (``max_distance=inf``
    selects the whole profile, which must not materialize l*m float64
    windows at once). Every window is centered by its own exact two-pass
    mean (shifted by its first element first, so a large common offset
    cannot poison the mean) and scaled by its own exact sigma: the squared
    distance is the sum of squared differences of the two z-normalized
    windows, cancellation-free down to 0. The `W@Q - m*mu_t*mu_q` form
    cancels catastrophically for near-constant windows at an offset, even
    in float64, and any form built on an externally supplied sigma leaves
    that sigma's rounding as a floor on perfect matches — so cached
    statistics passed to ``match`` are never used here (see ``mass``).
    ``q_const``/``t_const`` are the resolved constant flags (including any
    user overrides), applied exactly like the GPU path applies them.
    Unlike ``stump``, no P-norm zero-snap is applied: STUMPY's mass/match
    report raw float64 distances (exact and shifted duplicates still read
    0.0 because the sum of squares is exactly 0 for identical normalized
    windows).
    """
    js = np.asarray(js, dtype=np.int64)
    out = np.empty(js.size, dtype=np.float64)
    if js.size == 0:
        return out
    m = Q.shape[0]
    chunk = _refine_chunk_rows(m)
    if normalize:
        Wfull = np.lib.stride_tricks.sliding_window_view(T, m)
        # shift by the first element: `x - x0` errs with the window's SPREAD,
        # so a large common offset cannot poison the two-pass mean below.
        # The query goes through the same 2-D reductions as the windows:
        # numpy's 1-D mean/dot can differ from the axis-1 versions by an ulp
        # on identical data, which left exact duplicates at ~2e-15, not 0
        Qw = Q[None, :] - Q[0]
        Qw -= Qw.mean(axis=1)[:, None]
        sig_q = float(np.sqrt(np.einsum("ij,ij->i", Qw, Qw)[0] / m))
        sig_inv_q = 0.0 if (q_const or sig_q == 0.0) else 1.0 / sig_q
        u = Qw[0] * sig_inv_q
        for s in range(0, js.size, chunk):
            idx = js[s : s + chunk]
            W = Wfull[idx].astype(np.float64)
            tc = t_const[idx]
            W -= W[:, 0].copy()[:, None]
            W -= W.mean(axis=1)[:, None]
            sig_t = np.sqrt(np.einsum("ij,ij->i", W, W) / m)
            pos = (sig_t > 0.0) & ~tc
            sig_inv_t = np.where(pos, 1.0 / np.where(sig_t > 0.0, sig_t, 1.0), 0.0)
            W *= sig_inv_t[:, None]
            W -= u[None, :]
            d2 = np.einsum("ij,ij->i", W, W)
            # zero sig_inv on either side means rho == 0 on the GPU; mirror
            # it, the constant-flag rules overwrite as needed
            d2 = np.where((sig_inv_t == 0.0) | (sig_inv_q == 0.0), 2.0 * m, d2)
            d = np.sqrt(d2)
            out[s : s + chunk] = np.where(q_const & tc, 0.0, np.where(q_const ^ tc, np.sqrt(m), d))
    else:
        # the engine computes on the zero-filled series; mirror it so a user
        # T_subseq_isfinite override cannot inject NaN into the profile
        Tf = np.where(np.isfinite(T), T, 0.0)
        Wfull = np.lib.stride_tricks.sliding_window_view(Tf, m)
        for s in range(0, js.size, chunk):
            idx = js[s : s + chunk]
            diff = Wfull[idx].astype(np.float64) - Q[None, :]
            out[s : s + chunk] = np.sqrt(np.einsum("ij,ij->i", diff, diff))
    return out


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
    semantics. ``max_distance`` may be a number (fixed threshold) or a
    callable receiving the distance profile. A data-dependent threshold
    (the default, or a callable) is evaluated on successively refined
    profiles until it stops moving — the float32 GPU profile first, then
    once per float64 refinement round (typically 2-3 calls in total, at
    most 9; STUMPY calls it exactly once, on its exact float64 profile) —
    so the threshold that selects the matches comes from a profile that is
    float64-exact throughout the threshold band.
    ``normalize=False`` supports ``p=2.0`` only. Precomputed ``M_T``/``Σ_T``
    follow the ``mass`` contract: they are a cache, validated and otherwise
    unused (an infinite ``M_T`` marks its window non-finite); the search
    and the refinement use exact statistics, so the result equals the
    no-stats call.
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
    l = D.shape[0]

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

    fixed = isinstance(max_distance, (int, float, np.integer, np.floating)) and not isinstance(
        max_distance, (bool, np.bool_)
    )

    def _threshold(D):
        if max_distance is None:
            D_copy = D.copy()
            D_copy[np.isinf(D_copy)] = np.nan
            return float(
                np.nanmax([np.nanmean(D_copy) - 2.0 * np.nanstd(D_copy), np.nanmin(D_copy)])
            )
        if fixed:
            return float(max_distance)
        return float(max_distance(D))

    # float32 search noise bound (distance units); every entry within it of
    # the threshold is re-evaluated in float64 before the threshold is applied
    margin = 0.1 * (m / 50.0) ** 0.25
    if not normalize:
        # absolute distances (and their float32 noise) scale with the data's
        # units; use the same shared frame mass() standardized with
        finite = np.concatenate([Qf, Tf[np.isfinite(Tf)]])
        scale = float(finite.std()) if finite.size else 1.0
        if not np.isfinite(scale) or scale == 0.0:
            scale = 1.0
        # the engine's float32 cancellation noise also scales with each
        # window's OWN energy: an extreme-amplitude window's exact match can
        # read thousands above zero, so widen its refinement cutoff
        # per-window or it would never be re-evaluated
        _, sig_t_raw = rolling_mean_sigma(np.where(np.isfinite(Tf), Tf, 0.0), m)
        noise = 3.0 * np.sqrt(np.finfo(np.float32).eps * m)
        margin = margin * scale + noise * (sig_t_raw + float(Qf.std()))

    refined = np.zeros(l, dtype=bool)
    if query_idx is not None:
        # mass() already fixed this entry (0, or inf under a non-finite
        # T_subseq_isfinite override); STUMPY thresholds with it in place
        refined[query_idx] = True

    def _refine_upto(cutoff) -> int:
        js = np.nonzero((D <= cutoff) & np.isfinite(D) & ~refined)[0]
        if js.size:
            D[js] = _refine_candidates(Qf, Tf, js, normalize, q_const, t_const)
            refined[js] = True
        return int(js.size)

    # Estimate the threshold on the float32 profile and refine everything
    # near it. A data-dependent threshold is then recomputed on the refined
    # profile (STUMPY derives it from an exact float64 profile — e.g. its
    # min(D) term shifts when refinement corrects the best match), and the
    # newly exposed band is refined in turn, until a round refines nothing:
    # the profile is then unchanged, so the threshold is a fixed point and
    # every entry it could admit has been re-evaluated.
    md = _threshold(D)
    for _ in range(_MAX_REFINE_ROUNDS):
        # a NaN threshold admits nothing by comparison, but STUMPY's greedy
        # loop (`D > atol + NaN` is False) then returns every finite entry;
        # refine everything so those entries carry float64 values too
        cutoff = np.inf if np.isnan(md) else md + atol + margin
        n_new = _refine_upto(cutoff)
        if fixed:
            break
        md = _threshold(D)
        if n_new == 0:
            break

    return _find_matches(
        D,
        excl_zone,
        max_distance=md,
        max_matches=max_matches,
        query_idx=query_idx,
        atol=atol,
    )
