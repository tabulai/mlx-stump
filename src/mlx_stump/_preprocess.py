"""Input validation and float64 CPU-side preprocessing.

STUMPY computes everything in float64; Apple GPUs are float32-only. The
precision strategy starts here:

- the series is globally standardized before upload (exactly invariant for
  the z-normalized profile — the centered cross-covariance is bilinear, so
  per-series affine changes cancel even in AB-joins);
- per-window rolling mean and inverse standard deviation are computed on the
  CPU in float64 (O(n)) and only then cast to float32 for the GPU.

Semantics mirror STUMPY: float64 1-D input required, m >= 3, subsequences
containing NaN/inf get an infinite profile value and are never neighbors,
and a subsequence is "constant" when its rolling min equals its rolling max
(a window containing NaN is not constant).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np

# matches stumpy.config.STUMPY_EXCL_ZONE_DENOM
EXCL_ZONE_DENOM = 4


def check_series(T, name: str) -> np.ndarray:
    """Validate a time series the way STUMPY does; return a float64 copy."""
    T = np.asarray(T)
    if T.dtype != np.float64:
        raise TypeError(
            f"{np.float64} dtype expected but found {T.dtype} in {name}. "
            "Please change the input dtype with `.astype(np.float64)`."
        )
    if T.ndim != 1:
        raise ValueError(f"{name} is {T.ndim}-dimensional and must be 1-dimensional.")
    return T.copy()


def check_window_size(m, n: int | None = None, warn_n: int | None = None) -> int:
    """Validate ``m``; with ``warn_n`` (self-joins), also emit STUMPY's
    advisory when the exclusion zone starves the central subsequence."""
    if not np.issubdtype(type(m), np.integer):
        raise TypeError(f"`m` must be an integer but found {type(m)}.")
    m = int(m)
    if m < 3:
        raise ValueError("All window sizes must be greater than or equal to three.")
    if n is not None and m > n:
        raise ValueError(f"The window size must be less than or equal to {n}.")
    if warn_n is not None:
        excl_zone = int(np.ceil(m / EXCL_ZONE_DENOM))
        if (warn_n - m + 1) // 2 <= excl_zone:
            warnings.warn(
                f"The window size, 'm = {m}', may be too large and could lead to "
                "meaningless results. Consider reducing 'm' where necessary",
                stacklevel=3,
            )
    return m


def _rolling_reduce(a: np.ndarray, w: int, op: np.ufunc, fill: float) -> np.ndarray:
    """O(n) rolling window reduce (van Herk / two-pass block algorithm).

    NaNs propagate through np.minimum/np.maximum, so windows containing NaN
    reduce to NaN — exactly what constant detection needs.
    """
    n = a.shape[0]
    l = n - w + 1
    nblocks = -(-n // w)
    pad = nblocks * w - n
    ap = np.concatenate([a, np.full(pad, fill)]) if pad else a
    blocks = ap.reshape(nblocks, w)
    prefix = op.accumulate(blocks, axis=1).ravel()
    suffix = op.accumulate(blocks[:, ::-1], axis=1)[:, ::-1].ravel()
    return op(suffix[:l], prefix[w - 1 : w - 1 + l])


def rolling_isconstant(T: np.ndarray, m: int) -> np.ndarray:
    """A window is constant iff its min equals its max (NaN windows are not)."""
    lo = _rolling_reduce(T, m, np.minimum, np.inf)
    hi = _rolling_reduce(T, m, np.maximum, -np.inf)
    return lo == hi


def rolling_isfinite(isfinite_pt: np.ndarray, m: int) -> np.ndarray:
    """True where the length-m window contains only finite values."""
    bad = (~isfinite_pt).astype(np.int64)
    csum = np.zeros(bad.shape[0] + 1, dtype=np.int64)
    np.cumsum(bad, out=csum[1:])
    return (csum[m:] - csum[:-m]) == 0


_SIGMA_REPAIR_CHUNK = 1 << 14
# repair every window whose variance is within this factor of the cumsum
# noise floor: a window *at* K times the floor still carries ~1/K relative
# variance error, so a bare factor-8 bound left percent-level sigma errors
# on windows just above it (e.g. ordinary noise windows crushed by global
# standardization when the series contains a huge-amplitude segment),
# corrupting both the float32 search and the reported profile
_SIGMA_REPAIR_HEADROOM = 1 << 20


def rolling_mean_sigma(a: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Float64 rolling mean and standard deviation.

    The O(n) cumulative-sum pass is exact enough everywhere except
    small-variance windows (e.g. a flatlined sensor with tiny jitter, or any
    window whose variance global standardization crushed toward the cumsum
    noise floor), where ``E[x^2] - mu^2`` cancels and leaves large relative
    error. Windows whose computed variance falls within
    ``_SIGMA_REPAIR_HEADROOM`` of a per-window error bound are therefore
    recomputed exactly, two-pass, from the raw window values — an
    O(suspects * w) repair that caps the surviving relative variance error
    at ~1/headroom (~1e-6).
    """
    csum = np.zeros(a.shape[0] + 1)
    np.cumsum(a, out=csum[1:])
    mu = (csum[w:] - csum[:-w]) / w
    csq = np.zeros(a.shape[0] + 1)
    np.cumsum(a * a, out=csq[1:])
    var = (csq[w:] - csq[:-w]) / w - mu * mu
    np.maximum(var, 0.0, out=var)

    # cancellation bound: the cumsum difference errors scale with the prefix
    # magnitudes (csq is nondecreasing), the mu^2 term with |csum|
    eps = np.finfo(np.float64).eps
    bound = eps * (csq[w:] + 2.0 * np.abs(mu) * (np.abs(csum[w:]) + np.abs(csum[:-w]))) / w * 8.0
    suspects = np.nonzero(var <= bound * _SIGMA_REPAIR_HEADROOM)[0]
    if suspects.size:
        windows = np.lib.stride_tricks.sliding_window_view(a, w)
        for s in range(0, suspects.size, _SIGMA_REPAIR_CHUNK):
            idx = suspects[s : s + _SIGMA_REPAIR_CHUNK]
            wv = windows[idx]
            mu_exact = wv.mean(axis=1)
            mu[idx] = mu_exact
            var[idx] = ((wv - mu_exact[:, None]) ** 2).mean(axis=1)

    return mu, np.sqrt(var)


def process_isconstant(T_nan: np.ndarray, m: int, user_isconstant, name: str) -> np.ndarray:
    """Resolve a user-supplied isconstant spec against STUMPY's rules."""
    if user_isconstant is None:
        return rolling_isconstant(T_nan, m)
    if callable(user_isconstant):
        raise NotImplementedError(
            f"Callable `{name}` is not supported yet; pass a boolean array instead."
        )
    isconstant = np.asarray(user_isconstant)
    l = T_nan.shape[0] - m + 1
    if isconstant.dtype != np.bool_ or isconstant.shape != (l,):
        raise ValueError(f"`{name}` must be a boolean array of shape ({l},).")
    return isconstant.copy()


@dataclass
class PreprocessedSeries:
    """CPU (float64) and GPU (float32) views of one prepared series."""

    T: np.ndarray  # original float64 values, untouched
    m: int
    n: int
    l: int  # number of subsequences: n - m + 1
    center: float  # global standardization offset
    scale: float  # global standardization divisor
    Ts: np.ndarray  # standardized series, non-finite values zero-filled first
    isfinite: np.ndarray  # (l,) window all-finite
    isconstant: np.ndarray  # (l,) window min == max
    mu: np.ndarray  # (l,) float64 rolling mean of Ts
    sig_inv: np.ndarray  # (l,) float64 1/sigma of Ts windows (0 where constant)
    ssq: np.ndarray | None  # (l,) CENTERED sum of squares m*sigma^2 (normalize=False only)
    # device-side float32 copies. Windows are centered in float64 before
    # upload, so the GPU never re-derives the mean — but the non-normalized
    # distance needs the float32 mean itself for its m*(mu_q - mu_t)^2 term.
    Ts_mx: mx.array = field(repr=False, default=None)
    sig_inv_mx: mx.array = field(repr=False, default=None)
    isfinite_mx: mx.array = field(repr=False, default=None)
    isconstant_mx: mx.array = field(repr=False, default=None)
    ssq_mx: mx.array = field(repr=False, default=None)
    mu_mx: mx.array = field(repr=False, default=None)


def preprocess_series(
    T: np.ndarray,
    m: int,
    *,
    normalize: bool = True,
    center: float | None = None,
    scale: float | None = None,
    isconstant=None,
    isconstant_name: str = "T_subseq_isconstant",
) -> PreprocessedSeries:
    """Prepare one already-validated float64 series for the GPU engine.

    ``center``/``scale`` override the global standardization parameters; the
    non-normalized (aamp) path uses this to put both join series in one shared
    affine frame, which keeps their cross distances exactly invariant.
    """
    n = T.shape[0]
    l = n - m + 1

    isfinite_pt = np.isfinite(T)
    T_nan = np.where(np.isinf(T), np.nan, T)

    isfinite = rolling_isfinite(isfinite_pt, m)
    user_isconstant = isconstant is not None
    isconstant = process_isconstant(T_nan, m, isconstant, isconstant_name)
    fixed = isconstant & isfinite  # a window with NaN is never constant
    if user_isconstant and np.any(fixed != isconstant):
        warnings.warn(
            f"Subsequences located at indices {np.nonzero(fixed != isconstant)} "
            "contain one or more np.nan/np.inf and so their corresponding values "
            f"in `{isconstant_name}` have been automatically switched from True "
            "to False.",
            stacklevel=3,
        )
    isconstant = fixed

    T_filled = np.where(np.isnan(T_nan), 0.0, T_nan)

    if center is None or scale is None:
        finite_vals = T[isfinite_pt]
        c = float(finite_vals.mean()) if finite_vals.size else 0.0
        s = float(finite_vals.std()) if finite_vals.size else 1.0
        if not np.isfinite(s) or s == 0.0:
            s = 1.0
        center = c if center is None else center
        scale = s if scale is None else scale

    Ts = (T_filled - center) / scale

    mu, sigma = rolling_mean_sigma(Ts, m)
    sigma[isconstant] = 0.0
    with np.errstate(divide="ignore"):
        sig_inv = np.where(sigma > 0.0, 1.0 / sigma, 0.0)

    pos = sigma[sigma > 0.0]
    if pos.size and pos.min() < 1e-13 * max(1.0, float(np.max(np.abs(Ts)))):
        # e.g. a 1e17-amplitude segment next to unit noise: standardization
        # then re-rounds the noise below its own variation (float64 has ~16
        # digits total), so no downstream arithmetic can recover it
        warnings.warn(
            "The amplitude dynamic range of this series approaches the float64 "
            "standardization limit; distances involving its smallest-variance "
            "windows are unreliable.",
            stacklevel=3,
        )

    ssq = None
    if not normalize:
        # centered sum of squares: the engine computes the non-normalized
        # distance as ||qc - tc||^2 + m*(mu_q - mu_t)^2 (windows centered
        # before the float32 cast), which kills the ssq_q + ssq_t - 2*QT
        # cancellation on mixed-scale data
        ssq = m * sigma * sigma

    return PreprocessedSeries(
        T=T,
        m=m,
        n=n,
        l=l,
        center=float(center),
        scale=float(scale),
        Ts=Ts,
        isfinite=isfinite,
        isconstant=isconstant,
        mu=mu,
        sig_inv=sig_inv,
        ssq=ssq,
        Ts_mx=mx.array(Ts.astype(np.float32)),
        sig_inv_mx=mx.array(sig_inv.astype(np.float32)),
        isfinite_mx=mx.array(isfinite),
        isconstant_mx=mx.array(isconstant),
        ssq_mx=None if ssq is None else mx.array(ssq.astype(np.float32)),
        mu_mx=mx.array(mu.astype(np.float32)),
    )
