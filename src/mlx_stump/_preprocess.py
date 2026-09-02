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


def stable_center_scale(a: np.ndarray) -> tuple[float, float]:
    """Return a finite affine frame for finite values without squaring.

    The values are first mapped by a midpoint/max-deviation frame, then their
    mean and standard deviation are computed while bounded near one. Unlike
    raw ``mean/std``, this cannot overflow for huge finite values or
    underflow merely because all values use tiny units. Any positive affine
    frame is sufficient for the GPU arithmetic: normalized distances are
    invariant to it, while the absolute path multiplies by ``scale`` on
    return.
    """
    a = np.asarray(a, dtype=np.float64)
    finite_mask = np.isfinite(a)
    # Keep an already-finite input as a view.  In particular, non-normalized
    # joins deliberately assemble one shared finite frame; copying that whole
    # array again here would add a needless O(n) memory peak.  Genuinely
    # non-finite inputs are compacted exactly once.
    all_finite = bool(np.all(finite_mask))
    finite = a if all_finite else a[finite_mask]
    del finite_mask
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        if np.signbit(lo) == np.signbit(hi):
            midpoint = lo + (hi - lo) * 0.5
        else:
            midpoint = lo * 0.5 + hi * 0.5
        radius = max(abs(lo - midpoint), abs(hi - midpoint))
    if not np.isfinite(midpoint):  # finite endpoints imply a finite midpoint
        midpoint = 0.0
    if not np.isfinite(radius) or radius == 0.0:
        return float(midpoint), 1.0

    # Recover mean/std semantics in the bounded frame (some callers and
    # diagnostics rely on standardized data having mean 0 and sigma 1).
    # No raw value is squared until it is O(1).
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        if all_finite:
            bounded = np.subtract(finite, midpoint)
        else:
            # Boolean compaction already made `finite` a private writable
            # copy, so reuse it rather than holding two O(n) float64 arrays.
            bounded = finite
            bounded -= midpoint
        bounded /= radius
        mean_bounded = float(bounded.mean())
        bounded -= mean_bounded
        # Values are bounded near one, so a dot product cannot overflow or
        # underflow merely because the raw input units were extreme. Unlike
        # np.std, this does not allocate another n-wide centered temporary.
        std_bounded = float(np.sqrt((bounded @ bounded) / bounded.size))
        center = midpoint + radius * mean_bounded
        scale = radius * std_bounded
    if not np.isfinite(center):
        center = midpoint
    if not np.isfinite(scale) or scale == 0.0:
        scale = radius
    return float(center), float(scale)


def apply_affine_frame(a: np.ndarray, center: float, scale: float) -> np.ndarray:
    """Return ``(a - center) / scale`` without avoidable overflow.

    The direct subtraction is the accurate path for a large common offset,
    but it can overflow when two finite values have opposite signs near
    float64's maximum.  Division first is safe in exactly that case because
    the scale from :func:`stable_center_scale` is correspondingly large.
    Compute directly for every ordinary value and repair only the exceptional
    finite entries, preserving both precision and the common O(n) memory path.
    """
    a = np.asarray(a, dtype=np.float64)
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        out = np.subtract(a, center)
        out /= scale
        repair = np.isfinite(a) & ~np.isfinite(out)
        if np.any(repair):
            out[repair] = a[repair] / scale - center / scale
    return out


def center_rows_stable(a: np.ndarray) -> np.ndarray:
    """Center writable float64 rows in place after range preconditioning.

    A row is first mapped into a bounded midpoint/max-deviation frame and
    only then mean-centered.  Subsequent sums of squares therefore cannot
    overflow or underflow solely because the original units were huge or
    tiny.  Constant rows become exact zeros.
    """
    if a.ndim != 2:
        raise ValueError("`a` must be a 2-dimensional row matrix.")
    lo = np.min(a, axis=1)
    hi = np.max(a, axis=1)
    same_sign = np.signbit(lo) == np.signbit(hi)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        midpoint_same = lo + (hi - lo) * 0.5
        midpoint_cross = lo * 0.5 + hi * 0.5
        midpoint = np.where(same_sign, midpoint_same, midpoint_cross)
        scale = np.maximum(np.abs(lo - midpoint), np.abs(hi - midpoint))
        safe_scale = np.where((scale > 0.0) & np.isfinite(scale), scale, 1.0)
        a -= midpoint[:, None]
        a /= safe_scale[:, None]
    a -= a.mean(axis=1)[:, None]
    return a


def rowwise_l2_inplace(a: np.ndarray) -> np.ndarray:
    """Scale-safe Euclidean norm of writable float64 rows.

    The input is used as scratch space.  Finite norms are preserved in the
    original units; a mathematically unrepresentable result becomes ``inf``
    without emitting NumPy overflow/underflow warnings.
    """
    if a.ndim != 2:
        raise ValueError("`a` must be a 2-dimensional row matrix.")
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        scale = np.max(np.abs(a), axis=1)
        finite = np.isfinite(scale)
        safe_scale = np.where((scale > 0.0) & finite, scale, 1.0)
        a /= safe_scale[:, None]
        norm = np.sqrt(np.sum(a * a, axis=1)) * scale
    norm = np.where(scale == 0.0, 0.0, norm)
    return np.where(finite, norm, np.inf)


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


# byte bound on the float64 window copies held by one sigma-repair chunk
_SIGMA_REPAIR_BYTES = 1 << 25  # ~32 MiB
# repair every window whose variance is within this factor of the cumsum
# noise floor: a window *at* K times the floor still carries ~1/K relative
# variance error, so a bare factor-8 bound left percent-level sigma errors
# on windows just above it (e.g. ordinary noise windows crushed by global
# standardization when the series contains a huge-amplitude segment),
# corrupting both the float32 search and the reported profile
_SIGMA_REPAIR_HEADROOM = 1 << 20


def rolling_mean_sigma(
    a: np.ndarray, w: int, known_constant: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Float64 rolling mean and standard deviation.

    The O(n) cumulative-sum pass is exact enough everywhere except
    small-variance windows (e.g. a flatlined sensor with tiny jitter, or any
    window whose variance global standardization crushed toward the cumsum
    noise floor), where ``E[x^2] - mu^2`` cancels and leaves large relative
    error. Windows whose computed variance falls within
    ``_SIGMA_REPAIR_HEADROOM`` of a per-window error bound are therefore
    recomputed directly, two-pass, from the raw window values — an
    O(suspects * w) repair, streamed in byte-budgeted chunks, that caps the
    surviving relative variance error at ~1/headroom (~1e-6).

    ``known_constant`` (optional, ``(n-w+1,)`` bool) marks windows whose min
    equals their max: their mean is any of their values and their variance
    is exactly 0, so they are written directly instead of being re-read (a
    constant series would otherwise "repair" every window it has).
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
    if known_constant is not None:
        kc = np.nonzero(known_constant)[0]
        mu[kc] = a[kc]
        var[kc] = 0.0
        suspects = suspects[~known_constant[suspects]]
    if suspects.size:
        windows = np.lib.stride_tricks.sliding_window_view(a, w)
        chunk = max(1, _SIGMA_REPAIR_BYTES // (w * 8))
        for s in range(0, suspects.size, chunk):
            idx = suspects[s : s + chunk]
            wv = windows[idx]  # fancy indexing copies, so in-place is safe
            mu_exact = wv.mean(axis=1)
            mu[idx] = mu_exact
            wv -= mu_exact[:, None]
            var[idx] = np.einsum("ij,ij->i", wv, wv) / w

    return mu, np.sqrt(var)


def split_float32(x: np.ndarray) -> np.ndarray:
    """``(..., 2)`` float32 ``[hi, lo]`` with ``hi + lo == x`` to ~float64.

    ``lo`` is the float64 residual of the float32 rounding, itself rounded
    to float32 (relative 6e-8 of a quantity already 6e-8 of ``x``).
    """
    hi = np.asarray(x, dtype=np.float64).astype(np.float32)
    lo = (np.asarray(x, dtype=np.float64) - hi.astype(np.float64)).astype(np.float32)
    return np.stack([hi, lo], axis=-1)


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
    # distance needs the mean itself for its m*(mu_q - mu_t)^2 term. That
    # mean is carried as a float32 (hi, lo) pair: a single float32 holding
    # the shared frame's global offset has an ulp (3e-8 at |mu| ~ 0.5) far
    # coarser than the ~1e-5 spacing of neighboring window means on
    # mixed-scale data, and the difference of two such values decided
    # neighbors by rounding noise (5e-4 relative gaps vs STUMPY's aamp).
    # (hi_q - hi_t) is exact for nearby means (Sterbenz) and lo carries the
    # residual, so the device difference is accurate to float32 of the
    # difference itself.
    sig_inv_mx: mx.array = field(repr=False, default=None)
    isfinite_mx: mx.array = field(repr=False, default=None)
    isconstant_mx: mx.array = field(repr=False, default=None)
    ssq_mx: mx.array = field(repr=False, default=None)
    mu_mx: mx.array = field(repr=False, default=None)  # (l, 2) float32 [hi, lo]

    def release_device(self) -> None:
        """Drop the device-side copies (the CPU arrays stay).

        Call once the GPU phase is over and before ``mx.clear_cache()``:
        arrays still referenced when the cache is cleared land in it when
        this object is garbage-collected later (17 MiB after a
        ``mass(n=1e6)`` call, 86 MiB at n=5e6).
        """
        self.sig_inv_mx = self.isfinite_mx = self.isconstant_mx = None
        self.ssq_mx = self.mu_mx = None

    def release_search_arrays(self) -> None:
        """Drop CPU arrays needed only by the GPU search.

        The raw series plus finite/constant flags remain available for the
        float64 profile refinement. Calling this between the GPU sweep and
        refinement avoids retaining standardized series and rolling-stat
        arrays while a large object-dtype result is assembled.
        """
        self.Ts = self.mu = self.sig_inv = self.ssq = None


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
    # windows whose min equals their max (NaN windows never qualify): the
    # default constant flags, and the windows whose stats are known exactly
    detected = rolling_isconstant(T_nan, m)
    user_isconstant = isconstant is not None
    if user_isconstant:
        isconstant = process_isconstant(T_nan, m, isconstant, isconstant_name)
    else:
        isconstant = detected
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

    if center is None or scale is None:
        c, s = stable_center_scale(T)
        center = c if center is None else center
        scale = s if scale is None else scale

    # Invalid windows are masked from every ordinary result.  Represent their
    # bad points by the affine frame's center (standardized zero), not raw
    # zero: on a huge-offset, small-spread series, raw zero would become an
    # enormous sentinel whose contribution poisons cumulative rolling stats
    # for otherwise finite windows long after the bad point has left them.
    T_filled = np.where(isfinite_pt, T, center)
    Ts = apply_affine_frame(T_filled, center, scale)

    mu, sigma = rolling_mean_sigma(Ts, m, known_constant=detected)
    sigma[isconstant] = 0.0
    with np.errstate(divide="ignore"):
        sig_inv = np.where(sigma > 0.0, 1.0 / sigma, 0.0)

    pos = sigma[sigma > 0.0]
    # A user may deliberately mark a varying window as constant; we set its
    # sigma to zero above to implement that override, so it is not evidence
    # that global standardization lost the window's variation. Warn only for
    # windows that neither the data nor the resolved user flags call constant.
    lost_variation = isfinite & ~detected & ~isconstant & (sigma == 0.0)
    if np.any(lost_variation) or (
        pos.size and pos.min() < 1e-13 * max(1.0, float(np.max(np.abs(Ts))))
    ):
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
        sig_inv_mx=mx.array(sig_inv.astype(np.float32)),
        isfinite_mx=mx.array(isfinite),
        isconstant_mx=mx.array(isconstant),
        ssq_mx=None if ssq is None else mx.array(ssq.astype(np.float32)),
        mu_mx=mx.array(split_float32(mu)),
    )
