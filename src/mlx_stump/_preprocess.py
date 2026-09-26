"""Input validation and float64 CPU-side preprocessing.

STUMPY computes everything in float64; Apple GPUs are float32-only. Normalized
search copies every raw window into a bounded local frame, centers it, and
divides by its own RMS in float64 before the float32 upload. Raw-distance
search instead uses one shared, scale-safe affine frame for both join series
and carries its rolling statistics in float64/float32 high-low form.

Semantics mirror STUMPY: float64 1-D input required, m >= 3, subsequences
containing NaN/inf get an infinite profile value and are never neighbors,
and a subsequence is "constant" when its rolling min equals its rolling max
(a window containing NaN is not constant).
"""

from __future__ import annotations

import inspect
import numbers
import sys
import warnings
from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np

# default of stumpy.config.STUMPY_EXCL_ZONE_DENOM
EXCL_ZONE_DENOM = 4


def excl_zone_denom():
    """The exclusion-zone denominator in effect for this call.

    STUMPY reads ``stumpy.config.STUMPY_EXCL_ZONE_DENOM`` at call time, and it
    is the only knob STUMPY offers for the trivial-match exclusion zone
    ``ceil(m / denom)``. Honour it whenever STUMPY has been imported so a
    mixed pipeline uses one zone, and fall back to STUMPY's default of 4
    otherwise. STUMPY is not a dependency: never import it here. The raw
    value is kept (STUMPY accepts any positive real, e.g. 2.5 or 0.5).
    """
    cfg = sys.modules.get("stumpy.config")
    denom = EXCL_ZONE_DENOM
    if cfg is not None:
        denom = getattr(cfg, "STUMPY_EXCL_ZONE_DENOM", EXCL_ZONE_DENOM)
    if (
        isinstance(denom, (bool, np.bool_))
        or not isinstance(denom, numbers.Real)
        or not denom > 0
        or not np.isfinite(denom)
    ):
        raise ValueError(
            f"`STUMPY_EXCL_ZONE_DENOM` must be a finite positive number but found {denom!r}."
        )
    return denom


def exclusion_zone(m: int, denom=None) -> int:
    """``ceil(m / denom)``, STUMPY's trivial-match exclusion-zone half width."""
    if denom is None:
        denom = excl_zone_denom()
    return int(np.ceil(m / denom))


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


def check_series(T, name: str, copy: bool = True) -> np.ndarray:
    """Validate a time series the way STUMPY does; return a float64 copy.

    Byte-swapped float64 (``'>f8'`` on Apple Silicon, e.g. read from FITS or
    HDF5) is float64 all the same: it is accepted and returned in native
    byte order. ``copy=False`` validates only and returns the input array
    itself (for callers that hand the series on to a function making its
    own copy); a byte-swapped input is still converted, so the result is
    always native.
    """
    T = np.asarray(T)
    if T.dtype.newbyteorder("=") != np.float64:
        raise TypeError(
            f"{np.float64} dtype expected but found {T.dtype} in {name}. "
            "Please change the input dtype with `.astype(np.float64)`."
        )
    if T.ndim != 1:
        raise ValueError(f"{name} is {T.ndim}-dimensional and must be 1-dimensional.")
    if copy or not T.dtype.isnative:
        return T.astype(np.float64, order="C")
    return T


def check_window_size(
    m,
    n: int | None = None,
    warn_n: int | None = None,
    excl_zone_denom=EXCL_ZONE_DENOM,
    stacklevel: int = 3,
) -> int:
    """Validate ``m``; with ``warn_n`` (self-joins), also emit STUMPY's
    advisory when the exclusion zone starves the central subsequence.

    ``stacklevel`` is the advisory's ``warnings.warn`` level; the default
    points at the caller of a public function that calls this directly."""
    if not np.issubdtype(type(m), np.integer):
        raise TypeError(f"`m` must be an integer but found {type(m)}.")
    m = int(m)
    if m < 3:
        raise ValueError("All window sizes must be greater than or equal to three.")
    if n is not None and m > n:
        raise ValueError(f"The window size must be less than or equal to {n}.")
    if warn_n is not None:
        excl_zone = exclusion_zone(m, excl_zone_denom)
        if (warn_n - m + 1) // 2 <= excl_zone:
            warnings.warn(
                f"The window size, 'm = {m}', may be too large and could lead to "
                "meaningless results. Consider reducing 'm' where necessary",
                stacklevel=stacklevel,
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


# windows per chunk of the O(n) rolling passes: bounds their int64 / float64
# temporaries to ~12 MiB instead of several copies of the series (n=1e7:
# rolling_isconstant peaks at 0.26x the series instead of 5x, same speed)
_ROLLING_CHUNK = 1 << 18


def rolling_isconstant(T: np.ndarray, m: int) -> np.ndarray:
    """A window is constant iff its min equals its max (NaN windows are not).

    NaN propagates through the min/max, but an all-inf window compares
    equal (inf == inf), so callers holding a series with inf AND the result
    with the finite-window mask; that equals the result on the series with
    inf replaced by NaN. Min/max are exact, so evaluating chunks of windows
    (each on its own ``m - 1`` overlap) changes nothing but the temporaries.
    """
    l = T.shape[0] - m + 1
    out = np.empty(l, dtype=bool)
    step = max(_ROLLING_CHUNK, m)  # keeps the (m - 1)-sample overlap <= 2x
    for s in range(0, l, step):
        e = min(s + step, l)
        seg = T[s : e + m - 1]
        lo = _rolling_reduce(seg, m, np.minimum, np.inf)
        hi = _rolling_reduce(seg, m, np.maximum, -np.inf)
        np.equal(lo, hi, out=out[s:e])
        del lo, hi
    return out


def rolling_isfinite(isfinite_pt: np.ndarray, m: int) -> np.ndarray:
    """True where the length-m window contains only finite values."""
    l = isfinite_pt.shape[0] - m + 1
    if np.all(isfinite_pt):
        return np.ones(l, dtype=bool)
    out = np.empty(l, dtype=bool)
    step = max(_ROLLING_CHUNK, m)
    for s in range(0, l, step):
        e = min(s + step, l)
        # running count of non-finite points over this chunk's windows only
        bad = np.zeros(e - s + m, dtype=np.int64)
        np.cumsum(~isfinite_pt[s : e + m - 1], out=bad[1:])
        np.equal(bad[m:], bad[:-m], out=out[s:e])
    return out


# byte bound on the float64 window copies held by one sigma-repair chunk
_SIGMA_REPAIR_BYTES = 1 << 25  # ~32 MiB
# repair every window whose variance is within this factor of its one-pass
# rounding bound: a window *at* K times the bound can still carry ~1/K
# relative variance error, so a bare small factor left percent-level sigma
# errors on windows just above it (e.g. ordinary noise windows crushed by
# global standardization when the series contains a huge-amplitude
# segment), corrupting both the float32 search and the reported profile
_SIGMA_REPAIR_HEADROOM = 1 << 20


def _rolling_sum_local(x: np.ndarray, w: int) -> np.ndarray:
    """``sum(x[j:j+w])`` for every window, built from sub-sums of its own terms.

    The series is cut into length-``w`` blocks. Window ``j = b*w + r`` is the
    suffix of block ``b`` from ``r`` plus the length-``r`` prefix of block
    ``b+1``; block-end prefixes are zeroed, so aligned windows take the
    suffix alone. A difference of whole-series cumulative sums instead
    carries the rounding of the running prefix, which grows with position:
    here every partial sum is part of the window itself, so the error is at
    most ~(w-1)*eps/2 of the window's own sum of magnitudes wherever the
    window lies. Windows only start in full blocks; the trailing partial
    block (if any) contributes prefixes alone, so nothing is padded.
    """
    n = x.shape[0]
    l = n - w + 1
    nfull = n // w
    split = nfull * w
    blocks = x[:split].reshape(nfull, w)
    suffix = np.empty((nfull, w))
    np.cumsum(blocks[:, ::-1], axis=1, out=suffix[:, ::-1])  # sum(blocks[b, r:])
    prefix = np.empty(n)
    np.cumsum(blocks, axis=1, out=prefix[:split].reshape(nfull, w))
    prefix[w - 1 : split : w] = 0.0
    np.cumsum(x[split:], out=prefix[split:])
    sums = suffix.reshape(-1)[:l]
    sums += prefix[w - 1 :]
    return sums


def _two_pass_repair(
    a: np.ndarray, w: int, idx: np.ndarray, mu: np.ndarray, var: np.ndarray
) -> None:
    """Recompute ``mu``/``var`` of windows ``idx`` directly from their values.

    Streams the float64 window copies in ``_SIGMA_REPAIR_BYTES`` chunks.
    """
    windows = np.lib.stride_tricks.sliding_window_view(a, w)
    chunk = max(1, _SIGMA_REPAIR_BYTES // (w * 8))
    for s in range(0, idx.size, chunk):
        rows = idx[s : s + chunk]
        wv = windows[rows]  # fancy indexing copies, so in-place is safe
        mu_exact = wv.mean(axis=1)
        mu[rows] = mu_exact
        wv -= mu_exact[:, None]
        var[rows] = np.einsum("ij,ij->i", wv, wv) / w


def rolling_mean_sigma(
    a: np.ndarray, w: int, known_constant: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Float64 rolling mean and standard deviation.

    Window sums come from block-local partial sums (see
    :func:`_rolling_sum_local`), so the one-pass variance ``s2/w - mu^2``
    of a window with sum of squares ``s2`` is off by at most
    ``1.5*eps*s2`` to first order (``eps/2*(s2 + 2|mu|*sum|x|)``, and
    ``|mu|*sum|x| <= s2`` by Cauchy-Schwarz), independent of the window's
    position in the series. That bound is only relatively large for
    small-variance windows (e.g. a flatlined sensor with tiny jitter, or any
    window whose variance global standardization crushed toward it), where
    ``E[x^2] - mu^2`` cancels. Windows whose computed variance is within
    ``_SIGMA_REPAIR_HEADROOM`` of the bound are therefore recomputed
    directly, two-pass, from the raw window values — an O(suspects * w)
    repair, streamed in byte-budgeted chunks — so every other window keeps
    a relative variance error of at most ~1/headroom (~1e-6).

    ``known_constant`` (optional, ``(n-w+1,)`` bool) marks windows whose min
    equals their max: their mean is any of their values and their variance
    is exactly 0, so they are written directly instead of being re-read (a
    constant series would otherwise "repair" every window it has).
    """
    # tiny values may square or scale into the subnormal range: harmless
    # for these statistics, so a caller's underflow policy must not trip
    with np.errstate(under="ignore"):
        s2 = _rolling_sum_local(a * a, w)  # the squared series dies here
        mu = _rolling_sum_local(a, w)
        mu /= w
        var = s2 / w
        for s in range(0, var.shape[0], _ROLLING_CHUNK):
            e = s + _ROLLING_CHUNK
            var[s:e] -= mu[s:e] * mu[s:e]
        np.maximum(var, 0.0, out=var)
        s2 *= 1.5 * np.finfo(np.float64).eps * _SIGMA_REPAIR_HEADROOM  # the scaled bound
    suspects = np.nonzero(var <= s2)[0]
    del s2
    if known_constant is not None:
        kc = np.nonzero(known_constant)[0]
        mu[kc] = a[kc]
        var[kc] = 0.0
        suspects = suspects[~known_constant[suspects]]
    if suspects.size:
        _two_pass_repair(a, w, suspects, mu, var)

    return mu, np.sqrt(var, out=var)


def split_float32(x: np.ndarray) -> np.ndarray:
    """``(..., 2)`` float32 ``[hi, lo]`` with ``hi + lo == x`` to ~float64.

    ``lo`` is the float64 residual of the float32 rounding, itself rounded
    to float32 (relative 6e-8 of a quantity already 6e-8 of ``x``). Written
    in chunks straight into the result, so the float64 temporaries stay
    bounded instead of costing ~3x the input.
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.empty(x.shape + (2,), dtype=np.float32)
    flat, pairs = x.reshape(-1), out.reshape(-1, 2)
    for s in range(0, flat.shape[0], _ROLLING_CHUNK):
        chunk = flat[s : s + _ROLLING_CHUNK]
        hi = chunk.astype(np.float32)
        pairs[s : s + _ROLLING_CHUNK, 0] = hi
        pairs[s : s + _ROLLING_CHUNK, 1] = chunk - hi  # float64 residual, cast once
    return out


def call_isconstant(func, T: np.ndarray, m: int, name: str):
    """Evaluate a callable constant-flag spec with STUMPY's contract.

    STUMPY calls ``func(a, w)`` on the series with inf replaced by NaN and
    rejects a function with any other required argument (extra arguments
    are curried with ``functools.partial``). The callable gets a private
    copy, so it cannot modify the caller's series. Returns its raw result.
    """
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):  # no introspectable signature: just call it
        params = {}
    required = {k for k, v in params.items() if v.default is inspect.Parameter.empty}
    incompatible = required - {"a", "w"}
    if incompatible:
        raise ValueError(
            f"Incompatible arguments {incompatible} found in `{name}`. Please provide "
            f"the custom function `{name}` with arguments `a`, a 1-D array, and `w`, "
            "the window size."
        )
    return func(np.where(np.isinf(T), np.nan, T), m)


def process_isconstant(T: np.ndarray, m: int, user_isconstant, name: str) -> np.ndarray:
    """Resolve a user-supplied isconstant spec against STUMPY's rules.

    ``T`` is the series as given (inf allowed). ``None`` detects the
    constant finite windows. An array spec only needs ``T``'s length. A
    callable is evaluated once through :func:`call_isconstant`; like an
    array, its result must be a boolean array of shape ``(l,)``. Flags on
    non-finite windows are returned as given (the caller switches them off
    with STUMPY's warning).
    """
    if user_isconstant is None:
        detected = rolling_isconstant(T, m)
        detected &= rolling_isfinite(np.isfinite(T), m)
        return detected
    if callable(user_isconstant):
        user_isconstant = call_isconstant(user_isconstant, T, m, name)
    isconstant = np.asarray(user_isconstant)
    l = T.shape[0] - m + 1
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
    center: float  # shared-frame offset (raw mode; 0 in normalized mode)
    scale: float  # shared-frame divisor (raw mode; 1 in normalized mode)
    Ts: np.ndarray | None  # standardized series (raw mode only)
    isfinite: np.ndarray  # (l,) window all-finite
    isconstant: np.ndarray  # (l,) window min == max
    mu: np.ndarray | None  # (l,) rolling mean of Ts (raw mode only)
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
    # sig_inv_mx and isconstant_mx exist in normalized mode only. The device
    # windows are already divided by their locally recomputed RMS there, so
    # sig_inv_mx is a 1/0 varying-window mask rather than a rolling inverse
    # sigma. Raw mode uses ssq_mx/mu_mx instead (no constant special case).
    sig_inv_mx: mx.array = field(repr=False, default=None)
    isfinite_mx: mx.array = field(repr=False, default=None)
    isconstant_mx: mx.array = field(repr=False, default=None)
    ssq_mx: mx.array = field(repr=False, default=None)
    mu_mx: mx.array = field(repr=False, default=None)  # (l, 2) float32 [hi, lo]
    # (l,) rolling sigma of Ts, kept only on request (raw mode, keep_sigma)
    sigma: np.ndarray | None = field(repr=False, default=None)

    def release_device(self) -> None:
        """Drop the device-side copies (the CPU arrays stay).

        Call once the GPU phase is over and before ``mx.clear_cache()``:
        arrays still referenced when the cache is cleared land in it when
        this object is garbage-collected later instead of being returned to
        the system with the rest of the search phase.
        """
        self.sig_inv_mx = self.isfinite_mx = self.isconstant_mx = None
        self.ssq_mx = self.mu_mx = None

    def release_search_arrays(self) -> None:
        """Drop CPU arrays needed only by the GPU search.

        The raw series plus finite/constant flags remain available for the
        float64 profile refinement. Calling this between the GPU sweep and
        refinement avoids retaining the raw-mode standardized series and
        rolling-stat arrays while a large object-dtype result is assembled.
        """
        self.Ts = self.mu = self.ssq = self.sigma = None


def preprocess_series(
    T: np.ndarray,
    m: int,
    *,
    normalize: bool = True,
    center: float | None = None,
    scale: float | None = None,
    isconstant=None,
    isconstant_name: str = "T_subseq_isconstant",
    keep_sigma: bool = False,
    stacklevel: int = 3,
) -> PreprocessedSeries:
    """Prepare one already-validated float64 series for the GPU engine.

    ``center``/``scale`` override the global standardization parameters; the
    non-normalized (aamp) path uses this to put both join series in one shared
    affine frame, which keeps their cross distances exactly invariant.
    Normalized search needs only the raw series and finite/constant masks:
    every window is centered and scaled locally by the engine, so no global
    series copy or rolling statistics are built or retained in that mode.
    ``isconstant`` may be a boolean array or a callable (see
    :func:`process_isconstant`). ``keep_sigma`` (raw mode) also retains the
    rolling sigma of ``Ts``. ``stacklevel`` is that of this function's
    warnings, so wrappers can point them at the user's call.
    """
    n = T.shape[0]
    l = n - m + 1

    isfinite_pt = np.isfinite(T)
    isfinite = rolling_isfinite(isfinite_pt, m)
    user_isconstant = isconstant is not None
    detected = None
    if not normalize or not user_isconstant:
        # windows whose min equals their max (NaN windows never qualify, and
        # the all-inf windows that compare equal are masked here): the
        # default constant flags, and the windows whose stats are known
        # exactly. Normalized mode with user flags never reads it.
        detected = rolling_isconstant(T, m)
        detected &= isfinite
    if user_isconstant:
        isconstant = process_isconstant(T, m, isconstant, isconstant_name)
    else:
        isconstant = detected
    fixed = isconstant & isfinite  # a window with NaN is never constant
    if user_isconstant and np.any(fixed != isconstant):
        warnings.warn(
            f"Subsequences located at indices {np.nonzero(fixed != isconstant)} "
            "contain one or more np.nan/np.inf and so their corresponding values "
            f"in `{isconstant_name}` have been automatically switched from True "
            "to False.",
            stacklevel=stacklevel,
        )
    isconstant = fixed

    if normalize:
        del isfinite_pt
        active = isfinite & ~isconstant
        return PreprocessedSeries(
            T=T,
            m=m,
            n=n,
            l=l,
            center=0.0,
            scale=1.0,
            Ts=None,
            isfinite=isfinite,
            isconstant=isconstant,
            mu=None,
            ssq=None,
            # converted while copying: exact 0/1, no 4-byte/window host copy
            sig_inv_mx=mx.array(active, dtype=mx.float32),
            isfinite_mx=mx.array(isfinite),
            isconstant_mx=mx.array(isconstant),
            ssq_mx=None,
            mu_mx=None,
        )

    if center is None or scale is None:
        c, s = stable_center_scale(T)
        center = c if center is None else center
        scale = s if scale is None else scale

    # Invalid windows are masked from every ordinary result.  Represent their
    # bad points by the affine frame's center (standardized zero), not raw
    # zero: on a huge-offset, small-spread series, raw zero would become an
    # enormous sentinel whose contribution poisons cumulative rolling stats
    # for otherwise finite windows long after the bad point has left them.
    Ts = apply_affine_frame(np.where(isfinite_pt, T, center), center, scale)
    del isfinite_pt

    mu, sigma = rolling_mean_sigma(Ts, m, known_constant=detected)
    sigma[isconstant] = 0.0

    # A user may deliberately mark a varying window as constant; we set its
    # sigma to zero above to implement that override, so it is not evidence
    # that global standardization lost the window's variation. Warn only for
    # windows that neither the data nor the resolved user flags call constant.
    lost_variation = isfinite & ~detected & ~isconstant & (sigma == 0.0)
    smallest = np.min(sigma, where=sigma > 0.0, initial=np.inf)
    precision_limited = np.any(lost_variation) or (
        smallest < 1e-13 * max(1.0, float(max(np.max(Ts), -np.min(Ts))))
    )
    del lost_variation
    if precision_limited:
        # e.g. a 1e17-amplitude segment next to unit noise: standardization
        # then re-rounds the noise below its own variation (float64 has ~16
        # digits total), so no downstream arithmetic can recover it
        warnings.warn(
            "The amplitude dynamic range of this series approaches the float64 "
            "standardization limit for raw-distance search; distances involving "
            "its smallest-variance windows may be unreliable.",
            stacklevel=stacklevel,
        )

    # centered sum of squares: the engine computes the non-normalized
    # distance as ||qc - tc||^2 + m*(mu_q - mu_t)^2 (windows centered
    # before the float32 cast), which kills the ssq_q + ssq_t - 2*QT
    # cancellation on mixed-scale data. (m * sigma) * sigma, as before: the
    # same rounding, one l-wide buffer
    ssq = m * sigma
    ssq *= sigma
    if not keep_sigma:
        sigma = None

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
        ssq=ssq,
        isfinite_mx=mx.array(isfinite),
        # rounded while copying, exactly like astype(float32): no host copy
        ssq_mx=mx.array(ssq, dtype=mx.float32),
        mu_mx=mx.array(split_float32(mu)),
        sigma=sigma,
    )
