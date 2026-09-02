"""`match`: all occurrences of a query in a series, nearest first."""

from __future__ import annotations

import warnings
from decimal import Decimal, localcontext

import numpy as np

from ._mass import _as_flag, mass
from ._preprocess import (
    EXCL_ZONE_DENOM,
    apply_affine_frame,
    center_rows_stable,
    process_isconstant,
    rolling_isfinite,
    rolling_mean_sigma,
    rowwise_l2_inplace,
    stable_center_scale,
)
from ._stump import _refine_chunk_rows

# refinement/threshold rounds for a data-dependent max_distance (the loop
# converges as soon as a round refines nothing new, typically in 2-3 rounds)
_MAX_REFINE_ROUNDS = 8

# The refinement below forms each z-normalized row with float64 pairwise
# reductions and then sums squared component differences.  For two exactly
# positive-affine, representable rows, the remaining discrepancy is solely
# roundoff from centering, scaling, and subtraction.  Eight ulps per normalized
# component is a conservative envelope for that operation chain, so its
# squared L2 envelope is ``(8 eps)^2 * m``.  This is deliberately many orders
# tighter than STUMPY's general-purpose 1e-14 squared P-norm snap: a genuinely
# near-affine row must retain its non-zero distance here.
_ZNORM_ROUNDOFF_D2_PER_SAMPLE = (8.0 * np.finfo(np.float64).eps) ** 2
# Bound the vectorized exact-translation proof's scratch arrays. Rows/cells
# that cannot fit its signed-int64 representation simply use the general
# streaming affine certificate below.
_TRANSLATION_FAST_BYTES = 1 << 24
_TRANSLATION_TILE_CELL_BYTES = 512


def _canonical_dyadic_parts(x: np.ndarray):
    """Vectorized exact ``sign * odd_mantissa * 2**power`` decomposition."""
    x = np.asarray(x, dtype=np.float64)
    fraction, exponent = np.frexp(np.abs(x))
    # frexp's fraction has at most 53 binary digits, so this multiplication
    # and integer conversion are exact for normal and subnormal binary64.
    mantissa = (fraction * np.float64(2**53)).astype(np.uint64)
    nonzero = mantissa != 0
    trailing = np.zeros(mantissa.shape, dtype=np.int16)
    lowest = np.zeros_like(mantissa)
    # Unsigned two's complement without an underflow warning: ~u + 1 is
    # representable whenever u is nonzero.
    lowest[nonzero] = mantissa[nonzero] & (
        np.bitwise_not(mantissa[nonzero]) + np.uint64(1)
    )
    trailing[nonzero] = np.log2(lowest[nonzero].astype(np.float64)).astype(np.int16)
    mantissa[nonzero] = np.right_shift(
        mantissa[nonzero], trailing[nonzero].astype(np.uint64)
    )
    power = exponent.astype(np.int16) - 53 + trailing
    sign = np.where(np.signbit(x), -1, 1).astype(np.int64)
    bits = np.zeros(mantissa.shape, dtype=np.int16)
    bits[nonzero] = (
        np.floor(np.log2(mantissa[nonzero].astype(np.float64))).astype(np.int16) + 1
    )
    return mantissa, power, sign, bits, nonzero


def _exact_translation_rows(Q: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Exactly certify stored rows satisfying ``W = Q + b`` in bounded tiles.

    Each exact binary64 difference is reduced to a signed odd integer times
    a power of two and compared with the row's first difference. Cells whose
    aligned integers do not fit safely in signed int64 leave their row false;
    the caller then uses the general constant-memory affine certificate.
    Thus false is conservative, while true is an exact proof. Both row and
    column tiling keep expression temporaries bounded for every legal window.
    """
    Q = np.asarray(Q, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    if W.ndim == 1:
        W = W[None, :]
    if W.ndim != 2 or Q.ndim != 1 or W.shape[1] != Q.size:
        raise ValueError("`W` rows must have the same length as `Q`.")
    out = np.zeros(W.shape[0], dtype=bool)
    if W.shape[0] == 0 or Q.size == 0:
        return out

    max_cells = max(1, _TRANSLATION_FAST_BYTES // _TRANSLATION_TILE_CELL_BYTES)
    # The public caller already selects finite rows, but keeping this helper
    # total avoids feeding non-finite values to frexp/as_integer_ratio when it
    # is exercised directly.
    for c0 in range(0, Q.size, max_cells):
        if not np.all(np.isfinite(Q[c0 : c0 + max_cells])):
            return out

    int64_max = np.iinfo(np.int64).max
    int16 = np.iinfo(np.int16)
    sentinel = int16.max

    # A row tile is also required when a direct/internal caller supplies more
    # rows than fit even at one column per tile.
    for r0 in range(0, W.shape[0], max_cells):
        r1 = min(W.shape[0], r0 + max_cells)
        rows = W[r0:r1]
        n_rows = rows.shape[0]
        active = np.isfinite(rows[:, 0])
        ref_num = np.zeros(n_rows, dtype=np.int64)
        ref_power = np.zeros(n_rows, dtype=np.int16)
        for r in np.flatnonzero(active):
            numerator, power = _canonical_dyadic_difference(rows[r, 0], Q[0])
            # The vectorized delta below is signed-int64. An oversized exact
            # reference is not a rejection; false deliberately selects the
            # unbounded-integer fallback.
            if abs(numerator) > int64_max or not (int16.min <= power <= int16.max):
                active[r] = False
            else:
                ref_num[r] = numerator
                ref_power[r] = power

        columns = max(1, max_cells // n_rows)
        for c0 in range(0, Q.size, columns):
            c1 = min(Q.size, c0 + columns)
            raw_w = rows[:, c0:c1]
            finite_cells = np.isfinite(raw_w)
            active &= np.all(finite_cells, axis=1)
            if not np.any(active):
                break
            # Invalid rows must not reach frexp. This remains tile-bounded.
            w_tile = raw_w if np.all(finite_cells) else np.where(finite_cells, raw_w, 0.0)

            qm, qp, qs, qbits, qnz = _canonical_dyadic_parts(Q[c0:c1])
            wm, wp, ws, wbits, wnz = _canonical_dyadic_parts(w_tile)
            q_power = np.where(qnz, qp, sentinel)[None, :]
            w_power = np.where(wnz, wp, sentinel)
            pmin = np.minimum(q_power, w_power)
            pmin = np.where((~qnz[None, :]) & (~wnz), 0, pmin)

            q_width = np.where(qnz[None, :], qbits[None, :] + qp[None, :] - pmin, 0)
            w_width = np.where(wnz, wbits + wp - pmin, 0)
            eligible = active[:, None] & (q_width <= 62) & (w_width <= 62)

            # Zero shifts for ineligible cells prevent oversized/negative
            # shifts. Eligible aligned values have magnitude < 2**62.
            q_shift = np.where(
                eligible & qnz[None, :], qp[None, :] - pmin, 0
            ).astype(np.uint64)
            w_shift = np.where(eligible & wnz, wp - pmin, 0).astype(np.uint64)
            qi = np.left_shift(qm[None, :], q_shift).astype(np.int64) * qs[None, :]
            wi = np.left_shift(wm, w_shift).astype(np.int64) * ws
            # The difference cannot reach int64_min, so magnitude/negation is
            # safe before canonicalizing it.
            delta = wi - qi
            nonzero = delta != 0
            magnitude = np.where(delta < 0, -delta, delta).astype(np.uint64)
            lowest = np.zeros_like(magnitude)
            lowest[nonzero] = magnitude[nonzero] & (
                np.bitwise_not(magnitude[nonzero]) + np.uint64(1)
            )
            trailing = np.zeros(delta.shape, dtype=np.int16)
            trailing[nonzero] = np.log2(lowest[nonzero].astype(np.float64)).astype(
                np.int16
            )
            odd = np.right_shift(magnitude, trailing.astype(np.uint64)).astype(
                np.int64
            )
            odd *= np.where(delta < 0, -1, 1)
            power = np.where(nonzero, pmin + trailing, 0)

            active &= np.all(
                eligible
                & (odd == ref_num[:, None])
                & (power == ref_power[:, None]),
                axis=1,
            )

        out[r0:r1] = active
    return out


def _canonical_dyadic_difference(x: float, y: float) -> tuple[int, int]:
    """Return exact ``(odd_numerator, power)`` for ``x - y``.

    Finite binary64 numbers are dyadic rationals.  Reducing each exact
    difference to ``odd_numerator * 2**power`` lets the affine certificate
    compare products without aligning an entire million-element window to
    one denominator (and without retaining a million Python big integers).
    Zero is represented canonically by ``(0, 0)``.
    """
    x_num, x_den = float(x).as_integer_ratio()
    y_num, y_den = float(y).as_integer_ratio()
    x_power = x_den.bit_length() - 1
    y_power = y_den.bit_length() - 1
    common_power = max(x_power, y_power)
    numerator = (x_num << (common_power - x_power)) - (
        y_num << (common_power - y_power)
    )
    if numerator == 0:
        return 0, 0
    lowest_bit = abs(numerator) & -abs(numerator)
    trailing = lowest_bit.bit_length() - 1
    return numerator >> trailing, -common_power + trailing


def _exact_positive_affine_streaming_row(
    Q: np.ndarray, W: np.ndarray, pivot: int
) -> bool:
    """Certify one non-identical affine row with constant auxiliary memory."""
    dq_num, dq_power = _canonical_dyadic_difference(Q[pivot], Q[0])
    dw_num, dw_power = _canonical_dyadic_difference(W[pivot], W[0])
    if dq_num * dw_num <= 0:  # zero or negative scale
        return False

    for qi, wi in zip(Q, W, strict=True):
        dqi_num, dqi_power = _canonical_dyadic_difference(qi, Q[0])
        dwi_num, dwi_power = _canonical_dyadic_difference(wi, W[0])
        if dqi_num == 0 or dwi_num == 0:
            if dqi_num != dwi_num:
                return False
            continue

        # All four numerators are odd, so each product is already canonical.
        # Equality therefore requires both its signed numerator and its
        # power-of-two exponent to agree; no potentially enormous shift is
        # needed for the comparison.
        if (
            dwi_num * dq_num != dqi_num * dw_num
            or dwi_power + dq_power != dqi_power + dw_power
        ):
            return False
    return True


def _exact_positive_affine_rows(Q: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Certify which stored rows satisfy ``W = a * Q + b`` for some ``a > 0``.

    The test is exact over the real values represented by the input
    binary64 numbers.  It does not infer affinity from a small residual:
    each float is converted to an integer at a common power-of-two scale,
    then the two point sets are checked for exact collinearity with a
    positive slope using Python's unbounded integers.  Consequently a
    one-ULP perturbation cannot be certified accidentally.

    This deliberately expensive check is only called for rows whose
    normalized squared distance is already inside the tiny float64
    roundoff envelope.  Bitwise-identical rows take a fast path.
    """
    Q = np.asarray(Q, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    if W.ndim == 1:
        W = W[None, :]
    out = np.zeros(W.shape[0], dtype=bool)
    finite = np.all(np.isfinite(W), axis=1) & np.all(np.isfinite(Q))
    if not np.any(finite):
        return out

    # Equality itself is an exact positive-affine relation (a=1, b=0) and
    # covers the common duplicate-window case without allocating big ints.
    same = finite & np.all(W == Q[None, :], axis=1)
    out[same] = True
    remaining = np.nonzero(finite & ~same)[0]
    if remaining.size == 0:
        return out

    pivot_candidates = np.flatnonzero(Q != Q[0])
    if pivot_candidates.size == 0:
        # A positive affine image of a constant row is any constant row.
        out[remaining] = np.all(W[remaining] == W[remaining, :1], axis=1)
        return out

    qb = int(pivot_candidates[0])
    translated = _exact_translation_rows(Q, W[remaining])
    out[remaining[translated]] = True
    remaining = remaining[~translated]
    if remaining.size == 0:
        return out

    for row_idx in remaining:
        out[row_idx] = _exact_positive_affine_streaming_row(
            Q, W[row_idx], qb
        )
    return out


def _high_precision_znorm_rows(
    Q: np.ndarray, W: np.ndarray, row_indices: np.ndarray | None = None
) -> np.ndarray:
    """Evaluate tiny, non-affine z-distances beyond binary64 resolution.

    Binary64 normalization can occasionally round a one-ULP perturbation
    away completely.  Once exact dyadic arithmetic has proved that a row is
    *not* affine, recompute that row with enough decimal precision to span
    the entire binary64 exponent range.  The result is finally rounded back
    to the public float64 dtype; a positive value below float64's range is
    represented by its smallest positive subnormal so it cannot turn into a
    false exact match.

    This is not a symbolic distance oracle: the non-zero magnitude is a
    high-precision approximation rounded to float64. Exact-zero
    classification comes solely from
    ``_exact_positive_affine_rows``.
    """
    Q = np.asarray(Q, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    if W.ndim == 1:
        W = W[None, :]
    rows = (
        np.arange(W.shape[0], dtype=np.int64)
        if row_indices is None
        else np.asarray(row_indices, dtype=np.int64)
    )
    m = Q.size
    out = np.empty(rows.size, dtype=np.float64)

    # Find the exponent span with O(m) scratch, not an O(rows*m)
    # concatenate/frexp pair. A single extreme row must still set enough
    # Decimal precision for the shared query representation below.
    min_exponent = np.iinfo(np.int16).max
    max_exponent = np.iinfo(np.int16).min
    for values in (Q, *(W[row] for row in rows)):
        nonzero = values[values != 0.0]
        if nonzero.size:
            exponents = np.frexp(nonzero)[1]
            min_exponent = min(min_exponent, int(exponents.min()))
            max_exponent = max(max_exponent, int(exponents.max()))
    exponent_span = 0 if min_exponent > max_exponent else max_exponent - min_exponent
    # Ordinary binary64 rows need only ~17 significant decimal digits.  A
    # mixed-exponent row can encode a much smaller non-affine residual; two
    # times its binary exponent span covers the cross-products used by the
    # exact collinearity predicate, plus 50 guard digits for reductions and
    # square roots.  Thus common rows use 50 digits, while even the full
    # subnormal-to-max-float span is bounded at ~1,313 digits rather than an
    # unconditional multi-thousand-digit calculation.
    decimal_precision = max(50, int(np.ceil(2.0 * exponent_span * np.log10(2.0))) + 50)
    with localcontext() as ctx:
        ctx.prec = decimal_precision
        md = Decimal(m)
        q0 = Decimal.from_float(float(Q[0]))
        qd = [Decimal.from_float(float(v)) - q0 for v in Q]
        qmean = sum(qd, Decimal(0)) / md
        qc = [v - qmean for v in qd]
        qsig = (sum((v * v for v in qc), Decimal(0)) / md).sqrt()
        for out_idx, row_idx in enumerate(rows):
            row = W[row_idx]
            w0 = Decimal.from_float(float(row[0]))
            wd = [Decimal.from_float(float(v)) - w0 for v in row]
            wmean = sum(wd, Decimal(0)) / md
            wc = [v - wmean for v in wd]
            wsig = (sum((v * v for v in wc), Decimal(0)) / md).sqrt()
            if qsig == 0 or wsig == 0:
                dist = Decimal(0) if qsig == wsig else md.sqrt()
            else:
                dist = sum(
                    ((qv / qsig - wv / wsig) ** 2 for qv, wv in zip(qc, wc, strict=True)),
                    Decimal(0),
                ).sqrt()
            value = float(dist)
            if dist > 0 and value == 0.0:
                value = np.nextafter(0.0, 1.0)
            out[out_idx] = value
    return out


def _refine_candidates(Q, T, js, normalize, q_const, t_const):
    """Float64 re-evaluation of the finite profile entries ``js``.

    The GPU profile is float32, so a perfect occurrence reads ~1e-3 instead
    of ~0 and a tight ``max_distance`` would silently drop it; re-evaluating
    every candidate near the threshold restores STUMPY's behavior. Returns
    the refined distances for ``js`` (same order).

    Candidates are processed in byte-budgeted chunks (``max_distance=inf``
    selects the whole profile, which must not materialize l*m float64
    windows at once). Every window is centered by its own fresh two-pass
    mean (after a scale-safe midpoint/max-deviation preconditioner) and
    scaled by its own freshly recomputed sigma: the squared distance is the sum of
    squared differences of the two z-normalized windows,
    cancellation-free down to 0. The `W@Q - m*mu_t*mu_q` form
    cancels catastrophically for near-constant windows at an offset, even
    in float64, and any form built on an externally supplied sigma leaves
    that sigma's rounding as a floor on perfect matches — so precomputed
    statistics passed to ``match`` are never used here (see ``mass``).
    ``q_const``/``t_const`` are the resolved constant flags (including any
    user overrides), applied exactly like the GPU path applies them.
    Unlike ``stump``, no broad P-norm zero-snap is applied.  A squared
    distance inside the float64 operation's own roundoff envelope is set to
    zero only after the raw stored rows are certified exactly
    positive-affine using dyadic-rational arithmetic.  Thus independently
    normalized affine windows read exactly 0.0 while even a one-ULP
    non-affine perturbation retains its non-zero distance.
    """
    js = np.asarray(js, dtype=np.int64)
    out = np.empty(js.size, dtype=np.float64)
    if js.size == 0:
        return out
    m = Q.shape[0]
    chunk = _refine_chunk_rows(m)
    if normalize:
        Wfull = np.lib.stride_tricks.sliding_window_view(T, m)
        # Put every raw row into its own bounded midpoint/range frame before
        # centering and squaring. This retains the large-offset protection of
        # first-element shifting while also preventing overflow/underflow for
        # uniformly huge/tiny finite units.
        Qw = Q.copy()[None, :]
        center_rows_stable(Qw)
        # ``np.sum`` uses a partial pairwise reduction along this contiguous
        # axis.  Besides being more accurate than the BLAS-like accumulation
        # selected by ``einsum`` for long rows, it keeps the norm of an exactly
        # scaled row proportional to within a few ulps.
        sig_q = float(np.sqrt(np.sum(Qw * Qw, axis=1)[0] / m))
        sig_inv_q = 0.0 if (q_const or sig_q == 0.0) else 1.0 / sig_q
        u = Qw[0] * sig_inv_q
        for s in range(0, js.size, chunk):
            idx = js[s : s + chunk]
            collapsed_non_affine = np.zeros(idx.size, dtype=bool)
            tiny_distances = np.empty(0, dtype=np.float64)
            W = Wfull[idx].astype(np.float64)
            tc = t_const[idx]
            center_rows_stable(W)
            sig_t = np.sqrt(np.sum(W * W, axis=1) / m)
            pos = (sig_t > 0.0) & ~tc
            sig_inv_t = np.where(pos, 1.0 / np.where(sig_t > 0.0, sig_t, 1.0), 0.0)
            W *= sig_inv_t[:, None]
            W -= u[None, :]
            d2 = np.sum(W * W, axis=1)
            del W
            roundoff = d2 <= _ZNORM_ROUNDOFF_D2_PER_SAMPLE * m
            if np.any(roundoff):
                certified = np.zeros(idx.size, dtype=bool)
                # Re-read only the rare suspect rows: ``W`` has been
                # normalized in place, while the certificate must inspect
                # the original stored binary64 values.
                roundoff_rows = np.flatnonzero(roundoff)
                raw_roundoff = Wfull[idx[roundoff_rows]]
                certified_local = _exact_positive_affine_rows(Q, raw_roundoff)
                certified[roundoff_rows] = certified_local
                # A positive binary64 sum already preserves strict-zero
                # semantics. Decimal is needed only when normalization erased
                # a proven non-affine perturbation completely.
                collapsed_local = ~certified_local & (d2[roundoff_rows] == 0.0)
                if np.any(collapsed_local):
                    collapsed_rows = roundoff_rows[collapsed_local]
                    collapsed_non_affine[collapsed_rows] = True
                    tiny_distances = _high_precision_znorm_rows(
                        Q, raw_roundoff, np.flatnonzero(collapsed_local)
                    )
                d2 = np.where(roundoff & certified, 0.0, d2)
                del raw_roundoff
            # zero sig_inv on either side means rho == 0 on the GPU; mirror
            # it, the constant-flag rules overwrite as needed
            d2 = np.where((sig_inv_t == 0.0) | (sig_inv_q == 0.0), 2.0 * m, d2)
            # A normalized Euclidean distance cannot exceed 2*sqrt(m).
            # Roundoff at rho=-1 can overshoot by one ulp and incorrectly
            # exclude a match at an exactly inclusive theoretical threshold.
            d = np.minimum(np.sqrt(d2), 2.0 * np.sqrt(m))
            if tiny_distances.size:
                d[collapsed_non_affine] = np.minimum(tiny_distances, 2.0 * np.sqrt(m))
            out[s : s + chunk] = np.where(q_const & tc, 0.0, np.where(q_const ^ tc, np.sqrt(m), d))
    else:
        # the engine computes on the zero-filled series; mirror it so a user
        # T_subseq_isfinite override cannot inject NaN into the profile
        Tf = np.where(np.isfinite(T), T, 0.0)
        Wfull = np.lib.stride_tricks.sliding_window_view(Tf, m)
        for s in range(0, js.size, chunk):
            idx = js[s : s + chunk]
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                diff = Wfull[idx].astype(np.float64) - Q[None, :]
            out[s : s + chunk] = rowwise_l2_inplace(diff)
    return out


def _apply_exclusion_zone(a: np.ndarray, idx: int, excl_zone: int, val: float) -> None:
    zone_start = max(0, idx - excl_zone)
    zone_stop = min(a.shape[-1], idx + excl_zone)
    a[zone_start : zone_stop + 1] = val


def _default_max_distance(D: np.ndarray) -> float:
    """Scale-safe ``max(mean(D) - 2*std(D), min(D))`` over finite values.

    Squaring raw distances in ``np.nanstd`` overflows under a harmless large
    change of units and underflows under a small one. Map finite distances to
    [-1, 1], evaluate the same population moments there, then map the chosen
    convex point back. Empty profiles return NaN, which the greedy selector
    already treats as admitting no non-finite entries.
    """
    values = np.asarray(D, dtype=np.float64)
    finite_mask = np.isfinite(values)
    all_finite = bool(np.all(finite_mask))
    finite = values if all_finite else values[finite_mask]
    del finite_mask
    if finite.size == 0:
        return float("nan")
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if lo == hi:
        return lo
    midpoint = lo + (hi - lo) * 0.5
    radius = max(abs(lo - midpoint), abs(hi - midpoint))
    # Keep only one O(n) scratch allocation. If non-finite values had to be
    # filtered, ``finite`` is already an independent compact copy and can be
    # reused in place. Otherwise subtracting creates the sole scratch array.
    if all_finite:
        bounded = np.subtract(finite, midpoint)
    else:
        bounded = finite
        bounded -= midpoint
    bounded /= radius
    bounded_mean = float(bounded.mean())
    bounded_min = float(bounded.min())
    bounded -= bounded_mean
    variance = float(np.dot(bounded, bounded) / bounded.size)
    bounded_std = float(np.sqrt(max(variance, 0.0)))
    chosen = max(bounded_mean - 2.0 * bounded_std, bounded_min)
    # Invert the actual affine frame. The rounded midpoint is not guaranteed
    # to put adjacent binary64 endpoints at exactly -1 and +1, so mapping as
    # though it did would move the threshold by an ulp.
    restored = midpoint + radius * chosen
    # The mathematically equivalent inverse can round a few ulps beyond an
    # endpoint after the bounded mean/std reductions. Preserve the defining
    # max(..., min(D)) guarantee and keep every finite cutoff inside the
    # observed distance range.
    return float(min(hi, max(lo, restored)))


def _find_matches(
    D: np.ndarray,
    excl_zone: int,
    max_distance=None,
    max_matches=None,
    query_idx=None,
    atol=1e-8,
) -> np.ndarray:
    """Greedy nearest-first selection with exclusion zones (STUMPY port)."""
    D = np.array(D, dtype=np.float64, copy=True)
    if max_distance is None:
        max_distance = _default_max_distance

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
    most 9; STUMPY calls it exactly once, on its float64 profile) —
    so the threshold that selects the matches comes from a profile that is
    float64-refined throughout the threshold band.
    ``normalize=False`` supports ``p=2.0`` only. Precomputed ``M_T``/``Σ_T``
    follow the ``mass`` contract. In normalized mode they are compatibility
    metadata: an infinite ``M_T`` marks its window non-finite, while finite
    statistics use the same repaired float64 rolling path as a no-stats call.
    In raw mode the pair is only
    shape-validated and is otherwise ignored. It never skips preprocessing.
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
        q_const = bool(np.min(Qf) == np.max(Qf))
    if normalize:
        T_nan = np.where(np.isinf(Tf), np.nan, Tf)
        with warnings.catch_warnings():
            # the mass() call above already resolved these flags and warned
            warnings.simplefilter("ignore")
            t_const = process_isconstant(
                T_nan, m, T_subseq_isconstant, "T_subseq_isconstant"
            )
        t_const &= rolling_isfinite(np.isfinite(Tf), m)
    else:
        # mass() already validated the compatibility control. Constants have
        # no special case in raw Euclidean refinement.
        t_const = np.zeros(l, dtype=bool)

    fixed = isinstance(max_distance, (int, float, np.integer, np.floating)) and not isinstance(
        max_distance, (bool, np.bool_)
    )

    def _threshold(D):
        if max_distance is None:
            return _default_max_distance(D)
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
        center, scale = stable_center_scale(finite)
        del finite
        # the engine's float32 cancellation noise also scales with each
        # window's OWN energy: an extreme-amplitude window's exact match can
        # read thousands above zero, so widen its refinement cutoff
        # per-window or it would never be re-evaluated
        # Use standardized zero for invalid points, as preprocessing does;
        # otherwise a raw-zero sentinel can contaminate rolling margin stats
        # long after a NaN leaves a huge-offset window. The affine transform
        # also repairs the rare opposite-sign max-float subtraction overflow.
        T_scaled = apply_affine_frame(np.where(np.isfinite(Tf), Tf, center), center, scale)
        Q_scaled = apply_affine_frame(Qf, center, scale)
        _, sig_t_scaled = rolling_mean_sigma(T_scaled, m)
        sig_t_scaled[~rolling_isfinite(np.isfinite(Tf), m)] = np.inf
        noise = 3.0 * np.sqrt(np.finfo(np.float32).eps * m)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            margin = scale * (margin + noise * (sig_t_scaled + float(Q_scaled.std())))
        del Q_scaled, T_scaled, sig_t_scaled

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
    # profile (STUMPY derives it from a float64 profile — e.g. its
    # min(D) term shifts when refinement corrects the best match), and the
    # newly exposed band is refined in turn, until a round refines nothing:
    # the profile is then unchanged, so the threshold is a fixed point and
    # every entry it could admit has been re-evaluated.
    md = _threshold(D)
    for _ in range(_MAX_REFINE_ROUNDS):
        # a NaN threshold admits nothing by comparison, but STUMPY's greedy
        # loop (`D > atol + NaN` is False) then returns every finite entry;
        # refine everything so those entries carry float64 values too
        # A raw-mode margin is per-window and deliberately infinite at an
        # invalid target window. With a threshold of -inf, -inf + inf is NaN;
        # that row is non-finite in D and cannot be selected anyway, so form
        # the vector silently and let the finite mask below reject it.
        with np.errstate(invalid="ignore"):
            cutoff = np.inf if np.isnan(md) else md + atol + margin
        n_new = _refine_upto(cutoff)
        if fixed:
            break
        if n_new == 0:
            break
        md = _threshold(D)

    return _find_matches(
        D,
        excl_zone,
        max_distance=md,
        max_matches=max_matches,
        query_idx=query_idx,
        atol=atol,
    )
