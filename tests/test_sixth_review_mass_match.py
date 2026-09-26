"""Sixth review round: preprocessing, ``mass`` and ``match``.

1. ``match`` read ``Q`` (``np.isnan``, ``Q.shape[-1]``) before validating
   it: a 0-d query raised IndexError and an object query NumPy's ufunc
   error, and a float32 query holding NaN (or an int64 ``T``) raised
   ValueError where STUMPY's dtype check raises TypeError first. Both series
   are now validated up front, ``T`` without an extra copy.
2. A callable ``max_distance`` returning a size-1 array (e.g.
   ``np.nanpercentile(D, [1])``), which STUMPY accepts, raised TypeError.
   Size-1 ndarrays are unwrapped; ``None`` and lists are still rejected.
3. ``_find_matches`` ran one full ``argmin`` per accepted match, O(l·k): 23 s
   of a 23 s ``match`` at n=1e6, m=12. Beyond eight matches it now walks the
   candidates in stable sorted order (trimmed by a partition bound for a
   finite ``max_matches``), which is provably the same sequence; STUMPY's
   loop stays for NaN/-inf profiles and small ``max_matches``. A verbatim
   copy of the old loop is the oracle below.
4. Raw ``match`` recomputed the target's rolling sigma for its refinement
   margin without the known-constant mask, re-reading every flat window
   (O(l·m)), and repeated the sigma repair of near-flat windows that
   ``mass`` had already done. It now reuses ``mass``'s sigma for the
   identical frame.
5. A fixed threshold with ``max_matches=k`` refined the whole profile in
   float64 (all l rows for ``max_distance=np.inf``). Only windows that can
   be among the k greedy picks are refined now; the result is bitwise the
   same prefix.
6. Raw-mode rolling sums were differences of whole-series cumulative sums,
   whose rounding grows with position: 98% of the windows of a long random
   walk were "repaired" two-pass (O(n·m)), and the documented ~1e-6
   variance-error cap did not hold. Sums are block-local now, with a
   position-independent bound that makes the cap true.
7. Preprocessing's O(n) temporaries (int64 cumsums, van Herk copies, an
   inf->NaN copy of the series, unused raw-mode ``sig_inv``/``sig_inv_mx``/
   ``isconstant_mx``, and a second flag resolution in ``match``) peaked at
   5-11x the series. The masks are chunked and the dead arrays gone, with
   bitwise-identical fields (checked against verbatim copies of the old
   helpers).
8. Byte-swapped float64 (``'>f8'``) was rejected; it is converted to a
   native copy.
9. Callable ``T_subseq_isconstant``/``T_A_``/``T_B_``/``Q_subseq_isconstant``
   raised NotImplementedError. They follow STUMPY's contract now (``f(a,
   w)`` on the series with inf as NaN), evaluated once per call.
10. ``mass_absolute`` and ``aamp_match`` exist with STUMPY's exact
    signatures (``aamp_match`` keeps ``D[query_idx]`` at its true distance,
    as STUMPY's does), warnings point at the caller's line, and the public
    signatures are annotated.
"""

from __future__ import annotations

import inspect
import math
import tracemalloc
import typing
import warnings

import numpy as np
import pytest

import mlx_stump
import mlx_stump._match as match_mod
import mlx_stump._preprocess as prep_mod
from mlx_stump._match import _apply_exclusion_zone, _default_max_distance, _find_matches
from mlx_stump._preprocess import (
    PreprocessedSeries,
    _rolling_reduce,
    _rolling_sum_local,
    apply_affine_frame,
    check_series,
    preprocess_series,
    rolling_isconstant,
    rolling_isfinite,
    rolling_mean_sigma,
    split_float32,
    stable_center_scale,
)

from .conftest import (
    DATASETS,
    assert_dist_profiles_close,
    assert_profile_close,
    tie_tolerance,
)

stumpy = pytest.importorskip("stumpy")


def _walk(n, seed):
    return np.random.default_rng(seed).standard_normal(n).cumsum()


def _assert_same_matches(got, ref):
    """Identical rows, element types and bits (-0.0 is not 0.0)."""
    assert got.dtype == ref.dtype == object
    assert got.shape == ref.shape
    for g, r in zip(got.ravel(), ref.ravel(), strict=True):
        assert type(g) is type(r)
        if isinstance(r, float):
            assert np.float64(g).tobytes() == np.float64(r).tobytes(), (g, r)
        else:
            assert g == r


# ---------------------------------------------------------------- 1. match validates Q
def test_match_validates_query_before_reading_it():
    T = _walk(200, seed=1)
    with pytest.raises(ValueError, match="0-dimensional"):
        mlx_stump.match(np.float64(1.0), T)
    with pytest.raises(TypeError, match="float64"):
        mlx_stump.match(np.array([1.0, 2.0, 3.0], dtype=object), T)
    Q32 = T[:10].astype(np.float32)
    Q32[0] = np.nan
    with pytest.raises(TypeError, match="float32"):
        mlx_stump.match(Q32, T)
    with pytest.raises(TypeError):
        stumpy.match(Q32, T)  # the exception type STUMPY raises too
    Q = T[:10].copy()
    Q[0] = np.nan
    with pytest.raises(TypeError, match="int64"):
        mlx_stump.match(Q, np.arange(200))
    with pytest.raises(ValueError, match="illegal"):
        mlx_stump.match(Q, T)


def test_match_does_not_copy_the_series_to_validate_it(monkeypatch):
    calls = []
    real = match_mod.check_series

    def spy(T, name, copy=True):
        calls.append((name, copy))
        return real(T, name, copy)

    monkeypatch.setattr(match_mod, "check_series", spy)
    T = _walk(300, seed=2)
    mlx_stump.match(T[10:30].copy(), T, max_matches=2)
    assert calls == [("Q", True), ("T", False)]
    assert check_series(T, "T", copy=False) is T


# ------------------------------------------------- 2. size-1 array thresholds
@pytest.mark.parametrize(
    "threshold",
    [
        lambda D: np.nanpercentile(D, [1]),
        lambda D: np.quantile(D, [0.01]),
        lambda D: np.array([[3.0]]),
    ],
    ids=["nanpercentile", "quantile", "array_1x1"],
)
def test_match_callable_threshold_may_return_a_size1_array(threshold):
    T = np.random.default_rng(0).standard_normal(5000).cumsum()
    Q = T[100:150].copy()
    got = mlx_stump.match(Q, T, max_distance=threshold)
    ref = stumpy.match(Q, T, max_distance=threshold)
    assert got.shape == ref.shape and got.shape[0] >= 1
    np.testing.assert_array_equal(got[:, 1].astype(np.int64), ref[:, 1].astype(np.int64))
    np.testing.assert_allclose(got[:, 0].astype(float), ref[:, 0].astype(float), atol=1e-6)


@pytest.mark.parametrize("bad", [lambda D: None, lambda D: [3.0], lambda D: np.array([1.0, 2.0])])
def test_match_callable_threshold_rejects_non_scalars(bad):
    T = _walk(500, seed=3)
    with pytest.raises(TypeError):
        mlx_stump.match(T[10:60].copy(), T, max_distance=bad)


# ------------------------------------------------ 3. sorted-walk selection
def _legacy_find_matches(
    D, excl_zone, max_distance=None, max_matches=None, query_idx=None, atol=1e-8
):
    """Verbatim copy of the pre-round-6 ``_find_matches`` (STUMPY's loop)."""
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


def _fuzz_profile(rng):
    l = int(rng.integers(1, 300))
    kind = int(rng.integers(0, 4))
    if kind == 0:
        D = rng.integers(0, 5, l).astype(np.float64)  # heavy ties, many zeros
    elif kind == 1:
        D = np.round(rng.random(l) * 3, 1)
    elif kind == 2:
        D = rng.random(l)
    else:
        D = np.abs(rng.standard_normal(l)) * 10.0 ** rng.integers(-3, 4)
    zeros = D == 0.0
    D[zeros & (rng.random(l) < 0.5)] = -0.0
    D[rng.random(l) < 0.05 * rng.integers(0, 3)] = np.inf
    if rng.random() < 0.1:
        D[rng.random(l) < 0.02] = np.nan
    if rng.random() < 0.1:
        D[rng.random(l) < 0.02] = -np.inf
    return D


def _median_threshold(D):
    finite = D[np.isfinite(D)]
    return float(np.median(finite)) if finite.size else float("nan")


def test_sorted_walk_equals_stumpy_loop_exactly():
    rng = np.random.default_rng(2026)
    thresholds = [None, np.nan, np.inf, -np.inf, "fixed", _median_threshold]
    counts = [None, 0, 1, 2.5, 8, 9, 50, np.inf]
    walked = 0
    for _ in range(3000):
        D = _fuzz_profile(rng)
        l = D.shape[0]
        md = thresholds[int(rng.integers(0, len(thresholds)))]
        if isinstance(md, str):
            finite = D[np.isfinite(D)]
            md = float(np.quantile(finite, rng.random())) if finite.size else 1.0
        mm = counts[int(rng.integers(0, len(counts)))]
        atol = [0.0, 1e-8, 0.5][int(rng.integers(0, 3))]
        ez = [0, 1, 3, 10][int(rng.integers(0, 4))]
        qi = None if rng.random() < 0.6 else int(rng.integers(0, l))
        kwargs = dict(max_distance=md, max_matches=mm, query_idx=qi, atol=atol)
        ref = _legacy_find_matches(D, ez, **kwargs)
        got = _find_matches(D, ez, **kwargs)
        _assert_same_matches(got, ref)
        walked += (mm is None or mm > 8) and bool(np.all(D > -np.inf))
    assert walked > 1000  # the new path, not just the kept loop, was compared


def test_sorted_walk_handles_huge_and_nan_match_counts():
    D = np.random.default_rng(5).random(400)
    for mm in (10**400, np.nan):
        _assert_same_matches(
            _find_matches(D, 3, max_distance=np.inf, max_matches=mm),
            _legacy_find_matches(D, 3, max_distance=np.inf, max_matches=mm),
        )


# ------------------------------------------- 4. raw margin reuses mass's sigma
def test_raw_match_reuses_mass_sigma_and_never_rereads_flat_windows(monkeypatch):
    rng = np.random.default_rng(11)
    n, m = 20_000, 50
    T = np.zeros(n)
    for start in range(500, n - 200, 2000):  # ~5% active, the rest exactly flat
        T[start : start + 100] = rng.standard_normal(100).cumsum()
    T[7000] = np.nan
    Q = T[4500:4550].copy()

    sigma_calls = []
    real_sigma = prep_mod.rolling_mean_sigma

    def sigma_spy(a, w, known_constant=None):
        sigma_calls.append(known_constant is not None)
        return real_sigma(a, w, known_constant)

    repaired = []
    real_repair = prep_mod._two_pass_repair

    def repair_spy(a, w, idx, mu, var):
        repaired.append(idx.copy())
        return real_repair(a, w, idx, mu, var)

    monkeypatch.setattr(prep_mod, "rolling_mean_sigma", sigma_spy)
    monkeypatch.setattr(prep_mod, "_two_pass_repair", repair_spy)
    got = mlx_stump.match(Q, T, normalize=False, max_matches=5)

    # one rolling pass (mass's preprocessing), with the known-constant mask
    assert sigma_calls == [True]
    flat = rolling_isconstant(T, m) & rolling_isfinite(np.isfinite(T), m)
    assert flat.mean() > 0.8
    rows = np.concatenate(repaired) if repaired else np.zeros(0, dtype=np.int64)
    assert not np.any(flat[rows])

    ref = stumpy.match(Q, T, normalize=False, max_matches=5)
    np.testing.assert_array_equal(got[:, 1].astype(np.int64), ref[:, 1].astype(np.int64))
    np.testing.assert_allclose(got[:, 0].astype(float), ref[:, 0].astype(float), atol=1e-9)


# ------------------------------------------------ 5. top-k refinement band
@pytest.mark.parametrize("normalize", [True, False])
def test_topk_fixed_threshold_is_the_bitwise_prefix_and_refines_a_band(monkeypatch, normalize):
    T = _walk(20_000, seed=12)
    Q = T[5_000:5_064].copy()
    l = T.size - Q.size + 1
    full = mlx_stump.match(Q, T, max_distance=np.inf, normalize=normalize)
    assert full.shape[0] > 100

    rows = []
    real = match_mod._refine_candidates

    def spy(Q, T, js, *args):
        rows.append(len(js))
        return real(Q, T, js, *args)

    monkeypatch.setattr(match_mod, "_refine_candidates", spy)
    for k in (1, 4, 20):
        rows.clear()
        top = mlx_stump.match(Q, T, max_distance=np.inf, max_matches=k, normalize=normalize)
        _assert_same_matches(top, full[:k])
        assert sum(rows) < l // 10, (k, sum(rows))
    # a huge integer count is compared, never converted to float
    _assert_same_matches(
        mlx_stump.match(Q, T, max_distance=np.inf, max_matches=10**400, normalize=normalize),
        full,
    )


def test_topk_band_with_a_forced_first_pick():
    T = _walk(8_000, seed=13)
    Q = T[1_000:1_040].copy() + 0.05  # a mismatched query window: forced first
    full = mlx_stump.aamp_match(Q, T, max_distance=np.inf, query_idx=3_000)
    for k in (1, 3, 9):
        top = mlx_stump.aamp_match(Q, T, max_distance=np.inf, max_matches=k, query_idx=3_000)
        _assert_same_matches(top, full[:k])
    with pytest.warns(UserWarning, match="query_idx"):
        zq = mlx_stump.match(Q, T, max_distance=np.inf, query_idx=3_000)
    with pytest.warns(UserWarning, match="query_idx"):
        top = mlx_stump.match(Q, T, max_distance=np.inf, max_matches=3, query_idx=3_000)
    _assert_same_matches(top, zq[:3])
    assert zq[0, 1] == 3_000 and zq[0, 0] == 0.0


# ---------------------------------------------- 6. block-local rolling sums
EDGE_SHAPES = [
    (1, 1), (5, 1), (5, 5), (6, 5), (7, 3), (9, 3), (10, 4), (11, 10),
    (100, 99), (100, 100), (101, 10), (257, 16), (1000, 7), (1024, 1024), (1025, 512),
]  # fmt: skip


@pytest.mark.parametrize(("n", "w"), EDGE_SHAPES)
def test_block_local_sums_agree_with_fsum(n, w):
    rng = np.random.default_rng(n * 7919 + w)
    ints = rng.integers(-1000, 1000, n).astype(np.float64)
    exact = np.array([math.fsum(ints[j : j + w]) for j in range(n - w + 1)])
    np.testing.assert_array_equal(_rolling_sum_local(ints, w), exact)  # exact sub-sums

    x = rng.standard_normal(n) * 10.0 ** rng.integers(-5, 6, n)
    ref = np.array([math.fsum(x[j : j + w]) for j in range(n - w + 1)])
    mag = np.array([math.fsum(np.abs(x[j : j + w])) for j in range(n - w + 1)])
    err = np.abs(_rolling_sum_local(x, w) - ref)
    # every partial sum is part of the window: error bounded by its own magnitudes
    assert np.all(err <= max(w - 1, 1) * np.finfo(float).eps * mag)


def test_surviving_variance_error_is_within_the_headroom_cap():
    """A unit-noise prefix followed by an alternating block whose variance
    sits at 1.5x the old position-dependent repair threshold: the old
    cumsum differences left it 1.4e-4 relative variance error."""
    rng = np.random.default_rng(7)
    n0, w = 1_000_000, 10_000
    base = rng.standard_normal(n0)
    H = prep_mod._SIGMA_REPAIR_HEADROOM
    old_threshold = 8 * np.finfo(float).eps * float(np.sum(base * base)) / w * H
    alt = np.sqrt(old_threshold) * 1.5 * np.where(np.arange(4 * w) % 2 == 0, 1.0, -1.0)
    a = np.concatenate([base, alt])
    mu, sigma = rolling_mean_sigma(a, w)
    windows = np.lib.stride_tricks.sliding_window_view(a, w)
    rows = np.arange(n0 - 2 * w, a.size - w + 1, 97)
    W = windows[rows]
    c = W - W.mean(axis=1)[:, None]
    c -= c.mean(axis=1)[:, None]  # second correction pass
    var = np.einsum("ij,ij->i", c, c) / w
    rel = np.abs(sigma[rows] ** 2 - var) / var
    assert rel.max() <= 1.0 / H


def test_long_random_walk_needs_almost_no_sigma_repair(monkeypatch):
    repaired = []
    real = prep_mod._two_pass_repair

    def spy(a, w, idx, mu, var):
        repaired.append(idx.size)
        return real(a, w, idx, mu, var)

    monkeypatch.setattr(prep_mod, "_two_pass_repair", spy)
    T = _walk(3_000_000, seed=0)
    c, s = stable_center_scale(T)
    rolling_mean_sigma(apply_affine_frame(T, c, s), 100)
    # the global-prefix bound repaired 39.5% of these windows (1,186,301)
    assert sum(repaired) <= 50


# --------------------------------------------------- 7. lean preprocessing
def _legacy_rolling_isconstant(T, m):
    """Verbatim copy of the pre-round-6 helper (called on T with inf->NaN)."""
    lo = _rolling_reduce(T, m, np.minimum, np.inf)
    hi = _rolling_reduce(T, m, np.maximum, -np.inf)
    return lo == hi


def _legacy_rolling_isfinite(isfinite_pt, m):
    """Verbatim copy of the pre-round-6 helper."""
    bad = (~isfinite_pt).astype(np.int64)
    csum = np.zeros(bad.shape[0] + 1, dtype=np.int64)
    np.cumsum(bad, out=csum[1:])
    return (csum[m:] - csum[:-m]) == 0


def _legacy_split_float32(x):
    """Verbatim copy of the pre-round-6 helper."""
    hi = np.asarray(x, dtype=np.float64).astype(np.float32)
    lo = (np.asarray(x, dtype=np.float64) - hi.astype(np.float64)).astype(np.float32)
    return np.stack([hi, lo], axis=-1)


def test_chunked_split_float32_is_unchanged(monkeypatch):
    rng = np.random.default_rng(3)
    x = rng.standard_normal(1001) * 10.0 ** rng.integers(-40, 40, 1001)
    x[:4] = [0.0, -0.0, 1e300, -np.inf]
    monkeypatch.setattr(prep_mod, "_ROLLING_CHUNK", 7)
    with np.errstate(over="ignore", invalid="ignore"):
        for arr in (x, x[:1], np.array([x[5]])):
            got, ref = split_float32(arr), _legacy_split_float32(arr)
            assert got.dtype == ref.dtype and got.shape == ref.shape
            assert got.tobytes() == ref.tobytes()


def _legacy_preprocess_fields(T, m, *, normalize, center, scale, isconstant):
    """The pre-round-6 ``preprocess_series`` body with its own mask helpers
    (only ``rolling_mean_sigma`` is today's, which item 6 changed on
    purpose); returns the fields and the warnings it emitted."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        isfinite_pt = np.isfinite(T)
        T_nan = np.where(np.isinf(T), np.nan, T)
        isfinite = _legacy_rolling_isfinite(isfinite_pt, m)
        detected = _legacy_rolling_isconstant(T_nan, m)
        user = isconstant is not None
        isconstant = np.asarray(isconstant).copy() if user else detected
        fixed = isconstant & isfinite
        if user and np.any(fixed != isconstant):
            warnings.warn(f"switched {np.nonzero(fixed != isconstant)}", stacklevel=1)
        isconstant = fixed
        out = dict(T=T, isfinite=isfinite, isconstant=isconstant)
        if normalize:
            active = isfinite & ~isconstant
            out.update(
                center=0.0,
                scale=1.0,
                sig_inv_mx=active.astype(np.float32),
                isfinite_mx=isfinite,
                isconstant_mx=isconstant,
            )
            return out, [str(r.message) for r in rec]
        if center is None or scale is None:
            c, s = stable_center_scale(T)
            center = c if center is None else center
            scale = s if scale is None else scale
        T_filled = np.where(isfinite_pt, T, center)
        Ts = apply_affine_frame(T_filled, center, scale)
        mu, sigma = rolling_mean_sigma(Ts, m, known_constant=detected)
        sigma[isconstant] = 0.0
        pos = sigma[sigma > 0.0]
        lost_variation = isfinite & ~detected & ~isconstant & (sigma == 0.0)
        if np.any(lost_variation) or (
            pos.size and pos.min() < 1e-13 * max(1.0, float(np.max(np.abs(Ts))))
        ):
            warnings.warn("dynamic range", stacklevel=1)
        ssq = m * sigma * sigma
        out.update(
            center=float(center),
            scale=float(scale),
            Ts=Ts,
            mu=mu,
            ssq=ssq,
            isfinite_mx=isfinite,
            ssq_mx=ssq.astype(np.float32),
            mu_mx=_legacy_split_float32(mu),
        )
    return out, [str(r.message) for r in rec]


def _extra_series(n, seed):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(n)
    a[rng.integers(0, n, 5)] = np.inf
    a[rng.integers(0, n, 5)] = -np.inf
    a[rng.integers(0, n, 3)] = np.nan
    b = np.full(n, 3.0)
    b[n // 3 : n // 3 + 40] = np.inf  # an all-inf run longer than m
    b[2 * n // 3 : 2 * n // 3 + 40] = -np.inf
    c = rng.standard_normal(n) * 1e-9 + 1e9
    c[n // 2 : n // 2 + 50] = 1e9
    return {
        "mixed_nonfinite": a,
        "inf_runs_const": b,
        "all_nan": np.full(n, np.nan),
        "all_const": np.full(n, -7.25),
        "offset_flat": c,
    }


def _all_series(n):
    series = {name: gen(n, seed=n) for name, gen in DATASETS.items()}
    series.update(_extra_series(n, n + 1))
    return series


@pytest.mark.parametrize("chunk", [5, 17, None])
def test_chunked_masks_equal_the_old_helpers(monkeypatch, chunk):
    if chunk is not None:
        monkeypatch.setattr(prep_mod, "_ROLLING_CHUNK", chunk)
    for n in (60, 257, 1000):
        for name, T in _all_series(n).items():
            for m in (3, 7, 50, n // 2 + 1, n):
                fin = _legacy_rolling_isfinite(np.isfinite(T), m)
                np.testing.assert_array_equal(rolling_isfinite(np.isfinite(T), m), fin)
                T_nan = np.where(np.isinf(T), np.nan, T)
                np.testing.assert_array_equal(
                    rolling_isconstant(T, m) & fin,
                    _legacy_rolling_isconstant(T_nan, m),
                    err_msg=f"{name} n={n} m={m}",
                )


def _device(x):
    return None if x is None else np.array(x)


@pytest.mark.parametrize("chunk", [17, None])
def test_preprocess_fields_are_bitwise_unchanged(monkeypatch, chunk):
    if chunk is not None:
        monkeypatch.setattr(prep_mod, "_ROLLING_CHUNK", chunk)
    for n in (60, 257, 1000):
        for name, T in _all_series(n).items():
            for m in (3, 50, n // 2 + 1):
                flags = np.random.default_rng(m).random(n - m + 1) < 0.3
                for normalize in (True, False):
                    for user in (None, flags):
                        for center, scale in ((None, None), (0.5, 2.0)):
                            with warnings.catch_warnings(record=True) as rec:
                                warnings.simplefilter("always")
                                p = preprocess_series(
                                    T,
                                    m,
                                    normalize=normalize,
                                    center=center,
                                    scale=scale,
                                    isconstant=user,
                                )
                            ref, ref_warn = _legacy_preprocess_fields(
                                T,
                                m,
                                normalize=normalize,
                                center=center,
                                scale=scale,
                                isconstant=user,
                            )
                            where = f"{name} n={n} m={m} norm={normalize}"
                            assert len(rec) == len(ref_warn), where
                            for key, val in ref.items():
                                got = getattr(p, key)
                                if key.endswith("_mx"):
                                    got = _device(got)
                                if isinstance(val, float):
                                    assert got == val or (np.isnan(got) and np.isnan(val))
                                    continue
                                assert got.dtype == val.dtype, (where, key)
                                np.testing.assert_array_equal(got, val, err_msg=f"{where} {key}")
                            if not normalize:
                                assert p.sig_inv_mx is None and p.isconstant_mx is None


def test_normalized_preprocess_transient_is_about_one_series():
    n = 2_000_000 + 37
    T = _walk(n, seed=21)
    T[::99_991] = np.nan
    T[7::199_999] = np.inf
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        base = tracemalloc.get_traced_memory()[0]
        p = preprocess_series(T, 100)
        peak = tracemalloc.get_traced_memory()[1] - base
    finally:
        tracemalloc.stop()
    assert isinstance(p, PreprocessedSeries)
    # was 5.25x (6.25x when n % m != 0): int64 cumsums, van Herk copies and
    # a full inf->NaN copy of the series
    assert peak <= 2.0 * T.nbytes, peak / T.nbytes


@pytest.mark.parametrize("flags", [None, "array"])
def test_normalized_match_computes_the_window_masks_once(monkeypatch, flags):
    T = _walk(5_000, seed=22)
    T[::997] = np.nan
    T[5::1999] = np.inf
    Q = T[3_000:3_040].copy()
    if flags == "array":
        flags = np.random.default_rng(1).random(T.size - 39) < 0.1
        flags &= rolling_isfinite(np.isfinite(T), 40)
    ref = mlx_stump.match(Q, T, T_subseq_isconstant=flags)
    calls = []
    for name in ("rolling_isconstant", "rolling_isfinite"):
        real = getattr(prep_mod, name)

        def spy(*args, _real=real, _name=name):
            calls.append(_name)
            return _real(*args)

        monkeypatch.setattr(prep_mod, name, spy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = mlx_stump.match(Q, T, T_subseq_isconstant=flags)
    # mass's preprocessing only; match used to rebuild an inf->NaN copy of
    # the series and recompute both masks after mass returned
    expected = ["rolling_isfinite"] + (["rolling_isconstant"] if flags is None else [])
    assert calls == expected
    _assert_same_matches(got, ref)


# --------------------------------------------------- 8. byte-swapped float64
def test_byte_swapped_float64_is_accepted_everywhere():
    T = _walk(600, seed=31)
    T[100] = np.nan
    Tb = T.astype(">f8")
    Q = T[200:232].copy()
    Qb = Q.astype(">f8")
    assert not Tb.dtype.isnative
    out = check_series(Tb, "T", copy=False)
    assert out.dtype.isnative and out.dtype == np.float64
    np.testing.assert_array_equal(out, T)

    for normalize in (True, False):
        a = mlx_stump.stump(Tb, 32, normalize=normalize)
        b = mlx_stump.stump(T, 32, normalize=normalize)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(
            mlx_stump.mass(Qb, Tb, normalize=normalize), mlx_stump.mass(Q, T, normalize=normalize)
        )
        _assert_same_matches(
            mlx_stump.match(Qb, Tb, normalize=normalize), mlx_stump.match(Q, T, normalize=normalize)
        )
    np.testing.assert_array_equal(mlx_stump.mass_absolute(Qb, Tb), mlx_stump.mass_absolute(Q, T))
    _assert_same_matches(mlx_stump.aamp_match(Qb, Tb), mlx_stump.aamp_match(Q, T))


@pytest.mark.parametrize("dtype", [np.float32, ">f4", np.int64, np.complex128, object])
def test_other_dtypes_are_still_rejected(dtype):
    T = np.arange(50).astype(dtype)
    with pytest.raises(TypeError, match="float64"):
        check_series(T, "T")
    with pytest.raises(TypeError, match="float64"):
        mlx_stump.mass(np.arange(5.0), T)


# ------------------------------------------------ 9. callable constant flags
def isconstant_custom_func(a, w, quantile_threshold=0.05):
    """STUMPY-docs-style tolerance flag: the lowest-sigma windows."""
    sigma = np.nanstd(np.lib.stride_tricks.sliding_window_view(a, w), axis=1)
    return sigma < np.nanquantile(sigma, quantile_threshold)


def _flatline_series(n, seed):
    rng = np.random.default_rng(seed)
    T = rng.standard_normal(n).cumsum()
    T[n // 3 : n // 3 + 200] = T[n // 3 - 1] + 1e-3 * rng.standard_normal(200)
    T[n // 2] = np.nan
    return T


def _assert_flagged_neighbours_tie(I, I_ref, TA, TB, m, tie):
    """Tie-tolerant index check under the callable's constant flags (a
    flagged pair is at 0, a half-flagged one at sqrt(m))."""
    fa, fb = (
        isconstant_custom_func(np.where(np.isinf(X), np.nan, X), m)
        & rolling_isfinite(np.isfinite(X), m)
        for X in (TA, TB)
    )
    I, I_ref = np.asarray(I, np.int64), np.asarray(I_ref, np.int64)
    np.testing.assert_array_equal(I == -1, I_ref == -1)
    for i in np.flatnonzero((I != I_ref) & (I_ref >= 0)):
        d = []
        for j in (I[i], I_ref[i]):
            if fa[i] or fb[j]:
                d.append(0.0 if fa[i] and fb[j] else np.sqrt(m))
            else:
                a, b = TA[i : i + m], TB[j : j + m]
                d.append(np.linalg.norm((a - a.mean()) / a.std() - (b - b.mean()) / b.std()))
        assert abs(d[0] - d[1]) <= tie, (i, I[i], I_ref[i], d)


def test_callable_t_flag_matches_stumpy_stump():
    T = _flatline_series(3000, seed=41)
    m = 50
    ref = stumpy.stump(T, m, T_A_subseq_isconstant=isconstant_custom_func)
    got = mlx_stump.stump(T, m, T_A_subseq_isconstant=isconstant_custom_func)
    tie = tie_tolerance(m)
    _assert_flagged_neighbours_tie(got.I_, ref.I_, T, T, m, tie)
    assert_profile_close(got.P_, ref.P_, m=m, tie_atol=tie)
    # AB-join: each side resolves its own callable
    TB = _flatline_series(2000, seed=42)
    ref = stumpy.stump(
        T, m, TB, ignore_trivial=False,
        T_A_subseq_isconstant=isconstant_custom_func,
        T_B_subseq_isconstant=isconstant_custom_func,
    )  # fmt: skip
    got = mlx_stump.stump(
        T, m, TB, ignore_trivial=False,
        T_A_subseq_isconstant=isconstant_custom_func,
        T_B_subseq_isconstant=isconstant_custom_func,
    )  # fmt: skip
    _assert_flagged_neighbours_tie(got.I_, ref.I_, T, TB, m, tie)
    assert_profile_close(got.P_, ref.P_, m=m, tie_atol=tie)


def test_callable_flags_match_stumpy_mass_and_match():
    T = _flatline_series(3000, seed=43)
    m = 40
    Q = T[1_050 : 1_050 + m].copy()  # inside the jittered flatline

    def q_flag(a, w):
        return np.array([np.std(a) < 0.01])

    kwargs = dict(T_subseq_isconstant=isconstant_custom_func, Q_subseq_isconstant=q_flag)
    assert q_flag(Q, m)[0]
    # (STUMPY's match hands its callable the series with raw inf, its mass
    # and stump with NaN; mlx-stump always uses NaN, so compare without inf)
    got = mlx_stump.match(Q, T, max_distance=0.5, **kwargs)
    ref = stumpy.match(Q, T, max_distance=0.5, **kwargs)
    assert got.shape == ref.shape and got.shape[0] >= 2
    np.testing.assert_array_equal(got[:, 1].astype(np.int64), ref[:, 1].astype(np.int64))
    np.testing.assert_array_equal(got[:, 0].astype(float), ref[:, 0].astype(float))  # all 0

    T[2000] = np.inf
    D = mlx_stump.mass(Q, T, **kwargs)
    Dr = stumpy.mass(Q, T, **kwargs)
    assert_dist_profiles_close(D, Dr, m=m)
    np.testing.assert_array_equal(D == 0.0, Dr == 0.0)  # the same windows flagged


def test_callable_flag_equals_the_equivalent_array():
    T = _flatline_series(2500, seed=44)
    m = 30
    Q = T[900:930].copy()
    flags = isconstant_custom_func(np.where(np.isinf(T), np.nan, T), m)

    def as_callable(a, w):
        return flags

    np.testing.assert_array_equal(
        mlx_stump.stump(T, m, T_A_subseq_isconstant=as_callable),
        mlx_stump.stump(T, m, T_A_subseq_isconstant=flags),
    )
    for normalize in (True, False):
        np.testing.assert_array_equal(
            mlx_stump.mass(Q, T, normalize=normalize, T_subseq_isconstant=as_callable),
            mlx_stump.mass(Q, T, normalize=normalize, T_subseq_isconstant=flags),
        )
        _assert_same_matches(
            mlx_stump.match(
                Q, T, normalize=normalize, T_subseq_isconstant=as_callable,
                Q_subseq_isconstant=lambda a, w: np.array([False]),
            ),
            mlx_stump.match(
                Q, T, normalize=normalize, T_subseq_isconstant=flags,
                Q_subseq_isconstant=False,
            ),
        )  # fmt: skip


def test_match_evaluates_each_callable_flag_once_on_a_nan_copy():
    T = _flatline_series(2000, seed=45)
    T[1500] = np.inf
    T_before = T.copy()
    m = 25
    Q = T[300:325].copy()
    seen = {"T": 0, "Q": 0}

    def t_flag(a, w):
        seen["T"] += 1
        assert w == m and a.shape == T.shape
        assert np.isnan(a[1500]) and not np.isinf(a).any()
        a[:] = 0.0  # a private copy: mutating it must not matter
        return np.zeros(a.size - w + 1, dtype=bool)

    def q_flag(a, w):
        seen["Q"] += 1
        return np.array([False])

    got = mlx_stump.match(Q, T, T_subseq_isconstant=t_flag, Q_subseq_isconstant=q_flag)
    assert seen == {"T": 1, "Q": 1}
    np.testing.assert_array_equal(T, T_before)
    _assert_same_matches(
        got,
        mlx_stump.match(
            Q, T,
            T_subseq_isconstant=np.zeros(T.size - m + 1, dtype=bool),
            Q_subseq_isconstant=False,
        ),
    )  # fmt: skip


def test_callable_flags_are_validated_like_stumpy():
    T = _walk(300, seed=46)
    Q = T[10:30].copy()
    l = T.size - Q.size + 1
    with pytest.raises(ValueError, match="Incompatible arguments"):
        mlx_stump.mass(Q, T, T_subseq_isconstant=lambda x, m: np.zeros(l, dtype=bool))
    with pytest.raises(ValueError, match="Incompatible arguments"):
        mlx_stump.stump(T, 20, T_A_subseq_isconstant=lambda a, w, extra: None)
    with pytest.raises(ValueError, match="boolean array"):
        mlx_stump.mass(Q, T, T_subseq_isconstant=lambda a, w: np.zeros(l))
    with pytest.raises(ValueError, match="boolean array"):
        mlx_stump.mass(Q, T, T_subseq_isconstant=lambda a, w: np.zeros(l + 1, dtype=bool))
    with pytest.raises(ValueError, match="single boolean"):
        mlx_stump.mass(Q, T, Q_subseq_isconstant=lambda a, w: np.zeros(2, dtype=bool))
    # size-1 results in any shape STUMPY indexes are fine
    for res in (np.array([[True]]), np.array([True]), True):
        D = mlx_stump.mass(Q, T, Q_subseq_isconstant=lambda a, w, r=res: r)
        assert D[0] == np.sqrt(Q.size)  # constant query vs varying window
    # raw mode validates the callable, then ignores it
    np.testing.assert_array_equal(
        mlx_stump.mass(Q, T, normalize=False, T_subseq_isconstant=isconstant_custom_func),
        mlx_stump.mass(Q, T, normalize=False),
    )


def test_callable_flags_on_nonfinite_windows_are_switched_off_with_warning():
    T = _walk(400, seed=47)
    T[200] = np.nan
    m = 20
    with pytest.warns(UserWarning, match="switched from True"):
        mp = mlx_stump.stump(T, m, T_A_subseq_isconstant=lambda a, w: np.ones(a.size - w + 1, bool))
    assert np.all(np.isinf(mp.P_[181:201]))


# ---------------------------------------- 10. STUMPY aamp signatures, warnings
@pytest.mark.parametrize(
    ("ours", "theirs"),
    [
        (mlx_stump.mass_absolute, stumpy.core.mass_absolute),
        (mlx_stump.aamp_match, stumpy.aamp_match),
    ],
)
def test_aamp_aliases_have_stumpys_exact_signature(ours, theirs):
    def params(f):
        return [(p.name, p.default, p.kind) for p in inspect.signature(f).parameters.values()]

    assert params(ours) == params(theirs)
    assert {"mass_absolute", "aamp_match"} <= set(mlx_stump.__all__)


def test_mass_absolute_positional_calls_match_stumpy():
    T = _walk(1500, seed=51)
    T[700] = np.nan
    Q = T[300:340].copy()
    l = T.size - Q.size + 1
    scale = float(np.nanstd(T))
    for args in [(None, 2.0, None), (None, 2.0, 300)]:
        D = mlx_stump.mass_absolute(Q, T, *args)
        Dr = stumpy.core.mass_absolute(Q, T, *args)
        assert_dist_profiles_close(D, Dr, m=Q.size, scale=scale)
    assert mlx_stump.mass_absolute(Q, T, None, 2.0, 300)[300] == 0.0
    override = np.ones(l, dtype=bool)
    override[300] = False
    D = mlx_stump.mass_absolute(Q, T, override, 2.0, 300)
    assert np.isinf(D[300])  # the finite mask is applied after the zeroing
    same = mlx_stump.mass(Q, T, None, None, False, 2.0, override, None, None, 300)
    np.testing.assert_array_equal(D, same)
    with pytest.raises(NotImplementedError, match="p=2.0"):
        mlx_stump.mass_absolute(Q, T, None, 1.0)


def test_aamp_match_positional_calls_match_stumpy():
    T = _walk(3000, seed=52)
    Q = T[1000:1050].copy()
    for args in [
        (None, np.inf, 5, 1e-8, None),
        (None, None, None, 1e-8, None),
        (None, np.inf, 5, 1e-8, 1000),
        (None, 20.0, None, 1e-8, 1000),
    ]:
        got = mlx_stump.aamp_match(Q, T, *args)
        ref = stumpy.aamp_match(Q, T, *args)
        assert got.shape == ref.shape
        np.testing.assert_array_equal(got[:, 1].astype(np.int64), ref[:, 1].astype(np.int64))
        np.testing.assert_allclose(got[:, 0].astype(float), ref[:, 0].astype(float), atol=1e-9)
    assert mlx_stump.aamp_match(Q, T, None, np.inf, 1, 1e-8, 1000)[0, 0] == 0.0


def test_aamp_match_keeps_the_true_query_distance():
    """stumpy.aamp_match reports query_idx at its real distance (and stops
    there when it exceeds the threshold); stumpy.match zeroes it."""
    T = _walk(3000, seed=53)
    Q = T[1000:1050].copy()
    qi = 2000  # a different window
    true_d = float(np.linalg.norm(T[qi : qi + 50] - Q))
    ref = stumpy.aamp_match(Q, T, max_distance=np.inf, max_matches=3, query_idx=qi)
    got = mlx_stump.aamp_match(Q, T, max_distance=np.inf, max_matches=3, query_idx=qi)
    np.testing.assert_array_equal(got[:, 1].astype(np.int64), ref[:, 1].astype(np.int64))
    assert got[0, 1] == qi and got[0, 0] == pytest.approx(true_d, rel=1e-12)
    # the forced first candidate fails the threshold: nothing is returned
    for f in (stumpy.aamp_match, mlx_stump.aamp_match):
        assert f(Q, T, max_distance=true_d / 2, query_idx=qi).shape[0] == 0
    with pytest.warns(UserWarning, match="query_idx"):
        zeroed = mlx_stump.match(Q, T, max_distance=true_d / 2, query_idx=qi, normalize=False)
    assert zeroed[0, 1] == qi and zeroed[0, 0] == 0.0
    with pytest.raises(ValueError, match="query_idx"):
        mlx_stump.aamp_match(Q, T, query_idx=-1)


def test_warnings_point_at_the_callers_line():
    rng = np.random.default_rng(54)
    T = _walk(600, seed=54)
    T[300] = np.nan
    m = 20
    Q = T[50:70].copy()
    l = T.size - m + 1
    flags = np.ones(l, dtype=bool)
    wide = np.concatenate([rng.standard_normal(400), 1e17 * rng.standard_normal(400)])
    calls = [
        lambda: mlx_stump.mass(Q, T, T_subseq_isconstant=flags),
        lambda: mlx_stump.match(Q, T, T_subseq_isconstant=flags),
        lambda: mlx_stump.mass(Q, T, query_idx=100),
        lambda: mlx_stump.match(Q, T, query_idx=100),
        lambda: mlx_stump.mass_absolute(Q, T, query_idx=100),
        lambda: mlx_stump.mass_absolute(wide[:20].copy(), wide),
        lambda: mlx_stump.aamp_match(wide[:20].copy(), wide),
        lambda: mlx_stump.mass(Q[:, None], T),
    ]
    for call in calls:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            call()
        assert rec, call
        assert {r.filename for r in rec} == {__file__}


def test_public_signatures_are_annotated():
    for f in (mlx_stump.mass, mlx_stump.match, mlx_stump.mass_absolute, mlx_stump.aamp_match):
        hints = typing.get_type_hints(f)
        assert hints["return"] is np.ndarray
        assert "Q" in hints and "T" in hints
