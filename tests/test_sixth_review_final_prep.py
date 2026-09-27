"""Sixth review round, final adversarial review: preprocessing, mass/match, stimp.

1. [parity] Raw-mode (``normalize=False``) exact ties between bitwise-
   identical non-constant windows did not follow STUMPY's nearest-in-time
   rule: the block-local rolling sums rounded each window by its offset in
   its block, so identical windows got means a few ulps apart, the float32
   distance carried ``m * dmu**2`` noise, and the ``(d2, key)`` tie rule
   never saw a tie (I_ agreed with ``stumpy.aamp`` on 87% of the rows of a
   tiled integer pattern, and a k=3 row kept its 3 nearest copies on 857 of
   1091 rows). Window sums now come from one fixed pairwise (doubling) tree
   per window, so identical windows get identical statistics; the repair
   test uses the error bound re-derived for pairwise summation. Past
   ``_ROLLING_CHUNK`` the ~2w-long level buffers are summed in place, so
   very wide windows peak no higher than the old sums did.
2. [integration F1] ``match``/``aamp_match`` with an unsigned NumPy
   ``query_idx``: ``idx - excl_zone`` wrapped on NumPy 2 (the query was
   reported twice) and became a float on NumPy 1.24 (TypeError). The index
   is a Python int once ``mass`` has range-checked it.
3. [integration F3] Raw ``stimp.pan()`` raised under
   ``np.errstate(all="raise")`` on 2**-1060-scale, 1e305-scale and
   mixed-scale series (the pan scale's product/reciprocal and the pan
   multiply); both are guarded now, with unchanged values.
4. [memory MEM-5] The raw shared frame (``mass``/``match``/``stump``)
   compacted the finite values into copies shorter than the series, which
   macOS malloc cannot reuse for the n-sized preprocessing arrays that
   follow (+0.45-0.7 GiB RSS at n=3e7 with 0.1% NaN). ``finite_center_scale``
   gathers them chunk-wise into one full-length buffer, bit-identically.
5. [memory MEM-3] An error or Ctrl-C inside ``mass``'s GPU section left the
   window block and batch buffers in MLX's cache; an inline handler now
   releases them (and clears the traceback's mlx-stump frames).
6. [docs-2 / quality Q2] ``IsConstantFunc``/``IsConstantSpec`` are defined
   once, in ``_preprocess``; ``stump``/``gpu_stump`` annotate their constant
   flags with the alias that admits callables (see also
   ``test_sixth_review_stump.py::test_public_annotations_resolve``).
7. [quality Q5] ``check_window_size`` used a hard-coded denominator of 4 by
   default; it reads the configured STUMPY_EXCL_ZONE_DENOM now.
8. [quality Q7] ``process_isconstant`` lost its unreachable ``None`` branch,
   and ``_mass``/``_match`` require an explicit ``stacklevel``.
9. [quality Q1] The float32 margin of the fixed-threshold top-k band had no
   test: near-duplicate windows that float32 reorders around the band's
   bound U now pin it (the ``cap = U`` mutant fails here).
10. [docs-3] The ``stimp`` docstring states that ``min_m``/``max_m``/``step``
    must be integers, unlike STUMPY.
"""

from __future__ import annotations

import collections.abc
import gc
import inspect
import itertools
import math
import tracemalloc
import typing
import warnings

import mlx.core as mx
import numpy as np
import pytest

import mlx_stump
import mlx_stump._engine as eng
import mlx_stump._mass as mass_mod
import mlx_stump._match as match_mod
import mlx_stump._preprocess as prep_mod
import mlx_stump._stump as stump_mod
from mlx_stump._preprocess import (
    IsConstantFunc,
    IsConstantSpec,
    _rolling_sum_depth,
    _rolling_sum_local,
    check_window_size,
    exclusion_zone,
    finite_center_scale,
    preprocess_series,
    process_isconstant,
    rolling_mean_sigma,
    stable_center_scale,
)

from .conftest import random_walk
from .test_sixth_review_engine import _force_fallback, _force_tiles

stumpy = pytest.importorskip("stumpy")

# exact-tie inputs trigger STUMPY's (and mlx-stump's STUMPY-style) advisories
pytestmark = [
    pytest.mark.filterwarnings("ignore:A large number of values in `P`:UserWarning"),
    pytest.mark.filterwarnings("ignore:The window size:UserWarning"),
]

MIB = 1 << 20
EPS = np.finfo(np.float64).eps


def _run(monkeypatch, T, m, *, fused, tiled, columns=37, **kwargs):
    """stump() on the fused kernels or the forced compiled fallback, dense or
    forced tiled (with the helpers of test_sixth_review_engine.py)."""
    with monkeypatch.context() as mp:
        if not fused:
            _force_fallback(mp)
        if tiled:
            _force_tiles(mp, m, columns)
        return mlx_stump.stump(T, m, **kwargs)


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


# ------------------------------------------------ 1. raw exact ties (parity)
def _int_tile():
    rng = np.random.default_rng(0)
    return np.tile(rng.integers(-4, 5, 37).astype(float), 30), 20


def _dyadic_tile():
    rng = np.random.default_rng(1)
    return np.tile(rng.integers(-8, 9, 23) / 8, 60), 16


def _planted_motif():
    # non-periodic: a 40-sample integer motif planted 12 times in integer
    # noise (a whole-series cumsum difference fails this one too)
    rng = np.random.default_rng(123)
    T = rng.integers(-50, 51, 3000).astype(float)
    motif = rng.integers(-50, 51, 40).astype(float)
    for p in np.sort(rng.choice(np.arange(0, 3000 - 40, 200), 12, replace=False)):
        T[p : p + 40] = motif
    return T, 25


RAW_TIE_DATA = {"int_tile": _int_tile, "dyadic_tile": _dyadic_tile, "planted": _planted_motif}


@pytest.mark.parametrize("fused", [True, False])
@pytest.mark.parametrize("tiled", [False, True])
@pytest.mark.parametrize("name", sorted(RAW_TIE_DATA))
def test_raw_identical_window_ties_follow_stumpy(monkeypatch, name, tiled, fused):
    """stumpy.aamp is exact on integer/dyadic data: on every row whose nearest
    neighbour is an identical window it picks the nearest copy in time
    (left on an equal offset), and so does mlx-stump now."""
    T, m = RAW_TIE_DATA[name]()
    ref = stumpy.aamp(T, m)
    mp = _run(monkeypatch, T, m, fused=fused, tiled=tiled, normalize=False)
    rows = np.asarray(ref.P_, dtype=np.float64) == 0.0
    assert rows.sum() > 100
    np.testing.assert_array_equal(np.asarray(mp.P_, dtype=np.float64)[rows], 0.0)
    for f in ("I_", "left_I_", "right_I_"):
        np.testing.assert_array_equal(
            np.asarray(getattr(mp, f))[rows], np.asarray(getattr(ref, f))[rows], err_msg=f
        )


@pytest.mark.parametrize("fused", [True, False])
@pytest.mark.parametrize("tiled", [False, True])
def test_raw_topk_keeps_the_nearest_identical_windows(monkeypatch, tiled, fused):
    """k=3 on the integer tile: every row lists its 3 nearest identical
    windows outside the exclusion zone, nearest first and left first on an
    equal offset (a direct oracle: STUMPY's own tied top-k set depends on
    its thread count)."""
    T, m = _int_tile()
    mp = _run(monkeypatch, T, m, fused=fused, tiled=tiled, normalize=False, k=3)
    I, P = np.asarray(mp.I_), np.asarray(mp.P_, dtype=np.float64)
    W = np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(T, m))
    v = W.view(np.dtype((np.void, W.itemsize * m))).ravel()
    excl = exclusion_zone(m)
    for i in range(W.shape[0]):
        same = np.flatnonzero(v == v[i])
        same = same[np.abs(same - i) > excl]
        key = 2 * np.abs(same - i) + (same > i)
        nearest = same[np.argsort(key, kind="stable")[:3]]
        assert nearest.size == 3
        np.testing.assert_array_equal(I[i], nearest, err_msg=f"row {i}")
        np.testing.assert_array_equal(P[i], 0.0)


@pytest.mark.parametrize("chunk", [None, 5])
def test_identical_raw_windows_get_identical_statistics(monkeypatch, chunk):
    """preprocess_series(normalize=False): copies of a window at different
    offsets (in and across rolling chunks) share mu, ssq, sigma and the
    device mean pair bit for bit, on the plain path and on the two-pass
    repair path (a near-flat motif on a large offset)."""
    if chunk is not None:
        monkeypatch.setattr(prep_mod, "_ROLLING_CHUNK", chunk)
    rng = np.random.default_rng(3)
    m = 37
    T = rng.standard_normal(20_000).cumsum()
    plain = rng.standard_normal(m).cumsum() * 3.0 + 5.0
    flat = 1e6 + 1e-6 * rng.standard_normal(m)
    starts = {"plain": [101, 1000, 2047, 4109, 7777], "flat": [12_345, 15_000, 16_385, 19_000]}
    for s in starts["plain"]:
        T[s : s + m] = plain
    for s in starts["flat"]:
        T[s : s + m] = flat
    prep = preprocess_series(T, m, normalize=False, keep_sigma=True)
    mu_mx = np.array(prep.mu_mx)
    for group in starts.values():
        for field in (prep.mu, prep.ssq, prep.sigma, mu_mx):
            assert len({field[s].tobytes() for s in group}) == 1


SUM_SHAPES = [(1, 1), (9, 3), (100, 99), (1000, 7), (1025, 512), (4000, 1000)]


@pytest.mark.parametrize(("n", "w"), SUM_SHAPES)
def test_pairwise_sums_are_chunk_independent_and_within_their_bound(monkeypatch, n, w):
    rng = np.random.default_rng(n * 31 + w)
    x = rng.standard_normal(n) * 10.0 ** rng.integers(-5, 6, n)
    full = _rolling_sum_local(x, w)
    for chunk in (1, 3, 64):
        monkeypatch.setattr(prep_mod, "_ROLLING_CHUNK", chunk)
        assert _rolling_sum_local(x, w).tobytes() == full.tobytes()
    ref = np.array([math.fsum(x[j : j + w]) for j in range(n - w + 1)])
    mag = np.array([math.fsum(np.abs(x[j : j + w])) for j in range(n - w + 1)])
    # pairwise: at most depth * eps/2 of the window's own magnitudes (x2 slack)
    assert np.all(np.abs(full - ref) <= _rolling_sum_depth(w) * EPS * mag)
    ints = rng.integers(-1000, 1000, n).astype(np.float64)
    exact = np.array([math.fsum(ints[j : j + w]) for j in range(n - w + 1)])
    np.testing.assert_array_equal(_rolling_sum_local(ints, w), exact)


def test_wide_window_levels_are_summed_in_one_buffer(monkeypatch):
    """Past _ROLLING_CHUNK a chunk's level buffer is ~2w long; its levels are
    summed in place, so one such buffer is alive instead of two (two, plus
    a view pinning the previous chunk's, took raw preprocessing at w = n/2
    to 5.69x the series against 4.75x for the old sums), bit-identically."""
    n, w = 40_000, 20_000
    x = random_walk(n, seed=4)
    ref = _rolling_sum_local(x, w)  # w <= _ROLLING_CHUNK: the allocating path
    monkeypatch.setattr(prep_mod, "_ROLLING_CHUNK", 64)
    tracemalloc.start()
    try:
        base = tracemalloc.get_traced_memory()[0]
        got = _rolling_sum_local(x, w)
        peak = tracemalloc.get_traced_memory()[1] - base
    finally:
        tracemalloc.stop()
    assert got.tobytes() == ref.tobytes()
    level = (w + w - 1) * 8  # step = max(chunk, w) windows plus the w - 1 overlap
    assert peak - got.nbytes <= 1.25 * level, peak / level


def test_rolling_sum_depth_bounds_every_terms_additions():
    """The depth used by the error bound is the tree's true maximum."""
    for w in range(1, 300):
        bits = [b for b in range(w.bit_length()) if w >> b & 1]
        p = len(bits)
        depth = max([bits[0] + p - 1] + [bits[k] + 1 + (p - 1 - k) for k in range(1, p)])
        assert depth <= _rolling_sum_depth(w) <= depth + 1, w


def test_pairwise_repair_bound_caps_the_surviving_variance_error(monkeypatch):
    """Windows on a unit offset whose variance sits from 0.25x to 4x the
    pairwise repair threshold ``H*(1.5*D + 3)*eps*S2/w``: those well below
    it are repaired two-pass, those well above are not (the old
    ``1.5*eps*S2`` test, 6x wider at w=48, repaired both), and every window
    keeps a relative variance error of at most ``1/(H - 1)``."""
    H = prep_mod._SIGMA_REPAIR_HEADROOM
    w = 48
    rel_threshold = H * (1.5 * _rolling_sum_depth(w) + 3.0) * EPS  # var / (S2/w)
    rng = np.random.default_rng(5)
    factors = (0.25, 0.9, 1.1, 1.5, 4.0)
    seg = 40 * w
    parts = []
    for c in factors:
        # centered +-d signs: variance ~d^2 on a mean of 1 (S2/w ~ 1)
        signs = rng.choice([-1.0, 1.0], seg)
        parts.append(1.0 + np.sqrt(c * rel_threshold) * (signs - signs.mean()))
    a = np.concatenate(parts)

    repaired = []
    real = prep_mod._two_pass_repair

    def spy(a_, w_, idx, mu, var):
        repaired.append(idx.copy())
        return real(a_, w_, idx, mu, var)

    monkeypatch.setattr(prep_mod, "_two_pass_repair", spy)
    mu, sigma = rolling_mean_sigma(a, w)
    rows = np.concatenate(repaired) if repaired else np.zeros(0, dtype=np.int64)
    low = np.arange(0, seg - w + 1)  # windows inside the 0.25x segment
    high = np.arange(4 * seg, 5 * seg - w + 1)  # inside the 4x segment
    assert np.isin(low, rows).all()
    assert not np.isin(high, rows).any()

    W = np.lib.stride_tricks.sliding_window_view(a, w)
    c = W - W.mean(axis=1)[:, None]
    c -= c.mean(axis=1)[:, None]  # second correction pass
    var = np.einsum("ij,ij->i", c, c) / w
    rel = np.abs(sigma**2 - var) / var
    assert rel.max() <= 1.0 / (H - 1)


# ------------------------------------------- 2. unsigned query_idx (F1)
@pytest.mark.parametrize("max_matches", [1, 3, 8, 9, None])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint32, np.uint64])
@pytest.mark.parametrize("raw", [False, True])
def test_unsigned_query_idx_matches_stumpy(raw, dtype, max_matches):
    """qi=0 and 3 (below the exclusion zone: idx - excl_zone wrapped) and
    250 (uint8: idx + excl_zone wrapped past 255), against STUMPY and the
    plain-int call."""
    T = random_walk(500, seed=21)
    m = 25
    ours = mlx_stump.aamp_match if raw else mlx_stump.match
    ref = stumpy.aamp_match if raw else stumpy.match
    for qi in (0, 3, 250):
        Q = T[qi : qi + m].copy()
        for md in (None, np.inf):
            kw = dict(max_distance=md, max_matches=max_matches)
            got = ours(Q, T, query_idx=dtype(qi), **kw)
            _assert_same_matches(got, ours(Q, T, query_idx=qi, **kw))
            idx = got[:, 1].astype(np.int64)
            assert np.unique(idx).size == idx.size, (qi, md)
            want = ref(Q, T, query_idx=dtype(qi), **kw)
            np.testing.assert_array_equal(idx, want[:, 1].astype(np.int64), err_msg=f"{qi} {md}")
            np.testing.assert_allclose(
                got[:, 0].astype(float), want[:, 0].astype(float), rtol=1e-6, atol=1e-6
            )


# ---------------------------------------------- 3. stimp pan errstate (F3)
def _extreme_series():
    base = random_walk(400, seed=7)
    mixed = base.copy()
    mixed[:200] = base[:200] * 1e-300
    mixed[200:] = base[200:] / np.abs(base[200:]).max() * 1e305
    unit = (base - base.min()) / (base.max() - base.min())
    return {
        "tiny": base * 2.0**-1060,  # range * sqrt(m) underflows (reciprocal overflows)
        "huge": base / np.abs(base).max() * 1e307,  # the reciprocal underflows
        "near_max": unit * 1.7e308,  # range * sqrt(m) overflows
        "mixed": mixed,  # a subnormal-range distance times a tiny scale
    }


@pytest.mark.parametrize("cls", [mlx_stump.stimp, mlx_stump.gpu_stimp])
@pytest.mark.parametrize("name", ["tiny", "huge", "near_max", "mixed"])
def test_raw_pan_ignores_caller_trapping_on_extreme_scales(cls, name):
    T = _extreme_series()[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the small-P / dynamic-range advisories
        pmp = cls(T, 5, 30, 5, normalize=False)
        for _ in range(3):
            pmp.update()
    for flags in itertools.product([True, False], repeat=4):
        kw = dict(zip(("normalize", "contrast", "binary", "clip"), flags, strict=True))
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)  # no overflow/underflow noise
            want = pmp.pan(**kw)
        with np.errstate(all="raise"):
            got = pmp.pan(**kw)
        np.testing.assert_array_equal(got, want, err_msg=str(kw))


# -------------------------------------------- 4. MEM-5: the finite frame
def _old_frame(*parts):
    """The pre-fix call sites: compact, concatenate, then the frame."""
    return stable_center_scale(np.concatenate([p[np.isfinite(p)] for p in parts]))


def _frame_cases():
    rng = np.random.default_rng(17)
    walk = rng.standard_normal(3001).cumsum()
    holey = walk.copy()
    holey[::97] = np.nan
    holey[5] = np.inf
    holey[2000:2600] = -np.inf
    q = walk[100:150].copy()
    extreme = rng.standard_normal(1000) * 1e300
    extreme[::3] = np.nan
    opposite = np.where(rng.standard_normal(999) > 0, 1.7e308, -1.7e308)
    opposite[::4] = np.inf
    tiny = walk * 2.0**-1060
    tiny[7::11] = np.nan
    const = np.full(500, 2.5)
    const[::9] = np.nan
    return {
        "mass": (q, holey),
        "self_join": (holey, holey),
        "ab_join": (holey, walk),
        "extreme": (extreme,),
        "opposite": (opposite,),
        "tiny": (tiny, tiny[::-1].copy()),
        "constant": (const,),
        "all_finite": (walk,),
        "empty_part": (np.full(40, np.nan), walk[:60].copy()),
        "no_finite": (np.full(40, np.nan), np.full(3, np.inf)),
    }


@pytest.mark.parametrize("chunk", [None, 7])
@pytest.mark.parametrize("name", sorted(_frame_cases()))
def test_finite_frame_is_bit_identical_to_the_compacted_one(monkeypatch, name, chunk):
    parts = _frame_cases()[name]
    before = [p.copy() for p in parts]
    if chunk is not None:
        monkeypatch.setattr(prep_mod, "_ROLLING_CHUNK", chunk)
    with np.errstate(all="raise"):
        got = finite_center_scale(*parts)
    want = _old_frame(*parts)
    assert np.array(got).tobytes() == np.array(want).tobytes(), (got, want)
    for p, b in zip(parts, before, strict=True):  # the inputs are untouched
        assert p.tobytes() == b.tobytes()


def _nan_series(n, seed):
    T = random_walk(n, seed=seed)
    T[::97] = np.nan
    T[n // 3] = np.inf
    T[n // 2 : n // 2 + 300] = -np.inf
    return T


def test_raw_outputs_unchanged_by_the_finite_frame(monkeypatch):
    """Raw mass, match, aamp_match and stump (self- and AB-join) give the
    same bits whether the frame is built compacted (the old call sites) or
    in one full-length buffer."""
    T = _nan_series(6000, seed=2)
    T_B = _nan_series(5000, seed=3)
    m = 40
    Q = T[1000 : 1000 + m].copy()

    def run():
        mp = mlx_stump.stump(T, m, normalize=False)
        ab = mlx_stump.stump(T, m, T_B, ignore_trivial=False, normalize=False)
        return [
            mlx_stump.mass(Q, T, normalize=False),
            mlx_stump.mass_absolute(Q, T, query_idx=1000),
            mlx_stump.match(Q, T, normalize=False, max_distance=np.inf),
            mlx_stump.aamp_match(Q, T, max_matches=5),
            *(np.asarray(getattr(x, f), dtype=np.float64) for x in (mp, ab) for f in
              ("P_", "I_", "left_I_", "right_I_")),
        ]  # fmt: skip

    new = run()
    monkeypatch.setattr(mass_mod, "finite_center_scale", _old_frame)
    monkeypatch.setattr(stump_mod, "finite_center_scale", _old_frame)
    old = run()
    for a, b in zip(new, old, strict=True):
        if a.dtype == object:
            _assert_same_matches(a, b)
        else:
            assert a.tobytes() == b.tobytes()


def test_raw_mass_frame_makes_no_compacted_copies(monkeypatch):
    """Traced host memory at the end of the frame phase (the entry of
    preprocess_series): the private T copy, the profile buffer and the one
    full-length frame buffer (plus a mask byte per sample), where the
    compacted copies, their concatenation and the frame's own scratch copy
    peaked at ~2 more series."""
    n = 2_000_000
    T = random_walk(n, seed=5)
    T[::1000] = np.nan  # 0.1% NaN
    Q = T[1:101].copy()  # m = 100
    peaks = []
    real = mass_mod.preprocess_series

    def spy(*args, **kwargs):
        peaks.append(tracemalloc.get_traced_memory()[1])
        raise KeyboardInterrupt  # the frame phase is all this test needs

    monkeypatch.setattr(mass_mod, "preprocess_series", spy)
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        base = tracemalloc.get_traced_memory()[0]
        with pytest.raises(KeyboardInterrupt):
            mlx_stump.mass(Q, T, normalize=False)
    finally:
        tracemalloc.stop()
    assert real is prep_mod.preprocess_series
    # T copy (8n) + D2 (8l) + frame buffer (8(n+m)) + finite mask (n) + slack
    assert peaks[0] - base <= 25 * n + 8 * MIB, (peaks[0] - base) / n


# -------------------------------------------- 5. MEM-3: mass error cleanup
def _raise_on_call(orig, n):
    calls = [0]

    def wrapped(*args, **kwargs):
        calls[0] += 1
        if calls[0] == n:
            del args, kwargs  # hold nothing in this (test-code) frame
            raise KeyboardInterrupt
        return orig(*args, **kwargs)

    return wrapped


@pytest.mark.parametrize("normalize", [True, False])
def test_mass_interrupted_in_the_block_loop_leaves_nothing_cached(monkeypatch, normalize):
    T = random_walk(300_000, seed=9)
    Q = T[:100].copy()
    mlx_stump.mass(Q, T[:5000], normalize=normalize)  # compile/warm up first
    gc.collect()
    mx.clear_cache()
    active = mx.get_active_memory()
    name = "znorm_sq_distances" if normalize else "absolute_sq_distances"
    orig = getattr(eng.MassEngine, name)
    monkeypatch.setattr(eng.MassEngine, name, _raise_on_call(orig, 1))
    with pytest.raises(KeyboardInterrupt):
        mlx_stump.mass(Q, T, normalize=normalize)
    gc.collect()
    # < 1 MiB like test_third_review_fixes.py (MLX may keep a small runtime
    # baseline); the leak this guards against was the whole window block
    assert mx.get_cache_memory() < MIB
    assert mx.get_active_memory() - active < MIB
    monkeypatch.setattr(eng.MassEngine, name, orig)
    assert np.isfinite(mlx_stump.mass(Q, T[:5000], normalize=normalize)).all()


# ---------------------------------------- 6. docs-2 / Q2: one alias each
def test_constant_flag_aliases_are_defined_once():
    assert mass_mod.IsConstantSpec is IsConstantSpec
    assert match_mod.IsConstantSpec is IsConstantSpec
    assert stump_mod.IsConstantSpec is IsConstantSpec
    import mlx_stump._stimp as stimp_mod

    assert stimp_mod.IsConstantFunc is IsConstantFunc
    assert typing.get_origin(IsConstantFunc) is collections.abc.Callable
    assert IsConstantFunc in typing.get_args(IsConstantSpec)
    hints = {
        "stump": typing.get_type_hints(mlx_stump.stump)["T_A_subseq_isconstant"],
        "gpu_stump": typing.get_type_hints(mlx_stump.gpu_stump)["T_B_subseq_isconstant"],
        "mass": typing.get_type_hints(mlx_stump.mass)["T_subseq_isconstant"],
        "match": typing.get_type_hints(mlx_stump.match)["Q_subseq_isconstant"],
    }
    assert len(set(hints.values())) == 1, hints


# --------------------------------------------- 7. Q5: configured denominator
def test_check_window_size_reads_the_configured_denominator(monkeypatch):
    monkeypatch.setattr(stumpy.config, "STUMPY_EXCL_ZONE_DENOM", 1)
    # (200 - 70 + 1) // 2 = 65 <= ceil(70 / 1): STUMPY warns (not so for 4)
    with pytest.warns(UserWarning, match="may be too large"):
        check_window_size(70, warn_n=200)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert check_window_size(70, warn_n=200, denom=4) == 70
    assert "excl_zone_denom" not in inspect.signature(check_window_size).parameters


# ------------------------------------------------------------- 8. Q7
def test_process_isconstant_takes_a_spec_and_stacklevels_are_explicit():
    T = random_walk(100, seed=1)
    with pytest.raises(ValueError, match="boolean array"):
        process_isconstant(T, 10, None, "T_subseq_isconstant")
    flags = np.zeros(91, dtype=bool)
    np.testing.assert_array_equal(process_isconstant(T, 10, flags, "x"), flags)
    for fn in (mass_mod._mass, match_mod._match):
        param = inspect.signature(fn).parameters["stacklevel"]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY
        assert param.default is inspect.Parameter.empty, fn.__name__


# ----------------------------------------------- 9. Q1: the top-k band margin
def _near_duplicates(seed, reps=100, m=32, noise=1e-7):
    # a pattern repeated with ~1e-7 relative noise: float32 reorders the
    # candidates around the band's upper bound U, so only the margin in
    # `cap = U + margin` lets the true best window be refined and picked
    rng = np.random.default_rng(seed)
    pat = rng.standard_normal(m)
    parts = []
    for _ in range(reps):
        parts.append(rng.standard_normal(int(rng.integers(m, 3 * m))).cumsum())
        parts.append(pat * (1 + noise * rng.standard_normal(m)) + noise * rng.standard_normal(m))
    return pat.copy(), np.concatenate(parts)


@pytest.mark.parametrize("normalize", [True, False])
def test_topk_band_margin_admits_float32_reordered_windows(normalize):
    for seed in range(12):
        Q, T = _near_duplicates(seed)
        full = mlx_stump.match(Q, T, max_distance=np.inf, normalize=normalize)
        for k in (1, 3):
            top = mlx_stump.match(Q, T, max_distance=np.inf, max_matches=k, normalize=normalize)
            _assert_same_matches(top, full[:k])


# ------------------------------------------------------------ 10. docs-3
def test_stimp_documents_integer_window_arguments():
    doc = " ".join(mlx_stump.stimp.__doc__.split())
    assert "``min_m``, ``max_m`` and ``step`` must be integers" in doc
    with pytest.raises(TypeError):
        mlx_stump.stimp(random_walk(200), 8.0)
