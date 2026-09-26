"""Sixth review round: engine paths the suite never executed.

1. AB-join top-k, raw top-k self-joins, the tiled raw-distance sweep, the
   padding used when ``k`` exceeds the number of target windows (dense and
   tiled), raw AB-join ``T_B_subseq_isconstant`` validation and ``(n, 1)``
   ``match`` inputs had no test at all, so the README's "every code path is
   golden-tested" was not true. Each now runs against float64 STUMPY with
   the tie-tolerant golden helpers (``match`` with column inputs is a
   self-consistency check: flattening them is a documented divergence and
   STUMPY rejects that input).
2. Two branches only randomized testing reached: tiled column blocks
   narrower than ``k`` (padding a block's top-k set) and an explicit
   ``chunk_size`` larger than the number of query rows (clamping the batch
   start). A planted bug in either passed the whole suite; both are now
   pinned bit-equal to the dense / automatic-chunk result.
3. ``stumpy.stumpi(T, m, mp=mlx_stump.stump(T, m))`` starts a streaming
   profile from the GPU result; it worked but was neither documented nor
   tested. After a series of updates it matches stumpi's own start-up.
4. A seeded differential fuzz slice, stratified over join type x
   {dense, tiled} x k in {1, 5} x normalize so every combination runs on
   every CI run, checks each result against an independent float64 oracle
   and STUMPY. ``MLX_STUMP_FUZZ_CASES=<n>`` runs more seeds.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pytest

import mlx_stump
import mlx_stump._engine as eng

from .conftest import (
    DATASETS,
    assert_indices_tie_tolerant,
    assert_profile_close,
    tie_tolerance,
)

stumpy = pytest.importorskip("stumpy")


def _force_tiled(monkeypatch, tile_bytes=32 * 1024):
    monkeypatch.setattr(eng, "_MATMUL_WINDOW_BYTES", 0)
    monkeypatch.setattr(eng, "_TILE_WINDOW_BYTES", tile_bytes)


def _columns(mp, k):
    """(P, I) as (l, k) arrays for any k."""
    P = np.asarray(mp[:, :k], dtype=np.float64)
    I = np.asarray(mp[:, k : 2 * k], dtype=np.int64)
    return P, I


def _assert_bit_equal(mp, ref, k):
    P, I = _columns(mp, k)
    Pr, Ir = _columns(ref, k)
    np.testing.assert_array_equal(P, Pr)
    np.testing.assert_array_equal(I, Ir)
    np.testing.assert_array_equal(mp.left_I_, ref.left_I_)
    np.testing.assert_array_equal(mp.right_I_, ref.right_I_)


def _golden(T_A, m, T_B=None, *, k, normalize, **kwargs):
    """Every one of the k columns against float64 STUMPY, tie-tolerantly."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mp = mlx_stump.stump(T_A, m, T_B, k=k, normalize=normalize, **kwargs)
        ref = stumpy.stump(T_A, m, T_B, k=k, normalize=normalize, **kwargs)
    B = T_A if T_B is None else T_B
    tie = tie_tolerance(m)
    if not normalize:
        tie *= max(1.0, float(np.nanstd(T_A[np.isfinite(T_A)])))
    assert mp.shape == ref.shape
    P, I = _columns(mp, k)
    Pr, Ir = _columns(ref, k)
    for j in range(k):
        assert_profile_close(P[:, j], Pr[:, j], m=m, exact_mask=I[:, j] == Ir[:, j], tie_atol=tie)
        assert_indices_tie_tolerant(I[:, j], Ir[:, j], T_A, B, m, normalize=normalize, tie_atol=tie)
    np.testing.assert_array_equal(mp.left_I_ == -1, ref.left_I_ == -1)
    np.testing.assert_array_equal(mp.right_I_ == -1, ref.right_I_ == -1)
    return mp


# ------------------------------------------------------- 1: untested paths
@pytest.mark.parametrize("tiled", [False, True])
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("k", [2, 3])
def test_topk_ab_join(monkeypatch, tiled, normalize, k):
    if tiled:
        _force_tiled(monkeypatch)
    T_A = DATASETS["with_nans"](1100, seed=31)
    T_B = DATASETS["random_walk"](1500, seed=32)
    mp = _golden(T_A, 32, T_B, k=k, normalize=normalize, ignore_trivial=False)
    assert np.all(mp.left_I_ == -1) and np.all(mp.right_I_ == -1)


@pytest.mark.parametrize("tiled", [False, True])
@pytest.mark.parametrize("k", [1, 3])
def test_raw_self_join_dense_and_tiled(monkeypatch, tiled, k):
    T = DATASETS["with_nans"](1500, seed=22)
    m = 40
    if tiled:
        dense = mlx_stump.stump(T, m, k=k, normalize=False)
        _force_tiled(monkeypatch)
    mp = _golden(T, m, k=k, normalize=False)
    if tiled:
        _assert_bit_equal(mp, dense, k)


@pytest.mark.parametrize("tiled", [False, True])
@pytest.mark.parametrize("normalize", [True, False])
def test_k_exceeds_target_windows(monkeypatch, tiled, normalize):
    """m=8 keeps finite values to compare: the AB-join has l_B = 5 < k = 8,
    the self-join l = 33 < k = 40 with most rows short of k neighbours."""
    m = 8
    if tiled:
        _force_tiled(monkeypatch, 4 * m * 4)  # blocks of at most 4 windows
    rng = np.random.default_rng(0)
    T_A, T_B = rng.standard_normal(40), rng.standard_normal(12)
    mp = _golden(T_A, m, T_B, k=8, normalize=normalize, ignore_trivial=False)
    P, I = _columns(mp, 8)
    assert np.all(np.isinf(P[:, 5:])) and np.all(I[:, 5:] == -1)
    assert np.all(np.isfinite(P[:, :5]))
    mp = _golden(T_A, m, k=40, normalize=normalize)
    P, _ = _columns(mp, 40)
    assert np.isfinite(P).sum() > 0 and np.all(np.isinf(P[:, 33:]))


def test_raw_ab_join_validates_T_B_flags():
    rng = np.random.default_rng(1)
    T_A, T_B = rng.standard_normal(300), rng.standard_normal(400)
    with pytest.raises(ValueError, match="T_B_subseq_isconstant"):
        mlx_stump.stump(
            T_A, 20, T_B, ignore_trivial=False, normalize=False, T_B_subseq_isconstant="bad"
        )


@pytest.mark.parametrize("normalize", [True, False])
def test_match_column_inputs_equal_flat_inputs(normalize):
    T = DATASETS["random_walk"](2000, seed=5)
    Q = T[300:350].copy()
    flat = mlx_stump.match(Q, T, max_matches=5, normalize=normalize)
    column = mlx_stump.match(Q[:, None], T[:, None], max_matches=5, normalize=normalize)
    np.testing.assert_array_equal(np.asarray(column, float), np.asarray(flat, float))


# ------------------------------------------ 2: branches only fuzzing reached
@pytest.mark.parametrize("join", ["self", "ab"])
def test_tiled_blocks_narrower_than_k(monkeypatch, join):
    rng = np.random.default_rng(123)
    T = rng.standard_normal(300).cumsum()
    m = 16
    if join == "self":
        T_B, kwargs = None, dict(k=5)
    else:
        T_B, kwargs = rng.standard_normal(150).cumsum(), dict(k=6, ignore_trivial=False)
    dense = mlx_stump.stump(T, m, T_B, **kwargs)
    _force_tiled(monkeypatch, 4 * m * 4)
    l_b = (T if T_B is None else T_B).size - m + 1
    assert eng.resident_block_bytes(l_b, m) // (4 * m) < kwargs["k"]  # every block < k wide
    mp = _golden(T, m, T_B, normalize=True, **kwargs)
    _assert_bit_equal(mp, dense, kwargs["k"])


@pytest.mark.parametrize(
    "join,k,chunk", [("self", 1, 1000), ("self", 3, 1000), ("ab", 1, 500)]
)
def test_chunk_size_above_query_rows(join, k, chunk):
    rng = np.random.default_rng(123)
    T = rng.standard_normal(300).cumsum()
    m = 16
    if join == "self":
        args, kwargs = (T, m), dict(k=k)
    else:  # a short query series against a longer target
        args, kwargs = (rng.standard_normal(120).cumsum(), m, T), dict(ignore_trivial=False)
    auto = mlx_stump.stump(*args, **kwargs)
    assert chunk > auto.shape[0]
    mp = mlx_stump.stump(*args, chunk_size=chunk, **kwargs)
    _assert_bit_equal(mp, auto, k)


# -------------------------------------------------------- 3: stumpi seeding
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("k", [1, 2])
def test_stumpi_seeded_with_gpu_profile(k, normalize):
    rng = np.random.default_rng(7)
    T = rng.standard_normal(1200).cumsum()
    m = 32
    mp = mlx_stump.stump(T, m, k=k, normalize=normalize)
    seeded = stumpy.stumpi(T, m, egress=False, normalize=normalize, k=k, mp=mp)
    own = stumpy.stumpi(T, m, egress=False, normalize=normalize, k=k)
    for t in T[-1] + rng.standard_normal(15).cumsum():
        seeded.update(t)
        own.update(t)
    np.testing.assert_array_equal(seeded.T_, own.T_)
    tie = tie_tolerance(m)
    if not normalize:
        tie *= max(1.0, float(np.std(seeded.T_)))
    # egress=False: indices address stream.T_ directly
    P, Pr = np.reshape(seeded.P_, (-1, k)), np.reshape(own.P_, (-1, k))
    I, Ir = np.reshape(seeded.I_, (-1, k)), np.reshape(own.I_, (-1, k))
    for j in range(k):
        assert_profile_close(P[:, j], Pr[:, j], m=m, exact_mask=I[:, j] == Ir[:, j], tie_atol=tie)
        assert_indices_tie_tolerant(
            I[:, j], Ir[:, j], seeded.T_, seeded.T_, m, normalize=normalize, tie_atol=tie
        )
    assert_indices_tie_tolerant(
        seeded.left_I_, own.left_I_, seeded.T_, seeded.T_, m, normalize=normalize, tie_atol=tie
    )
    np.testing.assert_allclose(seeded.left_P_, own.left_P_, atol=tie, rtol=0)


def test_stumpi_rejects_a_k_mismatch():
    T = np.random.default_rng(8).standard_normal(300).cumsum()
    with pytest.raises(ValueError, match="shape of `mp`"):
        stumpy.stumpi(T, 16, k=2, mp=mlx_stump.stump(T, 16, k=1))


# ----------------------------------------------- 4: seeded differential fuzz
_FAMILIES = (
    "walk", "noise", "sine", "sine_clean", "piecewise", "integers", "nonfinite",
    "const_runs", "offset", "tiny", "huge", "mixed",
)


def _fuzz_series(family, n, rng):
    if family == "walk":
        return rng.standard_normal(n).cumsum()
    if family == "noise":
        return rng.standard_normal(n)
    if family in ("sine", "sine_clean"):
        t = np.linspace(0, rng.uniform(4, 40) * np.pi, n)
        return np.sin(t) + (0.3 if family == "sine" else 0.01) * rng.standard_normal(n)
    if family == "piecewise":
        return np.repeat(rng.integers(-5, 6, size=n) * 1.0, rng.integers(1, max(2, n // 5), n))[:n]
    if family == "integers":
        if rng.random() < 0.5:
            return rng.integers(-3, 4, size=n).astype(np.float64)
        return rng.integers(-2, 3, size=n).cumsum().astype(np.float64)
    if family == "nonfinite":  # includes the first and/or last window
        T = _fuzz_series(str(rng.choice(["walk", "noise", "sine"])), n, rng)
        pos = list(rng.integers(0, n, size=int(rng.integers(1, 6))))
        pos += [0] * (rng.random() < 0.5) + [n - 1] * (rng.random() < 0.5)
        T[pos] = rng.choice([np.nan, np.inf, -np.inf], size=len(pos))
        if rng.random() < 0.2:
            s = int(rng.integers(0, n))
            T[s : s + int(rng.integers(1, 30))] = np.nan
        return T
    if family == "const_runs":
        T = _fuzz_series(str(rng.choice(["walk", "noise"])), n, rng)
        for _ in range(int(rng.integers(1, 4))):
            s = int(rng.integers(0, n))
            T[s : s + int(rng.integers(1, max(2, n // 4)))] = rng.uniform(-3, 3)
        return T
    if family == "offset":
        T = rng.standard_normal(n).cumsum()
        s = int(rng.integers(0, n))
        T[s : s + int(rng.integers(1, n))] += 1e6
        return T
    if family == "tiny":
        return _fuzz_series(str(rng.choice(["walk", "noise", "sine"])), n, rng) * 1e-8
    if family == "huge":
        return _fuzz_series(str(rng.choice(["walk", "noise", "integers"])), n, rng) * 1e150
    a = _fuzz_series(str(rng.choice(["walk", "noise", "sine", "piecewise"])), n, rng)
    b = _fuzz_series(str(rng.choice(["nonfinite", "const_runs", "integers"])), n, rng)
    cut = int(rng.integers(0, n))
    a[cut:] = b[cut:]
    return a


def _windows(T, m):
    return np.lib.stride_tricks.sliding_window_view(T, m)


def _window_flags(T, m, isconstant=None):
    W = _windows(T, m)
    finite = np.all(np.isfinite(W), axis=1)
    detected = finite & (np.ptp(np.where(np.isfinite(W), W, 0.0), axis=1) == 0.0)
    return finite, detected if isconstant is None else isconstant & finite


def _znorm_rows(W):
    """Locally z-normalized float64 rows: midpoint/radius frame, two-pass mean."""
    W = np.array(W, dtype=np.float64)
    lo, hi = W.min(axis=1), W.max(axis=1)
    mid = lo + (hi - lo) * 0.5
    rad = np.maximum(hi - mid, mid - lo)
    W -= mid[:, None]
    W /= np.where(rad > 0, rad, 1.0)[:, None]
    W -= W.mean(axis=1)[:, None]
    W -= W.mean(axis=1)[:, None]
    rms = np.sqrt(np.sum(W * W, axis=1) / W.shape[1])
    W /= np.where(rms > 0, rms, 1.0)[:, None]
    return W


def _frame(T_A, T_B):
    """Power-of-two scale that keeps raw squares finite at any input scale."""
    finite = np.concatenate([T_A[np.isfinite(T_A)], T_B[np.isfinite(T_B)], [1.0]])
    return 2.0 ** np.floor(np.log2(np.max(np.abs(finite))))


def _oracle(T_A, T_B, m, normalize, excl, const_A, const_B):
    """Full float64 distance matrix with STUMPY's special cases; ``excl`` is
    the exclusion-zone half width (None for AB-joins)."""
    fa, ca = _window_flags(T_A, m, const_A)
    fb, cb = _window_flags(T_B, m, const_B)
    WA = _windows(np.where(np.isfinite(T_A), T_A, 0.0), m)
    WB = _windows(np.where(np.isfinite(T_B), T_B, 0.0), m)
    if normalize:
        d2 = np.clip(2.0 * m - 2.0 * (_znorm_rows(WA) @ _znorm_rows(WB).T), 0.0, 4.0 * m)
        both, one = ca[:, None] & cb[None, :], ca[:, None] ^ cb[None, :]
        D = np.sqrt(np.where(both, 0.0, np.where(one, float(m), d2)))
    else:
        s = _frame(T_A, T_B)
        A, B = WA / s, WB / s
        Ac, Bc = A - A.mean(axis=1)[:, None], B - B.mean(axis=1)[:, None]
        cross = np.sum(Ac * Ac, axis=1)[:, None] + np.sum(Bc * Bc, axis=1)[None, :]
        d2 = np.maximum(cross - 2.0 * (Ac @ Bc.T), 0.0)
        D = np.sqrt(d2 + m * (A.mean(axis=1)[:, None] - B.mean(axis=1)[None, :]) ** 2) * s
    D[~(fa[:, None] & fb[None, :])] = np.inf
    if excl is not None:
        rows, cols = np.indices(D.shape)
        D[np.abs(rows - cols) <= excl] = np.inf
    return D, ca, cb


def _pair_dist(T_A, T_B, m, rows, cols, normalize, ca, cb):
    """Direct float64 evaluation of each pair (rows[t], cols[t])."""
    A, B = _windows(T_A, m)[rows], _windows(T_B, m)[cols]
    finite = np.all(np.isfinite(A), axis=1) & np.all(np.isfinite(B), axis=1)
    A = np.where(finite[:, None], A, 0.0)
    B = np.where(finite[:, None], B, 0.0)
    if normalize:
        diff = _znorm_rows(A) - _znorm_rows(B)
        d = np.sqrt(np.sum(diff * diff, axis=1))
        qc, tc = ca[rows], cb[cols]
        d = np.where(qc & tc, 0.0, np.where(qc ^ tc, np.sqrt(m), d))
    else:
        diff = A - B
        s = np.max(np.abs(diff), axis=1)
        s1 = np.where(s > 0, s, 1.0)
        d = np.sqrt(np.sum((diff / s1[:, None]) ** 2, axis=1)) * s
    return np.where(finite, d, np.inf)


def _pair_energy(T_A, T_B, m, rows, cols, s):
    """Scale of the float32 error in raw d**2 (in units of s**2): the pair's
    centered energies plus its mean-offset term."""
    A, B = _windows(T_A, m)[rows] / s, _windows(T_B, m)[cols] / s
    Ac, Bc = A - A.mean(axis=1)[:, None], B - B.mean(axis=1)[:, None]
    return (
        np.sum(Ac * Ac, axis=1)
        + np.sum(Bc * Bc, axis=1)
        + m * (A.mean(axis=1) - B.mean(axis=1)) ** 2
    )


def _assert_near_optimal(name, T_A, T_B, m, rows, got, best_d, best_idx, normalize, ca, cb, s):
    """Each chosen neighbour is within the float32 tie tolerance of the
    oracle's optimum, and each reported distance is its direct evaluation."""
    d = _pair_dist(T_A, T_B, m, rows, got, normalize, ca, cb)
    assert np.all(np.isfinite(d)), f"{name}: a neighbour with a non-finite window"
    if normalize:
        gap = d - best_d
        assert np.all(gap <= tie_tolerance(m)), f"{name}: max gap {gap.max():.3g}"
    else:
        e = np.maximum(
            _pair_energy(T_A, T_B, m, rows, got, s), _pair_energy(T_A, T_B, m, rows, best_idx, s)
        )
        egap = (d / s) ** 2 - (best_d / s) ** 2
        assert np.all(egap <= 2e-6 * e), f"{name}: max energy gap {np.max(egap / e):.3g}"
    return d


_STRATA = [
    (join, tiled, k, normalize)
    for join in ("self", "ab", "ab_equal")
    for tiled in (False, True)
    for k in (1, 5)
    for normalize in (True, False)
]
_FUZZ_CASES = int(os.environ.get("MLX_STUMP_FUZZ_CASES", len(_STRATA)))


def _fuzz_id(case):
    join, tiled, k, normalize = _STRATA[case % len(_STRATA)]
    mode = "norm" if normalize else "raw"
    return f"{case}-{join}-{'tiled' if tiled else 'dense'}-k{k}-{mode}"


@pytest.mark.parametrize("case", range(_FUZZ_CASES), ids=_fuzz_id)
def test_seeded_differential_fuzz(monkeypatch, case):
    join, tiled, k, normalize = _STRATA[case % len(_STRATA)]
    rng = np.random.default_rng(6_000 + case)
    m = int(rng.choice([3, 4, 5, 7, 8, 12, 16, 25, 33, 50, 64]))
    T_A = _fuzz_series(str(rng.choice(_FAMILIES)), int(rng.integers(m, 601)), rng)
    if join == "self":
        T_B = None
    elif join == "ab":
        T_B = _fuzz_series(str(rng.choice(_FAMILIES)), int(rng.integers(m, 601)), rng)
    else:
        T_B = T_A.copy()
    ignore_trivial = bool(rng.random() < 0.5)
    B = T_A if T_B is None else T_B
    self_join = T_B is None or (ignore_trivial and np.array_equal(T_A, T_B, equal_nan=True))
    l_q, l_b = T_A.size - m + 1, B.size - m + 1

    kwargs = dict(ignore_trivial=ignore_trivial, normalize=normalize, k=k)
    const_A = const_B = None
    if normalize and rng.random() < 0.15:  # custom constant flags (a superset of the true ones)
        const_A = const_B = _window_flags(T_A, m)[1] | (rng.random(l_q) < 0.05)
        if join == "ab":
            const_B = _window_flags(T_B, m)[1] | (rng.random(l_b) < 0.05)
        kwargs["T_A_subseq_isconstant"] = const_A
        if T_B is not None:
            kwargs["T_B_subseq_isconstant"] = const_B
    # chunk_size=1 is excluded: its GEMV kernel may resolve float32 ties
    # differently (documented). Tile widths are log-uniform so blocks
    # narrower than k are common; the engine floors a block at 4 windows.
    chunk = [None, 2, 3, 7, 64, 173, 1000][int(rng.integers(0, 7))]
    rows = int(2 ** rng.uniform(0, 6))

    def dispatches():
        return -(-l_q // (chunk or l_q)) * (-(-l_b // max(4, rows)) if tiled else 1)

    while dispatches() > 200:  # bounded runtime; an automatic chunk is one batch here
        if tiled and max(4, rows) < l_b:
            rows *= 2
        else:
            chunk *= 2
    if tiled:
        _force_tiled(monkeypatch, 4 * m * rows)

    def run(fn, **extra):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = fn(T_A, m, T_B, **kwargs, **extra)
        return out, sorted(str(w.message) for w in caught if "implies" in str(w.message))

    ours, warned = run(mlx_stump.stump, chunk_size=chunk)
    ref, ref_warned = run(stumpy.stump)
    assert warned == ref_warned  # join disambiguation replicates STUMPY's
    assert ours.shape == ref.shape

    excl = int(np.ceil(m / stumpy.config.STUMPY_EXCL_ZONE_DENOM))
    D, ca, cb = _oracle(T_A, B, m, normalize, excl if self_join else None, const_A, const_B)
    s = _frame(T_A, B)
    order = np.argsort(D, axis=1, kind="stable")[:, :k]
    best = np.take_along_axis(D, order, axis=1)
    if best.shape[1] < k:  # fewer target windows than k
        pad = k - best.shape[1]
        best = np.pad(best, ((0, 0), (0, pad)), constant_values=np.inf)
        order = np.pad(order, ((0, 0), (0, pad)), constant_values=-1)
    P, I = _columns(ours, k)
    Pr, Ir = _columns(ref, k)
    np.testing.assert_array_equal(I == -1, ~np.isfinite(best))
    np.testing.assert_array_equal(I == -1, Ir == -1)
    np.testing.assert_array_equal(np.isinf(P), I == -1)
    np.testing.assert_array_equal(np.isinf(P), np.isinf(Pr))
    if k > 1:  # distinct neighbours, ascending distances
        for row in I:
            assert np.unique(row[row >= 0]).size == np.count_nonzero(row >= 0)
        with np.errstate(invalid="ignore"):  # inf - inf in the padded tail
            assert np.all(np.diff(P, axis=1)[np.isfinite(P[:, 1:])] >= 0)

    for j in range(k):
        r = np.nonzero(I[:, j] >= 0)[0]
        if self_join:
            assert np.all(np.abs(I[r, j] - r) > excl), "neighbour inside the exclusion zone"
        d = _assert_near_optimal(
            f"column {j}", T_A, B, m, r, I[r, j], best[r, j], order[r, j], normalize, ca, cb, s
        )
        if normalize:  # d**2 is bounded by 4m; compare on that scale near 0
            err = np.abs(P[r, j] ** 2 - d**2) <= 1e-9 * np.maximum(d**2, m)
        else:
            err = np.abs(P[r, j] - d) <= 1e-9 * d
        assert np.all(err), f"column {j}: reported P is not the pair's float64 distance"

    row_idx, col_idx = np.indices(D.shape)
    for side in ("left", "right"):
        got = np.asarray(getattr(ours, f"{side}_I_"), dtype=np.int64)
        np.testing.assert_array_equal(got == -1, getattr(ref, f"{side}_I_") == -1)
        if not self_join:
            assert np.all(got == -1)
            continue
        on_side = col_idx < row_idx if side == "left" else col_idx > row_idx
        D_side = np.where(on_side, D, np.inf)
        side_idx = np.argmin(D_side, axis=1)
        side_best = D_side[np.arange(l_q), side_idx]
        np.testing.assert_array_equal(got == -1, ~np.isfinite(side_best))
        r = np.nonzero(got >= 0)[0]
        assert np.all(got[r] < r - excl if side == "left" else got[r] > r + excl)
        _assert_near_optimal(
            side, T_A, B, m, r, got[r], side_best[r], side_idx[r], normalize, ca, cb, s
        )
