"""Regression tests for the sixth review round: the ``stump`` entry point.

1. A numpy-integer ``chunk_size`` passed validation but was never converted
   to a Python int, so ``mx.arange`` rejected the numpy batch bounds (a
   TypeError whenever ``chunk_size <= l_q``). Small unsigned types would
   also have wrapped in the batch arithmetic. It is now converted like ``k``.
2. Self-join detection compared the raw arrays, so an explicit ``T_B`` that
   differed from ``T_A`` only in the marker of a missing sample (inf vs NaN,
   -inf vs +inf) became an AB-join: every row matched itself at distance 0.
   STUMPY compares after replacing non-finite values. The comparison now
   ignores the marker. STUMPY's zero fill also equates a missing sample with
   a real 0.0; that quirk is deliberately not copied.
3. The small-P diagnostic took the profile mean without ``under="ignore"``,
   so a raw profile at subnormal scale raised FloatingPointError under a
   caller's ``np.seterr(under="raise")`` after all the work was done.
4. Top-k refinement called the float64 refine once per neighbor column,
   normalizing every query window ``k`` times, on one core. The refine now
   takes the whole ``(l, k)`` index matrix and writes into ``P`` in place.
   Each row chunk normalizes its query windows once, and the target gather
   doubles as the difference buffer. Chunks run on a small per-call thread
   pool that splits the same memory budget; its threads pull chunks from
   one shared cursor, so the scheduling state is one task per thread, not
   one future per chunk. The output is bit-identical to the per-column
   code; a verbatim copy of that code is kept below as the reference.
5. The object output was filled one strided column at a time; it is now
   filled by whole blocks, which boxes the same Python floats and ints.
6. A pickled ``mparray`` lost ``_m``/``_k``/``_excl_zone_denom`` (numpy
   rebuilds subclasses through ``__array_finalize__(None)``), so ``P_``
   and ``I_`` raised after pickle, ``np.save`` or a process pool. It now
   pickles through its constructor.
7. ``aamp``, ``gpu_stump`` and ``gpu_aamp`` wrappers with STUMPY's exact
   positional signatures (``stump``'s fifth positional argument is
   ``normalize``, ``aamp``'s is ``p``, ``gpu_stump``'s is ``device_id``).
   Their warnings, including those raised in preprocessing helpers, point
   at the user's call.
8. The public signatures are annotated, and the ``stump`` docstring states
   the call-time exclusion-zone denominator.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import io
import pathlib
import pickle
import threading
import typing
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import mlx_stump
import mlx_stump._engine as eng
import mlx_stump._stump as st
from mlx_stump._mparray import mparray
from mlx_stump._preprocess import center_rows_stable, rowwise_l2_inplace

from .conftest import assert_indices_tie_tolerant, assert_profile_close, tie_tolerance

stumpy = pytest.importorskip("stumpy")


def _walk(n, seed):
    return np.random.default_rng(seed).standard_normal(n).cumsum()


# ------------------------------------------------ 1. numpy-integer chunk_size
_L_Q = 291  # > 256, so a uint8 wrap in the batch arithmetic would be caught


@pytest.mark.parametrize(
    "chunk", [np.int64(3), np.int32(7), np.uint8(3), np.uint8(255), np.int64(_L_Q)]
)
@pytest.mark.parametrize("tiled", [False, True])
def test_numpy_integer_chunk_size(monkeypatch, chunk, tiled):
    m = 10
    T = _walk(_L_Q + m - 1, seed=1)
    T_B = _walk(240, seed=2)
    if tiled:
        monkeypatch.setattr(eng, "_MATMUL_WINDOW_BYTES", 0)
        monkeypatch.setattr(eng, "_TILE_WINDOW_BYTES", 37 * m * 4)
    for kwargs in ({"k": 1}, {"T_B": T_B, "ignore_trivial": False, "k": 3, "normalize": False}):
        got = mlx_stump.stump(T, m, chunk_size=chunk, **kwargs)
        ref = mlx_stump.stump(T, m, chunk_size=int(chunk), **kwargs)
        np.testing.assert_array_equal(np.asarray(got, float), np.asarray(ref, float))


# ------------------------------------------ 2. self-join detection and markers
def _marker_pair(case):
    T_A = _walk(300, seed=21)
    T_B = T_A.copy()
    if case == "inf/nan":
        T_A[50], T_B[50] = np.inf, np.nan
    elif case == "-inf/+inf":
        T_A[50], T_B[50] = -np.inf, np.inf
    else:  # mixed markers at several positions
        T_A[[30, 120, 200]] = [np.nan, np.inf, -np.inf]
        T_B[[30, 120, 200]] = [np.inf, -np.inf, np.nan]
    return T_A, T_B


def _join_warnings(record):
    return sorted(str(w.message) for w in record if "Arrays T_A, T_B" in str(w.message))


@pytest.mark.parametrize("case", ["inf/nan", "-inf/+inf", "mixed"])
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("ignore_trivial", [True, False])
def test_self_join_detection_ignores_nonfinite_marker(case, normalize, ignore_trivial):
    T_A, T_B = _marker_pair(case)
    m = 12
    with warnings.catch_warnings(record=True) as ours_w:
        warnings.simplefilter("always")
        ours = mlx_stump.stump(T_A, m, T_B, ignore_trivial=ignore_trivial, normalize=normalize)
    with warnings.catch_warnings(record=True) as ref_w:
        warnings.simplefilter("always")
        if normalize:
            ref = stumpy.stump(T_A, m, T_B, ignore_trivial=ignore_trivial)
        else:
            ref = stumpy.aamp(T_A, m, T_B, ignore_trivial=ignore_trivial)
    assert _join_warnings(ours_w) == _join_warnings(ref_w)
    if ignore_trivial:
        assert not _join_warnings(ours_w)
        I = ours.I_
        assert not np.any(I == np.arange(I.size))
    else:
        assert _join_warnings(ours_w) == [
            "Arrays T_A, T_B are equal, which implies a self-join. "
            "Try setting `ignore_trivial = True`."
        ]
    tie = tie_tolerance(m)
    assert_profile_close(ours.P_, ref.P_, m=m, tie_atol=tie)
    for attr in ("I_", "left_I_", "right_I_"):
        assert_indices_tie_tolerant(
            getattr(ours, attr), getattr(ref, attr), T_A, T_B, m,
            normalize=normalize, tie_atol=tie,
        )


def test_missing_sample_against_zero_is_an_ab_join():
    """Documented divergence: STUMPY's zero fill makes NaN equal a real 0.0
    (and then mirrors wrong distances onto the NaN rows); here the series
    differ, so the join is an AB-join with STUMPY's own warning."""
    T_A = _walk(200, seed=22)
    T_B = T_A.copy()
    T_A[60], T_B[60] = np.nan, 0.0
    with pytest.warns(UserWarning, match="are not equal, which implies an AB-join"):
        mp = mlx_stump.stump(T_A, 10, T_B)
    assert np.all(mp.left_I_ == -1) and np.all(mp.right_I_ == -1)


def test_different_shapes_are_an_ab_join():
    T_A = _walk(200, seed=23)
    with pytest.warns(UserWarning, match="are not equal"):
        mlx_stump.stump(T_A, 10, T_A[:150].copy())


# ------------------------------------------------ 3. small-P check underflow
def test_raw_stump_small_p_check_ignores_caller_underflow_policy():
    T = np.random.default_rng(7).standard_normal(500).cumsum() * 2.0**-1060
    with np.errstate(all="raise"), pytest.warns(UserWarning, match="smaller than"):
        out = mlx_stump.stump(T, 20, normalize=False)
    assert np.all(np.asarray(out[:, 0], float) > 0)


# ------------------------------------------ 4. one 2-D, threaded refinement
def _ref_refine_znorm(query, target, I):
    """Verbatim ``_refine_znorm`` at 4b0b29a (one neighbor column per call)."""
    refine_chunk_rows = eng.refine_chunk_rows
    P_NORM_THRESHOLD = st.P_NORM_THRESHOLD
    m = query.m
    out = np.full(I.shape, np.inf)
    valid = np.nonzero(I >= 0)[0]
    if valid.size == 0:
        return out
    WQ = np.lib.stride_tricks.sliding_window_view(query.T, m)
    WT = np.lib.stride_tricks.sliding_window_view(target.T, m)
    dmax = 2.0 * np.sqrt(m)  # rho >= -1; sqrt rounding can overshoot 1 ulp
    chunk = refine_chunk_rows(m)
    for s in range(0, valid.size, chunk):
        qi = valid[s : s + chunk]
        tj = I[qi]
        qc = query.isconstant[qi]
        tc = target.isconstant[tj]
        # raw windows may hold NaN/inf; those rows are overwritten with inf
        # below, so let the intermediate arithmetic run silently
        with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
            qw = WQ[qi]  # fancy indexing copies, so everything can be in place
            center_rows_stable(qw)
            sq = np.sqrt(np.sum(qw * qw, axis=1) / m)
            tw = WT[tj]
            center_rows_stable(tw)
            st_ = np.sqrt(np.sum(tw * tw, axis=1) / m)
            sq_inv = np.where((sq > 0.0) & ~qc, 1.0 / np.where(sq > 0.0, sq, 1.0), 0.0)
            st_inv = np.where((st_ > 0.0) & ~tc, 1.0 / np.where(st_ > 0.0, st_, 1.0), 0.0)
            qw *= sq_inv[:, None]
            tw *= st_inv[:, None]
            qw -= tw
            d2 = np.einsum("ij,ij->i", qw, qw)
            # a zero sig_inv on either side (constant flag, or a truly flat
            # window flagged non-constant) means rho == 0 on the GPU; mirror
            # that, and let the constant-flag rules below overwrite as needed
            d2 = np.where((sq_inv == 0.0) | (st_inv == 0.0), 2.0 * m, d2)
            d2[~np.isfinite(d2)] = 2.0 * m  # NaN/inf windows: masked below
            d2[d2 < P_NORM_THRESHOLD] = 0.0
            d = np.minimum(np.sqrt(d2), dmax)
        d = np.where(qc & tc, 0.0, np.where(qc ^ tc, np.sqrt(m), d))
        d[~(query.isfinite[qi] & target.isfinite[tj])] = np.inf
        out[qi] = d
    return out


def _ref_refine_absolute(query, target, I):
    """Verbatim ``_refine_absolute`` at 4b0b29a (one neighbor column per call)."""
    refine_chunk_rows = eng.refine_chunk_rows
    m = query.m
    out = np.full(I.shape, np.inf)
    valid = np.nonzero(I >= 0)[0]
    if valid.size == 0:
        return out
    WQ = np.lib.stride_tricks.sliding_window_view(query.T, m)
    WT = np.lib.stride_tricks.sliding_window_view(target.T, m)
    chunk = refine_chunk_rows(m)
    for s in range(0, valid.size, chunk):
        qi = valid[s : s + chunk]
        tj = I[qi]
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            diff = WQ[qi] - WT[tj]
        d = rowwise_l2_inplace(diff)
        d[~(query.isfinite[qi] & target.isfinite[tj])] = np.inf
        out[qi] = d
    return out


def _refine_case(name):
    """(T_A, m, stump kwargs) for the bit-identity cases."""
    rng = np.random.default_rng(31)
    walk = rng.standard_normal(420).cumsum()
    if name == "nan_inf":
        T = walk.copy()
        T[[40, 200, 330]] = [np.nan, np.inf, -np.inf]
        return T, 20, {"k": 4}
    if name == "constant_runs":
        T = rng.standard_normal(420)
        T[100:150] = 3.0
        T[260:300] = -1.5
        return T, 16, {"k": 3}
    if name == "offset_1e6":
        return walk + 1e6, 20, {"k": 5}
    if name == "scale_1e-200":
        return walk * 1e-200, 20, {"k": 3}
    if name == "scale_1e200":
        return walk * 1e200, 20, {"k": 3}
    if name == "user_isconstant":
        flags = rng.random(420 - 15 + 1) < 0.2
        return walk, 15, {"k": 3, "T_A_subseq_isconstant": flags}
    if name == "near_duplicates":
        T = rng.standard_normal(420)
        pat = rng.standard_normal(12)
        for p in (40, 170, 300):
            T[p : p + 12] = pat * (1 + 1e-9 * rng.standard_normal(12))
        return T, 12, {"k": 3}
    if name == "m3":
        return walk[:200], 3, {"k": 6}
    if name == "all_nan_rows":
        T = walk.copy()
        T[150:190] = np.nan
        return T, 10, {"k": 3}
    if name == "ab_short_target":  # l_B = 31 < k: trailing -1 columns
        return walk[:300], 10, {"k": 40, "T_B": rng.standard_normal(40)}
    if name == "k1":
        return walk, 25, {}
    if name == "threaded_default_budget":  # l > rows // workers at the default budget
        return rng.standard_normal(12000).cumsum(), 16, {"k": 2}
    raise AssertionError(name)


_REFINE_CASES = [
    "nan_inf",
    "constant_runs",
    "offset_1e6",
    "scale_1e-200",
    "scale_1e200",
    "user_isconstant",
    "near_duplicates",
    "m3",
    "all_nan_rows",
    "ab_short_target",
    "k1",
    "threaded_default_budget",
]


def _capture_refine_inputs(monkeypatch, normalize, T, m, kwargs):
    """Run stump and capture the real (A, B, I) handed to the refinement."""
    seen = []
    name = "_refine_znorm" if normalize else "_refine_absolute"
    real = getattr(st, name)

    def spy(A, B, I, out=None):
        seen.append((A, B, I.copy()))
        return real(A, B, I, out=out)

    with monkeypatch.context() as mp, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mp.setattr(st, name, spy)
        mlx_stump.stump(T, m, normalize=normalize, **kwargs)
    ((A, B, I),) = seen
    return A, B, I, real


class _CountingPool(ThreadPoolExecutor):
    made = 0

    def __init__(self, *args, **kwargs):
        type(self).made += 1
        super().__init__(*args, **kwargs)


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("case", _REFINE_CASES)
def test_refinement_bit_identical_to_per_column_code(monkeypatch, case, normalize):
    T, m, kwargs = _refine_case(case)
    if not normalize:
        kwargs.pop("T_A_subseq_isconstant", None)
    A, B, I, refine = _capture_refine_inputs(monkeypatch, normalize, T, m, kwargs)
    old = _ref_refine_znorm if normalize else _ref_refine_absolute
    ref = np.column_stack([old(A, B, I[:, j]) for j in range(I.shape[1])])
    if case == "ab_short_target":
        assert np.all(I[:, 31:] == -1)
    # 1- and 7-row chunks are exercised on the small cases; on 12k rows
    # they would only add thousands of tiny serial chunks
    budgets = (None, 64) if case == "threaded_default_budget" else (None, 1, 7, 64)
    for rows in budgets:
        for workers in (1, 8):
            with monkeypatch.context() as mp:
                if rows is not None:
                    mp.setattr(eng, "_REFINE_MEM_BUDGET", rows * m * 8 * 4)
                    assert eng.refine_chunk_rows(m) == rows
                mp.setattr(st, "_REFINE_MAX_WORKERS", workers)
                _CountingPool.made = 0
                mp.setattr(st, "ThreadPoolExecutor", _CountingPool)
                got = np.full(I.shape, np.nan)
                assert refine(A, B, I, out=got) is got
                one = refine(A, B, I[:, 0])  # 1-D indices, allocated output
                budget_rows = eng.refine_chunk_rows(m)
            assert got.view(np.uint64).tolist() == ref.view(np.uint64).tolist(), (rows, workers)
            assert one.view(np.uint64).tolist() == ref[:, 0].view(np.uint64).tolist()
            # the threaded path really ran whenever there was more than one chunk
            w = min(workers, st._cpu_count(), budget_rows)
            pooled = w > 1 and I.shape[0] > budget_rows // w
            assert _CountingPool.made == (2 if pooled else 0)  # one pool per refine call
            if rows == 64 and workers == 8 and st._cpu_count() > 1:
                assert pooled


def test_refinement_worker_error_propagates(monkeypatch):
    """A failing chunk raises in the caller rather than leaving stale rows."""
    T = _walk(600, seed=41)
    A, B, I, refine = _capture_refine_inputs(monkeypatch, True, T, 10, {"k": 2})
    monkeypatch.setattr(eng, "_REFINE_MEM_BUDGET", 64 * 10 * 8 * 4)
    real = st._znorm_rows
    calls = []

    def flaky(w, isconstant, m):
        calls.append(1)
        if len(calls) == 5:
            raise MemoryError("synthetic")
        return real(w, isconstant, m)

    monkeypatch.setattr(st, "_znorm_rows", flaky)
    with pytest.raises(MemoryError, match="synthetic"):
        refine(A, B, I, out=np.empty(I.shape))


class _SubmitCountingPool(ThreadPoolExecutor):
    tasks = 0

    def submit(self, *args, **kwargs):
        type(self).tasks += 1
        return super().submit(*args, **kwargs)


def _two_row_chunks(monkeypatch, m):
    """16-row refinement budget on 8 threads: 2-row chunks, pooled even on
    a one-CPU runner."""
    monkeypatch.setattr(eng, "_REFINE_MEM_BUDGET", 16 * m * 8 * 4)
    monkeypatch.setattr(st, "_REFINE_MAX_WORKERS", 8)
    monkeypatch.setattr(st, "_cpu_count", lambda: 8)
    monkeypatch.setattr(st, "ThreadPoolExecutor", _SubmitCountingPool)
    _SubmitCountingPool.tasks = 0


def test_refine_scheduling_holds_one_task_per_thread(monkeypatch):
    """The pool gets one task per thread rather than one future per chunk
    (~l*m/2**20 chunks at large m: 167 MiB of scheduling state at l=1.95e6,
    m=5e4), and at most one chunk of rows is in flight at once."""
    m, l = 10, 20001  # 10001 two-row chunks
    _two_row_chunks(monkeypatch, m)
    lock = threading.Lock()
    live = [0, 0]  # rows in flight, most seen
    spans = []

    def job(s, e):
        with lock:
            spans.append((s, e))
            live[0] += e - s
            live[1] = max(live[1], live[0])
        np.sqrt(np.arange(2000.0))  # release the GIL briefly, like a real chunk
        with lock:
            live[0] -= e - s

    st._map_refine_chunks(job, l, m)
    assert _SubmitCountingPool.tasks == 8
    assert sorted(spans) == [(s, min(s + 2, l)) for s in range(0, l, 2)]
    assert live[1] <= 16


def test_refine_failure_stops_the_other_threads(monkeypatch):
    m, l = 10, 20001
    _two_row_chunks(monkeypatch, m)
    lock = threading.Lock()
    calls = []

    def job(s, e):
        with lock:
            calls.append(s)
            n = len(calls)
        if n == 3:
            raise MemoryError("synthetic")

    with pytest.raises(MemoryError, match="synthetic"):
        st._map_refine_chunks(job, l, m)
    # each other thread finishes at most the chunk it holds and one more
    assert len(calls) <= 3 + 2 * 8


# ------------------------------------------------------ 5. block assembly
@pytest.mark.parametrize("k", [1, 4])
def test_object_output_boxes_python_scalars(k):
    T = _walk(300, seed=51)
    mp = mlx_stump.stump(T, 12, k=k)
    assert mp.dtype == object and mp.shape == (289, 2 * k + 2)
    for j in range(k):
        assert type(mp[5, j]) is float
        assert type(mp[5, k + j]) is int
    assert type(mp[5, 2 * k]) is int and type(mp[5, 2 * k + 1]) is int
    P = np.asarray(mp[:, :k], float)
    assert np.all(np.diff(P, axis=1) >= 0)  # columns still ascending


# ---------------------------------------------------------- 6. pickling
def _same_mp(a, b):
    assert type(a) is mparray
    assert (a._m, a._k, a._excl_zone_denom) == (b._m, b._k, b._excl_zone_denom)
    for attr in ("P_", "I_", "left_I_", "right_I_"):
        np.testing.assert_array_equal(getattr(a, attr), getattr(b, attr))


@pytest.mark.parametrize("k", [1, 3])
def test_mparray_pickle_round_trip(k):
    mp = mlx_stump.stump(_walk(260, seed=61), 11, k=k)
    for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
        _same_mp(pickle.loads(pickle.dumps(mp, protocol=proto)), mp)
    buf = io.BytesIO()
    np.save(buf, mp, allow_pickle=True)
    buf.seek(0)
    _same_mp(np.load(buf, allow_pickle=True), mp)
    for part in (mp[10:40], mp[::3]):
        _same_mp(pickle.loads(pickle.dumps(part)), part)


def test_mparray_pickle_loads_without_reduce(monkeypatch):
    """The pickle is a plain constructor call, so an mlx-stump without the
    new ``__reduce__`` (or with a different one) still loads it intact."""
    mp = mlx_stump.stump(_walk(200, seed=62), 10, k=2)
    blob = pickle.dumps(mp)
    monkeypatch.delattr(mparray, "__reduce__")
    _same_mp(pickle.loads(blob), mp)


# ------------------------------------------------- 7. STUMPY-named wrappers
def _stumpy_source_signature(module, func):
    """(name, default) pairs of a STUMPY function, read from its source.

    ``stumpy.gpu_stump``/``gpu_aamp`` cannot be imported without a CUDA
    driver, so their signatures are parsed rather than inspected.
    """
    pkg = pathlib.Path(importlib.util.find_spec("stumpy").origin).parent
    tree = ast.parse((pkg / f"{module}.py").read_text())
    (fn,) = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == func]
    args = fn.args
    assert not args.kwonlyargs and args.vararg is None and args.kwarg is None
    defaults = [inspect.Parameter.empty] * (len(args.args) - len(args.defaults))
    defaults += [ast.literal_eval(d) for d in args.defaults]
    return [(a.arg, d) for a, d in zip(args.args, defaults, strict=True)]


def _positional(func):
    params = list(inspect.signature(func).parameters.values())
    kw = [p for p in params if p.kind is p.KEYWORD_ONLY]
    assert [(p.name, p.default) for p in kw] == [("chunk_size", None)]
    pos = [p for p in params if p.kind is not p.KEYWORD_ONLY]
    assert all(p.kind is p.POSITIONAL_OR_KEYWORD for p in pos)
    return [(p.name, p.default) for p in pos]


def test_wrapper_signatures_match_stumpy():
    assert _positional(mlx_stump.stump) == _positional_ref(stumpy.stump)
    assert _positional(mlx_stump.aamp) == _positional_ref(stumpy.aamp)
    assert _positional(mlx_stump.gpu_stump) == _stumpy_source_signature(
        "gpu_stump", "gpu_stump"
    )
    assert _positional(mlx_stump.gpu_aamp) == _stumpy_source_signature("gpu_aamp", "gpu_aamp")
    for name in ("aamp", "gpu_stump", "gpu_aamp"):
        assert name in mlx_stump.__all__


def _positional_ref(func):
    return [(p.name, p.default) for p in inspect.signature(func).parameters.values()]


def _assert_matches(ours, ref, T_A, T_B, m, k, normalize):
    tie = tie_tolerance(m)
    P = np.atleast_2d(ours.P_.T).T
    Pr = np.atleast_2d(ref.P_.T).T
    I = np.atleast_2d(ours.I_.T).T
    Ir = np.atleast_2d(ref.I_.T).T
    for j in range(k):
        assert_profile_close(P[:, j], Pr[:, j], m=m, tie_atol=tie)
        assert_indices_tie_tolerant(
            I[:, j], Ir[:, j], T_A, T_B, m, normalize=normalize, tie_atol=tie
        )
    for attr in ("left_I_", "right_I_"):
        assert_indices_tie_tolerant(
            getattr(ours, attr), getattr(ref, attr), T_A, T_B, m,
            normalize=normalize, tie_atol=tie,
        )


def test_wrappers_positional_golden():
    T = _walk(500, seed=71)
    T_B = _walk(350, seed=72)
    m = 16
    # stumpy.aamp's fifth positional argument is p, sixth k
    _assert_matches(
        mlx_stump.aamp(T, m, None, True, 2.0, 3), stumpy.aamp(T, m, None, True, 2.0, 3),
        T, T, m, 3, False,
    )
    _assert_matches(
        mlx_stump.aamp(T, m, T_B, False, 2.0, 2), stumpy.aamp(T, m, T_B, False, 2.0, 2),
        T, T_B, m, 2, False,
    )
    # gpu_stump(T_A, m, T_B, ignore_trivial, device_id, normalize, p, k, ...);
    # stumpy.gpu_stump needs CUDA, so compare with the CPU functions it mirrors
    _assert_matches(
        mlx_stump.gpu_stump(T, m, None, True, 0, True, 2.0, 2),
        stumpy.stump(T, m, None, True, True, 2.0, 2),
        T, T, m, 2, True,
    )
    _assert_matches(
        mlx_stump.gpu_stump(T, m, T_B, False, [0, 1], False, 2.0, 1),
        stumpy.aamp(T, m, T_B, False, 2.0, 1),
        T, T_B, m, 1, False,
    )
    _assert_matches(
        mlx_stump.gpu_aamp(T, m, None, True, 0, 2.0, 3),
        stumpy.aamp(T, m, None, True, 2.0, 3),
        T, T, m, 3, False,
    )


def test_wrappers_run_the_same_computation():
    T = _walk(400, seed=73)
    T_B = _walk(300, seed=74)
    flags = np.random.default_rng(75).random(400 - 12 + 1) < 0.1
    pairs = [
        (mlx_stump.aamp(T, 12, k=2), mlx_stump.stump(T, 12, normalize=False, k=2)),
        (
            mlx_stump.gpu_aamp(T, 12, T_B, False, device_id=[0, 1], chunk_size=37),
            mlx_stump.stump(T, 12, T_B, False, normalize=False, chunk_size=37),
        ),
        (
            mlx_stump.gpu_stump(T, 12, T_A_subseq_isconstant=flags, k=3),
            mlx_stump.stump(T, 12, T_A_subseq_isconstant=flags, k=3),
        ),
    ]
    for got, ref in pairs:
        _same_mp(got, ref)


@pytest.mark.parametrize("device_id", [0, 3, np.int64(1), [0, 1], (0,), np.arange(2)])
def test_device_id_accepted_and_ignored(device_id):
    T = _walk(200, seed=76)
    ref = mlx_stump.stump(T, 10)
    _same_mp(mlx_stump.gpu_stump(T, 10, device_id=device_id), ref)
    _same_mp(
        mlx_stump.gpu_aamp(T, 10, device_id=device_id), mlx_stump.stump(T, 10, normalize=False)
    )


@pytest.mark.parametrize("device_id", [-1, True, 1.5, "0", [], [0, "1"], None])
def test_device_id_rejected(device_id):
    T = _walk(100, seed=77)
    for fn in (mlx_stump.gpu_stump, mlx_stump.gpu_aamp):
        with pytest.raises(ValueError, match="device_id"):
            fn(T, 10, device_id=device_id)


def test_aamp_rejects_other_p_norms():
    T = _walk(100, seed=78)
    for call in (
        lambda: mlx_stump.aamp(T, 10, p=1.0),
        lambda: mlx_stump.gpu_aamp(T, 10, p=1.0),
        lambda: mlx_stump.gpu_stump(T, 10, normalize=False, p=1.0),
    ):
        with pytest.raises(NotImplementedError, match="p=2.0 only"):
            call()


def _raw_precision_series():
    rng = np.random.default_rng(79)
    T = rng.standard_normal(400)
    T[150:200] += 1e17 * rng.standard_normal(50)
    return T


def _call(entry, T_A, m, T_B=None, ignore_trivial=True, normalize=True, flags=None):
    """One public entry point, called directly from this file."""
    if entry == "stump":
        return mlx_stump.stump(
            T_A, m, T_B, ignore_trivial, normalize, T_A_subseq_isconstant=flags
        )
    if entry == "gpu_stump":
        return mlx_stump.gpu_stump(
            T_A, m, T_B, ignore_trivial, normalize=normalize, T_A_subseq_isconstant=flags
        )
    assert not normalize and flags is None
    if entry == "aamp":
        return mlx_stump.aamp(T_A, m, T_B, ignore_trivial)
    return mlx_stump.gpu_aamp(T_A, m, T_B, ignore_trivial)


@pytest.mark.parametrize("entry", ["stump", "aamp", "gpu_stump", "gpu_aamp"])
def test_warnings_point_at_the_callers_line(entry):
    """Every warning, raised in the implementation or in a preprocessing
    helper one frame deeper, is attributed to this file (the caller)."""
    raw = entry.endswith("aamp")
    T = _walk(300, seed=80)
    calls = [
        # join disambiguation + small-P diagnostic (implementation frame)
        ("are equal", dict(T_A=T, m=10, T_B=T.copy(), ignore_trivial=False)),
        # window-size advisory (check_window_size)
        ("may be too large", dict(T_A=T[:30].copy(), m=20)),
        # raw standardization-limit warning (preprocess_series)
        ("standardization limit", dict(T_A=_raw_precision_series(), m=10, normalize=False)),
    ]
    if not raw:
        T_nan = T.copy()
        T_nan[100] = np.nan
        flags = np.ones(300 - 10 + 1, dtype=bool)
        # constant flags switched off at NaN windows (preprocess_series)
        calls.append(("automatically switched", dict(T_A=T_nan, m=10, flags=flags)))
    for expected, kwargs in calls:
        if raw:
            kwargs["normalize"] = False
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            _call(entry, **kwargs)
        assert any(expected in str(w.message) for w in rec), (entry, expected)
        assert {w.filename for w in rec} == {__file__}, [
            (w.filename, str(w.message)[:40]) for w in rec
        ]


# ------------------------------------------- 8. annotations and docstring
def test_public_annotations_resolve():
    for fn in (mlx_stump.stump, mlx_stump.aamp, mlx_stump.gpu_stump, mlx_stump.gpu_aamp):
        hints = typing.get_type_hints(fn)
        assert hints["return"] is mparray
        assert "T_A" in hints and "chunk_size" in hints
    assert "STUMPY_EXCL_ZONE_DENOM" in mlx_stump.stump.__doc__
