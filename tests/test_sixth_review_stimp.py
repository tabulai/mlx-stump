"""Sixth review round: the GPU pan matrix profile, ``stimp`` / ``gpu_stimp``.

The upgrade review found that STUMPY's pan matrix profile could not use
mlx-stump at all (``_stimp`` always hands ``stump`` a callable constant-flag
function, which was unsupported) and proposed a native ``stimp`` once
callables work. This round adds it:

1. ``mlx_stump.stimp(T, min_m, max_m, step, normalize, p,
   T_subseq_isconstant_func)`` computes one exact GPU matrix profile per
   ``update()``: the result of ``stumpy.stimp(..., percentage=1.0,
   pre_scrump=False)``, which is what ``stumpy.gpu_stimp`` computes.
   ``normalize=False`` follows ``stumpy.aamp_stimp`` (``p=2.0`` only).
2. The window sizes ``M_`` are visited in STUMPY's breadth-first order; the
   default ``max_m`` is STUMPY's largest self-join window under the
   exclusion-zone denominator in effect, and an explicit range is swapped
   and clamped as STUMPY does.
3. ``pan()`` reproduces STUMPY's row layout (+inf padding past each
   profile), normalization (including ``aamp_stimp``'s value-range factor),
   contrast, binarization, clipping, block repetition and NaN fill; the
   binary ``PAN_`` is identical to STUMPY's on random-walk and sine+noise
   data.
4. ``gpu_stimp`` has ``stumpy.gpu_stimp``'s exact positional signature;
   ``device_id`` is validated and ignored.
5. Warnings from the per-window ``stump`` calls point at the user's
   ``update()`` line.
6. Review follow-ups: ``P_`` returns read-only views instead of copies (no
   second copy of the pan array); with ``normalize=False`` a constant-flag
   function is validated once rather than evaluated at every update; on
   exact-tie data (exactly periodic series) the profiles still agree with
   STUMPY although the rank-based contrast may order the ties differently.
"""

from __future__ import annotations

import ast
import functools
import importlib.util
import inspect
import pathlib
import typing
import warnings

import numpy as np
import pytest

import mlx_stump
from mlx_stump._stimp import _bfs_order

from .conftest import assert_profile_close, random_walk, sine_noise, tie_tolerance, with_nans

stumpy = pytest.importorskip("stumpy")


def _ref_stimp(T, *, normalize=True, **kw):
    """The exact-profile STUMPY oracle (``percentage=1.0``, no pre-scrump)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # its window-size advisories
        if normalize:
            return stumpy.stimp(T, percentage=1.0, pre_scrump=False, **kw)
        return stumpy.aamp_stimp(T, percentage=1.0, pre_scraamp=False, **kw)


def _ref_profiles(ref):
    """STUMPY's ``P_`` (added in 1.14), rebuilt from its PAN on older
    versions (the lowest-dependency CI job runs STUMPY 1.13)."""
    if hasattr(ref, "P_"):
        return ref.P_
    n = len(ref._T)
    return [ref._PAN[row][: n - m + 1] for row, m in zip(ref._bfs_indices, ref._M, strict=True)]


def _update(pmp, count):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(count):
            pmp.update()


def _pan_atol(pmp, normalize):
    """Tie tolerance of a normalized ``pan`` value: the profile's tie
    tolerance times the row's normalization factor."""
    M = pmp.M_.astype(np.float64)
    if normalize:
        scale = 1.0 / (2.0 * np.sqrt(M))
    else:
        finite = pmp._T[np.isfinite(pmp._T)]
        scale = 1.0 / (np.ptp(finite) * np.sqrt(M))
    return max(tie_tolerance(int(m)) * s for m, s in zip(pmp.M_, scale, strict=True))


def _assert_matches(ours, ref, *, normalize=True, contrast_atol=1e-3):
    np.testing.assert_array_equal(ours.M_, ref.M_)
    assert ours.M_.dtype == np.int64
    P, Pr = ours.P_, _ref_profiles(ref)
    assert len(P) == len(Pr) == len(ours.M_)
    for m, row, row_ref in zip(ours.M_, P, Pr, strict=True):
        assert row.shape == row_ref.shape
        assert_profile_close(row, row_ref, m=int(m), tie_atol=tie_tolerance(int(m)))
    # the raw PAN rows, padding included, share one inf pattern
    np.testing.assert_array_equal(np.isinf(ours._PAN), np.isinf(ref._PAN))
    # the default (binary) PAN_ is identical
    np.testing.assert_array_equal(ours.PAN_, ref.PAN_)
    atol = _pan_atol(ours, normalize)
    np.testing.assert_allclose(
        ours.pan(binary=False, contrast=False),
        ref.pan(binary=False, contrast=False),
        atol=atol,
        rtol=0,
    )
    # contrast is rank based: a near-tie resolved the other way moves a
    # value by ~2.5 / (#values), far below this bound
    np.testing.assert_allclose(ours.pan(binary=False), ref.pan(binary=False), atol=contrast_atol)


# --------------------------------------------------- 2. window-size order
def test_bfs_order_matches_stumpy():
    for n in range(1, 301):
        np.testing.assert_array_equal(_bfs_order(n), stumpy.core._bfs_indices(n))
        assert _bfs_order(n).dtype == np.int64


# ------------------------------------------ 1, 3. parity with stumpy.stimp
@pytest.mark.parametrize("gen", [random_walk, sine_noise], ids=["random_walk", "sine_noise"])
@pytest.mark.parametrize(
    "n, kw",
    [
        (300, {}),  # the default window range (max_m=None)
        (1000, dict(min_m=5, max_m=120, step=3)),
    ],
)
def test_matches_stumpy_stimp_partial_and_full(gen, n, kw):
    T = gen(n, seed=11)
    ours = mlx_stump.stimp(T, **kw)
    ref = _ref_stimp(T, **kw)
    total = len(ref.M_)
    done = 0
    for count in (1, 3, 7, total):  # partial updates, then the full profile
        _update(ours, count - done)
        _update(ref, count - done)
        done = count
        _assert_matches(ours, ref)
    # past the end, update() is a no-op in both
    before = ours._PAN.copy()
    ours.update()
    np.testing.assert_array_equal(ours._PAN, before)


@pytest.mark.parametrize("gen", [random_walk, sine_noise], ids=["random_walk", "sine_noise"])
def test_matches_stumpy_stimp_step_8(gen):
    kw = dict(min_m=8, max_m=264, step=8)
    T = gen(2000, seed=5)
    ours = mlx_stump.stimp(T, **kw)
    ref = _ref_stimp(T, **kw)
    assert len(ours.M_) == 33
    _update(ours, 33)
    _update(ref, 33)
    _assert_matches(ours, ref)


@pytest.mark.parametrize("gen", [random_walk, sine_noise], ids=["random_walk", "sine_noise"])
@pytest.mark.parametrize("partial", [True, False])
def test_raw_mode_matches_stumpy_aamp_stimp(gen, partial):
    kw = dict(min_m=4, max_m=90, step=2)
    T = gen(800, seed=7)
    ours = mlx_stump.stimp(T, normalize=False, **kw)
    ref = _ref_stimp(T, normalize=False, **kw)
    count = 5 if partial else len(ref.M_)
    _update(ours, count)
    _update(ref, count)
    _assert_matches(ours, ref, normalize=False)


def test_raw_mode_scales_by_the_finite_value_range():
    T = random_walk(400, seed=3)
    T[50] = np.nan
    T[60] = np.inf
    ours = mlx_stump.stimp(T, min_m=10, max_m=40, normalize=False)
    ref = _ref_stimp(T, normalize=False, min_m=10, max_m=40)
    _update(ours, len(ours.M_))
    _update(ref, len(ref.M_))
    _assert_matches(ours, ref, normalize=False)


@pytest.mark.parametrize("flags", range(16))
def test_raw_mode_constant_series_matches_stumpy(flags):
    """A zero value range divides by zero in aamp_stimp's normalization;
    the NaN/1 outcome is STUMPY's, without floating-point warnings."""
    normalize, contrast, binary, clip = (bool(flags >> b & 1) for b in range(4))
    T = np.full(120, 2.5)
    ours = mlx_stump.stimp(T, 5, 30, 5, normalize=False)
    ref = _ref_stimp(T, normalize=False, min_m=5, max_m=30, step=5)
    _update(ours, 3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _update(ref, 3)
        want = ref.pan(normalize=normalize, contrast=contrast, binary=binary, clip=clip)
    with np.errstate(all="raise"):
        got = ours.pan(normalize=normalize, contrast=contrast, binary=binary, clip=clip)
    np.testing.assert_array_equal(got, want)  # NaN positions compare equal here


@pytest.fixture(scope="module")
def partial_pair():
    T = random_walk(500, seed=21)
    ours = mlx_stump.stimp(T, min_m=6, max_m=60, step=2)
    ref = _ref_stimp(T, min_m=6, max_m=60, step=2)
    _update(ours, 6)
    _update(ref, 6)
    return ours, ref


@pytest.mark.parametrize("flags", range(16))
@pytest.mark.parametrize("threshold", [0.1, 0.2, 0.5])
def test_pan_flags_match_stumpy(partial_pair, flags, threshold):
    """Every normalize/contrast/binary/clip combination after a partial
    update, which exercises the row repetition and the NaN fill."""
    normalize, contrast, binary, clip = (bool(flags >> b & 1) for b in range(4))
    ours, ref = partial_pair
    kw = dict(threshold=threshold, normalize=normalize, contrast=contrast, binary=binary, clip=clip)
    got, want = ours.pan(**kw), ref.pan(**kw)
    assert got.shape == want.shape == (len(ref.M_), 500)
    assert got.dtype == np.float64
    np.testing.assert_array_equal(np.isnan(got), np.isnan(want))
    np.testing.assert_allclose(got, want, atol=1e-9, rtol=0)


def test_nan_and_inf_windows_match_stumpy():
    T = with_nans(600, seed=4)
    ours = mlx_stump.stimp(T, min_m=5, max_m=70, step=5)
    ref = _ref_stimp(T, min_m=5, max_m=70, step=5)
    _update(ours, 4)
    _update(ref, 4)
    _assert_matches(ours, ref)
    _update(ours, len(ours.M_))
    _update(ref, len(ref.M_))
    _assert_matches(ours, ref)


def test_pan_before_any_update_is_all_nan_without_warning():
    T = random_walk(100, seed=1)
    ours = mlx_stump.stimp(T, min_m=5, max_m=20)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        PAN = ours.PAN_
    ref = _ref_stimp(T, min_m=5, max_m=20)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # STUMPY's nanmax of an all-NaN array
        ref_PAN = ref.PAN_
    assert PAN.shape == ref_PAN.shape == (16, 100)
    assert np.isnan(PAN).all() and np.isnan(ref_PAN).all()
    assert all(np.isinf(p).all() for p in ours.P_)


def test_raw_profiles_and_padding_layout():
    T = random_walk(200, seed=2)
    ours = mlx_stump.stimp(T, min_m=10, max_m=30, step=5)
    _update(ours, 2)
    P = ours.P_
    for i, (m, row) in enumerate(zip(ours.M_, P, strict=True)):
        assert row.shape == (200 - m + 1,) and row.dtype == np.float64
        pan_row = ours._PAN[ours._bfs_indices[i]]
        assert np.isinf(pan_row[200 - m + 1 :]).all()  # the padding
        if i < 2:
            np.testing.assert_array_equal(row, mlx_stump.stump(T, int(m)).P_)
        else:
            assert np.isinf(row).all()
    # P_ holds read-only views (no copy of the pan array, as in STUMPY, but
    # they cannot corrupt it); M_ is a copy
    for row in P:
        assert not row.flags.writeable and not row.flags.owndata
        assert np.shares_memory(row, ours._PAN)
    with pytest.raises(ValueError, match="read-only"):
        P[0][:] = 0.0
    M = ours.M_
    M[:] = 3
    assert np.isfinite(ours.P_[0]).all() and ours.P_[0].min() > 0.0
    assert ours.M_[0] == 20
    assert ours._PAN.flags.writeable  # the pan array itself stays writable
    # a view taken before an update is filled in place by it, as in STUMPY
    assert np.isinf(P[2]).all()
    _update(ours, 1)
    assert np.isfinite(P[2]).all()
    np.testing.assert_array_equal(P[2], mlx_stump.stump(T, int(ours.M_[2])).P_)


def test_custom_isconstant_func_matches_stumpy():
    def flat_below(a, w, tol):
        windows = np.lib.stride_tricks.sliding_window_view(a, w)
        return np.std(windows, axis=1) < tol

    rng = np.random.default_rng(9)
    T = random_walk(700, seed=9)
    T[200:320] = 5.0 + 1e-4 * rng.standard_normal(120)  # nearly flat
    T[500:620] = -3.0 + 1e-4 * rng.standard_normal(120)
    func = functools.partial(flat_below, tol=1e-3)
    kw = dict(min_m=6, max_m=60, step=6)
    ours = mlx_stump.stimp(T, T_subseq_isconstant_func=func, **kw)
    ref = _ref_stimp(T, T_subseq_isconstant_func=func, **kw)
    _update(ours, len(ours.M_))
    _update(ref, len(ref.M_))
    _assert_matches(ours, ref)
    # the flags change the result: the default rule sees the tiny noise
    default = mlx_stump.stimp(T, **kw)
    _update(default, len(default.M_))
    assert not np.array_equal(default._PAN, ours._PAN)


def test_raw_mode_validates_the_isconstant_func_once():
    """Raw distances ignore constant flags, so with ``normalize=False`` a
    user function is called once, by the first successful update, to
    validate it; a failing first update leaves the state unchanged."""
    T = random_walk(400, seed=21)
    kw = dict(min_m=5, max_m=40, step=5)
    calls = []

    def counting(a, w):
        calls.append(w)
        return np.zeros(len(a) - w + 1, dtype=bool)

    ours = mlx_stump.stimp(T, normalize=False, T_subseq_isconstant_func=counting, **kw)
    _update(ours, len(ours.M_))
    assert calls == [int(ours.M_[0])]
    plain = mlx_stump.stimp(T, normalize=False, **kw)
    _update(plain, len(plain.M_))
    np.testing.assert_array_equal(ours._PAN, plain._PAN)

    broken = mlx_stump.stimp(
        T, normalize=False, T_subseq_isconstant_func=lambda a, w: np.zeros(3, dtype=bool), **kw
    )
    for _ in range(2):  # still validated (and rejected) until an update succeeds
        with pytest.raises(ValueError, match="boolean array of shape"):
            broken.update()
    assert broken._n_processed == 0 and np.isinf(broken._PAN).all()

    # normalized mode keeps calling it: the flags matter for every window
    calls.clear()
    norm = mlx_stump.stimp(T, T_subseq_isconstant_func=counting, **kw)
    _update(norm, 3)
    assert calls == [int(m) for m in norm.M_[:3]]


def test_exact_tie_groups_keep_profiles_close():
    """Exactly periodic data has large groups of exactly tied distances.
    The rank-based contrast step may order those ties differently from
    STUMPY (documented), but the profiles and the normalized, uncontrasted
    pan still agree within float tolerance."""
    T = np.tile(np.random.default_rng(22).standard_normal(25), 16)
    kw = dict(min_m=4, max_m=60, step=4)
    ours = mlx_stump.stimp(T, **kw)
    ref = _ref_stimp(T, **kw)
    _update(ours, len(ours.M_))
    _update(ref, len(ref.M_))
    np.testing.assert_array_equal(ours.M_, ref.M_)
    for m, row, row_ref in zip(ours.M_, ours.P_, _ref_profiles(ref), strict=True):
        assert_profile_close(row, row_ref, m=int(m), tie_atol=1e-6)
    np.testing.assert_allclose(
        ours.pan(binary=False, contrast=False), ref.pan(binary=False, contrast=False), atol=1e-6
    )


# --------------------------------------------- 2. window range semantics
def test_min_m_greater_than_max_m_is_swapped():
    T = random_walk(300, seed=8)
    ours = mlx_stump.stimp(T, min_m=40, max_m=10, step=3)
    ref = _ref_stimp(T, min_m=40, max_m=10, step=3)
    np.testing.assert_array_equal(ours.M_, ref.M_)
    np.testing.assert_array_equal(ours.M_, mlx_stump.stimp(T, 10, 40, 3).M_)
    _update(ours, 3)
    _update(ref, 3)
    _assert_matches(ours, ref)


@pytest.mark.parametrize(
    "n, kw",
    [
        (5, {}),
        (6, {}),
        (37, {}),
        (300, dict(min_m=50)),
        (300, dict(min_m=5, step=4)),
        (300, dict(min_m=1, max_m=1000)),  # clamped to [3, max window]
        (300, dict(min_m=-4, max_m=12)),
        (60, dict(min_m=100)),  # max_m=None: not clamped, as in STUMPY
        (60, dict(min_m=1)),
    ],
)
def test_window_range_matches_stumpy(n, kw):
    T = random_walk(n, seed=1)
    np.testing.assert_array_equal(mlx_stump.stimp(T, **kw).M_, _ref_stimp(T, **kw).M_)


def test_unclamped_default_range_fails_at_the_offending_update():
    """With max_m=None STUMPY does not clamp: m > len(T) (or m < 3) raises
    only when that window's update runs, and the state is unchanged."""
    T = random_walk(60, seed=1)
    ours = mlx_stump.stimp(T, min_m=100)
    ref = _ref_stimp(T, min_m=100)
    np.testing.assert_array_equal(ours.M_, [101, 100])
    for pmp in (ours, ref):
        with pytest.raises(ValueError):
            pmp.update()
    with pytest.raises(ValueError, match="less than or equal to 60"):
        ours.update()
    assert ours._n_processed == 0

    ours = mlx_stump.stimp(T, min_m=2)
    assert 2 in ours.M_
    last = list(ours.M_).index(2)
    _update(ours, last)
    with pytest.raises(ValueError, match="greater than or equal to three"):
        ours.update()
    assert ours._n_processed == last


def test_empty_window_range_raises():
    with pytest.raises(ValueError, match="No window size"):
        mlx_stump.stimp(random_walk(4, seed=0), min_m=3, max_m=10)
    with pytest.raises(ValueError, match="No window size"):
        mlx_stump.stimp(random_walk(100, seed=0), min_m=90, max_m=95)


@pytest.mark.parametrize("denom", [1, 2.5, 8])
def test_honours_stumpy_excl_zone_denom(monkeypatch, denom):
    monkeypatch.setattr(stumpy.config, "STUMPY_EXCL_ZONE_DENOM", denom)
    T = random_walk(240, seed=6)
    for kw in ({}, dict(min_m=4, max_m=500, step=9)):
        ours = mlx_stump.stimp(T, **kw)
        ref = _ref_stimp(T, **kw)
        np.testing.assert_array_equal(ours.M_, ref.M_)
    # the per-window profiles use the same zone
    kw = dict(min_m=5, max_m=60, step=11)
    ours = mlx_stump.stimp(T, **kw)
    ref = _ref_stimp(T, **kw)
    _update(ours, len(ours.M_))
    _update(ref, len(ref.M_))
    _assert_matches(ours, ref)


# ------------------------------------------------------ 4. gpu_stimp
def _stumpy_class_init_signature(module, cls):
    """(name, default) pairs of a STUMPY class's ``__init__`` (minus
    ``self``), parsed from source: ``stumpy.gpu_stimp`` cannot be imported
    without a CUDA driver."""
    pkg = pathlib.Path(importlib.util.find_spec("stumpy").origin).parent
    tree = ast.parse((pkg / f"{module}.py").read_text())
    (node,) = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls]
    (init,) = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"]
    args = init.args
    assert not args.kwonlyargs and args.vararg is None and args.kwarg is None
    defaults = [inspect.Parameter.empty] * (len(args.args) - len(args.defaults))
    defaults += [ast.literal_eval(d) for d in args.defaults]
    pairs = [(a.arg, d) for a, d in zip(args.args, defaults, strict=True)]
    assert pairs[0] == ("self", inspect.Parameter.empty)
    return pairs[1:]


def _params(obj):
    params = list(inspect.signature(obj).parameters.values())
    assert all(p.kind is p.POSITIONAL_OR_KEYWORD for p in params)
    return [(p.name, p.default) for p in params]


def test_signatures_match_stumpy():
    assert _params(mlx_stump.gpu_stimp) == _stumpy_class_init_signature("gpu_stimp", "gpu_stimp")
    # stimp is stumpy.stimp without the SCRIMP controls
    ref = [
        (p.name, p.default)
        for p in inspect.signature(stumpy.stimp).parameters.values()
        if p.name not in ("percentage", "pre_scrump")
    ]
    assert _params(mlx_stump.stimp) == ref
    assert {"stimp", "gpu_stimp"} <= set(mlx_stump.__all__)
    assert issubclass(mlx_stump.gpu_stimp, mlx_stump.stimp)
    for cls in (mlx_stump.stimp, mlx_stump.gpu_stimp):  # annotations resolve
        hints = typing.get_type_hints(cls.__init__)
        assert "T" in hints and "T_subseq_isconstant_func" in hints


@pytest.mark.parametrize("device_id", [0, 1, np.int64(0), [0], [0, 1]])
@pytest.mark.gpu
def test_gpu_stimp_ignores_a_valid_device_id(device_id):
    T = random_walk(300, seed=12)
    ours = mlx_stump.gpu_stimp(T, 5, 40, 5, device_id)
    plain = mlx_stump.stimp(T, 5, 40, 5)
    _update(ours, len(ours.M_))
    _update(plain, len(plain.M_))
    np.testing.assert_array_equal(ours._PAN, plain._PAN)
    np.testing.assert_array_equal(ours.PAN_, plain.PAN_)


@pytest.mark.parametrize("device_id", [-1, None, [], "0", True, 1.0, [0, -2]])
def test_gpu_stimp_rejects_an_invalid_device_id(device_id):
    with pytest.raises(ValueError, match="device_id"):
        mlx_stump.gpu_stimp(random_walk(100, seed=0), 5, 20, 1, device_id)


@pytest.mark.gpu
def test_gpu_stimp_positional_normalize_is_the_sixth_argument():
    T = random_walk(300, seed=13)
    raw = mlx_stump.gpu_stimp(T, 5, 30, 5, 0, False)
    ref = _ref_stimp(T, normalize=False, min_m=5, max_m=30, step=5)
    _update(raw, 2)
    _update(ref, 2)
    _assert_matches(raw, ref, normalize=False)


# --------------------------------------------------- input validation
def test_input_validation():
    T = random_walk(100, seed=0)
    with pytest.raises(TypeError, match="float64"):
        mlx_stump.stimp(np.arange(100))
    with pytest.raises(TypeError, match="float64"):
        mlx_stump.stimp(T.astype(np.float32))
    with pytest.raises(ValueError, match="1-dimensional"):
        mlx_stump.stimp(T.reshape(10, 10))
    for name, value in [("min_m", 3.0), ("max_m", 20.5), ("step", True), ("min_m", "3")]:
        with pytest.raises(TypeError, match=f"`{name}` must be an integer"):
            mlx_stump.stimp(T, **{name: value})
    for step in (0, -2):
        with pytest.raises(ValueError, match="`step` must be a positive integer"):
            mlx_stump.stimp(T, step=step)
    with pytest.raises(NotImplementedError, match="p=2.0"):
        mlx_stump.stimp(T, normalize=False, p=1.0)
    mlx_stump.stimp(T, normalize=True, p=1.0)  # ignored, as in STUMPY
    with pytest.raises(ValueError, match="callable"):
        mlx_stump.stimp(T, T_subseq_isconstant_func=np.zeros(98, dtype=bool))
    with pytest.raises(ValueError, match="finite value"):
        mlx_stump.stimp(np.full(50, np.nan), normalize=False)
    # a function with another required argument is rejected when it is used
    pmp = mlx_stump.stimp(T, 5, 10, T_subseq_isconstant_func=lambda a, w, extra: None)
    with pytest.raises(ValueError, match="Incompatible arguments"):
        pmp.update()


def test_series_is_copied_and_byte_order_is_accepted():
    T = random_walk(300, seed=14)
    swapped = T.astype(">f8")
    ours = mlx_stump.stimp(swapped, 8, 40, 8)
    ref = _ref_stimp(T, min_m=8, max_m=40, step=8)
    swapped[:] = 0.0  # later edits of the caller's array do not leak in
    _update(ours, len(ours.M_))
    _update(ref, len(ref.M_))
    _assert_matches(ours, ref)


# ------------------------------------------------------- 5. warnings
@pytest.mark.parametrize(
    "cls", [mlx_stump.stimp, pytest.param(mlx_stump.gpu_stimp, marks=pytest.mark.gpu)]
)
def test_update_warnings_point_at_the_callers_line(cls):
    """Warnings from the per-window stump call and from its preprocessing
    helpers (one frame deeper) are attributed to the ``update()`` line."""

    def all_flat(a, w):
        return np.ones(len(a) - w + 1, dtype=bool)

    T_nan = random_walk(200, seed=3)
    T_nan[100] = np.nan
    cases = [
        # window-size advisory (check_window_size)
        ("may be too large", random_walk(30, seed=3), dict(min_m=20)),
        # user flags switched off at NaN windows (preprocess_series)
        ("automatically switched", T_nan, dict(min_m=5, max_m=20)),
        # small-profile diagnostic (_stump): every window is "constant"
        ("smaller than", random_walk(200, seed=3), dict(min_m=5, max_m=20)),
    ]
    for expected, T, kw in cases:
        if expected != "may be too large":
            kw["T_subseq_isconstant_func"] = all_flat
        pmp = cls(T, **kw)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            pmp.update()
        assert any(expected in str(w.message) for w in rec), expected
        assert {w.filename for w in rec} == {__file__}, [
            (w.filename, str(w.message)[:40]) for w in rec
        ]
