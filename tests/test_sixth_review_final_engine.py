"""Regression tests for the sixth review round's final audit: the GPU engine.

1. (F2) The threaded refinement stopped its lanes only when a worker
   raised. A caller interrupted while waiting on the pool (Ctrl-C) left
   every thread refining to the end in the background, and a script could
   not exit until they had. The stop event is now set inside the pool's
   ``with`` body, so any exception in the caller stops each lane after the
   chunk it holds, before the pool joins.
2. (F4) An extreme but accepted ``STUMPY_EXCL_ZONE_DENOM`` (``ceil(m /
   denom)`` near or above ``2**31``) overflowed the reducers' int32 zone
   arithmetic: the fused kernel raised ``std::bad_cast`` and the compiled
   fallback reported every row as its own neighbour at distance 0, where
   STUMPY returns an all-inf profile. The fused kernels also raised in
   AB-joins, which have no zone but still pass it. Both reducers clamp the
   zone to ``l``, which already excludes every candidate.
3. (MEM-1) Tiled blocks differ in width by one column, and MLX reuses a
   cached buffer only within 2 pages of the request. With an explicit
   ``chunk_size`` above ~4096 rows (and, for ``m >= ~8192``, for the window
   block itself, in ``mass`` too) the wider blocks' buffers stayed cached
   next to the narrower blocks' ones: about twice the modeled device
   memory. The cache is now cleared once, where the blocks narrow.
4. (MEM-2) Each window block was staged in NumPy before its device copy.
   On macOS the freed staging copy stayed resident, so RSS growth exceeded
   ``estimated_peak_bytes`` at large ``m`` (n=68000, m=1000: ~700 MiB
   against 642). Blocks are now centered straight into their own device
   buffer; the estimate's upload phase is block + centering temporary.
5. (MEM-3) A ``stump`` call that raised (Ctrl-C between batches, a failed
   allocation) left the window block and the batch buffers in MLX's cache,
   ~400 MiB, until the next call: the traceback's frames kept them alive
   past any ``clear_cache``. The handler now releases the device arrays,
   clears the finished mlx-stump frames, synchronizes and clears the cache
   (``_engine.free_gpu_after_error``). It also covers preprocessing: a
   normalized AB-join whose ``T_B`` constant flags fail after ``T_A``'s
   per-window arrays were uploaded no longer leaves those cached.
6. (Q3) The kernel-fallback RuntimeWarning had a fixed ``stacklevel`` and
   pointed into ``_engine``; it now points at the first frame outside the
   package, whichever entry point reached it.
7. (Q7) The sweeps converted and stored float32-derived profile values that
   the float64 refinement always overwrote. They now return only the
   neighbour indices, and ``stump`` refines into a fresh array: the output
   is bitwise identical and the sweep's host peak halves at large ``k``.

8. (CI, macos-14) The Metal compiler of macOS 14 ignores ``#pragma METAL
   fp contract(off)``, so the raw-mode kernel fused ``max(x, 0) +
   (m*dmu)*dmu`` into an FMA and resolved exact ties differently from the
   compiled fallback (the fused-vs-fallback bit-identity tests failed on
   every macos-14 CI job). The product is now OR-ed with a runtime zero,
   which no compiler can fold; the harness test builds the kernel's own
   snippet without the pragma, as that compiler does.

Q4 (the bit-identity helper in ``test_sixth_review_engine.py`` checks that
the fused reducer was built) and MEM-4 (re-measured per-window byte
figures) change tests and comments only.
"""

from __future__ import annotations

import gc
import signal
import threading
import time
import tracemalloc
import warnings

import mlx.core as mx
import numpy as np
import pytest

import mlx_stump
import mlx_stump._engine as eng
import mlx_stump._kernels as kern
import mlx_stump._stump as st
from mlx_stump._preprocess import exclusion_zone, preprocess_series, stable_center_scale

from .conftest import (
    assert_indices_tie_tolerant,
    assert_profile_close,
    run_isolated,
    tie_tolerance,
)

stumpy = pytest.importorskip("stumpy")

MIB = 1 << 20
FIELDS = ("P_", "I_", "left_I_", "right_I_")
needs_metal = pytest.mark.skipif(
    not (mx.metal.is_available() and mx.default_device() == mx.gpu),
    reason="requires an active Metal GPU",
)
real_sigint = pytest.mark.skipif(
    signal.getsignal(signal.SIGINT) is not signal.default_int_handler,
    reason="SIGINT does not raise KeyboardInterrupt here",
)


def _walk(n, seed):
    return np.random.default_rng(seed).standard_normal(n).cumsum()


def _force_fallback(monkeypatch):
    monkeypatch.setattr(eng, "_fused_reducer", lambda k: False)


def _force_tiles(monkeypatch, m, columns):
    monkeypatch.setattr(eng, "_MATMUL_WINDOW_BYTES", 0)
    monkeypatch.setattr(eng, "_TILE_WINDOW_BYTES", columns * m * 4)


def _assert_same(a, b, label=""):
    for f in FIELDS:
        np.testing.assert_array_equal(
            np.asarray(getattr(a, f), dtype=np.float64),
            np.asarray(getattr(b, f), dtype=np.float64),
            err_msg=f"{label} {f}",
        )


# ------------------------------------------- 1: interrupted refinement
@real_sigint
def test_interrupted_refinement_stops_every_thread(monkeypatch):
    """A real SIGINT at the main thread while it waits on the refinement
    pool (``_thread.interrupt_main`` would not wake that wait). Each lane
    stops after the chunk it holds, and the pool is joined before the
    KeyboardInterrupt reaches the caller; before the fix all 401 chunks
    ran, most of them after the caller had regained control."""
    m, l, lanes = 10, 801, 8  # 401 two-row chunks of 50 ms: ~2.5 s in all
    monkeypatch.setattr(eng, "_REFINE_MEM_BUDGET", 16 * m * 8 * 4)
    monkeypatch.setattr(st, "_REFINE_MAX_WORKERS", lanes)
    monkeypatch.setattr(st, "_cpu_count", lambda: lanes)
    lock = threading.Lock()
    calls = []
    fired = []
    main = threading.main_thread().ident

    def job(s, e):
        with lock:
            calls.append(s)
            if len(calls) == 3 * lanes:
                fired.append(len(calls))
                signal.pthread_kill(main, signal.SIGINT)
        time.sleep(0.05)

    with pytest.raises(KeyboardInterrupt):
        st._map_refine_chunks(job, l, m)
    with lock:
        at_return = len(calls)
    time.sleep(0.2)
    with lock:
        later = len(calls)
    assert fired == [3 * lanes]
    assert later == at_return, f"{later - at_return} chunks ran after the call raised"
    # each lane finishes the chunk it holds and starts at most one more
    assert at_return - fired[0] <= 2 * lanes


# ---------------------------------------- 2: extreme exclusion zones
def _denom_for_zone(m, zone):
    """A denominator whose zone ``ceil(m / denom)`` is exactly ``zone``."""
    d = m / zone
    for _ in range(64):
        z = exclusion_zone(m, d)
        if z == zone:
            return float(d)
        d = np.nextafter(d, np.inf) if z > zone else np.nextafter(d, 0.0)
    raise AssertionError(f"no float denominator gives zone {zone}")


@pytest.mark.filterwarnings("ignore:The window size:UserWarning")
@pytest.mark.parametrize("zone", ["denom=1e-9", "denom=1e-12", "zone=2**31-l"])
@pytest.mark.parametrize("normalize", [True, False])
def test_extreme_exclusion_zone_excludes_every_candidate(monkeypatch, zone, normalize):
    """Like STUMPY: an all-inf profile and -1 indices, on the fused kernel
    (GPU, k=1), the compiled fallback (k=20, forced, the CPU device), dense
    and tiled. A zone of ``2**31 - l`` fits in int32, but ``i + zone + 1``
    did not."""
    T = _walk(500, seed=61)
    m = 50
    l = T.size - m + 1
    denom = {"denom=1e-9": 1e-9, "denom=1e-12": 1e-12}.get(zone) or _denom_for_zone(
        m, 2**31 - l
    )
    monkeypatch.setattr(stumpy.config, "STUMPY_EXCL_ZONE_DENOM", denom)
    ref_fn = stumpy.stump if normalize else stumpy.aamp
    for k in (1, 20):
        ref = ref_fn(T, m, k=k)
        assert not np.isfinite(np.asarray(ref[:, :k], dtype=np.float64)).any()
        for device in ("default", "fallback", "cpu"):
            for tiled in (False, True):
                with monkeypatch.context() as mp:
                    if device == "fallback":
                        _force_fallback(mp)
                    if tiled:
                        _force_tiles(mp, m, 100)
                    if device == "cpu":
                        with mx.stream(mx.cpu):
                            got = mlx_stump.stump(T, m, k=k, normalize=normalize)
                    else:
                        got = mlx_stump.stump(T, m, k=k, normalize=normalize)
                assert got._excl_zone_denom == denom
                _assert_same(got, ref, f"k={k} {device} tiled={tiled}")


@pytest.mark.parametrize("denom", [1e-9, 1e-12])
@pytest.mark.parametrize("normalize", [True, False])
def test_extreme_exclusion_zone_leaves_ab_joins_unchanged(monkeypatch, denom, normalize):
    """An AB-join has no exclusion zone, but the fused kernels still pack it
    into their int32 parameters: before the fix the fused k=1 and top-k
    reductions raised ``std::bad_cast``. Now every path returns exactly
    what it returns at the default denominator, which matches STUMPY."""
    T_A = _walk(500, seed=63)
    T_B = _walk(300, seed=64)
    m = 50
    tie = tie_tolerance(m)
    default = stumpy.config.STUMPY_EXCL_ZONE_DENOM
    ref_fn = stumpy.stump if normalize else stumpy.aamp
    for k in (1, 5, 20):
        ref = ref_fn(T_A, m, T_B, ignore_trivial=False, k=k)
        for device in ("default", "fallback", "cpu"):
            for tiled in (False, True):
                runs = {}
                for d in (default, denom):
                    with monkeypatch.context() as mp:
                        mp.setattr(stumpy.config, "STUMPY_EXCL_ZONE_DENOM", d)
                        if device == "fallback":
                            _force_fallback(mp)
                        if tiled:
                            _force_tiles(mp, m, 100)
                        with mx.stream(mx.cpu if device == "cpu" else mx.default_device()):
                            runs[d] = mlx_stump.stump(
                                T_A, m, T_B, ignore_trivial=False, k=k, normalize=normalize
                            )
                got = runs[denom]
                label = f"k={k} {device} tiled={tiled}"
                assert got._excl_zone_denom == denom
                _assert_same(got, runs[default], label)
                P = np.asarray(got.P_, dtype=np.float64).reshape(-1, k)
                I = np.asarray(got.I_, dtype=np.int64).reshape(-1, k)
                Pr = np.asarray(ref.P_, dtype=np.float64).reshape(-1, k)
                Ir = np.asarray(ref.I_, dtype=np.int64).reshape(-1, k)
                for j in range(k):
                    assert_profile_close(P[:, j], Pr[:, j], m=m, tie_atol=tie)
                    assert_indices_tie_tolerant(
                        I[:, j], Ir[:, j], T_A, T_B, m, normalize=normalize, tie_atol=tie
                    )
                assert (np.asarray(got.left_I_) == -1).all()
                assert (np.asarray(got.right_I_) == -1).all()


# --------------------------------------- 3: blocks that narrow by one column
def _sample_tiled_batches(monkeypatch):
    """Record (block width, MLX active + cached bytes) at every tiled batch."""
    samples = []
    real = st.make_reducer

    def make(*args, **kwargs):
        red = real(*args, **kwargs)
        block = red.block

        def sampled(QT, s0, j0, j1):
            samples.append((j1 - j0, mx.get_active_memory() + mx.get_cache_memory()))
            return block(QT, s0, j0, j1)

        red.block = sampled
        return red

    monkeypatch.setattr(st, "make_reducer", make)
    return samples


@pytest.mark.gpu
@pytest.mark.parametrize("fused", [True, False])
def test_narrower_tiles_do_not_keep_a_second_batch_set(monkeypatch, fused):
    """Five blocks of 1639/1638 columns and one 8192-row batch per block:
    one column fewer shrinks the batch's QT by 32 KiB, a request MLX does
    not serve from the cached wider buffer. Before the fix the old set
    stayed cached: 1.86x one modeled set (fused), 1.47x (fallback)."""
    m, l, B = 16, 8192, 8192
    T = _walk(l + m - 1, seed=62)
    _force_tiles(monkeypatch, m, 2000)
    if not fused:
        _force_fallback(monkeypatch)
    samples = _sample_tiled_batches(monkeypatch)
    mx.synchronize()
    mx.clear_cache()
    base = mx.get_active_memory()
    mlx_stump.stump(T, m, chunk_size=B)
    assert [w for w, _ in samples] == [1639, 1639, 1638, 1638, 1638]
    one_set = B * eng._batch_row_bytes(1639, m, 1, True, fused)
    worst = max(b for _, b in samples) - base
    assert worst <= 1.25 * one_set, f"{worst / one_set:.2f} batch sets live or cached"


@pytest.mark.gpu
@pytest.mark.slow
def test_narrower_mass_block_does_not_keep_the_old_block_cached(monkeypatch):
    """m=8192: a block row is 32 KiB, so the 4095-column block cannot reuse
    the cached 4096-column one (128 MiB). Before the fix mass held 2.0 blocks
    at the last block."""
    m, n = 8192, 20_478
    T = _walk(n, seed=63)
    l = n - m + 1
    block = eng.resident_block_bytes(l, m)
    assert block < l * m * 4  # tiled: blocks of 4096, 4096, 4095 columns
    seen = []
    real = eng.MassEngine.znorm_sq_distances

    def sampled(self, QT, *args):
        seen.append(mx.get_active_memory() + mx.get_cache_memory())
        return real(self, QT, *args)

    monkeypatch.setattr(eng.MassEngine, "znorm_sq_distances", sampled)
    mx.synchronize()
    mx.clear_cache()
    base = mx.get_active_memory()
    mlx_stump.mass(T[:m].copy(), T)
    assert len(seen) == 3
    worst = max(seen) - base
    assert worst <= 1.25 * block, f"{worst / block:.2f} window blocks live or cached"


# ---------------------------------------------- 4: blocks built in place
@pytest.mark.parametrize("device", ["gpu", "cpu"])
def test_writable_host_view_aliases_a_private_device_buffer(device):
    """The contract the in-place build relies on (MLX 0.30 and 0.32): an
    evaluated ``mx.zeros`` exports its own writable, C-contiguous buffer,
    and later device work reads what the host wrote there."""
    if device == "gpu" and not mx.metal.is_available():
        pytest.skip("requires a Metal GPU")
    rng = np.random.default_rng(64)
    with mx.stream(mx.gpu if device == "gpu" else mx.cpu):
        a = mx.zeros((37, 11), dtype=mx.float32)
        b = mx.zeros((37, 11), dtype=mx.float32)
        mx.eval(a, b)
        h = eng._writable_host_view(a)
        assert h is not None and h.flags.writeable and h.flags.c_contiguous
        assert h.dtype == np.float32 and h.shape == (37, 11)
        vals = rng.standard_normal((37, 11)).astype(np.float32)
        h[...] = vals
        del h
        q = mx.array(rng.standard_normal((5, 11)).astype(np.float32))
        np.testing.assert_array_equal(np.array(q @ a.T), np.array(q @ mx.array(vals).T))
        assert float(mx.max(mx.abs(b)).item()) == 0.0  # no shared storage


@pytest.mark.parametrize("tiled", [False, True])
@pytest.mark.parametrize("normalize", [True, False])
def test_blocks_are_built_without_a_host_staging_copy(monkeypatch, tiled, normalize):
    """tracemalloc sees NumPy's allocations: building a block allocated a
    whole block-sized staging array; now only the centering chunks."""
    m = 500
    T = _walk(20_000, seed=65)
    T[3000:3010] = np.nan
    if normalize:
        prep = preprocess_series(T, m)
    else:
        center, scale = stable_center_scale(T[np.isfinite(T)])
        prep = preprocess_series(T, m, normalize=False, center=center, scale=scale)
    monkeypatch.setattr(eng, "_CENTER_BYTES", 1 << 21)  # 2 MiB centering steps
    if tiled:
        _force_tiles(monkeypatch, m, 6000)  # 4 blocks of 4876/4875 columns
    if tracemalloc.is_tracing():
        pytest.skip("tracemalloc is already in use")
    tracemalloc.start()
    try:
        engine = eng.MassEngine(prep, normalize=normalize)
        widths = []
        for j0, j1, W in engine.target_blocks():
            widths.append(j1 - j0)
            del W
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    block = max(widths) * m * 4
    assert (len(widths) == 4) == tiled
    assert peak < block / 2, f"host peak {peak / MIB:.1f} MiB for a {block / MIB:.1f} MiB block"


@pytest.mark.parametrize("tiled", [False, True])
def test_in_place_blocks_match_the_staged_build(monkeypatch, tiled):
    """The in-place build writes the same float32 values in the same layout
    as the NumPy-staged build it replaced (kept as the fallback should MLX
    stop exporting a writable buffer): stump and mass outputs are
    bit-identical, over several centering steps per block."""
    m = 64
    T = _walk(3000, seed=66)
    T[1000:1030] = np.nan
    T[2000:2100] = 1.5
    Q = T[400 : 400 + m].copy()
    if tiled:
        _force_tiles(monkeypatch, m, 700)
    monkeypatch.setattr(eng, "_CENTER_BYTES", 100 * (m * 8 + eng._CENTER_ROW_BYTES))
    in_place = [0]
    real = eng._writable_host_view

    def counted(a):
        view = real(a)
        in_place[0] += view is not None
        return view

    def run():
        out = []
        for normalize in (True, False):
            for k in (1, 3):
                mp = mlx_stump.stump(T, m, k=k, normalize=normalize)
                out.append(np.asarray(mp, dtype=np.float64))
            out.append(mlx_stump.mass(Q, T, normalize=normalize))
        return out

    monkeypatch.setattr(eng, "_writable_host_view", counted)
    built = run()
    assert in_place[0] >= 6  # every block of every call was built in place
    monkeypatch.setattr(eng, "_writable_host_view", lambda a: None)
    staged = run()
    for a, b in zip(built, staged, strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.slow
def test_large_window_dense_rss_within_estimate():
    """n=68000, m=1000: the dense block is ~256 MiB. With the staging copy
    RSS grew 696-722 MiB against the 642 MiB estimate; built in place,
    434-475 MiB (fresh interpreters, after a warm-up)."""
    n, m, k = 68_000, 1_000, 1
    l = n - m + 1
    assert eng.resident_block_bytes(l, m) == l * m * 4  # dense
    before, peak, _ = run_isolated(
        f"""
        T = np.random.default_rng(0).standard_normal({n}).cumsum()
        mlx_stump.stump(T[:4096], {m}, k={k})  # warm-up: Metal/JIT baseline
        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * _unit
        mlx_stump.stump(T, {m}, k={k})
        """
    )
    growth = peak - before
    est = eng.estimated_peak_bytes(l, m, k=k) / MIB
    assert growth <= est, f"RSS grew {growth:.0f} MiB vs estimate {est:.0f} MiB"


# ------------------------------------------ 5: cleanup after an exception
def _raise_on_call(orig, n, exc_type, message):
    """Raise a fresh ``exc_type`` at the n-th call (a stored instance would
    keep its traceback, and the frames in it, alive until the test ends)."""
    calls = [0]

    def wrapped(*args, **kwargs):
        calls[0] += 1
        if calls[0] == n:
            del args, kwargs  # this (test) frame must hold no device arrays
            raise exc_type(message)
        return orig(*args, **kwargs)

    return wrapped


def _sigint_on_call(orig, n):
    """Deliver a real SIGINT to the main thread at the n-th call."""
    calls = [0]
    main = threading.main_thread().ident

    def wrapped(*args, **kwargs):
        calls[0] += 1
        if calls[0] == n:
            del args, kwargs
            signal.pthread_kill(main, signal.SIGINT)
            time.sleep(10)  # the handler raises KeyboardInterrupt first
            raise AssertionError("SIGINT was not delivered")
        return orig(*args, **kwargs)

    return wrapped


def _check_released(call, exc_type, base):
    """Run ``call``, which raises ``exc_type``; afterwards MLX holds nothing
    beyond ``base`` and caches nothing, also once the traceback is gone
    (anything it still referenced would be freed into the cache)."""
    with pytest.raises(exc_type) as excinfo:
        call()
    cached, active = mx.get_cache_memory(), mx.get_active_memory()
    del excinfo
    gc.collect()
    late = mx.get_cache_memory()
    assert cached == 0 and active <= base + MIB and late == 0, (
        f"at the raise {cached / MIB:.1f} MiB cached and {(active - base) / MIB:.1f} MiB "
        f"active; {late / MIB:.1f} MiB cached once the traceback was freed"
    )


def _clean_slate():
    gc.collect()
    mx.synchronize()
    mx.clear_cache()
    return mx.get_active_memory()


@pytest.mark.parametrize("fused", [True, False])
@pytest.mark.parametrize("tiled", [False, True])
def test_failing_reducer_leaves_nothing_cached(monkeypatch, tiled, fused):
    """A batch that fails (as a Metal allocation failure would) at the third
    reducer call. Before the fix the window block, the reducer's inputs and
    the batch buffers stayed in MLX's cache."""
    m = 50
    T = _walk(20_000, seed=67)
    if tiled:
        _force_tiles(monkeypatch, m, 5_000)
    if not fused:
        _force_fallback(monkeypatch)
    eng._fused_reducer(1)  # probe the kernels first: the probe calls FusedReduce.full
    for cls in (kern.FusedReduce, eng.ReduceStep):
        for name in ("full", "block"):
            fail = _raise_on_call(getattr(cls, name), 3, RuntimeError, "allocation failed")
            monkeypatch.setattr(cls, name, fail)
    base = _clean_slate()
    _check_released(lambda: mlx_stump.stump(T, m, chunk_size=512), RuntimeError, base)


def test_failing_sweep_releases_the_window_block(monkeypatch):
    """The sweep raises before its first batch: the 46 MiB dense block the
    engine uploaded must not stay cached."""
    real = st._compute_profile
    monkeypatch.setattr(st, "_compute_profile", _raise_on_call(real, 1, MemoryError, "sweep"))
    T = _walk(60_000, seed=68)
    base = _clean_slate()
    _check_released(lambda: mlx_stump.stump(T, 200), MemoryError, base)


@real_sigint
@pytest.mark.parametrize("tiled", [False, True])
def test_ctrl_c_between_batches_leaves_nothing_cached(monkeypatch, tiled):
    """A real SIGINT while fetching the next query batch, with the current
    batch still in flight on the GPU."""
    m = 50
    T = _walk(20_000, seed=69)
    if tiled:
        _force_tiles(monkeypatch, m, 5_000)
    if tiled:
        monkeypatch.setattr(st, "query_windows", _sigint_on_call(st.query_windows, 6))
    else:
        # The dense implicit self-join now takes views of the already-packed
        # target rows. Interrupt its lazy batch generator instead of the old
        # CPU query packer; the 11th pull is the sixth query batch, fetched
        # while the fifth batch runs.
        real_batches = st._batches
        pulls = [0]
        main = threading.main_thread().ident

        def interrupted_batches(*args, **kwargs):
            for bounds in real_batches(*args, **kwargs):
                pulls[0] += 1
                if pulls[0] == 11:
                    signal.pthread_kill(main, signal.SIGINT)
                    time.sleep(10)
                    raise AssertionError("SIGINT was not delivered")
                yield bounds

        monkeypatch.setattr(st, "_batches", interrupted_batches)
    base = _clean_slate()
    _check_released(lambda: mlx_stump.stump(T, m, chunk_size=512), KeyboardInterrupt, base)


@pytest.mark.parametrize("bad_flag", ["array", "callable"])
def test_failing_t_b_preprocessing_releases_t_a(bad_flag):
    """A normalized AB-join whose T_B constant flags fail after T_A's
    per-window arrays were uploaded (~6 MiB here): the handler must cover
    preprocessing, not only the sweep."""
    n, m = 1_000_000, 50
    T_A, T_B = _walk(n, seed=71), _walk(n, seed=72)

    def boom(a, w):
        raise ValueError("flag callable failed")

    flag = np.zeros(3, dtype=bool) if bad_flag == "array" else boom
    base = _clean_slate()
    _check_released(
        lambda: mlx_stump.stump(T_A, m, T_B, ignore_trivial=False, T_B_subseq_isconstant=flag),
        ValueError,
        base,
    )


def test_successful_call_after_a_failure_is_unchanged(monkeypatch):
    """The handler re-raises and leaves no state behind: the next call's
    output is the same as a fresh one."""
    T = _walk(3_000, seed=70)
    ref = mlx_stump.stump(T, 40, k=2)
    with monkeypatch.context() as mp:
        mp.setattr(
            st, "_compute_profile", _raise_on_call(st._compute_profile, 1, ValueError, "x")
        )
        with pytest.raises(ValueError):
            mlx_stump.stump(T, 40, k=2)
    _assert_same(mlx_stump.stump(T, 40, k=2), ref)


# ---------------------------------- 6: the fallback warning's location
@needs_metal
def test_kernel_fallback_warning_points_at_the_caller(monkeypatch):
    """Each public entry point reaches the probe at a different depth (a
    fresh k for each, so every call probes and warns)."""
    monkeypatch.setattr(kern, "_LAUNCH", {})

    def broken(k, tg):
        raise RuntimeError("simulated pipeline failure")

    monkeypatch.setattr(kern, "_probe", broken)
    T = _walk(400, seed=71)

    def tiled():
        with monkeypatch.context() as mp:
            _force_tiles(mp, 10, 100)
            mlx_stump.stump(T, 10, k=8)

    calls = {
        "stump": lambda: mlx_stump.stump(T, 10, k=6),
        "estimated_peak_bytes": lambda: mlx_stump.estimated_peak_bytes(391, 10, k=7),
        "gpu_aamp": lambda: mlx_stump.gpu_aamp(T, 10, k=9),
        "tiled stump": tiled,
        "stimp": lambda: mlx_stump.stimp(T, min_m=8, max_m=12).update(),  # k=1
    }
    for name, call in calls.items():
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            call()
        hits = [w for w in rec if "compiled reduction" in str(w.message)]
        assert hits, name
        assert {w.filename for w in hits} == {__file__}, name


# ------------------------------------------ 7: the sweep keeps indices only
@pytest.mark.parametrize("tiled", [False, True])
@pytest.mark.parametrize("k", [1, 4])
def test_sweep_returns_only_neighbour_indices(monkeypatch, tiled, k):
    m = 12
    captured = []
    real = st._compute_profile

    def spy(*args, **kwargs):
        out = real(*args, **kwargs)
        captured.append(out)
        return out

    monkeypatch.setattr(st, "_compute_profile", spy)
    if tiled:
        _force_tiles(monkeypatch, m, 50)
    T = _walk(300, seed=72)
    T[100] = np.nan
    for T_B in (None, _walk(250, seed=73)):
        mp = mlx_stump.stump(T, m, T_B, k=k, ignore_trivial=T_B is None)
        I, IL, IR = captured.pop()
        assert I.dtype == IL.dtype == IR.dtype == np.int64
        assert I.shape == (T.size - m + 1, k)
        # the refinement may reorder near-tied top-k columns, not change them
        got = np.asarray(mp.I_, dtype=np.int64).reshape(I.shape)
        np.testing.assert_array_equal(np.sort(got, axis=1), np.sort(I, axis=1))
        np.testing.assert_array_equal(np.asarray(mp.left_I_, dtype=np.int64), IL)
        np.testing.assert_array_equal(np.asarray(mp.right_I_, dtype=np.int64), IR)


@pytest.mark.parametrize("tiled,ratio", [(False, 1.5), (True, 3.0)])
def test_sweep_host_peak_holds_no_profile_values(monkeypatch, tiled, ratio):
    """k=64, l=19951: the sweep's host peak was 20.5 MiB dense and 56.0 MiB
    tiled (float64 P, its sqrt temporaries and the tiled left/right
    combination for k > 1) against 9.7 MiB of indices; now 10.5 and 21.3."""
    m, k = 50, 64
    T = _walk(20_000, seed=74)
    l = T.size - m + 1
    if tiled:
        _force_tiles(monkeypatch, m, 5_000)
    if tracemalloc.is_tracing():
        pytest.skip("tracemalloc is already in use")
    peaks = []
    real = st._compute_profile

    def traced(*args, **kwargs):
        tracemalloc.start()
        try:
            return real(*args, **kwargs)
        finally:
            peaks.append(tracemalloc.get_traced_memory()[1])
            tracemalloc.stop()

    monkeypatch.setattr(st, "_compute_profile", traced)
    mlx_stump.stump(T, m, k=k)
    assert peaks[0] < ratio * l * k * 8, f"sweep host peak {peaks[0] / MIB:.1f} MiB"


def test_unused_sweep_names_are_gone():
    assert not hasattr(st, "_INF")


# --------------------- 8: raw distances without the contract pragma
def _raw_distance_kernel(name, d2_source):
    """The fused kernels' raw-distance snippets (``_ABS_PRE``/``_ABS_D2``)
    in a one-cell-per-thread harness, compiled WITHOUT ``#pragma METAL fp
    contract(off)``: what the Metal compiler of macOS 14 builds, since it
    ignores the pragma."""
    source = (
        """
        const uint e = thread_position_in_grid.x;
        const int i = 0;
        const size_t qbase = 0;
        const uint j = e;
        const int jg = (int)e;
        const float c_m = consts[0];
        const bool q_fin = qf[i];
        """
        + kern._ABS_PRE
        + d2_source
        + "\n        out[e] = d2;\n"
    )
    return mx.fast.metal_kernel(
        name=name,
        input_names=["QT", "qa", "qb", "qf", "ta", "tb", "tf", "par", "consts"],
        output_names=["out"],
        source=source,
        header=kern._HEADER.replace("#pragma METAL fp contract(off)", ""),
    )


@needs_metal
def test_raw_distance_matches_mlx_where_the_contract_pragma_is_ignored():
    """GitHub's macos-14 runners ignore ``#pragma METAL fp contract(off)``:
    there the kernel fused ``max(x, 0) + (m*dmu)*dmu`` into an FMA (120,313
    of 1,048,576 cells differed from MLX's strictly rounded ops), so raw
    self-joins resolved exact ties differently from the compiled fallback.
    The product now goes through an OR with a runtime zero, which rounds it
    on its own whatever the compiler does with the pragma."""
    n, m = 1 << 16, 7.0
    rng = np.random.default_rng(75)
    QT = rng.standard_normal(n).astype(np.float32) * 3
    ssq_t = rng.uniform(0.0, 12.0, n).astype(np.float32)
    mu_t = np.stack(
        [rng.standard_normal(n), rng.standard_normal(n) * 1e-8], axis=1
    ).astype(np.float32)
    ssq_q = np.array([5.0], dtype=np.float32)
    mu_q = np.array([[0.3, 1e-9]], dtype=np.float32)
    ones_q, ones_t = mx.array([True]), mx.array(np.ones(n, dtype=bool))
    consts = mx.array(np.array([m, 1 / m, 2 * m, 4 * m], dtype=np.float32))
    par = mx.array([0, 0, 0, 0], dtype=mx.int32)
    inputs = [
        mx.array(QT), mx.array(ssq_q), mx.array(mu_q.reshape(-1)), ones_q,
        mx.array(ssq_t), mx.array(mu_t.reshape(-1)), ones_t, par, consts,
    ]
    ref = eng._abs_sq(
        mx.array(QT)[None, :], mx.array(ssq_q), mx.array(mu_q), ones_q,
        mx.array(ssq_t), mx.array(mu_t), ones_t, m,
    )
    ref = np.array(ref)[0]

    def run(name, d2_source):
        (out,) = _raw_distance_kernel(name, d2_source)(
            inputs=inputs, grid=(n, 1, 1), threadgroup=(256, 1, 1),
            output_shapes=[(n,)], output_dtypes=[mx.float32],
        )
        return np.array(out)

    # the pre-fix one-expression form, for scale: without the pragma the
    # compiler fused it on every macOS measured (7,356 cells here)
    start = kern._ABS_D2.index("        // round the product")
    stop = kern._ABS_D2.index("+ p;") + len("+ p;")
    original = (
        kern._ABS_D2[:start]
        + "        float d2 = mlx_max(x, 0.0f) + (c_m * dmu) * dmu;"
        + kern._ABS_D2[stop:]
    )
    fused_cells = int(np.sum(run("mlx_stump_test_raw_d2_original", original) != ref))
    got = run("mlx_stump_test_raw_d2", kern._ABS_D2)
    assert np.array_equal(got, ref), (
        f"{int(np.sum(got != ref))} cells differ from MLX "
        f"({fused_cells} with the pre-fix expression)"
    )
