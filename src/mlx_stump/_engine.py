"""Batched-MASS GPU engine: matrix profiles via doubly-centered matmul.

For a query batch Q (B, m) and a target series T (n,), the sliding dot
products QT[b, j] = sum_k Q[b, k] * T[j + k] are one dense matmul against the
materialized (l, m) target window matrix. Each distance profile comes from a
fresh product rather than a long floating-point recurrence, so error does not
accumulate along the series.

EVERY window on both sides is mean-centered in float64 before the float32
cast. Normalized windows are also divided by their own locally computed RMS,
so their matmul is ``m * rho`` directly; raw windows retain the shared affine
frame and their product is the mean-centered cross-covariance:

- z-normalized: near-constant windows sitting at a large offset (a flatlined
  sensor) don't have their tiny covariance swamped by float32 rounding of
  the offset, and the catastrophic `QT - m*mu_q*mu_t` subtraction never
  happens in float32;
- non-normalized (aamp): the distance is ||qc - tc||^2 + m*(mu_q - mu_t)^2
  (exact algebra: the cross terms vanish because centered windows sum to 0),
  which kills the `ssq_q + ssq_t - 2*QT` cancellation that otherwise scales
  the noise with (segment offset / scale)^2 on mixed-scale data; the means
  travel as float32 (hi, lo) pairs so their difference is not limited to
  the ulp of the shared frame's global offset.

When the full window matrix would exceed the memory cap, the target is
processed in evenly-sized column blocks (each block built, centered, and
uploaded on demand), which preserves the identical per-window centering at
any length. (An FFT cross-correlation fallback was used here previously; it
operated on the raw float32 series, could not center per-window, and was
numerically wrong for near-constant data.)

Sweep reduction. Each batch's QT is reduced to per-row minima (``k == 1``;
self-joins also keep the left and right minima) or top-k lists by one of two
bit-identical implementations, chosen by the single predicate
``_fused_reducer(k)``:

- ``_kernels.FusedReduce`` (a Metal GPU that is MLX's default device,
  ``k <= 16``, and kernels that launch there, probed once per process): one
  threadgroup per query row reads each QT element once and evaluates the
  distance in registers, so QT is the only ``(B, width)`` buffer of the
  batch;
- ``ReduceStep``, one ``mx.compile`` graph (the CPU device, ``k > 16``, and
  the reference in the tests), which materializes the distances, masked
  variants and top-k sort buffers.

Both evaluate ``_znorm_sq`` / ``_abs_sq`` with the same float32 operations
and constants (``reduce_consts``: ``1/m`` is rounded once on the host) and
select lexicographically by ``(d2, key)``. The key reproduces STUMPY's
exact-tie rule: its self-join diagonal traversal visits candidates in
ascending offset, the left one first, and keeps the first of equal values,
so ``key = 2*|j - i| + (j > i)``; AB-joins keep the lowest column,
``key = j``. Candidates that tie only in float32 (their float64 distances
differ below float32 resolution) go through the same rule, so which of
those wins is arbitrary with respect to float64.

Memory accounting. Three byte budgets bound each ordinary computation;
documented one-row/one-block floors can exceed a nominal budget at extreme
``m``:

- the resident window block: the whole ``(m, l)`` float32 matrix when it fits
  ``_MATMUL_WINDOW_BYTES``, else one ``_TILE_WINDOW_BYTES`` column block at a
  time (each block is released before the next one is built);
- the live per-batch GPU intermediates, ``_CHUNK_MEM_BUDGET``
  (``default_chunk_size``/``tiled_chunk_size`` size the query batch to it at
  the measured bytes per QT cell of the reduction that runs, ``_FUSED_CELL``
  or ``_fallback_cell``; only one batch is in flight, the next batch's query
  windows being built on the CPU while it runs);
- CPU-side float64 temporaries: the block-centering step (``_CENTER_BYTES``),
  the sigma repair in preprocessing and the float64 refinement chunks (see
  ``_preprocess`` / ``_stump``), each a fixed budget independent of ``n``.

Building a block stages it in numpy before the device copy, so the block
exists twice for the duration of the upload. ``estimated_peak_bytes`` puts
the pieces together; the O(n) per-series arrays are on top of it.

Distance special cases follow STUMPY's semantics:
- z-normalized: d = sqrt(2m(1 - rho)), rho from the mean-centered covariance;
- both windows constant -> 0; exactly one constant -> sqrt(m);
- either window non-finite -> inf.
"""

from __future__ import annotations

from collections.abc import Iterator

import mlx.core as mx
import numpy as np

from ._preprocess import PreprocessedSeries, center_rows_stable

_INF = float("inf")
_U64_MAX = np.uint64(2**64 - 1)
# columns per first-stage chunk of the compiled fallback's top-k selection
_TOPK_CHUNK = 1024

# budget for the live per-batch GPU intermediates (the query window batch,
# QT, the reduction outputs and, on the compiled fallback, the squared
# distances, masked left/right variants and top-k sort/gather buffers).
# Actual peak memory also includes the per-call constants (the window
# matrix or one tile of it) and the O(l) per-series device arrays (window
# stats and masks, 14-18 B per window and series, and the fallback's 4 B
# column index) on top of this. The k == 1 sweeps fill the budget: their
# cells carry no headroom (the fused kernels on MLX 0.30 and 0.32, the
# fallback on 0.30), so measured peaks exceed block + budget by part of
# those arrays: at n=131072, m=50 by +1.5 MiB (fused, raw self-join; the
# arrays are 2.3 MiB) and +3.5 MiB (fallback on 0.30, raw AB-join; 5.0 MiB).
_CHUNK_MEM_BUDGET = 3 << 27  # ~384 MiB
# Device bytes per QT cell (batch row x target column) live during one
# batch, measured as the slope of MLX's peak over two explicit batch sizes
# (n=32768, m=50; normalized and raw alike). The fused kernels materialize
# QT and nothing else of that shape: 4.0 on MLX 0.30 and 0.32. The compiled
# fallback (_fallback_cell) also holds d2 and the masked left/right
# variants and, for k > 1, the uint64 (d2, key) selection keys and sort
# buffers.
_FUSED_CELL = 4
# host bytes per neighbour per row of the tiled top-k merge (_stump's
# _merge_topk: the block result, both 2k-wide concatenations, the self-join
# tie keys, the full lexsort permutation, the gathered outputs, and NumPy's
# sort workspace). tracemalloc peaks: 84.6 at k=2, 71 at k=16, 70 at k=100.
_TILED_MERGE_CELL = 96
# cap for materializing the (l, m) target window matrix on the GPU in one
# piece; above it the engine switches to tiled column blocks. The tiled
# sweep runs the same reduction per block and measured as fast as or faster
# than the dense one at about half the peak memory (n=524288, m=200, fused
# kernels: 26.2 s tiled/128 MiB vs 28.3 s dense, medians of 3 interleaved
# runs, peak 401 vs 785 MiB, identical output; the compiled step earlier
# gave 47.7 s vs 47.9 s), so the cap trades nothing for memory.
_MATMUL_WINDOW_BYTES = 1 << 28  # ~256 MiB
# size of one materialized target window block in tiled mode
_TILE_WINDOW_BYTES = 1 << 27  # ~128 MiB
# Byte bound on the complete CPU centering stage while building window blocks.
# Besides the rows*m float64 work matrix, center_rows_stable holds midpoint,
# scale, reduction, and mask vectors. 128 bytes per row conservatively covers
# their observed allocator peak (72-80 B/row) and ufunc transients.
_CENTER_BYTES = 1 << 26  # ~64 MiB
_CENTER_ROW_BYTES = 128
# byte budget for the float64 window copies held live by one refinement
# chunk (two fancy-indexed window blocks plus their centered copies); the
# refinement runs after the device memory has been released
_REFINE_MEM_BUDGET = 1 << 28  # ~256 MiB


def refine_chunk_rows(m: int) -> int:
    return max(1, min(1 << 16, _REFINE_MEM_BUDGET // (m * 8 * 4)))


def _center_rows(m: int) -> int:
    """Window rows centered per float64 step so the temporary fits ``_CENTER_BYTES``."""
    return max(1, _CENTER_BYTES // (m * 8 + _CENTER_ROW_BYTES))


def resident_block_bytes(l: int, m: int) -> int:
    """Bytes of the float32 window block the engine keeps on the device."""
    for name, value in (("l", l), ("m", m)):
        if not (
            isinstance(value, (int, np.integer))
            and not isinstance(value, (bool, np.bool_))
            and value >= 1
        ):
            raise ValueError(f"`{name}` must be a positive integer.")
    l, m = int(l), int(m)
    full = l * m * 4
    if full <= _MATMUL_WINDOW_BYTES:
        return full
    tile_rows = max(4, _TILE_WINDOW_BYTES // (4 * m))
    nblocks = -(-l // tile_rows)
    return -(-l // nblocks) * m * 4


def estimated_peak_bytes(
    l: int,
    m: int,
    k: int = 1,
    self_join: bool = True,
    l_q: int | None = None,
    chunk_size: int | None = None,
    fused: bool | None = None,
) -> int:
    """Estimate of the bytes one join needs beyond its O(n) per-series arrays.

    ``l`` is the number of target windows (the side the engine
    materializes), ``l_q`` the number of query windows (the output rows;
    defaults to ``l``), and ``chunk_size`` has the same meaning as in
    :func:`mlx_stump.stump`. ``fused`` selects the sweep reduction being
    modeled; by default it is the one this process would run
    (``_fused_reducer(k)``: the fused Metal kernels on the GPU for
    ``k <= 16``, else the compiled fallback, whose batches are smaller).
    With no explicit chunk size, the automatic byte-budgeted batch is
    modeled; with one, the requested batch (clamped to ``l_q``) is modeled,
    including requests that deliberately exceed the automatic ~384 MiB
    device budget. The largest of three phases:

    - upload: the block staged in numpy plus its device copy plus the
      centering temporary;
    - sweep: the resident block plus the per-batch intermediates budget (or
      one batch row, when even a single row exceeds the budget), plus the
      numeric profile/index outputs (float64 + int64 per neighbor,
      left/right indices; the tiled sweep also keeps float32/int64 top-k
      accumulators). The budget is enforced batch by batch — each batch is
      synchronized before the next allocates and the trailing batch is
      computed at full width — so exactly one set of intermediates exists
      (the next batch's query windows, built while it runs, are within the
      per-row query allowance). Tiled top-k joins also merge each device
      result into the running set on the host; the two concatenations, tie
      keys, full lexsort permutation, gather results, and sorting workspace
      are included here;
    - assembly: after the device memory is released, the float64
      refinement chunk plus the numeric outputs, the top-k reordering
      temporaries, and the object-dtype ``mparray`` STUMPY's output layout
      requires: an 8-byte pointer plus one CPython small-object allocation
      per cell. Although ``sys.getsizeof`` reports 24/28 bytes for a
      float/int, both occupy a 32-byte pymalloc size class, so the resident
      footprint is ~80 bytes per neighbor per row. This dominates for large
      ``k`` (n=50,000, m=50, k=100: ~385 MiB for the output alone).

    It is an estimate with headroom, not a hard cap: MLX's allocator rounds
    buffers up (about +0.5% observed), the O(n) series and stat arrays are
    not included, and the figures are MLX's own active-memory peak plus
    host memory (GPU-written buffers are invisible to RSS on macOS).
    """
    for name, value in (("l", l), ("m", m), ("k", k)):
        if not (
            isinstance(value, (int, np.integer))
            and not isinstance(value, (bool, np.bool_))
            and value >= 1
        ):
            raise ValueError(f"`{name}` must be a positive integer.")
    if l_q is None:
        l_q = l
    elif not (
        isinstance(l_q, (int, np.integer))
        and not isinstance(l_q, (bool, np.bool_))
        and l_q >= 1
    ):
        raise ValueError("`l_q` must be a positive integer.")
    if not isinstance(self_join, (bool, np.bool_)):
        raise ValueError("`self_join` must be a boolean.")
    if chunk_size is not None and not (
        isinstance(chunk_size, (int, np.integer))
        and not isinstance(chunk_size, (bool, np.bool_))
        and chunk_size >= 1
    ):
        raise ValueError("`chunk_size` must be a positive integer.")
    if fused is not None and not isinstance(fused, (bool, np.bool_)):
        raise ValueError("`fused` must be a boolean or None.")
    l, m, k, l_q = int(l), int(m), int(k), int(l_q)
    self_join = bool(self_join)
    fused = _fused_reducer(k) if fused is None else bool(fused)
    block = resident_block_bytes(l, m)
    full = l * m * 4
    tiled = full > _MATMUL_WINDOW_BYTES
    width = block // (m * 4)  # columns of the resident block
    one_row = _batch_row_bytes(width, m, k, self_join, fused)
    if chunk_size is None:
        # Match the actual sizing helpers. Tiled batches are sized against
        # MassEngine.tile_rows (the nominal upper bound), while blocks are
        # subsequently balanced and can be narrower than that bound.
        if tiled:
            batch = _tiled_batch(max(4, _TILE_WINDOW_BYTES // (4 * m)), m, k, self_join, fused)
        else:
            batch = _dense_batch(l, m, k, self_join, fused)
        batch = min(batch, max(1, l_q))
        # Keep the whole advertised device budget as conservative headroom
        # when automatic sizing is used. A one-row floor can exceed it.
        device_batch = max(_CHUNK_MEM_BUDGET, one_row)
    else:
        batch = min(int(chunk_size), max(1, l_q))
        device_batch = batch * one_row
    numeric = l_q * (16 * k + 16)  # P (float64) and I (int64) per neighbor, IL/IR
    accum = l_q * 12 * k if (tiled and k > 1) else 0  # tiled top-k merge state
    if tiled and k > 1:
        # the _merge_topk workspace (see _TILED_MERGE_CELL); the automatic
        # tiled batch already charges it per row against the budget, and it
        # is added here once more as headroom
        host_batch = batch * k * _TILED_MERGE_CELL
    elif k > 1:
        # Dense output conversion holds one float64 value copy and one int64
        # index copy for the current batch alongside the persistent outputs.
        host_batch = batch * k * 16
    else:
        # Value/index and (for self-joins) left/right conversion vectors.
        host_batch = batch * 32
    sweep = block + device_batch + numeric + accum + host_batch
    refine = refine_chunk_rows(m) * m * 8 * 4
    reorder = l_q * 16 * k if k > 1 else 0  # argsort order + one reordered copy live
    # Object-array pointers plus CPython's allocation-size footprint for the
    # boxed float/int in every cell. Both 24-byte floats and 28-byte ints use
    # the 32-byte pymalloc class; modeling logical `getsizeof` values
    # undercounted the canonical k=100 process peak by ~25 MiB.
    boxed = l_q * (2 * k + 2) * (8 + 32)
    assembly = refine + numeric + reorder + boxed
    # _center_rows has a one-row floor when a single float64 window plus its
    # rowwise scratch exceeds the nominal centering budget. Model it too.
    upload = 2 * block + max(_CENTER_BYTES, m * 8 + _CENTER_ROW_BYTES)
    return max(upload, sweep, assembly)


def _query_batch_bytes(m: int) -> int:
    # Float64 window copy + float32 cast/upload, plus the row reductions,
    # midpoint/scale vectors, masks, and ufunc transients held by local
    # normalization. Raw mode uses less, so this is conservative there.
    return m * 24 + _CENTER_ROW_BYTES


def _fallback_cell(k: int, self_join: bool) -> int:
    """Device bytes per QT cell of the compiled fallback, covering the larger
    of MLX 0.30 and 0.32 (which needs 4 fewer): measured 16/8 (self/AB) at
    k=1; with the two-stage top-k selection 52/32 at k=5-16, 56/36 at k=100
    and 62/42 at k=256 (the chunk winners add ~40*k/_TOPK_CHUNK); 68/48 for
    the single-stage selection used above k=256. The top-k cells keep 4 B of
    headroom on MLX 0.30; the k == 1 cells are exact there (see
    ``_CHUNK_MEM_BUDGET``)."""
    if k == 1:
        return 16 if self_join else 8
    if 4 * k <= _TOPK_CHUNK:
        return (56 if self_join else 36) + (40 * k) // _TOPK_CHUNK
    return 72 if self_join else 52


def _batch_row_bytes(width: int, m: int, k: int, self_join: bool, fused: bool) -> int:
    """Device bytes one query row of a batch holds against ``width`` target
    columns: its QT row (plus the fallback's distance/sort intermediates),
    its reduction outputs, and its query window."""
    cell = _FUSED_CELL if fused else _fallback_cell(k, bool(self_join))
    # k == 1: up to six 4-byte vectors; top-k: values + indices, left/right
    out = 24 if k == 1 else 8 * k + 16
    return width * cell + out + _query_batch_bytes(m)


def _dense_batch(l: int, m: int, k: int, self_join: bool, fused: bool) -> int:
    per_row = _batch_row_bytes(l, m, k, self_join, fused)
    return max(1, min(1024, _CHUNK_MEM_BUDGET // per_row))


def _tiled_batch(tile_rows: int, m: int, k: int, self_join: bool, fused: bool) -> int:
    per_row = _batch_row_bytes(tile_rows, m, k, self_join, fused)
    if k > 1:
        per_row += k * _TILED_MERGE_CELL  # the host merge runs while the block is live
    return max(1, min(4096, _CHUNK_MEM_BUDGET // per_row))


def default_chunk_size(
    engine: MassEngine, l_q: int, k: int = 1, self_join: bool = False, fused: bool | None = None
) -> int:
    """Query rows per GPU batch such that live intermediates fit the budget.

    The budget is enforced (floor of one row), never overridden for
    throughput: callers who want bigger batches pass ``chunk_size``. The
    per-row cost depends on the reduction that runs (``fused`` defaults to
    ``_fused_reducer(k)``); batches are capped at 1024 rows.
    """
    fused = _fused_reducer(k) if fused is None else fused
    return min(_dense_batch(engine.l, engine.m, k, self_join, fused), max(1, l_q))


def tiled_chunk_size(
    engine: MassEngine, l_q: int, k: int = 1, self_join: bool = False, fused: bool | None = None
) -> int:
    """Query rows per batch in tiled mode: intermediates span one tile, not l.

    Top-k batches also charge the host merge workspace per row; batches are
    capped at 4096 rows.
    """
    fused = _fused_reducer(k) if fused is None else fused
    return min(_tiled_batch(engine.tile_rows, engine.m, k, self_join, fused), max(1, l_q))


class MassEngine:
    """Holds the target series' centered windows and per-window stats on the GPU.

    Sliding dot products are dense matmuls against the materialized (l, m)
    doubly-centered target window matrix — built in one piece when it fits
    the memory cap, or streamed as evenly-sized column blocks
    (``target_blocks``) when it does not. Both forms apply the same float64
    per-window mean-centering before the float32 cast, so precision is
    identical at every series length.
    """

    def __init__(self, target: PreprocessedSeries, *, normalize: bool = True):
        self.target = target
        self.normalize = normalize
        self.m = target.m
        self.l = target.l
        self.tiled = self.l * self.m * 4 > _MATMUL_WINDOW_BYTES
        # floor of 4 rows: 1-2-wide blocks hit GEMV-style kernels whose
        # accumulation differs from the wide-block GEMM in the last float32
        # bit and flips near-ties (costs at most ~4x _TILE_WINDOW_BYTES per
        # block for gigantic m, where a single window dwarfs the tile anyway)
        self.tile_rows = self.l if not self.tiled else max(4, _TILE_WINDOW_BYTES // (4 * self.m))
        self.W_T = None
        if not self.tiled:
            self.W_T = self._build_block_T(0, self.l)
            mx.eval(self.W_T)

    def _build_block_T(self, j0: int, j1: int) -> mx.array:
        """Build one transposed float32 target-window block.

        Z-normalized windows are copied from the raw series, put into their
        own bounded frame, mean-centered, and divided by their own RMS before
        the cast.  A single global standardized copy can re-round a tiny
        window embedded in a much larger-range series enough to change its
        nearest neighbor; local normalization removes that conditioning.
        Raw-distance windows keep the shared affine frame and rolling mean.
        """
        source = self.target.T if self.normalize else self.target.Ts
        w = np.lib.stride_tricks.sliding_window_view(source, self.m)[j0:j1]
        out = np.empty((j1 - j0, self.m), dtype=np.float32)
        step = _center_rows(self.m)
        for s in range(0, j1 - j0, step):
            e = min(s + step, j1 - j0)
            if self.normalize:
                work = w[s:e].copy()
                center_rows_stable(work)
                rms = np.sqrt(np.einsum("ij,ij->i", work, work) / self.m)
                rows = slice(j0 + s, j0 + e)
                active = self.target.isfinite[rows] & ~self.target.isconstant[rows]
                safe_rms = np.where(active & (rms > 0.0), rms, 1.0)
                work /= safe_rms[:, None]
                work[~active] = 0.0
                out[s:e] = work
                del active, rms, safe_rms, work
            else:
                out[s:e] = w[s:e] - self.target.mu[j0 + s : j0 + e, None]
        return mx.array(out).T

    def target_blocks(self) -> Iterator[tuple[int, int, mx.array]]:
        """Yield (j0, j1, block) covering all target windows in column order.

        Blocks are split as evenly as possible (never wider than
        ``tile_rows``): a narrow trailing block — especially a single
        column — would be dispatched to a different matmul kernel whose
        accumulation order can differ in the last float32 bit and flip
        near-ties to a different (equally good) neighbor.

        Only one block is meant to be alive at a time: the generator drops
        its own reference after yielding, and callers must ``del`` theirs
        before advancing (a ``for`` target is only rebound on the next
        iteration), or the previous block stays resident while the next one
        is built.
        """
        if not self.tiled:
            yield 0, self.l, self.W_T
            return
        nblocks = -(-self.l // self.tile_rows)
        base, extra = divmod(self.l, nblocks)
        j0 = 0
        for b in range(nblocks):
            j1 = j0 + base + (1 if b < extra else 0)
            block = self._build_block_T(j0, j1)
            mx.eval(block)
            yield j0, j1, block
            del block
            j0 = j1

    def znorm_sq_distances(
        self,
        QT: mx.array,
        sig_inv_q: mx.array,
        isconstant_q: mx.array,
        isfinite_q: mx.array,
        j0: int = 0,
        j1: int | None = None,
    ) -> mx.array:
        """(B, j1-j0) *squared* z-normalized distances with STUMPY's special cases.

        ``QT`` comes from query and target windows that were independently
        centered and scaled to unit RMS in float64 before their float32 cast,
        so it is ``m * rho`` directly. No global-frame cancellation or
        ``QT - m*mu_q*mu_t`` subtraction occurs in float32.

        Squared distances are what the search runs on (sqrt is monotonic and
        the reported profile values are re-evaluated in float64 anyway).
        """
        m = float(self.m)
        t = self.target
        j1 = self.l if j1 is None else j1
        return _znorm_sq(
            QT,
            sig_inv_q,
            isconstant_q,
            isfinite_q,
            t.sig_inv_mx[j0:j1],
            t.isconstant_mx[j0:j1],
            t.isfinite_mx[j0:j1],
            m,
        )

    def absolute_sq_distances(
        self,
        QT: mx.array,
        ssq_q: mx.array,
        mu_q: mx.array,
        isfinite_q: mx.array,
        j0: int = 0,
        j1: int | None = None,
    ) -> mx.array:
        """(B, j1-j0) squared non-normalized (p=2) distances, shared standardized frame.

        ``QT`` comes from centered windows and ``ssq`` values are centered
        sums of squares, so d2 = ||qc - tc||^2 + m*(mu_q - mu_t)^2 with the
        offset carried stably by the mean term; ``mu_q`` is a ``(B, 2)``
        float32 ``[hi, lo]`` split (see ``_preprocess.split_float32``).
        Multiply distances by the shared ``scale`` to return to original
        units.
        """
        t = self.target
        j1 = self.l if j1 is None else j1
        return _abs_sq(
            QT,
            ssq_q,
            mu_q,
            isfinite_q,
            t.ssq_mx[j0:j1],
            t.mu_mx[j0:j1],
            t.isfinite_mx[j0:j1],
            float(self.m),
        )


def _znorm_sq(
    QT, sig_inv_q, isconstant_q, isfinite_q, sig_inv_t, isconst_t, isfinite_t, m, inv_m=None
):
    # ``m``/``inv_m`` are Python floats (eager ``mass``) or the 0-d float32
    # arrays of the compiled sweep step. The step must not compute 1/m on the
    # device or receive it as a Python constant: mx.compile writes scalar
    # constants into the kernel source with ~7 significant digits (1/7 came
    # out 3 ulp off) and recompiles for every new value.
    if inv_m is None:
        inv_m = 1.0 / m
    rho = QT * (sig_inv_q[:, None] * sig_inv_t[None, :] * inv_m)
    # Correlation is mathematically in [-1, 1]. Float32 covariance/stat
    # rounding can stray by an ulp on either side, so enforce both distance
    # bounds; a lower-only clamp allowed raw MASS to exceed 2*sqrt(m).
    d2 = mx.minimum(mx.maximum(2.0 * m * (1.0 - rho), 0.0), 4.0 * m)
    q_const = isconstant_q[:, None]
    c_const = isconst_t[None, :]
    both = mx.logical_and(q_const, c_const)
    one = mx.logical_and(mx.logical_or(q_const, c_const), mx.logical_not(both))
    d2 = mx.where(both, 0.0, mx.where(one, m, d2))
    bad = mx.logical_or(mx.logical_not(isfinite_q[:, None]), mx.logical_not(isfinite_t[None, :]))
    return mx.where(bad, _INF, d2)


def _abs_sq(QT, ssq_q, mu_q, isfinite_q, ssq_t, mu_t, isfinite_t, m):
    # mu_* are (.., 2) float32 [hi, lo] splits of the float64 window means:
    # hi differences are exact for nearby means and lo carries the residual,
    # so dmu is accurate to float32 of the difference itself rather than of
    # the means (which carry the shared frame's global offset)
    dmu = (mu_q[:, 0][:, None] - mu_t[:, 0][None, :]) + (mu_q[:, 1][:, None] - mu_t[:, 1][None, :])
    d2 = mx.maximum(ssq_q[:, None] + ssq_t[None, :] - 2.0 * QT, 0.0) + m * dmu * dmu
    bad = mx.logical_or(mx.logical_not(isfinite_q[:, None]), mx.logical_not(isfinite_t[None, :]))
    return mx.where(bad, _INF, d2)


def _argmin_and_value(d2):
    I = mx.argmin(d2, axis=1)
    P2 = mx.take_along_axis(d2, I[:, None], axis=1)[:, 0]
    return I, P2


def _last_argmin_and_value(d2, j):
    """Like ``_argmin_and_value`` but the LAST minimum wins exact ties.

    The block-local index comes from the ``(1, W)`` column-index input ``j``
    rather than from ``d2.shape``: a shape-derived Python int would enter the
    compiled graph as a literal and cost a fresh Metal compile (~0.2 s) for
    every new series length.
    """
    rev = d2[:, ::-1]
    r = mx.argmin(rev, axis=1)
    P2 = mx.take_along_axis(rev, r[:, None], axis=1)[:, 0]
    return mx.take(j[0, ::-1], r) - j[0, 0], P2


def _topk(d2, key, k: int):
    """Per-row k smallest ``(d2, key)`` pairs in lexicographic order: the
    squared distances and their (non-negative, < 2**31) keys.

    One uint64 per cell, ``|d2| bits << 32 | key``, orders exactly like
    ``(d2, key)`` for the non-negative (or infinite) distances used here
    (clearing the sign bit makes -0.0 tie with +0.0, as float comparisons
    do) and carries both halves, so a values-only partition suffices and the
    caller decodes the column from the key. Wide rows are selected in two
    stages: the k smallest of every ``_TOPK_CHUNK``-column chunk, then of
    those. That is exact, because each of a row's k smallest keys is among
    its chunk's k smallest, and it replaces the full-row multi-block sort
    with single-block sorts (a 65437-wide row: 1.3 ms vs 3.2 ms per 96 rows,
    and less memory). Padding columns hold the maximal key and never reach
    the output, since every row has at least ``kk`` real columns.
    """
    B, W = d2.shape
    kk = min(k, W)
    mag = mx.view(d2, mx.uint32) & 0x7FFFFFFF
    comp = (mag.astype(mx.uint64) << 32) | key.astype(mx.uint64)
    if W > _TOPK_CHUNK and 4 * kk <= _TOPK_CHUNK:
        nc = -(-W // _TOPK_CHUNK)
        comp = mx.pad(comp, ((0, 0), (0, nc * _TOPK_CHUNK - W)), constant_values=mx.array(_U64_MAX))
        comp = mx.partition(comp.reshape(B * nc, _TOPK_CHUNK), kth=kk - 1, axis=1)[:, :kk]
        comp = comp.reshape(B, nc * kk)
    top = mx.sort(mx.partition(comp, kth=kk - 1, axis=1)[:, :kk], axis=1)
    return mx.view((top >> 32).astype(mx.uint32), mx.float32), (top & 0xFFFFFFFF).astype(mx.int32)


def reduce_consts(m: int) -> np.ndarray:
    """``[m, 1/m, 2m, 4m]`` in float32; ``1/m`` is rounded once, on the host.

    The fused kernels read all four from one buffer; ``ReduceStep`` receives
    ``m`` and the same ``1/m`` as 0-d arrays and forms ``2m`` and ``4m`` on
    the device, which is exact in float32 for integer ``m``. Sharing these
    values is part of what makes the two reductions bit-identical.
    """
    m = float(m)
    return np.array([m, 1.0 / m, 2.0 * m, 4.0 * m], dtype=np.float32)


def _fused_reducer(k: int) -> bool:
    """The one dispatch predicate: does this process run the fused kernels?

    True only on a Metal GPU that is MLX's default device, for ``k`` within
    the top-k kernel's range, and when the ``k`` kernels launch on this GPU
    (``_kernels.launch_threadgroup``: probed once per process and ``k``;
    a failure warns and falls back). Both the reducer choice
    (``make_reducer``) and the batch sizing (``default_chunk_size``,
    ``tiled_chunk_size``, ``estimated_peak_bytes``) are derived from it, so a
    batch sized for the fused path's 4 B/cell never reaches the fallback.
    """
    from ._kernels import FUSED_TOPK_MAX, launch_threadgroup

    return (
        k <= FUSED_TOPK_MAX
        and mx.metal.is_available()
        and mx.default_device() == mx.gpu
        and launch_threadgroup(k) is not None
    )


def make_reducer(
    query: PreprocessedSeries,
    engine: MassEngine,
    *,
    normalize: bool,
    self_join: bool,
    excl: int,
    k: int,
    fused: bool,
):
    """The sweep reduction for one ``stump`` call: ``FusedReduce`` when
    ``fused`` (``_fused_reducer(k)``), else the compiled ``ReduceStep``."""
    kwargs = dict(
        normalize=normalize,
        self_join=self_join,
        excl=excl,
        k=k,
        consts=reduce_consts(engine.m),
    )
    if fused:
        from ._kernels import FusedReduce

        return FusedReduce(query, engine.target, **kwargs)
    return ReduceStep(query, engine.target, **kwargs)


class ReduceStep:
    """Compiled fallback reduction: squared distances + the per-row selections.

    This is the reference the fused Metal kernels (``_kernels.FusedReduce``)
    are tested against bit for bit, and it runs whenever they cannot: on the
    CPU device, without Metal, and for ``k > FUSED_TOPK_MAX``.

    Every array — the QT block, both series' stats, the row/column index
    vectors and the constants ``m``, ``1/m`` and ``excl`` (0-d arrays) — is an
    explicit input, so one trace serves every window size; only ``k`` is
    structural (the argpartition ``kth``). (Closure-capturing the target
    arrays would also alias a traced input for single-chunk self-joins, which
    MLX's compile rejects.) Running the chain as one compiled graph keeps the
    per-batch cost memory-bound instead of dispatch-bound; the uncompiled
    op-by-op form ran the tiled sweep ~2.5x slower.

    ``full(QT, s0)`` reduces the batch whose first query row is ``s0``
    against the whole target; ``block(QT, s0, j0, j1)`` against target
    columns ``j0:j1`` (tiled mode). Indices are block-local. Outputs:

    - self-join, k == 1: ``I, P2, Il, Pl2, Ir, Pr2``;
    - self-join, k > 1: ``vals2, idxs, Il, Pl2, Ir, Pr2`` (the top-k set
      excludes the trivial-match zone);
    - AB-join: ``I, P2`` (k == 1) or ``vals2, idxs``.

    Tie rule (STUMPY's): every selection is lexicographic in ``(d2, key)``.
    Self-joins use ``key = 2*|j - i| + (j > i)`` — the order in which
    STUMPY's diagonal traversal visits candidates (ascending offset, the left
    one first) and so the one its strict comparisons keep: the nearest-in-time
    candidate wins and the left one on an equal offset. Hence the left
    minimum is the LAST minimum over ``j <= i - excl - 1`` and the right one
    the FIRST over ``j >= i + excl + 1``. AB-joins use ``key = j`` (lowest
    column). Rows with no finite candidate report an infinite value; their
    index is meaningless and callers map it to -1.
    """

    def __init__(
        self,
        query: PreprocessedSeries,
        target: PreprocessedSeries,
        *,
        normalize: bool,
        self_join: bool,
        excl: int,
        k: int,
        consts: np.ndarray,
    ):
        self.k = k
        self.self_join = self_join
        if normalize:
            self._q = (query.sig_inv_mx, query.isconstant_mx, query.isfinite_mx)
            self._t = (target.sig_inv_mx, target.isconstant_mx, target.isfinite_mx)

            def dist(QT, a, b, qf, t_a, t_b, t_f, m, inv_m):
                return _znorm_sq(QT, a, b, qf, t_a, t_b, t_f, m, inv_m)

        else:
            self._q = (query.ssq_mx, query.mu_mx, query.isfinite_mx)
            self._t = (target.ssq_mx, target.mu_mx, target.isfinite_mx)

            def dist(QT, a, b, qf, t_a, t_b, t_f, m, inv_m):
                return _abs_sq(QT, a, b, qf, t_a, t_b, t_f, m)

        self._consts = (
            mx.array(consts[0]),
            mx.array(consts[1]),
            mx.array(int(excl), dtype=mx.int32),
        )
        self._j_full = mx.arange(target.l)[None, :]

        if self_join:

            def step(QT, a, b, qf, i_col, t_a, t_b, t_f, j, m, inv_m, excl):
                d2 = dist(QT, a, b, qf, t_a, t_b, t_f, m, inv_m)
                # nearest candidate on each side: last minimum on the left,
                # first on the right
                Il, Pl2 = _last_argmin_and_value(mx.where(j <= i_col - (excl + 1), d2, _INF), j)
                Ir, Pr2 = _argmin_and_value(mx.where(j >= i_col + (excl + 1), d2, _INF))
                if k == 1:
                    # combine by (d2, offset) on GLOBAL columns (block mode)
                    i, jr = i_col[:, 0], j[0]
                    left_better = (Pl2 < Pr2) | ((Pl2 == Pr2) & ((i - jr[Il]) <= (jr[Ir] - i)))
                    I = mx.where(left_better, Il, Ir)
                    P2 = mx.where(left_better, Pl2, Pr2)
                    return I, P2, Il, Pl2, Ir, Pr2
                off = j - i_col
                key = 2 * mx.abs(off) + (off > 0)
                vals2, top = _topk(mx.where(mx.abs(off) <= excl, _INF, d2), key, k)
                half = top >> 1
                idxs = i_col + mx.where((top & 1) == 1, half, -half) - j[0, 0]
                return vals2, idxs, Il, Pl2, Ir, Pr2

        else:

            def step(QT, a, b, qf, i_col, t_a, t_b, t_f, j, m, inv_m, excl):
                d2 = dist(QT, a, b, qf, t_a, t_b, t_f, m, inv_m)
                if k == 1:
                    return _argmin_and_value(d2)
                vals2, top = _topk(d2, j, k)
                return vals2, top - j[0, 0]

        self._compiled = mx.compile(step)

    def full(self, QT, s0: int):
        s0 = int(s0)
        e = s0 + QT.shape[0]
        qa, qb, qf = (x[s0:e] for x in self._q)
        i_col = mx.arange(s0, e)[:, None]
        return self._compiled(QT, qa, qb, qf, i_col, *self._t, self._j_full, *self._consts)

    def block(self, QT, s0: int, j0: int, j1: int):
        s0, j0, j1 = int(s0), int(j0), int(j1)
        e = s0 + QT.shape[0]
        qa, qb, qf = (x[s0:e] for x in self._q)
        i_col = mx.arange(s0, e)[:, None]
        t = (x[j0:j1] for x in self._t)
        j_row = mx.arange(j0, j1)[None, :]
        return self._compiled(QT, qa, qb, qf, i_col, *t, j_row, *self._consts)


def query_windows(
    query: PreprocessedSeries, start: int, stop: int, *, normalize: bool
) -> mx.array:
    """Return one float32 query-window batch for the selected distance mode."""
    if normalize:
        w = np.lib.stride_tricks.sliding_window_view(query.T, query.m)[start:stop].copy()
        center_rows_stable(w)
        rms = np.sqrt(np.einsum("ij,ij->i", w, w) / query.m)
        active = query.isfinite[start:stop] & ~query.isconstant[start:stop]
        safe_rms = np.where(active & (rms > 0.0), rms, 1.0)
        w /= safe_rms[:, None]
        w[~active] = 0.0
    else:
        source = np.lib.stride_tricks.sliding_window_view(query.Ts, query.m)[start:stop]
        w = source - query.mu[start:stop, None]
    return mx.array(w.astype(np.float32))
