"""Batched-MASS GPU engine: matrix profiles via doubly-centered matmul.

For a query batch Q (B, m) and a target series T (n,), the sliding dot
products QT[b, j] = sum_k Q[b, k] * T[j + k] are one dense matmul against the
materialized (l, m) target window matrix. Each distance profile comes from a
fresh product rather than a long floating-point recurrence, so error does not
accumulate along the series.

EVERY window on both sides is mean-centered in float64 before the float32
cast, so the matmul product IS the mean-centered cross-covariance:

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

Memory accounting. Three byte budgets bound each ordinary computation;
documented one-row/one-block floors can exceed a nominal budget at extreme
``m``:

- the resident window block: the whole ``(m, l)`` float32 matrix when it fits
  ``_MATMUL_WINDOW_BYTES``, else one ``_TILE_WINDOW_BYTES`` column block at a
  time (each block is released before the next one is built);
- the live per-batch GPU intermediates, ``_CHUNK_MEM_BUDGET``
  (``default_chunk_size``/``tiled_chunk_size`` size the query batch to it);
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

from ._preprocess import PreprocessedSeries

_INF = float("inf")

# budget for the live per-chunk GPU intermediates (the query window batch,
# QT, squared distances, the masked left/right variants, and the top-k
# sort/gather buffers). Actual peak memory also includes the per-call
# constants (the window matrix or one tile of it) on top of this.
_CHUNK_MEM_BUDGET = 3 << 27  # ~384 MiB
# cap for materializing the (l, m) target window matrix on the GPU in one
# piece; above it the engine switches to tiled column blocks. The tiled
# sweep runs the same compiled reduce step per block and measured as fast
# as or faster than the dense one (n=524288, m=200: 47.7 s tiled/128 MiB
# vs 47.9 s dense in one run, 38.8 s vs 52.8 s in another, at about half
# the peak memory), so the cap trades nothing for memory.
_MATMUL_WINDOW_BYTES = 1 << 28  # ~256 MiB
# size of one materialized target window block in tiled mode
_TILE_WINDOW_BYTES = 1 << 27  # ~128 MiB
# byte bound on the float64 centering temporary while building window blocks
_CENTER_BYTES = 1 << 26  # ~64 MiB
# byte budget for the float64 window copies held live by one refinement
# chunk (two fancy-indexed window blocks plus their centered copies); the
# refinement runs after the device memory has been released
_REFINE_MEM_BUDGET = 1 << 28  # ~256 MiB


def refine_chunk_rows(m: int) -> int:
    return max(1, min(1 << 16, _REFINE_MEM_BUDGET // (m * 8 * 4)))


def _center_rows(m: int) -> int:
    """Window rows centered per float64 step so the temporary fits ``_CENTER_BYTES``."""
    return max(1, _CENTER_BYTES // (m * 8))


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
) -> int:
    """Estimate of the bytes one join needs beyond its O(n) per-series arrays.

    ``l`` is the number of target windows (the side the engine
    materializes), ``l_q`` the number of query windows (the output rows;
    defaults to ``l``), and ``chunk_size`` has the same meaning as in
    :func:`mlx_stump.stump`. With no explicit chunk size, the automatic
    byte-budgeted batch is modeled; with one, the requested batch (clamped
    to ``l_q``) is modeled, including requests that deliberately exceed the
    automatic ~384 MiB device budget. The largest of three phases:

    - upload: the block staged in numpy plus its device copy plus the
      centering temporary;
    - sweep: the resident block plus the per-batch intermediates budget (or
      one batch row, when even a single row exceeds the budget), plus the
      numeric profile/index outputs (float64 + int64 per neighbor,
      left/right indices; the tiled sweep also keeps float32/int64 top-k
      accumulators). The budget is enforced batch by batch — each batch is
      synchronized before the next allocates and the trailing batch is
      computed at full width — so exactly one set of intermediates exists.
      Tiled top-k joins also merge each device result into the running set
      on the host; the two concatenations, full stable-argsort permutation,
      gather results, and sorting workspace are included here;
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
    l, m, k, l_q = int(l), int(m), int(k), int(l_q)
    self_join = bool(self_join)
    block = resident_block_bytes(l, m)
    full = l * m * 4
    tiled = full > _MATMUL_WINDOW_BYTES
    width = block // (m * 4)  # columns of the resident block
    cell = 16 if k == 1 else (48 if self_join else 40)
    one_row = width * cell + _query_batch_bytes(m)
    if chunk_size is None:
        # Match the actual sizing helpers. Tiled batches are sized against
        # MassEngine.tile_rows (the nominal upper bound), while blocks are
        # subsequently balanced and can be narrower than that bound.
        sizing_width = max(4, _TILE_WINDOW_BYTES // (4 * m)) if tiled else l
        sizing_row = sizing_width * cell + _query_batch_bytes(m)
        batch_cap = 4096 if tiled else 1024
        batch = max(1, min(batch_cap, _CHUNK_MEM_BUDGET // sizing_row))
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
        # _merge_topk holds the float32/int64 block result, two 2k-wide
        # concatenations, the full int64 stable-argsort result, both gathered
        # outputs, and NumPy's stable-sort workspace. 64 B/cell covers the
        # named arrays; 80 B/cell leaves headroom for the sort implementation.
        host_batch = batch * k * 80
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
    return max(2 * block + _CENTER_BYTES, sweep, assembly)


def _query_batch_bytes(m: int) -> int:
    # float64 window copy + centered copy + float32 cast + device upload
    return m * 24


def default_chunk_size(engine: MassEngine, l_q: int, k: int = 1, self_join: bool = False) -> int:
    """Query rows per GPU batch such that live intermediates fit the budget.

    The budget is enforced (floor of one row), never overridden for
    throughput: callers who want bigger batches pass ``chunk_size``.
    """
    if k == 1:
        per_row = engine.l * 16  # QT + d2 + two masked left/right variants, float32
    else:
        # the top-k path additionally holds argpartition/argsort index and
        # gather intermediates: ~48 bytes/cell measured for self-joins
        # (which also carry the masked left/right variants), ~40 without
        per_row = engine.l * (48 if self_join else 40)
    per_row += _query_batch_bytes(engine.m)
    b = max(1, min(1024, _CHUNK_MEM_BUDGET // per_row))
    return min(b, max(1, l_q))


def tiled_chunk_size(engine: MassEngine, l_q: int, k: int = 1, self_join: bool = False) -> int:
    """Query rows per batch in tiled mode: intermediates span one tile, not l."""
    cell = 16 if k == 1 else (48 if self_join else 40)
    per_row = engine.tile_rows * cell + _query_batch_bytes(engine.m)
    b = max(1, min(4096, _CHUNK_MEM_BUDGET // per_row))
    return min(b, max(1, l_q))


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
        """(m, j1-j0) float32 transposed window block for target windows
        ``j0:j1``, each window centered by its float64 rolling mean before
        the cast."""
        w = np.lib.stride_tricks.sliding_window_view(self.target.Ts, self.m)[j0:j1]
        out = np.empty((j1 - j0, self.m), dtype=np.float32)
        step = _center_rows(self.m)
        for s in range(0, j1 - j0, step):
            e = min(s + step, j1 - j0)
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

    def sliding_dot_products(self, Q_batch: mx.array) -> mx.array:
        """Centered QT for a (B, m) float32 centered query batch -> (B, l).

        Materializes the full (B, l) row. In tiled mode each block's product
        is evaluated before the next block is built, so only one block is
        resident at a time; callers that also need the distances bounded per
        block (``mass``, the tiled ``stump`` sweep) loop ``target_blocks``
        themselves instead.
        """
        if not self.tiled:
            return mx.matmul(Q_batch, self.W_T)
        parts = []
        for _, _, block in self.target_blocks():
            part = mx.matmul(Q_batch, block)
            mx.eval(part)  # a lazy product would pin every block until concatenation
            parts.append(part)
            del block
        return mx.concatenate(parts, axis=1)

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

        ``QT`` must come from *mean-centered* query windows against
        *mean-centered* target windows: the doubly-centered product IS the
        mean-centered cross-covariance, so no catastrophic `QT - m*mu_q*mu_t`
        subtraction ever happens in float32. Query-side stats live in the
        query series' own standardized frame; mixing frames is algebraically
        valid because the covariance is bilinear.

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


def _znorm_sq(QT, sig_inv_q, isconstant_q, isfinite_q, sig_inv_t, isconst_t, isfinite_t, m):
    rho = QT * (sig_inv_q[:, None] * sig_inv_t[None, :] * (1.0 / m))
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


def _topk(d2, k: int):
    """Per-row k smallest squared distances (ascending) and their columns."""
    kk = min(k, d2.shape[1])
    part = mx.argpartition(d2, kth=kk - 1, axis=1)[:, :kk]
    vals = mx.take_along_axis(d2, part, axis=1)
    order = mx.argsort(vals, axis=1)
    vals = mx.take_along_axis(vals, order, axis=1)
    idxs = mx.take_along_axis(part, order, axis=1)
    return vals, idxs


class ReduceStep:
    """One compiled kernel: squared distances + the per-row reductions.

    Only ``m``/``excl``/``k`` are baked in as constants; every array — the QT
    block, both series' stats, and the row/column index vectors — is an
    explicit argument. (Closure-capturing the target arrays would alias a
    traced input for single-chunk self-joins, which MLX's compile rejects.)
    Running the whole chain as one compiled graph is what keeps the
    per-chunk cost memory-bound instead of dispatch-bound — the uncompiled
    op-by-op form materializes every elementwise intermediate and ran the
    tiled sweep ~2.5x slower than the dense one.

    ``full`` reduces a chunk against the whole target row; ``block`` against
    the target columns ``j0:j1`` (tiled mode). Both take ``(QT, a, b, qf,
    i_col)`` where ``(a, b)`` are the query-side ``(sig_inv, isconstant)``
    slices (z-normalized) or ``(ssq, mu)`` slices (absolute) and ``i_col``
    the ``(B, 1)`` query row indices. Outputs, in order:

    - self-join, k == 1: ``I, P2, Il, Pl2, Ir, Pr2``;
    - self-join, k > 1: ``vals2, idxs, Il, Pl2, Ir, Pr2`` (the top-k set
      excludes the trivial-match zone);
    - AB-join: ``I, P2`` (k == 1) or ``vals2, idxs``.

    Left/right regions already exclude the trivial-match zone, and their
    combined minimum IS the global minimum; on exact ties the left
    (lower-index) candidate wins, matching a full-row argmin.
    """

    def __init__(
        self, engine: MassEngine, *, normalize: bool, self_join: bool, excl: int, k: int = 1
    ):
        t = engine.target
        m = float(engine.m)
        self.k = k
        self.self_join = self_join
        if normalize:
            self.t_a, self.t_b = t.sig_inv_mx, t.isconstant_mx

            def dist(QT, a, b, qf, t_a, t_b, t_f):
                return _znorm_sq(QT, a, b, qf, t_a, t_b, t_f, m)

        else:
            self.t_a, self.t_b = t.ssq_mx, t.mu_mx

            def dist(QT, a, b, qf, t_a, t_b, t_f):
                return _abs_sq(QT, a, b, qf, t_a, t_b, t_f, m)

        self.t_f = t.isfinite_mx
        self.j_row = mx.arange(engine.l)[None, :]

        if self_join:

            def step(QT, a, b, qf, i_col, t_a, t_b, t_f, j):
                d2 = dist(QT, a, b, qf, t_a, t_b, t_f)
                dl = mx.where(j <= i_col - (excl + 1), d2, _INF)
                Il, Pl2 = _argmin_and_value(dl)
                dr = mx.where(j >= i_col + (excl + 1), d2, _INF)
                Ir, Pr2 = _argmin_and_value(dr)
                if k == 1:
                    left_better = Pl2 <= Pr2
                    I = mx.where(left_better, Il, Ir)
                    P2 = mx.where(left_better, Pl2, Pr2)
                    return I, P2, Il, Pl2, Ir, Pr2
                vals2, idxs = _topk(mx.where(mx.abs(i_col - j) <= excl, _INF, d2), k)
                return vals2, idxs, Il, Pl2, Ir, Pr2

        else:

            def step(QT, a, b, qf, i_col, t_a, t_b, t_f, j):
                d2 = dist(QT, a, b, qf, t_a, t_b, t_f)
                if k == 1:
                    return _argmin_and_value(d2)
                return _topk(d2, k)

        self._compiled = mx.compile(step)

    def full(self, QT, a, b, qf, i_col):
        return self._compiled(QT, a, b, qf, i_col, self.t_a, self.t_b, self.t_f, self.j_row)

    def block(self, QT, a, b, qf, i_col, j0: int, j1: int, j_row):
        return self._compiled(
            QT, a, b, qf, i_col, self.t_a[j0:j1], self.t_b[j0:j1], self.t_f[j0:j1], j_row
        )


def query_windows(query: PreprocessedSeries, start: int, stop: int) -> mx.array:
    """Float32 (B, m) centered window batch from the standardized series.

    Each window's float64 rolling mean is subtracted *before* the float32
    cast, which is what keeps both the covariance (z-normalized) and the
    centered difference norm (absolute) well-conditioned on the GPU.
    """
    w = np.lib.stride_tricks.sliding_window_view(query.Ts, query.m)[start:stop]
    w = w - query.mu[start:stop, None]
    return mx.array(w.astype(np.float32))
