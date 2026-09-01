"""Batched-MASS GPU engine: exact distance profiles via doubly-centered matmul.

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
  the noise with (segment offset / scale)^2 on mixed-scale data.

When the full window matrix would exceed the memory cap, the target is
processed in evenly-sized column blocks (each block built, centered, and
uploaded on demand), which preserves the identical per-window centering at
any length. (An FFT cross-correlation fallback was used here previously; it
operated on the raw float32 series, could not center per-window, and was
numerically wrong for near-constant data.)

Distances follow STUMPY's semantics exactly:
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
# piece; above it the engine switches to tiled column blocks
_MATMUL_WINDOW_BYTES = 1 << 29
# size of one materialized target window block in tiled mode
_TILE_WINDOW_BYTES = 1 << 28
# bound on the float64 centering intermediate while building window blocks
_CENTER_STEP = 1 << 16


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
        self.tile_rows = self.l if not self.tiled else max(1, _TILE_WINDOW_BYTES // (4 * self.m))
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
        for s in range(0, j1 - j0, _CENTER_STEP):
            e = min(s + _CENTER_STEP, j1 - j0)
            out[s:e] = w[s:e] - self.target.mu[j0 + s : j0 + e, None]
        return mx.array(out).T

    def target_blocks(self) -> Iterator[tuple[int, int, mx.array]]:
        """Yield (j0, j1, block) covering all target windows in column order.

        Blocks are split as evenly as possible (never wider than
        ``tile_rows``): a narrow trailing block — especially a single
        column — would be dispatched to a different matmul kernel whose
        accumulation order can differ in the last float32 bit and flip
        near-ties to a different (equally good) neighbor.
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
            j0 = j1

    def sliding_dot_products(self, Q_batch: mx.array) -> mx.array:
        """Centered QT for a (B, m) float32 centered query batch -> (B, l).

        Materializes the full row; tiled callers that need bounded memory
        should loop ``target_blocks`` themselves instead.
        """
        if not self.tiled:
            return mx.matmul(Q_batch, self.W_T)
        parts = [mx.matmul(Q_batch, block) for _, _, block in self.target_blocks()]
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
        query series' own standardized frame; mixing frames is exact because
        the covariance is bilinear.

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
        offset carried exactly by the mean term. Multiply distances by the
        shared ``scale`` to return to original units.
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
    d2 = mx.maximum(2.0 * m * (1.0 - rho), 0.0)
    q_const = isconstant_q[:, None]
    c_const = isconst_t[None, :]
    both = mx.logical_and(q_const, c_const)
    one = mx.logical_and(mx.logical_or(q_const, c_const), mx.logical_not(both))
    d2 = mx.where(both, 0.0, mx.where(one, m, d2))
    bad = mx.logical_or(mx.logical_not(isfinite_q[:, None]), mx.logical_not(isfinite_t[None, :]))
    return mx.where(bad, _INF, d2)


def _abs_sq(QT, ssq_q, mu_q, isfinite_q, ssq_t, mu_t, isfinite_t, m):
    dmu = mu_q[:, None] - mu_t[None, :]
    d2 = mx.maximum(ssq_q[:, None] + ssq_t[None, :] - 2.0 * QT, 0.0) + m * dmu * dmu
    bad = mx.logical_or(mx.logical_not(isfinite_q[:, None]), mx.logical_not(isfinite_t[None, :]))
    return mx.where(bad, _INF, d2)


def _argmin_and_value(d2):
    I = mx.argmin(d2, axis=1)
    P2 = mx.take_along_axis(d2, I[:, None], axis=1)[:, 0]
    return I, P2


def make_reduce_step(engine: MassEngine, *, normalize: bool, self_join: bool, excl: int):
    """Compile one fused kernel: squared distances + (left/right) argmin.

    Only ``m``/``excl`` are baked in as constants; every array — the QT
    block, both series' stats, and the row-index column — is an explicit
    argument. (Closure-capturing the target arrays would alias a traced
    input for single-chunk self-joins, which MLX's compile rejects.) Running
    the whole chain as one compiled graph is what keeps the per-chunk cost
    memory-bound instead of dispatch-bound.

    The returned callable takes ``(QT, a, b, qf, i_col)`` where ``(a, b)``
    are the query-side ``(sig_inv, isconstant)`` slices (z-normalized) or
    ``(ssq, mu)`` slices (absolute).
    """
    t = engine.target
    m = float(engine.m)
    j_row = mx.arange(engine.l)[None, :]

    if normalize:
        t_a, t_b = t.sig_inv_mx, t.isconstant_mx

        def dist(QT, a, b, qf, t_a, t_b, t_f):
            return _znorm_sq(QT, a, b, qf, t_a, t_b, t_f, m)

    else:
        t_a, t_b = t.ssq_mx, t.mu_mx

        def dist(QT, a, b, qf, t_a, t_b, t_f):
            return _abs_sq(QT, a, b, qf, t_a, t_b, t_f, m)

    if self_join:

        def step(QT, a, b, qf, i_col, t_a, t_b, t_f, j):
            d2 = dist(QT, a, b, qf, t_a, t_b, t_f)
            # the left/right regions already exclude the trivial-match zone,
            # and their combined minimum IS the global minimum; on exact ties
            # the left (lower-index) candidate wins, matching full-row argmin
            dl = mx.where(j <= i_col - (excl + 1), d2, _INF)
            Il, Pl2 = _argmin_and_value(dl)
            dr = mx.where(j >= i_col + (excl + 1), d2, _INF)
            Ir, Pr2 = _argmin_and_value(dr)
            left_better = Pl2 <= Pr2
            I = mx.where(left_better, Il, Ir)
            P2 = mx.where(left_better, Pl2, Pr2)
            return I, P2, Il, Pl2, Ir, Pr2

    else:

        def step(QT, a, b, qf, i_col, t_a, t_b, t_f, j):
            I, P2 = _argmin_and_value(dist(QT, a, b, qf, t_a, t_b, t_f))
            return I, P2

    compiled = mx.compile(step)
    isfinite_t = t.isfinite_mx

    def run(QT, a, b, qf, i_col):
        return compiled(QT, a, b, qf, i_col, t_a, t_b, isfinite_t, j_row)

    return run


def query_windows(query: PreprocessedSeries, start: int, stop: int) -> mx.array:
    """Float32 (B, m) centered window batch from the standardized series.

    Each window's float64 rolling mean is subtracted *before* the float32
    cast, which is what keeps both the covariance (z-normalized) and the
    centered difference norm (absolute) well-conditioned on the GPU.
    """
    w = np.lib.stride_tricks.sliding_window_view(query.Ts, query.m)[start:stop]
    w = w - query.mu[start:stop, None]
    return mx.array(w.astype(np.float32))
