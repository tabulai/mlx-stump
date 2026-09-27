"""Research prototype: triangular dense self-join for ``k == 1`` on Metal.

The packed query and target windows are the same object in this case, so a
window pair has the same score in both orientations. An off-diagonal tile
computes its GEMM only once. The ordinary row reducer finds right neighbors
of the first tile; a column reducer reads the *same* row-major product to find
left neighbors of the second tile. Its lanes span adjacent columns, making
the strided orientation coalesced without allocating ``QT.T``.

One tile product uses at most ``4*C*C`` bytes at tile width ``C`` (256 MiB at
``C=8192``), plus linear-size outputs and the separately resident packed
windows. Only compact left/right winners cross to the CPU. Keeping their float32 scores
until every tile has merged preserves this prototype's ``(score, tie key)``
order. Different GEMM shapes and transposed orientations can change float32
accumulation near a tie, including nonzero ties. The production zero-band
repair does not certify general rank parity, so this path is deliberately
**not** selected by the library. This benchmark compares *exact* public
outputs with the production sweep and reports every mismatch.

Usage:
    PYTHONPATH=src python bench/bench_symmetric.py --n 16384 --m 200 --repeat 3
    PYTHONPATH=src python bench/bench_symmetric.py --n 8192 --m 200 --tile-caps 4096
"""

from __future__ import annotations

import argparse
import gc
import time

import mlx.core as mx
import numpy as np

import mlx_stump
from mlx_stump import _engine, _stump
from mlx_stump._engine import make_reducer
from mlx_stump._kernels import _ABS_D2, _ABS_PRE, _HEADER, _ZNORM_D2, _ZNORM_PRE

_COL_LANES = 32
_ROW_LANES = 8
_COL_TG = _COL_LANES * _ROW_LANES
_COLUMN_KERNELS: dict[bool, object] = {}


def balanced_tiles(length: int, cap: int) -> list[tuple[int, int]]:
    """Return almost-equal tiles, or none if triangular reuse is impossible."""
    if length < 2 or cap < 1:
        return []
    count = -(-length // cap)
    if count < 2:
        return []
    width, extra = divmod(length, count)
    tiles = []
    start = 0
    for t in range(count):
        stop = start + width + (t < extra)
        tiles.append((start, stop))
        start = stop
    return tiles


def _column_kernel(normalize: bool):
    if normalize not in _COLUMN_KERNELS:
        pre = _ZNORM_PRE if normalize else _ABS_PRE
        distance = _ZNORM_D2 if normalize else _ABS_D2
        source = (
            r"""
    const uint tid = thread_position_in_threadgroup.x;
    const uint col_lane = tid % COL_LANES;
    const uint row_lane = tid / COL_LANES;
    const uint col = threadgroup_position_in_grid.x * COL_LANES + col_lane;
    const int width = QT_shape[1];
    const int height = QT_shape[0];
    const int i0 = par[0];
    const int j0 = par[1];
    const int EXCL = par[2];
    const float c_m = consts[0];
    const float c_inv_m = consts[1];
    const float c_2m = consts[2];
    const float c_4m = consts[3];
    float best = INFINITY;
    uint best_key = 0xFFFFFFFFu;
    if (col < (uint)width) {
        const int i = j0 + (int)col;  // output row in the right-hand tile
        const int j = (int)col;        // its column in the row-major QT tile
        const bool q_fin = qf[i];
"""
            + pre
            + r"""
        // The nearest left candidate has the smallest tie key. Descending
        // traversal also avoids repeated top-k insertion on exact plateaus.
        for (int local = height - 1 - (int)row_lane; local >= 0;
             local -= ROW_LANES) {
            const int jg = i0 + local;
            const size_t qbase = (size_t)local * (size_t)width;
"""
            + distance
            + r"""
            if (jg <= i - (EXCL + 1)) {
                const uint key = 2u * (uint)(i - jg);
                lex_min(best, best_key, d2, key);
            }
        }
    }
    threadgroup float partial_v[COL_LANES * ROW_LANES];
    threadgroup uint partial_k[COL_LANES * ROW_LANES];
    partial_v[tid] = best;
    partial_k[tid] = best_key;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row_lane == 0 && col < (uint)width) {
        float value = INFINITY;
        uint key = 0xFFFFFFFFu;
        for (uint lane = 0; lane < ROW_LANES; lane++) {
            const uint slot = lane * COL_LANES + col_lane;
            lex_min(value, key, partial_v[slot], partial_k[slot]);
        }
        outI[col] = metal::isfinite(value) ? j0 + (int)col - (int)(key / 2u) : -1;
        outP[col] = value;
    }
"""
        )
        _COLUMN_KERNELS[normalize] = mx.fast.metal_kernel(
            name=f"mlx_stump_symmetric_column_{'z' if normalize else 'a'}",
            input_names=["QT", "qa", "qb", "qf", "ta", "tb", "tf", "par", "consts"],
            output_names=["outI", "outP"],
            source=source,
            header=(
                f"#define COL_LANES {_COL_LANES}\n#define ROW_LANES {_ROW_LANES}\n" + _HEADER
            ),
        )
    return _COLUMN_KERNELS[normalize]


class _ColumnReduce:
    def __init__(self, series, normalize: bool, excl: int, consts: mx.array):
        if normalize:
            self._stats = (series.sig_inv_mx, series.isconstant_mx, series.isfinite_mx)
        else:
            self._stats = (series.ssq_mx, series.mu_mx, series.isfinite_mx)
        self._kernel = _column_kernel(normalize)
        self._consts = consts
        self._excl = min(int(excl), int(series.l))

    def __call__(self, QT: mx.array, i0: int, j0: int):
        width = QT.shape[1]
        par = mx.array([i0, j0, self._excl, 0], dtype=mx.int32)
        return self._kernel(
            inputs=[QT, *self._stats, *self._stats, par, self._consts],
            grid=(_COL_TG * ((width + _COL_LANES - 1) // _COL_LANES), 1, 1),
            threadgroup=(_COL_TG, 1, 1),
            output_shapes=[(width,), (width,)],
            output_dtypes=[mx.int32, mx.float32],
        )


def _merge_side(
    values: np.ndarray,
    indices: np.ndarray,
    start: int,
    candidate_values: np.ndarray,
    candidate_indices: np.ndarray,
    *,
    left: bool,
) -> None:
    """Merge one tile's winners with the global side profile in tie order."""
    end = start + candidate_values.size
    old_values = values[start:end]
    old_indices = indices[start:end]
    valid = (candidate_indices >= 0) & np.isfinite(candidate_values)
    closer = candidate_indices > old_indices if left else candidate_indices < old_indices
    better = valid & (
        (candidate_values < old_values)
        | ((candidate_values == old_values) & ((old_indices < 0) | closer))
    )
    old_values[better] = candidate_values[better]
    old_indices[better] = candidate_indices[better]


def compute_symmetric_k1(query, engine, *, normalize: bool, excl: int, tiles):
    """Return ``(I, IL, IR, P2, PL2, PR2)`` for an implicit dense self-join.

    All index arrays use global window indices; scores stay float32 so the
    caller can apply its ordinary near-zero ambiguity check unchanged.
    """
    length = query.l
    left_p = np.full(length, np.inf, dtype=np.float32)
    right_p = np.full(length, np.inf, dtype=np.float32)
    left_i = np.full(length, -1, dtype=np.int64)
    right_i = np.full(length, -1, dtype=np.int64)
    reducer = make_reducer(
        query, engine, normalize=normalize, self_join=True, excl=excl, k=1, fused=True
    )
    column = _ColumnReduce(query, normalize, excl, reducer._consts)
    packed = engine.W_T.T

    for a, (i0, i1) in enumerate(tiles):
        first = packed[i0:i1]
        for b in range(a, len(tiles)):
            j0, j1 = tiles[b]
            second = packed[j0:j1]
            QT = mx.matmul(first, second.T)
            row_out = reducer.block(QT, i0, j0, j1)
            col_out = None if a == b else column(QT, i0, j0)
            if col_out is None:
                mx.eval(*row_out)
            else:
                mx.eval(*row_out, *col_out)
            mx.synchronize()

            row_left_i = np.array(row_out[2], dtype=np.int64)
            row_left_p = np.array(row_out[3], dtype=np.float32)
            row_right_i = np.array(row_out[4], dtype=np.int64)
            row_right_p = np.array(row_out[5], dtype=np.float32)
            row_left_i = np.where(row_left_i >= 0, row_left_i + j0, -1)
            row_right_i = np.where(row_right_i >= 0, row_right_i + j0, -1)
            _merge_side(left_p, left_i, i0, row_left_p, row_left_i, left=True)
            _merge_side(right_p, right_i, i0, row_right_p, row_right_i, left=False)

            if col_out is not None:
                col_i = np.array(col_out[0], dtype=np.int64)
                col_p = np.array(col_out[1], dtype=np.float32)
                _merge_side(left_p, left_i, j0, col_p, col_i, left=True)
            del QT, row_out, col_out, second
        del first

    rows = np.arange(length)
    left_key = 2 * (rows - left_i)
    right_key = 2 * (right_i - rows) + 1
    choose_left = (left_p < right_p) | (
        (left_p == right_p) & (left_key < right_key)
    )
    profile_p = np.where(choose_left, left_p, right_p)
    profile_i = np.where(choose_left, left_i, right_i)
    profile_i[~np.isfinite(profile_p)] = -1
    return profile_i.reshape(length, 1), left_i, right_i, profile_p, left_p, right_p


def triangular_profile(
    query,
    engine,
    *,
    self_join: bool,
    normalize: bool,
    k: int,
    chunk_size: int | None,
    excl: int,
    uncertain_out: np.ndarray | None = None,
    cap: int,
):
    """Adapter for replacing the production sweep within this benchmark only."""
    if (
        not self_join
        or query is not engine.target
        or engine.tiled
        or k != 1
        or chunk_size is not None
        or not _engine._fused_reducer(1)
    ):
        raise ValueError("triangle prototype requires a dense implicit k=1 Metal self-join")
    tiles = balanced_tiles(query.l, cap)
    if len(tiles) < 2:
        raise ValueError("triangle prototype requires at least two balanced tiles")
    I, IL, IR, P2, PL2, PR2 = compute_symmetric_k1(
        query, engine, normalize=normalize, excl=excl, tiles=tiles
    )
    if uncertain_out is not None:
        rows = np.arange(query.l)
        uncertain_out[:] = (
            _stump._ambiguous_scores(query, engine.target, rows, I[:, 0], P2, normalize)
            | _stump._ambiguous_scores(query, engine.target, rows, IL, PL2, normalize)
            | _stump._ambiguous_scores(query, engine.target, rows, IR, PR2, normalize)
        )
    return I, IL, IR


def _example_series(n: int, m: int, dataset: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    series = rng.standard_normal(n).cumsum()
    if dataset in ("repeat", "near"):
        if n < 4 * m + 3:
            raise ValueError("repeat/near dataset requires n >= 4*m + 3")
        motif = rng.standard_normal(m)
        starts = (0, n // 2, n - m)
        for start in starts:
            series[start : start + m] = motif
        if dataset == "near":
            series[starts[1] : starts[1] + m] += 1e-5 * rng.standard_normal(m)
    return series


def _snapshot(result) -> dict[str, np.ndarray]:
    return {
        name: np.array(getattr(result, name), copy=True)
        for name in ("P_", "I_", "left_I_", "right_I_")
    }


def _memory_api():
    if hasattr(mx, "reset_peak_memory"):
        return mx
    metal = getattr(mx, "metal", None)
    return metal if metal is not None and hasattr(metal, "reset_peak_memory") else None


def main() -> None:
    from bench_stump import provenance

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=16384)
    ap.add_argument("--m", type=int, default=200)
    ap.add_argument("--tile-caps", type=int, nargs="+", default=[4096, 8192])
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--seed", type=int, default=145)
    ap.add_argument("--dataset", choices=("walk", "repeat", "near"), default="walk")
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--strict", action="store_true", help="exit with error on output mismatch")
    args = ap.parse_args()
    if not (args.n >= args.m >= 3 and args.repeat >= 1):
        ap.error("require n >= m >= 3 and repeat >= 1")
    if any(cap < 16 or cap > 8192 for cap in args.tile_caps):
        ap.error("tile caps must be between 16 and 8192")
    if len(set(args.tile_caps)) != len(args.tile_caps):
        ap.error("tile caps must be distinct")
    length = args.n - args.m + 1
    if length * args.m * 4 > _engine._MATMUL_WINDOW_BYTES:
        ap.error("the target is tiled; this prototype requires dense storage")
    if any(len(balanced_tiles(length, cap)) < 2 for cap in args.tile_caps):
        ap.error("each tile cap must split the windows into at least two tiles")
    try:
        series = _example_series(args.n, args.m, args.dataset, args.seed)
    except ValueError as exc:
        ap.error(str(exc))
    if not _engine._fused_reducer(1):
        ap.error("the k=1 Metal reducer is unavailable on this device")

    original = _stump._compute_profile
    variants = ["production", *(f"triangle-{cap}" for cap in args.tile_caps)]
    timings = {name: [] for name in variants}
    peaks = {name: [] for name in variants}
    snapshots = {}
    uncertain_flags = {}
    memory = _memory_api()

    def run(name: str):
        if name == "production":
            selected = original
        else:
            cap = int(name.split("-", 1)[1])

            def triangle(*pos, **kw):
                return triangular_profile(*pos, **kw, cap=cap)

            selected = triangle

        def capture(*pos, **kw):
            result = selected(*pos, **kw)
            if kw.get("uncertain_out") is not None:
                uncertain_flags[name] = kw["uncertain_out"].copy()
            return result

        _stump._compute_profile = capture
        return mlx_stump.stump(series, args.m, normalize=not args.raw)

    try:
        # Compile each variant before timing. Releasing all results also makes
        # the peak measurement start from the same allocator state.
        for name in variants:
            warm = run(name)
            del warm
        gc.collect()
        mx.clear_cache()
        for r in range(args.repeat):
            order = variants if r % 2 == 0 else list(reversed(variants))
            for name in order:
                if memory is not None:
                    memory.reset_peak_memory()
                start = time.perf_counter()
                result = run(name)
                elapsed = time.perf_counter() - start
                timings[name].append(elapsed)
                if memory is not None:
                    peaks[name].append(memory.get_peak_memory())
                snapshots[name] = _snapshot(result)
                del result
                gc.collect()
                mx.clear_cache()
    finally:
        _stump._compute_profile = original

    print(provenance())
    print(
        f"n={args.n} m={args.m} windows={length} dataset={args.dataset} "
        f"normalize={not args.raw} repeats={args.repeat}"
    )
    print("Research prototype only: changed GEMM accumulation can alter near-tie ranks.")
    reference = snapshots["production"]
    base_time = float(np.median(timings["production"]))
    mismatch = False
    for name in variants:
        median = float(np.median(timings[name]))
        peak = float(np.median(peaks[name])) / 2**20 if peaks[name] else float("nan")
        print(
            f"{name}: median={median:.4f}s speedup={base_time / median:.2f}x "
            f"peak={peak:.1f}MiB"
        )
        if name == "production":
            continue
        uncertain_diff = int(
            np.count_nonzero(uncertain_flags[name] != uncertain_flags["production"])
        )
        mismatch |= uncertain_diff > 0
        print(f"  sweep uncertainty flags: exact={uncertain_diff == 0} mismatches={uncertain_diff}")
        for field, expected in reference.items():
            actual = snapshots[name][field]
            different = ~np.equal(actual, expected)
            count = int(np.count_nonzero(different))
            mismatch |= count > 0
            first = tuple(int(x) for x in np.argwhere(different)[0]) if count else None
            detail = (
                ""
                if first is None
                else f" first={first}: {expected[first]!r} -> {actual[first]!r}"
            )
            print(f"  {field}: exact={count == 0} mismatches={count}{detail}")
    if mismatch and args.strict:
        raise SystemExit("triangular prototype changed exact public output")


if __name__ == "__main__":
    main()
