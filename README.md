# mlx-stump

**Matrix profile on Apple Silicon GPUs, with a STUMPY-compatible API.**

`mlx-stump` computes the [matrix profile](https://www.cs.ucr.edu/~eamonn/MatrixProfile.html)
— the foundation for motif discovery, discord (anomaly) detection, and semantic
segmentation of time series — on the Metal GPU of any Apple Silicon Mac, via
[MLX](https://github.com/ml-explore/mlx).

STUMPY's GPU path (`gpu_stump`) requires an NVIDIA GPU, so on a Mac it simply
does not exist: Mac users are limited to the multi-core CPU path. `mlx-stump`
is the missing Mac GPU backend. It is a drop-in for `stumpy.stump`: same
signature, same output layout (columns `P`, `I`, `I_left`, `I_right`), so
STUMPY's downstream functions (`fluss`, `atsc`, `motifs`, …) consume its output
unchanged — and because the computation happens in Apple Silicon's unified
memory, there is no discrete-GPU transfer anywhere in the pipeline; the
resulting NumPy profile feeds straight into MLX models for downstream anomaly
classification on the same silicon.

> STUMPY is a trademark of TD Ameritrade IP Company, Inc. `mlx-stump` is an
> independent project that implements a STUMPY-compatible API; it is not
> affiliated with or endorsed by the STUMPY project or TD Ameritrade.

## Status

**v0.1 development — the batched-MASS engine is implemented and golden-tested
against STUMPY** (123 golden and regression tests). Distance profiles are
computed in bulk on the GPU as dense matmuls against the doubly-centered
subsequence matrix — materialized in one piece for moderate `n*m`, streamed
as column blocks beyond that — with a fused `mx.compile` distance+argmin
step per chunk. It is already 1.2–2.6x faster
than STUMPY using all CPU cores on the same machine, with a max profile error
around 1e-5 (see [Benchmarks](#benchmarks)). The SCAMP-style diagonal Metal
kernel — the headline speed path, O(1) work per cell instead of O(m) — is the
next milestone. See [Roadmap](#roadmap).

## Install

Not on PyPI yet (that lands with the v0.1 release — `pip install mlx-stump`
once it does). Until then, install from a checkout:

```bash
pip install .
```

Requires macOS on Apple Silicon and Python ≥ 3.10. `stumpy` (≥ 1.13) is an
optional extra used only for golden tests and CPU benchmarking (`pip install
".[dev]"`).

## Quickstart

```python
import numpy as np
import mlx_stump

T = np.random.randn(100_000).cumsum()   # a random walk
m = 200                                  # subsequence window

mp = mlx_stump.stump(T, m)               # same layout as stumpy.stump
P, I = mp[:, 0], mp[:, 1]                # profile values and neighbor indices
discord = np.argmax(P)                   # most anomalous subsequence
motif = np.argmin(P)                     # best-matching subsequence pair
```

The result also exposes STUMPY-style attributes: `mp.P_`, `mp.I_`,
`mp.left_I_`, `mp.right_I_`.

### API

| Function | STUMPY equivalent | Notes |
|---|---|---|
| `stump(T_A, m, T_B=None, ignore_trivial=True, normalize=True, p=2.0, k=1)` | `stumpy.stump` / `stumpy.aamp` | self-joins and AB-joins; `normalize=False` supports `p=2.0` only |
| `mass(Q, T, ...)` | `stumpy.mass` | z-normalized distance profile of one query |
| `match(Q, T, max_distance=..., max_matches=...)` | `stumpy.match` | all matches of a query, nearest first |

`stump` output feeds `stumpy.fluss`, `stumpy.motifs`, etc. directly.

## Precision

STUMPY computes in float64; Apple GPUs are float32-only. `mlx-stump` keeps the
error negligible in practice by:

1. **Global standardization** before upload — exactly invariant for the
   z-normalized profile, and removes large-offset cancellation.
2. **Float64 rolling statistics** — per-window mean and inverse standard deviation are
   computed on the CPU in float64 (O(n), trivial cost), then uploaded as float32.
3. **Doubly-centered covariance** — every window on *both* sides has its
   float64 mean subtracted *before* the float32 cast, so the sliding dot
   product IS the centered cross-covariance and the catastrophic
   `QT - m·μ_Q·μ_T` subtraction never happens in float32 — at any series
   length (the streamed column blocks of the large-`n` path are centered
   identically). The non-normalized (`aamp`) profile uses the same
   centering — `d² = ‖q_c − t_c‖² + m·(μ_q − μ_t)²` — so mixed-scale data
   doesn't hit the `ssq_q + ssq_t − 2·QT` cancellation either. Each profile
   also comes from a fresh product rather than a long floating-point
   recurrence, so error does not accumulate along the series.
4. **Float64 refinement** — profile values at the chosen indices are
   re-evaluated on the CPU in float64 (O(n·m), a few percent of runtime) as
   sums of squared differences of the z-normalized windows, with each
   window's mean and sigma recomputed exactly two-pass — a cancellation-free
   form that stays relatively accurate down to distance 0 — so reported `P`
   values are float64-exact for the reported neighbor.
5. **STUMPY-exact semantics** for constant subsequences, NaN/inf handling,
   exclusion zones, and left/right profiles, verified by a golden test suite
   that compares every code path against float64 STUMPY.

Precision metrics (max |ΔP|, index-agreement rate with tie tolerance) are
asserted in the test suite and published next to every benchmark number: an
fp32 matrix profile that disagrees with STUMPY on a discord is worthless no
matter how fast. What the float32 search costs in practice: a small fraction
of near-tied neighbors resolve to a different — equally close — index than
STUMPY's (0–0.05% of rows in the benchmarks below; every such disagreement
in the golden suite is verified in float64 to be a true near-tie). The effect
is largest for very small windows on smooth series: at `m=3` the worst
measured discrepancy between tied neighbors is ≈ 0.04, while at `m ≥ 8` the
golden-suite runs (n ≤ 3000) show none and the large-`n` benchmarks show
only the ~1e-5-scale near-ties reported in the table.

## Benchmarks

Honesty rules (see `bench/`):

- STUMPY is always benchmarked on the **same Mac with all cores** (never
  against published Xeon tables).
- Timings are end-to-end, including validation, float64 preprocessing,
  host↔GPU transfer, and float64 profile refinement.
- Precision metrics are published next to the speed numbers.

Self-join on a random walk, `m=200`, best of 2 (M-series Mac, mlx 0.32.2,
STUMPY 1.14.1 with numba parallel on all cores):

| n | mlx-stump (s) | stumpy all-cores (s) | speedup | max \|ΔP\| | idx agree | top-10 discords |
|---|---|---|---|---|---|---|
| 16,384 | 0.039 | 0.102 | 2.6x | 1.5e-05 | 99.99% | 10/10 |
| 65,536 | 0.548 | 0.759 | 1.4x | 2.5e-05 | 99.99% | 10/10 |
| 131,072 | 2.142 | 2.567 | 1.2x | 3.6e-05 | 99.99% | 10/10 |
| 262,144 | 9.080 | 11.698 | 1.3x | 2.9e-05 | 99.99% | 10/10 |

The MASS engine does O(m) work per distance-matrix cell where STUMPY's
recurrence does O(1), so the current speedup is structural, not a tuning
gap — the planned diagonal Metal kernel removes that factor. At `m=50` the
same benchmark shows 1.7x with 100% index agreement and max |ΔP| of 1.5e-08.

Run them yourself:

```bash
python bench/bench_stump.py --sizes 16384 65536 262144 --m 200
```

## Roadmap

- **v0.1** (this cycle): exact batched-MASS engine (`stump`, `mass`, `match`,
  AB-joins, `k>1`, `normalize=False`), golden harness, benchmark harness.
- **next**: SCAMP-style diagonal Metal kernel (one thread per diagonal,
  tiled, register-resident running covariances) for the large-`n` regime;
  `stump_batch` for many-short-series workloads (no STUMPY equivalent);
  chip-specific tuning (M1–M4).
- **backlog**: pan matrix profile (`stimp`), `mstump`, batched `stumpi`,
  snippets, MPdist.

## Known limitations

- `normalize=False` (non-normalized, `aamp`-style) supports `p=2.0` only.
- Custom `T_subseq_isconstant` must be a boolean array; callables are not
  supported yet.
- `mass`/`match` accept 1-D series only (STUMPY's multi-dimensional `Q`/`T`
  averaging is not implemented).
- In the rare corner of a self-join with an explicit `T_B` and *asymmetric*
  custom constant flags, distances follow consistent row-wise semantics;
  STUMPY's diagonal mirroring can report the unflagged distance for rows
  below a flagged target window.
- One deliberate accuracy divergence: for near-constant windows (rolling σ
  below ~1e-7 of the data scale — e.g. a flatlined sensor with tiny jitter),
  STUMPY's denominator clamp turns flat-vs-flat pairs into spurious
  zero-distance matches; mlx-stump computes those windows' float64 rolling σ
  exactly and its doubly-centered covariance resolves their true distances.
- A raw `mass` profile is the float32 GPU result: near-perfect matches read
  ~1e-3 rather than ~1e-8. `stump` and `match` re-evaluate their reported
  distances in float64, so their outputs don't carry this floor.
- Live per-batch GPU intermediates are kept under a ~384 MiB budget (an
  enforced ceiling, including the top-k buffers for `k > 1`); when the
  subsequence matrix would exceed ~512 MiB it is streamed as ~256 MiB
  column blocks with identical numerics. Pass `chunk_size` to trade memory
  for larger batches.
- Out-of-range `query_idx` values (including negative ones) raise
  `ValueError`. STUMPY silently wraps `query_idx <= -m` through numpy
  negative indexing and fabricates a zero-distance match at a negative
  index; rejecting is a deliberate, stricter divergence.
- A callable `max_distance` passed to `match` is invoked twice (once on the
  float32 profile, once on the float64-refined one); STUMPY calls it once on
  its exact profile.
- A truly constant window that a user flag array explicitly marks
  non-constant is ranked and reported at `sqrt(2m)` (its `1/σ` is taken as
  0, so `ρ = 0`); STUMPY's `σ = 0 → 1` clamp yields a slightly different
  convention for the same undefined quantity.
- With `normalize=False`, a `T_subseq_isfinite` override marking a
  NaN-containing window as finite computes its distance against the
  zero-filled series; STUMPY propagates NaN there.
- Like STUMPY's `gpu_stump`, streaming (`stumpi`) is out of scope for the GPU
  path for now.

## License

MIT
