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
against STUMPY** (207 golden and regression tests). Distance profiles are
computed in bulk on the GPU as dense matmuls against the doubly-centered
subsequence matrix — materialized in one piece for moderate `n*m`, streamed
as column blocks beyond that — with a fused `mx.compile` distance+argmin
step per chunk. It is already 1.3–2.0x faster
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

### Releasing (maintainers)

Pushing a `v<version>` tag builds the wheel and sdist, tests the *built
wheel* in a clean environment, and publishes to PyPI via trusted publishing.
The workflow refuses a tag whose commit is not on `main`, one that does not
exactly match `__version__` in `src/mlx_stump/__init__.py` (in canonical
PEP 440 form), and `.dev`/local versions — so bump the version (e.g. to
`0.1.0`) on `main`, let CI pass, then tag that commit `v0.1.0`. Two
repository settings should back the workflow up and are not yet
configured: a tag ruleset restricting who may create `v*` tags, and
required reviewers on the `pypi` environment that the PyPI trusted
publisher is scoped to.

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
   doesn't hit the `ssq_q + ssq_t − 2·QT` cancellation either, and the
   window means travel to the GPU as float32 (hi, lo) pairs so their
   difference is not limited to the ulp of the global offset. Each profile
   also comes from a fresh product rather than a long floating-point
   recurrence, so error does not accumulate along the series.
4. **Float64 refinement** — profile values at the chosen indices are
   re-evaluated on the CPU in float64 (O(n·m), a few percent of runtime) as
   sums of squared differences of the z-normalized windows, with each
   window's mean and sigma recomputed exactly two-pass — a cancellation-free
   form that stays relatively accurate down to distance 0 — so reported `P`
   values are float64-exact for the reported neighbor. `match` refines every
   candidate near its threshold the same way (with a user-supplied `Σ_T`,
   STUMPY's `2m(1 − ρ)` is formed from that exact centered covariance and
   the user's scale, never from `QT − m·μ_Q·M_T`).
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

Self-join on a random walk, `m=200`, best of 2. Provenance (the script
prints this line above its table): Apple M4 Max, macOS 26.2, Python 3.12.12,
mlx 0.32.2, numpy 2.5.2, STUMPY 1.14.1 with numba parallel on all 16 cores,
mlx-stump 0.1.0.dev0 at commit `4755d34`, 2026-09-01. Timings on a laptop
vary by tens of percent with thermal state and background load; compare
runs from the same session.

| n | mlx-stump (s) | stumpy all-cores (s) | speedup | max \|ΔP\| | idx agree | top-10 discords |
|---|---|---|---|---|---|---|
| 16,384 | 0.051 | 0.104 | 2.0x | 1.5e-05 | 99.99% | 10/10 |
| 65,536 | 0.573 | 0.718 | 1.3x | 2.5e-05 | 99.99% | 10/10 |
| 131,072 | 2.145 | 2.695 | 1.3x | 3.6e-05 | 99.99% | 10/10 |
| 262,144 | 9.328 | 12.393 | 1.3x | 2.9e-05 | 99.99% | 10/10 |

The MASS engine does O(m) work per distance-matrix cell where STUMPY's
recurrence does O(1), so the current speedup is structural, not a tuning
gap — the planned diagonal Metal kernel removes that factor. At `m=50` the
same benchmark (same provenance) shows 1.5–2.5x with 100% index agreement
and max |ΔP| ≤ 6e-6.
Above a 256 MiB subsequence matrix (n ≈ 335k at `m=200`) the target is
streamed in 128 MiB column blocks through the same compiled kernel: at
n=524,288 that path measured as fast as or faster than the dense sweep
(47.7 s vs 47.9 s in one run, 38.8 s vs 52.8 s in another), at about half
the peak memory.

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
- Memory is bounded by three fixed budgets rather than by `n·m`: the
  resident window block (the whole float32 subsequence matrix when it is
  ≤ 256 MiB, otherwise ~128 MiB column blocks, at least four windows wide,
  streamed one at a time — same compiled kernel, bit-identical numerics,
  measured as fast as or faster than the dense sweep), ~384 MiB of live
  per-batch GPU intermediates (an enforced ceiling, including the top-k
  buffers for `k > 1`: each batch is synchronized before the next one
  allocates, and the trailing batch is computed at full width so no second
  set of buffers ever exists), and ≤ 64 MiB float64 CPU temporaries per
  stage (block centering, sigma repair; the refinement chunk is ≤ 256 MiB
  and runs after the window matrix and MLX's cached batch buffers have been
  released). Each block exists twice while it is uploaded (numpy staging
  plus the device copy), so the process-wide peak for one call is
  estimated as `2·block + 64 MiB` during upload or `block + 384 MiB`
  during the sweep, plus the O(l·k) outputs and the O(n) per-series arrays
  — ~640 MiB for the largest dense block, ~512 MiB in tiled mode
  (`mlx_stump._engine.estimated_peak_bytes(l, m, k)` gives the number).
  That is an estimate with headroom, not a literal cap: MLX's allocator
  rounds buffers up (about +0.5% observed), a gigantic `l` can make even a
  one-row batch exceed the intermediates budget, and the figures are MLX's
  own active-memory peak plus host memory (GPU-written buffers do not show
  up in RSS on macOS, only in the process footprint). `mass`/`match`
  evaluate one block at a time and never hold more, and every device array
  is dropped before the cache is cleared, so nothing stays cached in MLX
  after a call returns. Pass `chunk_size` to trade memory for larger
  batches.
- Out-of-range `query_idx` values (including negative ones) raise
  `ValueError`. STUMPY silently wraps `query_idx <= -m` through numpy
  negative indexing and fabricates a zero-distance match at a negative
  index; rejecting is a deliberate, stricter divergence. With
  `normalize=False`, `match` zeroes `D[query_idx]` the way
  `stumpy.mass_absolute` does (when the window is finite); `stumpy.aamp_match`
  leaves that entry at its true distance, so a *mismatched* `query_idx`
  yields a zero-distance first match here and an empty result there.
- A data-dependent `max_distance` in `match` (the default, or a callable)
  is evaluated on successively refined profiles until it stops moving —
  the float32 profile first, then once per float64 refinement round,
  typically 2–3 calls in total and at most 9 — so the threshold that
  selects the matches comes from a profile that is float64-exact
  throughout the threshold band. STUMPY calls it once, on its exact
  profile; a callable with side effects would observe the difference.
- Precomputed `M_T`/`Σ_T` for `mass`/`match`: `Σ_T` is honored as the
  per-window scale of the distance (STUMPY's ranking is reproduced from its
  own `compute_mean_std` output), but `M_T` does not enter the covariance — every window
  is centered by its own exact float64 mean, because STUMPY's
  `QT − m·μ_Q·M_T` form amplifies `M_T`'s rounding by `(μ/σ)²` and collapses
  on offset data (self-match errors up to ~0.1 at offset 1e6, seed-dependent,
  and a meaningless ranking at 1e9). A deliberately biased `M_T` therefore
  changes nothing here, whereas it changes STUMPY's result. With a
  user-supplied `Σ_T`, a near-perfect match is reported through STUMPY's
  `2m(1 − ρ)`, so `Σ_T`'s own relative rounding δ surfaces as a floor of
  `sqrt(2m·δ)`: ≈ 1e-7 at unit scale, but ~8e-4 at offset 1e12 and ~0.06 at
  offset 1e14 even with `compute_mean_std`'s own output. Windows
  bitwise-identical to the query are reported as exactly 0 regardless;
  for jittered near-duplicates under thresholds tighter than that floor,
  omit `M_T`/`Σ_T` (the exact-stats path has no such floor). A window
  whose `M_T` is not finite is reported as inf ahead of the constant-window
  rules, as in STUMPY; a non-finite `Σ_T` entry acts as a zero sigma
  (`sqrt(2m)`, or the constant-window value when the window is flagged
  constant — STUMPY's behavior for `inf`; it yields NaN for a NaN `Σ_T`).
- `Q_subseq_isconstant` must be a boolean (Python/NumPy bool, or a boolean
  array of size 1): non-boolean values such as `"False"` or `0` raise
  `ValueError` rather than being coerced. Plain booleans and boolean *lists*
  (for the `T_*_subseq_isconstant` arrays) are a leniency — STUMPY accepts
  only an `np.ndarray` there.
- A window whose sigma is 0 without being flagged constant — a truly
  constant window that a user flag array marks non-constant, or a
  user-supplied `Σ_T` entry that is 0 (or negative) on a non-constant
  window — is ranked and reported at `sqrt(2m)` (its `1/σ` is taken as 0,
  so `ρ = 0`). STUMPY's denominator clamp yields a different convention
  (0 or a huge value) for the same undefined quantity.
- With `normalize=False`, a `T_subseq_isfinite` override marking a
  NaN-containing window as finite computes its distance against the
  zero-filled series; STUMPY propagates NaN there.
- `normalize=False` searches in float32, and its neighbor choice is not
  identical to STUMPY's `aamp` on mixed-scale data: with a 1e5 constant
  segment in unit noise (`m=7`) index agreement is ~83%, with a 1e6 offset
  segment in a random walk (`m=50`) ~99%. Most of those disagreements are
  exact ties (identical constant windows, where the chosen index is
  arbitrary on both sides); every one is a float32 near-tie *relative to
  the distance itself* — the two candidates' true distances differ by
  ~1e-7 of their magnitude (≤ 5e-7 in those two cases, which the test
  suite asserts: 3.6e-5 raw units on a ~1.7e5 distance, 0.08 on a ~4e6
  one; up to ~5e-6 on long series with large `m`). The reported `P` stays
  float64-exact for the chosen neighbor, and `match` widens its
  re-evaluation cutoff per-window so true matches are not dropped. From
  ~1e6 of dynamic range on, STUMPY's own CPU `aamp` — a float64 diagonal
  recurrence over terms of the segment's squared magnitude — drifts (its
  `P` is off by ~1e-2 at 1e6, ~0.5 at 1e7, ~5 at 1e8 on unit-scale rows,
  and it can pick neighbors far from the true nearest), so agreement with
  it stops being a precision metric there; measured against exact float64
  truth, mlx-stump's neighbor gaps stay ≤ ~1e-7 relative and its `P`
  float64-exact up to the ~1e13 standardization limit.
- On smooth, highly self-similar series (a clean periodic signal with
  small noise) most rows have many near-tied period-repeat candidates, and
  the float32 search often resolves them differently from STUMPY (≈60% of
  rows on a unit-amplitude sine at `m=100`, in both normalize modes). The
  gaps are within the golden-suite tie tolerance in absolute terms (≤ 3e-3
  on nearest distances of ~1e-2) but can be tens of percent of those small
  distances; `P` is exact for the chosen index.
- A series whose amplitude dynamic range approaches float64's ~16 digits
  (≳1e13 between its largest values and its smallest window variation)
  triggers a warning: global standardization cannot faithfully represent
  the smallest-variance windows at all.
- Like STUMPY's `gpu_stump`, streaming (`stumpi`) is out of scope for the GPU
  path for now.

## License

MIT
