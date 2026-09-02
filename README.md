# mlx-stump

**Matrix profile on Apple Silicon GPUs, with a STUMPY-compatible API.**

`mlx-stump` computes the [matrix profile](https://www.cs.ucr.edu/~eamonn/MatrixProfile.html)
— the foundation for motif discovery, discord (anomaly) detection, and semantic
segmentation of time series — on the Metal GPU of any Apple Silicon Mac, via
[MLX](https://github.com/ml-explore/mlx).

STUMPY's GPU path (`gpu_stump`) requires an NVIDIA GPU, so on a Mac it simply
does not exist: Mac users are limited to the multi-core CPU path. `mlx-stump`
is the missing Mac GPU backend. It is a drop-in for `stumpy.stump`: same
parameters, same output layout (profile value/index, then left/right profile
indices), and
the same `P_`/`I_` accessors used by STUMPY's downstream functions. Pass the
appropriate accessor—for example, `mp.I_` to `fluss` or `mp.P_` to `motifs`—
without conversion. Because the computation happens in Apple Silicon's
unified memory, there is no discrete-GPU transfer anywhere in the pipeline;
the resulting NumPy profile feeds straight into MLX models for downstream
anomaly classification on the same silicon.

> STUMPY is a trademark of TD Ameritrade IP Company, Inc. `mlx-stump` is an
> independent project that implements a STUMPY-compatible API; it is not
> affiliated with or endorsed by the STUMPY project or TD Ameritrade.

## Status

**v0.1 development — the batched-MASS engine is implemented and golden-tested
against STUMPY** (280 golden and regression tests). Distance profiles are
computed in bulk on the GPU as dense matmuls against a locally z-normalized
subsequence matrix (or a doubly-centered shared-frame matrix for raw
distances) — materialized in one piece for moderate `n*m`, streamed as column
blocks beyond that — with a fused `mx.compile` distance+argmin step per chunk.
It is already 1.1–2.1x faster
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

Releases are published by running the `Publish to PyPI` workflow by hand from
`main`, after bumping `__version__` in `src/mlx_stump/__init__.py`,
committing on `main`, and letting CI pass:

```bash
gh workflow run publish-pypi.yml --ref main -f version=0.1.0
```

The workflow refuses to run from any other branch, requires a successful CI
run for that exact `main` commit, and refuses a version that does not equal
`__version__` (in canonical PEP 440 form, never a `.dev`/local version). It
serializes releases, installs fully transitive
hash-locked build/test environments, builds the wheel and sdist twice from
independent archives of the commit, and requires byte-identical artifacts.
It installs and tests both the wheel and the sdist in separate clean
environments, creates (or verifies) an annotated `v<version>` tag at the exact
tested commit, and only then hands the artifacts to a minimal OIDC-only PyPI
job. A failed upload can therefore
be retried against the same verified tag; an immutable PyPI release cannot
be left without its source tag. If PyPI accepts only part of an upload, use
GitHub's **Re-run all jobs** action (not **Re-run failed jobs**): the full
rerun rebuilds the original event commit, rechecks PyPI's exact filenames
and hashes, and sends only the missing distributions.

The publisher deliberately uses the never-before-used identity
`publish-pypi.yml` / `pypi-release-v2`. PyPI must trust exactly that pair and
must **not** retain the historical `release.yml` / `pypi` identity: old
commits contain tag-triggered copies of `release.yml`, and GitHub evaluates a
workflow at the pushed tag's commit. The GitHub environment requires review,
allows deployments only from `main`, and holds `RELEASE_TAG_DEPLOY_KEY`. Its
public half is the repository's only deploy key. GitHub models ruleset bypass
for deploy keys as a repository-wide actor class, so adding any other deploy
key would also grant that key the `v*`-tag bypass and must be treated as a
release-security change. The ruleset protects creation, update, and deletion
of `v*` tags; the ordinary workflow token remains read-only.

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
| `stump(T_A, m, T_B=None, …, T_A_subseq_isconstant=None, T_B_subseq_isconstant=None, *, chunk_size=None)` | `stumpy.stump` / `stumpy.aamp` | self-joins and AB-joins; `normalize=False` supports `p=2.0` only; `chunk_size` is an mlx-stump extension |
| `mass(Q, T, ...)` | `stumpy.mass` | normalized or raw (`p=2`) distance profile of one query |
| `match(Q, T, max_distance=..., max_matches=...)` | `stumpy.match` | normalized or raw (`p=2`) matches of a query, nearest first |

The typed accessors feed downstream STUMPY functions directly: for example,
`stumpy.fluss(mp.I_, ...)` and `stumpy.motifs(T, mp.P_, ...)`. The full 2-D
`mparray` is not itself the one-dimensional input those functions expect.

## Precision

STUMPY computes in float64; Apple GPUs are float32-only. `mlx-stump` keeps the
error negligible in practice by:

1. **Scale-safe affine frames** — midpoint/max-deviation preconditioners keep
   uniformly tiny or huge finite units from underflowing or overflowing.
2. **Local float64 z-normalization** — every normalized query and target
   window is read from the raw series, placed in its own bounded frame,
   centered, and divided by its own RMS *before* the float32 cast. The GPU
   product is therefore `m·ρ` directly. A tiny exact window embedded beside
   values 10¹⁶ times larger is not re-rounded by one global standardized
   copy, and `QT - m·μ_Q·μ_T` never occurs.
3. **Doubly-centered raw covariance** — the non-normalized (`aamp`) profile
   uses one shared scale-safe frame and
   `d² = ‖q_c − t_c‖² + m·(μ_q − μ_t)²`, so mixed-scale data
   doesn't hit the `ssq_q + ssq_t − 2·QT` cancellation either, and the
   window means travel to the GPU as float32 (hi, lo) pairs so their
   difference is not limited to the ulp of the global offset. Each profile
   also comes from a fresh product rather than a long floating-point
   recurrence, so error does not accumulate along the series.
4. **Float64 refinement** — profile values at the chosen indices are
   re-evaluated on the CPU in float64 (O(n·m), a modest fraction of runtime) as
   sums of squared differences of the z-normalized windows, with each
   window's mean and sigma freshly recomputed two-pass — a cancellation-free
   form that stays relatively accurate down to distance 0 — so reported `P`
   values use well-conditioned float64 arithmetic for the reported neighbor.
   `match` refines every
   candidate near its threshold the same way; precomputed `M_T`/`Σ_T` never enter
   the arithmetic (see [Known limitations](#known-limitations)).
5. **STUMPY-compatible handling** of constant subsequences, NaN/inf values,
   exclusion zones, and left/right profiles, verified by a golden test suite
   that compares every code path against float64 STUMPY.

Precision metrics are asserted in the test suite and published next to every
benchmark number. Published `idx agree` is strict index equality; the golden
tests additionally verify that any differing index is a float64 near-tie. An
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

- STUMPY is always benchmarked on the **same Mac with all logical CPUs**
  (never against published Xeon tables). The script prints Numba's
  actual thread count and refuses a CPU baseline when it differs from the
  host's logical-CPU count.
- Timings are end-to-end, including validation, float64 preprocessing,
  host↔GPU transfer, and float64 profile refinement.
- Precision metrics are published next to the speed numbers.
- The provenance line marks the commit `+dirty` whenever tracked or untracked
  workspace changes mean the benchmark is not running that exact commit.

Self-join on a random walk, `m=200`, best of 2. Provenance (the script
prints this line above its table): Apple M4 Max, macOS 26.2, Python 3.12.12,
mlx 0.32.2, numpy 2.5.2, STUMPY 1.14.1 with numba parallel on all 16 cores,
mlx-stump 0.1.0.dev0 at commit `9c4dcae`, 2026-09-01. Timings on a laptop
vary by tens of percent with thermal state and background load; compare
runs from the same session.

| n | mlx-stump (s) | stumpy all-cores (s) | speedup | max \|ΔP\| | idx agree | top-10 discords |
|---|---|---|---|---|---|---|
| 16,384 | 0.061 | 0.107 | 1.8x | 1.5e-05 | 99.99% | 10/10 |
| 65,536 | 0.680 | 0.756 | 1.1x | 2.5e-05 | 99.99% | 10/10 |
| 131,072 | 2.499 | 2.788 | 1.1x | 3.6e-05 | 99.99% | 10/10 |
| 262,144 | 10.622 | 12.882 | 1.2x | 2.9e-05 | 99.99% | 10/10 |

The MASS engine does O(m) work per distance-matrix cell where STUMPY's
recurrence does O(1), so the current speedup is structural, not a tuning
gap — the planned diagonal Metal kernel removes that factor. At `m=50` the
same benchmark (same provenance) shows 1.4–2.1x with 100% index agreement
and max |ΔP| ≤ 6e-6.
Above a 256 MiB subsequence matrix (n ≈ 335k at `m=200`) the target is
streamed in 128 MiB column blocks through the same compiled kernel: at
n=524,288 that path measured as fast as or faster than the dense sweep
(47.7 s vs 47.9 s in one run, 38.8 s vs 52.8 s in another), at about half
the peak memory.

Run them yourself:

```bash
python bench/bench_stump.py --sizes 16384 65536 131072 262144 --m 200 --repeat 2 --seed 0
```

## Roadmap

- **v0.1** (this cycle): batched-MASS engine (`stump`, `mass`, `match`,
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
- `mass`/`match` operate on 1-D series. For compatibility, a single-column
  `(n, 1)` input is flattened (with STUMPY's warning for `mass(Q, ...)`), but
  STUMPY's genuinely multi-dimensional `Q`/`T` averaging is not implemented.
- In the rare corner of a self-join with an explicit `T_B` and *asymmetric*
  custom constant flags, distances follow consistent row-wise semantics;
  STUMPY's diagonal mirroring can report the unflagged distance for rows
  below a flagged target window.
- One deliberate accuracy divergence: for near-constant windows (rolling σ
  below ~1e-7 of the data scale — e.g. a flatlined sensor with tiny jitter),
  STUMPY's denominator clamp turns flat-vs-flat pairs into spurious
  zero-distance matches; mlx-stump instead centers and RMS-normalizes each raw
  window in its own bounded float64 frame before the GPU cast, resolving their
  true distances.
- The unrefined `mass` output is the float32 GPU result: near-perfect matches read
  ~1e-3 rather than ~1e-8. `stump` and `match` re-evaluate their reported
  distances in float64, so their outputs don't carry this floor.
- Memory is bounded by three fixed budgets rather than by `n·m`: the
  resident window block (the whole float32 subsequence matrix when it is
  ≤ 256 MiB, otherwise ~128 MiB column blocks, at least four windows wide,
  streamed one at a time — same compiled kernel, bit-identical numerics,
  measured as fast as or faster than the dense sweep), ~384 MiB of live
  per-batch GPU intermediates (an enforced ceiling under automatic chunking,
  including the top-k device buffers for `k > 1`: each batch is synchronized
  before the next one allocates, and the trailing batch is computed at full
  width so no second set of buffers ever exists), and bounded CPU
  temporaries (block centering and sigma repair are ≤ 64 MiB each; the
  refinement chunk is ≤ 256 MiB and runs after the window matrix and MLX's
  cached batch buffers have been released). Tiled top-k joins additionally
  need a batch-sized host merge workspace for concatenation, stable sorting,
  and gathering; it scales with `batch_rows·k` and is included in the peak
  estimate. Each block exists twice while it is uploaded (numpy staging
  plus the device copy), so the dominant modeled peak beyond the O(n)
  series arrays is estimated as `2·block + max(64 MiB, 8·m + 128 B)` during
  upload (the second term includes the documented one-window float64 floor)
  or `block + 384 MiB`
  during the sweep, plus numeric/object outputs —
  ~640 MiB for the largest dense block, ~512 MiB in tiled mode at `k=1`
  (`mlx_stump._engine.estimated_peak_bytes(l, m, k, self_join, l_q,
  chunk_size)` gives that phase estimate). The O(n) input series and window
  masks are additional; raw mode also retains its standardized series and
  rolling statistics. Those arrays are deliberately excluded from the helper.
  For large `k` the output itself dominates:
  STUMPY's
  object-dtype `mparray` layout costs a pointer plus a CPython allocator
  block per cell, ~80 resident bytes per neighbor per row (n=50,000, m=50,
  k=100: ~385 MiB for the returned array alone), which the estimate includes
  together with the top-k reordering temporaries. The canonical case now
  measures ~566 MiB against a ~638 MiB estimate. It is an estimate with
  headroom, not a literal cap: MLX's allocator rounds buffers up (about
  +0.5% observed), a
  gigantic `l` can make even a one-row batch exceed the intermediates
  budget, and the figures are MLX's own active-memory peak plus host
  memory (GPU-written buffers do not show up in RSS on macOS, only in the
  process footprint). `mass`/`match`
  evaluate one block at a time and never hold more, and every device array
  is dropped before the cache is cleared, so no per-series allocation stays
  cached after a call returns. MLX may retain a small runtime/allocator
  baseline (2.6 MiB on one hosted-runner image). For `stump`, pass
  `chunk_size` to trade memory for larger batches, and pass the same value to
  `estimated_peak_bytes` because an explicit batch is allowed to exceed the
  automatic 384 MiB budget. `mass` and `match` always use automatic block
  streaming.
- Out-of-range `query_idx` values (including negative ones) raise
  `ValueError`. STUMPY silently wraps `query_idx <= -m` through numpy
  negative indexing and fabricates a zero-distance match at a negative
  index; rejecting is a deliberate, stricter divergence. With
  `normalize=False`, `match` zeroes `D[query_idx]` the way
  `stumpy.mass_absolute` does (when the window is finite); `stumpy.aamp_match`
  leaves that entry at its true distance. A *mismatched* `query_idx` is
  therefore the first zero-distance match here but retains its true distance
  in STUMPY; with a threshold below that true distance, STUMPY omits it.
- A data-dependent `max_distance` in `match` (the default, or a callable)
  is evaluated on successively refined profiles until it stops moving —
  the float32 profile first, then once per float64 refinement round,
  typically 2–3 calls in total and at most 9 — so the threshold that
  selects the matches comes from a profile that is float64-refined
  throughout the threshold band. STUMPY calls it once, on its float64
  profile; a callable with side effects would observe the difference.
- In normalized `mass`/`match`, precomputed `M_T`/`Σ_T` are accepted as
  compatibility metadata, not used as a computational cache or as an input
  to the arithmetic: they are validated (shape `(l,)`), an infinite `M_T`
  marks its window non-finite (STUMPY's convention for windows containing
  NaN), and finite entries otherwise use the same raw-window local float64
  centering and RMS normalization as the no-stats call, so the two results
  are identical. Refined `match` output
  reports bitwise, shifted, and exactly representable positive-affine
  duplicates at exactly 0. A tiny numerical
  residual is zeroed only after exact dyadic-rational collinearity proves
  that the *stored float64 rows* satisfy `W = a·Q + b` with `a > 0`; the
  certificate does not claim that a mathematically affine transform performed
  with intervening rounding is still affine in its stored values. A
  non-affine row inside that roundoff band (even a one-ULP perturbation)
  remains non-zero; if ordinary float64 normalization collapses it all the
  way to zero, it is re-evaluated at high precision. `mass` in either mode
  remains float32 and retains the small floor described above.
  This is a deliberate choice of mathematical semantics over STUMPY's
  literal use of the supplied values, whose rounding STUMPY lets into the
  distance: its `QT − m·μ_Q·M_T` amplifies `M_T`'s rounding by `(μ/σ)²`
  (self-match errors up to ~0.1 at offset 1e6, a meaningless ranking at
  1e9), and its `1/(σ_Q·Σ_T)` leaves a `sqrt(2m·δ)` floor on perfect
  matches from `Σ_T`'s own relative rounding δ (~1e-3 at offsets 1e9–1e12,
  ~0.06 at 1e14, even with `compute_mean_std`'s own output). The
  consequences: a deliberately scaled or biased `M_T`/`Σ_T` changes
  STUMPY's result but not ours; a NaN `M_T` yields NaN in STUMPY and is
  ignored here (constant-window rules still apply, as in STUMPY); a
  non-finite or zero `Σ_T` entry yields `sqrt(2m)`/NaN in STUMPY and is
  ignored here. Passing `compute_mean_std`'s output reproduces STUMPY's
  ranking to within STUMPY's own rounding. With `normalize=False`, the pair
  is shape-validated and then entirely ignored: even an infinite `M_T` is
  not a raw-window finiteness marker, and positive-affine rows do not have
  zero Euclidean distance unless they are identical.
- `Q_subseq_isconstant` must be a boolean (Python/NumPy bool, or a boolean
  array of size 1): non-boolean values such as `"False"` or `0` raise
  `ValueError` rather than being coerced. A plain boolean is accepted for
  the scalar `Q` flag; boolean *lists* of the required length are accepted
  for the `T_*_subseq_isconstant` arrays, where STUMPY requires an
  `np.ndarray`. All constant-flag controls are shape/type
  validated when `normalize=False` and then ignored because raw Euclidean
  distance has no constant-window special case. STUMPY instead routes raw
  calls to separate functions that reject these normalized-only keywords.
- A window whose sigma is 0 without being flagged constant — a truly
  constant window that a user flag array marks non-constant — is ranked
  and reported at `sqrt(2m)` (its `1/σ` is taken as 0, so `ρ = 0`).
  STUMPY's denominator clamp yields a different convention (0 or a huge
  value) for the same undefined quantity.
- With `normalize=False`, a `T_subseq_isfinite` override marking a
  NaN-containing window as finite computes its distance against the
  zero-filled series; STUMPY propagates NaN there.
- `match(..., normalize=False)` interprets `atol` in the raw distance units,
  as STUMPY does. When rescaling a series and expecting the same match set,
  rescale an explicitly supplied `atol` too (or use `atol=0`); the default
  `max_distance` calculation itself is scale-safe.
- The refined non-normalized `stump` profile preserves Euclidean distances in
  the input's raw units at every representable scale. STUMPY's `aamp` applies
  its fixed `1e-14` squared P-norm threshold in raw units, so it snaps every
  distance below `1e-7` to zero (for example, an ordinary nonzero profile
  uniformly scaled by `2**-700`). mlx-stump deliberately does not apply that
  unit-dependent snap; exact duplicate windows still evaluate to exactly 0.
- `normalize=False` searches in float32, and its neighbor choice is not
  identical to STUMPY's `aamp` on mixed-scale data: with a 1e5 constant
  segment in unit noise (`m=7`) index agreement is ~83%, with a 1e6 offset
  segment in a random walk (`m=50`) ~99%. Most of those disagreements are
  exact ties (identical constant windows, where the chosen index is
  arbitrary on both sides); every one is a float32 near-tie *relative to
  the distance itself* — the two candidates' true distances differ by
  ~1e-7 of their magnitude (≤ 5e-7 in those two cases, which the test
  suite asserts: 3.6e-5 raw units on a ~1.7e5 distance, 0.08 on a ~4e6
  one; up to ~5e-6 on long series with large `m`). The reported `P` is
  directly recomputed in float64 for the chosen neighbor, and `match` widens its
  re-evaluation cutoff per-window so true matches are not dropped. From
  ~1e6 of dynamic range on, STUMPY's own CPU `aamp` — a float64 diagonal
  recurrence over terms of the segment's squared magnitude — drifts (its
  `P` is off by ~1e-2 at 1e6, ~0.5 at 1e7, ~5 at 1e8 on unit-scale rows,
  and it can pick neighbors far from the true nearest), so agreement with
  it stops being a precision metric there; measured against direct float64
  evaluation, mlx-stump's neighbor gaps stay ≤ ~1e-7 relative and its `P`
  agrees with that evaluation up to the ~1e13 standardization limit.
- On smooth, highly self-similar series (a clean periodic signal with
  small noise) most rows have many near-tied period-repeat candidates, and
  the float32 search often resolves them differently from STUMPY (≈60% of
  rows on a unit-amplitude sine at `m=100`, in both normalize modes). The
  gaps are within the golden-suite tie tolerance in absolute terms (≤ 3e-3
  on nearest distances of ~1e-2) but can be tens of percent of those small
  distances; `P` is directly recomputed in float64 for the chosen index.
- Normalized search has no global-standardization dynamic-range limit: each
  raw window is centered and scaled independently before upload. Raw-distance
  search necessarily uses one shared affine frame for cross-window units; a
  series with ≳1e13 between its largest variation and its smallest window
  variation warns that those smallest raw distances may be unreliable. This
  is not an absolute-units limit: uniformly rescaling an ordinary finite
  series from subnormal-scale values through ~1e300 remains scale-safe.
- Like STUMPY's `gpu_stump`, streaming (`stumpi`) is out of scope for the GPU
  path for now.

## License

MIT
