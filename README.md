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
indices), and the same `P_`/`I_` accessors used by STUMPY's downstream
functions. Pass the
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
against STUMPY** (893 golden and regression tests). Distance profiles are
computed in bulk on the GPU as dense matmuls against a locally z-normalized
subsequence matrix (or a doubly-centered shared-frame matrix for raw
distances) — materialized in one piece for moderate `n*m`, streamed as column
blocks beyond that — with the distance evaluation and the per-row
argmin/top-k selection fused into one-pass Metal kernels
(`mx.fast.metal_kernel`, `k ≤ 16`); a bit-identical `mx.compile` step serves
the CPU device and `k > 16`. It is 2.9–3.0× as fast as STUMPY using all 16
CPU cores on the same machine at `m=200` and 3.5–5.9× at `m=50`, with max
profile error ≤ 4.4e-5 at `m=200` and ≤ 4.9e-6 at `m=50` in the current
benchmarks (see [Benchmarks](#benchmarks)). The engine still does O(m) work
per distance-matrix cell; a SCAMP-style diagonal Metal kernel (O(1) per
cell) measured only 1.2–2.2× faster than the fused engine in a prototype, so
it remains future work. See [Roadmap](#roadmap).

CI runs the suite on macOS 14/15 with Python 3.10/3.12/3.14 plus one job at
the exact declared dependency floors (numpy 1.24.0, mlx 0.30.0 and STUMPY
1.13.0, on the newest Python 3.10.x), and fails any test run whose MLX
default device is not the Metal GPU (`MLX_STUMP_REQUIRE_METAL=1`).

## Install

Not on PyPI yet (that lands with the v0.1 release — `pip install mlx-stump`
once it does). Until then, install from a checkout:

```bash
pip install .
```

Requires macOS on Apple Silicon and Python ≥ 3.10. `stumpy` (≥ 1.13) is an
optional extra used only by the golden/regression test suite and CPU
benchmarking (`pip install ".[dev]"`); without it, the test modules that
need the oracle skip themselves and the rest of the suite still runs.

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
serializes releases, installs fully transitive hash-locked build/test
environments, builds the wheel and sdist twice from
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
`mp.left_I_`, `mp.right_I_`. Results keep these attributes through `pickle`,
`np.save(..., allow_pickle=True)` and process pools (`multiprocessing`,
`concurrent.futures`), so profiles can be computed in worker processes.
STUMPY's own `mparray` loses them on a round trip.

### API

| Function | STUMPY equivalent | Notes |
|---|---|---|
| `stump(T_A, m, T_B=None, ignore_trivial=True, normalize=True, p=2.0, k=1, T_A_subseq_isconstant=None, T_B_subseq_isconstant=None, *, chunk_size=None)` | `stumpy.stump` | self-joins and AB-joins; `normalize=False` computes the non-normalized profile (`p=2.0` only); constant flags may be boolean arrays or STUMPY-style callables; `chunk_size` is an mlx-stump extension |
| `aamp(T_A, m, T_B=None, ignore_trivial=True, p=2.0, k=1, *, chunk_size=None)` | `stumpy.aamp` | `stump(..., normalize=False)` under aamp's positional signature, where the fifth argument is `p`; `p=2.0` only |
| `gpu_stump(T_A, m, T_B=None, ignore_trivial=True, device_id=0, normalize=True, p=2.0, k=1, T_A_subseq_isconstant=None, T_B_subseq_isconstant=None, *, chunk_size=None)` | `stumpy.gpu_stump` | same computation as `stump`; `device_id` (an int or a list of ints) is validated and ignored, because the Mac's one GPU is always used |
| `gpu_aamp(T_A, m, T_B=None, ignore_trivial=True, device_id=0, p=2.0, k=1, *, chunk_size=None)` | `stumpy.gpu_aamp` | same computation as `aamp`; `device_id` is validated like `gpu_stump`'s and ignored |
| `mass(Q, T, ...)` | `stumpy.mass` | normalized or raw (`p=2`) distance profile of one query; constant flags may be boolean arrays or STUMPY-style callables |
| `mass_absolute(Q, T, T_subseq_isfinite=None, p=2.0, query_idx=None)` | `stumpy.core.mass_absolute` | STUMPY's exact signature, so positional calls port unchanged; `p=2.0` only |
| `match(Q, T, max_distance=..., max_matches=...)` | `stumpy.match` | normalized or raw (`p=2`) matches of a query, nearest first; `max_distance` may be a number or a callable returning a number or a size-1 array |
| `aamp_match(Q, T, T_subseq_isfinite=None, max_distance=None, max_matches=None, atol=1e-8, query_idx=None, p=2.0)` | `stumpy.aamp_match` | STUMPY's exact signature and `query_idx` semantics (the query window keeps its true distance); `p=2.0` only |
| `stimp(T, min_m=3, max_m=None, step=1, normalize=True, p=2.0, T_subseq_isconstant_func=None)` | `stumpy.stimp` with `percentage=1.0, pre_scrump=False` | pan matrix profile: `update()`, `pan(...)`, `PAN_`, `M_`, `P_`; every `update()` computes one exact profile; `normalize=False` follows `stumpy.aamp_stimp`, `p=2.0` only |
| `gpu_stimp(T, min_m=3, max_m=None, step=1, device_id=0, normalize=True, p=2.0, T_subseq_isconstant_func=None)` | `stumpy.gpu_stimp` | same computation as `stimp`; `device_id` is validated like `gpu_stump`'s and ignored |
| `estimated_peak_bytes(l, m, k=1, self_join=True, l_q=None, chunk_size=None, fused=None)` | — | mlx-stump extension: the modeled peak bytes of one join beyond its O(n) per-series arrays (see [Known limitations](#known-limitations)) |

Use these wrappers rather than aliasing `stump`: in a positional call,
`gpu_stump`'s fifth argument (`device_id`) and `aamp`'s (`p`) would otherwise
bind to `normalize`. Inputs are 1-D float64 series, as in STUMPY.
Byte-swapped float64 (`'>f8'`, e.g. read from FITS or HDF5) is accepted and
converted to native byte order once; STUMPY rejects it.

The typed accessors feed downstream STUMPY functions directly: for example,
`stumpy.fluss(mp.I_, ...)` and `stumpy.motifs(T, mp.P_, ...)`. The full 2-D
`mparray` is not itself the one-dimensional input those functions expect.
The exception is `stumpy.stumpi(T, m, k=k, normalize=normalize, mp=...)`,
which takes the full 2-D result of
`mlx_stump.stump(T, m, k=k, normalize=normalize)` as its initial profile
(see the streaming note under [Known limitations](#known-limitations)).

### Pan matrix profile

`stimp` is the GPU equivalent of `stumpy.gpu_stimp` — equivalently, of an
exact `stumpy.stimp(T, min_m, max_m, step, percentage=1.0,
pre_scrump=False, ...)`: every `update()` computes one exact matrix profile
for the next window size in `M_` (which lists them in STUMPY's
breadth-first order), with no `scrump` approximation. `PAN_`, `M_`, `P_`
and `pan(threshold=0.2, normalize=True, contrast=True, binary=True,
clip=True)` follow STUMPY, except for read-only `P_` views, integer-only
window arguments and the order of exactly tied cells (see the `stimp` note
under [Known limitations](#known-limitations)). `normalize=False` follows
STUMPY's `aamp_stimp` (`percentage=1.0, pre_scraamp=False`), `p=2.0` only.
`gpu_stimp` takes `stumpy.gpu_stimp`'s signature and runs the same
computation, so its positional calls port unchanged. `stimp` has no
`percentage` or `pre_scrump` parameter (it always computes the exact
profile), so its fifth positional parameter is `normalize` where
`stumpy.stimp`'s is `percentage`: when porting a `stumpy.stimp` call, pass
everything after `step` by keyword (a positional `0.01` would otherwise
bind to `normalize`).

```python
pan = mlx_stump.stimp(T, min_m=8, max_m=264, step=8)
for _ in range(pan.M_.shape[0]):   # one exact profile per window size
    pan.update()
PAN = pan.PAN_
```

On a random walk with window sizes 8 to 264 in steps of 8 (33 profiles),
the updates took 0.485 s against 1.634 s for
`stumpy.stimp(percentage=1.0, pre_scrump=False)` on all 16 cores at
n=8192 (3.4x; 3.3x in a repeat run) and 2.130 s against 9.325 s at
n=32,768 (4.4x), in 3 interleaved runs each on a shared M4 Max. The binary
`PAN_` was identical and the raw profiles agreed to max |ΔP| ≤ 3.4e-5.

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
   re-evaluated on the CPU in float64 (O(l·k·m) work: a small share of the
   runtime at `k=1` but a sizeable one for large `k`, so each query window
   is normalized once for all `k` neighbors and row chunks run on up to 8
   CPU threads per call) as sums of squared differences of the z-normalized
   windows, with each window's mean and sigma freshly recomputed two-pass —
   a cancellation-free form that stays relatively accurate down to distance
   0 — so reported `P` values use well-conditioned float64 arithmetic for
   the reported neighbor. `match` refines, the same way, every candidate
   that can pass its threshold. With a fixed threshold and a finite
   `max_matches`, that means only the windows that can be among the greedy
   picks, so `max_distance=np.inf, max_matches=k` refines a narrow band
   instead of the whole profile; that top-k result relies on the same
   float32 noise margin as any finite threshold. Precomputed `M_T`/`Σ_T`
   never enter the arithmetic (see [Known limitations](#known-limitations)).
5. **STUMPY-compatible handling** of constant subsequences, NaN/inf values,
   exclusion zones, and left/right profiles, verified by a golden test suite
   that compares every `stump` mode (self- and AB-joins, `k ≥ 1`, normalized
   and raw, single-block and tiled engines) and `mass`/`match` against
   float64 STUMPY, plus a seeded differential fuzz slice (stratified over
   join type × dense/tiled × `k` × normalize) that checks each result
   against an independent float64 oracle and STUMPY. The exclusion zone is
   `ceil(m / stumpy.config.STUMPY_EXCL_ZONE_DENOM)`, read at call time
   whenever STUMPY has been imported (otherwise STUMPY's default
   denominator of 4). Any positive denominator whose zone `ceil(m/denom)`
   is finite works (the suite tests down to 1e-12): a zone wider than the
   series gives STUMPY's all-inf self-join profile. A denominator so small
   that `m/denom` overflows (any subnormal one) raises `OverflowError`, as
   STUMPY does. Exact ties resolve by STUMPY's own rule: in self-joins the
   nearest-in-time candidate wins, and the left one on an equal offset (the
   order of STUMPY's diagonal traversal), for `I_`, `left_I_`, `right_I_`
   and every top-k column; AB-joins take the lowest index (raw top-k ties
   can differ from STUMPY in order and membership; see
   [Known limitations](#known-limitations)). In raw mode
   (`normalize=False`) this covers zero-distance ties (identical windows,
   constant or not): every window's statistics come from one fixed
   pairwise summation of its own values, so identical windows are exactly
   tied in the float32 search as well. Exact ties at nonzero distance
   between different raw windows cannot be ordered by a float32 search
   (e.g. 619 of 2,493 `I_` rows on iid {0, 1, 2} integers, `m=8`). The
   golden suite asserts exact index equality with
   `stumpy.stump`/`stumpy.aamp` on constant, flatline and NaN-segmented
   series, and with `stumpy.aamp` on the zero-distance rows of periodic
   integer, periodic dyadic and planted-motif series (dense and tiled,
   fused kernels and fallback).

Precision metrics are asserted in the test suite and published next to every
benchmark number. Published `idx agree` is strict index equality; the golden
tests additionally verify that any differing index is a float64 near-tie. An
fp32 matrix profile that disagrees with STUMPY on a discord is worthless no
matter how fast. What the float32 search costs in practice: a small fraction
of near-tied neighbors resolve to a different — equally close — index than
STUMPY's (99.98–100.00% index agreement in the benchmarks below; every such
disagreement in the golden suite is verified in float64 to be a true
near-tie). The effect is largest for very small windows on smooth series:
at `m=3` the worst measured discrepancy between tied neighbors is ≈ 0.04,
while at `m ≥ 8` the golden-suite runs (n ≤ 3000) show none and the
large-`n` benchmarks show only the ~1e-5-scale near-ties reported in the
table.

## Benchmarks

Honesty rules (see `bench/`):

- STUMPY is always benchmarked on the **same Mac with all logical CPUs**
  (never against published Xeon tables). The script prints Numba's
  actual thread count and refuses a CPU baseline when it differs from the
  host's logical-CPU count.
- Timings are end-to-end, including validation, float64 preprocessing,
  host↔GPU transfer, and float64 profile refinement.
- Precision metrics are published next to the speed numbers.
- The provenance line names the commit of the git tree the imported
  `mlx_stump` was loaded from, marked `+dirty` whenever tracked or untracked
  workspace changes mean the benchmark is not running that exact commit. An
  installed, non-editable build (e.g. `pip install .`, even into the
  checkout's own `.venv`) prints `installed build, commit unknown` instead of
  vouching for whatever checkout the shell is in. The precision columns
  reuse the last timed call's result, so no untimed extra runs are made.

Self-join on a random walk, `m=200`, best of 2. Provenance (the script
prints this line above its table): Apple M4 Max, macOS 26.6.2, Python
3.12.12, mlx 0.32.2, numpy 2.5.2, STUMPY 1.14.1 with numba parallel on all
16 cores, mlx-stump 0.1.0.dev0 at commit `50da5a6`, 2026-09-26. Timings on a
laptop vary by tens of percent with thermal state and background load;
compare runs from the same session.

| n | mlx-stump (s) | stumpy all-cores (s) | speedup | max \|ΔP\| | idx agree | top-10 discords |
|---|---|---|---|---|---|---|
| 16,384 | 0.037 | 0.105 | 2.9x | 1.21e-06 | 99.99% | 10/10 |
| 65,536 | 0.244 | 0.724 | 3.0x | 2.90e-05 | 99.98% | 10/10 |
| 131,072 | 0.872 | 2.552 | 2.9x | 4.39e-05 | 99.99% | 10/10 |
| 262,144 | 3.731 | 10.791 | 2.9x | 3.14e-05 | 99.99% | 10/10 |

At `m=50` the same benchmark (same provenance) shows 3.5–5.9x (3.5x at
n=16,384, 5.9x at 65,536, 5.6x at 131,072 and 262,144) with 100.00% index
agreement and max |ΔP| ≤ 4.9e-6.

The MASS engine still does O(m) work per distance-matrix cell where STUMPY's
recurrence does O(1). With the reduction fused, the matmul is now more than
half of each batch's time, and its share grows with `m`: at n=131,072,
`m=50`, one 766-row batch's matmul takes 1.80 ms of the 3.31 ms
matmul-plus-fused-reduction, against 20.99 ms with the compiled reduction.
A SCAMP-style diagonal kernel removes the O(m) factor, but a prototype
measured only 1.2–2.2x faster than the fused engine at n=65,536–262,144
(1.16x at n=65,536, `m=100`; 1.69x at n=262,144, `m=100`; 2.21x at
n=262,144, `m=200`). It covers only normalized `k=1` self-joins and needs a
conditioning gate, so it stays on the [Roadmap](#roadmap) as future work; a
symmetric SCAMP variant, which computes each cell once, could roughly double
that.

Above a 256 MiB subsequence matrix (n ≈ 335k at `m=200`) the target is
streamed in 128 MiB column blocks through the same fused reduction: at
n=524,288 that path measured as fast as the dense sweep (26.2 s vs 28.3 s,
medians of 3 interleaved runs on a GPU shared with other work) at about half
the peak memory (401 vs 785 MiB), with identical output. In the session of
the table above, the tiled path took 15.9 s at n=524,288 (`m=200`, one run)
and 81 s at n=1,048,576 (one run, thermally affected); STUMPY was not run at
those sizes. The tiled peak of 401 MiB at n=524,288 compares with a 500 MiB
`estimated_peak_bytes` estimate.

Run them yourself:

```bash
python bench/bench_stump.py --sizes 16384 65536 131072 262144 --m 200 --repeat 2 --seed 0
python bench/bench_stump.py --sizes 16384 65536 131072 262144 --m 50 --repeat 2 --seed 0
python bench/bench_stump.py --sizes 524288 1048576 --m 200 --repeat 1 --seed 0 --no-stumpy
```

## Roadmap

- **v0.1** (this cycle): batched-MASS engine (`stump`, `mass`, `match`,
  AB-joins, `k>1`, `normalize=False`) with the distance evaluation and
  argmin/top-k selection fused into one-pass Metal kernels; STUMPY's
  exact-tie rule; the `aamp`, `gpu_stump` and `gpu_aamp` wrappers;
  `mass_absolute` and `aamp_match`; callable constant flags; the pan matrix
  profile (`stimp`, `gpu_stimp`); golden harness, benchmark harness.
- **next**: SCAMP-style diagonal Metal kernel (one thread per diagonal,
  tiled, register-resident running covariances) for normalized `k=1`
  self-joins — a prototype measured 1.2–2.2x faster than the fused engine
  at n=65,536–262,144 (see [Benchmarks](#benchmarks)), and it needs an
  error-bound re-anchoring guard plus a conditioning gate that keeps
  large-level-shift series on MASS, where its global-frame recurrence loses
  accuracy and MASS does not (a symmetric variant could roughly double the
  gain); chip-specific tuning (M1–M4).
- **dropped**: `stump_batch` for many short series. A `stump` call now
  takes a median ~1.4 ms at n=1000, `m=50` (best ~1.25 ms), below the
  batched prototype's ~2.1 ms/series there, and ~0.55 ms at n=256, `m=16`,
  level with its 0.55 ms/series, so batching no longer pays for a separate
  API.
- **backlog**: `mstump`, GPU `stumpi` updates, snippets, MPdist.

## Known limitations

- `normalize=False` (non-normalized, `aamp`-style) supports `p=2.0` only.
- Callable constant flags (`T_subseq_isconstant`,
  `T_A_subseq_isconstant`/`T_B_subseq_isconstant`, `Q_subseq_isconstant`)
  follow STUMPY's contract: `f(a, w)` is called on a copy of the series with
  inf replaced by NaN and returns a boolean array. Each callable is
  evaluated once per call. Of the STUMPY functions mlx-stump mirrors, only
  `match` hands its callable the series with raw inf (as do STUMPY's
  `stumpi` and `motifs`), so a callable that treats inf and NaN differently
  can flag different windows there. With `normalize=False` the flags are
  validated (so a callable is evaluated) and then ignored; STUMPY does not
  evaluate them in that mode.
- `mass`/`match` operate on 1-D series. For compatibility, a single-column
  `(n, 1)` input is flattened (with STUMPY's warning for `mass(Q, ...)`), but
  STUMPY's genuinely multi-dimensional `Q`/`T` averaging is not implemented.
- In the rare corner of a self-join with an explicit `T_B` and *asymmetric*
  custom constant flags, distances follow consistent row-wise semantics;
  STUMPY's diagonal mirroring can report the unflagged distance for rows
  below a flagged target window.
- With an explicit `T_B` and `ignore_trivial=True`, the join is a self-join
  only when `T_B` equals `T_A`. As in STUMPY, the marker of a missing sample
  (NaN, inf or -inf) does not matter. Unlike STUMPY, a missing sample in one
  series against a real 0.0 (or -0.0) at the same position in the other
  makes them different series: the result is an AB-join, with STUMPY's 'not
  equal' warning. STUMPY zero-fills missing samples before comparing, so it
  treats that pair as a self-join and reports wrong mirrored distances for
  the rows whose `T_A` window contains the missing sample.
- One deliberate accuracy divergence: for near-constant windows (rolling σ
  below ~1e-7 of the data scale — e.g. a flatlined sensor with tiny jitter),
  STUMPY's denominator clamp turns flat-vs-flat pairs into spurious
  zero-distance matches; mlx-stump instead centers and RMS-normalizes each raw
  window in its own bounded float64 frame before the GPU cast, resolving their
  true distances.
- The unrefined `mass` output is the float32 GPU result: near-perfect matches
  read ~1e-3 rather than ~1e-8. `stump` and `match` re-evaluate their reported
  distances in float64, so their outputs don't carry this floor.
- The fused kernels are test-launched once per process and `k`. If a GPU
  cannot run them, mlx-stump emits a `RuntimeWarning` (attributed to the
  user's calling line) and uses the compiled reduction, which is slower
  (about 2–9× here) and uses smaller batches. The results are the same bit
  for bit. Only `k` and the kernel threadgroup width are compile-time
  parameters, so a new window size `m` does not trigger a new Metal
  compile.
- Memory is bounded by three fixed budgets rather than by `n·m`: the
  resident window block (the whole float32 subsequence matrix when it is
  ≤ 256 MiB, otherwise ~128 MiB column blocks, at least four windows wide,
  streamed one at a time — same reduction, bit-identical numerics,
  measured as fast as or faster than the dense sweep), ~384 MiB of live
  per-batch GPU intermediates (an enforced ceiling under automatic chunking,
  including the top-k device buffers for `k > 1`: batches are sized at the
  measured device bytes per distance-matrix cell of the reduction that
  runs — 4 B for the fused Metal kernels, which materialize only the QT
  product, 8–72 B for the compiled fallback used on the CPU device and for
  `k > 16`; the O(n) per-series device arrays (window statistics and
  masks: 6 B per window in normalized mode, 13 B in raw mode, per series,
  plus a 4 B column index per target window on the compiled fallback) come
  on top of it; each batch is synchronized before the next one allocates,
  the trailing batch is computed at full width, and MLX's cache is cleared
  once where tiled blocks narrow by a column, so no second set of buffers
  ever exists), and bounded CPU temporaries (block centering and sigma
  repair are ≤ 64 MiB each; the refinement's float64 windows are ≤ 256 MiB
  in total across its threads and are processed after the window matrix
  and MLX's cached batch buffers have been released). Tiled top-k joins
  additionally need a batch-sized host merge workspace for concatenation,
  tie keys, lexicographic sorting, and gathering (≤ 85 B per neighbor per
  row measured); it scales with `batch_rows·k`, is charged per row when
  the batch is sized, and is included in the peak estimate. Each block is
  built in place: its centered float64 chunks are cast straight into the
  block's own device buffer (unified memory), so no host staging copy
  exists (checked on MLX 0.30 and 0.32; an MLX that does not export the
  buffer writable gets a NumPy-staged build, which holds one more block).
  The dominant modeled peak beyond the O(n) series arrays is therefore
  estimated as `block + max(64 MiB, 8·m + 128 B)` during the build (the
  second term includes the documented one-window float64 floor) or
  `block + 384 MiB` during the sweep, plus numeric/object outputs —
  ~640 MiB for the largest dense block (n=68,000, m=1000, k=1, where RSS
  grew 434–475 MiB in fresh interpreters), ~512 MiB in tiled mode at `k=1`
  (`mlx_stump.estimated_peak_bytes(l, m, k, self_join, l_q, chunk_size,
  fused)` gives that phase estimate; `fused` defaults to the reduction this
  process runs). The O(n) arrays come on top and are deliberately excluded
  from the helper. `mass`/`match` hold one private float64 copy of the
  series (also for byte-swapped or `(n, 1)` input, which is converted once)
  and the float64 profile. Preprocessing's own host transient peaks at
  about 0.5x the series in normalized mode (chunked window masks). In raw
  mode it peaks at about 4.5x, of which about 3.25x is retained for the
  search: the standardized series, its rolling means and centered sums of
  squares, and the window masks (tracemalloc, n=1e7, m=100). End to end at
  n=3e7 (a 229 MiB series, random walk, m=100), `mass`/`match` grow RSS
  above the series by about 1,060 MiB normalized and by about 2,080 /
  2,310–2,365 MiB raw, whatever the NaN density (measured with 0, 300,
  3,000 and 30,000 = 0.1% NaNs; raw `match` varies by ~50 MiB from run to
  run). For large `k` the output itself dominates: STUMPY's object-dtype
  `mparray` layout costs a pointer plus a CPython allocator block per cell,
  ~80 resident bytes per neighbor per row (n=50,000, m=50, k=100: ~385 MiB
  for the returned array alone), which the estimate includes together with
  the top-k reordering temporaries. The canonical case's process RSS
  growth after a warm-up (measured as in `test_large_topk_within_estimate`)
  is ~588 MiB (585.3–592.8 MiB over 10 fresh-interpreter runs on an M4
  Max, macOS 26.6.2) against a ~638 MiB estimate: about 8% headroom, 7%
  for the largest run. It is an estimate with headroom, not a literal cap:
  MLX's allocator rounds buffers up (about +0.5% observed), a gigantic `l`
  can make even a one-row batch exceed the intermediates budget, the
  figures are MLX's own active-memory peak plus host memory (GPU-written
  buffers do not show up in RSS on macOS, only in the process footprint),
  and memory one phase frees is not always returned before the next
  allocates: Metal releases cleared buffers asynchronously (tens to
  ~200 ms after `mx.clear_cache()`) and macOS keeps freed large host
  temporaries resident, so at large `m` the refinement's float64 chunks
  (≤ 256 MiB) can land on part of the sweep's footprint (tiled n=60,000,
  m=4000, k=1 and k=5: RSS grew 555–571 MiB against 508–516 MiB
  estimates). `mass`/`match` evaluate one block at a time and never hold
  more, and every device array is dropped before the cache is cleared, so
  no per-series allocation stays cached after a call returns or raises: an
  error or Ctrl-C in `stump`, `mass` or `match` (during preprocessing or
  the GPU sweep) releases the per-window arrays, window block and batch
  buffers before the exception propagates. MLX may retain a small
  runtime/allocator baseline (2.6 MiB on one hosted-runner image). For
  `stump`, pass `chunk_size` to trade memory for larger batches, and pass
  the same value to `estimated_peak_bytes` because an explicit batch is
  allowed to exceed the automatic 384 MiB budget. `mass` and `match` always
  use automatic block streaming.
- Out-of-range `query_idx` values (including negative ones) raise
  `ValueError`. STUMPY silently wraps `query_idx <= -m` through numpy
  negative indexing and fabricates a zero-distance match at a negative
  index; rejecting is a deliberate, stricter divergence. With
  `normalize=False`, `match` zeroes `D[query_idx]` the way
  `stumpy.mass_absolute` does (when the window is finite).
  `stumpy.match(normalize=False)` dispatches to `stumpy.aamp_match`, which
  leaves that entry at its true distance. A *mismatched* `query_idx` is
  therefore the first zero-distance match of `mlx_stump.match`, but keeps
  its true distance in STUMPY, which returns nothing when that distance
  exceeds the threshold. `mlx_stump.aamp_match` reproduces
  `stumpy.aamp_match` exactly, including this.
- A data-dependent `max_distance` in `match` (the default, or a callable)
  is evaluated on successively refined profiles until it stops moving —
  the float32 profile first, then once per float64 refinement round,
  typically 2–3 calls in total and at most 9 — so the threshold that
  selects the matches comes from a profile that is float64-refined
  throughout the threshold band. STUMPY calls it once, on its float64
  profile; a callable with side effects would observe the difference. The
  callable may return a number or a size-1 array (e.g.
  `np.nanpercentile(D, [1])`), as in STUMPY.
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
  array of size 1) or a callable returning one: non-boolean values such as
  `"False"` or `0` raise `ValueError` rather than being coerced. A plain
  boolean is accepted for the scalar `Q` flag; boolean *lists* of the
  required length are accepted for the `T_*_subseq_isconstant` arrays, where
  STUMPY requires an `np.ndarray`. All constant-flag controls are shape/type
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
  segment in unit noise (`m=7`) index agreement is ~83% (`I_` 83.3%,
  `left_I_` 83.7%, `right_I_` 83.8%), with a 1e6 offset segment in a random
  walk (`m=50`) ~99%. Nearly all of the first case's disagreements (244 of
  250 rows) are rows on the constant segment itself, whose candidates are
  all exactly 0 apart. mlx-stump resolves them by STUMPY's nearest-in-time
  tie rule, but STUMPY's float64 `aamp` recurrence leaves ~1.4e-3 of
  rounding over that segment's 1e10-scale squared terms, different on each
  diagonal, so its own noise picks the neighbor. Where STUMPY's recurrence
  is exact, zero-distance ties between identical windows agree exactly
  (integer or dyadic data: all 1,091 rows of a 30-fold tile of a 37-sample
  integer pattern at `m=20`). Every disagreement is a float32 near-tie
  *relative to the distance itself* — the two candidates' true distances
  differ by ~1e-7 of their magnitude (≤ 5e-7 in those two cases, which the
  test suite asserts: 3.6e-5 raw units on a ~1.7e5 distance, 0.08 on a
  ~4e6 one; up to ~5e-6 on long series with large `m`). The reported `P`
  is directly recomputed in float64 for the chosen neighbor, and `match`
  widens its re-evaluation cutoff per-window so true matches are not
  dropped. From ~1e6 of dynamic range on, STUMPY's own CPU `aamp` — a
  float64 diagonal recurrence over terms of the segment's squared
  magnitude — drifts (its `P` is off by ~1e-2 at 1e6, ~0.5 at 1e7, ~5 at
  1e8 on unit-scale rows, and it can pick neighbors far from the true
  nearest), so agreement with it stops being a precision metric there;
  measured against direct float64 evaluation, mlx-stump's neighbor gaps
  stay ≤ ~1e-7 relative and its `P` agrees with that evaluation up to the
  ~1e13 standardization limit.
- With `normalize=False` and `k > 1`, `stumpy.aamp`'s choice among exactly
  tied top-k candidates depends on its traversal. Within a thread it
  inserts a new tie in front of equal entries, so a later smaller distance
  evicts the *nearest* member of a tied group; its cross-thread merge then
  fills the remaining slots from each later thread in that reversed order.
  Both the order and the set of tied members therefore vary with the
  series and with `NUMBA_NUM_THREADS`: on the integer tile above at `k=3`,
  148 of 1,091 rows keep their 3 nearest copies at 16 threads, against all
  1,091 single-threaded. mlx-stump orders the candidates its float32 search
  sees as exactly tied by (distance, nearest in time, left first on an
  equal offset), the rule it uses everywhere else; identical raw windows
  are such ties (see [Precision](#precision)), so on that tile every row
  keeps its 3 nearest copies. Compare raw top-k results with STUMPY by
  distance, not by index set. Normalized `stump` (any `k`) is
  thread-independent in STUMPY, and its exact ties resolve exactly as
  there; with raw `k = 1`, zero-distance ties do wherever STUMPY's float64
  recurrence is exact (e.g. integer or dyadic data).
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
  `stimp`'s raw `pan()` also runs cleanly under `np.errstate(all="raise")`
  on `2**-1060`-, 1e307- and 1.7e308-scale series and on one mixing
  1e-300- and 1e305-scale halves, but it is not scale-invariant at the
  extremes: its normalization is STUMPY's `(max(T) − min(T))·sqrt(m)`,
  which under- or overflows there, so the pan saturates exactly as
  STUMPY's does (a unit-amplitude random walk rescaled to `2**-1060` or
  1.7e308 changes 296 of 2,400 binary pan cells, bit for bit like
  `stumpy.aamp_stimp`; at 1e307 none change).
- `stimp`/`gpu_stimp` differ from STUMPY's pan matrix profile in three
  ways. `P_` returns read-only views of the internal pan array (use
  `.copy()` to edit one), where STUMPY's are writeable. `min_m`, `max_m`
  and `step` must be integers (a float raises `TypeError`); STUMPY also
  accepts floats and truncates them to int64. On series with large groups
  of exactly tied distances (exactly periodic or discrete-valued data, or
  constant runs with `normalize=False`), the rank-based contrast can order
  tied cells differently, so some binary `PAN_` cells differ while `P_`
  still agrees within float tolerance (292 of 6,000 cells, max |ΔP|
  3.5e-7, on a 16-fold tile of 25 Gaussian samples at `m` = 4 to 60 in
  steps of 4).
- Streaming updates (`stumpi`) are not GPU-accelerated: `stumpi.update` runs
  on STUMPY's CPU path. The expensive initial profile can come from the GPU:
  ```python
  stream = stumpy.stumpi(T, m, k=k, normalize=normalize,
                         mp=mlx_stump.stump(T, m, k=k, normalize=normalize))
  ```
  `k` and `normalize` must match: a `k` mismatch raises, but a `normalize`
  mismatch is accepted silently and gives wrong distances (raw mode is `p=2`
  only). With a custom `T_subseq_isconstant_func`, pass the same mask to
  `mlx_stump.stump` as
  `T_A_subseq_isconstant=stumpy.core.process_isconstant(T, m, func)`
  (pass this mask, not `func` itself: stumpi evaluates `func` on the raw
  series, inf included, whereas `mlx_stump.stump` replaces inf by NaN
  first, so the two masks can differ). Row indices differ from stumpi's
  own start-up only where `mlx_stump.stump`'s indices already differ from
  `stumpy.stump`'s (see the tie notes under [Precision](#precision)).

## License

MIT
