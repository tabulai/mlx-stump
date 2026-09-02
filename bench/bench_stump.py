"""Benchmark mlx_stump.stump against stumpy.stump on this machine.

Honesty rules:
- STUMPY runs on THIS Mac with all cores (never quoted from published tables).
- Timings are end-to-end: validation, float64 preprocessing, host<->GPU
  transfer, and the float64 profile refinement are all inside the clock.
- Precision metrics are printed next to the speed numbers: max |dP| against
  float64 STUMPY, exact index agreement, and top-10 discord overlap.

Usage:
    python bench/bench_stump.py --sizes 16384 65536 131072 --m 200 --repeat 3
    python bench/bench_stump.py --sizes 1048576 --no-stumpy   # GPU-only scaling
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

import mlx_stump


def _time(fn, repeat: int) -> float:
    best = np.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _discord_topk(P: np.ndarray, excl: int, k: int = 10) -> set[int]:
    """Top-k discord positions with an exclusion zone (greedy, like analysis code)."""
    P = P.copy()
    P[~np.isfinite(P)] = -np.inf
    out = []
    for _ in range(k):
        i = int(np.argmax(P))
        if not np.isfinite(P[i]):
            break
        out.append(i)
        P[max(0, i - excl) : i + excl + 1] = -np.inf
    return set(out)


def provenance(stumpy_threads: int | None = None) -> str:
    """One line naming the machine, software and commit the numbers came from."""
    import datetime
    import platform
    import subprocess

    import mlx.core as mx

    def _run(*cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
        except Exception:  # noqa: BLE001 - provenance is best effort
            return "unknown"

    chip = _run("sysctl", "-n", "machdep.cpu.brand_string")
    commit = _run("git", "rev-parse", "--short", "HEAD")
    tree_state = _run("git", "status", "--porcelain", "--untracked-files=normal")
    if tree_state not in ("", "unknown"):
        commit += "+dirty"
    try:
        import stumpy

        stumpy_v = stumpy.__version__
    except ImportError:
        stumpy_v = "not installed"
    thread_info = "" if stumpy_threads is None else f", numba threads {stumpy_threads}"
    return (
        f"{chip}, macOS {platform.mac_ver()[0]}, Python {platform.python_version()}, "
        f"mlx {mx.__version__}, numpy {np.__version__}, stumpy {stumpy_v}{thread_info}, "
        f"mlx-stump {mlx_stump.__version__} @ {commit}, {datetime.date.today().isoformat()}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", type=int, nargs="+", default=[16384, 65536, 131072])
    ap.add_argument("--m", type=int, default=200)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-stumpy", action="store_true", help="skip the CPU baseline")
    args = ap.parse_args()

    use_stumpy = not args.no_stumpy
    stumpy = None
    stumpy_threads = None
    if use_stumpy:
        try:
            import stumpy  # noqa: F811
        except ImportError:
            print("stumpy not installed - GPU-only run (from a checkout: pip install '.[bench]')")
            use_stumpy = False
        else:
            import numba

            stumpy_threads = numba.get_num_threads()
            available_threads = os.cpu_count()
            if available_threads is not None and stumpy_threads != available_threads:
                raise RuntimeError(
                    "The STUMPY baseline must use every logical CPU: "
                    f"Numba has {stumpy_threads}, this host reports {available_threads}. "
                    "Unset NUMBA_NUM_THREADS or pass --no-stumpy."
                )

    rng = np.random.default_rng(args.seed)
    m = args.m
    if args.repeat < 1:
        ap.error("--repeat must be at least 1")
    if m < 3:
        ap.error("--m must be at least 3")
    too_short = [n for n in args.sizes if n < m]
    if too_short:
        ap.error(f"every --sizes value must be at least m={m}; found {too_short}")
    excl = int(np.ceil(m / 4))

    # warm up: Metal kernel compilation and numba JIT are one-time costs
    T_warm = rng.standard_normal(max(4096, 2 * m + 10)).cumsum()
    mlx_stump.stump(T_warm, m)
    if use_stumpy:
        stumpy.stump(T_warm, m)

    stumpy_label = (
        "stumpy skipped" if stumpy_threads is None else f"stumpy {stumpy_threads}-thread"
    )
    header = (
        f"| n | mlx-stump (s) | {stumpy_label} (s) | speedup "
        "| max dP | idx agree | top-10 discords |"
    )
    rule = "|---|---|---|---|---|---|---|"
    print(f"\n{provenance(stumpy_threads)}")
    print(f"random walk, m={m}, best of {args.repeat}\n")
    print(header)
    print(rule)

    for n in args.sizes:
        T = rng.standard_normal(n).cumsum()
        t_ours = _time(lambda T=T: mlx_stump.stump(T, m), args.repeat)
        mp = mlx_stump.stump(T, m)
        if use_stumpy:
            t_ref = _time(lambda T=T: stumpy.stump(T, m), args.repeat)
            ref = stumpy.stump(T, m)
            dP = float(np.max(np.abs(mp.P_ - ref.P_)))
            agree = float(np.mean(mp.I_ == ref.I_))
            overlap = len(_discord_topk(mp.P_, excl) & _discord_topk(ref.P_, excl))
            print(
                f"| {n:,} | {t_ours:.3f} | {t_ref:.3f} | {t_ref / t_ours:.1f}x "
                f"| {dP:.2e} | {agree:.2%} | {overlap}/10 |"
            )
        else:
            print(f"| {n:,} | {t_ours:.3f} | - | - | - | - | - |")


if __name__ == "__main__":
    main()
