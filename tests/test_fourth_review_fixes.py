"""Regression tests for the fourth external review (two rounds).

1. Cached statistics: honoring a supplied ``Σ_T`` through ``2m(1 - rho)``
   turned its own rounding into a ``sqrt(2m*delta)`` floor on perfect
   matches (8e-4 at offset 1e12, 0.06 at 1e14 with STUMPY's
   ``compute_mean_std``), a raw-equality snap then missed shifted (affine)
   duplicates, and non-finite entries mixed STUMPY's and mathematical
   semantics. The explicit choice now: ``M_T``/``Σ_T`` are compatibility
   metadata — validated, with an infinite ``M_T`` marking its window
   non-finite, but otherwise absent from the arithmetic — so results equal
   the no-stats call exactly.
2. ``mx.clear_cache()`` ran while the per-series device arrays were still
   alive, so they entered MLX's cache when the call returned (17 MiB after
   ``mass(n=1e6)``, 86 MiB at 5e6) despite the README's "nothing stays
   cached".
3. ``estimated_peak_bytes`` was presented as a ceiling and undercounted large
   top-k results by ~37% (the object-dtype ``mparray`` and the reorder
   temporaries); it now models the assembly phase, takes separate query
   and target lengths, and is documented as an estimate.
4. Any matching ``v*`` tag could publish, from any commit, running that
   commit's own copy of the workflow; releases are now a manual workflow
   that only runs from ``main`` and creates the tag itself.
5. Tests bundled in the sdist read repository files that the sdist does not
   ship; they skip when those files are absent, and CI runs the sdist's
   tests.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import textwrap
import warnings

import numpy as np
import pytest
import stumpy

import mlx_stump
from mlx_stump._engine import _CHUNK_MEM_BUDGET, estimated_peak_bytes, resident_block_bytes

MIB = 1 << 20
_REPO = pathlib.Path(__file__).resolve().parents[1]
_RELEASE_YML = _REPO / ".github/workflows/publish-pypi.yml"


@pytest.mark.parametrize("offset,jitter", [(1e9, 1e-3), (1e12, 1e-3), (1e14, 0.1)])
def test_cached_stats_change_nothing_and_duplicates_read_zero(offset, jitter):
    """Bitwise duplicates and shifted (affine) copies are at z-distance 0;
    with STUMPY's own cached stats they must be found under
    max_distance=0, and every result must equal the no-stats call. (The
    jitter must exceed the float64 ulp at the offset — 1.2e-4 at 1e12,
    1.6e-2 at 1e14 — or the whole series quantizes to a constant.)"""
    rng = np.random.default_rng(3)
    T = offset + jitter * rng.standard_normal(6000)
    assert np.ptp(T) > 0
    m = 100
    Q = T[1000 : 1000 + m].copy()
    T[3000 : 3000 + m] = Q  # bitwise duplicate
    T[5000 : 5000 + m] = Q + 100.0 * jitter  # shifted copy (exact in float64 here)
    assert np.array_equal(T[5000 : 5000 + m] - T[5000], Q - Q[0])
    M_T, Σ_T = stumpy.core.compute_mean_std(T, m)
    for kw in ({"max_distance": 0.0}, {"max_distance": 0.01}, {}):
        M = mlx_stump.match(Q, T, M_T=M_T, Σ_T=Σ_T, **kw)
        M0 = mlx_stump.match(Q, T, **kw)
        np.testing.assert_array_equal(M.astype(float), M0.astype(float))
        assert sorted(int(i) for _, i in M[:3]) == [1000, 3000, 5000], (offset, kw)
        assert all(float(d) == 0.0 for d, _ in M[:3])
    np.testing.assert_array_equal(mlx_stump.mass(Q, T, M_T=M_T, Σ_T=Σ_T), mlx_stump.mass(Q, T))


def test_cached_stats_semantics_vs_stumpy():
    """The explicit divergence: STUMPY uses the supplied values literally,
    mlx-stump only keeps the infinite-M_T marker. Constant-window rules
    therefore agree with STUMPY for a NaN M_T (which STUMPY skips) while a
    non-finite or scaled Σ_T is ignored here and changes STUMPY's result."""
    rng = np.random.default_rng(5)
    T = rng.standard_normal(300)
    T[100:140] = 2.0  # windows 100..124 are constant at m=16
    m = 16
    Q = T[10 : 10 + m].copy()
    D0 = mlx_stump.mass(Q, T)
    M_T, Σ_T = stumpy.core.compute_mean_std(T, m)
    Mb, S = M_T.copy(), Σ_T.copy()
    Mb[50] = np.inf  # STUMPY: inf mean -> inf distance, ahead of everything
    Mb[110] = np.nan  # constant window: STUMPY's constant rule applies (isinf is False)
    S[200] = np.inf  # STUMPY: 1/inf -> rho = 0 -> sqrt(2m); ignored here
    S[210] = np.nan  # STUMPY: NaN; ignored here
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        D = mlx_stump.mass(Q, T, M_T=Mb, Σ_T=S)
        Dr = stumpy.mass(Q, T, M_T=Mb, Σ_T=S)
    assert np.isinf(D[50]) and np.isinf(Dr[50])
    assert D[110] == Dr[110] == np.sqrt(m)  # Q is not constant, window 110 is
    assert D[200] == D0[200] and Dr[200] == np.sqrt(2 * m)
    assert D[210] == D0[210] and np.isnan(Dr[210])
    keep = np.ones(D.shape[0], dtype=bool)
    keep[50] = False
    np.testing.assert_array_equal(D[keep], D0[keep])
    M = mlx_stump.match(Q, T, M_T=Mb, Σ_T=S, max_distance=float("inf"))
    assert 50 not in {int(i) for _, i in M}


@pytest.mark.slow
def test_nothing_cached_after_large_calls():
    import mlx.core as mx

    rng = np.random.default_rng(4)
    # MLX retains a small, version/runner-dependent kernel/allocator baseline
    # even after clear_cache (2.6 MiB on GitHub's macos-15 image, <1 MiB on
    # the development host). The regression is that per-series buffers must
    # not scale that baseline with n: the old ordering left 17/86+ MiB behind.
    mx.clear_cache()
    warm = rng.standard_normal(2_000).cumsum()
    mlx_stump.mass(warm[:100].copy(), warm)
    mass_baseline = mx.get_cache_memory()

    T = rng.standard_normal(1_000_000).cumsum()
    Q = T[100:200].copy()
    mlx_stump.mass(Q, T)
    del T, Q
    mass_cached = mx.get_cache_memory()
    assert mass_cached <= mass_baseline + 4 * MIB, (
        f"mass cache grew by {(mass_cached - mass_baseline) / MIB:.1f} MiB"
    )

    mlx_stump.stump(warm, 50)
    stump_baseline = mx.get_cache_memory()
    T = rng.standard_normal(200_000).cumsum()
    mlx_stump.stump(T, 50)
    del T, warm
    stump_cached = mx.get_cache_memory()
    assert stump_cached <= stump_baseline + 4 * MIB, (
        f"stump cache grew by {(stump_cached - stump_baseline) / MIB:.1f} MiB"
    )
    assert mx.get_active_memory() < MIB


def test_peak_estimate_terms():
    l, m = 63_337, 2_200
    base = estimated_peak_bytes(l, m)
    assert base >= resident_block_bytes(l, m) + _CHUNK_MEM_BUDGET + l * 32
    # the object-dtype output dominates for large k: ~80 bytes/neighbor/row
    l, m, k = 49_951, 50, 100
    est = estimated_peak_bytes(l, m, k=k)
    assert est >= l * k * 68
    assert 550 * MIB <= est <= 720 * MIB, est / MIB
    # AB-joins: the block follows the target, the outputs follow the query
    assert estimated_peak_bytes(1_000, 50, l_q=200_000) > estimated_peak_bytes(1_000, 50)
    assert estimated_peak_bytes(200_000, 50, l_q=1_000) < estimated_peak_bytes(200_000, 50)
    # a single batch row that exceeds the budget is what gets allocated
    huge_l = 100_000_000
    assert estimated_peak_bytes(huge_l, 3) > resident_block_bytes(huge_l, 3) + _CHUNK_MEM_BUDGET


@pytest.mark.slow
def test_large_topk_within_estimate():
    """The canonical top-k process peak stays below the published estimate."""
    n, m, k = 50_000, 50, 100
    l = n - m + 1
    src = textwrap.dedent(
        f"""
        import resource, sys
        import numpy as np
        import mlx_stump
        unit = 1 if sys.platform == "darwin" else 1024
        T = np.random.default_rng(0).standard_normal({n}).cumsum()
        mlx_stump.stump(T[:4096], {m}, k={k})  # warm-up: Metal/JIT baseline
        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * unit
        mp = mlx_stump.stump(T, {m}, k={k})
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * unit
        print("RESULT", (peak - before) / 2**20)
        """
    )
    out = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True, check=True)
    growth = float([ln for ln in out.stdout.splitlines() if ln.startswith("RESULT")][-1].split()[1])
    est = estimated_peak_bytes(l, m, k=k) / MIB
    assert growth <= est, f"RSS grew {growth:.0f} MiB vs estimate {est:.0f} MiB"


@pytest.mark.skipif(not _RELEASE_YML.exists(), reason="workflow files are not shipped in the sdist")
def test_release_workflow_is_manual_main_only_and_tags_before_publish():
    text = _RELEASE_YML.read_text()
    assert "workflow_dispatch" in text
    assert "tags:" not in text.split("jobs:")[0]  # no tag trigger
    assert '"${GITHUB_REF}" != "refs/heads/main"' in text
    guard = text.index("Refuse to release from anything but main")
    assert guard < text.index("check_tag_version.py")
    # The reviewed environment exposes the sole tag-bypass deploy key to the
    # tag job; the ordinary workflow token stays read-only.  The verified tag
    # must exist before the separate OIDC-only publish job can run.
    tag_push = text.index('"git@github.com:${GITHUB_REPOSITORY}.git" "refs/tags/${tag}"')
    assert "environment: pypi-release-v2" in text[text.index("  tag:") : text.index("  publish:")]
    assert tag_push < text.index("pypa/gh-action-pypi-publish")
    # the version input reaches shell steps only through an environment variable
    assert "VERSION: ${{ inputs.version }}" in text
    assert "${{ inputs.version }}" not in text.replace("VERSION: ${{ inputs.version }}", "")
