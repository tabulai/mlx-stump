"""An exact occurrence must survive a float32 tie with earlier near copies."""

from __future__ import annotations

import numpy as np
import pytest

import mlx_stump
import mlx_stump._engine as engine

pytestmark = pytest.mark.filterwarnings("ignore:A large number of values in `P`:UserWarning")


def _mode(monkeypatch, *, fallback: bool, tiled: bool) -> None:
    if fallback:
        monkeypatch.setattr(engine, "_fused_reducer", lambda k: False)
    if tiled:
        monkeypatch.setattr(engine, "_MATMUL_WINDOW_BYTES", 0)
        monkeypatch.setattr(engine, "_TILE_WINDOW_BYTES", 12 * 8 * 4)


@pytest.mark.parametrize(
    "fallback,tiled", [(False, False), (True, False), (False, True), (True, True)]
)
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("k", [1, 2])
def test_ab_exact_copy_beats_multiple_near_copies(monkeypatch, fallback, tiled, normalize, k):
    _mode(monkeypatch, fallback=fallback, tiled=tiled)
    q = np.arange(8, dtype=np.float64)
    target = np.full(40, 100.0)
    for j in (0, 16, 32):
        target[j : j + 8] = q
    for j in (0, 16):
        target[j] += 1e-4
        if not normalize:
            target[j + 1] -= 1e-4

    mp = mlx_stump.stump(
        q, 8, target, ignore_trivial=False, normalize=normalize, k=k, chunk_size=3
    )
    expected = [32] if k == 1 else [32, 0]
    np.testing.assert_array_equal(np.atleast_1d(mp.I_[0]), expected)
    assert np.atleast_1d(mp.P_[0])[0] == 0.0


@pytest.mark.parametrize(
    "fallback,tiled", [(False, False), (True, False), (False, True), (True, True)]
)
@pytest.mark.parametrize("k", [1, 2])
def test_self_exact_copy_replaces_near_global_and_side_neighbors(monkeypatch, fallback, tiled, k):
    _mode(monkeypatch, fallback=fallback, tiled=tiled)
    q = np.arange(8, dtype=np.float64)
    series = np.full(72, 100.0)
    for j in (0, 16, 32, 48, 64):
        series[j : j + 8] = q
    series[[0, 64]] += 1e-4

    mp = mlx_stump.stump(series, 8, k=k, chunk_size=3)
    expected = [16] if k == 1 else [16, 48]
    np.testing.assert_array_equal(np.atleast_1d(mp.I_[32]), expected)
    np.testing.assert_array_equal(np.atleast_1d(mp.P_[32]), np.zeros(k))
    assert mp.left_I_[32] == 16
    assert mp.right_I_[32] == 48


@pytest.mark.parametrize("exact", [lambda q: q, lambda q: 3 * q + 5])
@pytest.mark.parametrize("perturbation", [1e-4, 1e-9])
def test_exact_normalized_match_beats_near_copy_even_below_reported_zero_snap(
    exact, perturbation
):
    q = np.arange(8, dtype=np.float64)
    near = q.copy()
    near[0] += perturbation
    target = np.r_[near, np.full(8, 10.0), exact(q)]

    mp = mlx_stump.stump(q, 8, target, ignore_trivial=False)
    assert mp.I_[0] == 16
    assert mp.P_[0] == 0.0
