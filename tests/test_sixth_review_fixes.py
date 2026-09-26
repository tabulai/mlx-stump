"""Regression tests for the sixth review round (multi-agent audit, 2026-09-26).

This round's tests are split by area; this module holds the shared
groundwork:

1. ``stumpy.config.STUMPY_EXCL_ZONE_DENOM`` was ignored: mlx-stump hard-coded
   a denominator of 4, so a user who configured STUMPY's only exclusion-zone
   knob got neighbours inside STUMPY's zone from the claimed drop-in, and
   ``match`` returned overlapping matches. The denominator is now read once
   per ``stump``/``match`` call (the raw positive value, as STUMPY does) and
   recorded on the returned ``mparray``.
2. Two copies of the refinement chunk helper read different module globals,
   so the memory estimate and the real stump/match refinement chunk could
   diverge. There is now one helper, ``_engine.refine_chunk_rows``.
3. ``MassEngine.sliding_dot_products`` had an unreachable tiled branch that
   would have broken the per-batch memory bound if anyone called it; the
   method is gone and the dense sweep multiplies against ``W_T`` directly.

See ``test_sixth_review_engine.py``, ``test_sixth_review_stump.py``,
``test_sixth_review_mass_match.py`` and ``test_sixth_review_infra.py`` for
the rest of the round.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import mlx_stump
import mlx_stump._engine as eng

from .conftest import assert_indices_tie_tolerant, assert_profile_close, tie_tolerance

stumpy = pytest.importorskip("stumpy")


def _walk(n, seed):
    return np.random.default_rng(seed).standard_normal(n).cumsum()


@pytest.mark.parametrize("denom", [0.5, 1, 2.5, 8])
@pytest.mark.parametrize("k", [1, 3])
@pytest.mark.parametrize("normalize", [True, False])
def test_stump_honours_stumpy_excl_zone_denom(monkeypatch, denom, k, normalize):
    monkeypatch.setattr(stumpy.config, "STUMPY_EXCL_ZONE_DENOM", denom)
    T = _walk(400, seed=3)
    m = 20
    mp = mlx_stump.stump(T, m, k=k, normalize=normalize)
    ref = stumpy.stump(T, m, k=k) if normalize else stumpy.aamp(T, m, k=k)
    assert mp._excl_zone_denom == denom
    excl = int(np.ceil(m / denom))
    I = np.atleast_2d(mp.I_.T).T if k == 1 else mp.I_
    rows = np.arange(I.shape[0])[:, None]
    valid = I >= 0
    # no neighbour may fall inside the configured zone
    assert np.all(np.abs(I - rows)[valid] > excl)
    tie = tie_tolerance(m)
    P, Pr = np.atleast_2d(mp.P_.T).T, np.atleast_2d(ref.P_.T).T
    Ir = np.atleast_2d(ref.I_.T).T
    for j in range(k):
        assert_indices_tie_tolerant(
            I[:, j], Ir[:, j], T, T, m, normalize=normalize, tie_atol=tie
        )
        assert_profile_close(P[:, j], Pr[:, j], m=m, tie_atol=tie)
    assert_indices_tie_tolerant(
        mp.left_I_, ref.left_I_, T, T, m, normalize=normalize, tie_atol=tie
    )
    assert_indices_tie_tolerant(
        mp.right_I_, ref.right_I_, T, T, m, normalize=normalize, tie_atol=tie
    )


@pytest.mark.parametrize("denom", [1, 3])
def test_tiled_sweep_honours_excl_zone_denom(monkeypatch, denom):
    monkeypatch.setattr(stumpy.config, "STUMPY_EXCL_ZONE_DENOM", denom)
    T = _walk(500, seed=4)
    m = 24
    dense = mlx_stump.stump(T, m)
    monkeypatch.setattr(eng, "_MATMUL_WINDOW_BYTES", 0)
    monkeypatch.setattr(eng, "_TILE_WINDOW_BYTES", 37 * m * 4)
    tiled = mlx_stump.stump(T, m)
    for attr in ("P_", "I_", "left_I_", "right_I_"):
        np.testing.assert_array_equal(getattr(tiled, attr), getattr(dense, attr))
    excl = int(np.ceil(m / denom))
    I = tiled.I_
    assert np.all(np.abs(I - np.arange(I.size))[I >= 0] > excl)


@pytest.mark.parametrize("denom", [1, 2.5])
@pytest.mark.parametrize("normalize", [True, False])
def test_match_honours_excl_zone_denom(monkeypatch, denom, normalize):
    monkeypatch.setattr(stumpy.config, "STUMPY_EXCL_ZONE_DENOM", denom)
    T = _walk(1500, seed=5)
    Q = T[700:750].copy()
    ours = mlx_stump.match(Q, T, max_matches=12, normalize=normalize)
    ref = stumpy.match(Q, T, max_matches=12, normalize=normalize)
    assert [int(i) for _, i in ours] == [int(i) for _, i in ref]
    idx = np.sort(np.array([int(i) for _, i in ours]))
    assert np.all(np.diff(idx) > int(np.ceil(50 / denom)))


def test_window_advisory_uses_configured_denominator(monkeypatch):
    T = _walk(200, seed=6)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mlx_stump.stump(T, 70)  # (131 // 2) > ceil(70 / 4): no advisory
    monkeypatch.setattr(stumpy.config, "STUMPY_EXCL_ZONE_DENOM", 1)
    with pytest.warns(UserWarning, match="may be too large"):
        mlx_stump.stump(T, 70)


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf"), "4", None, True])
def test_invalid_excl_zone_denom_is_rejected(monkeypatch, bad):
    monkeypatch.setattr(stumpy.config, "STUMPY_EXCL_ZONE_DENOM", bad)
    T = _walk(120, seed=7)
    with pytest.raises(ValueError, match="STUMPY_EXCL_ZONE_DENOM"):
        mlx_stump.stump(T, 10)
    with pytest.raises(ValueError, match="STUMPY_EXCL_ZONE_DENOM"):
        mlx_stump.match(T[:10].copy(), T)


def test_default_denominator_is_four():
    assert stumpy.config.STUMPY_EXCL_ZONE_DENOM == 4
    assert mlx_stump.stump(_walk(100, seed=8), 8)._excl_zone_denom == 4


def test_one_refinement_budget_helper():
    import mlx_stump._match as match_mod
    import mlx_stump._stump as stump_mod

    assert stump_mod.refine_chunk_rows is eng.refine_chunk_rows
    assert match_mod.refine_chunk_rows is eng.refine_chunk_rows
    assert not hasattr(stump_mod, "_REFINE_MEM_BUDGET")
    assert not hasattr(eng.MassEngine, "sliding_dot_products")
