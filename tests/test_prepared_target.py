"""Repeated MASS/match queries against one owned target snapshot."""

from __future__ import annotations

import numpy as np
import pytest

import mlx_stump
from mlx_stump import _engine


def _walk(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(n).cumsum().astype(np.float64)


def test_prepared_normalized_reuses_dense_packing(monkeypatch):
    T = _walk(1024)
    m = 48
    calls = 0
    original = _engine.MassEngine._build_block_T

    def counted(self, j0, j1):
        nonlocal calls
        calls += 1
        return original(self, j0, j1)

    monkeypatch.setattr(_engine.MassEngine, "_build_block_T", counted)
    with mlx_stump.prepare_target(T, m) as target:
        assert calls == 1
        for i in (9, 173, 555):
            Q = T[i : i + m].copy()
            before_reference = calls
            expected = mlx_stump.mass(Q, T)
            assert calls == before_reference + 1
            np.testing.assert_array_equal(target.mass(Q), expected)
            assert calls == before_reference + 1

            expected_matches = mlx_stump.match(
                Q, T, max_distance=np.inf, max_matches=3, query_idx=i
            )
            after_reference = calls
            np.testing.assert_array_equal(
                target.match(Q, max_distance=np.inf, max_matches=3, query_idx=i),
                expected_matches,
            )
            assert calls == after_reference


def test_prepared_target_owns_series_flags_and_stats():
    T = _walk(640, seed=1)
    T[45:52] = 1.0
    original = T.copy()
    m = 16
    l = T.size - m + 1
    flags = np.zeros(l, dtype=bool)
    flags[100] = True
    stats_mean = np.zeros(l)
    stats_std = np.ones(l)
    stats_mean[30] = np.inf
    expected = mlx_stump.mass(
        original[210 : 210 + m].copy(),
        original,
        M_T=stats_mean,
        Σ_T=stats_std,
        T_subseq_isconstant=flags,
    )
    with mlx_stump.prepare_target(
        T,
        m,
        M_T=stats_mean,
        Σ_T=stats_std,
        T_subseq_isconstant=flags,
    ) as target:
        T[:] = -999.0
        flags[:] = True
        stats_mean[:] = 0.0
        Q = original[210 : 210 + m].copy()
        np.testing.assert_array_equal(target.mass(Q), expected)
        expected_match = mlx_stump.match(
            Q,
            original,
            M_T=np.where(np.arange(l) == 30, np.inf, 0.0),
            Σ_T=stats_std,
            T_subseq_isconstant=np.arange(l) == 100,
            max_distance=np.inf,
            max_matches=5,
        )
        np.testing.assert_array_equal(
            target.match(Q, max_distance=np.inf, max_matches=5), expected_match
        )


@pytest.mark.parametrize("layout", ["column", "byte_swapped"])
def test_prepared_target_owns_converted_layout(layout):
    T = _walk(256, seed=11)
    source = T[:, None].copy() if layout == "column" else T.astype(">f8")
    Q = T[70:94].copy()
    expected = mlx_stump.mass(Q, T)
    with mlx_stump.prepare_target(source, 24) as target:
        source[...] = 0.0
        np.testing.assert_array_equal(target.mass(Q), expected)


def test_prepared_raw_keeps_per_query_frame_and_override():
    T = _walk(512, seed=2)
    T[300] = np.nan
    original = T.copy()
    m = 32
    l = T.size - m + 1
    override = np.ones(l, dtype=bool)
    flags = np.zeros(l, dtype=bool)
    with mlx_stump.prepare_target(
        T,
        m,
        normalize=False,
        T_subseq_isfinite=override,
        T_subseq_isconstant=flags,
    ) as target:
        T[:] = 0.0
        override[:] = False
        flags[:] = True
        for Q in (original[10:42].copy(), original[350:382].copy() * 10.0):
            np.testing.assert_array_equal(
                target.mass(Q),
                mlx_stump.mass(
                    Q,
                    original,
                    normalize=False,
                    T_subseq_isfinite=np.ones(l, dtype=bool),
                    T_subseq_isconstant=np.zeros(l, dtype=bool),
                ),
            )
            np.testing.assert_array_equal(
                target.match(Q, max_distance=np.inf, max_matches=3),
                mlx_stump.match(
                    Q,
                    original,
                    normalize=False,
                    max_distance=np.inf,
                    max_matches=3,
                    T_subseq_isfinite=np.ones(l, dtype=bool),
                    T_subseq_isconstant=np.zeros(l, dtype=bool),
                ),
            )


def test_prepared_callable_target_flags_are_resolved_once():
    T = _walk(256, seed=5)
    m = 24
    l = T.size - m + 1
    flags = np.zeros(l, dtype=bool)
    flags[100] = True
    calls = 0

    def isconstant(a, w):
        nonlocal calls
        calls += 1
        np.testing.assert_array_equal(a, T)
        assert w == m
        return flags

    Q = T[40 : 40 + m].copy()
    reference = mlx_stump.mass(Q, T, T_subseq_isconstant=flags)
    with mlx_stump.prepare_target(T, m, T_subseq_isconstant=isconstant) as target:
        assert calls == 1
        np.testing.assert_array_equal(target.mass(Q), reference)
        np.testing.assert_array_equal(target.mass(Q), reference)
        assert calls == 1


def test_prepared_tiled_target_keeps_bounded_streaming(monkeypatch):
    monkeypatch.setattr(_engine, "_MATMUL_WINDOW_BYTES", 1 << 14)
    monkeypatch.setattr(_engine, "_TILE_WINDOW_BYTES", 1 << 13)
    T = _walk(384, seed=3)
    Q = T[50:82].copy()
    expected = mlx_stump.mass(Q, T)
    with mlx_stump.prepare_target(T, 32) as target:
        assert target._engine.tiled
        assert target._engine.W_T is None
        np.testing.assert_array_equal(target.mass(Q), expected)


def test_prepared_size_and_lifecycle():
    T = _walk(128, seed=4)
    with mlx_stump.prepare_target(T, 16) as target:
        with pytest.raises(ValueError, match="length 16"):
            target.mass(T[:15])
        with pytest.raises(ValueError, match="length 16"):
            target.match(T[:15])
        with pytest.raises(ValueError, match="query_idx"):
            target.mass(T[:16], query_idx=10_000)
    with pytest.raises(RuntimeError, match="closed"):
        target.mass(T[:16])
    with pytest.raises(RuntimeError, match="closed"):
        target.match(T[:16])
    target.close()
