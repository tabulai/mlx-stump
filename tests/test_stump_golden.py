"""Golden tests: mlx_stump.stump vs float64 stumpy.stump, self-joins."""

from __future__ import annotations

import numpy as np
import pytest
import stumpy

import mlx_stump

from .conftest import (
    DATASETS,
    assert_indices_tie_tolerant,
    assert_profile_close,
    tie_tolerance,
)


def golden_self_join(T, m, **kwargs):
    mp = mlx_stump.stump(T, m, **kwargs)
    ref = stumpy.stump(T, m, **kwargs)
    tie = tie_tolerance(m)

    assert mp.shape == ref.shape
    assert mp.dtype == object

    agree = mp.I_ == ref.I_
    assert_indices_tie_tolerant(mp.I_, ref.I_, T, T, m, tie_atol=tie)
    assert_profile_close(mp.P_, ref.P_, m=m, exact_mask=agree, tie_atol=tie)
    assert_indices_tie_tolerant(mp.left_I_, ref.left_I_, T, T, m, tie_atol=tie)
    assert_indices_tie_tolerant(mp.right_I_, ref.right_I_, T, T, m, tie_atol=tie)

    # what users actually consume: the best motif and the top discord
    finite = np.isfinite(ref.P_)
    if finite.sum() > 10:
        assert abs(mp.P_[np.argmin(mp.P_)] - ref.P_[np.argmin(ref.P_)]) <= tie
        p_ours = np.where(finite, mp.P_, -np.inf)
        p_ref = np.where(finite, ref.P_, -np.inf)
        assert abs(p_ours.max() - p_ref.max()) <= tie
    return mp, ref


@pytest.mark.parametrize("name", sorted(DATASETS))
@pytest.mark.parametrize("m", [8, 64])
def test_datasets(name, m):
    T = DATASETS[name](2000, seed=1)
    golden_self_join(T, m)


@pytest.mark.parametrize("m", [3, 101])
def test_window_extremes(m):
    T = DATASETS["random_walk"](1500, seed=2)
    golden_self_join(T, m)


def test_multi_chunk():
    # force several GPU chunks to cover the chunk-boundary logic
    T = DATASETS["sine_noise"](3000, seed=3)
    m = 32
    mp = mlx_stump.stump(T, m, chunk_size=173)
    ref = mlx_stump.stump(T, m)
    np.testing.assert_array_equal(mp.I_, ref.I_)
    np.testing.assert_allclose(mp.P_, ref.P_, atol=0, rtol=0)


def test_minimal_length_self_join():
    # n == m: a single subsequence, everything inside the exclusion zone
    T = DATASETS["white_noise"](64, seed=4)
    mp = mlx_stump.stump(T, 64)
    ref = stumpy.stump(T, 64)
    assert mp.shape == ref.shape == (1, 4)
    assert np.isinf(mp[0, 0]) and np.isinf(ref[0, 0])
    assert list(mp[0])[1:] == list(ref[0])[1:] == [-1, -1, -1]


def test_all_constant_series():
    T = np.full(500, 7.0)
    mp, ref = golden_self_join(T, 25)
    assert np.all(mp.P_ == 0.0)


def test_all_nan_series():
    T = np.full(300, np.nan)
    mp = mlx_stump.stump(T, 20)
    ref = stumpy.stump(T, 20)
    np.testing.assert_array_equal(np.isinf(mp.P_), np.isinf(ref.P_))
    np.testing.assert_array_equal(mp.I_, ref.I_)


def test_isconstant_override():
    T = DATASETS["white_noise"](800, seed=5)
    m = 16
    l = T.shape[0] - m + 1
    isconst = np.zeros(l, dtype=bool)
    isconst[100:140] = True
    mp = mlx_stump.stump(T, m, T_A_subseq_isconstant=isconst)
    ref = stumpy.stump(T, m, T_A_subseq_isconstant=isconst)
    tie = tie_tolerance(m)
    np.testing.assert_array_equal(np.isinf(mp.P_), np.isinf(ref.P_))
    assert np.mean(mp.I_ == ref.I_) >= 0.95
    assert_profile_close(mp.P_, ref.P_, m=m, exact_mask=(mp.I_ == ref.I_), tie_atol=tie)


def test_validation_matches_stumpy():
    T = np.random.default_rng(0).standard_normal(100)
    with pytest.raises(TypeError):
        mlx_stump.stump(T.astype(np.float32), 8)
    with pytest.raises(TypeError):
        mlx_stump.stump(np.arange(100), 8)
    with pytest.raises(ValueError):
        mlx_stump.stump(T.reshape(10, 10), 8)
    with pytest.raises(ValueError):
        mlx_stump.stump(T, 2)
    with pytest.raises(ValueError):
        mlx_stump.stump(T, 101)
    with pytest.raises(ValueError):
        mlx_stump.stump(T, 8, k=0)


def _flatline(n, seed=0):
    # near-constant segment at an offset: the case the old FFT fallback got
    # catastrophically wrong (it could not center target windows per-window)
    rng = np.random.default_rng(seed)
    T = rng.standard_normal(n)
    T[n // 2 - 100 : n // 2 + 100] = 5.0 + 1e-9 * rng.standard_normal(200)
    return T


@pytest.mark.parametrize("maker", [lambda n: DATASETS["random_walk"](n, seed=6), _flatline])
@pytest.mark.parametrize("k", [1, 3])
def test_tiled_engine_matches_single_block(monkeypatch, maker, k):
    """Large joins stream the target window matrix as column blocks; the
    tiled sweep applies the identical doubly-centered float32 arithmetic, so
    its output is bit-equal to the single-block engine — including on
    near-constant data, where the FFT fallback it replaced was badly wrong."""
    import mlx_stump._engine as eng
    from mlx_stump._preprocess import preprocess_series

    T = maker(2500)
    m = 64
    ref = mlx_stump.stump(T, m, k=k)
    monkeypatch.setattr(eng, "_MATMUL_WINDOW_BYTES", 0)
    monkeypatch.setattr(eng, "_TILE_WINDOW_BYTES", 64 * 1024)  # force ~10 blocks
    assert eng.MassEngine(preprocess_series(T, m)).tiled
    mp = mlx_stump.stump(T, m, k=k)
    assert np.array_equal(np.asarray(mp[:, :k], dtype=float), np.asarray(ref[:, :k], dtype=float))
    assert np.array_equal(
        np.asarray(mp[:, k : 2 * k], dtype=int), np.asarray(ref[:, k : 2 * k], dtype=int)
    )
    np.testing.assert_array_equal(mp.left_I_, ref.left_I_)
    np.testing.assert_array_equal(mp.right_I_, ref.right_I_)


def test_tiled_engine_multi_query_chunks(monkeypatch):
    # tiled target blocks combined with several query chunks (merge state
    # crosses both loop boundaries), AB-join included
    import mlx_stump._engine as eng

    T_A = _flatline(1200, seed=8)
    T_B = DATASETS["sine_noise"](900, seed=9)
    ref_self = mlx_stump.stump(T_A, 32, chunk_size=97)
    ref_ab = mlx_stump.stump(T_A, 32, T_B, ignore_trivial=False, chunk_size=97)
    monkeypatch.setattr(eng, "_MATMUL_WINDOW_BYTES", 0)
    monkeypatch.setattr(eng, "_TILE_WINDOW_BYTES", 32 * 1024)
    mp_self = mlx_stump.stump(T_A, 32, chunk_size=97)
    mp_ab = mlx_stump.stump(T_A, 32, T_B, ignore_trivial=False, chunk_size=97)
    for mp, ref in ((mp_self, ref_self), (mp_ab, ref_ab)):
        np.testing.assert_array_equal(mp.I_, ref.I_)
        np.testing.assert_allclose(mp.P_, ref.P_, atol=0, rtol=0)
        np.testing.assert_array_equal(mp.left_I_, ref.left_I_)
        np.testing.assert_array_equal(mp.right_I_, ref.right_I_)
