"""Output-contract tests: mparray layout and downstream STUMPY interop."""

from __future__ import annotations

import numpy as np
import stumpy

import mlx_stump

from .conftest import DATASETS


def test_output_layout_k1():
    T = DATASETS["random_walk"](600, seed=50)
    m = 24
    mp = mlx_stump.stump(T, m)
    assert isinstance(mp, mlx_stump.mparray)
    assert mp.dtype == object
    assert mp.shape == (T.shape[0] - m + 1, 4)
    assert isinstance(mp[10, 0], float | np.floating)
    assert isinstance(mp[10, 1], int | np.integer)
    assert mp.P_.dtype == np.float64 and mp.P_.ndim == 1
    assert mp.I_.dtype == np.int64 and mp.I_.ndim == 1
    assert mp.left_I_.dtype == np.int64
    assert mp.right_I_.dtype == np.int64
    # left neighbors precede, right neighbors follow (self-join)
    idx = np.arange(mp.shape[0])
    valid = mp.left_I_ >= 0
    assert np.all(mp.left_I_[valid] < idx[valid])
    valid = mp.right_I_ >= 0
    assert np.all(mp.right_I_[valid] > idx[valid])


def test_output_layout_topk():
    T = DATASETS["random_walk"](600, seed=51)
    m = 24
    k = 3
    mp = mlx_stump.stump(T, m, k=k)
    assert mp.shape == (T.shape[0] - m + 1, 2 * k + 2)
    assert mp.P_.shape == (mp.shape[0], k)
    assert mp.I_.shape == (mp.shape[0], k)
    assert mp.left_I_.ndim == 1 and mp.right_I_.ndim == 1


def test_slicing_preserves_attrs():
    T = DATASETS["white_noise"](400, seed=52)
    mp = mlx_stump.stump(T, 16)
    sliced = mp[10:20]
    assert isinstance(sliced, mlx_stump.mparray)
    assert sliced.P_.shape == (10,)


def test_fluss_consumes_output():
    """The week-1 end-to-end contract: our profile feeds stumpy.fluss unchanged."""
    rng = np.random.default_rng(53)
    # two clearly different regimes
    a = np.sin(np.linspace(0, 20 * np.pi, 1500)) + 0.1 * rng.standard_normal(1500)
    b = 0.5 * rng.standard_normal(1500)
    T = np.concatenate([a, b])
    m = 50
    mp = mlx_stump.stump(T, m)
    ref = stumpy.stump(T, m)
    cac, regimes = stumpy.fluss(mp[:, 1], L=m, n_regimes=2, excl_factor=1)
    cac_ref, regimes_ref = stumpy.fluss(ref[:, 1], L=m, n_regimes=2, excl_factor=1)
    assert cac.shape == cac_ref.shape
    # the detected regime boundary should agree closely with STUMPY's
    assert abs(int(regimes[0]) - int(regimes_ref[0])) <= m


def test_motifs_consumes_output():
    rng = np.random.default_rng(54)
    T = rng.standard_normal(2000)
    m = 50
    pattern = np.sin(np.linspace(0, 4 * np.pi, m)) * 3.0
    T[300 : 300 + m] = pattern + 0.01 * rng.standard_normal(m)
    T[1400 : 1400 + m] = pattern + 0.01 * rng.standard_normal(m)
    mp = mlx_stump.stump(T, m)
    motif_distances, motif_indices = stumpy.motifs(T, mp.P_, max_motifs=1)
    ref = stumpy.stump(T, m)
    ref_distances, ref_indices = stumpy.motifs(T, ref.P_, max_motifs=1)
    assert motif_indices.shape == ref_indices.shape
    assert motif_indices.size > 0
    np.testing.assert_array_equal(np.sort(motif_indices[0]), np.sort(ref_indices[0]))
