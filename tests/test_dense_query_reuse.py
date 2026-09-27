"""Dense self-joins reuse the target's packed windows as query batches."""

from __future__ import annotations

import importlib

import mlx.core as mx
import numpy as np
import pytest

import mlx_stump
from mlx_stump._engine import MassEngine, query_windows
from mlx_stump._preprocess import preprocess_series

stump_mod = importlib.import_module("mlx_stump._stump")


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("with_missing", [False, True])
def test_packed_dense_rows_equal_fresh_query_rows(normalize, with_missing):
    rng = np.random.default_rng(83)
    T = rng.standard_normal(291).cumsum().astype(np.float64)
    T[55:92] = 17.0  # constant windows use special normalized rows
    T[151:183] = T[11:43]  # exact copies keep their tie behavior
    if with_missing:
        T[137] = np.nan
        T[219] = np.inf
    m = 32
    prep = preprocess_series(T, m, normalize=normalize)
    try:
        engine = MassEngine(prep, normalize=normalize)
        assert not engine.tiled
        for start, stop in ((0, 43), (71, 108), (prep.l - 37, prep.l)):
            packed = np.array(engine.W_T.T[start:stop])
            fresh = np.array(query_windows(prep, start, stop, normalize=normalize))
            np.testing.assert_array_equal(packed, fresh)
    finally:
        engine = None
        prep.release_device()
        mx.clear_cache()


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("k", [1, 3])
def test_dense_self_join_result_equals_fresh_query_path(monkeypatch, normalize, k):
    rng = np.random.default_rng(84)
    T = rng.standard_normal(257).cumsum().astype(np.float64)
    T[85:117] = T[12:44]
    T[163] = np.nan
    kwargs = {"normalize": normalize, "k": k, "chunk_size": 37}

    got = mlx_stump.stump(T, 32, **kwargs)
    original = stump_mod._query_args

    def fresh_queries(batches, query, mode, packed_target=None):
        return original(batches, query, mode)

    monkeypatch.setattr(stump_mod, "_query_args", fresh_queries)
    ref = mlx_stump.stump(T, 32, **kwargs)
    np.testing.assert_array_equal(
        np.asarray(got, dtype=np.float64), np.asarray(ref, dtype=np.float64)
    )


def test_dense_implicit_self_join_never_repacks_queries(monkeypatch):
    T = np.random.default_rng(85).standard_normal(113).astype(np.float64)

    def unexpected(*args, **kwargs):
        raise AssertionError("dense self-join repacked a query batch")

    monkeypatch.setattr(stump_mod, "query_windows", unexpected)
    mlx_stump.stump(T, 16, chunk_size=19)
