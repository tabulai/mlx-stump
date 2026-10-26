"""GPU benchmark provenance and accuracy summaries for missing neighbors."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def bench():
    path = pathlib.Path(__file__).resolve().parents[1] / "bench" / "bench_stump.py"
    if not path.exists():
        pytest.skip("bench/ is not present")
    spec = importlib.util.spec_from_file_location("bench_stump_accuracy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("metal,device", [(False, "cpu"), (True, "cpu"), (False, "gpu")])
def test_benchmark_rejects_inactive_metal_before_computation(bench, monkeypatch, metal, device):
    monkeypatch.setattr(bench.mx.metal, "is_available", lambda: metal)
    monkeypatch.setattr(bench.mx, "default_device", lambda: getattr(bench.mx, device))
    monkeypatch.setattr(sys, "argv", ["bench_stump.py", "--no-stumpy"])

    def unexpected(*args, **kwargs):
        pytest.fail("Benchmark started computing without an active Metal GPU")

    monkeypatch.setattr(bench.mlx_stump, "stump", unexpected)
    with pytest.raises(RuntimeError, match="active Metal GPU"):
        bench.main()


def test_benchmark_provenance_records_device(bench, monkeypatch):
    monkeypatch.setattr(bench.mx.metal, "is_available", lambda: False)
    monkeypatch.setattr(bench.mx, "default_device", lambda: bench.mx.cpu)
    provenance = bench.provenance()
    assert f"device {bench.mx.cpu}" in provenance
    assert "metal False" in provenance


@pytest.mark.parametrize(
    "ours,reference,error,nonfinite_agree",
    [
        ([1.0, 2.5], [1.0, 2.0], 0.5, True),
        ([np.inf, 2.0], [np.inf, 3.0], 1.0, True),
        ([np.inf], [np.inf], None, True),
        ([np.inf, 2.0], [1.0, 2.0], 0.0, False),
        ([np.nan], [np.nan], None, False),
        ([np.nan], [np.inf], None, False),
        ([-np.inf], [np.inf], None, False),
    ],
)
def test_benchmark_accuracy_handles_nonfinite_profiles(
    bench, ours, reference, error, nonfinite_agree
):
    with np.errstate(invalid="raise"):
        assert bench._profile_error(ours, reference) == (error, nonfinite_agree)


@pytest.mark.parametrize(
    "profile,expected",
    [
        ([np.inf], "| n/a | yes | 100.00% | n/a (0 valid) |"),
        ([1.0, 2.0, np.inf], "| 0.00e+00 | yes | 100.00% | 1/1 |"),
    ],
)
def test_benchmark_table_reports_available_comparisons(
    bench, monkeypatch, capsys, profile, expected
):
    result = SimpleNamespace(P_=np.array(profile), I_=np.zeros(len(profile), dtype=int))
    monkeypatch.setattr(bench.mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(bench.mx, "default_device", lambda: bench.mx.gpu)
    monkeypatch.setattr(bench.mlx_stump, "stump", lambda *args, **kwargs: result)
    monkeypatch.setitem(
        sys.modules,
        "stumpy",
        SimpleNamespace(stump=lambda *args, **kwargs: result, __version__="test"),
    )
    monkeypatch.setitem(sys.modules, "numba", SimpleNamespace(get_num_threads=lambda: 1))
    monkeypatch.setattr(bench.os, "cpu_count", lambda: 1)
    monkeypatch.setattr(
        sys, "argv", ["bench_stump.py", "--sizes", "64", "--m", "64", "--repeat", "1"]
    )
    with np.errstate(invalid="raise"):
        bench.main()
    assert expected in capsys.readouterr().out
