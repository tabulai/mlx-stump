"""Sixth review round: test infrastructure, CI, packaging and the benchmark.

1. ``tests/conftest.py`` ran ``pytest.importorskip("stumpy")`` at import
   time. A skip escaping an initial conftest is a hard error, so without the
   optional oracle the whole session crashed (exit 1), taking down the
   modules that never use STUMPY. Each module that needs the oracle now
   skips itself after its imports.
2. The golden tie oracle ``_znorm_dist`` used the one-pass
   ``mean(a*b) - mean(a)*mean(b)`` covariance, which cancels on
   ``large_offset`` (random walk + 1e6) to errors of up to ~1.3e-2 in
   distance at m=8, about 40% of the tie tolerance it adjudicates. It is now
   the two-pass z-difference form, pinned here against exact rational
   arithmetic (the old form misses that bound by six orders of magnitude).
   It also detects constant windows by STUMPY's ptp == 0 rule: ``std == 0``
   missed a constant 0.1 window (std ~1e-17) and z-normalized it.
3. The fresh-interpreter RSS harness existed in five copies. It is now one
   ``conftest.run_isolated`` whose prologue and body are dedented
   separately (a column-0 body broke the old single-dedent template).
4. Lint hygiene: stale ``noqa`` codes, unused unpacked names, bare
   ``pytest.raises(ValueError)`` and an unescaped ``match`` pattern. Ruff's
   RUF100 now keeps stale ``noqa`` comments out.
5. CI could not show that MLX ran on Metal (a CPU fallback passes
   silently), linted with a floating ruff in every matrix job, had no job
   timeouts and never resolved the declared dependency floors. CI now
   reports the device, sets ``MLX_STUMP_REQUIRE_METAL=1`` (the conftest
   gate below fails a non-GPU run), lints once with the locked ruff, bounds
   both jobs, adds a lowest-dependency job pinned to the exact floor
   releases (``==1.24`` is 1.24.0; ``==1.24.*`` resolved to 1.24.4) and
   runs ``--strict-markers``.
6. The benchmark's provenance line reported the git state of the current
   directory rather than of the imported package, so a stale install was
   labelled with a clean commit; each size also ran one extra untimed call
   per library. Provenance now asks the imported package's own tree (and
   trusts it only if the file is tracked there), and ``_time`` returns the
   last result for the precision columns.
7. The distributions shipped no ``py.typed`` marker.
8. ``estimated_peak_bytes``, the documented memory estimator, was reachable
   only through the private ``mlx_stump._engine``.
"""

from __future__ import annotations

import importlib.resources
import importlib.util
import math
import os
import pathlib
import re
import shutil
import subprocess
import sys
from decimal import Decimal, localcontext
from fractions import Fraction

import mlx.core as mx
import numpy as np
import pytest

import mlx_stump
import mlx_stump._engine as eng

from . import conftest
from .conftest import DATASETS, _znorm_dist, run_isolated

_REPO = pathlib.Path(__file__).resolve().parents[1]
_CI_WORKFLOW = _REPO / ".github/workflows/ci.yml"
_BENCH = _REPO / "bench/bench_stump.py"


# ------------------------------------------------ 1: STUMPY is optional
def test_suite_runs_without_stumpy(tmp_path):
    """Shadow STUMPY with a module that fails to import: the oracle modules
    skip, the rest still runs, and the session exits 0."""
    (tmp_path / "stumpy.py").write_text(
        'raise ModuleNotFoundError("No module named \'stumpy\'", name="stumpy")\n'
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(tmp_path), env.get("PYTHONPATH")]))

    def pytest_run(*args):
        return subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider", *args],
            cwd=_REPO,
            env=env,
            capture_output=True,
            text=True,
        )

    # one oracle module (skipped) and one oracle-free test (runs)
    args = ["tests/test_output_format.py", "tests/test_sixth_review_infra.py"]
    out = pytest_run(*args, "-k", "not without_stumpy and (layout or estimator)")
    assert out.returncode == 0, out.stdout + out.stderr
    assert "could not import 'stumpy'" in out.stdout
    assert re.search(r"\b1 passed, 1 skipped\b", out.stdout), out.stdout
    # every module and the conftest still collect: a bare `import stumpy`
    # anywhere is a collection error (exit 2)
    out = pytest_run("--collect-only", "tests")
    assert out.returncode == 0, out.stdout + out.stderr
    assert "ERROR collecting" not in out.stdout and "could not import 'stumpy'" in out.stdout


# ---------------------------------------------------- 2: exact tie oracle
def _exact_znorm_dist(a, b):
    """z-normalized distance from exact rational moments (60-digit sqrt)."""
    m = len(a)
    fa, fb = [Fraction(x) for x in a], [Fraction(x) for x in b]
    ma, mb = sum(fa) / m, sum(fb) / m
    ca, cb = [x - ma for x in fa], [x - mb for x in fb]
    saa = sum(x * x for x in ca)
    sbb = sum(x * x for x in cb)
    sab = sum(x * y for x, y in zip(ca, cb, strict=True))

    def dec(q):
        return Decimal(q.numerator) / Decimal(q.denominator)

    with localcontext() as ctx:
        ctx.prec = 60
        rho = dec(sab) / (dec(saa) * dec(sbb)).sqrt()
        return float((2 * m * (1 - rho)).sqrt())


@pytest.mark.parametrize("m", [8, 64])
def test_tie_oracle_is_exact_at_a_large_offset(m):
    """Pairs outside the exclusion zone, including each row's nearest
    neighbour (the pairs the tie-tolerant helper adjudicates): the old
    one-pass form was off by up to ~1.3e-2 on this series at m=8."""
    T = DATASETS["large_offset"](2000, seed=1)
    rng = np.random.default_rng(m)
    excl = math.ceil(m / 4)
    i = rng.integers(0, T.size - m + 1, 60)
    j = rng.integers(0, T.size - m + 1, 60)
    far = np.abs(i - j) > excl
    rows = rng.integers(0, T.size - m + 1, 20)
    nn = np.asarray(mlx_stump.stump(T, m).I_)[rows]
    pairs = list(zip(i[far], j[far], strict=True)) + list(zip(rows, nn, strict=True))
    err = [
        abs(_znorm_dist(T, T, m, a, b) - _exact_znorm_dist(T[a : a + m], T[b : b + m]))
        for a, b in pairs
    ]
    assert max(err) <= 1e-9, max(err)


def test_tie_oracle_uses_stumpys_constant_rule():
    """Constants whose float64 mean is inexact: STUMPY calls two constant
    windows 0 apart and a constant vs a varying window sqrt(m) apart."""
    m = 3
    T = np.r_[np.full(10, 0.1), np.random.default_rng(0).standard_normal(20), np.full(10, 0.7)]
    assert T[:m].std() > 0.0 and T[-m:].std() > 0.0  # the trap std == 0 fell into
    assert _znorm_dist(T, T, m, 0, 31) == 0.0
    assert _znorm_dist(T, T, m, 0, 15) == pytest.approx(np.sqrt(m), rel=1e-15)
    assert _znorm_dist(T, T, m, 15, 31) == pytest.approx(np.sqrt(m), rel=1e-15)


# ------------------------------------------- 3: one isolated-memory harness
def test_run_isolated_accepts_bodies_at_any_indentation():
    indented = """
        x = np.ones(1 << 20)
        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * _unit
        y = x * 2.0
    """
    column0 = "x = np.ones(1 << 20)\nassert mlx_stump.__version__ and mx.metal is not None\n"
    for body in (indented, column0):
        before, peak, mlx_peak = run_isolated(body)
        assert 0 < before <= peak and mlx_peak >= 0


# --------------------------------------------------------------- 5: CI gate
def test_metal_gate_fails_a_session_off_the_gpu(monkeypatch):
    monkeypatch.delenv("MLX_STUMP_REQUIRE_METAL", raising=False)
    monkeypatch.setattr(mx, "default_device", lambda: mx.Device(mx.cpu, 0))
    conftest.pytest_sessionstart(None)  # opt-in: nothing happens without the variable
    monkeypatch.setenv("MLX_STUMP_REQUIRE_METAL", "1")
    with pytest.raises(pytest.exit.Exception, match="default_device=Device\\(cpu") as exc:
        conftest.pytest_sessionstart(None)
    assert exc.value.returncode == 1
    monkeypatch.setattr(mx, "default_device", lambda: mx.Device(mx.gpu, 0))
    monkeypatch.setattr(mx.metal, "is_available", lambda: False)
    with pytest.raises(pytest.exit.Exception, match="metal=False") as exc:
        conftest.pytest_sessionstart(None)
    assert exc.value.returncode == 1
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    conftest.pytest_sessionstart(None)


def test_unknown_markers_are_errors(request):
    assert request.config.getoption("strict_markers")


@pytest.mark.skipif(not _CI_WORKFLOW.exists(), reason="workflow files are not shipped in the sdist")
def test_ci_lowest_dependency_job_pins_the_declared_floors():
    pyproject = (_REPO / "pyproject.toml").read_text(encoding="utf-8")
    ci = _CI_WORKFLOW.read_text(encoding="utf-8")
    floors = dict(re.findall(r'"(mlx|numpy|stumpy)>=([0-9.]+)"', pyproject))
    assert set(floors) == {"mlx", "numpy", "stumpy"}
    for name, floor in floors.items():
        # exactly the floor release: PEP 440 zero-pads "==0.30" to 0.30.0,
        # while "==0.30.*" would install the newest 0.30.x patch (the
        # closing quote rejects that form)
        assert f'"{name}=={floor}"' in ci, name
    python = re.search(r'requires-python = ">=([0-9.]+)"', pyproject).group(1)
    include = ci[ci.index("include:") : ci.index("runs-on:")]
    assert f'python-version: "{python}"' in include and "deps: lowest" in include


# ------------------------------------------------------ 6: bench provenance
@pytest.fixture
def bench():
    if not _BENCH.exists():
        pytest.skip("bench/ is not present")
    spec = importlib.util.spec_from_file_location("bench_stump", _BENCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_package(root):
    init = root / "mlx_stump" / "__init__.py"
    init.parent.mkdir(parents=True)
    init.write_text("")
    return type(mlx_stump)("mlx_stump"), init


def test_bench_provenance_does_not_vouch_for_an_installed_build(bench, tmp_path, monkeypatch):
    fake, init = _fake_package(tmp_path / "site-packages")
    fake.__file__, fake.__version__ = str(init), "9.9.9"
    monkeypatch.setattr(bench, "mlx_stump", fake)
    assert "mlx-stump 9.9.9 @ installed build, commit unknown, " in bench.provenance()
    if shutil.which("git"):
        # an untracked copy inside a work tree (a `pip install .` into the
        # checkout's gitignored .venv) is not that tree's HEAD either
        repo = tmp_path / "checkout"
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        fake.__file__ = str(_fake_package(repo / ".venv")[1])
        assert "@ installed build, commit unknown, " in bench.provenance()


def test_bench_provenance_names_the_imported_checkout(bench):
    module = pathlib.Path(mlx_stump.__file__).resolve()
    if not ((_REPO / ".git").exists() and module.is_relative_to(_REPO / "src")):
        pytest.skip("mlx_stump is not imported from this git checkout")
    assert re.search(r"mlx-stump \S+ @ [0-9a-f]{7,}(\+dirty)?, ", bench.provenance())


def test_bench_timer_returns_the_last_result(bench):
    calls = []

    def fn():
        calls.append(len(calls))
        return len(calls)

    best, out = bench._time(fn, 3)
    assert calls == [0, 1, 2] and out == 3 and 0 <= best < math.inf


# ----------------------------------------------------- 7-8: packaging, API
def test_package_ships_the_typing_marker():
    assert importlib.resources.files("mlx_stump").joinpath("py.typed").is_file()


def test_public_estimator_is_the_engine_estimator():
    assert mlx_stump.estimated_peak_bytes is eng.estimated_peak_bytes
    assert "estimated_peak_bytes" in mlx_stump.__all__
