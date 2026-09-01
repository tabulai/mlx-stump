"""The release workflow refuses tags that do not name the packaged version."""

from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import sys

import pytest

import mlx_stump

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / ".github/scripts/check_tag_version.py"
pytestmark = pytest.mark.skipif(
    not _SCRIPT.exists(), reason="release guard script not present (tests run outside the repo)"
)


@pytest.fixture(scope="module")
def guard():
    spec = importlib.util.spec_from_file_location("check_tag_version", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reads_the_package_version(guard):
    assert guard.package_version() == mlx_stump.__version__


@pytest.mark.parametrize(
    "tag,version,fragment",
    [
        ("v0.1.0", "0.1.0", None),
        ("v0.1.0rc1", "0.1.0rc1", None),
        ("v1.2.3", "1.2.3", None),
        ("v0.1.0.post1", "0.1.0.post1", None),
        ("v0.1.0a1", "0.1.0a1", None),
        ("v1!0.1.0", "1!0.1.0", None),
        ("v0.1.0", "0.1.0.dev0", "does not match"),
        ("v0.1.0.dev0", "0.1.0.dev0", "development"),
        ("v0.1.0+local", "0.1.0+local", "canonical"),
        ("v0.2.0", "0.1.0", "does not match"),
        ("0.1.0", "0.1.0", "must look like"),
        ("v0.1.0 ", "0.1.0", "does not match"),
        # non-canonical spellings would be normalized by hatchling/PyPI to a
        # version the tag does not name (0.1.0.DEV0 -> 0.1.0.dev0 reached PyPI)
        ("v0.1.0.DEV0", "0.1.0.DEV0", "canonical"),
        ("v0.1.0.Dev1", "0.1.0.Dev1", "canonical"),
        ("v0.1.0-rc1", "0.1.0-rc1", "canonical"),
        ("v0.1.0RC1", "0.1.0RC1", "canonical"),
        ("v0.1.0dev0", "0.1.0dev0", "canonical"),
        ("vv0.1.0", "v0.1.0", "canonical"),
        ("v01.1.0", "01.1.0", "canonical"),
    ],
)
def test_check(guard, tag, version, fragment):
    problem = guard.check(tag, version)
    if fragment is None:
        assert problem is None
    else:
        assert problem is not None and fragment in problem


def test_cli_refuses_mismatched_tag():
    # whatever the tree's version is, this tag cannot match it
    out = subprocess.run(
        [sys.executable, str(_SCRIPT), "v999.999.999"], capture_output=True, text=True
    )
    assert out.returncode == 1
    assert "release refused" in out.stderr


def test_cli_accepts_the_matching_tag_only_for_publishable_versions():
    version = mlx_stump.__version__
    out = subprocess.run(
        [sys.executable, str(_SCRIPT), f"v{version}"], capture_output=True, text=True
    )
    if ".dev" in version or "+" in version:
        assert out.returncode == 1 and "development" in out.stderr
    else:
        assert out.returncode == 0, out.stderr
