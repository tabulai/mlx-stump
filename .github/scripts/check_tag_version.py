"""Refuse to release unless the pushed tag names the packaged version.

Usage::

    python .github/scripts/check_tag_version.py v0.1.0

Exits 0 when the tag is exactly ``"v" + __version__`` (as declared in
``src/mlx_stump/__init__.py``), the version is in canonical PEP 440 form
(what hatchling and PyPI would publish it as — ``0.1.0.DEV0`` or
``0.1.0-rc1`` would be normalized to something the tag does not name), and
it is publishable (no ``.devN`` or ``+local`` segment); exits 1 with an
explanation otherwise. Without this guard any ``v*`` tag would publish
whatever version happens to be in the tree — e.g. tagging ``v0.1.0`` while
the package still says ``0.1.0.dev0``.
"""

from __future__ import annotations

import pathlib
import re
import sys

_INIT = pathlib.Path(__file__).resolve().parents[2] / "src" / "mlx_stump" / "__init__.py"
# PEP 440 canonical form (appendix regex, no local segment): lower-case,
# dotted pre/post/dev segments, no leading "v"
_CANONICAL = re.compile(
    r"^([1-9][0-9]*!)?(0|[1-9][0-9]*)(\.(0|[1-9][0-9]*))*"
    r"((a|b|rc)(0|[1-9][0-9]*))?(\.post(0|[1-9][0-9]*))?(\.dev(0|[1-9][0-9]*))?$"
)


def package_version(init_path: pathlib.Path = _INIT) -> str:
    text = init_path.read_text(encoding="utf-8")
    found = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
    if found is None:
        raise ValueError(f'no `__version__ = "..."` line in {init_path}')
    return found.group(1)


def check(tag: str, version: str) -> str | None:
    """Return None when ``tag`` may publish ``version``, else the reason it may not."""
    if not tag.startswith("v"):
        return f"tag {tag!r} must look like 'v<version>'"
    if _CANONICAL.match(version) is None:
        return (
            f"version {version!r} is not in canonical PEP 440 form (e.g. 0.1.0, 0.1.0rc1, "
            "0.1.0.post1): hatchling/PyPI would normalize it to a version the tag cannot name"
        )
    if tag[1:] != version:
        return (
            f"tag {tag!r} does not match the package version {version!r}: bump "
            "`__version__` in src/mlx_stump/__init__.py (and commit) before tagging"
        )
    if ".dev" in version or "+" in version:
        return (
            f"version {version!r} is a development/local build and must not be "
            "published; release a final or pre-release version (e.g. 0.1.0, 0.1.0rc1)"
        )
    return None


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    version = package_version()
    problem = check(argv[1], version)
    if problem is not None:
        print(f"release refused: {problem}", file=sys.stderr)
        return 1
    print(f"tag {argv[1]} matches mlx_stump {version}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
