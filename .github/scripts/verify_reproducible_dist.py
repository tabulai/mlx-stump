"""Require two release builds to contain the same wheel/sdist bytes."""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import shutil


def _artifacts(directory: pathlib.Path) -> dict[str, pathlib.Path]:
    files = {path.name: path for path in directory.iterdir() if path.is_file()}
    wheels = [name for name in files if name.endswith(".whl")]
    sdists = [name for name in files if name.endswith(".tar.gz")]
    if len(files) != 2 or len(wheels) != 1 or len(sdists) != 1:
        raise SystemExit(
            f"release refused: {directory} must contain exactly one wheel and one sdist; "
            f"found {sorted(files)}"
        )
    return files


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("first", type=pathlib.Path)
    parser.add_argument("second", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    args = parser.parse_args()

    first = _artifacts(args.first)
    second = _artifacts(args.second)
    if first.keys() != second.keys():
        raise SystemExit(
            "release refused: independent builds produced different filenames: "
            f"{sorted(first)} != {sorted(second)}"
        )
    for name in sorted(first):
        left, right = _sha256(first[name]), _sha256(second[name])
        if left != right:
            raise SystemExit(
                f"release refused: independent builds differ for {name}: {left} != {right}"
            )

    args.output.mkdir(parents=True, exist_ok=False)
    for name, source in first.items():
        shutil.copy2(source, args.output / name)
        print(f"verified reproducible {name}: {_sha256(source)}")


if __name__ == "__main__":
    main()
