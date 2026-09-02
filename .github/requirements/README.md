# Release dependency locks

Generate these locks on macOS arm64 with CPython 3.12 and
`pip-tools==7.6.1`:

```bash
python -m pip install pip-tools==7.6.1

python -m piptools compile \
  --resolver=backtracking \
  --generate-hashes \
  --allow-unsafe \
  --strip-extras \
  --no-emit-index-url \
  --output-file=.github/requirements/release-build.txt \
  .github/requirements/release-build.in

python -m piptools compile \
  --resolver=backtracking \
  --generate-hashes \
  --allow-unsafe \
  --strip-extras \
  --no-emit-index-url \
  --output-file=.github/requirements/release-test.txt \
  .github/requirements/release-test.in
```

`pip-tools` 7.6.1 incorrectly adds `--no-index` to its generated command
comment when `--no-emit-index-url` is used, even though resolution used the
package index. The two command comments are corrected after generation so
they remain runnable on a clean machine. Review every regenerated lock diff
and verify both files install with `--require-hashes --only-binary=:all:`.
