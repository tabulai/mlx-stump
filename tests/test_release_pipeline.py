"""Static security checks for the release pipeline.

The workflow is intentionally absent from sdists, so this module skips there.
"""

from __future__ import annotations

import pathlib
import re

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[1]
_WORKFLOW = _REPO / ".github/workflows/publish-pypi.yml"
_HISTORICAL_WORKFLOW = _REPO / ".github/workflows/release.yml"
_CI_WORKFLOW = _REPO / ".github/workflows/ci.yml"
_BUILD_LOCK = _REPO / ".github/requirements/release-build.txt"
_TEST_LOCK = _REPO / ".github/requirements/release-test.txt"
_LOCK_README = _REPO / ".github/requirements/README.md"


@pytest.mark.skipif(not _WORKFLOW.exists(), reason="workflow files are not shipped in the sdist")
def test_release_uses_a_fresh_identity_and_separates_permissions():
    text = _WORKFLOW.read_text(encoding="utf-8")
    assert not _HISTORICAL_WORKFLOW.exists()
    assert "environment: pypi-release-v2" in text
    assert "concurrency:" in text and "cancel-in-progress: false" in text

    prepare_job = text[text.index("  prepare_publish:") : text.index("  tag:")]
    tag_job = text[text.index("  tag:") : text.index("  publish:")]
    publish_job = text[text.index("  publish:") :]
    assert "permissions: {}" in prepare_job
    assert "id-token: write" not in prepare_job
    assert "environment: pypi-release-v2" in tag_job
    assert "contents: read" in tag_job
    assert "contents: write" not in tag_job
    assert "id-token: write" not in tag_job
    assert "RELEASE_TAG_DEPLOY_KEY" in tag_job
    assert "id-token: write" in publish_job
    assert "contents: write" not in publish_job
    assert "actions/checkout" not in publish_job
    assert "run:" not in publish_job
    assert "needs: [prepare_publish, tag]" in publish_job
    assert text.index("  tag:") < text.index("  publish:")


@pytest.mark.skipif(not _WORKFLOW.exists(), reason="workflow files are not shipped in the sdist")
def test_release_tag_is_idempotent_only_for_the_verified_commit():
    text = _WORKFLOW.read_text(encoding="utf-8")
    assert 'git cat-file -t "refs/tags/${tag}"' in text
    assert 'git rev-parse "refs/tags/${tag}^{commit}"' in text
    assert '"${target}" != "${GITHUB_SHA}"' in text
    assert 'git tag --annotate "${tag}" "${GITHUB_SHA}"' in text
    assert '"git@github.com:${GITHUB_REPOSITORY}.git" "refs/tags/${tag}"' in text
    assert "GIT_SSH_COMMAND" in text
    assert 'git merge-base --is-ancestor "${GITHUB_SHA}"' in text
    assert "verify_remote_tag()" in text
    assert '"https://github.com/${GITHUB_REPOSITORY}.git"' in text
    assert '"refs/tags/${tag}:refs/tags/${tag}"' in text
    push_success = text[text.index('if GIT_SSH_COMMAND="${ssh_command}" git push') :]
    assert push_success.index("verify_remote_tag") < push_success.index("exit 0")
    existing = text[text.index('if git show-ref --verify --quiet "refs/tags/${tag}"') :]
    assert existing.index("verify_remote_tag") < existing.index("exit 0")


@pytest.mark.skipif(not _WORKFLOW.exists(), reason="workflow files are not shipped in the sdist")
def test_partial_pypi_retry_skips_only_matching_artifacts():
    text = _WORKFLOW.read_text(encoding="utf-8")
    start = text.index("Check for an exact partial")
    preflight = text[start : text.index("gh-action-pypi-publish")]
    assert "hashlib.sha256" in preflight
    assert "hash_mismatch" in preflight
    assert "unexpected=" in preflight
    assert "publish_needed=false" in preflight
    assert "local.keys() - remote.keys()" in preflight
    assert "shutil.copy2" in preflight
    assert "skip-existing" not in text
    assert "packages-dir: dist-pending/" in text
    # Recovery reruns every job so PyPI state is queried again. Artifact v4
    # names are immutable within one run unless the verified rebuild opts in
    # to replacing them.
    assert text.count("overwrite: true") == 2


@pytest.mark.skipif(not _WORKFLOW.exists(), reason="workflow files are not shipped in the sdist")
def test_all_workflow_actions_are_pinned_to_full_commit_shas():
    workflows = list((_REPO / ".github/workflows").glob("*.yml"))
    action = re.compile(r"^\s*- uses:\s+[^@\s]+@([^\s#]+)", re.MULTILINE)
    for workflow in workflows:
        refs = action.findall(workflow.read_text(encoding="utf-8"))
        assert refs, workflow
        assert all(re.fullmatch(r"[0-9a-f]{40}", ref) for ref in refs), (workflow, refs)


@pytest.mark.skipif(not _WORKFLOW.exists(), reason="workflow files are not shipped in the sdist")
def test_release_build_is_locked_reproducible_and_tests_both_artifacts():
    text = _WORKFLOW.read_text(encoding="utf-8")
    assert "actions: read" in text
    assert "Require successful CI for this exact commit" in text
    assert '"head_sha": sha' in text
    assert 'run.get("head_branch") == "main"' in text
    assert 'run.get("conclusion") == "success"' in text
    assert text.index("actions/setup-python@") < text.index(
        "Require successful CI for this exact commit"
    )
    assert 'python-version: "3.12.10"' in text
    assert "architecture: arm64" in text
    assert ".github/requirements/release-build.txt" in text
    assert text.count(".github/requirements/release-test.txt") >= 2
    assert text.count("--require-hashes --only-binary=:all:") >= 3
    assert "SOURCE_DATE_EPOCH" in text and 'git show -s --format=%ct "${GITHUB_SHA}"' in text
    assert text.count('git archive "${GITHUB_SHA}"') == 2
    assert "verify_reproducible_dist.py" in text
    assert "python -m build --no-isolation" in text
    assert "Build, install, and test from the sdist" in text
    assert "--no-deps --no-build-isolation" in text
    assert text.count("-m pip check") >= 2
    assert text.count('"site-packages" in module.as_posix()') >= 2
    for floating in (
        "pip install build",
        "pip install --upgrade pip",
        '"pytest>=',
        '"stumpy>=',
        '"ruff>=',
    ):
        assert floating not in text


@pytest.mark.skipif(not _CI_WORKFLOW.exists(), reason="workflow files are not shipped in the sdist")
def test_ci_exercises_the_locked_reproducible_artifact_path():
    text = _CI_WORKFLOW.read_text(encoding="utf-8")
    build = text[text.index("  build:") :]
    assert 'python-version: "3.12.10"' in build
    assert ".github/requirements/release-build.txt" in build
    assert build.count(".github/requirements/release-test.txt") >= 2
    assert "verify_reproducible_dist.py" in build
    assert "--no-isolation" in build
    assert "--no-deps --no-build-isolation" in build


@pytest.mark.skipif(not _WORKFLOW.exists(), reason="workflow files are not shipped in the sdist")
def test_release_locks_are_exact_and_hash_complete():
    assert _BUILD_LOCK.exists(), _BUILD_LOCK
    assert _TEST_LOCK.exists(), _TEST_LOCK
    assert _LOCK_README.exists(), _LOCK_README
    for path in (_BUILD_LOCK, _TEST_LOCK):
        text = path.read_text(encoding="utf-8")
        assert "--no-index" not in "\n".join(text.splitlines()[:7])
        starts = [
            match
            for match in re.finditer(
                r"(?m)^([A-Za-z0-9][A-Za-z0-9._-]*==[^\\\s]+) \\$", text
            )
        ]
        assert starts, path
        for idx, match in enumerate(starts):
            end = starts[idx + 1].start() if idx + 1 < len(starts) else len(text)
            block = text[match.start() : end]
            assert "--hash=sha256:" in block, (path, match.group(1))
        requirement_lines = [
            line
            for line in text.splitlines()
            if line and not line[0].isspace() and not line.startswith("#")
        ]
        assert len(requirement_lines) == len(starts)
        assert all("==" in line and line.endswith("\\") for line in requirement_lines)
