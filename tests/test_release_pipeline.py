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
