"""Release controls reject unreviewed refs, stale CI and foreign candidates."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_release.py"
COMMIT = "a" * 40
REPOSITORY = "Synthefy/nori-rel"


@pytest.fixture
def release_evidence(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("verify_nori_release", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "0.3.0"\n')
    output = tmp_path / "output"
    for key, value in {
        "GITHUB_REPOSITORY": REPOSITORY,
        "GITHUB_REF": "refs/heads/main",
        "GITHUB_SHA": COMMIT,
        "GITHUB_OUTPUT": str(output),
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "v0.3.0", "--target", "testpypi"])
    responses = {
        "git/ref/tags/v0.3.0": {"object": {"type": "commit", "sha": COMMIT}},
        "branches/main": {"protected": True, "commit": {"sha": COMMIT}},
        f"commits/{COMMIT}/check-runs?per_page=100": {
            "check_runs": [
                {
                    "name": "release-readiness",
                    "app": {"slug": "github-actions"},
                    "id": 100,
                    "conclusion": "success",
                    "details_url": f"https://github.com/{REPOSITORY}/actions/runs/123",
                }
            ]
        },
        "actions/runs/123": {
            "path": ".github/workflows/ci.yml",
            "event": "push",
            "head_branch": "main",
            "head_sha": COMMIT,
            "conclusion": "success",
        },
        "actions/runs/456": {
            "path": ".github/workflows/publish.yml",
            "event": "workflow_dispatch",
            "head_branch": "main",
            "head_sha": COMMIT,
            "conclusion": "success",
            "repository": {"full_name": REPOSITORY},
            "head_repository": {"full_name": REPOSITORY},
        },
    }
    monkeypatch.setattr(module, "read_github_repository_json", responses.__getitem__)
    return SimpleNamespace(module=module, responses=responses, output=output)


def test_verified_candidate_emits_only_the_reviewed_identifiers(release_evidence):
    release_evidence.module.main()

    assert release_evidence.output.read_text() == f"commit={COMMIT}\nversion=0.3.0\n"


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("GITHUB_REPOSITORY", "someone/nori-rel"),
        ("GITHUB_REF", "refs/heads/dev-experiment"),
        ("GITHUB_REF", "refs/tags/v0.3.0"),
        ("GITHUB_SHA", "b" * 40),
    ],
)
def test_rejects_unapproved_dispatch_context(
    release_evidence, monkeypatch, variable, value
):
    monkeypatch.setenv(variable, value)

    with pytest.raises(ValueError):
        release_evidence.module.main()

    assert not release_evidence.output.exists()


def test_rejects_unprotected_main(release_evidence):
    release_evidence.responses["branches/main"]["protected"] = False

    with pytest.raises(ValueError, match="protection"):
        release_evidence.module.main()


def test_rejects_tag_outside_main(release_evidence):
    release_evidence.responses["git/ref/tags/v0.3.0"]["object"]["sha"] = "b" * 40

    with pytest.raises(ValueError, match="current protected main"):
        release_evidence.module.main()


def test_rejects_version_mismatch(release_evidence):
    Path("pyproject.toml").write_text('[project]\nversion = "0.3.1"\n')

    with pytest.raises(ValueError, match="project version differ"):
        release_evidence.module.main()


def test_latest_failed_readiness_cannot_reuse_an_older_success(release_evidence):
    checks = release_evidence.responses[f"commits/{COMMIT}/check-runs?per_page=100"][
        "check_runs"
    ]
    checks.append(checks[0] | {"id": 101, "conclusion": "failure"})

    with pytest.raises(ValueError, match="release-readiness"):
        release_evidence.module.main()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("path", ".github/workflows/other.yml"),
        ("event", "pull_request"),
        ("head_sha", "b" * 40),
        ("head_branch", "dev-experiment"),
        ("conclusion", "failure"),
    ],
)
def test_readiness_must_certify_this_main_push(release_evidence, field, value):
    release_evidence.responses["actions/runs/123"][field] = value

    with pytest.raises(ValueError, match="successful CI on the main push"):
        release_evidence.module.main()


def test_annotated_release_tags_resolve_to_the_reviewed_commit(release_evidence):
    tag_sha = "c" * 40
    release_evidence.responses["git/ref/tags/v0.3.0"]["object"] = {
        "type": "tag",
        "sha": tag_sha,
    }
    release_evidence.responses[f"git/tags/{tag_sha}"] = {
        "object": {"type": "commit", "sha": COMMIT}
    }

    release_evidence.module.main()

    assert f"commit={COMMIT}" in release_evidence.output.read_text()


def test_production_requires_a_successful_testpypi_run(release_evidence, monkeypatch):
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "v0.3.0", "--target", "pypi"])

    with pytest.raises(ValueError, match="TestPyPI candidate run ID"):
        release_evidence.module.main()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("path", ".github/workflows/other.yml"),
        ("event", "pull_request"),
        ("head_sha", "b" * 40),
        ("head_branch", "dev-experiment"),
        ("conclusion", "failure"),
        ("repository", {"full_name": "someone/nori-rel"}),
        ("head_repository", {"full_name": "someone/nori-rel"}),
    ],
)
def test_rejects_foreign_or_stale_production_candidate(
    release_evidence, monkeypatch, field, value
):
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT), "v0.3.0", "--target", "pypi", "--candidate-run", "456"],
    )
    release_evidence.responses["actions/runs/456"][field] = value

    with pytest.raises(ValueError, match="successful release run at this commit"):
        release_evidence.module.main()

    assert not release_evidence.output.exists()


def test_accepts_the_successful_candidate_at_the_exact_commit(
    release_evidence, monkeypatch
):
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT), "v0.3.0", "--target", "pypi", "--candidate-run", "456"],
    )

    release_evidence.module.main()

    assert f"commit={COMMIT}" in release_evidence.output.read_text()
