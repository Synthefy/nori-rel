"""Release controls reject unreviewed refs, stale CI and foreign candidates."""

from __future__ import annotations

import hashlib
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
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n## 0.3.0 \u2014 2026-10-14\n\nInitial public release.\n"
        "\n## 0.2.0 \u2014 2026-09-01\n\nOlder notes.\n"
    )
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


@pytest.mark.parametrize(
    "changelog",
    [
        "# Changelog\n\n## 0.2.0 \u2014 2026-09-01\n\nOlder notes.\n",
        "# Changelog\n\n## Unreleased\n\nNew notes.\n",
        "# Changelog\n\n## 0.3.0\n\nNew notes.\n",
        "# Changelog\n\n## 0.3.0rc1 \u2014 2026-10-01\n\nCandidate notes.\n",
        "# Changelog\n\n## 0.3.0 \u2014 2026-13-45\n\nNew notes.\n",
        "# Changelog\n\n## 0.3.0 \u2014 2026-10-14\n\n"
        "## 0.2.0 \u2014 2026-09-01\n\nOlder.\n",
        "# Changelog\n\n## 0.3.0 \u2014 2026-10-14\n",
    ],
    ids=[
        "no-entry",
        "unreleased",
        "undated",
        "other-version",
        "invalid-date",
        "empty-notes",
        "empty-last-entry",
    ],
)
def test_requires_dated_release_notes_for_the_tagged_version(
    release_evidence, changelog
):
    Path("CHANGELOG.md").write_text(changelog)

    with pytest.raises(ValueError, match=r"CHANGELOG\.md needs release notes"):
        release_evidence.module.main()

    assert not release_evidence.output.exists()


def test_changelog_entry_may_use_an_ascii_hyphen(release_evidence):
    Path("CHANGELOG.md").write_text("## 0.3.0 - 2026-10-14\n\nInitial release.\n")

    release_evidence.module.main()

    assert release_evidence.output.read_text() == f"commit={COMMIT}\nversion=0.3.0\n"


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


def test_explicit_direct_release_needs_no_testpypi_run(release_evidence, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT), "v0.3.0", "--target", "pypi", "--direct-publish"],
    )
    del release_evidence.responses["actions/runs/456"]

    release_evidence.module.main()

    assert release_evidence.output.read_text() == f"commit={COMMIT}\nversion=0.3.0\n"


@pytest.mark.parametrize(
    ("target", "candidate_run", "direct_publish"),
    [
        ("testpypi", "", True),
        ("testpypi", "456", True),
        ("pypi", "456", True),
        ("testpypi", "456", False),
    ],
)
def test_incompatible_release_options_fail(
    release_evidence, monkeypatch, target, candidate_run, direct_publish
):
    arguments = [
        str(SCRIPT),
        "v0.3.0",
        "--target",
        target,
        "--candidate-run",
        candidate_run,
    ]
    if direct_publish:
        arguments.append("--direct-publish")
    monkeypatch.setattr(sys, "argv", arguments)

    with pytest.raises(ValueError, match="candidate_run"):
        release_evidence.module.main()

    assert not release_evidence.output.exists()


@pytest.mark.parametrize("failure", ["protection", "tag", "version", "ci"])
def test_direct_release_preserves_protected_main_and_ci_guards(
    release_evidence, monkeypatch, failure
):
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT), "v0.3.0", "--target", "pypi", "--direct-publish"],
    )
    if failure == "protection":
        release_evidence.responses["branches/main"]["protected"] = False
    elif failure == "tag":
        release_evidence.responses["git/ref/tags/v0.3.0"]["object"]["sha"] = "b" * 40
    elif failure == "version":
        Path("pyproject.toml").write_text('[project]\nversion = "0.3.1"\n')
    else:
        release_evidence.responses["actions/runs/123"]["conclusion"] = "failure"

    with pytest.raises(ValueError):
        release_evidence.module.main()

    assert not release_evidence.output.exists()


@pytest.fixture
def candidate_artifacts(release_evidence, tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    file_metadata_by_name = {}
    for name in ("nori_rel-0.3.0-py3-none-any.whl", "nori_rel-0.3.0.tar.gz"):
        contents = f"synthetic distribution: {name}".encode()
        (dist / name).write_bytes(contents)
        file_metadata_by_name[name] = {"sha256": hashlib.sha256(contents).hexdigest()}
    return SimpleNamespace(
        module=release_evidence.module,
        dist=dist,
        manifest={
            "commit": COMMIT,
            "version": "0.3.0",
            "candidate": {
                "repository": REPOSITORY,
                "workflow": ".github/workflows/publish.yml",
                "run_id": "456",
                "target": "testpypi",
            },
            "files": file_metadata_by_name,
        },
    )


@pytest.mark.parametrize("target", ["testpypi", "pypi"])
def test_candidate_provenance_records_the_actual_build_target(release_evidence, target):
    assert release_evidence.module.build_candidate_provenance("456", target) == {
        "repository": REPOSITORY,
        "workflow": ".github/workflows/publish.yml",
        "run_id": "456",
        "target": target,
    }


@pytest.mark.parametrize(
    ("run_id", "target"),
    [("", "pypi"), ("0", "pypi"), ("abc", "pypi"), ("456", "other")],
)
def test_invalid_candidate_provenance_is_rejected(release_evidence, run_id, target):
    with pytest.raises(ValueError, match="provenance"):
        release_evidence.module.build_candidate_provenance(run_id, target)


@pytest.mark.parametrize("target", ["testpypi", "pypi"])
def test_validated_candidate_keeps_exact_distribution_bytes(
    candidate_artifacts, target
):
    case = candidate_artifacts
    case.manifest["candidate"]["target"] = target
    original_bytes_by_name = {
        path.name: path.read_bytes() for path in case.dist.iterdir()
    }

    case.module.validate_candidate_artifacts(
        case.manifest,
        case.dist,
        commit=COMMIT,
        version="0.3.0",
        candidate_run="456",
        candidate_target=target,
    )

    assert {
        path.name: path.read_bytes() for path in case.dist.iterdir()
    } == original_bytes_by_name


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("repository", "someone/nori-rel"),
        ("workflow", ".github/workflows/other.yml"),
        ("run_id", "789"),
        ("target", "pypi"),
    ],
)
def test_promotion_rejects_foreign_run_or_direct_pypi_candidate(
    candidate_artifacts, field, value
):
    case = candidate_artifacts
    case.manifest["candidate"][field] = value

    with pytest.raises(ValueError, match="identity"):
        case.module.validate_candidate_artifacts(
            case.manifest,
            case.dist,
            commit=COMMIT,
            version="0.3.0",
            candidate_run="456",
            candidate_target="testpypi",
        )


@pytest.mark.parametrize(
    "change", ["commit", "version", "missing-provenance", "digest", "file-list"]
)
def test_candidate_identity_and_distribution_tampering_fail(
    candidate_artifacts, change
):
    case = candidate_artifacts
    if change in {"commit", "version"}:
        case.manifest[change] = "unreviewed"
    elif change == "missing-provenance":
        del case.manifest["candidate"]
    elif change == "digest":
        (case.dist / "nori_rel-0.3.0-py3-none-any.whl").write_bytes(b"tampered")
    else:
        (case.dist / "unexpected.whl").write_bytes(b"unexpected")

    with pytest.raises(ValueError):
        case.module.validate_candidate_artifacts(
            case.manifest,
            case.dist,
            commit=COMMIT,
            version="0.3.0",
            candidate_run="456",
            candidate_target="testpypi",
        )
