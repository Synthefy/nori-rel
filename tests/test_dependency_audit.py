"""Dependency exceptions apply only to complete, unchanged reviewed evidence."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

AUDIT_CHECKER_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts/audit_dependencies.py"
)
REVIEWED_ADVISORY_IDS = ["PYSEC-2026-1000", "CVE-2026-12345", "GHSA-abcd-efgh-ijkl"]
REVIEWED_ADVISORY_DESCRIPTION = (
    "Synthetic advisory for a reviewed optional execution path."
)


@pytest.fixture
def audit_case(tmp_path, monkeypatch):
    specification = importlib.util.spec_from_file_location(
        "dependency_audit", AUDIT_CHECKER_SCRIPT
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    monkeypatch.chdir(tmp_path)
    files = module.REQUIRED_REVIEW_FILES | {"src/example.py", "src/py.typed"}
    for name in files:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("reviewed bytes\n")
    today = datetime.now(UTC).date()
    policy = {
        "schema_version": 1,
        "owner": "Example maintainers",
        "reviewed_on": today.isoformat(),
        "expires_on": (today + timedelta(days=30)).isoformat(),
        "reviewed_files": {
            name: {"sha256": hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()}
            for name in files
        },
        "exceptions": [
            {
                "package": "example-package",
                "version": "1.0",
                "advisory_ids": REVIEWED_ADVISORY_IDS,
                "rationale": "The affected path is absent from this reviewed snapshot.",
                "source_urls": ["https://example.org/advisory"],
                "description_sha256": [
                    hashlib.sha256(REVIEWED_ADVISORY_DESCRIPTION.encode()).hexdigest()
                ],
            }
        ],
    }
    vulnerability = {
        "id": REVIEWED_ADVISORY_IDS[0],
        "aliases": REVIEWED_ADVISORY_IDS[1:],
        "description": REVIEWED_ADVISORY_DESCRIPTION,
        "fix_versions": ["2.0"],
    }
    report = {
        "dependencies": [
            {"name": "example-package", "version": "1.0", "vulns": [vulnerability]}
        ],
        "fixes": [],
    }
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("Example_Package==1.0\n")
    case = SimpleNamespace(
        module=module,
        root=tmp_path,
        policy=policy,
        report=report,
        requirements=requirements,
        exit_code=1,
        stdout="",
        stderr=None,
        raw_report=None,
        calls=[],
    )

    def fake_scanner(command, **options):
        case.calls.append((command, options))
        report_path = Path(command[command.index("--output") + 1])
        report_path.write_text(
            case.raw_report if case.raw_report is not None else json.dumps(case.report)
        )
        if case.stderr is None:
            dependencies = case.report.get("dependencies", [])
            count = sum(len(item.get("vulns", [])) for item in dependencies)
            affected = sum(bool(item.get("vulns")) for item in dependencies)
            stderr = (
                f"Found {count} known "
                f"{'vulnerability' if count == 1 else 'vulnerabilities'} "
                f"in {affected} {'package' if affected == 1 else 'packages'}\n"
                if count
                else "No known vulnerabilities found\n"
            )
        else:
            stderr = case.stderr
        return SimpleNamespace(
            returncode=case.exit_code, stdout=case.stdout, stderr=stderr
        )

    monkeypatch.setattr(module.subprocess, "run", fake_scanner)

    def invoke_audit():
        policy_path = tmp_path / "policy.json"
        policy_path.write_text(json.dumps(case.policy))
        output = tmp_path / "audit"
        status = module.audit_dependencies(requirements, output, policy_path)
        assessment = json.loads((output / "assessment.json").read_text())
        return status, assessment

    case.invoke = invoke_audit
    return case


def test_exact_review_preserves_raw_findings_and_scanner_exit(audit_case):
    status, assessment = audit_case.invoke()
    assert status == 0 and assessment["passed"]
    assert assessment["dispositions"][0]["accepted"]
    output = audit_case.root / "audit"
    assert json.loads((output / "report.json").read_text()) == audit_case.report
    assert json.loads((output / "scanner-exit.json").read_text()) == {"returncode": 1}
    assert (output / "scanner.stderr.log").read_text().startswith("Found 1 known")
    command, options = audit_case.calls[0]
    assert command[:3] == [audit_case.module.sys.executable, "-m", "pip_audit"]
    assert {"--strict", "--no-deps", "--disable-pip"} <= set(command)
    assert "--ignore-vuln" not in command and options["check"] is False


def test_clean_report_needs_no_exceptions_and_honors_markers(audit_case):
    audit_case.policy["exceptions"] = []
    audit_case.report["dependencies"][0]["vulns"] = []
    audit_case.requirements.write_text(
        'Example_Package==1.0\nignored==9.0 ; python_version < "2"\n'
    )
    audit_case.exit_code = 0
    assert audit_case.invoke() == (
        0,
        {"passed": True, "dispositions": [], "unaccepted_findings": []},
    )


@pytest.mark.parametrize(
    "change", ["new-id", "new-alias", "version", "description", "no-exception"]
)
def test_unreviewed_vulnerability_fails(audit_case, change):
    dependency = audit_case.report["dependencies"][0]
    vulnerability = dependency["vulns"][0]
    if change == "new-id":
        vulnerability["id"] = "CVE-2026-99999"
    elif change == "new-alias":
        vulnerability["aliases"].append("GHSA-xxxx-yyyy-zzzz")
    elif change == "version":
        dependency["version"] = "1.1"
        audit_case.requirements.write_text("example-package==1.1\n")
    elif change == "description":
        vulnerability["description"] += " Newly affected path."
    else:
        audit_case.policy["exceptions"] = []
    status, assessment = audit_case.invoke()
    assert status == 1 and len(assessment["unaccepted_findings"]) == 1


def test_aliases_cannot_be_combined_across_exception_entries(audit_case):
    original = audit_case.policy["exceptions"][0]
    audit_case.policy["exceptions"] = [
        dict(original, advisory_ids=[known]) for known in REVIEWED_ADVISORY_IDS
    ]
    assert audit_case.invoke()[0] == 1


def test_duplicate_advisories_with_reviewed_descriptions_are_retained(audit_case):
    vulnerability = copy.deepcopy(audit_case.report["dependencies"][0]["vulns"][0])
    vulnerability["description"] = "Another reviewed description."
    audit_case.policy["exceptions"][0]["description_sha256"].append(
        hashlib.sha256(vulnerability["description"].encode()).hexdigest()
    )
    audit_case.report["dependencies"][0]["vulns"].append(vulnerability)
    status, assessment = audit_case.invoke()
    assert status == 0 and len(assessment["dispositions"]) == 2


@pytest.mark.parametrize(
    "change",
    [
        "expired",
        "future",
        "hash",
        "new-source",
        "removed-source",
        "missing-required",
        "nonlocal",
        "owner",
        "rationale",
        "urls",
        "description-hashes",
    ],
)
def test_review_binding_fails_closed(audit_case, change):
    today = datetime.now(UTC).date()
    if change == "expired":
        audit_case.policy["expires_on"] = (today - timedelta(days=1)).isoformat()
    elif change == "future":
        audit_case.policy["reviewed_on"] = (today + timedelta(days=1)).isoformat()
    elif change == "hash":
        (audit_case.root / "src/example.py").write_text("changed")
    elif change == "new-source":
        (audit_case.root / "src/native.so").write_bytes(b"new native file")
    elif change == "removed-source":
        (audit_case.root / "src/py.typed").unlink()
    elif change == "missing-required":
        audit_case.policy["reviewed_files"].pop("scripts/audit_dependencies.py")
    elif change == "nonlocal":
        audit_case.policy["reviewed_files"]["../outside"] = "a" * 64
    elif change == "owner":
        audit_case.policy["owner"] = " "
    else:
        field = {
            "rationale": "rationale",
            "urls": "source_urls",
            "description-hashes": "description_sha256",
        }[change]
        audit_case.policy["exceptions"][0][field] = []
    status, assessment = audit_case.invoke()
    assert status == 1 and "error" in assessment


def test_generated_bytecode_does_not_invalidate_source_review(audit_case):
    cache = audit_case.root / "src/__pycache__"
    cache.mkdir()
    (cache / "example.cpython-312.pyc").write_bytes(b"cache")
    (audit_case.root / "src/example.pyc").write_bytes(b"legacy cache")
    assert audit_case.invoke()[0] == 0


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "extra",
        "skipped",
        "duplicate",
        "duplicate-version",
        "missing-aliases",
        "bad-aliases",
        "bad-description",
        "bad-vulns",
        "malformed-json",
        "duplicate-json-key",
    ],
)
def test_incomplete_or_malformed_report_fails(audit_case, change):
    dependency = audit_case.report["dependencies"][0]
    if change == "missing":
        audit_case.report["dependencies"] = []
    elif change == "extra":
        audit_case.report["dependencies"].append(
            {"name": "extra", "version": "1", "vulns": []}
        )
    elif change == "skipped":
        dependency["skip_reason"] = "package could not be audited"
    elif change.startswith("duplicate") and change != "duplicate-json-key":
        duplicate = copy.deepcopy(dependency)
        if change == "duplicate-version":
            duplicate["version"] = "2.0"
        audit_case.report["dependencies"].append(duplicate)
    elif change == "missing-aliases":
        del dependency["vulns"][0]["aliases"]
    elif change == "bad-aliases":
        dependency["vulns"][0]["aliases"] = "CVE-2026-12345"
    elif change == "bad-description":
        dependency["vulns"][0]["description"] = ""
    elif change == "bad-vulns":
        dependency["vulns"] = None
        audit_case.stderr = "Found 1 known vulnerability in 1 package\n"
    else:
        audit_case.raw_report = (
            '{"dependencies":'
            if change == "malformed-json"
            else '{"dependencies": [], "dependencies": [], "fixes": []}'
        )
    assert audit_case.invoke()[0] == 1


@pytest.mark.parametrize(
    "exit_code,has_findings",
    [(2, False), (2, True), (-9, False), (0, True), (1, False)],
)
def test_process_errors_and_exit_report_mismatch_fail(
    audit_case, exit_code, has_findings
):
    audit_case.exit_code = exit_code
    if not has_findings:
        audit_case.report["dependencies"][0]["vulns"] = []
    assert audit_case.invoke()[0] == 1


@pytest.mark.parametrize(
    "diagnostic",
    [
        "ERROR:pip_audit:network failure",
        "Traceback: failed",
        "Unknown warning",
        "Found 0 known vulnerabilities in 0 packages",
    ],
)
def test_diagnostic_errors_fail_even_with_complete_report(audit_case, diagnostic):
    audit_case.stderr = "Found 1 known vulnerability in 1 package\n" + diagnostic
    assert audit_case.invoke()[0] == 1


def test_documented_scanner_warnings_are_allowed(audit_case):
    audit_case.stderr = "\n".join(sorted(audit_case.module.SCANNER_WARNINGS))
    audit_case.stderr += "\nFound 1 known vulnerability in 1 package\n"
    assert audit_case.invoke()[0] == 0


def test_known_bit_alias_can_be_explicitly_reviewed(audit_case):
    alias = "BIT-example-2026-12345"
    audit_case.policy["exceptions"][0]["advisory_ids"] = [*REVIEWED_ADVISORY_IDS, alias]
    audit_case.report["dependencies"][0]["vulns"][0]["aliases"].append(alias)
    assert audit_case.invoke()[0] == 0


def test_review_cannot_authorize_more_than_thirty_days(audit_case):
    audit_case.policy["expires_on"] = (
        datetime.now(UTC).date() + timedelta(days=31)
    ).isoformat()
    assert audit_case.invoke()[0] == 1


@pytest.mark.parametrize("review_age_days", [0, 1])
def test_review_expires_at_start_of_expiry_date(audit_case, review_age_days):
    today = datetime.now(UTC).date()
    audit_case.policy["reviewed_on"] = (
        today - timedelta(days=review_age_days)
    ).isoformat()
    audit_case.policy["expires_on"] = today.isoformat()
    assert audit_case.invoke()[0] == 1


def test_scanner_launch_failure_leaves_diagnostic_evidence(audit_case, monkeypatch):
    def failed_scanner(*arguments, **options):
        raise OSError("synthetic launch failure")

    monkeypatch.setattr(audit_case.module.subprocess, "run", failed_scanner)
    assert audit_case.invoke()[0] == 1
    output = audit_case.root / "audit"
    assert json.loads((output / "scanner-exit.json").read_text()) == {
        "returncode": None
    }
    assert "synthetic launch failure" in (output / "scanner.stderr.log").read_text()


@pytest.mark.parametrize(
    "pin",
    [
        "example-package>=1.0",
        "example-package==1.*",
        "-r hidden.txt",
        "example-package @ https://example.org/package.whl",
    ],
)
def test_nonexact_requirements_are_rejected(audit_case, pin):
    audit_case.requirements.write_text(pin + "\n")
    assert audit_case.invoke()[0] == 1
