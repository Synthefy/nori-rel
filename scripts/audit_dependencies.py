"""Run pip-audit and assess only reviewed, expiring dependency exceptions."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import UTC, date, datetime
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

REQUIRED_REVIEW_FILES = frozenset(
    {
        ".gitattributes",
        "pyproject.toml",
        "uv.lock",
        "scripts/audit_dependencies.py",
        "scripts/validate_artifacts.py",
        *(
            f".github/workflows/{name}.yml"
            for name in (
                "ci",
                "dependency-audit",
                "publish",
                "release-matrix",
                "warehouse-tests",
            )
        ),
    }
)
ADVISORY_ID_PATTERN = re.compile(
    r"(?:PYSEC-\d{4}-\d+|CVE-\d{4}-\d{4,}|GHSA-[a-z0-9]{4}(?:-[a-z0-9]{4}){2}"
    r"|BIT-[a-z0-9-]+-\d{4}-\d+)"
)
MAX_REVIEW_AGE_DAYS = 30
SCANNER_WARNINGS = frozenset(
    {
        "WARNING:pip_audit._cli:--no-deps is supported, but users are encouraged "
        "to fully hash their pinned dependencies",
        "WARNING:pip_audit._cli:Consider using a tool like `pip-compile`: "
        "https://pip-tools.readthedocs.io/en/latest/#using-hashes",
    }
)


def require_valid_evidence(condition: bool, message: str) -> None:
    """Reject incomplete or inconsistent audit evidence."""
    if not condition:
        raise ValueError(message)


def read_json_object(path: Path) -> dict:
    """Load JSON while rejecting duplicate keys and non-object roots."""

    def collect_unique_json_fields(pairs):
        result = {}
        for name, value in pairs:
            require_valid_evidence(name not in result, f"duplicate JSON field: {name}")
            result[name] = value
        return result

    value = json.loads(path.read_text(), object_pairs_hook=collect_unique_json_fields)
    require_valid_evidence(isinstance(value, dict), "JSON root must be an object")
    return value


def is_nonempty_text(value: object) -> bool:
    """Return whether a field contains a nonempty string."""
    return isinstance(value, str) and bool(value.strip())


def read_pinned_requirements(path: Path) -> dict[str, str]:
    """Resolve applicable exact pins using the scanner interpreter's markers."""
    versions_by_package = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        requirement = Requirement(line)
        pins = list(requirement.specifier)
        require_valid_evidence(
            not requirement.url and len(pins) == 1 and pins[0].operator == "==",
            "every requirement must have one exact version pin",
        )
        version = pins[0].version
        Version(version)
        if requirement.marker and not requirement.marker.evaluate():
            continue
        package = canonicalize_name(requirement.name, validate=True)
        require_valid_evidence(
            versions_by_package.get(package, version) == version,
            f"conflicting requirement pins: {package}",
        )
        versions_by_package[package] = version
    require_valid_evidence(
        bool(versions_by_package), "requirements contain no applicable pins"
    )
    return versions_by_package


def validate_exception_policy(policy: dict, root: Path, today: date) -> list[dict]:
    """Bind exceptions to reviewed dates, exact source files, and known IDs."""
    require_valid_evidence(
        type(policy.get("schema_version")) is int, "invalid policy schema"
    )
    require_valid_evidence(policy["schema_version"] == 1, "unsupported policy schema")
    require_valid_evidence(
        is_nonempty_text(policy.get("owner")), "policy owner is required"
    )
    reviewed = date.fromisoformat(policy["reviewed_on"])
    expires = date.fromisoformat(policy["expires_on"])
    require_valid_evidence(
        reviewed <= today < expires, "policy is future-dated or expired"
    )
    require_valid_evidence(
        (expires - reviewed).days <= MAX_REVIEW_AGE_DAYS, "review exceeds 30 days"
    )
    reviewed_files = policy.get("reviewed_files")
    require_valid_evidence(
        isinstance(reviewed_files, dict), "reviewed_files must be an object"
    )
    source_files = set()
    for path in (root / "src").rglob("*"):
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        require_valid_evidence(
            not path.is_symlink(), "source symlinks cannot be reviewed"
        )
        if path.is_file():
            source_files.add(path.relative_to(root).as_posix())
    require_valid_evidence(bool(source_files), "source inventory is empty")
    require_valid_evidence(
        {name for name in reviewed_files if name.startswith("src/")} == source_files,
        "reviewed source inventory changed",
    )
    require_valid_evidence(
        reviewed_files.keys() >= REQUIRED_REVIEW_FILES, "review files are missing"
    )
    for name, hash_metadata in reviewed_files.items():
        relative = PurePosixPath(name)
        require_valid_evidence(
            bool(name)
            and not relative.is_absolute()
            and ".." not in relative.parts
            and "\\" not in name
            and ":" not in name
            and relative.as_posix() == name,
            "reviewed file path must be local and normalized",
        )
        path = root / relative
        require_valid_evidence(
            path.resolve().is_relative_to(root.resolve())
            and not any(parent.is_symlink() for parent in (path, *path.parents))
            and path.is_file(),
            f"reviewed file is missing or nonlocal: {name}",
        )
        require_valid_evidence(
            isinstance(hash_metadata, dict) and set(hash_metadata) == {"sha256"},
            "reviewed file must declare a SHA-256 hash",
        )
        expected_hash = hash_metadata["sha256"]
        require_valid_evidence(
            isinstance(expected_hash, str)
            and re.fullmatch(r"[a-f0-9]{64}", expected_hash) is not None
            and hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash,
            f"reviewed file hash changed: {name}",
        )
    exceptions = policy.get("exceptions")
    require_valid_evidence(isinstance(exceptions, list), "exceptions must be a list")
    for exception in exceptions:
        require_valid_evidence(
            isinstance(exception, dict), "exception must be an object"
        )
        canonicalize_name(exception["package"], validate=True)
        Version(exception["version"])
        advisory_ids = exception.get("advisory_ids")
        require_valid_evidence(
            isinstance(advisory_ids, list)
            and bool(advisory_ids)
            and all(
                isinstance(item, str) and ADVISORY_ID_PATTERN.fullmatch(item)
                for item in advisory_ids
            ),
            "exception must enumerate known advisory IDs",
        )
        require_valid_evidence(
            is_nonempty_text(exception.get("rationale")), "rationale is required"
        )
        description_hashes = exception.get("description_sha256")
        require_valid_evidence(
            isinstance(description_hashes, list)
            and bool(description_hashes)
            and all(
                isinstance(item, str) and re.fullmatch(r"[a-f0-9]{64}", item)
                for item in description_hashes
            ),
            "reviewed description hashes are required",
        )
        urls = exception.get("source_urls")
        require_valid_evidence(
            isinstance(urls, list) and bool(urls), "source URLs are required"
        )
        for url in urls:
            require_valid_evidence(isinstance(url, str), "source URL must be text")
            parsed = urlsplit(url)
            require_valid_evidence(
                parsed.scheme == "https"
                and bool(parsed.hostname)
                and parsed.username is None,
                "source URL must use HTTPS",
            )
    return exceptions


def assess_report(
    report: dict, pins: dict[str, str], exceptions: list[dict]
) -> list[dict]:
    """Validate complete coverage and classify every unfiltered advisory."""
    require_valid_evidence(
        set(report) == {"dependencies", "fixes"}, "invalid report fields"
    )
    require_valid_evidence(report["fixes"] == [], "scanner must not apply fixes")
    dependencies = report["dependencies"]
    require_valid_evidence(
        isinstance(dependencies, list), "dependencies must be a list"
    )
    seen_packages = set()
    dispositions = []
    for dependency in dependencies:
        require_valid_evidence(
            isinstance(dependency, dict), "dependency must be an object"
        )
        require_valid_evidence(
            set(dependency) == {"name", "version", "vulns"},
            "skipped/malformed dependency",
        )
        package = canonicalize_name(dependency["name"], validate=True)
        version = dependency["version"]
        require_valid_evidence(
            package not in seen_packages, f"duplicate dependency: {package}"
        )
        require_valid_evidence(
            pins.get(package) == version, f"unexpected dependency or version: {package}"
        )
        seen_packages.add(package)
        require_valid_evidence(
            isinstance(dependency["vulns"], list), "vulns must be a list"
        )
        for vulnerability in dependency["vulns"]:
            require_valid_evidence(
                isinstance(vulnerability, dict), "vulnerability must be an object"
            )
            require_valid_evidence(
                set(vulnerability) == {"id", "aliases", "fix_versions", "description"},
                "malformed vulnerability fields",
            )
            require_valid_evidence(
                is_nonempty_text(vulnerability["id"]), "advisory ID is required"
            )
            for field in ("aliases", "fix_versions"):
                require_valid_evidence(
                    isinstance(vulnerability[field], list)
                    and all(is_nonempty_text(item) for item in vulnerability[field]),
                    f"invalid vulnerability {field}",
                )
            require_valid_evidence(
                is_nonempty_text(vulnerability["description"]), "invalid description"
            )
            description_hash = hashlib.sha256(
                vulnerability["description"].encode()
            ).hexdigest()
            observed_ids = {vulnerability["id"], *vulnerability["aliases"]}
            matched = next(
                (
                    exception
                    for exception in exceptions
                    if canonicalize_name(exception["package"]) == package
                    and exception["version"] == version
                    and observed_ids <= set(exception["advisory_ids"])
                    and description_hash in exception["description_sha256"]
                ),
                None,
            )
            dispositions.append(
                {
                    "package": package,
                    "version": version,
                    "advisory_ids": sorted(observed_ids),
                    "description_sha256": description_hash,
                    "accepted": matched is not None,
                    "rationale": matched["rationale"] if matched else None,
                }
            )
    require_valid_evidence(
        seen_packages == pins.keys(), "report omits applicable dependencies"
    )
    return dispositions


def audit_dependencies(requirements: Path, output: Path, policy_path: Path) -> int:
    """Preserve scanner evidence and fail on unreviewed or invalid findings."""
    output.mkdir(parents=True, exist_ok=True)
    assessment = {"passed": False, "dispositions": [], "unaccepted_findings": []}
    report_path = output / "report.json"
    try:
        require_valid_evidence(
            not report_path.exists(), "output already contains a scanner report"
        )
        command = [
            sys.executable,
            "-m",
            "pip_audit",
            "--strict",
            "--no-deps",
            "--disable-pip",
            "--progress-spinner",
            "off",
            "--format",
            "json",
            "--output",
            str(report_path),
            "-r",
            str(requirements),
        ]
        try:
            scanner = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
        except OSError as error:
            (output / "scanner.stdout.log").write_text("")
            (output / "scanner.stderr.log").write_text(str(error))
            (output / "scanner-exit.json").write_text(json.dumps({"returncode": None}))
            raise
        (output / "scanner.stdout.log").write_text(scanner.stdout)
        (output / "scanner.stderr.log").write_text(scanner.stderr)
        (output / "scanner-exit.json").write_text(
            json.dumps({"returncode": scanner.returncode})
        )
        require_valid_evidence(scanner.returncode in (0, 1), "pip-audit process failed")
        pins = read_pinned_requirements(requirements)
        policy = read_json_object(policy_path)
        exceptions = validate_exception_policy(
            policy, Path.cwd(), datetime.now(UTC).date()
        )
        dispositions = assess_report(read_json_object(report_path), pins, exceptions)
        assessment["dispositions"] = dispositions
        assessment["unaccepted_findings"] = [
            item for item in dispositions if not item["accepted"]
        ]
        require_valid_evidence(
            scanner.returncode == int(bool(dispositions)),
            "scanner exit/report mismatch",
        )
        affected = len({item["package"] for item in dispositions})
        summary = (
            f"Found {len(dispositions)} known "
            f"{'vulnerability' if len(dispositions) == 1 else 'vulnerabilities'} "
            f"in {affected} {'package' if affected == 1 else 'packages'}"
            if dispositions
            else "No known vulnerabilities found"
        )
        diagnostic_lines = [
            line for line in scanner.stderr.splitlines() if line.strip()
        ]
        require_valid_evidence(
            not scanner.stdout.strip()
            and diagnostic_lines.count(summary) == 1
            and all(
                line in SCANNER_WARNINGS or line == summary for line in diagnostic_lines
            ),
            "unexpected or missing scanner diagnostics",
        )
        assessment["passed"] = not assessment["unaccepted_findings"]
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        assessment["error"] = str(error)
    (output / "assessment.json").write_text(json.dumps(assessment, indent=2) + "\n")
    return 0 if assessment["passed"] else 1


def main() -> int:
    """Audit pinned requirements from the repository root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--policy", type=Path, default=Path("docs/dependency-exceptions.json")
    )
    arguments = parser.parse_args()
    return audit_dependencies(
        arguments.requirements, arguments.output, arguments.policy
    )


if __name__ == "__main__":
    sys.exit(main())
