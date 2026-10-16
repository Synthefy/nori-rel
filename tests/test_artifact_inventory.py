"""Release archives must match reviewed Git inputs, including non-package files."""

from __future__ import annotations

import importlib.util
import io
import os
import subprocess
import tarfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

ARTIFACT_VALIDATOR_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts/validate_artifacts.py"
)


def write_distribution_archives(case):
    """Build tiny wheel/sdist fixtures without invoking any package backend."""
    with zipfile.ZipFile(case.dist / "nori_rel-0.3.0-py3-none-any.whl", "w") as archive:
        for name, content in case.wheel_files.items():
            archive.writestr(name, content)
    with tarfile.open(case.dist / "nori_rel-0.3.0.tar.gz", "w:gz") as archive:
        for name, content in case.sdist_files.items():
            member = tarfile.TarInfo(f"nori_rel-0.3.0/{name}")
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))


@pytest.fixture
def distribution_case(tmp_path):
    specification = importlib.util.spec_from_file_location(
        "artifact_validator", ARTIFACT_VALIDATOR_SCRIPT
    )
    validator = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(validator)
    source = tmp_path / "source"
    dist = tmp_path / "dist"
    source.mkdir()
    dist.mkdir()
    pyproject = (
        b"# Reviewed comment removed by uv_build normalization.\n"
        b'[build-system]\nrequires = ["uv_build==0.12.21"]\n'
        b'build-backend = "uv_build"\n'
        b'[project]\nname = "nori-rel"\nversion = "0.3.0"\n'
        b'description = "A literal # inside a string is valid"\n'
        b'license-files = ["LICENSE", "NOTICE", "ATTRIBUTION.md"]\n'
        b"[tool.release-test]\nreviewed = true\n"
    )
    files = {
        "pyproject.toml": pyproject,
        "README.md": b"Public client\n",
        "CHANGELOG.md": b"Release notes\n",
        "LICENSE": b"MIT license fixture\n",
        "NOTICE": b"Public notices\n",
        "ATTRIBUTION.md": b"Public attribution\n",
        "src/nori_rel/__init__.py": b'"""Public client fixture."""\n',
        "src/nori_rel/py.typed": b"",
        "docs/usage.md": b"Public usage\n",
        "examples/README.md": b"Synthetic examples\n",
        "tests/test_behavior.py": b"def test_behavior():\n    assert True\n",
        "scripts/public-package-files.txt": (
            b"nori_rel/__init__.py\nnori_rel/py.typed\n"
        ),
    }
    for name, content in files.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    git_environment = dict(os.environ) | {
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
    }
    for arguments in (
        ["init", "-q"],
        ["add", "."],
        [
            "-c",
            "user.name=Artifact Test",
            "-c",
            "user.email=artifact@example.test",
            "commit",
            "-qm",
            "Reviewed source",
        ],
    ):
        subprocess.run(["git", *arguments], cwd=source, env=git_environment, check=True)
    metadata = (
        b"Metadata-Version: 2.4\nName: nori-rel\nVersion: 0.3.0\n"
        b"License-Expression: MIT AND Apache-2.0\n"
        b"Provides-Extra: database\nProvides-Extra: snowflake\n"
        b"Provides-Extra: databricks\nProvides-Extra: local\n"
        b"Provides-Extra: agentic\n"
        b"License-File: LICENSE\nLicense-File: NOTICE\n"
        b"License-File: ATTRIBUTION.md\n\nPublic client fixture.\n"
    )
    wheel_files = {
        name.removeprefix("src/"): content
        for name, content in files.items()
        if name.startswith("src/")
    }
    wheel_files["nori_rel-0.3.0.dist-info/METADATA"] = metadata
    for name in ("LICENSE", "NOTICE", "ATTRIBUTION.md"):
        wheel_files[f"nori_rel-0.3.0.dist-info/licenses/{name}"] = files[name]
    case = SimpleNamespace(
        source=source,
        dist=dist,
        validator=validator,
        wheel_files=wheel_files,
        sdist_files=files
        | {
            "pyproject.toml": pyproject.split(b"\n", 1)[1],
            "pyproject.toml.orig": pyproject,
            "PKG-INFO": metadata,
        },
    )
    return case


def test_reviewed_archives_with_uv_build_normalization_pass(distribution_case):
    case = distribution_case
    write_distribution_archives(case)

    manifest = case.validator.validate_public_distributions(case.dist, case.source)

    assert manifest["name"] == "nori-rel"
    assert manifest["version"] == "0.3.0"
    assert manifest["package_files"] == ["nori_rel/__init__.py", "nori_rel/py.typed"]


def test_tampered_wheel_license_fails(distribution_case):
    case = distribution_case
    case.wheel_files["nori_rel-0.3.0.dist-info/licenses/LICENSE"] = b"unreviewed text"
    write_distribution_archives(case)

    with pytest.raises(ValueError, match=r"wheel license differs.*LICENSE"):
        case.validator.validate_public_distributions(case.dist, case.source)


def test_unreviewed_license_declaration_fails(distribution_case):
    case = distribution_case
    metadata_path = "nori_rel-0.3.0.dist-info/METADATA"
    case.wheel_files[metadata_path] = case.wheel_files[metadata_path].replace(
        b"License-File: LICENSE\n",
        b"License-File: LICENSE\nLicense-File: private.txt\n",
    )
    case.sdist_files["PKG-INFO"] = case.wheel_files[metadata_path]
    case.wheel_files["nori_rel-0.3.0.dist-info/licenses/private.txt"] = b"private text"
    write_distribution_archives(case)

    with pytest.raises(ValueError, match="wheel license inventory differs"):
        case.validator.validate_public_distributions(case.dist, case.source)


@pytest.mark.parametrize("directory", ["docs", "tests", "scripts", "examples"])
def test_unreviewed_archive_file_fails(distribution_case, directory):
    case = distribution_case
    case.sdist_files[f"{directory}/private.txt"] = b"synthetic private text"
    write_distribution_archives(case)

    with pytest.raises(ValueError, match="unexpected sdist member"):
        case.validator.validate_public_distributions(case.dist, case.source)


def test_staging_private_file_does_not_make_it_reviewed(distribution_case):
    case = distribution_case
    private_file = case.source / "docs/private.txt"
    private_file.write_bytes(b"synthetic private text")
    subprocess.run(["git", "add", "docs/private.txt"], cwd=case.source, check=True)
    case.sdist_files["docs/private.txt"] = private_file.read_bytes()
    write_distribution_archives(case)

    with pytest.raises(ValueError, match="unexpected sdist member"):
        case.validator.validate_public_distributions(case.dist, case.source)


def test_changed_worktree_and_archive_bytes_still_fail(distribution_case):
    case = distribution_case
    case.sdist_files["docs/usage.md"] = b"unreviewed replacement"
    (case.source / "docs/usage.md").write_bytes(case.sdist_files["docs/usage.md"])
    write_distribution_archives(case)

    with pytest.raises(ValueError, match=r"sdist bytes differ.*docs/usage\.md"):
        case.validator.validate_public_distributions(case.dist, case.source)


def test_tampering_both_package_archives_cannot_bypass_git_check(distribution_case):
    case = distribution_case
    case.sdist_files["src/nori_rel/__init__.py"] = b"unreviewed_code = True\n"
    case.wheel_files["nori_rel/__init__.py"] = case.sdist_files[
        "src/nori_rel/__init__.py"
    ]
    write_distribution_archives(case)

    with pytest.raises(ValueError, match=r"sdist bytes differ.*src/nori_rel/__init__"):
        case.validator.validate_public_distributions(case.dist, case.source)


@pytest.mark.parametrize(
    "name", ["docs/usage.md", "examples/README.md", "tests/test_behavior.py"]
)
def test_omitting_reviewed_included_files_fails(distribution_case, name):
    case = distribution_case
    del case.sdist_files[name]
    write_distribution_archives(case)

    with pytest.raises(ValueError, match="sdist omits reviewed source files"):
        case.validator.validate_public_distributions(case.dist, case.source)


@pytest.mark.parametrize(
    "addition",
    [b"\n# synthetic private comment\n", b'\nprivate_setting = "synthetic"\n'],
)
def test_unverified_project_normalization_fails(distribution_case, addition):
    case = distribution_case
    case.sdist_files["pyproject.toml"] += addition
    write_distribution_archives(case)

    with pytest.raises(ValueError, match="not verified uv_build normalization"):
        case.validator.validate_public_distributions(case.dist, case.source)


def test_normalized_project_cannot_change_boolean_to_integer(distribution_case):
    case = distribution_case
    case.sdist_files["pyproject.toml"] = case.sdist_files["pyproject.toml"].replace(
        b"reviewed = true", b"reviewed = 1"
    )
    write_distribution_archives(case)

    with pytest.raises(ValueError, match="not verified uv_build normalization"):
        case.validator.validate_public_distributions(case.dist, case.source)


@pytest.mark.parametrize("name", ["pyproject.toml.orig", "PKG-INFO"])
def test_generated_provenance_must_match_original(distribution_case, name):
    case = distribution_case
    case.sdist_files[name] += b"unreviewed content"
    write_distribution_archives(case)

    with pytest.raises(ValueError, match="differs from"):
        case.validator.validate_public_distributions(case.dist, case.source)


def test_normalized_project_requires_original_bytes(distribution_case):
    case = distribution_case
    del case.sdist_files["pyproject.toml.orig"]
    write_distribution_archives(case)

    with pytest.raises(ValueError, match="requires the reviewed original"):
        case.validator.validate_public_distributions(case.dist, case.source)


def test_unexpected_archive_directory_fails(distribution_case):
    case = distribution_case
    write_distribution_archives(case)
    path = case.dist / "nori_rel-0.3.0.tar.gz"
    with tarfile.open(path) as archive:
        members = [
            (member, archive.extractfile(member).read())
            for member in archive.getmembers()
        ]
    with tarfile.open(path, "w:gz") as archive:
        for member, content in members:
            archive.addfile(member, io.BytesIO(content))
        directory = tarfile.TarInfo("nori_rel-0.3.0/docs/private-directory")
        directory.type = tarfile.DIRTYPE
        archive.addfile(directory)

    with pytest.raises(ValueError, match="unexpected sdist directory"):
        case.validator.validate_public_distributions(case.dist, case.source)


@pytest.mark.parametrize(
    "requirement",
    [
        b"anthropic<2,>=1",
        b'openai<3,>=1.60; extra == "database"',
        b"duckdb<2,>=1.1",
        b'duckdb<2,>=1.1; extra == "agentic" or extra == "local"',
        b'duckdb<2,>=1.1; extra == "database" and python_version < "3.12"',
    ],
)
def test_optional_dependency_outside_its_extras_fails(distribution_case, requirement):
    case = distribution_case
    metadata_path = "nori_rel-0.3.0.dist-info/METADATA"
    case.wheel_files[metadata_path] = case.wheel_files[metadata_path].replace(
        b"License-File: LICENSE\n",
        b"Requires-Dist: " + requirement + b"\nLicense-File: LICENSE\n",
    )
    case.sdist_files["PKG-INFO"] = case.wheel_files[metadata_path]
    write_distribution_archives(case)

    with pytest.raises(ValueError, match="outside its extras"):
        case.validator.validate_public_distributions(case.dist, case.source)


@pytest.mark.parametrize(
    "requirement",
    [
        b'anthropic<2,>=1; extra == "agentic"',
        b'duckdb<2,>=1.1; extra == "database"',
        b'duckdb<2,>=1.1; extra == "agentic" or extra == "snowflake"',
    ],
)
def test_optional_dependency_under_its_extras_passes(distribution_case, requirement):
    case = distribution_case
    metadata_path = "nori_rel-0.3.0.dist-info/METADATA"
    case.wheel_files[metadata_path] = case.wheel_files[metadata_path].replace(
        b"License-File: LICENSE\n",
        b"Requires-Dist: " + requirement + b"\nLicense-File: LICENSE\n",
    )
    case.sdist_files["PKG-INFO"] = case.wheel_files[metadata_path]
    write_distribution_archives(case)

    case.validator.validate_public_distributions(case.dist, case.source)
