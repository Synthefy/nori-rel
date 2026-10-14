"""Validate the public distributions and record their immutable release hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tarfile
import tomllib
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

REQUIRED_EXTRAS = {"database", "snowflake", "databricks", "local", "agentic"}
FORBIDDEN_COMPONENTS = {
    "relarena",
    "finetune",
    "benchmarks",
    "notebooks",
    "__pycache__",
}
#: Dependencies only the opt-in agent search uses. The wheel may require them
#: under the ``agentic`` extra and nowhere else, so a core install stays light.
AGENTIC_ONLY_REQUIREMENTS = {"anthropic", "duckdb", "openai"}
SOURCE_ROOT_FILES = {
    "pyproject.toml",
    "README.md",
    "LICENSE",
    "NOTICE",
    "ATTRIBUTION.md",
    "CHANGELOG.md",
}
SOURCE_DIRECTORIES = ("src/nori_rel/", "tests/", "scripts/", "docs/", "examples/")
GENERATED_SOURCE_FILES = {"PKG-INFO", "pyproject.toml.orig"}


def read_committed_distribution_sources(source: Path) -> dict[str, bytes]:
    """Read the complete distribution input inventory from the reviewed HEAD."""
    entries = subprocess.check_output(
        ["git", "ls-tree", "-rz", "--full-tree", "HEAD"], cwd=source
    )
    source_bytes_by_path = {}
    for entry in entries.split(b"\0"):
        if not entry:
            continue
        attributes, raw_name = entry.split(b"\t", 1)
        mode, object_type, object_id = attributes.decode("ascii").split()
        name = raw_name.decode("utf-8")
        if name not in SOURCE_ROOT_FILES and not name.startswith(SOURCE_DIRECTORIES):
            continue
        validate_archive_member_path(name)
        if mode not in {"100644", "100755"} or object_type != "blob":
            raise ValueError(f"distribution source is not a regular Git file: {name}")
        source_bytes_by_path[name] = subprocess.check_output(
            ["git", "cat-file", "blob", object_id], cwd=source
        )
    return source_bytes_by_path


def toml_values_match(reviewed: object, normalized: object) -> bool:
    """Compare parsed TOML without treating booleans, integers, and floats alike."""
    if type(reviewed) is not type(normalized):
        return False
    if isinstance(reviewed, dict):
        return reviewed.keys() == normalized.keys() and all(
            toml_values_match(value, normalized[key]) for key, value in reviewed.items()
        )
    if isinstance(reviewed, list):
        return len(reviewed) == len(normalized) and all(
            toml_values_match(left, right)
            for left, right in zip(reviewed, normalized, strict=True)
        )
    return reviewed == normalized


def contains_toml_comment(document: str) -> bool:
    """Find comments outside TOML strings, including multiline string forms."""
    delimiter = ""
    offset = 0
    while offset < len(document):
        character = document[offset]
        if not delimiter:
            if character == "#":
                return True
            if character in {"'", '"'}:
                delimiter = character * (
                    3 if document.startswith(character * 3, offset) else 1
                )
                offset += len(delimiter)
                continue
        elif delimiter.startswith('"') and character == "\\":
            offset += 2
            continue
        elif document.startswith(delimiter, offset):
            offset += len(delimiter)
            if len(delimiter) == 3:
                # Four/five ending quotes include one/two literal quote characters.
                while offset < len(document) and document[offset] == delimiter[0]:
                    offset += 1
            delimiter = ""
            continue
        offset += 1
    return False


def get_requirement_name(requirement: str) -> str:
    """Return a Requires-Dist entry's normalized project name."""
    name = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
    if name is None:
        raise ValueError(f"unreadable wheel requirement: {requirement}")
    return re.sub(r"[-_.]+", "-", name.group(1)).lower()


def is_agentic_extra_requirement(requirement: str) -> bool:
    """Return whether a Requires-Dist entry applies only with the agentic extra."""
    _, separator, marker = requirement.partition(";")
    return bool(separator) and (
        re.fullmatch(r"\s*extra\s*==\s*[\"']agentic[\"']\s*", marker) is not None
    )


def validate_archive_member_path(name: str) -> None:
    """Reject private content and unsafe archive paths without extracting them."""
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"unsafe archive member: {name}")
    if FORBIDDEN_COMPONENTS.intersection(path.parts):
        raise ValueError(f"internal content in public distribution: {name}")
    if any(part.startswith(".env") for part in path.parts):
        raise ValueError(f"environment file in public distribution: {name}")
    if path.suffix in {".pyc", ".pt", ".safetensors", ".pem", ".key", ".ipynb"}:
        raise ValueError(f"unexpected binary or credential file: {name}")


def validate_public_distributions(dist: Path, source: Path) -> dict:
    """Check metadata, the reviewed package inventory, and wheel/sdist parity."""
    committed_sources = read_committed_distribution_sources(source)
    original_pyproject = committed_sources["pyproject.toml"]
    source_configuration = tomllib.loads(original_pyproject.decode("utf-8"))
    project = source_configuration["project"]
    version = project["version"]
    expected = {
        line
        for line in committed_sources["scripts/public-package-files.txt"]
        .decode("utf-8")
        .splitlines()
        if line and not line.startswith("#")
    }
    wheel = dist / f"nori_rel-{version}-py3-none-any.whl"
    sdist = dist / f"nori_rel-{version}.tar.gz"
    artifacts = {path for path in dist.iterdir() if not path.name.startswith(".")}
    if artifacts != {wheel, sdist}:
        raise ValueError(
            "distribution directory must contain only this wheel and sdist"
        )
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("duplicate wheel members")
        for name in names:
            validate_archive_member_path(name)
            if not name.startswith(("nori_rel/", f"nori_rel-{version}.dist-info/")):
                raise ValueError(f"unexpected wheel member: {name}")
        package = {
            name: archive.read(name)
            for name in names
            if name.startswith("nori_rel/") and not name.endswith("/")
        }
        if set(package) != expected:
            raise ValueError(
                f"package inventory changed; added={sorted(set(package) - expected)}, "
                f"missing={sorted(expected - set(package))}"
            )
        wheel_metadata = archive.read(f"nori_rel-{version}.dist-info/METADATA")
        metadata = BytesParser().parsebytes(wheel_metadata)
        if metadata["Name"] != "nori-rel" or metadata["Version"] != version:
            raise ValueError("wheel name/version does not match pyproject.toml")
        if metadata["License-Expression"] != "MIT AND Apache-2.0":
            raise ValueError("project license metadata is missing")
        if set(metadata.get_all("Provides-Extra", [])) != REQUIRED_EXTRAS:
            raise ValueError("published extras do not match the public client contract")
        license_files = metadata.get_all("License-File", [])
        if not {"LICENSE", "NOTICE", "ATTRIBUTION.md"}.issubset(license_files):
            raise ValueError("project license or notices are missing")
        if len(license_files) != len(set(license_files)) or set(license_files) != set(
            project["license-files"]
        ):
            raise ValueError("wheel license inventory differs from reviewed project")
        for name in license_files:
            contents = archive.read(f"nori_rel-{version}.dist-info/licenses/{name}")
            if contents != committed_sources[name]:
                raise ValueError(
                    f"wheel license differs from reviewed Git source: {name}"
                )
        for requirement in metadata.get_all("Requires-Dist", []):
            if any(
                name in requirement.lower() for name in ("relarena", "relbench", "git+")
            ):
                raise ValueError(f"internal dependency in wheel: {requirement}")
            if get_requirement_name(
                requirement
            ) in AGENTIC_ONLY_REQUIREMENTS and not is_agentic_extra_requirement(
                requirement
            ):
                raise ValueError(
                    f"agent-search dependency outside the agentic extra: {requirement}"
                )
        if any(name.endswith("/entry_points.txt") for name in names):
            raise ValueError("the public Python client must not register research CLIs")
        if "nori_rel/py.typed" not in package:
            raise ValueError("typed public API marker is missing")
    with tarfile.open(sdist) as archive:
        source_package = {}
        seen = set()
        sdist_bytes_by_path = {}
        permitted_names = set(committed_sources) | GENERATED_SOURCE_FILES
        permitted_directories = {
            parent.as_posix()
            for name in permitted_names
            for parent in PurePosixPath(name).parents
            if parent.as_posix() != "."
        }
        expected_root = f"nori_rel-{version}"
        for member in archive.getmembers():
            validate_archive_member_path(member.name)
            if member.name in seen or not (member.isfile() or member.isdir()):
                raise ValueError(f"duplicate or linked sdist member: {member.name}")
            seen.add(member.name)
            prefix, separator, name = member.name.partition("/")
            if prefix != expected_root:
                raise ValueError("sdist root does not match the release version")
            if member.isdir():
                if separator and name not in permitted_directories:
                    raise ValueError(f"unexpected sdist directory: {member.name}")
                continue
            if not separator or name not in permitted_names:
                raise ValueError(f"unexpected sdist member: {member.name}")
            stream = archive.extractfile(member)
            assert stream is not None
            contents = stream.read()
            sdist_bytes_by_path[name] = contents
            if name == "PKG-INFO":
                if contents != wheel_metadata:
                    raise ValueError("sdist PKG-INFO differs from wheel METADATA")
            elif name == "pyproject.toml.orig":
                if contents != original_pyproject:
                    raise ValueError(
                        "sdist pyproject.toml.orig differs from reviewed Git source"
                    )
            elif name == "pyproject.toml" and contents != original_pyproject:
                normalized_document = contents.decode("utf-8")
                if (
                    source_configuration.get("build-system", {}).get("build-backend")
                    != "uv_build"
                    or not toml_values_match(
                        source_configuration, tomllib.loads(normalized_document)
                    )
                    or contains_toml_comment(normalized_document)
                ):
                    raise ValueError(
                        "sdist pyproject.toml is not verified uv_build normalization"
                    )
            elif contents != committed_sources[name]:
                raise ValueError(f"sdist bytes differ from reviewed Git source: {name}")
            if name.startswith("src/nori_rel/"):
                source_package[name.removeprefix("src/")] = contents
        missing = (set(committed_sources) | {"PKG-INFO"}) - set(sdist_bytes_by_path)
        if missing:
            raise ValueError(f"sdist omits reviewed source files: {sorted(missing)}")
        if (
            sdist_bytes_by_path["pyproject.toml"] != original_pyproject
            and "pyproject.toml.orig" not in sdist_bytes_by_path
        ):
            raise ValueError("normalized pyproject.toml requires the reviewed original")
        if source_package != package:
            raise ValueError("wheel and sdist package contents differ")
        if not any(name.startswith("tests/") for name in sdist_bytes_by_path):
            raise ValueError("source distribution omits the public regression tests")
    return {
        "name": "nori-rel",
        "version": version,
        "files": {
            path.name: {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size": path.stat().st_size,
            }
            for path in sorted(artifacts)
        },
        "package_files": sorted(package),
    }


def main() -> None:
    """Validate a build and optionally write its release manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dist", type=Path)
    parser.add_argument("--source", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    manifest = validate_public_distributions(args.dist, args.source)
    manifest["commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.source, text=True
    ).strip()
    if args.manifest:
        args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Validated nori-rel {manifest['version']}: wheel, sdist, metadata, inventory"
    )


if __name__ == "__main__":
    main()
