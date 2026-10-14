"""Validate the public distributions and record their immutable release hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
import tomllib
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

REQUIRED_EXTRAS = {"database", "snowflake", "databricks", "local"}
FORBIDDEN_COMPONENTS = {
    "relarena",
    "agentic",
    "finetune",
    "noridfs",
    "benchmarks",
    "notebooks",
    "__pycache__",
}
SOURCE_ROOT_FILES = {
    "PKG-INFO",
    "pyproject.toml",
    "pyproject.toml.orig",
    "README.md",
    "LICENSE",
    "NOTICE",
    "ATTRIBUTION.md",
    "CHANGELOG.md",
}


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
    project = tomllib.loads((source / "pyproject.toml").read_text())["project"]
    version = project["version"]
    expected = {
        line
        for line in (source / "scripts/public-package-files.txt")
        .read_text()
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
        metadata = BytesParser().parsebytes(
            archive.read(f"nori_rel-{version}.dist-info/METADATA")
        )
        if metadata["Name"] != "nori-rel" or metadata["Version"] != version:
            raise ValueError("wheel name/version does not match pyproject.toml")
        if metadata["License-Expression"] != "MIT AND Apache-2.0":
            raise ValueError("project license metadata is missing")
        if set(metadata.get_all("Provides-Extra", [])) != REQUIRED_EXTRAS:
            raise ValueError("published extras do not match the public client contract")
        license_files = metadata.get_all("License-File", [])
        if not {"LICENSE", "NOTICE", "ATTRIBUTION.md"}.issubset(license_files):
            raise ValueError("project license or notices are missing")
        for name in license_files:
            archive.read(f"nori_rel-{version}.dist-info/licenses/{name}")
        for requirement in metadata.get_all("Requires-Dist", []):
            if any(
                name in requirement.lower()
                for name in ("relarena", "relbench", "anthropic", "openai", "git+")
            ):
                raise ValueError(f"internal dependency in wheel: {requirement}")
        if any(name.endswith("/entry_points.txt") for name in names):
            raise ValueError("the public Python client must not register research CLIs")
        if "nori_rel/py.typed" not in package:
            raise ValueError("typed public API marker is missing")
    with tarfile.open(sdist) as archive:
        source_package = {}
        seen = set()
        source_tests = []
        for member in archive.getmembers():
            validate_archive_member_path(member.name)
            if member.name in seen or not (member.isfile() or member.isdir()):
                raise ValueError(f"duplicate or linked sdist member: {member.name}")
            seen.add(member.name)
            if member.isdir():
                continue
            prefix, name = member.name.split("/", 1)
            if prefix != f"nori_rel-{version}":
                raise ValueError("sdist root does not match the release version")
            if name.startswith("src/nori_rel/"):
                stream = archive.extractfile(member)
                assert stream is not None
                source_package[name.removeprefix("src/")] = stream.read()
            elif name.startswith(("tests/", "scripts/", "docs/", "examples/")):
                if name.startswith("tests/"):
                    source_tests.append(name)
            elif name not in SOURCE_ROOT_FILES:
                raise ValueError(f"unexpected sdist member: {name}")
        if source_package != package:
            raise ValueError("wheel and sdist package contents differ")
        if not source_tests:
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
