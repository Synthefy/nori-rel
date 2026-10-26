"""The agent skills in skills/ stay consistent with each other and the plugin."""

from __future__ import annotations

import json
import py_compile
import re
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
SKILLS = REPOSITORY / "skills"
SKILL_FILES = sorted(SKILLS.glob("*/SKILL.md"))
LINK = re.compile(r"\[[^\]]*\]\(([^)#\s]+)(?:#[^)]*)?\)")


def read_frontmatter(path: Path) -> dict[str, str]:
    """Return the key: value pairs between the leading --- lines."""
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n"), f"{path} has no frontmatter"
    body = text.split("---\n", 2)[1]
    fields = {}
    for line in body.splitlines():
        key, separator, value = line.partition(":")
        assert separator, f"{path}: frontmatter line without a colon: {line!r}"
        fields[key.strip()] = value.strip()
    return fields


@pytest.mark.parametrize("path", SKILL_FILES, ids=lambda p: p.parent.name)
def test_skill_frontmatter_names_its_folder(path: Path) -> None:
    fields = read_frontmatter(path)
    assert fields["name"] == path.parent.name
    assert 20 < len(fields["description"]) <= 1024
    assert set(fields) <= {"name", "description"}


@pytest.mark.parametrize("path", SKILL_FILES, ids=lambda p: p.parent.name)
def test_skill_links_resolve(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    for target in LINK.findall(text):
        if target.startswith(("http://", "https://")):
            continue
        assert (path.parent / target).resolve().exists(), f"{path}: {target}"


@pytest.mark.parametrize(
    "script",
    sorted(SKILLS.glob("*/scripts/*.py")),
    ids=lambda p: f"{p.parent.parent.name}/{p.name}",
)
def test_skill_scripts_compile_and_document_usage(script: Path) -> None:
    py_compile.compile(str(script), doraise=True)
    assert "Usage:" in script.read_text(encoding="utf-8")


def test_every_skill_is_listed_in_the_skills_readme() -> None:
    readme = (SKILLS / "README.md").read_text(encoding="utf-8")
    for path in SKILL_FILES:
        assert f"`{path.parent.name}`" in readme, path.parent.name


def test_plugin_manifest_matches_marketplace_entry() -> None:
    plugin = json.loads((REPOSITORY / ".claude-plugin/plugin.json").read_text())
    marketplace = json.loads(
        (REPOSITORY / ".claude-plugin/marketplace.json").read_text()
    )
    entries = {entry["name"]: entry for entry in marketplace["plugins"]}
    assert plugin["name"] in entries
    assert entries[plugin["name"]]["source"] == "./"
    assert re.fullmatch(r"[a-z0-9-]+", plugin["name"])
    assert re.fullmatch(r"[a-z0-9-]+", marketplace["name"])
