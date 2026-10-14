"""Assert the grammar artefacts stay consistent, without needing Java."""

import re
from hashlib import sha256
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[2] / "src/nori_rel/pql"
VENDOR = PACKAGE / "_vendor/kumo"
_MANIFEST_ROW = re.compile(r"^([0-9a-f]{64})  (\S+)$", re.MULTILINE)


def digest(path: Path) -> str:
    """Return one file's SHA-256, matching `shasum -a 256` output."""
    return sha256(path.read_bytes()).hexdigest()


def rows(text: str) -> list[tuple[str, str]]:
    """Return (hash, relative path) pairs from a checksum manifest."""
    return _MANIFEST_ROW.findall(text)


def test_vendored_grammar_still_matches_upstream():
    """SOURCE.md claims the vendored file is unmodified; hold it to that."""
    recorded = rows((VENDOR / "SOURCE.md").read_text())
    assert recorded, "SOURCE.md lost its hash manifest"
    for expected, name in recorded:
        assert (VENDOR / name).is_file(), f"SOURCE.md lists a missing file: {name}"
        assert digest(VENDOR / name) == expected, f"{name} differs from upstream"


def test_source_md_lists_every_file_it_ships():
    """A manifest that omits a shipped file would not detect editing it."""
    listed = {name for _, name in rows((VENDOR / "SOURCE.md").read_text())}
    shipped = {p.name for p in VENDOR.iterdir() if p.name != "SOURCE.md"}
    assert listed == shipped, (
        "SOURCE.md and _vendor/kumo/ disagree about what is shipped"
    )


@pytest.mark.parametrize("name", ["LICENSE", "NOTICE"])
def test_apache_notices_ship_with_the_grammar(name):
    """Apache-2.0 requires these alongside the redistributed source."""
    assert (VENDOR / name).read_text().strip()


def test_generated_parser_matches_the_grammar_it_came_from():
    """Catch a grammar edit that was never regenerated.

    The regeneration recipe in _vendor/kumo/SOURCE.md rewrites CHECKSUMS.txt
    last, so a stale entry here means the committed parser and the committed
    grammar have drifted apart. Java only reproduces the parser; this only
    needs hashlib, so it runs in the ordinary test job.
    """
    recorded = rows((PACKAGE / "grammar/CHECKSUMS.txt").read_text())
    assert len(recorded) == 3
    for expected, name in recorded:
        assert digest(PACKAGE / name) == expected, (
            f"{name} changed without regenerating; see _vendor/kumo/SOURCE.md"
        )


def test_our_grammar_is_a_real_delta_from_upstream():
    """The fork exists to carry changes; an identical copy would be dead weight."""

    def rules(path: Path) -> list[str]:
        return [
            line
            for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("//")
        ]

    assert rules(PACKAGE / "grammar/PQLGrammar.g4") != rules(VENDOR / "PQLGrammar.g4")
