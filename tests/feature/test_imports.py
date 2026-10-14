"""Feature interfaces import without optional runtime stacks."""

import subprocess
import sys

import pytest

FORBIDDEN = (
    "fastdfs",
    "featuretools",
    "woodwork",
    "relbench",
    "relarena",
    "torch",
    "synthefy",
    "nori_rel.connectors",
    "nori_rel._runtime",
)
#: The agent search's own dependencies, from the ``agentic`` extra, load only
#: when a search is built, so a core install can import its interfaces.
AGENTIC_EXTRA = ("duckdb", "anthropic", "openai")


@pytest.mark.parametrize(
    "module",
    [
        "nori_rel.feature",
        "nori_rel.feature.dfs",
        "nori_rel.feature.dfs.noridfs",
        "nori_rel.feature.agentic",
    ],
)
def test_feature_packages_do_not_import_optional_stacks(module: str) -> None:
    forbidden = FORBIDDEN + AGENTIC_EXTRA
    subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sys\nimport {module}\n"
            f"for name in {forbidden!r}:\n"
            "    assert name not in sys.modules, name\n",
        ],
        check=True,
    )
