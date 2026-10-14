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


@pytest.mark.parametrize("module", ["nori_rel.feature", "nori_rel.feature.dfs"])
def test_dfs_packages_do_not_import_optional_stacks(module: str) -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sys\nimport {module}\n"
            f"for name in {FORBIDDEN!r}:\n"
            "    assert name not in sys.modules, name\n",
        ],
        check=True,
    )
