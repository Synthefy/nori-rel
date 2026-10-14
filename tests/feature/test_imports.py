"""Check that the old feature import paths alias the new DFS package."""

import subprocess
import sys
from importlib.util import find_spec

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


@pytest.mark.parametrize(
    "module", ["nori_rel.feature.dfs", "nori_rel.feature.dfs.noridfs"]
)
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


def test_old_fastdfs_exports_alias_the_new_implementation() -> None:
    if find_spec("fastdfs") is None:
        pytest.skip("FastDFS is an optional backend")
    from nori_rel._features import _CUTOFF, FeatureMatrices, RelationalFeatureEngine
    from nori_rel.feature.dfs import CUTOFF_COLUMN
    from nori_rel.feature.dfs import FeatureMatrices as NewMatrices
    from nori_rel.feature.dfs.fastdfs import FastDFS

    assert RelationalFeatureEngine is FastDFS
    assert FeatureMatrices is NewMatrices
    assert _CUTOFF == CUTOFF_COLUMN
