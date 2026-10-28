"""Exercise an installed artifact without relying on its source checkout."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.resources
import sys
from pathlib import Path
from types import SimpleNamespace


def check_feature_generation() -> None:
    """Exercise the installed adapter and its optional dependency imports."""
    import pandas as pd

    from nori_rel.feature.dfs import CUTOFF_COLUMN
    from nori_rel.feature.dfs.fastdfs import FastDFS
    from nori_rel.feature.dfs.noridfs import NativeFeatureConfig, NoriDFS

    snapshot = SimpleNamespace(
        tables={"users": pd.DataFrame({"id": [1, 2], "age": [40.0, 30.0]})},
        primary_keys={"users": ["id"]},
        foreign_keys=[],
        time_columns={},
    )
    context_rows = pd.DataFrame(
        {
            "id": [1, 2],
            CUTOFF_COLUMN: pd.to_datetime(["2026-01-01"] * 2, utc=True),
            "row_id": ["context:1", "context:2"],
        }
    )
    query_rows = context_rows.iloc[::-1].copy()
    query_rows["row_id"] = ["query:2", "query:1"]
    backends = (
        FastDFS(max_depth=1),
        NoriDFS(NativeFeatureConfig(max_depth=1), horizon=pd.Timedelta(days=30)),
    )
    for backend in backends:
        features = backend.compute(
            snapshot, context_rows, query_rows, "users", row_id_column="row_id"
        )
        if len(features.train) != 2 or len(features.query) != 2:
            raise RuntimeError("feature generation changed the requested row counts")
        if not features.feature_names:
            raise RuntimeError("feature generation returned no model inputs")
        pd.testing.assert_frame_equal(
            features.train.iloc[::-1].reset_index(drop=True), features.query
        )


def main() -> None:
    """Check an independently installed optional dependency selection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extras", default="")
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    extras = set(filter(None, args.extras.split(",")))
    import nori_rel
    from nori_rel.pql.parser import parse_pql

    if importlib.metadata.version("nori-rel") != args.version:
        raise RuntimeError("installed package version differs from the built artifact")
    if "site-packages" not in Path(nori_rel.__file__).parts:
        raise RuntimeError("smoke test imported the checkout instead of the artifact")
    parse_pql("PREDICT COUNT(orders.*,0,30,days) FOR EACH users.user_id")
    importlib.resources.files("nori_rel").joinpath("py.typed").read_bytes()
    importlib.resources.files("nori_rel.pql").joinpath(
        "grammar/PQLGrammar.g4"
    ).read_bytes()
    if not extras and any(
        name in sys.modules
        for name in (
            "torch",
            "fastdfs",
            "psycopg",
            "pymysql",
            "synthefy_nori",
            "duckdb",
            "anthropic",
            "openai",
        )
    ):
        raise RuntimeError("base import eagerly loaded an optional dependency")
    connector_modules = {
        "database": "postgres",
        "mysql": "mysql",
        "snowflake": "snowflake",
        "databricks": "databricks",
    }
    for extra, module in connector_modules.items():
        if extra in extras:
            importlib.import_module(f"nori_rel.connectors.{module}")
    if extras.intersection(connector_modules):
        check_feature_generation()
        # Creating and closing a client must not contact a database or model service.
        with nori_rel.NoriRelClient(mode="hosted", api_key="artifact-smoke-test"):
            pass
    if "agentic" in extras:
        import duckdb

        for module in ("anthropic", "openai"):
            importlib.import_module(module)
        importlib.import_module("nori_rel.feature.agentic.agent_claude")
        importlib.import_module("nori_rel.feature.agentic.agent_openai")
        if duckdb.sql("SELECT 1").fetchone() != (1,):
            raise RuntimeError("DuckDB did not run a query")
        if not callable(nori_rel.AgenticSearch):
            raise RuntimeError("the agent search is not exposed")
    if "local" in extras:
        from synthefy_nori import NoriRegressor

        if not callable(NoriRegressor):
            raise RuntimeError("local runtime did not expose its supported API")
        if extras.intersection(connector_modules):
            with nori_rel.NoriRelClient(mode="local", device="cpu"):
                pass
    print(f"Installed artifact OK: {args.version}, extras={','.join(sorted(extras))}")


if __name__ == "__main__":
    main()
