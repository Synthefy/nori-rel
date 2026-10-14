"""Exercise an installed artifact without relying on its source checkout."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.resources
import sys
from pathlib import Path


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
        name in sys.modules for name in ("torch", "fastdfs", "psycopg", "synthefy_nori")
    ):
        raise RuntimeError("base import eagerly loaded an optional dependency")
    connector_modules = {
        "database": "postgres",
        "snowflake": "snowflake",
        "databricks": "databricks",
    }
    for extra, module in connector_modules.items():
        if extra in extras:
            importlib.import_module(f"nori_rel.connectors.{module}")
    if extras.intersection(connector_modules):
        # Creating and closing a client must not contact a database or model service.
        with nori_rel.NoriRelClient(mode="hosted", api_key="artifact-smoke-test"):
            pass
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
