"""Predict from synthetic PostgreSQL data, save a pipeline, and replay it.

Set DATABASE_URL to a disposable TLS-enabled PostgreSQL database where your
identity can create schemas. Run with --mode hosted to use the Synthefy API;
local inference is the default. See examples/README.md for installation.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from uuid import uuid4

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict
from synthetic_data import (
    CUSTOMER_SPEND_QUERY,
    HISTORICAL_ANCHOR_TIMES,
    SCORING_TIME,
    make_customer_order_fixture,
)

from nori_rel import EnvironmentCredential, NoriRelClient, NoriRelPipeline


def run_example(mode: str, device: str | None = None) -> None:
    """Own and clean one schema, including when prediction or replay fails."""
    connection_options = conninfo_to_dict(os.environ["DATABASE_URL"])
    connection_options.setdefault("sslmode", "require")
    schema_name = f"nori_rel_example_{uuid4().hex}"
    connection_request = dict(
        engine="postgresql",
        credential=EnvironmentCredential(variable="DATABASE_URL"),
        schema_name=schema_name,
        tables=["users", "orders"],
        time_columns={"orders": "event_time"},
    )
    users, orders = make_customer_order_fixture()
    with psycopg.connect(**connection_options, autocommit=True) as database:
        database.execute(
            sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema_name))
        )
        try:
            database.execute(
                sql.SQL("SET search_path TO {}").format(sql.Identifier(schema_name))
            )
            database.execute(
                "CREATE TABLE users (user_id bigint PRIMARY KEY, "
                "created_at timestamptz NOT NULL, country text NOT NULL)"
            )
            database.execute(
                "CREATE TABLE orders (order_id bigint PRIMARY KEY, "
                "user_id bigint REFERENCES users(user_id), "
                "event_time timestamptz NOT NULL, amount numeric(12, 2) NOT NULL)"
            )
            with database.cursor() as cursor:
                cursor.executemany("INSERT INTO users VALUES (%s, %s, %s)", users)
                cursor.executemany("INSERT INTO orders VALUES (%s, %s, %s, %s)", orders)

            with tempfile.TemporaryDirectory(prefix="nori_rel_pipeline_") as directory:
                pipeline_path = Path(directory) / "customer_spend.json"
                with NoriRelClient(mode=mode, device=device) as client:
                    details = client.predict(
                        source=client.connect(**connection_request),
                        query=CUSTOMER_SPEND_QUERY,
                        anchor_times=HISTORICAL_ANCHOR_TIMES,
                        anchor_time=SCORING_TIME,
                        observed_through=SCORING_TIME,
                        entity_created_at="created_at",
                        return_details=True,
                    )
                    if details.pipeline is None:
                        raise RuntimeError("The prediction did not produce a pipeline")
                    details.pipeline.save(pipeline_path)

                # New client, new connection pool, and the saved feature recipe.
                with NoriRelClient(mode=mode, device=device) as client:
                    replayed = client.predict(
                        source=client.connect(**connection_request),
                        pipeline=NoriRelPipeline.load(pipeline_path),
                        as_of=SCORING_TIME,
                        observed_through=SCORING_TIME,
                        expected_context_sha256=details.metadata["context_rows_sha256"],
                    )
                if (
                    replayed.attrs["feature_matrix_sha256"]
                    != details.metadata["feature_matrix_sha256"]
                ):
                    raise RuntimeError("Replay changed the synthetic feature matrix")
                print(
                    replayed[["entity_id", "prediction"]].head().to_string(index=False)
                )
                print(f"Replayed {len(replayed)} rows; context and features match.")
        finally:
            database.execute(
                sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema_name))
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("local", "hosted"), default="local")
    parser.add_argument("--device", help="Local inference device, for example cpu")
    arguments = parser.parse_args()
    run_example(arguments.mode, arguments.device)
