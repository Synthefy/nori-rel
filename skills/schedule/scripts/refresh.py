"""Predict again from a saved pipeline, with fresh rows and no feature search.

Usage:
    python refresh.py nori_rel_source.json pipeline.json \
        --as-of 2026-04-06 --observed-through 2026-04-06 \
        [--task nori_rel_task.json] [--out predictions.csv] [--query-filter ...]

``--as-of`` is the date to predict from and ``--observed-through`` the date
through which the database is verified complete. The task file supplies the
client mode, model and device the pipeline was saved with. Exit code 2 means
the pipeline no longer fits the source or the client; the message says why.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from nori_rel import (
    AwsSecret,
    DatabricksConnection,
    EnvironmentCredential,
    ForeignKeySchema,
    NoriRelClient,
    NoriRelPipeline,
    PipelineError,
    SnowflakeConnection,
)


def load_json(path: Path) -> dict:
    """Return the JSON object stored at ``path``."""
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def parse_time(text: str) -> datetime:
    """Return an aware UTC datetime from an ISO-8601 string such as 2026-03-01."""
    value = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def make_credential(source: dict):
    """Return the credential reference described by the source file."""
    credential = source["credential"]
    if "environment" in credential:
        return EnvironmentCredential(variable=credential["environment"])
    if "aws_secret" in credential:
        return AwsSecret(**credential["aws_secret"])
    if "rds_iam" in credential:
        # Imported here so the other credentials work with releases before it.
        from nori_rel import RdsIamToken

        return RdsIamToken(**credential["rds_iam"])
    raise ValueError('credential needs "environment", "aws_secret" or "rds_iam"')


def connect_source(client: NoriRelClient, source: dict):
    """Register the database described by a nori_rel_source.json file."""
    engine = source["engine"]
    connection = None
    if engine == "snowflake":
        connection = SnowflakeConnection(**source["connection"])
    elif engine == "databricks":
        connection = DatabricksConnection(**source["connection"])
    foreign_keys = {
        table: [ForeignKeySchema(**key) for key in keys]
        for table, keys in (source.get("foreign_keys") or {}).items()
    }
    return client.connect(
        engine=engine,
        credential=make_credential(source),
        schema_name=source.get("schema_name"),
        tables=source.get("tables"),
        time_columns=source.get("time_columns") or {},
        primary_keys=source.get("primary_keys") or {},
        foreign_keys=foreign_keys,
        connection=connection,
    )


def make_client(task: dict) -> NoriRelClient:
    """Return a client with the mode, model and device the task file names."""
    options = {"mode": task.get("mode") or "local"}
    if task.get("model"):
        options["model"] = task["model"]
    if task.get("device"):
        options["device"] = task["device"]
    return NoriRelClient(**options)


def main() -> int:
    """Replay the pipeline at --as-of and write the predictions."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", type=Path, help="nori_rel_source.json")
    parser.add_argument("pipeline", type=Path, help="pipeline.json")
    parser.add_argument("--as-of", required=True, help="date to predict from")
    parser.add_argument(
        "--observed-through", required=True, help="verified completeness date"
    )
    parser.add_argument("--task", type=Path, help="nori_rel_task.json")
    parser.add_argument("--out", type=Path, help="default predictions_<as-of>.csv")
    parser.add_argument(
        "--query-filter", help="PQL entity filter, e.g. \"users.country='CA'\""
    )
    arguments = parser.parse_args()
    source = load_json(arguments.source)
    task = load_json(arguments.task) if arguments.task else {}
    as_of = parse_time(arguments.as_of)
    out = arguments.out or Path(f"predictions_{as_of:%Y-%m-%d}.csv")
    pipeline = NoriRelPipeline.load(arguments.pipeline)
    try:
        with make_client(task) as client:
            database = connect_source(client, source)
            frame = client.predict(
                source=database,
                pipeline=pipeline,
                as_of=as_of,
                observed_through=parse_time(arguments.observed_through),
                query_filter=arguments.query_filter,
                return_details=True,
            )
    except PipelineError as error:
        print(f"pipeline {pipeline.pipeline_id} cannot run: {type(error).__name__}")
        print(error)
        print("Fix the source or client, or predict again and save a new pipeline.")
        return 2
    frame.predictions.to_csv(out, index=False)
    metadata = frame.metadata
    print(
        f"{len(frame.predictions)} predictions from {metadata.get('usable_rows')} "
        f"context rows; pipeline {metadata.get('pipeline_id')}; "
        f"context sha256 {metadata.get('context_rows_sha256')}"
    )
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
