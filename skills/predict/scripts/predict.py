"""Run one Nori-Rel prediction from a source file and a task file.

Usage:
    python predict.py nori_rel_source.json nori_rel_task.json [--out-dir DIR]

Writes predictions.csv, context.csv (the labelled rows Nori learned from),
features.csv, feature_table.csv, metadata.json and, when the prediction can be
saved, pipeline.json into the output directory (default: nori_rel_output).
Credentials come from the environment or AWS Secrets Manager, as the source file
says, and never enter the output.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from nori_rel import (
    AwsSecret,
    DatabricksConnection,
    EnvironmentCredential,
    ForeignKeySchema,
    NoriRelClient,
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
    raise ValueError('credential needs "environment" or "aws_secret"')


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


def make_feature_strategy(task: dict):
    """Return the feature search: "fastdfs", "noridfs" or a configured agent."""
    strategy = task.get("feature_strategy") or "fastdfs"
    if isinstance(strategy, str):
        return strategy
    from nori_rel import AgenticSearch

    options = {
        key: strategy[key] for key in ("model", "max_cost_usd") if key in strategy
    }
    if "prices" in strategy:
        from nori_rel.feature.agentic.usage import TokenPrices

        options["prices"] = TokenPrices(**strategy["prices"])
    if strategy.get("agent", "claude") == "openai":
        options.pop("max_cost_usd", None)
        options.pop("prices", None)
        if "base_url" in strategy:
            options["base_url"] = strategy["base_url"]
        return AgenticSearch.openai(**options)
    return AgenticSearch.claude(**options)


def get_offset(spec: dict) -> pd.DateOffset:
    """Return a pandas offset for {"end": 30, "unit": "days"} or a spacing."""
    return pd.DateOffset(**{spec["unit"]: spec.get("value", spec.get("end"))})


def get_past_dates(task: dict, anchor_time: datetime) -> list[datetime]:
    """Return the historical anchors: past_dates steps before anchor_time."""
    step = get_offset(task.get("anchor_spacing") or task["window"])
    anchor = pd.Timestamp(anchor_time)
    return [
        (anchor - k * step).to_pydatetime() for k in range(1, task["past_dates"] + 1)
    ]


def build_request(task: dict, anchor_time: datetime | None) -> dict:
    """Return the predict() keyword arguments for the task at anchor_time."""
    request = {"query": task["query"]}
    for key in ("output_type", "quantiles", "query_filter", "metric", "task"):
        if task.get(key) is not None:
            request[key] = task[key]
    if task.get("max_context_rows"):
        request["max_context_rows"] = task["max_context_rows"]
    if anchor_time is None:
        return request  # a static target takes no dates
    observed_through = parse_time(task["observed_through"])
    request.update(
        anchor_times=get_past_dates(task, anchor_time),
        anchor_time=anchor_time,
        observed_through=min(anchor_time, observed_through),
        entity_created_at=task["entity_created_at"],
    )
    return request


def write_outputs(details, out_dir: Path) -> None:
    """Write the prediction's tables, metadata and pipeline to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    details.predictions.to_csv(out_dir / "predictions.csv", index=False)
    details.context.to_csv(out_dir / "context.csv", index=False)
    details.features.to_csv(out_dir / "features.csv")
    details.get_feature_table().to_csv(out_dir / "feature_table.csv", index=False)
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(details.metadata, file, indent=2, default=str)
    if details.pipeline is not None:
        details.pipeline.save(out_dir / "pipeline.json")


def print_summary(details) -> None:
    """Print what Nori learned from, what it scored, and the predictions."""
    metadata = details.metadata
    print(
        f"\ntask {metadata['task']}; model {metadata['model']}; "
        f"features {metadata['features']} ({metadata.get('feature_count')} columns)"
    )
    print(
        f"context rows: {metadata.get('usable_rows')} usable of "
        f"{metadata.get('candidate_rows')} candidates "
        f"({metadata.get('dropped_unknown_labels')} unknown labels dropped)"
    )
    if "anchor_time" in details.context:
        per_anchor = details.context.groupby("anchor_time")["label"].agg(
            ["count", "mean"]
        )
        print("labelled rows per past date:")
        print(per_anchor.to_string())
    print(f"scored rows: {len(details.predictions)}")
    print(details.predictions["prediction"].describe().to_string())
    if details.pipeline is not None:
        print(f"pipeline {details.pipeline.pipeline_id} saved")
    else:
        print("no pipeline: this prediction cannot be replayed from a file")


def main() -> int:
    """Run the prediction and write its outputs."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", type=Path, help="nori_rel_source.json")
    parser.add_argument("task", type=Path, help="nori_rel_task.json")
    parser.add_argument("--out-dir", type=Path, default=Path("nori_rel_output"))
    arguments = parser.parse_args()
    source = load_json(arguments.source)
    task = load_json(arguments.task)
    anchor_time = parse_time(task["anchor_time"]) if task.get("anchor_time") else None
    request = build_request(task, anchor_time)
    with make_client(task) as client:
        database = connect_source(client, source)
        details = client.predict(
            source=database,
            feature_strategy=make_feature_strategy(task),
            agent_instructions=task.get("agent_instructions"),
            return_details=True,
            on_event=lambda event: print(f"{event.type}: {event.message}"),
            **request,
        )
    write_outputs(details, arguments.out_dir)
    print_summary(details)
    print(f"\noutputs in {arguments.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
