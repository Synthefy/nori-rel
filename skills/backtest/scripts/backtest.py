"""Check a Nori-Rel task on windows whose outcome is already known.

Usage:
    python backtest.py nori_rel_source.json nori_rel_task.json \
        [--folds 3] [--holdout 200] [--out-dir DIR]

A temporal task is predicted from each of ``--folds`` earlier cutoffs, one
label window apart, using only data up to that cutoff, and compared with what
happened. A static task holds out ``--holdout`` rows whose value is stored and
predicts them from the rest. Both report the error of Nori next to two
references: repeating each entity's previous window (temporal only) and the
median of the rows Nori learned from. Writes backtest.csv (one row per fold)
and backtest_rows.csv (one row per scored entity) to the output directory.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
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


def get_errors(frame: pd.DataFrame, column: str) -> dict[str, float]:
    """Return the MAE and RMSE of a column of forecasts against ``actual``."""
    error = frame[column] - frame["actual"]
    return {
        f"mae_{column}": float(error.abs().mean()),
        f"rmse_{column}": float(np.sqrt((error**2).mean())),
    }


def get_coverage(frame: pd.DataFrame, quantiles: list[float] | None) -> float | None:
    """Return the share of actuals inside the outer quantile band, if any."""
    if not quantiles or "quantiles" not in frame or len(frame) == 0:
        return None
    low, high = min(quantiles), max(quantiles)
    bands = frame["quantiles"].apply(
        lambda value: (
            (value[str(low)], value[str(high)])
            if isinstance(value, dict)
            else (value[quantiles.index(low)], value[quantiles.index(high)])
        )
    )
    inside = [
        band[0] <= actual <= band[1]
        for band, actual in zip(bands, frame["actual"], strict=True)
    ]
    return float(np.mean(inside))


def run_temporal(client, database, task: dict, folds: int) -> tuple[list, list]:
    """Predict from each earlier cutoff and compare with the known outcomes."""
    anchor_time = parse_time(task["anchor_time"])
    step = get_offset(task["window"])
    cutoffs = [
        (pd.Timestamp(anchor_time) - k * step).to_pydatetime()
        for k in range(1, folds + 2)
    ]
    # One read of the outcomes at every cutoff, plus one older cutoff for the
    # repeat-previous-window reference. Its label windows are all complete.
    truth = client.predict(
        source=database,
        **{**build_request(task, anchor_time), "anchor_times": cutoffs},
        return_details=True,
        log_summary=False,
    )
    labels = truth.context.pivot(
        index="entity_id", columns="anchor_time", values="label"
    )
    cap = task.get("max_context_rows") or 5000
    if truth.metadata.get("candidate_rows", 0) >= cap:
        print(
            f"WARN: the outcome read hit max_context_rows={cap}; only the newest "
            "cutoffs' entities are checked. Raise max_context_rows in the task."
        )
    summaries, rows = [], []
    for fold, cutoff in enumerate(cutoffs[:folds], start=1):
        stamp = pd.Timestamp(cutoff)
        if stamp not in labels.columns:
            print(f"fold {fold} at {cutoff:%Y-%m-%d}: no complete outcome window")
            continue
        check = client.predict(
            source=database,
            **build_request(task, cutoff),
            return_details=True,
            log_summary=False,
        )
        scored = check.predictions.set_index("entity_id")
        frame = pd.DataFrame({"nori": scored["prediction"]})
        if "quantiles" in scored:
            frame["quantiles"] = scored["quantiles"]
        frame["actual"] = labels[stamp]
        previous = pd.Timestamp(cutoffs[fold])
        frame["repeat_previous"] = labels.get(previous, np.nan)
        frame["context_median"] = float(check.context["label"].median())
        frame = frame.dropna(subset=["actual", "nori"])
        summary = {
            "fold": fold,
            "cutoff": cutoff.isoformat(),
            "entities": len(frame),
            "context_rows": len(check.context),
            **get_errors(frame, "nori"),
            **get_errors(frame.dropna(subset=["repeat_previous"]), "repeat_previous"),
            **get_errors(frame, "context_median"),
            "coverage": get_coverage(frame, task.get("quantiles")),
        }
        summaries.append(summary)
        rows.append(frame.assign(fold=fold, cutoff=cutoff).reset_index())
        print(
            f"fold {fold} at {cutoff:%Y-%m-%d}: {len(frame)} entities; MAE nori "
            f"{summary['mae_nori']:.4g}, repeat previous "
            f"{summary['mae_repeat_previous']:.4g}, context median "
            f"{summary['mae_context_median']:.4g}"
        )
    return summaries, rows


def quote_ids(values) -> str:
    """Return the ids as a PQL IN list: numbers bare, anything else quoted."""
    parts = []
    for value in values:
        if isinstance(value, int | float | np.integer | np.floating):
            parts.append(str(value))
        else:
            parts.append("'" + str(value).replace("'", "''") + "'")
    return ", ".join(parts)


def run_static(client, database, task: dict, holdout: int) -> tuple[list, list]:
    """Hold out stored values of a static target and predict them."""
    full = client.predict(
        source=database,
        **build_request(task, None),
        return_details=True,
        log_summary=False,
    )
    known = full.context.set_index("entity_id")["label"]
    # Nori needs at least 32 rows to learn from, so the holdout leaves them.
    holdout = min(holdout, len(known) - 32)
    if holdout < 1:
        raise ValueError(f"only {len(known)} stored values: too few to hold any out")
    sample = known.sample(holdout, random_state=0)
    match = re.search(r"\bFOR\s+EACH\s+([^\s]+)", task["query"], flags=re.IGNORECASE)
    if match is None:
        raise ValueError("a static backtest needs a FOR EACH query")
    query = task["query"].replace(
        match.group(0), f"FOR {match.group(1)} IN ({quote_ids(sample.index)})"
    )
    check = client.predict(
        source=database,
        **{**build_request(task, None), "query": query},
        return_details=True,
        log_summary=False,
    )
    frame = pd.DataFrame(
        {"nori": check.predictions.set_index("entity_id")["prediction"]}
    )
    frame["actual"] = sample
    frame["context_median"] = float(check.context["label"].median())
    frame = frame.dropna(subset=["actual", "nori"])
    summary = {
        "fold": 1,
        "cutoff": None,
        "entities": len(frame),
        "context_rows": len(check.context),
        **get_errors(frame, "nori"),
        **get_errors(frame, "context_median"),
        "coverage": get_coverage(frame, task.get("quantiles")),
    }
    print(
        f"held out {len(frame)} rows; MAE nori {summary['mae_nori']:.4g}, "
        f"context median {summary['mae_context_median']:.4g}"
    )
    return [summary], [frame.assign(fold=1).reset_index()]


def main() -> int:
    """Run the backtest and write its tables."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", type=Path, help="nori_rel_source.json")
    parser.add_argument("task", type=Path, help="nori_rel_task.json")
    parser.add_argument("--folds", type=int, default=3, help="earlier cutoffs")
    parser.add_argument("--holdout", type=int, default=200, help="static rows")
    parser.add_argument("--out-dir", type=Path, default=Path("nori_rel_output"))
    arguments = parser.parse_args()
    source = load_json(arguments.source)
    task = load_json(arguments.task)
    with make_client(task) as client:
        database = connect_source(client, source)
        if task.get("anchor_time"):
            summaries, rows = run_temporal(client, database, task, arguments.folds)
        else:
            summaries, rows = run_static(client, database, task, arguments.holdout)
    if not summaries:
        print("no fold had a complete outcome window; nothing to compare")
        return 1
    arguments.out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(summaries)
    table.to_csv(arguments.out_dir / "backtest.csv", index=False)
    pd.concat(rows).to_csv(arguments.out_dir / "backtest_rows.csv", index=False)
    pooled = pd.concat(rows)
    nori = pooled.eval("abs(nori - actual)").mean()
    median = pooled.eval("abs(context_median - actual)").mean()
    print(f"\nall folds, {len(pooled)} rows: MAE nori {nori:.4g}")
    print(f"context median: {median:.4g}")
    if "repeat_previous" in pooled:
        repeat = pooled.dropna(subset=["repeat_previous"])
        print(
            f"repeat previous window ({len(repeat)} rows): "
            f"{repeat.eval('abs(repeat_previous - actual)').mean():.4g}"
        )
    print(f"outputs in {arguments.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
