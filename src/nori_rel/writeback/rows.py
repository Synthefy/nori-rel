"""The rows write-back stores: one run's predictions, and the run's record.

Each engine turns a :class:`Column` kind into its own SQL type and writes the
same row values, so the three engines hold the same table shape.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Literal

import pandas as pd

from nori_rel.connectors.base import RUN_RECORDS_TABLE
from nori_rel.run.records import RunRecord

#: The SQL type family of a written column, mapped per engine by the writers.
ColumnKind = Literal[
    "integer", "decimal", "text", "timestamp", "double", "boolean", "json"
]

_INT64_RANGE = range(-(2**63), 2**63)
#: Digits a ``decimal`` column holds; wider whole ids are stored as text.
_DECIMAL_DIGITS = 38


@dataclass(frozen=True)
class Column:
    """A written column: its lowercase name and its kind of SQL type."""

    name: str
    kind: ColumnKind


@dataclass(frozen=True)
class TableRows:
    """One run's rows for one table; writing them replaces that run's rows.

    A ``json`` value is a Python list or dict; the writer serializes it.
    """

    table: str
    run_id: str
    columns: tuple[Column, ...]
    rows: tuple[tuple[Any, ...], ...]


RUN_RECORD_COLUMNS = (
    Column("run_id", "text"),
    Column("pipeline_id", "text"),
    Column("status", "text"),
    Column("as_of", "timestamp"),
    Column("started_at", "timestamp"),
    Column("finished_at", "timestamp"),
    Column("output_table", "text"),
    Column("rows_written", "integer"),
    Column("written_at", "timestamp"),
    Column("record", "json"),
)


def get_completed_run_record(predictions: pd.DataFrame) -> RunRecord:
    """Return the record of the run that produced ``predictions``.

    Raises:
        ValueError: The frame did not come from a completed ``predict`` run,
            or lacks the columns every prediction frame has.
    """
    if not isinstance(predictions, pd.DataFrame):
        raise TypeError("predictions must be the DataFrame predict returned")
    record = RunRecord.from_prediction(predictions)
    if record is None or record.status != "completed":
        raise ValueError(
            "predictions carry no completed run record; pass the frame that "
            "predict or run.result() returned"
        )
    missing = sorted({"entity_id", "anchor_time", "prediction"} - set(predictions))
    if missing:
        raise ValueError(f"predictions lack the columns {missing}")
    return record


def get_prediction_rows(
    predictions: pd.DataFrame,
    *,
    table: str,
    record: RunRecord,
    written_at: datetime,
) -> TableRows:
    """Return the rows that store ``predictions`` for ``record``'s run.

    Columns: ``entity_id``, ``as_of`` (the scoring time), ``prediction`` (the
    requested mean or median; a Boolean for a binary target), ``probability``
    for a binary target, ``quantiles`` as a JSON list when quantiles were
    requested, ``run_id``, ``pipeline_id`` and ``written_at``.

    Raises:
        ValueError: An entity id or a prediction is missing or not finite.
    """
    binary = "probability" in predictions
    columns = [
        Column("entity_id", _get_entity_id_kind(predictions["entity_id"])),
        Column("as_of", "timestamp"),
        Column("prediction", "boolean" if binary else "double"),
    ]
    if binary:
        columns.append(Column("probability", "double"))
    if "quantiles" in predictions:
        columns.append(Column("quantiles", "json"))
    columns += [
        Column("run_id", "text"),
        Column("pipeline_id", "text"),
        Column("written_at", "timestamp"),
    ]
    entity_kind = columns[0].kind
    rows = []
    for item in predictions.itertuples(index=False):
        row: list[Any] = [
            _get_entity_id(item.entity_id, entity_kind),
            _get_utc_datetime(item.anchor_time),
            bool(item.prediction) if binary else _get_finite_float(item.prediction),
        ]
        if binary:
            row.append(_get_finite_float(item.probability))
        if "quantiles" in predictions:
            row.append([_get_finite_float(value) for value in item.quantiles])
        row += [record.run_id, record.pipeline_id, written_at]
        rows.append(tuple(row))
    return TableRows(table, record.run_id, tuple(columns), tuple(rows))


def get_run_record_rows(record: RunRecord, *, written_at: datetime) -> TableRows:
    """Return the ``nori_rel_runs`` row of ``record``, with the record as JSON."""
    row = (
        record.run_id,
        record.pipeline_id,
        record.status,
        record.as_of,
        record.started_at,
        record.finished_at,
        record.output_table,
        record.rows_written,
        written_at,
        record.model_dump(mode="json"),
    )
    return TableRows(RUN_RECORDS_TABLE, record.run_id, RUN_RECORD_COLUMNS, (row,))


def get_written_at() -> datetime:
    """Return the time a write stamps on its rows, in UTC microseconds."""
    return datetime.now(UTC)


def _get_entity_id_kind(entity_ids: pd.Series) -> ColumnKind:
    """Return the column kind of entity ids: ``integer``, ``decimal`` or ``text``.

    64-bit whole ids are ``integer``, whole ids up to 38 digits ``decimal``,
    anything else ``text``. A run with no rows has no ids to tell; its table
    is never created from it (see ``write_predictions``).
    """
    values = entity_ids.dropna().tolist()
    if len(values) != len(entity_ids):
        raise ValueError("predictions have a missing entity_id")
    whole = all(
        (isinstance(value, numbers.Integral) and not isinstance(value, bool))
        or (
            isinstance(value, Decimal)
            and value.is_finite()
            and value == value.to_integral_value()
        )
        for value in values
    )
    if not whole or any(
        len(str(abs(int(value)))) > _DECIMAL_DIGITS for value in values
    ):
        return "text"
    return (
        "integer" if all(int(value) in _INT64_RANGE for value in values) else "decimal"
    )


def _get_entity_id(value: Any, kind: ColumnKind) -> Any:
    """Return an entity id as the Python value its column kind stores."""
    if kind == "integer":
        return int(value)
    if kind == "decimal":
        return Decimal(int(value))
    return str(value)


def _get_utc_datetime(value: Any) -> datetime:
    """Return a timezone-aware time as a UTC datetime; refuse a naive one."""
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError("predictions need timezone-aware anchor_time values")
    return stamp.tz_convert("UTC").to_pydatetime()


def _get_finite_float(value: Any) -> float:
    """Return ``value`` as a float; refuse a missing or infinite one."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("predictions hold a missing or infinite value")
    return number


__all__ = [
    "RUN_RECORDS_TABLE",
    "RUN_RECORD_COLUMNS",
    "Column",
    "ColumnKind",
    "TableRows",
    "get_completed_run_record",
    "get_prediction_rows",
    "get_run_record_rows",
    "get_written_at",
]
