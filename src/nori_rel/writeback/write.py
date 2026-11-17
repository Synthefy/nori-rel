"""Write a run's predictions, and every run's record, to a destination schema.

``write_predictions`` stores one completed run's predictions in a table the
caller names and the run's record, with ``output_table`` and
``rows_written``, in ``nori_rel_runs`` beside it. ``TableRunRecordStore``
keeps every run's record in that same table, failed and cancelled runs
included, when a client uses it as ``run_records``.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pandas as pd

from nori_rel.contracts import PredictionDestination
from nori_rel.limits import RelationalLimits
from nori_rel.run.records import RunRecord
from nori_rel.writeback.rows import (
    RUN_RECORDS_TABLE,
    EarlierRuns,
    TableRows,
    get_completed_run_record,
    get_prediction_rows,
    get_run_record_rows,
    get_written_at,
)
from nori_rel.writeback.writers import TableWriter, open_table_writer


def write_predictions(
    predictions: pd.DataFrame,
    *,
    destination: PredictionDestination,
    table: str,
    create_table: bool = False,
    replace_earlier_runs: bool = False,
    limits: RelationalLimits | None = None,
) -> RunRecord:
    """Store one run's predictions in ``table``, replacing that run's earlier rows.

    Returns the run's record with ``output_table`` and ``rows_written``; the
    same record is stored in ``nori_rel_runs`` in the destination schema,
    which is created when missing. ``table`` must exist with the prediction
    columns unless ``create_table`` is true. With ``replace_earlier_runs``,
    the rows other runs of the same pipeline wrote for the same ``as_of``
    are replaced too, so a scheduled job that predicts again on a retry
    leaves one batch; their records stay in ``nori_rel_runs``.

    Raises:
        ValueError: The frame has no completed run record, the table is
            missing, or its columns differ; the message names the columns.
            ``replace_earlier_runs`` for a run without a pipeline.
        PredictionWriteError: The destination refused the write.
    """
    _require_table_name(table)
    if not isinstance(destination, PredictionDestination):
        raise TypeError("destination must be a PredictionDestination")
    record = get_completed_run_record(predictions)
    written_at = get_written_at()
    rows = get_prediction_rows(
        predictions, table=table, record=record, written_at=written_at
    )
    if replace_earlier_runs:
        rows = replace(rows, earlier_runs=_get_earlier_runs(record, rows))
    writer = open_table_writer(destination, limits)
    try:
        if rows.rows or writer.get_column_names(table) is not None:
            rows = _get_rows_in_table_order(writer, rows, create=create_table)
            writes = [rows]
        else:
            # A run that scored nobody writes no rows, so a missing table is
            # left missing: its entity_id type is unknown.
            writes = []
        written = record.model_copy(
            update={
                "output_table": writer.get_qualified_name(table),
                "rows_written": len(rows.rows),
            }
        )
        runs = _get_rows_in_table_order(
            writer, get_run_record_rows(written, written_at=written_at), create=True
        )
        writer.replace_run_rows([*writes, runs])
    finally:
        writer.close()
    return written


class TableRunRecordStore:
    """Keep run records in the ``nori_rel_runs`` table of a destination schema.

    Pass it as ``NoriRelClient(run_records=...)`` to keep every run's record
    in the warehouse: the ``running`` record each run stores first and its
    final one, failed and cancelled runs included. Each write opens a
    connection with the destination's writer credential and replaces the
    run's row. The table is created when missing.
    """

    def __init__(
        self,
        destination: PredictionDestination,
        *,
        limits: RelationalLimits | None = None,
    ) -> None:
        """Keep the destination; nothing connects until a record is written."""
        if not isinstance(destination, PredictionDestination):
            raise TypeError("destination must be a PredictionDestination")
        self.destination = destination
        self.limits = limits

    def __repr__(self) -> str:
        """Name the engine and schema, never the credential."""
        return (
            f"{type(self).__name__}(engine={self.destination.engine.value!r}, "
            f"schema_name={self.destination.schema_name!r})"
        )

    def write(self, record: RunRecord) -> None:
        """Replace the run's row in ``nori_rel_runs`` with ``record``."""
        writer = open_table_writer(self.destination, self.limits)
        try:
            rows = _get_rows_in_table_order(
                writer,
                get_run_record_rows(record, written_at=get_written_at()),
                create=True,
            )
            writer.replace_run_rows([rows])
        finally:
            writer.close()

    def read(self, run_id: str) -> RunRecord:
        """Return the record of ``run_id``; ``KeyError`` if the table has none."""
        writer = open_table_writer(self.destination, self.limits)
        try:
            if writer.get_column_names(RUN_RECORDS_TABLE) is None:
                stored = None
            else:
                stored = writer.read_run_record(run_id)
        finally:
            writer.close()
        if stored is None:
            raise KeyError(f"no record of run {run_id!r} in {RUN_RECORDS_TABLE}")
        return RunRecord.model_validate(
            json.loads(stored) if isinstance(stored, str | bytes) else stored
        )


def _get_rows_in_table_order(
    writer: TableWriter, rows: TableRows, *, create: bool
) -> TableRows:
    """Return ``rows`` in the column order of their table, creating it if asked.

    Names are compared as the destination stores them (uppercase on
    Snowflake), so a column only a quoted name reaches does not pass.

    Raises:
        ValueError: The table is missing and ``create`` is false, or its
            columns are not the written ones.
    """
    expected = [writer.get_column_name(column.name) for column in rows.columns]
    existing = writer.get_column_names(rows.table)
    name = writer.get_qualified_name(rows.table)
    if existing is None:
        if not create:
            raise ValueError(
                f"table {name} does not exist; pass create_table=True to create it"
            )
        writer.create_table(rows.table, rows.columns)
        return rows
    missing = [column for column in expected if column not in existing]
    unexpected = [column for column in existing if column not in expected]
    if missing or unexpected:
        raise ValueError(
            f"table {name} does not have the written columns: missing {missing}, "
            f"unexpected {unexpected}; write to another table or change it"
        )
    order = [expected.index(column) for column in existing]
    return replace(
        rows,
        columns=tuple(rows.columns[index] for index in order),
        rows=tuple(tuple(row[index] for index in order) for row in rows.rows),
    )


def _get_earlier_runs(record: RunRecord, rows: TableRows) -> EarlierRuns:
    """Return the pipeline and prediction times whose earlier rows are replaced.

    A pipeline replay scores every row at its ``as_of``; the record's
    ``as_of`` counts too, so a retry that scored nobody still clears its batch.

    Raises:
        ValueError: The run has no pipeline, so other runs cannot be matched.
    """
    if record.pipeline_id is None:
        raise ValueError(
            "replace_earlier_runs needs predictions from a pipeline run; this "
            "run has no pipeline, so its earlier runs cannot be told apart"
        )
    as_of_index = [column.name for column in rows.columns].index("as_of")
    as_of_values = {row[as_of_index] for row in rows.rows}
    if record.as_of is not None:
        as_of_values.add(record.as_of)
    return EarlierRuns(record.pipeline_id, tuple(sorted(as_of_values)))


def _require_table_name(table: Any) -> None:
    """Refuse a blank table name, or the run records table itself."""
    if not isinstance(table, str) or not table.strip() or "\x00" in table:
        raise ValueError("table must name a table in the destination schema")
    if table.lower() == RUN_RECORDS_TABLE:
        raise ValueError(f"{RUN_RECORDS_TABLE} holds run records, not predictions")


__all__ = ["TableRunRecordStore", "write_predictions"]
