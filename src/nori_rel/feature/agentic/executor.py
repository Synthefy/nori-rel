"""Run SQL feature programs against an in-memory DuckDB copy of a snapshot."""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping

import numpy as np
import pandas as pd

from ..dfs.noridfs.contracts import DatabaseView
from .contracts import (
    CUTOFF_ALIAS,
    EVAL_TABLE,
    ROW_COLUMN,
    ExecutionResult,
    FeatureProgram,
    QueryDiagnostics,
)

_SOURCE = "__nr_source"
#: Values past this magnitude overflow the float32 cast in Nori's scaler.
_FLOAT32_SAFE = 1e30
#: A high-cardinality string column is prose rather than a broken category when
#: it reads like prose, which is a mean length of at least 20 characters over
#: a sample of its rows.
_TEXT_SAMPLE_ROWS = 10_000
_MIN_MEAN_TEXT_LENGTH = 20


class ProgramExecutor:
    """Materialize one row per anchor for every query of a program.

    Tables load into an in-memory DuckDB catalog with external file access
    off. ``eval_table`` carries ``row_id``, the entity key, and ``cutoff``.
    A query must return ``row_id`` at most once per anchor; a missing anchor
    becomes a missing value and is reported as coverage. A preflight count
    rejects fan-out beyond ``fanout_limit`` rows per anchor, and a timer
    interrupts statements past ``timeout_seconds``.
    """

    def __init__(
        self,
        database: DatabaseView,
        *,
        entity_column: str,
        timeout_seconds: float,
        fanout_limit: int = 4,
        max_categorical_cardinality: int = 100,
        allow_text: bool = True,
    ) -> None:
        """Bind a snapshot view and the execution limits.

        `allow_text` off restores the behaviour that predates the text path:
        a string above the cardinality cap is refused rather than encoded.
        It exists so the text path can be ablated within one implementation
        instead of against a different codebase.
        """
        import duckdb

        self.database = database
        self.entity_column = entity_column
        self.timeout_seconds = timeout_seconds
        self.fanout_limit = fanout_limit
        self.max_categorical_cardinality = max_categorical_cardinality
        self.allow_text = allow_text
        self._connection = duckdb.connect(
            database=":memory:", config={"enable_external_access": "false"}
        )
        for name, frame in database.tables.items():
            self._load(name, frame)

    def bind_anchors(self, anchors: pd.DataFrame) -> pd.Series:
        """Expose anchors as ``eval_table`` and return their row identifiers."""
        frame = self._eval_frame(anchors)
        self._load(EVAL_TABLE, frame)
        return frame[ROW_COLUMN]

    def run(self, program: FeatureProgram, anchors: pd.DataFrame) -> ExecutionResult:
        """Execute every query, coerce types, and align features to anchors."""
        row_ids = self.bind_anchors(anchors)
        blocks: list[pd.DataFrame] = []
        categorical: list[str] = []
        text: list[str] = []
        diagnostics: dict[str, QueryDiagnostics] = {}
        for query in program.queries:
            started = time.perf_counter()
            raw = self._execute(query.name, query.sql, expected_rows=len(row_ids))
            block, kinds, prose = self._align(query.name, raw, row_ids)
            blocks.append(block)
            categorical.extend(kinds)
            text.extend(prose)
            diagnostics[query.name] = QueryDiagnostics(
                columns=tuple(block.columns),
                coverage=float(raw[ROW_COLUMN].nunique() / max(len(row_ids), 1)),
                missing_fraction=(
                    float(block.isna().to_numpy().mean()) if block.size else 1.0
                ),
                seconds=time.perf_counter() - started,
            )
        frame = pd.concat(blocks, axis=1) if blocks else pd.DataFrame(index=row_ids)
        if not frame.columns.is_unique:
            raise ValueError("feature column names collide across queries")
        return ExecutionResult(
            frame=frame.reset_index(drop=True),
            categorical_columns=tuple(categorical),
            text_columns=tuple(text),
            diagnostics=diagnostics,
        )

    def sql(self, statement: str, *, limit: int) -> pd.DataFrame:
        """Run one read-only exploration statement with a row limit."""
        trimmed = statement.strip().rstrip(";")
        return self._execute("exploration", f"SELECT * FROM ({trimmed}) LIMIT {limit}")

    def close(self) -> None:
        """Release the connection."""
        self._connection.close()

    def _load(self, name: str, frame: pd.DataFrame) -> None:
        # A cursor does not see frames registered on its parent connection, and
        # every query runs on a cursor so it can be interrupted, so each table
        # is materialized into the catalog instead of registered.
        self._connection.register(_SOURCE, frame)
        quoted = name.replace('"', '""')
        self._connection.execute(
            f'CREATE OR REPLACE TABLE "{quoted}" AS SELECT * FROM {_SOURCE}'
        )
        self._connection.unregister(_SOURCE)

    def _eval_frame(self, anchors: pd.DataFrame) -> pd.DataFrame:
        if len(anchors.columns) != 2:
            raise ValueError("anchors must hold exactly the entity key and cutoff")
        entity, cutoff = anchors.columns
        return pd.DataFrame(
            {
                ROW_COLUMN: np.arange(len(anchors), dtype=np.int64),
                self.entity_column: anchors[entity].to_numpy(),
                CUTOFF_ALIAS: anchors[cutoff].to_numpy(),
            }
        )

    def _execute(
        self, name: str, sql: str, *, expected_rows: int | None = None
    ) -> pd.DataFrame:
        cursor = self._connection.cursor()
        timer = threading.Timer(self.timeout_seconds, cursor.interrupt)
        timer.start()
        try:
            if expected_rows is not None:
                self._preflight(cursor, name, sql, expected_rows)
            return cursor.execute(sql).fetch_df()
        except ValueError:
            raise
        except Exception as error:
            if "interrupt" in str(error).lower():
                raise TimeoutError(
                    f"query {name!r} exceeded {self.timeout_seconds:g}s"
                ) from error
            raise ValueError(f"query {name!r}: {error}") from error
        finally:
            timer.cancel()
            cursor.close()

    def _preflight(self, cursor, name: str, sql: str, expected_rows: int) -> None:
        inner = sql.rstrip().rstrip(";")
        count = cursor.execute(
            f"SELECT COUNT(*) FROM ({inner}) AS __nr_count"
        ).fetchone()
        if count is not None and count[0] > expected_rows * self.fanout_limit:
            raise ValueError(
                f"query {name!r} returns {count[0]} rows for {expected_rows} "
                "anchors; aggregate to one row per row_id before returning"
            )

    def _align(
        self, name: str, raw: pd.DataFrame, row_ids: pd.Series
    ) -> tuple[pd.DataFrame, list[str], list[str]]:
        if ROW_COLUMN not in raw:
            raise ValueError(f"query {name!r} must return {ROW_COLUMN} from eval_table")
        if raw[ROW_COLUMN].duplicated().any():
            raise ValueError(f"query {name!r} returns several rows for one anchor")
        drop = [c for c in (self.entity_column, CUTOFF_ALIAS) if c in raw]
        values = raw.drop(columns=drop).set_index(ROW_COLUMN)
        values = values.reindex(row_ids.to_numpy())
        columns: dict[str, pd.Series] = {}
        categorical: list[str] = []
        text: list[str] = []
        for column in values.columns:
            output = f"{name}__{column}"
            series, kind = _coerce(
                values[column],
                self.max_categorical_cardinality,
                allow_text=self.allow_text,
            )
            if kind is None:
                raise ValueError(
                    f"column {column!r} of query {name!r} is neither numeric, prose, "
                    f"nor a category with at most "
                    f"{self.max_categorical_cardinality} levels"
                )
            columns[output] = series
            if kind == "categorical":
                categorical.append(output)
            elif kind == "text":
                text.append(output)
        frame = pd.DataFrame(columns, index=values.index).reset_index(drop=True)
        return frame, categorical, text


def _coerce(
    series: pd.Series, cardinality: int, *, allow_text: bool = True
) -> tuple[pd.Series, str | None]:
    """Return a Nori-ready column and its kind, or ``None`` when unusable.

    Missing values survive on purpose: Nori encodes them natively, and an
    internal study measured passthrough well ahead of any imputation.
    """
    if pd.api.types.is_bool_dtype(series):
        mapped = series.map({True: "true", False: "false"}).astype("object")
        return mapped.where(series.notna(), None), "categorical"
    if pd.api.types.is_datetime64_any_dtype(series):
        # DuckDB hands back microsecond timestamps, pandas often nanosecond, so
        # subtract the epoch instead of reinterpreting the integer view.
        zone = getattr(series.dt, "tz", None)
        epoch = pd.Timestamp(0, tz=zone) if zone is not None else pd.Timestamp(0)
        days = (series - epoch) / pd.Timedelta(days=1)
        return days.where(series.notna()), "numeric"
    if pd.api.types.is_numeric_dtype(series):
        numeric = series.astype(float).replace([np.inf, -np.inf], np.nan)
        return numeric.where(numeric.abs() < _FLOAT32_SAFE), "numeric"
    present = series.dropna()
    strings = series.astype("object").map(lambda v: None if pd.isna(v) else str(v))
    if present.nunique() <= cardinality:
        return strings, "categorical"
    if allow_text and is_prose(present):
        # Above the cardinality cap this used to be an error, so a program that
        # selected a description was rejected outright rather than scored with
        # the text path the DFS entry has always had.
        return strings, "text"
    return series, None


def subset_view(database: DatabaseView, before: pd.Timestamp) -> DatabaseView:
    """Return the snapshot with every temporal table truncated before a time."""
    tables: dict[str, pd.DataFrame] = {}
    for name, frame in database.tables.items():
        column = database.time_columns.get(name)
        if column is None:
            tables[name] = frame
            continue
        stamps = frame[column]
        tables[name] = frame[stamps.notna() & (stamps < before)].reset_index(drop=True)
    return DatabaseView.from_parts(
        tables=tables,
        primary_keys=database.primary_keys,
        foreign_keys=database.foreign_keys,
        time_columns=database.time_columns,
    )


def is_prose(present: pd.Series) -> bool:
    """Separate prose from high-cardinality identifiers by mean length."""
    sample = present.head(_TEXT_SAMPLE_ROWS).astype(str)
    return not sample.empty and bool(sample.str.len().mean() >= _MIN_MEAN_TEXT_LENGTH)


def frames_match(left: pd.DataFrame, right: pd.DataFrame) -> Mapping[str, bool]:
    """Compare two one-row feature frames per column, treating NaN as equal."""
    result: dict[str, bool] = {}
    for column in left.columns:
        if column not in right:
            result[column] = False
            continue
        a = pd.Series(left[column].to_numpy()).reset_index(drop=True)
        b = pd.Series(right[column].to_numpy()).reset_index(drop=True)
        result[column] = bool(a.equals(b))
    return result
