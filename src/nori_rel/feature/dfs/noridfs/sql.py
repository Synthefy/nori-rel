"""DuckDB SQL that computes each feature of a fitted NoriDFS plan.

This module states each planned feature as one DuckDB query, so a reader can
see what a column holds, and ``get_features_from_sql`` runs those queries. It
is NoriDFS's DuckDB executor; ``engine`` holds the pandas executor, and its
tests hold the two to the same values. A query reads the snapshot tables and
``eval_table``, which holds one row per prediction row: ``row_id``, the entity
key under its own column name, and ``cutoff``. The agent search's SQL reads the
same table.

Two helper columns keep the executor's tie-breaks exact:

- ``__nr_tie_p<n>``, on the table a path's rows come from, orders rows with
  equal times for the ``last`` features of the plan's path ``n``. It holds the
  row hash the executor sorts by.
- ``<column>__nr_rank``, on the table of a categorical column, orders that
  column's values the way the executor breaks equal counts in a mode: by
  Python type, then ``repr``.

``get_features_from_sql`` adds both to its DuckDB copy of the snapshot. DuckDB
is imported there, so writing SQL needs no optional dependency.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .contracts import DatabaseView, FeaturePlan, FeatureSpec, ForeignKey, RelationPath
from .engine import (
    _CATEGORY_ESCAPE,
    _MISSING_CATEGORY,
    _category,
    _time_ns,
    get_path_row_ties,
)

#: The anchor table and its columns, named as the agent search's SQL names
#: them (``feature.agentic.contracts``).
EVAL_TABLE = "eval_table"
ROW_COLUMN = "row_id"
CUTOFF_COLUMN = "cutoff"
#: The entity key's column in ``eval_table`` when its own name is taken.
ENTITY_FALLBACK_COLUMN = "entity_key"
#: Each query returns ``row_id`` and this column, one row per anchor.
FEATURE_COLUMN = "feature"
TIE_COLUMN_PREFIX = "__nr_tie_p"
RANK_COLUMN_SUFFIX = "__nr_rank"
#: The most anchor and row pairs, summed over a plan's queries, that
#: ``executor="auto"`` joins in DuckDB. A query joins each anchor to every row
#: of its entity before it filters them by time, so a few entities with long
#: histories and many anchors make it slower than the pandas prefix sums.
AUTO_MAX_JOINED_PAIRS = 1_000_000_000
#: Queries ``get_features_from_sql`` runs at once. Each runs on one DuckDB
#: thread, so it adds a group's values in the same order on every run, and the
#: matrix does not depend on this number.
QUERY_WORKERS = min(8, os.cpu_count() or 1)


_INDENT = "    "
_INTERVAL_UNITS = (
    ("days", 86_400 * 10**9),
    ("hours", 3_600 * 10**9),
    ("minutes", 60 * 10**9),
    ("seconds", 10**9),
    ("milliseconds", 10**6),
    ("microseconds", 10**3),
)
_ROW_TIME_OPERATIONS = {
    "recency": "(EPOCH({cutoff}) - EPOCH(MAX({rows}.row_time))) / 86400.0",
    "mean_age": "(EPOCH({cutoff}) - AVG(EPOCH({rows}.row_time))) / 86400.0",
    "oldest_age": "(EPOCH({cutoff}) - EPOCH(MIN({rows}.row_time))) / 86400.0",
}
_VALUE_FUNCTIONS = {
    "mean": "AVG",
    "sum": "SUM",
    "min": "MIN",
    "max": "MAX",
    "std": "STDDEV_SAMP",
}


@dataclass(frozen=True)
class ColumnType:
    """The facts about one column that decide how the executor compares it."""

    dtype: str
    integer: bool
    numeric: bool
    text: bool
    categorical: bool

    @classmethod
    def of(cls, values: pd.Series) -> ColumnType:
        """Return a column's type as the executor's ``_aligned_keys`` reads it."""
        return cls(
            dtype=str(values.dtype),
            integer=pd.api.types.is_integer_dtype(values),
            numeric=pd.api.types.is_numeric_dtype(values),
            text=pd.api.types.is_string_dtype(values),
            categorical=isinstance(values.dtype, pd.CategoricalDtype),
        )


@dataclass(frozen=True)
class SQLSchema:
    """The schema facts a plan's SQL needs beyond the plan itself.

    ``column_types`` covers the join keys, with ``(EVAL_TABLE, <entity
    column>)`` for the anchors' keys.
    """

    primary_keys: Mapping[str, str]
    time_columns: Mapping[str, str]
    column_types: Mapping[tuple[str, str], ColumnType] = field(default_factory=dict)


def get_sql_schema(
    database: DatabaseView,
    plan: FeaturePlan,
    anchor_keys: pd.Series,
) -> SQLSchema:
    """Return what the plan's SQL needs to know about this snapshot.

    Only key columns are inspected, so this stays cheap on wide tables.
    """
    entity_table = plan.anchors.entity_table
    columns = {(entity_table, database.primary_keys[entity_table])}
    for path in plan.paths:
        for edge in path.edges:
            columns.add((edge.child_table, edge.child_column))
            columns.add((edge.parent_table, edge.parent_column))
    types = {
        (table, column): ColumnType.of(database.tables[table][column])
        for table, column in sorted(columns)
    }
    types[(EVAL_TABLE, get_eval_entity_column(plan))] = ColumnType.of(anchor_keys)
    return SQLSchema(
        primary_keys=dict(database.primary_keys),
        time_columns=dict(database.time_columns),
        column_types=types,
    )


def get_plan_sql(plan: FeaturePlan, schema: SQLSchema) -> dict[str, str]:
    """Return the DuckDB query that computes each planned feature, by name.

    Each query returns ``row_id`` and ``feature``, one row per ``eval_table``
    row, in ``row_id`` order.
    """
    writer = _PlanSQLWriter(plan, schema)
    return {feature.name: writer.get_feature_sql(feature) for feature in plan.features}


def get_eval_entity_column(plan: FeaturePlan) -> str:
    """Return the entity key's column name in ``eval_table``."""
    column = plan.anchors.entity_column
    if column.lower() in {ROW_COLUMN, CUTOFF_COLUMN}:
        return ENTITY_FALLBACK_COLUMN
    return column


def get_tie_column(plan: FeaturePlan, path: RelationPath) -> str:
    """Return the helper column that orders a path's rows with equal times."""
    return f"{TIE_COLUMN_PREFIX}{plan.paths.index(path)}"


def get_rank_column(column: str) -> str:
    """Return the helper column that orders a categorical column's values."""
    return f"{column}{RANK_COLUMN_SUFFIX}"


def get_path_row_table(path: RelationPath) -> str:
    """Return the table a path's rows come from, which holds its tie column."""
    return path.edges[0].child_table if path.is_bridge else path.edges[-1].child_table


def get_features_from_sql(
    database: DatabaseView,
    plan: FeaturePlan,
    anchors: pd.DataFrame,
    *,
    max_joined_pairs: int | None = None,
) -> pd.DataFrame | None:
    """Return the plan's features computed by its SQL on an in-memory DuckDB.

    ``anchors`` holds the entity key and, for a temporal plan, the cutoff, as
    ``NativeRelationalFeaturizer.transform`` takes them. The result has one
    row per anchor, in order, and one column per planned feature, with the
    pandas executor's types: numbers are float64, and categories are Python
    values with None for missing. The queries run ``QUERY_WORKERS`` at a time,
    each on one DuckDB thread: with more threads, DuckDB adds a group's floats
    in an order that changes between runs, and so do the last digits of means,
    sums and standard deviations. External file access is off, as it is for
    the agent search's SQL. With ``max_joined_pairs``, the result is None when
    the queries would join more anchor and row pairs than that. Needs
    ``duckdb``.
    """
    import duckdb

    schema = get_sql_schema(database, plan, anchors[plan.anchors.entity_column])
    writer = _PlanSQLWriter(plan, schema)
    connection = duckdb.connect(
        database=":memory:", config={"enable_external_access": "false"}
    )
    try:
        for name, frame in _get_sql_tables(database, plan).items():
            _load_table(connection, name, frame)
        _load_table(connection, EVAL_TABLE, _get_eval_table(plan, anchors))
        # Pair counts are integers, so they are exact on any number of threads.
        if (
            max_joined_pairs is not None
            and _get_joined_pairs(connection, writer) > max_joined_pairs
        ):
            return None
        connection.execute("SET threads = 1")
        with ThreadPoolExecutor(QUERY_WORKERS) as pool:
            columns = pool.map(
                lambda feature: _get_feature_values(
                    connection, writer, feature, len(anchors)
                ),
                plan.features,
            )
            values = dict(zip(plan.columns, columns, strict=True))
        return pd.DataFrame(values, index=pd.RangeIndex(len(anchors)))
    finally:
        connection.close()


def _get_joined_pairs(connection, writer: _PlanSQLWriter) -> int:
    """Return the anchor and row pairs the plan's queries join, summed."""
    pairs: dict[RelationPath, int] = {}
    total = 0
    for feature in writer.plan.features:
        if feature.path is None or feature.operation == "lookup":
            continue
        if feature.path not in pairs:
            row = connection.execute(
                writer.get_joined_pairs_sql(feature.path)
            ).fetchone()
            pairs[feature.path] = int(row[0])
        total += pairs[feature.path]
    return total


def _get_feature_values(
    connection, writer: _PlanSQLWriter, feature: FeatureSpec, rows: int
) -> pd.Series | np.ndarray:
    """Run one feature's query; return its values as the pandas executor types them.

    Numbers become float64. Categories are fetched as Python values, so a
    boolean stays a bool and an integer stays an integer where a data frame
    would turn either into a float or a nullable type. The query runs on its
    own cursor, so several can run at once.
    """
    sql = writer.get_feature_sql(feature)
    cursor = connection.cursor()
    try:
        if feature.kind == "categorical":
            result = cursor.execute(sql).fetchall()
            row_ids = [row[0] for row in result]
            values = [row[1] for row in result]
            if (
                feature.operation == "mode"
                and writer.plan.config.include_missing_in_modes
            ):
                categories = _get_mode_categories(cursor, writer, feature)
                values = [
                    None if value is None else categories[value] for value in values
                ]
            output: pd.Series | np.ndarray = pd.Series(values, dtype=object)
        else:
            frame = cursor.execute(sql).df()
            row_ids = frame[ROW_COLUMN].tolist()
            output = frame[FEATURE_COLUMN].to_numpy(dtype=float, na_value=np.nan)
    finally:
        cursor.close()
    if row_ids != list(range(rows)):
        raise RuntimeError(f"the SQL of {feature.name!r} lost or repeated rows")
    return output


def _get_mode_categories(
    connection, writer: _PlanSQLWriter, feature: FeatureSpec
) -> dict[str, object]:
    """Return the value behind each text a mode that counts missing values returns.

    That query returns text, since one column holds both the missing category
    and the database's values. The executor returns the value itself, so each
    text maps back to its value, normalized as the executor's ``_category``
    normalizes it.
    """
    if feature.column is None:
        raise ValueError(f"{feature.name!r} has no source column")
    source = _column(feature.table, feature.column)
    rows = connection.execute(
        f"SELECT DISTINCT {writer._get_mode_value(source)}, {source} "
        f"FROM {_quote(feature.table)}"
    ).fetchall()
    categories = {}
    for label, value in rows:
        category = _category(value, include_missing=True)
        if category is not None:
            categories[label] = category[1]
    return categories


@dataclass(frozen=True)
class _PathTable:
    """One table a path joins, and the name the query gives it."""

    table: str
    name: str

    def get_from_item(self) -> str:
        """Return the table as a FROM or JOIN item, with its name if it differs."""
        if self.name == self.table:
            return _quote(self.table)
        return f"{_quote(self.table)} AS {_quote(self.name)}"


class _PlanSQLWriter:
    """Write each feature of one plan as a self-contained DuckDB query."""

    def __init__(self, plan: FeaturePlan, schema: SQLSchema) -> None:
        self.plan = plan
        self.types = schema.column_types
        self.temporal = plan.anchors.cutoff_column is not None
        self.entity_table = plan.anchors.entity_table
        self.entity_key = schema.primary_keys[self.entity_table]
        self.eval_entity = get_eval_entity_column(plan)
        tables = {self.entity_table}
        for path in plan.paths:
            for edge in path.edges:
                tables.update({edge.child_table, edge.parent_table})
        self.time_columns = {
            table: column
            for table, column in schema.time_columns.items()
            if table in tables
        }
        self.taken = {table.lower() for table in tables} | {EVAL_TABLE}
        taken = self.taken
        self.anchor = _get_unused_name("e", taken)
        self.rows = _get_unused_name("p", taken)
        self.counts = _get_unused_name("c", taken)
        self.path_rows = _get_unused_name("path_rows", taken)
        self.value_counts = _get_unused_name("value_counts", taken)
        self.cutoff = f"{self.anchor}.{CUTOFF_COLUMN}"

    def get_feature_sql(self, feature: FeatureSpec) -> str:
        """Return the query for one planned feature."""
        if feature.operation == "anchor":
            return self._get_select(self._get_anchor_value(feature), [])
        if feature.operation == "direct":
            joins = [self._get_entity_join()]
            visible = self._get_visibility([self.entity_table])
            value = self._get_value(feature, self.entity_table)
            return self._get_select(_when(visible, value), joins)
        path = feature.path
        if path is None:
            raise ValueError(f"{feature.name!r} has no relation path")
        if feature.operation == "lookup":
            joins = [self._get_entity_join()]
            joins.extend(
                f"LEFT JOIN {_quote(edge.parent_table)} "
                f"ON {self._join(edge, edge.parent_table)}"
                for edge in path.edges
            )
            tables = [self.entity_table, *(edge.parent_table for edge in path.edges)]
            visible = self._get_visibility(tables)
            value = self._get_value(feature, path.terminal_table)
            return self._get_select(_when(visible, value), joins)
        if feature.operation == "mode":
            return self._get_mode_sql(feature, path)
        return self._get_aggregate_sql(feature, path)

    def get_joined_pairs_sql(self, path: RelationPath) -> str:
        """Return the number of anchor and path-row pairs whose keys match.

        A feature query joins each anchor to its entity's path rows before it
        filters them by time, so this count bounds the query's work.
        """
        path_rows = self._get_path_rows(path, column=None, tie=False, mode=False)
        rows, anchor = self.rows, self.anchor
        entity = _quote(self.eval_entity)
        key = self._equals(
            f"{rows}.entity_key",
            (self.entity_table, self.entity_key),
            f"{anchor}.{entity}",
            (EVAL_TABLE, self.eval_entity),
        )
        return "\n".join(
            [
                f"WITH {self.path_rows} AS (",
                *_indent(path_rows),
                ")",
                f"SELECT COALESCE(SUM({rows}.n * {anchor}.n), 0) AS pairs",
                "FROM (",
                f"{_INDENT}SELECT entity_key, COUNT(*) AS n",
                f"{_INDENT}FROM {self.path_rows}",
                f"{_INDENT}GROUP BY entity_key",
                f") AS {rows}",
                "JOIN (",
                f"{_INDENT}SELECT {entity}, COUNT(*) AS n",
                f"{_INDENT}FROM {EVAL_TABLE}",
                f"{_INDENT}GROUP BY {entity}",
                f") AS {anchor}",
                f"ON {key}",
            ]
        )

    def _get_anchor_value(self, feature: FeatureSpec) -> str:
        values = {
            "entity_key": f"{self.anchor}.{_quote(self.eval_entity)}",
            "cutoff_epoch_days": f"EPOCH({self.cutoff}) / 86400.0",
            "cutoff_year": f"YEAR({self.cutoff})",
            "cutoff_month": f"MONTH({self.cutoff})",
            "cutoff_day": f"DAY({self.cutoff})",
            "cutoff_day_of_week": f"ISODOW({self.cutoff}) - 1",
        }
        if feature.column not in values:
            raise ValueError(f"unknown anchor feature {feature.column!r}")
        return values[feature.column]

    def _get_aggregate_sql(self, feature: FeatureSpec, path: RelationPath) -> str:
        """Return an aggregate over the path rows each anchor can see."""
        operation, rows = feature.operation, self.rows
        value = f"{rows}.value"
        if operation == "count":
            aggregate = f"COUNT({rows}.entity_key)"
        elif operation in _ROW_TIME_OPERATIONS and self.temporal:
            aggregate = _ROW_TIME_OPERATIONS[operation].format(
                cutoff=self.cutoff, rows=rows
            )
        elif operation in _VALUE_FUNCTIONS and feature.column is not None:
            aggregate = f"{_VALUE_FUNCTIONS[operation]}({value})"
        elif operation == "last" and feature.column is not None:
            order = f"{rows}.row_tie"
            if self.temporal:
                order = f"{rows}.row_time, {order}"
            aggregate = (
                f"LAST({value} ORDER BY {order}) FILTER (WHERE {value} IS NOT NULL)"
            )
        elif operation == "missing_fraction" and feature.column is not None:
            aggregate = (
                f"AVG(CASE WHEN {value} IS NULL THEN 1.0 ELSE 0.0 END) "
                f"FILTER (WHERE {rows}.entity_key IS NOT NULL)"
            )
        else:
            raise ValueError(f"cannot write {feature.name!r} as SQL")
        path_rows = self._get_path_rows(
            path, column=feature.column, tie=operation == "last", mode=False
        )
        group_by = [f"{self.anchor}.{ROW_COLUMN}"]
        if operation in _ROW_TIME_OPERATIONS:
            group_by.append(self.cutoff)
        return "\n".join(
            [
                f"WITH {self.path_rows} AS (",
                *_indent(path_rows),
                ")",
                "SELECT",
                f"{_INDENT}{self.anchor}.{ROW_COLUMN},",
                f"{_INDENT}{aggregate} AS {FEATURE_COLUMN}",
                f"FROM {EVAL_TABLE} AS {self.anchor}",
                f"LEFT JOIN {self.path_rows} AS {rows}",
                *_indent(self._get_row_conditions(feature)),
                f"GROUP BY {', '.join(group_by)}",
                f"ORDER BY {self.anchor}.{ROW_COLUMN}",
            ]
        )

    def _get_mode_sql(self, feature: FeatureSpec, path: RelationPath) -> str:
        """Return the most frequent value; equal counts take the lowest rank."""
        if feature.column is None:
            raise ValueError(f"{feature.name!r} has no source column")
        anchor, rows, counts = self.anchor, self.rows, self.counts
        path_rows = self._get_path_rows(
            path, column=feature.column, tie=False, mode=True
        )
        value_counts = [
            "SELECT",
            f"{_INDENT}{anchor}.{ROW_COLUMN},",
            f"{_INDENT}{rows}.value,",
            f"{_INDENT}{rows}.value_rank,",
            f"{_INDENT}COUNT(*) AS n",
            f"FROM {EVAL_TABLE} AS {anchor}",
            f"JOIN {self.path_rows} AS {rows}",
            *_indent(self._get_row_conditions(feature)),
            f"WHERE {rows}.value IS NOT NULL",
            f"GROUP BY {anchor}.{ROW_COLUMN}, {rows}.value, {rows}.value_rank",
        ]
        return "\n".join(
            [
                f"WITH {self.path_rows} AS (",
                *_indent(path_rows),
                f"), {self.value_counts} AS (",
                *_indent(value_counts),
                ")",
                "SELECT",
                f"{_INDENT}{anchor}.{ROW_COLUMN},",
                f"{_INDENT}FIRST({counts}.value ORDER BY {counts}.n DESC, "
                f"{counts}.value_rank) AS {FEATURE_COLUMN}",
                f"FROM {EVAL_TABLE} AS {anchor}",
                f"LEFT JOIN {self.value_counts} AS {counts}",
                f"{_INDENT}ON {counts}.{ROW_COLUMN} = {anchor}.{ROW_COLUMN}",
                f"GROUP BY {anchor}.{ROW_COLUMN}",
                f"ORDER BY {anchor}.{ROW_COLUMN}",
            ]
        )

    def _get_mode_value(self, source: str) -> str:
        """Return the counted value: the column, or text with missing as a value.

        With missing values counted, one column holds them and the database
        values, so all become text. A database value spelled like the missing
        category is escaped, as the executor's ``_category`` does, so it never
        counts as missing.
        """
        if not self.plan.config.include_missing_in_modes:
            return source
        missing = _literal(_MISSING_CATEGORY)
        escape = _literal(_CATEGORY_ESCAPE)
        text = f"CAST({source} AS VARCHAR)"
        return (
            f"CASE WHEN {source} IS NULL THEN {missing} "
            f"WHEN {text} = {missing} OR STARTS_WITH({text}, {escape}) "
            f"THEN {escape} || {text} ELSE {text} END"
        )

    def _get_path_tables(self, path: RelationPath) -> list[_PathTable]:
        """Return the tables a path joins, in order, each with its SQL name.

        A table joined a second time, as when a bridge's association table
        is the entity table itself, gets a numbered name.
        """
        tables = [_PathTable(self.entity_table, self.entity_table)]
        used = {self.entity_table.lower()}
        for position, edge in enumerate(path.edges):
            outward = path.direction == "descendant" or position == 0
            table = edge.child_table if outward else edge.parent_table
            name = table
            if name.lower() in used:
                name = _get_unused_name(f"{table}_2", self.taken | used)
            used.add(name.lower())
            tables.append(_PathTable(table, name))
        return tables

    def _get_path_rows(
        self,
        path: RelationPath,
        *,
        column: str | None,
        tie: bool,
        mode: bool,
    ) -> list[str]:
        """Return the CTE body: one row per occurrence the path reaches.

        Inner joins from the entity table drop rows whose keys reach no
        entity, as the executor's joins do. A temporal plan keeps only rows
        whose declared times are all present and dates each row by the latest
        of them.
        """
        tables = self._get_path_tables(path)
        joins = []
        for position, edge in enumerate(path.edges):
            previous, joined = tables[position], tables[position + 1]
            outward = path.direction == "descendant" or position == 0
            child, parent = (joined, previous) if outward else (previous, joined)
            joined_column = edge.child_column if outward else edge.parent_column
            previous_column = edge.parent_column if outward else edge.child_column
            condition = self._equals(
                _column(joined.name, joined_column),
                (joined.table, joined_column),
                _column(previous.name, previous_column),
                (previous.table, previous_column),
            )
            if child.table != edge.child_table or parent.table != edge.parent_table:
                raise RuntimeError(f"{path.identity!r} does not join as its edges say")
            joins.append(f"JOIN {joined.get_from_item()} ON {condition}")
        entity = tables[0]
        selected = [f"{_column(entity.name, self.entity_key)} AS entity_key"]
        times = [
            _column(item.name, self.time_columns[item.table])
            for item in tables
            if item.table in self.time_columns
        ]
        if self.temporal:
            if not times:
                row_time = "CAST(NULL AS TIMESTAMP_NS)"
            elif len(times) == 1:
                row_time = times[0]
            else:
                row_time = f"GREATEST({', '.join(times)})"
            selected.append(f"{row_time} AS row_time")
        if tie:
            # A bridge's rows are its association rows; a descendant's, its
            # terminal rows. That table holds the path's tie column.
            row_table = tables[1] if path.is_bridge else tables[-1]
            tie_column = _column(row_table.name, get_tie_column(self.plan, path))
            selected.append(f"{tie_column} AS row_tie")
        if column is not None:
            source = _column(tables[-1].name, column)
            value = self._get_mode_value(source) if mode else source
            selected.append(f"{value} AS value")
            if mode:
                rank = _column(tables[-1].name, get_rank_column(column))
                selected.append(f"{rank} AS value_rank")
        lines = [
            "SELECT",
            *(
                f"{_INDENT}{item}{',' if position < len(selected) - 1 else ''}"
                for position, item in enumerate(selected)
            ),
            f"FROM {entity.get_from_item()}",
            *joins,
        ]
        if self.temporal and times:
            lines.append(
                "WHERE " + " AND ".join(f"{time} IS NOT NULL" for time in times)
            )
        return lines

    def _get_row_conditions(self, feature: FeatureSpec) -> list[str]:
        """Return the ON clause that matches path rows an anchor can see."""
        entity = self._equals(
            f"{self.rows}.entity_key",
            (self.entity_table, self.entity_key),
            f"{self.anchor}.{_quote(self.eval_entity)}",
            (EVAL_TABLE, self.eval_entity),
        )
        conditions = [f"ON {entity}"]
        if self.temporal:
            conditions.append(f"AND {self.rows}.row_time < {self.cutoff}")
            if feature.window_ns is not None:
                conditions.append(f"AND {self._get_window_start(feature.window_ns)}")
        return conditions

    def _get_window_start(self, window_ns: int) -> str:
        """Return the condition that a row falls in the window before the cutoff."""
        for unit, unit_ns in _INTERVAL_UNITS:
            if window_ns % unit_ns == 0:
                interval = f"INTERVAL '{window_ns // unit_ns} {unit}'"
                return f"{self.rows}.row_time >= {self.cutoff} - {interval}"
        return (
            f"EPOCH_NS({self.rows}.row_time) >= EPOCH_NS({self.cutoff}) - {window_ns}"
        )

    def _get_select(self, value: str, joins: list[str]) -> str:
        return "\n".join(
            [
                "SELECT",
                f"{_INDENT}{self.anchor}.{ROW_COLUMN},",
                f"{_INDENT}{value} AS {FEATURE_COLUMN}",
                f"FROM {EVAL_TABLE} AS {self.anchor}",
                *joins,
                f"ORDER BY {self.anchor}.{ROW_COLUMN}",
            ]
        )

    def _get_entity_join(self) -> str:
        condition = self._equals(
            _column(self.entity_table, self.entity_key),
            (self.entity_table, self.entity_key),
            f"{self.anchor}.{_quote(self.eval_entity)}",
            (EVAL_TABLE, self.eval_entity),
        )
        return f"LEFT JOIN {_quote(self.entity_table)} ON {condition}"

    def _get_value(self, feature: FeatureSpec, table: str) -> str:
        """Return a direct or lookup value; a datetime becomes a number of days."""
        if feature.column is None:
            raise ValueError(f"{feature.name!r} has no source column")
        value = _column(table, feature.column)
        if feature.kind != "datetime":
            return value
        if self.temporal:
            return f"(EPOCH({self.cutoff}) - EPOCH({value})) / 86400.0"
        return f"EPOCH({value}) / 86400.0"

    def _get_visibility(self, tables: list[str]) -> list[str]:
        """Return, per timed table, the condition that its time is before the cutoff."""
        if not self.temporal:
            return []
        return [
            f"{_column(table, self.time_columns[table])} < {self.cutoff}"
            for table in tables
            if table in self.time_columns
        ]

    def _join(self, edge: ForeignKey, joined_table: str) -> str:
        """Return the key condition for ``edge``, the joined table's side first."""
        child = (edge.child_table, edge.child_column)
        parent = (edge.parent_table, edge.parent_column)
        first, second = (
            (child, parent) if joined_table == edge.child_table else (parent, child)
        )
        return self._equals(_column(*first), first, _column(*second), second)

    def _equals(
        self,
        left: str,
        left_column: tuple[str, str],
        right: str,
        right_column: tuple[str, str],
    ) -> str:
        """Return a key comparison; keys the executor compares as text are cast."""
        if _compares_as_text(self.types.get(left_column), self.types.get(right_column)):
            return f"CAST({left} AS VARCHAR) = CAST({right} AS VARCHAR)"
        return f"{left} = {right}"


def _compares_as_text(left: ColumnType | None, right: ColumnType | None) -> bool:
    """Return whether the executor's ``_aligned_keys`` compares two keys as text.

    It keeps keys of one dtype, and integer, numeric or string pairs, as they
    are; any other pair becomes strings on both sides.
    """
    if left is None or right is None:
        return False
    if left.dtype == right.dtype and not left.categorical:
        return False
    return not (
        (left.integer and right.integer)
        or (left.numeric and right.numeric)
        or (left.text and right.text)
    )


def _quote(name: str) -> str:
    """Return a DuckDB identifier, quoted so that any database name survives."""
    return '"' + name.replace('"', '""') + '"'


def _literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _column(table: str, column: str) -> str:
    return f"{_quote(table)}.{_quote(column)}"


def _when(conditions: list[str], value: str) -> str:
    """Return ``value`` where every condition holds, otherwise NULL."""
    if not conditions:
        return value
    return f"CASE WHEN {' AND '.join(conditions)} THEN {value} END"


def _indent(lines: list[str]) -> list[str]:
    return [f"{_INDENT}{line}" for line in lines]


def _get_unused_name(preferred: str, taken: set[str]) -> str:
    """Return ``preferred``, numbered if needed, so no table name shadows it."""
    name, number = preferred, 1
    while name.lower() in taken:
        name, number = f"{preferred}{number}", number + 1
    return name


def _get_sql_tables(
    database: DatabaseView, plan: FeaturePlan
) -> dict[str, pd.DataFrame]:
    """Return the columns the plan's SQL reads, table by table, for DuckDB.

    Each table keeps its keys, its time column and the columns features read.
    Time columns and datetimes hold the executor's own UTC nanoseconds, so
    both read the same instants. Paths with a ``last`` feature get their tie
    column, and each column a mode reads gets its rank column.
    """
    entity_table = plan.anchors.entity_table
    used: dict[str, set[str]] = {entity_table: {database.primary_keys[entity_table]}}
    for path in plan.paths:
        for edge in path.edges:
            used.setdefault(edge.child_table, set()).add(edge.child_column)
            used.setdefault(edge.parent_table, set()).add(edge.parent_column)
    for feature in plan.features:
        if feature.operation != "anchor" and feature.column is not None:
            used.setdefault(feature.table, set()).add(feature.column)
    for table, columns in used.items():
        if table in database.time_columns:
            columns.add(database.time_columns[table])
    lowered = [name.lower() for name in used]
    if EVAL_TABLE in lowered or len(set(lowered)) != len(lowered):
        raise ValueError(
            "DuckDB cannot hold these tables: one is named eval_table, or two "
            "names differ only in case"
        )
    tables = {}
    for name in sorted(used):
        source = database.tables[name]
        columns = [column for column in source.columns if column in used[name]]
        frame = source[columns].reset_index(drop=True).copy()
        for column in frame.columns:
            values = frame[column]
            if column == database.time_columns.get(name) or (
                pd.api.types.is_datetime64_any_dtype(values)
            ):
                frame[column] = _get_naive_utc(values)
            elif isinstance(values.dtype, pd.CategoricalDtype):
                frame[column] = values.astype(object)
        tables[name] = frame
    for path in plan.paths:
        features = [feature for feature in plan.features if feature.path == path]
        if any(feature.operation == "last" for feature in features):
            columns = sorted({f.column for f in features if f.column is not None})
            _add_helper_column(
                tables[get_path_row_table(path)],
                get_tie_column(plan, path),
                get_path_row_ties(database, path, columns),
            )
    ranked: set[tuple[str, str]] = set()
    for feature in plan.features:
        if feature.operation != "mode" or feature.column is None:
            continue
        if (feature.table, feature.column) in ranked:
            continue
        ranked.add((feature.table, feature.column))
        _add_helper_column(
            tables[feature.table],
            get_rank_column(feature.column),
            _get_category_ranks(
                database.tables[feature.table][feature.column],
                include_missing=plan.config.include_missing_in_modes,
            ),
        )
    return tables


def _add_helper_column(frame: pd.DataFrame, column: str, values: object) -> None:
    if column in frame:
        raise ValueError(f"helper column {column!r} is already a database column")
    frame[column] = values


def _get_category_ranks(values: pd.Series, *, include_missing: bool) -> pd.Series:
    """Return each value's place in the executor's order for equal mode counts.

    Text and boolean columns are ranked once per distinct value; any other
    column, which may mix types, value by value.
    """
    if pd.api.types.is_bool_dtype(values) or (
        values.dtype == object
        and pd.api.types.infer_dtype(values, skipna=True) in ("string", "empty")
    ):
        return _get_distinct_ranks(values, include_missing=include_missing)
    return _get_scanned_ranks(values, include_missing=include_missing)


def _get_scanned_ranks(values: pd.Series, *, include_missing: bool) -> pd.Series:
    """Rank a column as ``_get_category_ranks`` does, value by value."""
    cache: dict[tuple[type, object], tuple[str, str] | None] = {}

    def get_identity(value: object) -> tuple[str, str] | None:
        try:
            return cache[(type(value), value)]
        except KeyError:
            category = _category(value, include_missing=include_missing)
            identity = None if category is None else category[0]
            cache[(type(value), value)] = identity
            return identity
        except TypeError:
            category = _category(value, include_missing=include_missing)
            return None if category is None else category[0]

    identities = [get_identity(value) for value in values]
    order = {
        identity: position
        for position, identity in enumerate(
            sorted({identity for identity in identities if identity is not None})
        )
    }
    return pd.Series(
        [None if identity is None else order[identity] for identity in identities],
        dtype="Int64",
    )


def _get_distinct_ranks(values: pd.Series, *, include_missing: bool) -> pd.Series:
    """Rank text or booleans as ``_get_category_ranks`` does, once per distinct value.

    Equal values of one type share an identity, so ranking each distinct value
    once gives every row the rank the value-by-value scan gives it.
    """
    codes, distinct = pd.factorize(values)
    identities = []
    for value in distinct:
        category = _category(value, include_missing=include_missing)
        if category is None:
            raise RuntimeError("a present value has no category")
        identities.append(category[0])
    missing = codes < 0
    if include_missing and missing.any():
        category = _category(None, include_missing=True)
        if category is None:
            raise RuntimeError("the missing category has no identity")
        codes = np.where(missing, len(identities), codes)
        identities.append(category[0])
        missing = np.zeros(len(codes), dtype=bool)
    order = {
        identity: position for position, identity in enumerate(sorted(set(identities)))
    }
    # The last slot only fills rows that stay missing.
    ranks = np.array([order[identity] for identity in identities] + [0])
    result = pd.Series(ranks[np.where(missing, len(identities), codes)], dtype="Int64")
    result[missing] = pd.NA
    return result


def _get_naive_utc(values: pd.Series) -> pd.Series:
    """Return times as the executor reads them: naive UTC nanoseconds."""
    nanoseconds = _time_ns(values)
    present = ~np.isnan(nanoseconds)
    integers = np.zeros(len(nanoseconds), dtype=np.int64)
    integers[present] = nanoseconds[present].astype(np.int64)
    times = pd.Series(integers.view("datetime64[ns]"))
    times[~present] = pd.NaT
    return times


def _get_eval_table(plan: FeaturePlan, anchors: pd.DataFrame) -> pd.DataFrame:
    columns: dict[str, object] = {
        ROW_COLUMN: np.arange(len(anchors), dtype=np.int64),
        get_eval_entity_column(plan): anchors[plan.anchors.entity_column].to_numpy(),
    }
    if plan.anchors.cutoff_column is not None:
        cutoffs = _get_naive_utc(anchors[plan.anchors.cutoff_column])
        columns[CUTOFF_COLUMN] = cutoffs.to_numpy()
    return pd.DataFrame(columns)


def _load_table(connection, name: str, frame: pd.DataFrame) -> None:
    """Copy a frame into the DuckDB catalog under its own name."""
    connection.register("__nr_source__", frame)
    try:
        connection.execute(
            f"CREATE TABLE {_quote(name)} AS SELECT * FROM __nr_source__"
        )
    finally:
        connection.unregister("__nr_source__")


__all__ = [
    "CUTOFF_COLUMN",
    "EVAL_TABLE",
    "FEATURE_COLUMN",
    "ROW_COLUMN",
    "SQLSchema",
    "get_eval_entity_column",
    "get_features_from_sql",
    "get_plan_sql",
    "get_sql_schema",
]
