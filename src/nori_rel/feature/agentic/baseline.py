"""Plain-language descriptions of the DFS columns an extending search starts from.

NoriDFS names its columns by path identity, URL-escaped, which is exact and
unreadable: an agent handed forty of them cannot tell a count from a mean or
see that a table's value columns were never aggregated. This module turns the
fitted plan into one sentence per column and a list of what the plan leaves
unread, which is where an agent's own queries have room to help.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from ..dfs.noridfs.contracts import DatabaseView, FeaturePlan, FeatureSpec, RelationPath

_DAY_NS = 24 * 60 * 60 * 1_000_000_000

#: What each row-level operation measures, with the table's name filled in.
_ROW_OPERATIONS = {
    "count": "number of {table} rows",
    "recency": "time since the latest {table} row",
    "mean_age": "mean age of the {table} rows",
    "oldest_age": "age of the oldest {table} row",
}
#: What each column-level operation measures, with ``table.column`` filled in.
_COLUMN_OPERATIONS = {
    "mean": "mean of {source}",
    "sum": "sum of {source}",
    "min": "minimum of {source}",
    "max": "maximum of {source}",
    "std": "standard deviation of {source}",
    "last": "latest {source}",
    "mode": "most common {source}",
    "missing_fraction": "fraction of {source} that is missing",
    "lookup": "{source} of the linked row",
    "direct": "the entity's own {source}",
    "anchor": "{source} of the prediction anchor",
    "count": "number of non-missing {source} values",
}


@dataclass(frozen=True)
class BaselineColumn:
    """One DFS column as the agent sees it.

    ``name`` is the exact column name, which is what a program drops by.
    ``reads`` is the ``table.column`` it aggregates, or ``table`` for a row
    count or age, so the agent can tell which source columns are covered.
    """

    name: str
    kind: str
    reads: str
    description: str


def describe_plan(plan: FeaturePlan) -> tuple[BaselineColumn, ...]:
    """Return one plain-language description per column of a fitted plan."""
    return tuple(_describe(feature) for feature in plan.features)


def unread_columns(plan: FeaturePlan, database: DatabaseView) -> dict[str, list[str]]:
    """Return, per table, the value columns no DFS column reads.

    Keys, foreign keys, and time columns are structure rather than values and
    are left out, as are tables where every value column is already read.
    """
    read = {
        (feature.table, feature.column)
        for feature in plan.features
        if feature.column is not None
    }
    keys = {(edge.child_table, edge.child_column) for edge in database.foreign_keys}
    unread: dict[str, list[str]] = {}
    for table, frame in database.tables.items():
        structural = {
            database.primary_keys.get(table),
            database.time_columns.get(table),
        }
        columns = [
            column
            for column in frame.columns
            if column not in structural
            and (table, column) not in keys
            and (table, column) not in read
            and _is_value(frame[column])
        ]
        if columns:
            unread[table] = columns
    return unread


def _describe(feature: FeatureSpec) -> BaselineColumn:
    source = (
        feature.table if feature.column is None else f"{feature.table}.{feature.column}"
    )
    if feature.column is None and feature.operation in _ROW_OPERATIONS:
        what = _ROW_OPERATIONS[feature.operation].format(table=feature.table)
    else:
        template = _COLUMN_OPERATIONS.get(feature.operation, "{source}")
        what = template.format(source=source)
    parts = [what]
    if feature.path is not None:
        parts.append(_route(feature.path))
    if feature.operation not in {"anchor", "direct", "lookup"}:
        parts.append(_window(feature.window_ns))
    return BaselineColumn(
        name=feature.name,
        kind=feature.kind,
        reads=source,
        description=", ".join(parts),
    )


def _route(path: RelationPath) -> str:
    """Say how the path reaches its rows from the entity."""
    if path.is_parent:
        hops = " then ".join(edge.parent_table for edge in path.edges)
        return f"on the parent {hops}"
    if path.is_bridge:
        first, *parents = path.edges
        hops = " then ".join(edge.parent_table for edge in parents)
        return (
            f"over the {hops} rows reached through the entity's "
            f"{first.child_table} rows"
        )
    hops = " <- ".join(
        f"{edge.child_table} (by {edge.child_column})" for edge in path.edges
    )
    return f"over the entity's {hops} rows"


def _window(window_ns: int | None) -> str:
    if window_ns is None:
        return "all history"
    days = window_ns / _DAY_NS
    return f"last {days:g} days" if days >= 1 else f"last {window_ns / 3.6e12:g} hours"


def _is_value(series: pd.Series) -> bool:
    """Return whether a column carries values an aggregate could use."""
    return bool(
        pd.api.types.is_numeric_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )


def catalog(
    plan: FeaturePlan | None,
    features: pd.DataFrame,
    kinds: Mapping[str, object],
) -> list[dict[str, object]]:
    """Return the baseline columns as the schema reports them.

    With a plan each entry carries the description and what it reads; without
    one, as for a fallback that is not NoriDFS, only the name and kind.
    """
    described = {} if plan is None else {c.name: c for c in describe_plan(plan)}
    entries: list[dict[str, object]] = []
    for name in features.columns:
        entry: dict[str, object] = {
            "name": name,
            "kind": kinds[name],
            "null_fraction": round(float(features[name].isna().mean()), 3),
        }
        column = described.get(name)
        if column is not None:
            entry["reads"] = column.reads
            entry["description"] = column.description
        entries.append(entry)
    return entries


__all__ = ["BaselineColumn", "catalog", "describe_plan", "unread_columns"]
