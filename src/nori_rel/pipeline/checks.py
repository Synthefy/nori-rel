"""Checks that a saved pipeline still fits the database and the request."""

from __future__ import annotations

import importlib.metadata
import json
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

from nori_rel.contracts import SchemaGraph, TableSchema
from nori_rel.pipeline.errors import (
    PipelineContextError,
    PipelineFeatureError,
    PipelineSchemaError,
    PipelineTaskError,
    PipelineVersionWarning,
)
from nori_rel.pipeline.task import get_pql_task_signature

# The manifest records load the NoriDFS and agent modules; the annotations
# need them, the checks do not, so a FastDFS-only client can import this file.
if TYPE_CHECKING:
    from nori_rel.pipeline.manifest import ColumnsRecord, PipelineManifest, SourceRecord

#: At most this many differences are listed in one schema error.
MAX_LISTED_SCHEMA_DIFFERENCES = 10


def get_pipeline_query(manifest: PipelineManifest, query: str | None) -> str:
    """Return the PQL to run: the saved query, or one for other entities.

    A different ``query`` may change only its FOR clause; its target, entity
    and WHERE must be the saved ones.
    """
    if query is None:
        return manifest.task.pql_query
    if get_pql_task_signature(query) != manifest.task.signature:
        raise PipelineTaskError(
            "the query predicts a different task than the pipeline: only the "
            "FOR clause may change. Pipeline query: "
            f"{manifest.task.pql_query!r}"
        )
    return query


def require_same_schema(saved: SourceRecord, engine: str, schema: SchemaGraph) -> None:
    """Raise ``PipelineSchemaError`` naming each difference from the saved schema."""
    differences = get_schema_differences(saved, engine, schema)
    if differences:
        listed = differences[:MAX_LISTED_SCHEMA_DIFFERENCES]
        more = len(differences) - len(listed)
        raise PipelineSchemaError(
            "the database schema changed after the pipeline was saved: "
            + "; ".join(listed)
            + (f"; and {more} more" if more else "")
        )


def get_schema_differences(
    saved: SourceRecord, engine: str, schema: SchemaGraph
) -> list[str]:
    """Return readable differences between the saved and the current schema."""
    differences = []
    if str(engine) != saved.engine.value:
        differences.append(f"engine is {engine}, saved {saved.engine.value}")
    if schema.schema_name != saved.schema_name:
        differences.append(
            f"schema is {schema.schema_name!r}, saved {saved.schema_name!r}"
        )
    current = {table.name: table for table in schema.tables}
    before = {table.name: table for table in saved.tables}
    differences += [
        f"table {name!r} is missing" for name in sorted(before - current.keys())
    ]
    differences += [
        f"table {name!r} is new" for name in sorted(current - before.keys())
    ]
    for name in sorted(before.keys() & current.keys()):
        differences += _get_table_differences(before[name], current[name])
    return differences


def require_expected_context(expected_sha256: str | None, actual_sha256: str) -> None:
    """Raise ``PipelineContextError`` if a requested context hash is not met.

    A pipeline builds fresh context on every run, so by default nothing is
    compared. Passing the ``context_rows_sha256`` of an earlier run asks to
    reproduce that run, and a run on other context rows fails.
    """
    if expected_sha256 is not None and expected_sha256 != actual_sha256:
        raise PipelineContextError(
            f"the context rows hash to {actual_sha256}, not the expected "
            f"{expected_sha256}: the history changed since that run, or the "
            "request uses another as_of, observed_through or query_filter"
        )


def warn_if_other_nori_rel(manifest: PipelineManifest) -> None:
    """Warn when the pipeline was saved by another version of nori-rel.

    The saved feature definitions are checked on their own, so another version
    is not an error. A new version can still change how a saved definition
    runs, so the difference is reported.
    """
    installed = importlib.metadata.version("nori-rel")
    if manifest.nori_rel_version != installed:
        warnings.warn(
            f"pipeline {manifest.pipeline_id} was saved with nori-rel "
            f"{manifest.nori_rel_version}; nori-rel {installed} is installed",
            PipelineVersionWarning,
            stacklevel=2,
        )


def require_same_columns(saved: ColumnsRecord, names: Sequence[str]) -> None:
    """Raise ``PipelineFeatureError`` unless the features match the saved order."""
    if tuple(names) == saved.names:
        return
    missing = [name for name in saved.names if name not in names]
    added = [name for name in names if name not in saved.names]
    detail = (
        f"missing {missing[:5]}, new {added[:5]}"
        if missing or added
        else "the same columns in another order"
    )
    raise PipelineFeatureError(
        f"the saved features produced different columns ({detail}); "
        "export the pipeline again"
    )


def _get_table_differences(saved: TableSchema, current: TableSchema) -> list[str]:
    """Return the differences between one saved table and its current form."""
    name = saved.name
    differences = []
    before = {column.name: column for column in saved.columns}
    after = {column.name: column for column in current.columns}
    differences += [
        f"column {name}.{column} is missing" for column in sorted(before - after.keys())
    ]
    differences += [
        f"column {name}.{column} is new" for column in sorted(after - before.keys())
    ]
    for column in sorted(before.keys() & after.keys()):
        was, now = before[column], after[column]
        if (was.data_type, was.nullable) != (now.data_type, now.nullable):
            differences.append(
                f"column {name}.{column} is {_get_column_type(now)}, "
                f"saved {_get_column_type(was)}"
            )
    if saved.primary_key != current.primary_key:
        differences.append(
            f"primary key of {name} is {current.primary_key}, saved {saved.primary_key}"
        )
    if _get_foreign_key_names(saved) != _get_foreign_key_names(current):
        differences.append(f"foreign keys of {name} changed")
    if saved.time_column != current.time_column:
        differences.append(
            f"time column of {name} is {current.time_column!r}, "
            f"saved {saved.time_column!r}"
        )
    return differences


def _get_column_type(column) -> str:
    """Return a column's data type and nullability as text."""
    return f"{column.data_type} {'NULL' if column.nullable else 'NOT NULL'}"


def _get_foreign_key_names(table: TableSchema) -> set[str]:
    """Return each foreign key of a table as one comparable JSON string."""
    return {
        json.dumps(key.model_dump(mode="json"), sort_keys=True)
        for key in table.foreign_keys
    }


__all__ = [
    "MAX_LISTED_SCHEMA_DIFFERENCES",
    "get_pipeline_query",
    "get_schema_differences",
    "require_expected_context",
    "require_same_columns",
    "require_same_schema",
    "warn_if_other_nori_rel",
]
