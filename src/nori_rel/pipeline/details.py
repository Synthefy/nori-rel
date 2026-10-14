"""What ``predict(return_details=True)`` returns."""

from __future__ import annotations

from typing import Any
from urllib.parse import unquote

import pandas as pd
from pydantic import BaseModel, ConfigDict, SkipValidation

from nori_rel.pipeline.artifact import NoriRelPipeline


class PredictionDetails(BaseModel):
    """A prediction with its metadata and the pipeline that produced it.

    ``details["predictions"]``, ``details["metadata"]`` and
    ``details["pipeline"]`` work as they did when ``return_details`` returned
    a dict.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    predictions: pd.DataFrame
    metadata: dict[str, Any]
    #: ``None`` when the prediction cannot be exported (a custom feature
    #: backend, or a model that cannot be pinned).
    pipeline: SkipValidation[NoriRelPipeline | None]
    #: The scored rows' feature matrix the model predicted from: one row per
    #: entity_id, one column per feature. ``None`` when not kept.
    features: pd.DataFrame | None = None

    def __getitem__(self, key: str) -> Any:
        """Return a field by name, for code written against the old dict."""
        if key not in type(self).model_fields:
            raise KeyError(key)
        return getattr(self, key)

    def get_feature_table(self) -> pd.DataFrame:
        """Return one row per feature column the model used, for reading.

        Columns:
            feature: the column name in ``features``.
            source: "agent" for a column an agent's SQL query produced,
                "noridfs", or the prediction's feature strategy otherwise
                (for example "fastdfs").
            definition: in words. For an agent column, the column and the
                query that produced it (the query's SQL is in
                ``metadata["agentic_program"]["queries"]``); for NoriDFS, the
                aggregation, column, window and table, such as "mean of
                QTY_CANS over the last 30 days, in SHIPMENTS matched on
                CUSTOMER_ID"; otherwise the column name.
            dtype: the column's data type.
            missing_share: the share of scored entities with no value.
            mean, min, max: over the scored entities, for numeric columns.

        Raises:
            ValueError: the details hold no features.
        """
        if self.features is None:
            raise ValueError(
                "these details hold no features; call predict(return_details=True)"
            )
        program = self.metadata.get("agentic_program") or {}
        agent_queries = [query["name"] for query in program.get("queries", [])]
        strategy = self.metadata.get("features", "unknown")
        rows = []
        for name in self.features.columns:
            values = self.features[name]
            source, definition = _get_feature_source_and_definition(
                str(name), agent_queries, strategy
            )
            numeric = pd.api.types.is_numeric_dtype(
                values
            ) and not pd.api.types.is_bool_dtype(values)
            rows.append(
                {
                    "feature": name,
                    "source": source,
                    "definition": definition,
                    "dtype": str(values.dtype),
                    "missing_share": float(values.isna().mean()),
                    "mean": float(values.mean()) if numeric else None,
                    "min": float(values.min()) if numeric else None,
                    "max": float(values.max()) if numeric else None,
                }
            )
        return pd.DataFrame(rows)


_DAY_NS = 86_400 * 10**9

_NORIDFS_ANCHOR_DEFINITIONS = {
    "cutoff_epoch_days": "the anchor time, in days since 1970-01-01",
    "cutoff_year": "the anchor time's year",
    "cutoff_month": "the anchor time's month",
    "cutoff_day": "the anchor time's day of the month",
    "cutoff_day_of_week": "the anchor time's day of the week",
}

_NORIDFS_ROW_DEFINITIONS = {
    "count": "number of rows",
    "recency": "days since the latest row",
    "oldest_age": "days since the oldest row",
    "mean_age": "mean age of the rows in days",
}


def _get_feature_source_and_definition(
    name: str, agent_queries: list[str], strategy: str
) -> tuple[str, str]:
    """Return a feature column's source and its definition in words.

    An agent query's columns are named ``<query>__<column>``; NoriDFS names
    start with ``nr|``.
    """
    for query in agent_queries:
        if name.startswith(f"{query}__"):
            column = name.removeprefix(f"{query}__")
            return "agent", f"{column}, from the agent's {query!r} query"
    if name.startswith("nr|"):
        return "noridfs", _get_noridfs_definition(name)
    return strategy, name


def _get_noridfs_definition(name: str) -> str:
    """Return a NoriDFS column's definition in words.

    Names are ``nr|anchor|<component>``, ``nr|direct|<table>.<column>`` or
    ``nr|path|<path>|<column or "rows">|<operation>|<window>``, each part
    percent-escaped. A name of another form is returned with its escapes
    decoded.
    """
    parts = [unquote(part) for part in name.split("|")]
    if parts[1:2] == ["anchor"] and len(parts) == 3:
        return _NORIDFS_ANCHOR_DEFINITIONS.get(
            parts[2], f"{parts[2]} of the anchor time"
        )
    if parts[1:2] == ["direct"] and len(parts) == 3:
        return f"{parts[2]}, the entity's own column"
    if parts[1:2] != ["path"] or len(parts) != 6:
        return unquote(name)
    _, _, path, column, operation, window = parts
    if operation == "lookup":
        return f"{column}, looked up {_get_noridfs_path_phrase(path)}"
    if column == "rows":
        measure = _NORIDFS_ROW_DEFINITIONS.get(operation, f"{operation} of rows")
    else:
        measure = {
            "last": f"latest {column}",
            "mode": f"most common {column}",
            "std": f"standard deviation of {column}",
            "missing_fraction": f"share of rows missing {column}",
        }.get(operation, f"{operation} of {column}")
    if window == "all" and operation in {"last", "recency", "oldest_age"}:
        return f"{measure}, {_get_noridfs_path_phrase(path)}"
    return (
        f"{measure} {_get_noridfs_window_phrase(window)}, "
        f"{_get_noridfs_path_phrase(path)}"
    )


def _get_noridfs_window_phrase(window: str) -> str:
    """Return a NoriDFS window ("all" or "<nanoseconds>ns") in words."""
    if window.endswith("ns") and window[:-2].isdigit():
        return f"over the last {int(window[:-2]) / _DAY_NS:g} days"
    if window == "all":
        return "over all history"
    return f"over window {window}"


def _get_noridfs_path_phrase(path: str) -> str:
    """Return where a NoriDFS path reads from, in words.

    A one-step child path ``ENTITY<-TABLE.KEY`` reads "in TABLE matched on
    KEY"; a one-step parent path ``ENTITY=>ENTITY.KEY->PARENT.ID`` reads
    "from PARENT through KEY"; any other path is named as it is.
    """
    _, separator, step = path.partition("<-")
    if separator and "<-" not in step and "." in step:
        table, key = step.rsplit(".", 1)
        return f"in {table} matched on {key}"
    _, separator, step = path.partition("=>")
    if separator and "/" not in step and "->" in step:
        child, parent = step.split("->", 1)
        return f"from {parent.rsplit('.', 1)[0]} through {child.rsplit('.', 1)[-1]}"
    return f"along {path}"


__all__ = ["PredictionDetails"]
