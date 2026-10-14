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
            definition: the agent query's SQL, the NoriDFS name with its
                escapes decoded (path, column, aggregation and window), or the
                column name.
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
        sql_by_query = {
            query["name"]: query["sql"] for query in program.get("queries", [])
        }
        strategy = self.metadata.get("features", "unknown")
        rows = []
        for name in self.features.columns:
            values = self.features[name]
            source, definition = _get_feature_source_and_definition(
                str(name), sql_by_query, strategy
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


def _get_feature_source_and_definition(
    name: str, sql_by_query: dict[str, str], strategy: str
) -> tuple[str, str]:
    """Return a feature column's source and a readable definition.

    An agent query's columns are named ``<query>__<column>``; NoriDFS names
    start with ``nr|`` and percent-escape their parts.
    """
    for query, sql in sql_by_query.items():
        if name.startswith(f"{query}__"):
            return "agent", sql
    if name.startswith("nr|"):
        return "noridfs", unquote(name)
    return strategy, name


__all__ = ["PredictionDetails"]
