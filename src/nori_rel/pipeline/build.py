"""Record a finished prediction as a pipeline manifest."""

from __future__ import annotations

import importlib.metadata
import math
import numbers
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from nori_rel._feature_program import SavedFeatureProgram, program_record
from nori_rel.context_policy import ContextPolicy
from nori_rel.contracts import PredictionTask, RelationalOutputType, SchemaGraph
from nori_rel.feature import FeatureBackend, FeatureMatrices
from nori_rel.feature.dfs.noridfs import NoriDFS
from nori_rel.pipeline.features import (
    get_installed_fastdfs_version,
    get_plan_record,
    get_table_dtypes,
)
from nori_rel.pipeline.manifest import (
    AgenticFeaturesRecord,
    ContextPolicyRecord,
    FastDFSFeaturesRecord,
    FeaturesRecord,
    NoriDFSFeaturesRecord,
    PipelineManifest,
    get_pql_task_signature,
)

#: Prediction metadata copied into a pipeline's provenance.
PROVENANCE_KEYS = (
    "features",
    "candidate_rows",
    "usable_rows",
    "dropped_unknown_labels",
    "feature_count",
    "agentic_model",
    "agentic_cost_usd",
    "agentic_usage",
    "agentic_stop_reason",
    "agentic_trials",
    "context_rows_sha256",
    "feature_matrix_sha256",
    "context_policy",
)


def get_pipeline_manifest(
    *,
    engine: str,
    query: str,
    entity_created_at: str,
    task: PredictionTask,
    as_of: datetime,
    anchor_times: Sequence[datetime],
    max_context_rows: int,
    context_policy: ContextPolicy | None = None,
    schema: SchemaGraph,
    tables: dict[str, pd.DataFrame],
    backend: FeatureBackend,
    matrices: FeatureMatrices,
    categorical_columns: Sequence[str] | None,
    predictor: Any,
    output_type: RelationalOutputType | None,
    quantiles: Sequence[float] | None,
    decision_threshold: float | None,
    metadata: dict[str, Any],
) -> PipelineManifest | None:
    """Return the manifest of a finished prediction, or None if it has none.

    A prediction has no pipeline when its features came from a custom
    backend, or when its model cannot be pinned (a local checkpoint outside
    the Hugging Face cache, or a predictor without a model record).

    Args:
        engine: Database engine of the source, for example ``"postgresql"``.
        query: PQL text that was predicted.
        entity_created_at: Column of the entity table that holds the time
            each entity was created. Rows are only used from that time on.
        task: ``regression`` or ``classification``, resolved from ``query``.
        as_of: The prediction time of this run (the ``anchor_time`` the
            scoring rows were built at). Each context anchor is saved as its
            distance before ``as_of``.
        anchor_times: The historical times the context rows were built at.
        max_context_rows: Most labeled rows that Nori reads as context,
            under the newest-candidates rule.
        context_policy: The ContextPolicy that chose the context rows, or
            None for the newest-candidates rule; saved for replay.
        schema: Tables, columns, keys and time columns the run read.
        tables: The snapshot tables of the run, for their dtypes.
        backend: The feature backend that built the matrices.
        matrices: The feature matrices Nori read. Their columns, in order,
            are saved.
        categorical_columns: Columns declared categorical to Nori, or None
            to keep Nori's own detection.
        predictor: The Nori predictor of the run; it pins the model.
        output_type: Regression output, or None for classification.
        quantiles: Quantile levels for the quantile output, or None.
        decision_threshold: Probability cutoff for classification, or None.
        metadata: Prediction metadata; the keys in ``PROVENANCE_KEYS`` are
            copied to the pipeline's provenance.
    """
    features = _get_features_record(
        backend, query=query, entity_created_at=entity_created_at, schema=schema
    )
    record_model = getattr(predictor, "get_pipeline_model_record", None)
    model = None if record_model is None else record_model()
    if features is None or model is None:
        return None
    columns = list(matrices.feature_names)
    return PipelineManifest.create(
        created_at=datetime.now(UTC),
        nori_rel_version=importlib.metadata.version("nori-rel"),
        task={
            "pql_query": query,
            "signature": get_pql_task_signature(query),
            "entity_created_at": entity_created_at,
            "task": task,
        },
        source={
            "engine": engine,
            "schema_name": schema.schema_name,
            "tables": schema.tables,
            "column_dtypes": get_table_dtypes(tables),
        },
        features=features,
        columns={
            "names": columns,
            "dtypes": [str(matrices.train[name].dtype) for name in columns],
            "categorical": (
                None if categorical_columns is None else list(categorical_columns)
            ),
        },
        model=model,
        prediction={
            "output_type": output_type,
            "quantiles": None if quantiles is None else list(quantiles),
            "decision_threshold": decision_threshold,
        },
        context_policy=ContextPolicyRecord.from_anchor_times(
            [_get_utc_datetime(value) for value in anchor_times],
            as_of=_get_utc_datetime(as_of),
            max_context_rows=max_context_rows,
            row_policy=context_policy,
        ),
        provenance=_get_json_ready_metadata(
            {key: metadata[key] for key in PROVENANCE_KEYS if key in metadata}
        ),
    )


def _get_features_record(
    backend: FeatureBackend,
    *,
    query: str,
    entity_created_at: str,
    schema: SchemaGraph,
) -> FeaturesRecord | None:
    """Return the record of the feature backend.

    The result is None when the backend has not run yet, and for a custom
    backend, which has no pipeline form.
    """
    if isinstance(backend, NoriDFS):
        return (
            None
            if backend.last_plan is None
            else NoriDFSFeaturesRecord(plan=get_plan_record(backend.last_plan))
        )
    from nori_rel.feature.agentic import AgenticFeatures

    if isinstance(backend, AgenticFeatures):
        return _get_agentic_record(
            backend, query=query, entity_created_at=entity_created_at, schema=schema
        )
    fastdfs = sys.modules.get("nori_rel.feature.dfs.fastdfs")
    if fastdfs is not None and type(backend) is fastdfs.FastDFS:
        if backend.last_column_types is None:
            return None
        return FastDFSFeaturesRecord(
            max_depth=backend.max_depth,
            max_features=backend.max_features,
            fastdfs_version=get_installed_fastdfs_version(),
            column_types=backend.last_column_types,
        )
    return None


def _get_agentic_record(
    backend: Any, *, query: str, entity_created_at: str, schema: SchemaGraph
) -> FeaturesRecord | None:
    """Record the saved program on its NoriDFS plan, or the plan alone."""
    fallback = backend.fallback
    plan = fallback.last_plan if isinstance(fallback, NoriDFS) else None
    config = backend.config
    if plan is None or not config.extend_baseline or config.allow_text:
        return None
    if backend.last_program is None:
        return NoriDFSFeaturesRecord(plan=get_plan_record(plan))
    program = program_record(
        backend.last_program,
        query=query,
        entity_created_at=entity_created_at,
        schema=schema,
        baseline_columns=backend.last_baseline_columns,
    )
    return AgenticFeaturesRecord(
        plan=get_plan_record(plan),
        program=SavedFeatureProgram.model_validate(program),
    )


def _get_utc_datetime(value: Any) -> datetime:
    """Return a timezone-aware time as a UTC ``datetime``."""
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise ValueError("pipeline times must be timezone-aware")
    return stamp.tz_convert(UTC).to_pydatetime()


def _get_json_ready_metadata(value: Any) -> Any:
    """Return metadata as plain JSON values; NaN and infinity become None."""
    if isinstance(value, dict):
        return {str(key): _get_json_ready_metadata(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_get_json_ready_metadata(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value) if math.isfinite(value) else None
    return str(value)


__all__ = ["PROVENANCE_KEYS", "get_pipeline_manifest"]
