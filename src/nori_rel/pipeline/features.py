"""Saved feature definitions: NoriDFS plans in JSON, and their dtypes."""

from __future__ import annotations

import dataclasses
import importlib.metadata
from collections.abc import Mapping

import pandas as pd

from nori_rel.feature import FeatureBackend
from nori_rel.feature.dfs.noridfs import NoriDFS
from nori_rel.feature.dfs.noridfs.contracts import (
    AnchorDefinition,
    FeaturePlan,
    FeatureSpec,
    ForeignKey,
    NativeFeatureConfig,
    RelationPath,
)
from nori_rel.pipeline.errors import PipelineFeatureError, PipelineSchemaError
from nori_rel.pipeline.manifest import (
    AgenticFeaturesRecord,
    AnchorRecord,
    FastDFSFeaturesRecord,
    FeaturePlanRecord,
    FeatureSpecRecord,
    FeaturesRecord,
    ForeignKeyRecord,
    NoriDFSFeaturesRecord,
    PlanConfigRecord,
    RelationPathRecord,
)


def get_plan_record(plan: FeaturePlan) -> FeaturePlanRecord:
    """Return a fitted NoriDFS plan as JSON-ready data."""
    return FeaturePlanRecord(
        schema_digest=plan.schema_digest,
        anchors=AnchorRecord(**dataclasses.asdict(plan.anchors)),
        config=PlanConfigRecord(**dataclasses.asdict(plan.config)),
        paths=tuple(_get_path_record(path) for path in plan.paths),
        features=tuple(_get_feature_record(feature) for feature in plan.features),
        digest=plan.digest,
    )


def get_plan_from_record(record: FeaturePlanRecord) -> FeaturePlan:
    """Rebuild a saved plan; refuse one that does not reproduce its digest."""
    try:
        plan = FeaturePlan(
            schema_digest=record.schema_digest,
            anchors=AnchorDefinition(**record.anchors.model_dump()),
            config=NativeFeatureConfig(**record.config.model_dump()),
            paths=tuple(_get_relation_path(path) for path in record.paths),
            features=tuple(_get_feature_spec(feature) for feature in record.features),
        )
    except (TypeError, ValueError) as exc:
        raise PipelineFeatureError(f"invalid saved NoriDFS plan: {exc}") from exc
    if plan.digest != record.digest:
        raise PipelineFeatureError(
            "the saved NoriDFS plan does not reproduce its digest; "
            "it was written by an incompatible nori-rel"
        )
    return plan


def get_saved_feature_backend(features: FeaturesRecord) -> FeatureBackend:
    """Return the backend that recomputes saved features without a search."""
    if isinstance(features, NoriDFSFeaturesRecord):
        return NoriDFS.from_plan(get_plan_from_record(features.plan))
    if isinstance(features, AgenticFeaturesRecord):
        from nori_rel._feature_search import (
            DEFAULT_SEARCH_CONFIG,
            require_agentic_modules,
        )
        from nori_rel.feature.agentic import AgenticFeatures

        require_agentic_modules("duckdb")
        return AgenticFeatures(
            program=features.program.to_program(),
            fallback=NoriDFS.from_plan(get_plan_from_record(features.plan)),
            config=DEFAULT_SEARCH_CONFIG,
            expected_baseline_columns=features.program.baseline_columns,
        )
    assert isinstance(features, FastDFSFeaturesRecord)
    installed = get_installed_fastdfs_version()
    if installed != features.fastdfs_version:
        raise PipelineFeatureError(
            f"the pipeline was saved with fastdfs {features.fastdfs_version}, "
            f"but fastdfs {installed} is installed; a feature of the same name "
            "can be computed differently. Install the saved version or export "
            "the pipeline again"
        )
    from nori_rel.feature.dfs.fastdfs import FastDFS

    return FastDFS(
        max_depth=features.max_depth,
        max_features=features.max_features,
        type_hints=features.column_types,
    )


def get_installed_fastdfs_version() -> str:
    """Return the version of the installed ``fastdfs`` package."""
    return importlib.metadata.version("fastdfs")


def get_table_dtypes(tables: Mapping[str, pd.DataFrame]) -> dict[str, dict[str, str]]:
    """Return each table's column dtypes, for ``get_tables_with_saved_dtypes`` later."""
    return {
        name: {str(column): str(dtype) for column, dtype in frame.dtypes.items()}
        for name, frame in tables.items()
    }


def get_tables_with_saved_dtypes(
    tables: Mapping[str, pd.DataFrame], saved: Mapping[str, Mapping[str, str]]
) -> dict[str, pd.DataFrame]:
    """Give all-NULL columns and empty tables their saved dtypes.

    A snapshot reads an all-NULL column as ``string`` and an empty table as
    ``object``, whatever the database type is. A saved NoriDFS plan refuses
    any dtype change, so those columns take the dtype they had at save time.
    A column with values and another dtype is a schema change and an error.
    """
    if set(tables) != set(saved):
        raise PipelineSchemaError(
            f"the snapshot has tables {sorted(tables)}; "
            f"the pipeline read {sorted(saved)}"
        )
    return {
        name: get_frame_with_saved_dtypes(
            frame, saved[name], location=f"table {name!r}"
        )
        for name, frame in tables.items()
    }


def get_frame_with_saved_dtypes(
    frame: pd.DataFrame, saved: Mapping[str, str], *, location: str
) -> pd.DataFrame:
    """Cast all-missing columns of ``frame`` to ``saved`` dtypes; see above."""
    if set(map(str, frame.columns)) != set(saved):
        raise PipelineSchemaError(
            f"{location} has columns {sorted(map(str, frame.columns))}; "
            f"the pipeline read {sorted(saved)}"
        )
    changed = {
        column: _get_restored_column(frame[column], saved[str(column)], location)
        for column in frame.columns
        if str(frame[column].dtype) != saved[str(column)]
    }
    return frame.assign(**changed) if changed else frame


def _get_restored_column(series: pd.Series, dtype: str, location: str) -> pd.Series:
    """Return an empty or all-NULL column with its saved dtype.

    A column with values and another dtype is a schema error.
    """
    current = str(series.dtype)
    if series.notna().any():
        raise PipelineSchemaError(
            f"column {series.name!r} of {location} reads as {current}; "
            f"the pipeline saved {dtype}"
        )
    if series.empty:
        return series.astype(dtype)
    restored = pd.Series(index=series.index, dtype=dtype, name=series.name)
    if str(restored.dtype) != dtype or restored.notna().any():
        raise PipelineSchemaError(
            f"column {series.name!r} of {location} is all NULL, which the "
            f"saved dtype {dtype} cannot hold"
        )
    return restored


def _get_path_record(path: RelationPath) -> RelationPathRecord:
    """Return the record of a relation path."""
    return RelationPathRecord(
        entity_table=path.entity_table,
        edges=tuple(
            ForeignKeyRecord(**dataclasses.asdict(edge)) for edge in path.edges
        ),
        direction=path.direction,
    )


def _get_relation_path(record: RelationPathRecord) -> RelationPath:
    """Return the relation path that a record holds."""
    return RelationPath(
        entity_table=record.entity_table,
        edges=tuple(ForeignKey(**edge.model_dump()) for edge in record.edges),
        direction=record.direction,
    )


def _get_feature_record(feature: FeatureSpec) -> FeatureSpecRecord:
    """Return the record of a feature spec."""
    return FeatureSpecRecord(
        name=feature.name,
        operation=feature.operation,
        kind=feature.kind,
        table=feature.table,
        column=feature.column,
        path=None if feature.path is None else _get_path_record(feature.path),
        window_ns=feature.window_ns,
        paired_path=None
        if feature.paired_path is None
        else _get_path_record(feature.paired_path),
        paired_column=feature.paired_column,
    )


def _get_feature_spec(record: FeatureSpecRecord) -> FeatureSpec:
    """Return the feature spec that a record holds."""
    return FeatureSpec(
        name=record.name,
        operation=record.operation,
        kind=record.kind,
        table=record.table,
        column=record.column,
        path=None if record.path is None else _get_relation_path(record.path),
        window_ns=record.window_ns,
        paired_path=None
        if record.paired_path is None
        else _get_relation_path(record.paired_path),
        paired_column=record.paired_column,
    )


__all__ = [
    "get_frame_with_saved_dtypes",
    "get_installed_fastdfs_version",
    "get_plan_from_record",
    "get_plan_record",
    "get_saved_feature_backend",
    "get_table_dtypes",
    "get_tables_with_saved_dtypes",
]
