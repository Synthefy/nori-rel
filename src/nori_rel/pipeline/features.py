"""Saved FastDFS settings and stable input dtypes."""

from __future__ import annotations

import importlib.metadata
from collections.abc import Mapping

import pandas as pd

from nori_rel.feature import FeatureBackend
from nori_rel.pipeline.errors import PipelineFeatureError, PipelineSchemaError
from nori_rel.pipeline.manifest import FeaturesRecord


def get_saved_feature_backend(features: FeaturesRecord) -> FeatureBackend:
    """Return the backend that recomputes saved features without a search."""
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
    ``object``, whatever the database type is. Restore those columns to the
    dtype they had at save time.
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


__all__ = [
    "get_frame_with_saved_dtypes",
    "get_installed_fastdfs_version",
    "get_saved_feature_backend",
    "get_table_dtypes",
    "get_tables_with_saved_dtypes",
]
