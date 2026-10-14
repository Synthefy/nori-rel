"""Fingerprints of one run's inputs, recorded so a past prediction can be traced.

A pipeline runs again on new data, so these values are expected to change from
run to run. They are never compared with the saved pipeline. Each run reports
them in its prediction metadata and run summary; to reproduce one past run
exactly, pass its context hash as ``expected_context_sha256``.

The bytes are encoded here, not with ``pd.util.hash_pandas_object``, because
pandas does not promise to keep that function's output across versions.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from nori_rel.feature import FeatureMatrices

_NANOSECONDS_DTYPE = "datetime64[ns]"


def get_context_rows_sha256(context: pd.DataFrame, labels: pd.Series) -> str:
    """Hash the labeled context rows: entity, anchor time and label, in order.

    Anchor times are UTC nanoseconds as int64 and labels are little-endian
    float64. Integer entity IDs are int64 bytes; any other ID is a string, and
    a type tag keeps ``7`` and ``"7"`` apart.
    """
    if len(context) != len(labels):
        raise ValueError("labels must align with the context rows")
    digest = hashlib.sha256()
    digest.update(len(context).to_bytes(8, "little"))
    digest.update(_get_entity_bytes(context["entity_id"]))
    anchors = pd.to_datetime(context["anchor_time"], utc=True)
    digest.update(
        anchors.dt.tz_convert(None).to_numpy(dtype=_NANOSECONDS_DTYPE).astype("<i8")
    )
    digest.update(labels.to_numpy(dtype="<f8").tobytes())
    return digest.hexdigest()


def get_feature_matrix_sha256(matrices: FeatureMatrices) -> str:
    """Hash the feature names and the values of the train and query matrices.

    Numeric columns hash as little-endian float64, with every NaN the same.
    Other columns hash as their strings, with missing values as ``null``.
    """
    digest = hashlib.sha256()
    digest.update(json.dumps(list(matrices.feature_names)).encode())
    for frame in (matrices.train, matrices.query):
        digest.update(len(frame).to_bytes(8, "little"))
        for name in matrices.feature_names:
            digest.update(_get_column_bytes(frame[name]))
    return digest.hexdigest()


def _get_entity_bytes(entities: pd.Series) -> bytes:
    """Return the bytes of the entity IDs: int64 for integers, else strings."""
    if pd.api.types.infer_dtype(entities, skipna=False) == "integer":
        return b"int64" + entities.to_numpy(dtype="<i8").tobytes()
    text = entities.map(str).tolist()
    return b"str" + json.dumps(text, ensure_ascii=False).encode()


def _get_column_bytes(column: pd.Series) -> bytes:
    """Return the bytes of a feature column: float64 for numbers, else strings."""
    if pd.api.types.is_bool_dtype(column) or pd.api.types.is_numeric_dtype(column):
        values = column.to_numpy(dtype="<f8", na_value=np.nan)
        values = np.where(np.isnan(values), np.nan, values)
        return b"num" + values.astype("<f8").tobytes()
    text = [None if pd.isna(value) else str(value) for value in column]
    return b"str" + json.dumps(text, ensure_ascii=False).encode()


__all__ = ["get_context_rows_sha256", "get_feature_matrix_sha256"]
