import hashlib
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from nori_rel.feature import FeatureMatrices
from nori_rel.pipeline.audit import get_context_rows_sha256, get_feature_matrix_sha256

JAN = datetime(2026, 1, 1, tzinfo=UTC)


def _context(entities=(7, 42), anchors=(JAN, JAN)) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "context_row_id": range(1, len(entities) + 1),
            "entity_id": pd.Series(list(entities), dtype=object),
            "anchor_time": list(anchors),
        }
    )


def test_context_hash_follows_rows_labels_and_entity_types() -> None:
    labels = pd.Series([1.0, 2.0])
    base = get_context_rows_sha256(_context(), labels)

    assert base == get_context_rows_sha256(_context(), labels.copy())
    assert base != get_context_rows_sha256(_context(), pd.Series([1.0, 2.5]))
    assert base != get_context_rows_sha256(_context(entities=(42, 7)), labels)
    assert base != get_context_rows_sha256(_context(entities=("7", "42")), labels)
    as_other_zone = pd.Timestamp(JAN).tz_convert("America/New_York")
    assert base == get_context_rows_sha256(
        _context(anchors=(as_other_zone, as_other_zone)), labels
    )
    with pytest.raises(ValueError, match="align"):
        get_context_rows_sha256(_context(), pd.Series([1.0]))


def test_context_hash_is_fixed_by_its_byte_encoding() -> None:
    # The encoding is part of the audit record: a change here changes every
    # hash that earlier runs reported.
    expected = hashlib.sha256()
    expected.update((2).to_bytes(8, "little"))
    expected.update(b"int64" + np.array([7, 42], dtype="<i8").tobytes())
    january_ns = int(pd.Timestamp(JAN).value)
    expected.update(np.array([january_ns] * 2, dtype="<i8").tobytes())
    expected.update(np.array([1.0, 2.0], dtype="<f8").tobytes())

    assert get_context_rows_sha256(_context(), pd.Series([1.0, 2.0])) == (
        expected.hexdigest()
    )


def test_context_hash_of_a_large_context_is_fast() -> None:
    rows = 200_000
    context = pd.DataFrame(
        {
            "entity_id": np.arange(rows),
            "anchor_time": pd.Timestamp(JAN) + pd.to_timedelta(np.arange(rows), "s"),
        }
    )

    assert len(get_context_rows_sha256(context, pd.Series(np.arange(rows) / 7))) == 64


def test_feature_hash_follows_names_and_values() -> None:
    def matrices(train_a, names=("a", "b")) -> FeatureMatrices:
        return FeatureMatrices(
            train=pd.DataFrame({"a": train_a, "b": ["x", None]}),
            query=pd.DataFrame({"a": [9.0], "b": ["y"]}),
            feature_names=list(names),
        )

    base = get_feature_matrix_sha256(matrices([1.0, np.nan]))

    assert base == get_feature_matrix_sha256(matrices([1.0, float("nan")]))
    assert base != get_feature_matrix_sha256(matrices([1.0, 2.0]))
    assert base != get_feature_matrix_sha256(matrices([1.0, np.nan], names=("b", "a")))
