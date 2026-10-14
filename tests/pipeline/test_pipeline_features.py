import pandas as pd
import pytest

from nori_rel.pipeline import PipelineFeatureError, PipelineSchemaError
from nori_rel.pipeline.features import (
    get_installed_fastdfs_version,
    get_saved_feature_backend,
    get_table_dtypes,
    get_tables_with_saved_dtypes,
)
from nori_rel.pipeline.manifest import (
    FastDFSFeaturesRecord,
)


def _tables(amount: list | None = None) -> dict[str, pd.DataFrame]:
    users = pd.DataFrame(
        {
            "user_id": pd.array([7, 42, 99], dtype="Int64"),
            "created_at": pd.to_datetime(["2025-01-01"] * 3, utc=True),
            "country": pd.array(["US", "US", "CA"], dtype="string"),
        }
    )
    orders = pd.DataFrame(
        {
            "id": pd.array([1, 2, 3, 4], dtype="Int64"),
            "buyer_id": pd.array([7, 7, 99, 42], dtype="Int64"),
            "event_time": pd.to_datetime(
                ["2025-11-02", "2025-12-03", "2025-12-20", "2026-01-15"], utc=True
            ),
            "amount": pd.array(amount or [1.25, 2.5, 99.0, 4.0], dtype="Float64"),
            "channel": pd.array(["web", "store", "web", "web"], dtype="string"),
        }
    )
    return {"users": users, "orders": orders}


def test_all_null_columns_and_empty_tables_take_their_saved_dtypes() -> None:
    saved_dtypes = get_table_dtypes(_tables())
    tables = _tables()
    tables["orders"]["amount"] = pd.array([None] * 4, dtype="string")
    aligned = get_tables_with_saved_dtypes(tables, saved_dtypes)

    assert str(aligned["orders"]["amount"].dtype) == "Float64"
    assert aligned["orders"]["amount"].isna().all()

    empty = {
        name: pd.DataFrame(columns=frame.columns) for name, frame in tables.items()
    }
    restored = get_tables_with_saved_dtypes(empty, saved_dtypes)
    assert get_table_dtypes(restored) == saved_dtypes


def test_a_column_with_values_and_another_dtype_is_a_schema_change() -> None:
    tables = _tables()
    saved_dtypes = get_table_dtypes(tables)
    tables["orders"]["amount"] = pd.array(["1.25", None, None, None], dtype="string")

    with pytest.raises(PipelineSchemaError, match="'amount' of table 'orders'"):
        get_tables_with_saved_dtypes(tables, saved_dtypes)
    with pytest.raises(PipelineSchemaError, match="the pipeline read"):
        get_tables_with_saved_dtypes({"users": tables["users"]}, saved_dtypes)


def test_saved_feature_backends_run_without_a_search() -> None:
    fastdfs = get_saved_feature_backend(_fastdfs_record())

    assert (fastdfs.max_depth, fastdfs.max_features) == (2, 9)
    assert fastdfs.type_hints == {"orders": {"amount": "float"}}


def _fastdfs_record(version: str | None = None) -> FastDFSFeaturesRecord:
    return FastDFSFeaturesRecord(
        max_depth=2,
        max_features=9,
        fastdfs_version=version or get_installed_fastdfs_version(),
        column_types={"orders": {"amount": "float"}},
    )


def test_a_pipeline_saved_with_another_fastdfs_is_refused() -> None:
    with pytest.raises(PipelineFeatureError, match=r"saved with fastdfs 0\.1"):
        get_saved_feature_backend(_fastdfs_record("0.1"))


def test_a_misspelled_fastdfs_type_fails_on_load() -> None:
    with pytest.raises(ValueError, match="column_types"):
        FastDFSFeaturesRecord(
            max_depth=2,
            max_features=9,
            fastdfs_version="1.1",
            column_types={"orders": {"amount": "numeric"}},
        )
