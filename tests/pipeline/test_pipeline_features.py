from dataclasses import dataclass, field

import pandas as pd
import pytest

from nori_rel.feature import CUTOFF_COLUMN
from nori_rel.feature.dfs.noridfs import NoriDFS
from nori_rel.feature.dfs.noridfs.contracts import NativeFeatureConfig
from nori_rel.pipeline import PipelineFeatureError, PipelineSchemaError
from nori_rel.pipeline.features import (
    get_installed_fastdfs_version,
    get_plan_from_record,
    get_plan_record,
    get_saved_feature_backend,
    get_table_dtypes,
    get_tables_with_saved_dtypes,
)
from nori_rel.pipeline.manifest import (
    FastDFSFeaturesRecord,
    FeaturePlanRecord,
    NoriDFSFeaturesRecord,
)

ROW_ID = "row"
CONFIG = NativeFeatureConfig(max_features=48, max_direct_features=1)


@dataclass(frozen=True)
class Tables:
    tables: dict[str, pd.DataFrame]
    primary_keys: dict = field(
        default_factory=lambda: {"users": ["user_id"], "orders": ["id"]}
    )
    foreign_keys: list = field(
        default_factory=lambda: [("orders", "buyer_id", "users", "user_id")]
    )
    time_columns: dict = field(default_factory=lambda: {"orders": "event_time"})


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


def _rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoffs = pd.to_datetime(["2025-12-01", "2026-01-01"], utc=True)
    train = pd.DataFrame(
        {
            "user_id": pd.array([7, 42, 99, 7, 42, 99], dtype="Int64"),
            CUTOFF_COLUMN: [cutoffs[0]] * 3 + [cutoffs[1]] * 3,
        }
    )
    train[ROW_ID] = [f"context:{i}" for i in range(len(train))]
    query = pd.DataFrame(
        {
            "user_id": pd.array([7, 99], dtype="Int64"),
            CUTOFF_COLUMN: pd.to_datetime(["2026-02-01"] * 2, utc=True),
        }
    )
    query[ROW_ID] = ["query:0", "query:1"]
    return train, query


def _compute(backend, tables):
    train, query = _rows()
    return backend.compute(Tables(tables), train, query, "users", row_id_column=ROW_ID)


def _fitted() -> tuple[NoriDFS, object]:
    backend = NoriDFS(CONFIG, horizon=pd.Timedelta(days=30))
    return backend, _compute(backend, _tables())


def test_plan_round_trips_through_json_and_reproduces_the_matrices() -> None:
    backend, fitted = _fitted()
    record = get_plan_record(backend.last_plan)
    restored = get_plan_from_record(
        FeaturePlanRecord.model_validate_json(record.model_dump_json())
    )

    replayed = _compute(NoriDFS.from_plan(restored), _tables())

    assert restored == backend.last_plan
    assert replayed.feature_names == fitted.feature_names
    pd.testing.assert_frame_equal(replayed.train, fitted.train)
    pd.testing.assert_frame_equal(replayed.query, fitted.query)


def test_saved_plan_is_not_planned_again_on_new_data() -> None:
    backend, fitted = _fitted()
    saved = NoriDFS.from_plan(backend.last_plan)
    changed = _tables(amount=[1.0, 1.0, 1.0, 1.0])
    changed["orders"]["channel"] = pd.array(["a", "b", "c", "d"], dtype="string")

    replayed = _compute(saved, changed)

    assert replayed.feature_names == fitted.feature_names
    assert saved.last_plan is backend.last_plan


def test_all_null_columns_and_empty_tables_take_their_saved_dtypes() -> None:
    backend, _ = _fitted()
    saved_dtypes = get_table_dtypes(_tables())
    tables = _tables()
    tables["orders"]["amount"] = pd.array([None] * 4, dtype="string")
    saved = NoriDFS.from_plan(backend.last_plan)

    with pytest.raises(ValueError, match="does not match the fitted feature plan"):
        _compute(saved, tables)
    aligned = get_tables_with_saved_dtypes(tables, saved_dtypes)

    assert str(aligned["orders"]["amount"].dtype) == "Float64"
    assert aligned["orders"]["amount"].isna().all()
    assert tuple(_compute(saved, aligned).feature_names) == backend.last_plan.columns

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


def test_a_plan_that_does_not_reproduce_its_digest_is_refused() -> None:
    backend, _ = _fitted()
    record = get_plan_record(backend.last_plan).model_copy(update={"digest": "0" * 32})

    with pytest.raises(PipelineFeatureError, match="does not reproduce its digest"):
        get_plan_from_record(record)


def test_saved_plan_refuses_another_entity_table() -> None:
    backend, _ = _fitted()
    train, query = _rows()
    orders_rows = train.rename(columns={"user_id": "id"})

    with pytest.raises(ValueError, match="another entity table"):
        NoriDFS.from_plan(backend.last_plan).compute(
            Tables(_tables()),
            orders_rows,
            query.rename(columns={"user_id": "id"}),
            "orders",
            row_id_column=ROW_ID,
        )


def test_saved_feature_backends_run_without_a_search() -> None:
    backend, fitted = _fitted()
    plan = get_plan_record(backend.last_plan)

    noridfs = get_saved_feature_backend(NoriDFSFeaturesRecord(plan=plan))
    fastdfs = get_saved_feature_backend(_fastdfs_record())

    assert noridfs.saved_plan == backend.last_plan
    assert _compute(noridfs, _tables()).feature_names == fitted.feature_names
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
