"""Check the shared matrix boundary against the unchanged native executor."""

from importlib.util import find_spec
from types import SimpleNamespace

import pandas as pd
import pytest

from nori_rel.feature.dfs import CUTOFF_COLUMN, BaseDFS, FeatureMatrices
from nori_rel.feature.dfs.noridfs import (
    AnchorDefinition,
    DatabaseView,
    NativeFeatureConfig,
    NativeRelationalFeaturizer,
    NoriDFS,
)

AnchorRows = tuple[pd.DataFrame, pd.DataFrame]


@pytest.fixture
def snapshot() -> SimpleNamespace:
    return SimpleNamespace(
        tables={
            "users": pd.DataFrame({"id": [1, 2], "age": [40.0, 30.0]}),
            "orders": pd.DataFrame(
                {
                    "id": [1, 2, 3, 4],
                    "user_id": [1, 1, 2, 2],
                    "amount": [4.0, 20.0, 7.0, 999.0],
                    "at": pd.to_datetime(
                        ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-05"],
                        utc=True,
                    ),
                }
            ),
        },
        primary_keys={"users": ["id"], "orders": ["id"]},
        foreign_keys=[("orders", "user_id", "users", "id")],
        time_columns={"orders": "at"},
    )


@pytest.fixture
def anchors() -> AnchorRows:
    train = pd.DataFrame(
        {
            "id": [2, 1],
            CUTOFF_COLUMN: pd.to_datetime(["2026-01-03", "2026-01-02"], utc=True),
            "row_id": ["context:2", "context:1"],
        },
        index=[25, 10],
    )
    query = pd.DataFrame(
        {
            "id": [2, 1],
            CUTOFF_COLUMN: pd.to_datetime(["2026-01-06", "2026-01-04"], utc=True),
            "row_id": ["query:2", "query:1"],
        },
        index=[30, 20],
    )
    return train, query


def _compute(
    engine: BaseDFS, snapshot: SimpleNamespace, anchors: AnchorRows
) -> FeatureMatrices:
    train, query = anchors
    return engine.compute(snapshot, train, query, "users", row_id_column="row_id")


@pytest.fixture(params=["noridfs", "fastdfs"])
def engine(request: pytest.FixtureRequest) -> BaseDFS:
    if request.param == "noridfs":
        return NoriDFS(NativeFeatureConfig(max_features=64), horizon="2 days")
    if (
        find_spec("fastdfs") is None
        or find_spec("nori_rel.feature.dfs.fastdfs") is None
    ):
        pytest.skip("The database FastDFS wrapper is optional and not exported")
    from nori_rel.feature.dfs.fastdfs import FastDFS

    return FastDFS(max_features=64)


def test_base_dfs_requires_an_implementation() -> None:
    with pytest.raises(TypeError, match="abstract"):
        BaseDFS()


def test_backends_share_matrix_shape_without_mutating_inputs(
    engine: BaseDFS, snapshot: SimpleNamespace, anchors: AnchorRows
) -> None:
    tables_before = {
        name: frame.copy(deep=True) for name, frame in snapshot.tables.items()
    }
    rows_before = tuple(frame.copy(deep=True) for frame in anchors)
    matrices = _compute(engine, snapshot, anchors)
    assert isinstance(engine, BaseDFS)
    assert isinstance(matrices, FeatureMatrices)
    assert len(matrices.train) == len(anchors[0])
    assert len(matrices.query) == len(anchors[1])
    assert list(matrices.train.columns) == list(matrices.query.columns)
    assert matrices.feature_names == list(matrices.train.columns)
    assert matrices.train.columns.is_unique
    assert not any("row_id" in name for name in matrices.feature_names)
    for name, frame in tables_before.items():
        pd.testing.assert_frame_equal(snapshot.tables[name], frame)
    for frame, before in zip(anchors, rows_before, strict=True):
        pd.testing.assert_frame_equal(frame, before)


def test_backends_keep_features_aligned_after_row_reordering(
    engine: BaseDFS, snapshot: SimpleNamespace, anchors: AnchorRows
) -> None:
    original = _compute(engine, snapshot, anchors)
    reordered = _compute(engine, snapshot, tuple(frame.iloc[::-1] for frame in anchors))
    for before, after in (
        (original.train, reordered.train),
        (original.query, reordered.query),
    ):
        pd.testing.assert_frame_equal(before.iloc[::-1].reset_index(drop=True), after)


def test_native_boundary_matches_direct_fit_transform(
    snapshot: SimpleNamespace, anchors: AnchorRows
) -> None:
    config = NativeFeatureConfig(max_features=64)
    engine = NoriDFS(config, horizon="2 days")
    matrices = _compute(engine, snapshot, anchors)
    database = DatabaseView.from_parts(**vars(snapshot))
    definition = AnchorDefinition.create(
        entity_table="users",
        entity_column="id",
        cutoff_column=CUTOFF_COLUMN,
        horizon="2 days",
    )
    fitted = NativeRelationalFeaturizer.fit(database, definition, config)
    expected = fitted.transform(database, pd.concat(anchors, ignore_index=True)).frame
    pd.testing.assert_frame_equal(
        pd.concat([matrices.train, matrices.query], ignore_index=True), expected
    )
    total = next(
        feature.name
        for feature in fitted.plan.features
        if feature.column == "amount"
        and feature.operation == "sum"
        and feature.window_ns is None
    )
    # Equal-to-cutoff events are excluded, and the original row order survives.
    assert pd.isna(matrices.train[total].iloc[0])
    assert matrices.train[total].iloc[1] == 4.0
    assert matrices.query[total].tolist() == [1006.0, 24.0]


def test_native_categories_come_from_the_plan_not_matrix_dtypes(
    snapshot: SimpleNamespace, anchors: AnchorRows, monkeypatch
) -> None:
    transform = NativeRelationalFeaturizer.transform

    def transform_with_numeric_categories(featurizer, database, rows):
        result = transform(featurizer, database, rows)
        for column in featurizer.plan.categorical_columns:
            result.frame[column] = pd.to_numeric(result.frame[column])
        return result

    monkeypatch.setattr(
        NativeRelationalFeaturizer, "transform", transform_with_numeric_categories
    )
    engine = NoriDFS()
    assert engine.categorical_columns() is None
    matrices = _compute(engine, snapshot, anchors)
    categories = engine.categorical_columns()
    assert categories == engine.last_plan.categorical_columns
    assert categories
    assert all(
        pd.api.types.is_numeric_dtype(matrices.train[name]) for name in categories
    )
    assert matrices.train[categories[0]].tolist() == [2, 1]


def test_fastdfs_keeps_automatic_categories() -> None:
    pytest.importorskip("fastdfs")
    fastdfs = pytest.importorskip("nori_rel.feature.dfs.fastdfs")
    assert fastdfs.FastDFS().categorical_columns() is None


def test_native_boundary_does_not_use_extra_anchor_columns(
    snapshot: SimpleNamespace, anchors: AnchorRows
) -> None:
    expected = _compute(NoriDFS(), snapshot, anchors)
    changed = tuple(frame.assign(label=999.0, unrelated="ignored") for frame in anchors)
    actual = _compute(NoriDFS(), snapshot, changed)
    pd.testing.assert_frame_equal(actual.train, expected.train)
    pd.testing.assert_frame_equal(actual.query, expected.query)


def test_future_values_do_not_change_historical_rows(
    snapshot: SimpleNamespace, anchors: AnchorRows
) -> None:
    original = _compute(NoriDFS(), snapshot, anchors)
    orders = snapshot.tables["orders"]
    orders.loc[orders["id"] == 4, "amount"] = -1000.0
    changed = _compute(NoriDFS(), snapshot, anchors)
    pd.testing.assert_frame_equal(changed.train, original.train)
    assert not changed.query.equals(original.query)


@pytest.mark.parametrize("empty_split", [0, 1])
def test_native_boundary_accepts_an_empty_split(
    snapshot: SimpleNamespace, anchors: AnchorRows, empty_split: int
) -> None:
    rows = tuple(
        frame.iloc[:0] if i == empty_split else frame for i, frame in enumerate(anchors)
    )
    matrices = _compute(NoriDFS(), snapshot, rows)
    assert len(matrices.train) == len(rows[0])
    assert len(matrices.query) == len(rows[1])
    assert matrices.train.columns.equals(matrices.query.columns)


@pytest.mark.parametrize("row_id", [None, "context:1"])
def test_native_boundary_rejects_null_or_duplicate_row_ids(
    snapshot: SimpleNamespace, anchors: AnchorRows, row_id: str | None
) -> None:
    train, query = anchors
    query.loc[query.index[0], "row_id"] = row_id
    with pytest.raises(ValueError, match="row identifiers"):
        _compute(NoriDFS(), snapshot, (train, query))


def test_native_boundary_rejects_missing_row_ids(
    snapshot: SimpleNamespace, anchors: AnchorRows
) -> None:
    rows = tuple(frame.drop(columns="row_id") for frame in anchors)
    with pytest.raises(ValueError, match="row identifiers"):
        _compute(NoriDFS(), snapshot, rows)


def test_native_boundary_rejects_composite_entity_keys(
    snapshot: SimpleNamespace, anchors: AnchorRows
) -> None:
    snapshot.primary_keys["users"] = ["id", "age"]
    with pytest.raises(ValueError, match="one primary-key column"):
        _compute(NoriDFS(), snapshot, anchors)
