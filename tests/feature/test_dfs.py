"""FastDFS alignment, input ownership and point-in-time behavior."""

from types import SimpleNamespace

import pandas as pd
import pytest

from nori_rel.feature.dfs import CUTOFF_COLUMN, BaseDFS, FeatureMatrices
from nori_rel.feature.dfs.fastdfs import FastDFS

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


@pytest.fixture
def engine() -> BaseDFS:
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


def test_fastdfs_keeps_automatic_categories() -> None:
    pytest.importorskip("fastdfs")
    fastdfs = pytest.importorskip("nori_rel.feature.dfs.fastdfs")
    assert fastdfs.FastDFS().categorical_columns() is None


def test_future_values_do_not_change_historical_rows(
    snapshot: SimpleNamespace, anchors: AnchorRows
) -> None:
    original = _compute(FastDFS(), snapshot, anchors)
    orders = snapshot.tables["orders"]
    orders.loc[orders["id"] == 4, "amount"] = -1000.0
    changed = _compute(FastDFS(), snapshot, anchors)
    pd.testing.assert_frame_equal(changed.train, original.train)
    assert not changed.query.equals(original.query)
