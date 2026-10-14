"""FastDFS reports the column types it inferred and replays them."""

from types import SimpleNamespace

import pandas as pd
import pytest

from nori_rel.feature.dfs import CUTOFF_COLUMN

pytest.importorskip("fastdfs")

from nori_rel.feature.dfs.fastdfs import FastDFS


def _snapshot(statuses: list[str]) -> SimpleNamespace:
    users = pd.DataFrame({"id": [1, 2], "age": [40.0, 30.0]})
    orders = pd.DataFrame(
        {
            "id": range(len(statuses)),
            "user_id": [1, 2] * (len(statuses) // 2),
            "status": statuses,
            "at": pd.to_datetime(["2026-01-01"] * len(statuses), utc=True),
        }
    )
    return SimpleNamespace(
        tables={"users": users, "orders": orders},
        primary_keys={"users": ["id"], "orders": ["id"]},
        foreign_keys=[("orders", "user_id", "users", "id")],
        time_columns={"orders": "at"},
    )


def _rows(prefix: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2],
            CUTOFF_COLUMN: pd.to_datetime(["2026-02-01"] * 2, utc=True),
            "row_id": [f"{prefix}:1", f"{prefix}:2"],
        }
    )


def _compute(engine: FastDFS, snapshot: SimpleNamespace) -> list[str]:
    result = engine.compute(
        snapshot, _rows("context"), _rows("query"), "users", row_id_column="row_id"
    )
    return result.feature_names


def test_fastdfs_reports_the_types_it_inferred() -> None:
    engine = FastDFS(max_depth=2)
    _compute(engine, _snapshot(["new", "paid"] * 2))

    assert engine.last_column_types == {
        "users": {"age": "float"},
        "orders": {"status": "category", "at": "datetime"},
    }


def test_saved_types_keep_the_features_when_the_data_changes_type() -> None:
    few = _snapshot(["new", "paid"] * 2)
    many = _snapshot([f"status{n}" for n in range(12)])
    first = FastDFS(max_depth=2)
    saved = _compute(first, few)

    assert first.last_column_types is not None
    assert _compute(FastDFS(max_depth=2), many) != saved
    replay = FastDFS(max_depth=2, type_hints=first.last_column_types)
    assert _compute(replay, many) == saved
