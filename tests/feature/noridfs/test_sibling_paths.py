"""Sibling paths: features from a parent's other time-stamped rows, before each cutoff.

Every feature is checked against a direct loop over the same rows, on random data with
repeated timestamps, so the sorted as-of executor must agree with the plain definition.
"""

import numpy as np
import pandas as pd
import pytest

from nori_rel.feature.dfs.noridfs import (
    AnchorDefinition,
    DatabaseView,
    NativeFeatureConfig,
    NativeRelationalFeaturizer,
    SiblingPathConfig,
)
from nori_rel.feature.dfs.noridfs.planner import (
    get_sibling_match_columns,
    get_sibling_paths,
)

HOUR = pd.Timedelta(hours=1)
DAY_NS = 86_400_000_000_000


def get_page_view_database(seed: int = 0, views: int = 400) -> DatabaseView:
    """Return devices, drugs and pharmacies with time-stamped views pointing at them."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-07-01", tz="UTC")
    # Whole minutes on a short calendar give many repeated timestamps per device.
    minutes = rng.integers(0, 14 * 24 * 60, size=views)
    frame = pd.DataFrame(
        {
            "view_id": np.arange(1, views + 1),
            "device_id": rng.integers(1, 25, size=views),
            "drug_id": rng.integers(1, 6, size=views),
            "pharmacy_id": rng.integers(1, 4, size=views),
            "view_time": start + pd.to_timedelta(minutes // 7 * 7, unit="min"),
            "price": rng.normal(10, 3, size=views).round(2),
        }
    )
    return DatabaseView.from_parts(
        tables={
            "devices": pd.DataFrame({"device_id": np.arange(1, 25)}),
            "drugs": pd.DataFrame({"drug_id": np.arange(1, 6)}),
            "pharmacies": pd.DataFrame({"pharmacy_id": np.arange(1, 4)}),
            "views": frame,
        },
        primary_keys={
            "devices": "device_id",
            "drugs": "drug_id",
            "pharmacies": "pharmacy_id",
            "views": "view_id",
        },
        foreign_keys=[
            ("views", "device_id", "devices", "device_id"),
            ("views", "drug_id", "drugs", "drug_id"),
            ("views", "pharmacy_id", "pharmacies", "pharmacy_id"),
        ],
        time_columns={"views": "view_time"},
    )


def get_expected_device_features(
    views: pd.DataFrame, view_id: int, cutoff: pd.Timestamp
) -> dict[str, float]:
    """Return one view's device features by looping over the device's earlier views."""
    row = views.set_index("view_id").loc[view_id]
    earlier = views.loc[
        (views["device_id"] == row["device_id"]) & (views["view_time"] < cutoff)
    ]
    expected = {
        "count|all": float(len(earlier)),
        "recency": (cutoff - earlier["view_time"].max()) / pd.Timedelta(days=1)
        if len(earlier)
        else np.nan,
        "oldest_age": (cutoff - earlier["view_time"].min()) / pd.Timedelta(days=1)
        if len(earlier)
        else np.nan,
        "distinct|drug_id": float(earlier["drug_id"].nunique()),
        "distinct|pharmacy_id": float(earlier["pharmacy_id"].nunique()),
        "distinct|view_time": float(earlier["view_time"].nunique()),
    }
    for hours in (1, 6, 24, 168):
        expected[f"count|{hours}h"] = float(
            (earlier["view_time"] >= cutoff - hours * HOUR).sum()
        )
    previous_at = earlier["view_time"].max() if len(earlier) else None
    for column in ("drug_id", "pharmacy_id"):
        same = earlier.loc[earlier[column] == row[column]]
        expected[f"match_count|{column}"] = float(len(same))
        expected[f"match_recency|{column}"] = (
            (cutoff - same["view_time"].max()) / pd.Timedelta(days=1)
            if len(same)
            else np.nan
        )
        expected[f"match_previous|{column}"] = float(
            len(same) > 0 and same["view_time"].max() == previous_at
        )
    return expected


def get_device_columns(frame: pd.DataFrame) -> dict[str, str]:
    """Return {short name: column} for the device-to-views sibling features."""
    prefix = (
        "nr|path|views%7CS%3Aviews.device_id-%3Edevices.device_id%7Cviews.device_id|"
    )
    columns = {}
    for column in frame.columns:
        if not column.startswith(prefix):
            continue
        source, operation, window = column[len(prefix) :].split("|")
        if operation == "count":
            hours = (
                "all"
                if window == "all"
                else f"{int(window[:-2]) // 3_600_000_000_000}h"
            )
            columns[f"count|{hours}"] = column
        elif operation in {"recency", "oldest_age"}:
            columns[operation] = column
        else:
            columns[f"{operation}|{source}"] = column
    return columns


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_event_time_sibling_features_match_a_direct_loop(seed):
    database = get_page_view_database(seed)
    views = database.tables["views"]
    anchors = AnchorDefinition.create(
        entity_table="views", entity_column="view_id", cutoff_column="cutoff"
    )
    featurizer = NativeRelationalFeaturizer.fit(
        database,
        anchors,
        NativeFeatureConfig(max_features=256),
        sibling_paths=SiblingPathConfig(),
    )
    # Event-time rows: each view anchored at its own time.
    rows = views[["view_id"]].assign(cutoff=views["view_time"])
    frame = featurizer.transform(database, rows).frame
    columns = get_device_columns(frame)
    assert set(columns) == {
        "count|all",
        "count|1h",
        "count|6h",
        "count|24h",
        "count|168h",
        "recency",
        "oldest_age",
        "distinct|drug_id",
        "distinct|pharmacy_id",
        "distinct|view_time",
        "match_count|drug_id",
        "match_recency|drug_id",
        "match_previous|drug_id",
        "match_count|pharmacy_id",
        "match_recency|pharmacy_id",
        "match_previous|pharmacy_id",
    }
    for position, view_id in enumerate(views["view_id"]):
        cutoff = views.loc[views["view_id"] == view_id, "view_time"].iloc[0]
        expected = get_expected_device_features(views, view_id, cutoff)
        for name, column in columns.items():
            assert frame[column].iloc[position] == pytest.approx(
                expected[name], nan_ok=True, abs=1e-9
            ), (view_id, name)


def test_a_shared_cutoff_counts_every_row_before_it():
    database = get_page_view_database(3)
    views = database.tables["views"]
    anchors = AnchorDefinition.create(
        entity_table="views", entity_column="view_id", cutoff_column="cutoff"
    )
    featurizer = NativeRelationalFeaturizer.fit(
        database, anchors, NativeFeatureConfig(), sibling_paths=SiblingPathConfig()
    )
    cutoff = pd.Timestamp("2026-07-08", tz="UTC")
    frame = featurizer.transform(
        database, views[["view_id"]].assign(cutoff=cutoff)
    ).frame
    counts = frame[get_device_columns(frame)["count|all"]]
    expected = views["device_id"].map(
        views.loc[views["view_time"] < cutoff, "device_id"].value_counts()
    )
    assert counts.tolist() == expected.fillna(0).astype(float).tolist()


def test_sibling_paths_are_opt_in_and_listed_same_table_first():
    database = get_page_view_database()
    anchors = AnchorDefinition.create(
        entity_table="views", entity_column="view_id", cutoff_column="cutoff"
    )
    plain = NativeRelationalFeaturizer.fit(database, anchors, NativeFeatureConfig())
    assert not any(path.is_sibling for path in plain.plan.paths)
    assert not any("%7CS%3A" in name for name in plain.plan.columns)
    paths = get_sibling_paths(database, anchors, SiblingPathConfig())
    assert [path.edges[0].parent_table for path in paths] == [
        "devices",
        "drugs",
        "pharmacies",
    ]
    assert get_sibling_match_columns(database, paths[0]) == [
        ("drug_id", "drug_id"),
        ("pharmacy_id", "pharmacy_id"),
    ]
    assert (
        len(get_sibling_paths(database, anchors, SiblingPathConfig(max_paths=1))) == 1
    )
    only_devices = get_sibling_paths(
        database, anchors, SiblingPathConfig(parent_tables=("devices",))
    )
    assert [path.edges[0].parent_table for path in only_devices] == ["devices"]


def test_static_anchors_plan_no_sibling_paths():
    database = get_page_view_database()
    anchors = AnchorDefinition.create(
        entity_table="views", entity_column="view_id", cutoff_column=None
    )
    plan = NativeRelationalFeaturizer.fit(
        database, anchors, NativeFeatureConfig(), sibling_paths=SiblingPathConfig()
    ).plan
    assert not any(path.is_sibling for path in plan.paths)


def test_sibling_config_validation():
    with pytest.raises(ValueError, match="max_paths"):
        SiblingPathConfig(max_paths=0)
    with pytest.raises(ValueError, match="positive"):
        SiblingPathConfig(window_ns=(0,))
    with pytest.raises(ValueError, match="unique"):
        SiblingPathConfig(window_ns=(DAY_NS, DAY_NS))
    with pytest.raises(ValueError, match="parent_tables"):
        SiblingPathConfig(parent_tables=())


def test_a_sibling_plan_survives_the_pipeline_record():
    from nori_rel.pipeline.features import get_plan_from_record, get_plan_record

    database = get_page_view_database()
    anchors = AnchorDefinition.create(
        entity_table="views", entity_column="view_id", cutoff_column="cutoff"
    )
    plan = NativeRelationalFeaturizer.fit(
        database, anchors, NativeFeatureConfig(), sibling_paths=SiblingPathConfig()
    ).plan
    assert any(path.is_sibling for path in plan.paths)
    restored = get_plan_from_record(get_plan_record(plan))
    assert restored.digest == plan.digest
    views = database.tables["views"]
    rows = views[["view_id"]].assign(cutoff=views["view_time"])
    assert (
        NativeRelationalFeaturizer(restored)
        .transform(database, rows)
        .frame.equals(NativeRelationalFeaturizer(plan).transform(database, rows).frame)
    )


@pytest.mark.parametrize("seed", [0, 1])
def test_numeric_sibling_aggregates_match_a_direct_loop(seed):
    database = get_page_view_database(seed)
    views = database.tables["views"].copy()
    # Some prices are missing: sums skip them and means divide by the rest.
    views.loc[views["view_id"] % 7 == 0, "price"] = np.nan
    database = DatabaseView.from_parts(
        tables={**database.tables, "views": views},
        primary_keys=database.primary_keys,
        foreign_keys=[
            (fk.child_table, fk.child_column, fk.parent_table, fk.parent_column)
            for fk in database.foreign_keys
        ],
        time_columns=database.time_columns,
    )
    anchors = AnchorDefinition.create(
        entity_table="views", entity_column="view_id", cutoff_column="cutoff"
    )
    config = SiblingPathConfig(
        parent_tables=("devices",), include_numeric_aggregates=True
    )
    featurizer = NativeRelationalFeaturizer.fit(
        database, anchors, NativeFeatureConfig(max_features=256), sibling_paths=config
    )
    frame = featurizer.transform(
        database, views[["view_id"]].assign(cutoff=views["view_time"])
    ).frame
    prefix = (
        "nr|path|views%7CS%3Aviews.device_id-%3Edevices.device_id%7Cviews.device_id"
        "|price|"
    )
    windows = [f"{window}ns" for window in config.window_ns]
    for position, row in enumerate(views.itertuples(index=False)):
        earlier = views.loc[
            (views["device_id"] == row.device_id) & (views["view_time"] < row.view_time)
        ]
        prices = earlier["price"].dropna()
        expected = {
            "sum|all": prices.sum(),
            "mean|all": prices.mean() if len(prices) else np.nan,
            "max|all": prices.max() if len(prices) else np.nan,
            "min|all": prices.min() if len(prices) else np.nan,
        }
        for window in windows:
            recent = earlier.loc[
                earlier["view_time"] >= row.view_time - pd.Timedelta(window)
            ]
            recent_prices = recent["price"].dropna()
            expected[f"sum|{window}"] = recent_prices.sum()
            expected[f"mean|{window}"] = (
                recent_prices.mean() if len(recent_prices) else np.nan
            )
        for name, value in expected.items():
            assert frame[prefix + name].iloc[position] == pytest.approx(
                value, nan_ok=True, abs=1e-9
            ), (row.view_id, name)


def test_numeric_sibling_aggregates_are_opt_in():
    database = get_page_view_database()
    anchors = AnchorDefinition.create(
        entity_table="views", entity_column="view_id", cutoff_column="cutoff"
    )
    plan = NativeRelationalFeaturizer.fit(
        database,
        anchors,
        NativeFeatureConfig(max_features=256),
        sibling_paths=SiblingPathConfig(),
    ).plan
    assert not any(
        spec.path is not None
        and spec.path.is_sibling
        and spec.operation in {"sum", "mean", "max", "min"}
        for spec in plan.features
    )


def test_numeric_columns_limit_the_aggregated_columns():
    database = get_page_view_database()
    anchors = AnchorDefinition.create(
        entity_table="views", entity_column="view_id", cutoff_column="cutoff"
    )

    def get_numeric_sources(config: SiblingPathConfig) -> set[str]:
        """Return the columns the plan sums or averages over sibling rows."""
        plan = NativeRelationalFeaturizer.fit(
            database,
            anchors,
            NativeFeatureConfig(max_features=512),
            sibling_paths=config,
        ).plan
        return {
            spec.column
            for spec in plan.features
            if spec.path is not None
            and spec.path.is_sibling
            and spec.operation in {"sum", "mean"}
        }

    assert get_numeric_sources(SiblingPathConfig(include_numeric_aggregates=True)) == {
        "price"
    }
    assert (
        get_numeric_sources(
            SiblingPathConfig(include_numeric_aggregates=True, numeric_columns=())
        )
        == set()
    )


def get_database_with_drug_classes(seed: int = 0) -> DatabaseView:
    """Return the page-view database with each drug in one of two drug classes."""
    database = get_page_view_database(seed)
    tables = dict(database.tables)
    tables["drugs"] = tables["drugs"].assign(drug_class_id=[1, 1, 2, 2, 1])
    tables["drug_classes"] = pd.DataFrame({"drug_class_id": [1, 2]})
    return DatabaseView.from_parts(
        tables=tables,
        primary_keys={**database.primary_keys, "drug_classes": "drug_class_id"},
        foreign_keys=[
            (fk.child_table, fk.child_column, fk.parent_table, fk.parent_column)
            for fk in database.foreign_keys
        ]
        + [("drugs", "drug_class_id", "drug_classes", "drug_class_id")],
        time_columns=database.time_columns,
    )


@pytest.mark.parametrize("seed", [0, 1])
def test_siblings_grouped_by_a_parent_attribute_match_a_direct_loop(seed):
    database = get_database_with_drug_classes(seed)
    views = database.tables["views"]
    drug_class = database.tables["drugs"].set_index("drug_id")["drug_class_id"]
    anchors = AnchorDefinition.create(
        entity_table="views", entity_column="view_id", cutoff_column="cutoff"
    )
    config = SiblingPathConfig(
        parent_tables=("drug_classes",), include_numeric_aggregates=True
    )
    featurizer = NativeRelationalFeaturizer.fit(
        database, anchors, NativeFeatureConfig(max_features=256), sibling_paths=config
    )
    grouped = [path for path in featurizer.plan.paths if path.is_sibling]
    assert [path.identity for path in grouped] == [
        "views|S:views.drug_id->drugs.drug_id/G:drugs.drug_class_id->"
        "drug_classes.drug_class_id|views.drug_id"
    ]
    frame = featurizer.transform(
        database, views[["view_id"]].assign(cutoff=views["view_time"])
    ).frame
    prefix = (
        "nr|path|views%7CS%3Aviews.drug_id-%3Edrugs.drug_id%2FG%3Adrugs.drug_class_id"
        "-%3Edrug_classes.drug_class_id%7Cviews.drug_id|"
    )
    day = f"{24 * 3_600_000_000_000}ns"
    view_class = views["drug_id"].map(drug_class)
    for position, row in enumerate(views.itertuples(index=False)):
        earlier = views.loc[
            (view_class == drug_class[row.drug_id])
            & (views["view_time"] < row.view_time)
        ]
        recent = earlier.loc[earlier["view_time"] >= row.view_time - pd.Timedelta(day)]
        expected = {
            "rows|count|all": float(len(earlier)),
            f"rows|count|{day}": float(len(recent)),
            "price|sum|all": earlier["price"].sum(),
            f"price|mean|{day}": recent["price"].mean() if len(recent) else np.nan,
            "rows|recency|all": (row.view_time - earlier["view_time"].max())
            / pd.Timedelta(days=1)
            if len(earlier)
            else np.nan,
        }
        for name, value in expected.items():
            assert frame[prefix + name].iloc[position] == pytest.approx(
                value, nan_ok=True, abs=1e-9
            ), (row.view_id, name)


def test_grouped_sibling_paths_are_planned_only_for_named_tables():
    database = get_database_with_drug_classes()
    anchors = AnchorDefinition.create(
        entity_table="views", entity_column="view_id", cutoff_column="cutoff"
    )
    every_parent = get_sibling_paths(database, anchors, SiblingPathConfig())
    assert all(len(path.edges) == 2 for path in every_parent)
    # The entity references drugs directly, so drugs is not a grouping table.
    named = get_sibling_paths(
        database, anchors, SiblingPathConfig(parent_tables=("drugs", "drug_classes"))
    )
    assert sorted(len(path.edges) for path in named) == [2, 3]


def test_a_grouped_sibling_plan_survives_the_pipeline_record():
    from nori_rel.pipeline.features import get_plan_from_record, get_plan_record

    database = get_database_with_drug_classes()
    anchors = AnchorDefinition.create(
        entity_table="views", entity_column="view_id", cutoff_column="cutoff"
    )
    plan = NativeRelationalFeaturizer.fit(
        database,
        anchors,
        NativeFeatureConfig(),
        sibling_paths=SiblingPathConfig(parent_tables=("drug_classes",)),
    ).plan
    assert any(len(path.edges) == 3 for path in plan.paths)
    restored = get_plan_from_record(get_plan_record(plan))
    assert restored.digest == plan.digest


def test_excluded_paths_leave_out_parent_and_child_pairs():
    database = get_database_with_drug_classes()
    anchors = AnchorDefinition.create(
        entity_table="views", entity_column="view_id", cutoff_column="cutoff"
    )
    config = SiblingPathConfig(
        parent_tables=("devices", "drug_classes"),
        excluded_paths=(("drug_classes", "views"),),
    )
    paths = get_sibling_paths(database, anchors, config)
    assert [path.edges[0].parent_table for path in paths] == ["devices"]


def test_max_features_caps_the_sibling_family():
    database = get_page_view_database()
    anchors = AnchorDefinition.create(
        entity_table="views", entity_column="view_id", cutoff_column="cutoff"
    )

    def get_sibling_count(config: SiblingPathConfig) -> int:
        """Return how many planned features come from sibling paths."""
        plan = NativeRelationalFeaturizer.fit(
            database,
            anchors,
            NativeFeatureConfig(max_features=512),
            sibling_paths=config,
        ).plan
        return sum(
            spec.path is not None and spec.path.is_sibling for spec in plan.features
        )

    assert get_sibling_count(SiblingPathConfig()) > 10
    assert get_sibling_count(SiblingPathConfig(max_features=10)) == 10
    with pytest.raises(ValueError, match="max_features"):
        SiblingPathConfig(max_features=0)
