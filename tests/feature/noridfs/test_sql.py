"""Each NoriDFS feature's SQL gives the pandas executor's values on DuckDB."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nori_rel._pql_runtime import get_feature_sql_by_name
from nori_rel.feature.agentic import AgenticFeatures, FeatureProgram, FeatureQuery
from nori_rel.feature.agentic import contracts as agentic_contracts
from nori_rel.feature.dfs import CUTOFF_COLUMN as ROWS_CUTOFF_COLUMN
from nori_rel.feature.dfs.noridfs import (
    AnchorDefinition,
    DatabaseView,
    FeaturePlan,
    NativeFeatureConfig,
    NativeRelationalFeaturizer,
    NoriDFS,
)
from nori_rel.feature.dfs.noridfs.sql import (
    CUTOFF_COLUMN,
    EVAL_TABLE,
    ROW_COLUMN,
    get_features_from_sql,
    get_plan_sql,
    get_sql_schema,
)

_START = pd.Timestamp("2024-01-01", tz="UTC")
#: Category values whose Python repr order differs from plain string order:
#: "'ab cd'" sorts before "'ab'", and a value holding a quote is written
#: with double quotes, which sort before every single-quoted value. The last
#: two are spelled like the missing category and its escape.
_LABELS = [
    "ab",
    "ab cd",
    "it's",
    'say "hi"',
    "z",
    "__nori_rel_missing__",
    "__nori_rel_escape__:z",
]
_HORIZON = "2 days"
#: Features computed without arithmetic, which must match exactly.
_EXACT_OPERATIONS = {"count", "min", "max", "last", "lookup", "direct"}
_EXACT_ANCHORS = {"cutoff_year", "cutoff_month", "cutoff_day", "cutoff_day_of_week"}
#: The executor's std subtracts prefix sums of squares, so where the true
#: spread is zero it returns noise of about sqrt(eps * mean square), around
#: 1e-6 here. SQL's STDDEV_SAMP returns the exact zero.
_STD_TOLERANCE = {"rtol": 1e-6, "atol": 1e-4}
_TOLERANCE = {"rtol": 1e-9, "atol": 1e-9}


def _duckdb() -> None:
    """Skip unless DuckDB, from the agentic extra, can run the SQL."""
    pytest.importorskip("duckdb")


def _times(rng: np.random.Generator, size: int, *, missing: float) -> pd.Series:
    """Return UTC times on a coarse grid, so equal times are common."""
    offsets = pd.to_timedelta(rng.integers(0, 30, size), unit="D") + pd.to_timedelta(
        rng.choice([0, 6, 12], size), unit="h"
    )
    times = pd.Series(_START + offsets)
    times[rng.random(size) < missing] = pd.NaT
    return times


def _labels(rng: np.random.Generator, size: int) -> pd.Series:
    return pd.Series(rng.choice(np.array([*_LABELS, None], dtype=object), size))


def _keys(rng: np.random.Generator, size: int, valid: int) -> pd.Series:
    """Return foreign keys with some missing and some reaching no parent."""
    choices = np.array([*range(1, valid + 1), 999, None], dtype=object)
    return pd.Series(pd.array(rng.choice(choices, size), dtype="Int64"))


def _random_database(seed: int, *, timed_entity: bool) -> DatabaseView:
    """Return a database with every path NoriDFS plans and messy values.

    users is the entity. It points to plans, which point to regions (parent
    lookups). orders, and their items, are its descendants; visits link it
    to stores (a bridge); events has no primary key.
    """
    rng = np.random.default_rng(seed)
    users = pd.DataFrame(
        {
            "user_id": np.arange(1, 26),
            "age": np.where(
                rng.random(25) < 0.2, np.nan, rng.integers(18, 70, 25).astype(float)
            ),
            "tier": _labels(rng, 25),
            "vip": pd.Series(
                rng.choice(np.array([True, False, None], dtype=object), 25)
            ),
            "joined": _times(rng, 25, missing=0.1),
            "plan_id": _keys(rng, 25, 4),
        }
    )
    if timed_entity:
        users["created_at"] = _START + pd.to_timedelta(rng.integers(-5, 20, 25), "D")
        users.loc[rng.random(25) < 0.1, "created_at"] = pd.NaT
    plans = pd.DataFrame(
        {
            "plan_id": np.arange(1, 5),
            "launched_at": _times(rng, 4, missing=0.2),
            "fee": rng.normal(10, 3, 4),
            "name": _labels(rng, 4),
            "region_id": _keys(rng, 4, 3),
        }
    )
    regions = pd.DataFrame(
        {
            "region_id": np.arange(1, 4),
            "region_name": _labels(rng, 3),
            "area": rng.normal(100, 10, 3),
        }
    )
    orders = pd.DataFrame(
        {
            "order_id": np.arange(1, 121),
            "user_id": _keys(rng, 120, 25),
            "placed_at": _times(rng, 120, missing=0.1),
            "amount": np.where(rng.random(120) < 0.15, np.nan, rng.normal(50, 20, 120)),
            "status": _labels(rng, 120),
            "express": pd.Series(
                rng.choice(np.array([True, False, None], dtype=object), 120)
            ),
        }
    )
    items = pd.DataFrame(
        {
            "item_id": np.arange(1, 201),
            "order_id": _keys(rng, 200, 120),
            "shipped_at": _times(rng, 200, missing=0.1),
            "price": rng.normal(5, 2, 200),
            "kind": _labels(rng, 200),
        }
    )
    stores = pd.DataFrame(
        {
            "store_id": np.arange(1, 7),
            "opened_at": _times(rng, 6, missing=0.2),
            "size": rng.normal(300, 50, 6),
            "store_kind": _labels(rng, 6),
        }
    )
    visits = pd.DataFrame(
        {
            "user_id": _keys(rng, 90, 25),
            "store_id": _keys(rng, 90, 6),
            "visited_at": _times(rng, 90, missing=0.1),
        }
    )
    events = pd.DataFrame(
        {
            "user_id": _keys(rng, 80, 25),
            "at": _times(rng, 80, missing=0.1),
            "value": np.where(rng.random(80) < 0.15, np.nan, rng.normal(0, 1, 80)),
            "label": _labels(rng, 80),
        }
    )
    time_columns = {
        "plans": "launched_at",
        "orders": "placed_at",
        "items": "shipped_at",
        "stores": "opened_at",
        "visits": "visited_at",
        "events": "at",
    }
    if timed_entity:
        time_columns["users"] = "created_at"
    return DatabaseView.from_parts(
        tables={
            "users": users,
            "plans": plans,
            "regions": regions,
            "orders": orders,
            "items": items,
            "stores": stores,
            "visits": visits,
            "events": events,
        },
        primary_keys={
            "users": "user_id",
            "plans": "plan_id",
            "regions": "region_id",
            "orders": "order_id",
            "items": "item_id",
            "stores": "store_id",
        },
        foreign_keys=[
            ("users", "plan_id", "plans", "plan_id"),
            ("plans", "region_id", "regions", "region_id"),
            ("orders", "user_id", "users", "user_id"),
            ("items", "order_id", "orders", "order_id"),
            ("visits", "user_id", "users", "user_id"),
            ("visits", "store_id", "stores", "store_id"),
            ("events", "user_id", "users", "user_id"),
        ],
        time_columns=time_columns,
    )


def _random_anchors(
    seed: int, database: DatabaseView, *, temporal: bool = True
) -> pd.DataFrame:
    """Return anchors for known and unknown users, several per user.

    Cutoffs fall on event times and on window starts, so rows exactly at a
    cutoff or a window bound are common.
    """
    rng = np.random.default_rng(seed + 1000)
    users = database.tables["users"]["user_id"].to_numpy()
    keys = rng.choice(np.append(users, 404), 90)
    if not temporal:
        return pd.DataFrame({"user": keys})
    event_times = pd.concat(
        [database.tables["orders"]["placed_at"], database.tables["events"]["at"]]
    ).dropna()
    shifts = pd.to_timedelta(rng.choice([0, 2, 8], 90), unit="D")
    cutoffs = pd.Series(rng.choice(event_times.to_numpy(), 90)) + shifts
    return pd.DataFrame({"user": keys, "cutoff": pd.to_datetime(cutoffs, utc=True)})


def _config(*, missing: bool) -> NativeFeatureConfig:
    """Return a budget wide enough to plan every candidate feature."""
    return NativeFeatureConfig(
        max_depth=2,
        max_paths=64,
        max_features=10_000,
        max_direct_features=100,
        window_multipliers=(1, 4),
        include_missing_in_modes=missing,
        include_missing_fractions=missing,
    )


def _definition(
    *, temporal: bool = True, entity_table: str = "users"
) -> AnchorDefinition:
    return AnchorDefinition.create(
        entity_table=entity_table,
        entity_column="user",
        cutoff_column="cutoff" if temporal else None,
        horizon=_HORIZON if temporal else None,
    )


def _canonical(value: object) -> object:
    """Return a value as both sides spell it: missing is None, bools as text.

    A mode that counts missing values over a boolean column returns text in
    SQL (``'true'``), since one SQL column cannot hold booleans and the
    missing category together.
    """
    if value is None or (np.ndim(value) == 0 and pd.isna(value)):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return str(value).lower()
    return value


def _assert_same_features(
    plan: FeaturePlan, expected: pd.DataFrame, actual: pd.DataFrame
) -> None:
    assert list(actual.columns) == list(expected.columns)
    for feature in plan.features:
        name = feature.name
        if feature.kind == "categorical":
            assert [_canonical(value) for value in actual[name]] == [
                _canonical(value) for value in expected[name]
            ], name
            continue
        actual_values = pd.to_numeric(actual[name]).to_numpy(dtype=float)
        expected_values = pd.to_numeric(expected[name]).to_numpy(dtype=float)
        exact = (
            feature.operation in _EXACT_OPERATIONS and feature.kind == "numeric"
        ) or feature.column in _EXACT_ANCHORS
        if exact:
            np.testing.assert_array_equal(actual_values, expected_values, err_msg=name)
        else:
            tolerance = _STD_TOLERANCE if feature.operation == "std" else _TOLERANCE
            np.testing.assert_allclose(
                actual_values,
                expected_values,
                equal_nan=True,
                err_msg=name,
                **tolerance,
            )


def _plan_and_features(
    database: DatabaseView,
    anchors: pd.DataFrame,
    definition: AnchorDefinition,
    config: NativeFeatureConfig,
) -> tuple[FeaturePlan, pd.DataFrame, pd.DataFrame]:
    featurizer = NativeRelationalFeaturizer.fit(database, definition, config)
    expected = featurizer.transform(database, anchors).frame
    actual = get_features_from_sql(database, featurizer.plan, anchors)
    return featurizer.plan, expected, actual


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("timed_entity", [False, True])
@pytest.mark.parametrize("missing", [False, True])
def test_every_feature_sql_gives_the_executor_values(
    seed: int, timed_entity: bool, missing: bool
) -> None:
    _duckdb()
    database = _random_database(seed, timed_entity=timed_entity)
    plan, expected, actual = _plan_and_features(
        database,
        _random_anchors(seed, database),
        _definition(),
        _config(missing=missing),
    )
    planned = {
        (feature.operation, None if feature.path is None else feature.path.direction)
        for feature in plan.features
    }
    required = {("anchor", None), ("direct", None), ("lookup", "parent")}
    for direction in ("descendant", "bridge"):
        required.update(
            (operation, direction)
            for operation in (
                "count",
                "recency",
                "mean_age",
                "oldest_age",
                "mean",
                "std",
                "min",
                "max",
                "sum",
                "last",
                "mode",
            )
        )
        if missing:
            required.add(("missing_fraction", direction))
    assert required <= planned
    assert any(feature.window_ns is not None for feature in plan.features)
    assert any(len(path.edges) == 2 and path.is_parent for path in plan.paths)
    assert any(
        len(path.edges) == 2 and path.direction == "descendant" for path in plan.paths
    )
    _assert_same_features(plan, expected, actual)


@pytest.mark.parametrize("seed", range(3))
def test_static_plan_sql_gives_the_executor_values(seed: int) -> None:
    _duckdb()
    database = _random_database(seed, timed_entity=True)
    plan, expected, actual = _plan_and_features(
        database,
        _random_anchors(seed, database, temporal=False),
        _definition(temporal=False),
        _config(missing=True),
    )
    assert {"last", "mode", "lookup", "sum"} <= {f.operation for f in plan.features}
    _assert_same_features(plan, expected, actual)


def test_unusual_names_are_quoted_and_still_match() -> None:
    _duckdb()
    times = pd.to_datetime(
        ["2024-01-01", "2024-01-02", "2024-01-02", None, "2024-01-05"], utc=True
    )
    database = DatabaseView.from_parts(
        tables={
            "select": pd.DataFrame(
                {"User.Id": [1, 2, 3], 'Name "Display"': ["a", "b", None]}
            ),
            "Order Lines": pd.DataFrame(
                {
                    "line id": [1, 2, 3, 4, 5],
                    "User.Id": [1, 1, 2, 2, 3],
                    "order": times,
                    "Amount 'USD'": [1.0, 2.0, np.nan, 4.0, 5.0],
                    "group": ["x", "y", "x", None, "y"],
                }
            ),
        },
        primary_keys={"select": "User.Id", "Order Lines": "line id"},
        foreign_keys=[("Order Lines", "User.Id", "select", "User.Id")],
        time_columns={"Order Lines": "order"},
    )
    anchors = pd.DataFrame(
        {
            "user": [1, 2, 3, 1],
            "cutoff": pd.to_datetime(
                ["2024-01-03", "2024-01-06", "2024-01-06", "2024-01-02"], utc=True
            ),
        }
    )
    plan, expected, actual = _plan_and_features(
        database, anchors, _definition(entity_table="select"), _config(missing=False)
    )
    assert {"direct", "count", "last", "mode"} <= {f.operation for f in plan.features}
    _assert_same_features(plan, expected, actual)


def test_a_bridge_through_the_entity_table_names_it_twice() -> None:
    """A post's answers are posts too: the bridge joins posts to itself."""
    _duckdb()
    rng = np.random.default_rng(7)
    posts = pd.DataFrame(
        {
            "Id": np.arange(1, 41),
            "ParentId": pd.array(
                [None] * 10 + list(rng.integers(1, 11, 30)), dtype="Int64"
            ),
            "OwnerUserId": pd.array(rng.integers(1, 6, 40), dtype="Int64"),
            "CreationDate": _times(rng, 40, missing=0.1),
            "Score": rng.integers(-2, 10, 40).astype(float),
        }
    )
    users = pd.DataFrame(
        {
            "Id": np.arange(1, 6),
            "CreationDate": _times(rng, 5, missing=0.0),
            "Reputation": rng.integers(1, 500, 5).astype(float),
        }
    )
    database = DatabaseView.from_parts(
        tables={"posts": posts, "users": users},
        primary_keys={"posts": "Id", "users": "Id"},
        foreign_keys=[
            ("posts", "ParentId", "posts", "Id"),
            ("posts", "OwnerUserId", "users", "Id"),
        ],
        time_columns={"posts": "CreationDate", "users": "CreationDate"},
    )
    anchors = pd.DataFrame(
        {
            "post": rng.integers(1, 12, 40),
            "cutoff": _START + pd.to_timedelta(rng.integers(5, 35, 40), unit="D"),
        }
    )
    definition = AnchorDefinition.create(
        entity_table="posts",
        entity_column="post",
        cutoff_column="cutoff",
        horizon=_HORIZON,
    )
    plan, expected, actual = _plan_and_features(
        database, anchors, definition, _config(missing=False)
    )
    bridges = [path for path in plan.paths if path.is_bridge]
    assert any(path.edges[0].child_table == "posts" for path in bridges)
    _assert_same_features(plan, expected, actual)


def test_a_text_key_joins_a_number_key_as_text() -> None:
    """The executor compares a text key with an integer key as strings."""
    _duckdb()
    database = DatabaseView.from_parts(
        tables={
            "users": pd.DataFrame({"user_id": ["1", "2"], "age": [30.0, 40.0]}),
            "orders": pd.DataFrame(
                {
                    "order_id": [1, 2, 3],
                    "user_id": [1, 1, 2],
                    "at": pd.to_datetime(
                        ["2024-01-01", "2024-01-02", "2024-01-03"], utc=True
                    ),
                    "amount": [1.0, 2.0, 3.0],
                }
            ),
        },
        primary_keys={"users": "user_id", "orders": "order_id"},
        foreign_keys=[("orders", "user_id", "users", "user_id")],
        time_columns={"orders": "at"},
    )
    anchors = pd.DataFrame(
        {
            "user": ["1", "2"],
            "cutoff": pd.to_datetime(["2024-01-05", "2024-01-05"], utc=True),
        }
    )
    plan, expected, actual = _plan_and_features(
        database, anchors, _definition(), _config(missing=False)
    )
    count = next(f.name for f in plan.features if f.operation == "count")
    assert expected[count].tolist() == [2.0, 1.0]
    schema = get_sql_schema(database, plan, anchors["user"])
    assert (
        'JOIN "orders" ON CAST("orders"."user_id" AS VARCHAR) '
        '= CAST("users"."user_id" AS VARCHAR)'
    ) in get_plan_sql(plan, schema)[count]
    _assert_same_features(plan, expected, actual)


def _orders_database() -> DatabaseView:
    return DatabaseView.from_parts(
        tables={
            "users": pd.DataFrame({"user_id": [1, 2], "plan_id": [10, 20]}),
            "plans": pd.DataFrame({"plan_id": [10, 20], "fee": [5.0, 7.0]}),
            "orders": pd.DataFrame(
                {
                    "order_id": [1, 2, 3],
                    "user_id": [1, 1, 2],
                    "placed_at": pd.to_datetime(
                        ["2024-01-01", "2024-01-02", "2024-01-03"], utc=True
                    ),
                    "amount": [1.0, 2.0, 3.0],
                }
            ),
        },
        primary_keys={"users": "user_id", "plans": "plan_id", "orders": "order_id"},
        foreign_keys=[
            ("users", "plan_id", "plans", "plan_id"),
            ("orders", "user_id", "users", "user_id"),
        ],
        time_columns={"orders": "placed_at"},
    )


def _feature_sql(operation: str, column: str | None, window: bool = False) -> str:
    database = _orders_database()
    plan = NativeRelationalFeaturizer.fit(
        database, _definition(), _config(missing=False)
    ).plan
    feature = next(
        f
        for f in plan.features
        if f.operation == operation
        and f.column == column
        and (f.window_ns is not None) == window
    )
    anchors = pd.Series([1], name="user")
    return get_plan_sql(plan, get_sql_schema(database, plan, anchors))[feature.name]


def test_a_mean_reads_the_entity_rows_before_each_cutoff() -> None:
    assert _feature_sql("mean", "amount", window=True) == "\n".join(
        [
            "WITH path_rows AS (",
            "    SELECT",
            '        "users"."user_id" AS entity_key,',
            '        "orders"."placed_at" AS row_time,',
            '        "orders"."amount" AS value',
            '    FROM "users"',
            '    JOIN "orders" ON "orders"."user_id" = "users"."user_id"',
            '    WHERE "orders"."placed_at" IS NOT NULL',
            ")",
            "SELECT",
            "    e.row_id,",
            "    AVG(p.value) AS feature",
            "FROM eval_table AS e",
            "LEFT JOIN path_rows AS p",
            '    ON p.entity_key = e."user"',
            "    AND p.row_time < e.cutoff",
            "    AND p.row_time >= e.cutoff - INTERVAL '2 days'",
            "GROUP BY e.row_id",
            "ORDER BY e.row_id",
        ]
    )


def test_a_parent_lookup_reads_the_linked_row() -> None:
    assert _feature_sql("lookup", "fee") == "\n".join(
        [
            "SELECT",
            "    e.row_id,",
            '    "plans"."fee" AS feature',
            "FROM eval_table AS e",
            'LEFT JOIN "users" ON "users"."user_id" = e."user"',
            'LEFT JOIN "plans" ON "plans"."plan_id" = "users"."plan_id"',
            "ORDER BY e.row_id",
        ]
    )


def test_sql_names_the_anchor_table_as_the_agent_search_does() -> None:
    assert EVAL_TABLE == agentic_contracts.EVAL_TABLE
    assert ROW_COLUMN == agentic_contracts.ROW_COLUMN
    assert CUTOFF_COLUMN == agentic_contracts.CUTOFF_ALIAS


@pytest.mark.xfail(
    strict=True,
    reason=(
        "the executor computes std from prefix sums of squares, which cancel "
        "when values are large next to their spread"
    ),
)
def test_std_of_large_values_with_a_small_spread_matches_the_sql() -> None:
    _duckdb()
    database = DatabaseView.from_parts(
        tables={
            "users": pd.DataFrame({"user_id": [1]}),
            "orders": pd.DataFrame(
                {
                    "order_id": [1, 2, 3, 4],
                    "user_id": [1, 1, 1, 1],
                    "placed_at": pd.to_datetime(
                        ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
                        utc=True,
                    ),
                    "amount": [1e8 + 0.1, 1e8 + 0.2, 1e8 + 0.3, 1e8 + 0.4],
                }
            ),
        },
        primary_keys={"users": "user_id", "orders": "order_id"},
        foreign_keys=[("orders", "user_id", "users", "user_id")],
        time_columns={"orders": "placed_at"},
    )
    anchors = pd.DataFrame(
        {"user": [1], "cutoff": pd.to_datetime(["2024-02-01"], utc=True)}
    )
    plan, expected, actual = _plan_and_features(
        database, anchors, _definition(), _config(missing=False)
    )
    std = next(
        f.name for f in plan.features if f.operation == "std" and f.window_ns is None
    )
    np.testing.assert_allclose(actual[std], [0.1290994], rtol=1e-6)
    np.testing.assert_allclose(expected[std], actual[std], rtol=1e-6)


def _snapshot() -> SimpleNamespace:
    database = _orders_database()
    return SimpleNamespace(
        tables=dict(database.tables),
        primary_keys={table: [key] for table, key in database.primary_keys.items()},
        foreign_keys=[
            (edge.child_table, edge.child_column, edge.parent_table, edge.parent_column)
            for edge in database.foreign_keys
        ],
        time_columns=dict(database.time_columns),
    )


def _compute(backend: NoriDFS) -> list[str]:
    rows = pd.DataFrame(
        {
            "user_id": [1, 2],
            ROWS_CUTOFF_COLUMN: pd.to_datetime(["2024-01-05", "2024-01-05"], utc=True),
            "row_id": ["context:1", "query:1"],
        }
    )
    matrices = backend.compute(
        _snapshot(), rows.iloc[:1], rows.iloc[1:], "users", row_id_column="row_id"
    )
    return matrices.feature_names


def test_noridfs_reports_sql_for_every_feature_it_computed() -> None:
    backend = NoriDFS(horizon=_HORIZON)
    assert backend.last_feature_sql == {}
    names = _compute(backend)
    assert list(backend.last_feature_sql) == names
    assert all(
        f"FROM {EVAL_TABLE} AS e" in sql for sql in backend.last_feature_sql.values()
    )


def test_prediction_details_get_noridfs_sql_also_from_an_agent_search() -> None:
    backend = NoriDFS(horizon=_HORIZON)
    names = _compute(backend)
    assert get_feature_sql_by_name(backend, names) == backend.last_feature_sql
    program = FeatureProgram(
        (FeatureQuery("history", "SELECT row_id, 1 AS one FROM eval_table"),)
    )
    search = AgenticFeatures(program=program, fallback=backend)
    kept = [*names[1:], "history__one"]
    assert get_feature_sql_by_name(search, kept) == {
        name: backend.last_feature_sql[name] for name in names[1:]
    }
    assert get_feature_sql_by_name(AgenticFeatures(program=program), kept) == {}
