"""Prove snapshot consistency, row identity and past-only FastDFS inputs."""

from dataclasses import replace
from datetime import UTC, datetime

import pandas as pd
import pytest

from nori_rel._features import RelationalFeatureEngine
from nori_rel._pql_features import build_pql_features, build_prediction_rows

JAN = datetime(2026, 1, 1, tzinfo=UTC)
FEB = datetime(2026, 2, 1, tzinfo=UTC)
MAR = datetime(2026, 3, 3, tzinfo=UTC)
QUERY = (
    "PREDICT SUM(orders.amount,0,30,days) FOR users.user_id=99 WHERE users.country='US'"
)
OPTIONS = dict(
    anchor_times=[JAN],
    anchor_time=FEB,
    observed_through=FEB,
    entity_created_at="created_at",
)


def test_context_and_feature_tables_share_snapshot(source, monkeypatch):
    connector, db = source
    original = connector._snapshot

    def concurrent_write(connection, schema, as_of, **options):
        db.execute("INSERT INTO orders VALUES (4,42,'2026-01-05',123,'web')")
        return original(connection, schema, as_of, **options)

    monkeypatch.setattr(connector, "_snapshot", concurrent_write)
    snapshot, context, counts = connector.prepare_inputs("test", QUERY, **OPTIONS)
    assert context["label"].tolist() == [3.75, 0]
    assert snapshot.tables["orders"]["id"].tolist() == [1, 2, 3]
    assert counts["usable_rows"] == 2


def test_fastdfs_features_use_each_anchor_and_preserve_context_ids(source):
    connector, db = source
    db.execute("INSERT INTO orders VALUES (4,7,'2025-12-01',10,'web')")
    snapshot, context, _ = connector.prepare_inputs("test", QUERY, **OPTIONS)
    # Deliberately reverse ID order; features and labels must stay aligned.
    context = context.iloc[::-1].copy()
    engine = RelationalFeatureEngine(max_depth=2, max_features=32)
    scoring = build_prediction_rows(snapshot, QUERY, "created_at")
    matrices, scoring = build_pql_features(snapshot, context, QUERY, scoring, engine)
    assert matrices.train.index.tolist() == [2, 1]
    assert scoring["entity_id"].tolist() == [99]  # WHERE US only restricts context.
    assert matrices.query.index.tolist() == scoring["query_row_id"].tolist()
    assert list(matrices.train.columns) == list(matrices.query.columns)
    assert not any("nori_pql_row_id" in str(c) for c in matrices.feature_names)
    mean_cols = [
        c
        for c in matrices.feature_names
        if "MEAN" in c.upper() and "AMOUNT" in c.upper()
    ]
    assert mean_cols
    assert matrices.train.loc[1, mean_cols[0]] == 10
    changed = snapshot.tables["orders"].copy()
    changed.loc[changed["id"].isin([1, 2]), "amount"] = 999
    altered = replace(snapshot, tables={**snapshot.tables, "orders": changed})
    again, _ = build_pql_features(altered, context, QUERY, scoring, engine)
    pd.testing.assert_frame_equal(matrices.train, again.train)


def test_delayed_window_features_match_the_immediate_window(source):
    """The gap before a delayed interval is observed, adjacent, and off limits.

    A (30,60,days) target leaves 30 days between the anchor and its label
    interval. The snapshot holds that gap in full, so a feature builder that
    cut at anchor + start would find it. At scoring time the same 30 days lie
    in the future, so a model trained on them would lean on an input that is
    always absent in production. Features must stop at the anchor for both
    windows and come out identical.
    """
    connector, db = source
    db.execute("INSERT INTO orders VALUES (4,7,'2026-01-20',10,'web')")  # the gap
    delayed = QUERY.replace("0,30,days", "30,60,days")
    options = dict(OPTIONS, anchor_time=MAR, observed_through=MAR)
    engine = RelationalFeatureEngine(max_depth=2, max_features=32)
    runs = {}
    for query in (QUERY, delayed):
        snapshot, context, _ = connector.prepare_inputs("test", query, **options)
        scoring = build_prediction_rows(snapshot, query, "created_at")
        matrices, _ = build_pql_features(snapshot, context, query, scoring, engine)
        runs[query] = (context, matrices.train)
    (context, immediate), (later, delayed_train) = runs[QUERY], runs[delayed]
    assert context["context_row_id"].tolist() == later["context_row_id"].tolist()
    assert context["label"].tolist() == [13.75, 0]  # the gap order is a label here
    assert later["label"].tolist() == [0, 0]  # and outside the delayed interval
    pd.testing.assert_frame_equal(immediate, delayed_train)


def test_scoring_each_in_and_missing_ids(source):
    connector, _ = source
    snapshot, _, _ = connector.prepare_inputs("test", QUERY, **OPTIONS)
    assert build_prediction_rows(
        snapshot, QUERY.replace("=99", " IN (42,7)"), "created_at"
    )["entity_id"].tolist() == [7, 42]
    assert build_prediction_rows(
        snapshot,
        QUERY.replace("FOR users.user_id=99", "FOR EACH users.user_id"),
        "created_at",
    )["entity_id"].tolist() == [7, 42, 99, 500]
    with pytest.raises(RuntimeError, match="missing"):
        build_prediction_rows(snapshot, QUERY.replace("=99", "=1234"), "created_at")


@pytest.mark.parametrize("key", [7, 2**53 + 1])
def test_nullable_integer_keys_preserve_identity_and_features(source, key):
    connector, db = source
    db.execute("DELETE FROM orders")
    db.execute("DELETE FROM users")
    db.execute("INSERT INTO users VALUES (%s,'2025-01-01','US')", (key,))
    db.execute(
        "INSERT INTO orders VALUES (1,%s,'2025-12-01',12,'web'), "
        "(2,NULL,'2025-12-01',999,'web'), (3,%s,'2026-01-02',55,'web')",
        (key, key),
    )
    query = f"PREDICT SUM(orders.amount,0,30,days) FOR users.user_id={key}"
    snapshot, context, _ = connector.prepare_inputs("test", query, **OPTIONS)
    keys = snapshot.tables["orders"].set_index("id")["buyer_id"]
    assert keys[1] == key and keys[3] == key
    assert pd.isna(keys[2])
    assert context["entity_id"].tolist() == [key]
    assert context["label"].tolist() == [55]
    scoring = build_prediction_rows(snapshot, query, "created_at")
    matrices, _ = build_pql_features(
        snapshot, context, query, scoring, RelationalFeatureEngine(max_features=32)
    )
    assert scoring["entity_id"].tolist() == [key]
    assert matrices.train["d2__users.COUNT(orders)"].tolist() == [1]
    assert matrices.train["d2__users.MEAN(orders.amount)"].tolist() == [12]
    assert matrices.query["d2__users.COUNT(orders)"].tolist() == [2]
    assert matrices.query["d2__users.MEAN(orders.amount)"].tolist() == [33.5]


def _pipeline(connector, query):
    """Run prepare_inputs through FastDFS and return context, matrices, scoring."""
    snapshot, context, _ = connector.prepare_inputs("test", query, **OPTIONS)
    scoring = build_prediction_rows(snapshot, query, "created_at")
    matrices, _ = build_pql_features(
        snapshot, context, query, scoring, RelationalFeatureEngine(max_features=32)
    )
    return context, matrices, scoring


@pytest.mark.parametrize("kind", ["table", "key", "fk", "child", "time"])
def test_dotted_names_match_an_otherwise_identical_ordinary_schema(source, kind):
    connector, db = source
    context, matrices, scoring = _pipeline(connector, QUERY)
    query = QUERY
    if kind == "table":
        db.execute('ALTER TABLE users RENAME TO "User.Table"')
        connector.config.tables = ["User.Table", "orders"]
        query = query.replace("users.", "`User.Table`.")
    elif kind == "key":
        db.execute('ALTER TABLE users RENAME COLUMN user_id TO "user.id"')
        query = query.replace("users.user_id", "users.`user.id`")
    elif kind == "fk":
        db.execute('ALTER TABLE orders RENAME COLUMN buyer_id TO "buyer.id"')
    elif kind == "child":
        db.execute('ALTER TABLE orders RENAME TO "order.items"')
        connector.config.tables = ["users", "order.items"]
        connector.config.time_columns = {"order.items": "event_time"}
        query = query.replace("orders.", "`order.items`.")
    else:
        db.execute('ALTER TABLE orders RENAME COLUMN event_time TO "event.time"')
        connector.config.time_columns = {"orders": "event.time"}
    renamed, renamed_matrices, renamed_scoring = _pipeline(connector, query)

    assert renamed["entity_id"].tolist() == context["entity_id"].tolist()
    assert renamed["label"].tolist() == context["label"].tolist()
    assert renamed_scoring["entity_id"].tolist() == scoring["entity_id"].tolist()
    # Compare by position: only the names differ, and dtypes must survive too.
    for renamed_frame, frame in (
        (renamed_matrices.train, matrices.train),
        (renamed_matrices.query, matrices.query),
    ):
        assert renamed_frame.shape == frame.shape
        width = range(frame.shape[1])
        pd.testing.assert_frame_equal(
            renamed_frame.set_axis(width, axis=1), frame.set_axis(width, axis=1)
        )
    assert not any("_x" in name for name in renamed_matrices.feature_names)


def test_dotted_feature_column_reaches_features_under_its_original_name(source):
    connector, db = source
    db.execute('ALTER TABLE users ADD COLUMN "profile.score" integer')
    db.execute('UPDATE users SET "profile.score" = user_id')
    context, matrices, _ = _pipeline(connector, QUERY)
    dotted = [name for name in matrices.feature_names if "`profile.score`" in name]
    assert dotted == ["d1__users.`profile.score`"]
    # The restored column must hold the source values, not merely the name.
    assert (
        matrices.train[dotted[0]].tolist() == context["entity_id"].tolist() == [7, 42]
    )
    assert not any("_x" in name for name in matrices.feature_names)


def test_dotted_and_underscore_tables_both_survive_the_alias(source):
    connector, db = source
    db.execute('ALTER TABLE users RENAME TO "a.b"')
    db.execute("CREATE TABLE a_b (id bigint PRIMARY KEY, note text)")
    connector.config.tables = ["a.b", "a_b", "orders"]
    query = QUERY.replace("users.", "`a.b`.")
    context, matrices, _ = _pipeline(connector, query)
    assert context["label"].tolist() == [3.75, 0]
    assert any("`a.b`" in name for name in matrices.feature_names)
    assert not any("_x" in name for name in matrices.feature_names)


@pytest.mark.parametrize("name", ["__target_index__", "__time_for_cutoff__"])
def test_entity_key_named_after_a_fastdfs_internal_column(source, name):
    connector, db = source
    db.execute(f'ALTER TABLE users RENAME COLUMN user_id TO "{name}"')
    query = QUERY.replace("users.user_id", f"users.`{name}`")
    context, matrices, scoring = _pipeline(connector, query)
    assert context["label"].tolist() == [3.75, 0]
    assert scoring["entity_id"].tolist() == [99]
    # Context rows sit at Jan 1, before every order; user 99 is scored at Feb 1,
    # after its single January order.
    assert matrices.train["d2__users.COUNT(orders)"].tolist() == [0, 0]
    assert matrices.query["d2__users.COUNT(orders)"].tolist() == [1]


def test_uppercase_alias_lookalike_does_not_shadow_a_dotted_column(source):
    """DuckDB folds identifiers, so an alias must not be reachable by casing."""
    from nori_rel._names import _alias

    connector, db = source
    lookalike = _alias("a.b").upper()
    db.execute('ALTER TABLE orders ADD COLUMN "a.b" numeric')
    db.execute(f'ALTER TABLE orders ADD COLUMN "{lookalike}" numeric')
    db.execute(f'UPDATE orders SET "a.b" = 10, "{lookalike}" = 20')
    _, matrices, _ = _pipeline(connector, QUERY)

    dotted = matrices.query["d2__users.MEAN(orders.`a.b`)"].tolist()
    plain = matrices.query[f"d2__users.MEAN(orders.{lookalike})"].tolist()
    assert dotted == [10] and plain == [20]


def test_sql_only_preparation_still_supports_quoted_names(source):
    connector, db = source
    db.execute('ALTER TABLE users RENAME COLUMN user_id TO "user.id"')
    rows, _ = connector.prepare_context(
        "test", QUERY.replace("users.user_id", "users.`user.id`"), **OPTIONS
    )
    assert rows["label"].tolist() == [3.75, 0]


def test_scoring_filter_uses_same_snapshot_as_source_tables(source, monkeypatch):
    connector, db = source
    original = connector._snapshot

    def change_after_filter(connection, schema, as_of, **options):
        db.execute("UPDATE users SET country='US' WHERE user_id=99")
        return original(connection, schema, as_of, **options)

    monkeypatch.setattr(connector, "_snapshot", change_after_filter)
    snapshot, _, _ = connector.prepare_inputs(
        "test", QUERY, **OPTIONS, query_filter="users.country='CA'"
    )
    assert snapshot.scoring_filter_ids == (99,)
    assert snapshot.tables["users"].set_index("user_id").loc[99, "country"] == "CA"
    assert build_prediction_rows(snapshot, QUERY, "created_at")[
        "entity_id"
    ].tolist() == [99]


def test_scoring_filter_literal_is_bound(source):
    connector, db = source
    value = "x'); DROP TABLE users;--"
    db.execute("UPDATE users SET country=%s WHERE user_id=99", (value,))
    snapshot, _, _ = connector.prepare_inputs(
        "test", QUERY, **OPTIONS, query_filter='users.country="' + value + '"'
    )
    assert snapshot.scoring_filter_ids == (99,)
    assert len(snapshot.tables["users"]) == 4


def test_boolean_scoring_filter_and_null_exclusion(source):
    connector, db = source
    db.execute("ALTER TABLE users ADD COLUMN active boolean")
    db.execute("UPDATE users SET active=true WHERE user_id=99")
    db.execute("UPDATE users SET active=false WHERE user_id=7")
    snapshot, _, _ = connector.prepare_inputs(
        "test",
        QUERY,
        **OPTIONS,
        query_filter="users.active=true AND users.country IN ('CA','US')",
    )
    assert snapshot.scoring_filter_ids == (99,)


def test_scoring_filter_is_bounded_before_snapshot(source, monkeypatch):
    from nori_rel.connectors.postgres import SnapshotLimitError
    from nori_rel.limits import RelationalLimits

    connector, _ = source
    connector.settings = RelationalLimits(max_rows_per_table=1)

    def forbidden(*args, **kwargs):
        pytest.fail("filter row bound must fail before table loading")

    monkeypatch.setattr(connector, "_snapshot", forbidden)
    with pytest.raises(SnapshotLimitError, match="scoring filter"):
        connector.prepare_inputs(
            "test",
            QUERY,
            **OPTIONS,
            max_context_rows=1,
            query_filter="users.country IS NOT NULL",
        )
