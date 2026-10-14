"""Verify live connector context rows, exact labels and transaction guarantees."""

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from sqlalchemy import text

from nori_rel.connectors.postgres import PostgreSQLConnector, SnapshotLimitError
from nori_rel.contracts import DatabaseCreateRequest, EnvironmentCredential
from nori_rel.limits import RelationalLimits
from nori_rel.pql.sql_expressions import namespace_sql

JAN = datetime(2026, 1, 1, tzinfo=UTC)
FEB = datetime(2026, 2, 1, tzinfo=UTC)


def prepare(connector, target="COUNT(orders.*,0,30,days)", **options):
    return connector.prepare_context(
        "test",
        f"PREDICT {target} FOR users.user_id=500 WHERE users.country='US'",
        anchor_times=[JAN],
        anchor_time=FEB,
        observed_through=FEB,
        entity_created_at="created_at",
        **options,
    )


def test_parameter_namespacing_preserves_identifier_escapes():
    sql, params = namespace_sql(
        '"%%(value_0)s" = %(value_0)s AND "%%%s" IS NULL',
        {"value_0": "US"},
        "context",
    )
    assert sql == '"%%(value_0)s" = %(context_value_0)s AND "%%%s" IS NULL'
    assert params == {"context_value_0": "US"}


@pytest.mark.parametrize(
    "target,expected,dropped",
    [
        ("COUNT(orders.*,0,30,days)", [(1, 7, JAN, 2), (2, 42, JAN, 0)], 0),
        (
            "SUM(orders.amount,0,30,days)",
            [(1, 7, JAN, Decimal("3.75")), (2, 42, JAN, 0)],
            0,
        ),
        ("AVG(orders.amount,0,30,days)", [(1, 7, JAN, Decimal("1.875"))], 1),
        # A zero denominator yields an unknown label; the row is excluded and
        # reported in dropped_unknown_labels rather than silently vanishing.
        (
            "SUM(orders.amount,0,30,days)/COUNT(orders.*,0,30,days)",
            [(1, 7, JAN, Decimal("1.875"))],
            1,
        ),
        (
            "COUNT(orders.* WHERE orders.channel='web',0,30,days)=0",
            [(1, 7, JAN, False), (2, 42, JAN, True)],
            0,
        ),
    ],
)
def test_pql_connector_returns_exact_rows_and_counts(source, target, expected, dropped):
    connector, _ = source
    frame, counts = prepare(connector, target)
    assert list(frame.columns) == [
        "context_row_id",
        "entity_id",
        "anchor_time",
        "label",
    ]
    assert list(frame.itertuples(index=False, name=None)) == expected
    assert counts == {
        "candidate_rows": 2,
        "usable_rows": len(expected),
        "dropped_unknown_labels": dropped,
    }


def test_zero_false_and_large_integers_are_not_lost(source):
    connector, db = source
    db.execute("UPDATE orders SET amount=NULL")
    frame, counts = prepare(connector, "SUM(orders.amount,0,30,days)")
    assert list(frame.itertuples(index=False, name=None)) == [(2, 42, JAN, 0)]
    assert counts == {
        "candidate_rows": 2,
        "usable_rows": 1,
        "dropped_unknown_labels": 1,
    }
    db.execute("UPDATE orders SET amount=9007199254740993")
    frame, _ = prepare(connector, "AVG(orders.amount,0,30,days)")
    assert frame.iloc[0]["label"] == Decimal("9007199254740993")


def test_empty_and_all_unknown_context_retain_columns(source):
    connector, db = source
    frame, counts = prepare(connector, "AVG(orders.amount,0,30,days)")
    assert counts["usable_rows"] == 1
    db.execute("UPDATE orders SET amount=NULL")
    frame, counts = prepare(connector, "AVG(orders.amount,0,30,days)")
    assert frame.empty and counts == {
        "candidate_rows": 2,
        "usable_rows": 0,
        "dropped_unknown_labels": 2,
    }
    db.execute("UPDATE users SET country='CA'")
    frame, counts = prepare(connector)
    assert frame.empty and list(frame.columns) == [
        "context_row_id",
        "entity_id",
        "anchor_time",
        "label",
    ]
    assert counts == {
        "candidate_rows": 0,
        "usable_rows": 0,
        "dropped_unknown_labels": 0,
    }


def test_bound_before_labels_and_no_implicit_refill(source):
    connector, _ = source
    frame, counts = prepare(
        connector,
        "AVG(orders.amount WHERE orders.channel='missing',0,30,days)",
        max_context_rows=1,
    )
    assert frame.empty
    assert counts == {
        "candidate_rows": 1,
        "usable_rows": 0,
        "dropped_unknown_labels": 1,
    }


def test_read_only_utc_snapshot_survives_concurrent_commit(source, monkeypatch):
    connector, db = source
    discover = connector._discover

    def concurrent_update(connection, database_id):
        schema = discover(connection, database_id)
        for setting, expected in [
            ("transaction_read_only", "on"),
            ("transaction_isolation", "repeatable read"),
            ("TimeZone", "UTC"),
            ("statement_timeout", "5min"),
        ]:
            assert connection.execute(text(f"SHOW {setting}")).scalar_one() == expected
        db.execute("INSERT INTO orders VALUES (4,42,'2026-01-05',10,'web')")
        return schema

    monkeypatch.setattr(connector, "_discover", concurrent_update)
    frame, _ = prepare(connector)
    assert frame["label"].tolist() == [2, 0]
    monkeypatch.setattr(connector, "_discover", discover)
    frame, _ = prepare(connector)
    assert frame["label"].tolist() == [2, 1]


def test_memory_limit_fails_without_leaving_open_transaction(source):
    connector, db = source
    connector.settings = RelationalLimits(max_snapshot_bytes=1)
    with pytest.raises(SnapshotLimitError, match="resource limits"):
        prepare(connector)
    assert (
        db.execute(
            "SELECT COUNT(*) FROM pg_stat_activity "
            "WHERE application_name='nori-rel' AND state='idle in transaction'"
        ).fetchone()[0]
        == 0
    )


@pytest.mark.parametrize("limit", [0, -1, True, 60_001])
def test_invalid_row_budget_fails_before_connection(monkeypatch, limit):
    connector = PostgreSQLConnector(
        DatabaseCreateRequest(
            engine="postgresql", credential=EnvironmentCredential(variable="UNUSED")
        ),
        RelationalLimits(),
    )
    monkeypatch.setattr(connector, "_engine", lambda: pytest.fail("must not connect"))
    with pytest.raises(SnapshotLimitError, match="max_context_rows"):
        prepare(connector, max_context_rows=limit)
