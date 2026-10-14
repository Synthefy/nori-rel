"""Verify physical connection reuse, fresh snapshots, and pool cleanup."""

from datetime import UTC, datetime

import pytest
from sqlalchemy import event, text
from sqlalchemy.exc import DBAPIError

from nori_rel.connectors.postgres import ConnectorError

FEB = datetime(2026, 2, 1, tzinfo=UTC)


def prepare(connector):
    return connector.prepare_context(
        "test",
        "PREDICT COUNT(orders.*,0,30,days) FOR users.user_id=99 "
        "WHERE users.country='US'",
        anchor_times=[datetime(2026, 1, 1, tzinfo=UTC)],
        anchor_time=FEB,
        observed_through=FEB,
        entity_created_at="created_at",
    )


def test_pool_reuses_connection_but_reads_fresh_context(source):
    connector, db = source
    engine = connector._engine()
    backend_pids = []
    event.listen(
        engine,
        "checkout",
        lambda connection, *_: backend_pids.append(connection.info.backend_pid),
    )
    first, _ = prepare(connector)
    assert first["label"].tolist() == [2, 0]
    db.execute("INSERT INTO orders VALUES (4,7,'2026-01-05',10,'web')")
    second, _ = prepare(connector)
    assert second["label"].tolist() == [3, 0]
    assert len(backend_pids) == 2 and len(set(backend_pids)) == 1
    assert engine.pool.checkedout() == 0

    # A failed read-only transaction returns its connection without poisoning it.
    with (
        pytest.raises(DBAPIError),
        connector._read_only_transaction(engine) as connection,
    ):
        connection.execute(text("CREATE TABLE forbidden_pool_write (id int)"))
    third, _ = prepare(connector)
    assert third["label"].tolist() == [3, 0]
    assert len(set(backend_pids)) == 1
    connector.close()
    connector.close()
    with pytest.raises(ConnectorError, match="closed"):
        connector.test()


def test_timestamp_override_does_not_change_registered_source(source):
    connector, _ = source
    snapshot = connector.snapshot("test", FEB, time_columns={"users": "created_at"})
    assert snapshot.time_columns["users"] == "created_at"
    assert connector.config.time_columns == {"orders": "event_time"}
    assert "users" not in connector.snapshot("test", FEB).time_columns
