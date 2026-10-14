from __future__ import annotations

import os
from datetime import UTC, datetime

import psycopg
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import InternalError

from nori_rel.connectors.postgres import PostgreSQLConnector
from nori_rel.contracts import DatabaseCreateRequest, EnvironmentCredential
from nori_rel.limits import RelationalLimits

pytestmark = pytest.mark.skipif(
    "NORI_REL_TEST_DATABASE_URL" not in os.environ,
    reason="NORI_REL_TEST_DATABASE_URL is not configured",
)


@pytest.fixture(scope="module", autouse=True)
def managed_database() -> None:
    if os.getenv("NORI_REL_TEST_MANAGED") != "1":
        return
    raw_url = os.environ["NORI_REL_TEST_DATABASE_URL"]
    url = make_url(raw_url)
    if url.host not in {"127.0.0.1", "localhost"} or not (
        url.database or ""
    ).startswith("nori_rel_test"):
        raise RuntimeError("managed integration tests require a local test database")
    with psycopg.connect(raw_url) as connection, connection.cursor() as cursor:
        cursor.execute("DROP TABLE IF EXISTS results")
        cursor.execute("DROP TABLE IF EXISTS drivers")
        cursor.execute(
            "CREATE TABLE drivers (driver_id integer PRIMARY KEY, name text NOT NULL)"
        )
        cursor.execute(
            "CREATE TABLE results ("
            "result_id integer PRIMARY KEY, "
            "driver_id integer NOT NULL REFERENCES drivers(driver_id), "
            "race_date timestamptz NOT NULL, position double precision NOT NULL)"
        )
        cursor.execute("INSERT INTO drivers VALUES (1, 'Ada'), (2, 'Lin')")
        cursor.execute(
            "INSERT INTO results VALUES "
            "(1, 1, '2026-01-08T00:00:00Z', 2.0), "
            "(2, 2, '2026-01-10T00:00:00Z', 4.0), "
            "(3, 1, '2026-01-11T00:00:00Z', 1.0)"
        )


def _connector() -> PostgreSQLConnector:
    source = DatabaseCreateRequest(
        engine="postgresql",
        credential=EnvironmentCredential(variable="NORI_REL_TEST_DATABASE_URL"),
        tables=["drivers", "results"],
        time_columns={"results": "race_date"},
    )
    return PostgreSQLConnector(
        source,
        RelationalLimits(max_rows_per_table=100, max_total_rows=200),
    )


def test_live_postgres_discovery_snapshot_and_cutoff() -> None:
    connector = _connector()

    assert connector.test() >= 0
    schema = connector.discover("db-integration")
    assert [table.name for table in schema.tables] == ["drivers", "results"]
    results = next(table for table in schema.tables if table.name == "results")
    assert results.primary_key == ["result_id"]
    assert results.foreign_keys[0].referred_table == "drivers"
    assert results.time_column == "race_date"

    snapshot = connector.snapshot("db-integration", datetime(2026, 1, 10, tzinfo=UTC))
    assert snapshot.tables["drivers"]["driver_id"].tolist() == [1, 2]
    assert snapshot.tables["results"]["result_id"].tolist() == [1, 2]
    assert snapshot.foreign_keys == [("results", "driver_id", "drivers", "driver_id")]


def test_every_connector_session_is_database_read_only() -> None:
    connector = _connector()
    engine = connector._engine()
    try:
        with engine.connect() as connection:
            assert (
                connection.execute(
                    text("SHOW default_transaction_read_only")
                ).scalar_one()
                == "on"
            )
            with pytest.raises(InternalError, match="read-only transaction"):
                connection.execute(text("CREATE TABLE forbidden_write (id int)"))
    finally:
        engine.dispose()


def test_transaction_read_only_is_enforced_inside_the_snapshot() -> None:
    raw_url = make_url(os.environ["NORI_REL_TEST_DATABASE_URL"]).set(
        drivername="postgresql+psycopg"
    )
    engine = create_engine(raw_url)
    try:
        with (
            pytest.raises(InternalError, match="read-only transaction"),
            PostgreSQLConnector._read_only_transaction(engine) as connection,
        ):
            assert (
                connection.execute(text("SHOW transaction_read_only")).scalar_one()
                == "on"
            )
            connection.execute(text("CREATE TABLE forbidden_fallback (id int)"))
    finally:
        engine.dispose()


def test_snapshot_source_tables_use_server_side_cursors():
    from sqlalchemy import event

    connector = _connector()
    source_cursors = []

    def record_cursor(connection, cursor, statement, parameters, context, many):
        if " LIMIT " in " ".join(statement.split()).upper():
            source_cursors.append(cursor.name)

    engine = connector._engine()
    event.listen(engine, "after_cursor_execute", record_cursor)
    try:
        connector.snapshot("db-integration", datetime(2026, 1, 10, tzinfo=UTC))
        assert len(source_cursors) == 2
        assert all(source_cursors)
    finally:
        connector.close()
