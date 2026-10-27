"""Opt-in warehouse fixtures with owned schemas and separate writer credentials."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest

from nori_rel import EnvironmentCredential, RelationalLimits
from nori_rel.contracts import DatabaseCreateRequest, ForeignKeySchema

JANUARY_ANCHOR = datetime(2026, 1, 1, tzinfo=UTC)
SCORING_CUTOFF = datetime(2026, 2, 1, tzinfo=UTC)
LARGE_ENTITY_ID = 9_007_199_254_740_993
WAREHOUSE_CONFIG_ENV = {
    "snowflake": "NORI_REL_TEST_SNOWFLAKE_CONFIG",
    "databricks": "NORI_REL_TEST_DATABRICKS_CONFIG",
}


def pytest_addoption(parser):
    parser.addoption(
        "--require-warehouse",
        action="store_true",
        help="Fail selected warehouse tests when their configuration is absent.",
    )


def _load_test_configuration(engine: str, request) -> dict[str, Any]:
    variable = WAREHOUSE_CONFIG_ENV[engine]
    value = os.environ.get(variable)
    if not value:
        message = f"{variable} is required for {engine} live integration tests"
        if request.config.getoption("--require-warehouse"):
            pytest.fail(message, pytrace=False)
        pytest.skip(message)
    try:
        configuration = json.loads(value)
        for key in ("reader_credential_env", "writer_credential_env"):
            credential_variable = configuration[key]
            if not os.environ.get(credential_variable):
                raise ValueError("missing referenced credential environment variable")
        if not isinstance(configuration["connection"], dict):
            raise ValueError("connection must be an object")
    except (ValueError, KeyError, TypeError):
        # Configuration may accidentally contain secrets: never echo its value.
        pytest.fail(
            f"Invalid {variable} or missing credential reference", pytrace=False
        )
    return configuration


def _connector_class(engine: str):
    if engine == "snowflake":
        from nori_rel.connectors.snowflake import SnowflakeConnector

        return SnowflakeConnector
    from nori_rel.connectors.databricks import DatabricksConnector

    return DatabricksConnector


def _connection_settings(engine: str, configuration: dict):
    from nori_rel.contracts import DatabricksConnection, SnowflakeConnection

    settings = SnowflakeConnection if engine == "snowflake" else DatabricksConnection
    return settings(**configuration)


@dataclass
class WarehouseSource:
    engine: str
    configuration: dict
    config: DatabaseCreateRequest
    connector: Any
    writer: Any

    def identifier(self, value: str) -> str:
        quote = "`" if self.engine in {"databricks", "mysql"} else '"'
        return quote + value.replace(quote, quote * 2) + quote

    def reference(self, table: str | None = None) -> str:
        if self.engine in {"postgresql", "mysql"}:
            parts = [self.config.schema_name]
            if table is not None:
                parts.append(table)
            return ".".join(self.identifier(part) for part in parts)
        settings = self.configuration["connection"]
        namespace = settings["database" if self.engine == "snowflake" else "catalog"]
        parts = [namespace, self.config.schema_name]
        if table is not None:
            parts.append(table)
        return ".".join(self.identifier(part) for part in parts)

    def execute_write(self, statement: str, parameters=None) -> None:
        with self.writer.cursor() as cursor:
            if parameters is None:
                cursor.execute(statement)
            else:
                cursor.execute(statement, parameters)

    def insert_rows(self, table: str, rows: list[tuple]) -> None:
        parameter = "%s" if self.engine in {"postgresql", "mysql"} else "?"
        placeholders = ",".join([parameter] * len(rows[0]))
        if self.engine == "snowflake":
            rows = [
                tuple(
                    ("TIMESTAMP_TZ", value)
                    if isinstance(value, datetime) and value.utcoffset() is not None
                    else value
                    for value in row
                )
                for row in rows
            ]
        with self.writer.cursor() as cursor:
            if self.engine == "databricks":
                # The native driver executes each executemany row separately.
                # These small fixtures fit one statement with native bindings.
                placeholder_groups = ",".join(f"({placeholders})" for _ in rows)
                bound_values = [value for row in rows for value in row]
                cursor.execute(
                    f"INSERT INTO {self.reference(table)} VALUES {placeholder_groups}",
                    bound_values,
                )
            else:
                cursor.executemany(
                    f"INSERT INTO {self.reference(table)} VALUES ({placeholders})", rows
                )

    def connect(self, client):
        return client.connect(
            engine=self.config.engine,
            connection=self.config.connection,
            credential=self.config.credential,
            schema_name=self.config.schema_name,
            tables=self.config.tables,
            time_columns=self.config.time_columns,
            primary_keys=self.config.primary_keys,
            foreign_keys=self.config.foreign_keys,
        )


def _populate_fixture(source: WarehouseSource) -> None:
    timestamp_type = {
        "snowflake": "TIMESTAMP_TZ",
        "databricks": "TIMESTAMP",
        "postgresql": "TIMESTAMPTZ",
        # MySQL has no time zone type; the fixture writer's session is UTC.
        "mysql": "DATETIME(6)",
    }[source.engine]
    enforces_keys = source.engine in {"postgresql", "mysql"}
    key_constraint = "PRIMARY KEY" if enforces_keys else "NOT NULL"
    parent = f"{source.reference('users')}({source.identifier('user_id')})"
    foreign_key = f"REFERENCES {parent}" if source.engine == "postgresql" else ""
    # MySQL ignores a column's inline REFERENCES; only a table constraint works.
    table_foreign_key = (
        f", FOREIGN KEY ({source.identifier('user_id')}) REFERENCES {parent}"
        if source.engine == "mysql"
        else ""
    )
    amount_type = "DOUBLE PRECISION" if source.engine == "postgresql" else "DOUBLE"
    source.execute_write(
        f"CREATE TABLE {source.reference('users')} ("
        f"{source.identifier('user_id')} BIGINT {key_constraint}, "
        f"{source.identifier('created_at')} {timestamp_type}, "
        f"{source.identifier('country')} VARCHAR(8))"
    )
    source.execute_write(
        f"CREATE TABLE {source.reference('orders')} ("
        f"{source.identifier('order_id')} BIGINT {key_constraint}, "
        f"{source.identifier('user_id')} BIGINT {foreign_key}, "
        f"{source.identifier('event_time')} {timestamp_type}, "
        f"{source.identifier('amount')} {amount_type}, "
        f"{source.identifier('channel')} VARCHAR(16){table_foreign_key})"
    )
    created_at = datetime(2025, 1, 1, tzinfo=UTC)
    users = [(entity, created_at, "US") for entity in range(1, 7)]
    users += [(LARGE_ENTITY_ID, created_at, "CA")]
    source.insert_rows("users", users)
    orders = []
    for entity in range(1, 5):
        for timestamp, value, channel in (
            ("2025-12-15", entity, "web"),
            ("2026-01-01", 999, "web"),  # Lower boundary is excluded.
            ("2026-01-02", None, "web"),  # FIRST must preserve NULL.
            ("2026-01-02", 10 + entity, "web"),
            ("2026-01-03", 10 + entity, "web"),
            ("2026-01-05", 30 + entity, "ordered"),
            ("2026-01-05", 40 + entity, "ordered"),  # Ordered tie uses key.
            ("2026-01-31", 20 + entity, "web"),  # Upper boundary included.
            ("2026-01-31", None, "web"),  # LAST must preserve NULL.
            ("2026-01-31T00:00:01", 777, "web"),
            ("2026-03-01", 999_999, "web"),  # Future feature row excluded.
        ):
            instant = datetime.fromisoformat(timestamp).replace(tzinfo=UTC)
            orders.append((len(orders) + 1, entity, instant, value, channel))
    orders.append((len(orders) + 1, 6, datetime(2026, 1, 10, tzinfo=UTC), None, "web"))
    orders.append(
        (len(orders) + 1, LARGE_ENTITY_ID, datetime(2025, 12, 1, tzinfo=UTC), 5, "web")
    )
    source.insert_rows("orders", orders)


def _grant_reader(source: WarehouseSource) -> None:
    if source.engine == "snowflake":
        role = source.configuration.get("reader_role")
        if role:
            quoted_role = source.identifier(role)
            source.execute_write(
                f"GRANT USAGE ON SCHEMA {source.reference()} TO ROLE {quoted_role}"
            )
            source.execute_write(
                f"GRANT SELECT ON ALL TABLES IN SCHEMA {source.reference()} "
                f"TO ROLE {quoted_role}"
            )
    else:
        principal = source.configuration.get("reader_principal")
        if principal:
            quoted_principal = source.identifier(principal)
            source.execute_write(
                f"GRANT USE SCHEMA ON SCHEMA {source.reference()} TO {quoted_principal}"
            )
            source.execute_write(
                f"GRANT SELECT ON SCHEMA {source.reference()} TO {quoted_principal}"
            )


@contextmanager
def _create_warehouse_source(engine: str, request):
    configuration = _load_test_configuration(engine, request)
    schema_name = "nori_rel_test_" + uuid4().hex
    config = DatabaseCreateRequest(
        engine=engine,
        connection=_connection_settings(engine, configuration["connection"]),
        credential=EnvironmentCredential(
            variable=configuration["reader_credential_env"]
        ),
        schema_name=schema_name,
        tables=["users", "orders"],
        time_columns={"orders": "event_time"},
        primary_keys={"users": ["user_id"], "orders": ["order_id"]},
        foreign_keys={
            "orders": [
                ForeignKeySchema(
                    columns=["user_id"],
                    referred_table="users",
                    referred_columns=["user_id"],
                )
            ]
        },
    )
    connector_class = _connector_class(engine)
    connector = connector_class(config, RelationalLimits(min_training_rows=2))
    writer_settings = {
        **configuration["connection"],
        **configuration.get("writer_connection", {}),
    }
    writer_config = config.model_copy(
        update={
            "connection": _connection_settings(engine, writer_settings),
            "credential": EnvironmentCredential(
                variable=configuration["writer_credential_env"]
            ),
            # OpenSession may select its schema eagerly. The owned test schema
            # does not exist until this writer has provisioned it.
            "schema_name": configuration.get(
                "writer_schema", "PUBLIC" if engine == "snowflake" else "default"
            ),
        }
    )
    writer_connector = connector_class(writer_config, RelationalLimits())
    # Only fixture provisioning accesses the native writer connection. Connector
    # reads use a separate credential and never execute CREATE/INSERT/GRANT/DROP.
    writer = writer_connector._open_connection()
    source = WarehouseSource(engine, configuration, config, connector, writer)
    owned_schema_created = False
    try:
        source.execute_write(
            "ALTER SESSION SET TIMEZONE='UTC'"
            if engine == "snowflake"
            else "SET TIME ZONE 'UTC'"
        )
        retention_setting = (
            " DATA_RETENTION_TIME_IN_DAYS=1" if engine == "snowflake" else ""
        )
        source.execute_write(f"CREATE SCHEMA {source.reference()}{retention_setting}")
        owned_schema_created = True
        _populate_fixture(source)
        _grant_reader(source)
        yield source
    finally:
        connector.close()
        try:
            if owned_schema_created:
                # The random schema was created by this fixture, never supplied
                # by environment configuration or reused from another test.
                source.execute_write(f"DROP SCHEMA {source.reference()} CASCADE")
        finally:
            writer.close()
            writer_connector.close()


@pytest.fixture(
    params=[
        pytest.param("snowflake", marks=[pytest.mark.warehouse, pytest.mark.snowflake]),
        pytest.param(
            "databricks", marks=[pytest.mark.warehouse, pytest.mark.databricks]
        ),
    ]
)
def warehouse_source(request):
    with _create_warehouse_source(request.param, request) as source:
        yield source


@pytest.fixture
def warehouse_pair(request):
    # Validate both configurations before allocating either paid service.
    for engine in WAREHOUSE_CONFIG_ENV:
        _load_test_configuration(engine, request)
    with (
        _create_warehouse_source("snowflake", request) as snowflake,
        _create_warehouse_source("databricks", request) as databricks,
    ):
        yield snowflake, databricks


@pytest.fixture
def postgres_parity_source():
    """Populate the exact same independently expected fixture in PostgreSQL."""
    variable = "NORI_REL_TEST_DATABASE_URL"
    if not os.environ.get(variable):
        pytest.skip("PostgreSQL integration database is supplied by CI")
    import psycopg

    from nori_rel.connectors.postgres import PostgreSQLConnector

    config = DatabaseCreateRequest(
        engine="postgresql",
        credential=EnvironmentCredential(variable=variable),
        schema_name="nori_rel_test_" + uuid4().hex,
        tables=["users", "orders"],
        time_columns={"orders": "event_time"},
    )
    connector = PostgreSQLConnector(config, RelationalLimits(min_training_rows=2))
    with psycopg.connect(os.environ[variable], autocommit=True) as writer:
        source = WarehouseSource("postgresql", {}, config, connector, writer)
        source.execute_write("SET TIME ZONE 'UTC'")
        source.execute_write(f"CREATE SCHEMA {source.reference()}")
        try:
            _populate_fixture(source)
            yield source
        finally:
            connector.close()
            source.execute_write(f"DROP SCHEMA {source.reference()} CASCADE")


def connect_mysql_writer(database_url: str):
    """Open an autocommit PyMySQL session in UTC for fixture provisioning."""
    import pymysql
    from sqlalchemy.engine import make_url

    url = make_url(database_url)
    return pymysql.connect(
        host=url.host,
        port=url.port or 3306,
        user=url.username,
        password=url.password or "",
        autocommit=True,
        init_command="SET time_zone = '+00:00'",
    )


@contextmanager
def create_mysql_source(variable: str = "NORI_REL_TEST_MYSQL_URL", **options):
    """Yield the shared fixture in a new MySQL database, dropped afterward.

    The connector reads with the URL in ``variable``; the fixture is written
    with NORI_REL_TEST_MYSQL_URL. Options override the source configuration,
    e.g. tables=[...], or set limits=RelationalLimits(). Tables beyond users
    and orders are the caller's to create.
    """
    from nori_rel.connectors.mysql import MySQLConnector

    limits = options.pop("limits", RelationalLimits(min_training_rows=2))
    config = DatabaseCreateRequest(
        engine="mysql",
        credential=EnvironmentCredential(variable=variable),
        schema_name="nori_rel_test_" + uuid4().hex,
        **{
            "tables": ["users", "orders"],
            "time_columns": {"orders": "event_time"},
            **options,
        },
    )
    connector = MySQLConnector(config, limits)
    writer = connect_mysql_writer(os.environ["NORI_REL_TEST_MYSQL_URL"])
    source = WarehouseSource("mysql", {}, config, connector, writer)
    try:
        source.execute_write(f"CREATE DATABASE {source.reference()}")
        _populate_fixture(source)
        yield source
    finally:
        connector.close()
        try:
            source.execute_write(f"DROP DATABASE IF EXISTS {source.reference()}")
        finally:
            writer.close()


@pytest.fixture
def mysql_parity_source():
    """Populate the exact same independently expected fixture in MySQL."""
    if not os.environ.get("NORI_REL_TEST_MYSQL_URL"):
        pytest.skip("MySQL integration database is supplied by CI")
    with create_mysql_source() as source:
        yield source
