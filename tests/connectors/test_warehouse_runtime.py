"""Shared warehouse preparation, native binding, limits, and client contracts."""

from __future__ import annotations

import json
import re
import sqlite3
import sys
import time
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock

import pandas as pd
import pytest

from nori_rel import (
    DatabaseCreateRequest,
    DatabaseEngine,
    DatabricksConnection,
    EnvironmentCredential,
    ForeignKeySchema,
    NoriRelClient,
    RelationalLimits,
    SnowflakeConnection,
)
from nori_rel._credentials import CredentialResolutionError, CredentialResolver
from nori_rel.connectors.base import SnapshotProvenance
from nori_rel.connectors.errors import ConnectorError, SnapshotLimitError
from nori_rel.connectors.warehouse import (
    WarehouseConnector,
    WarehouseSession,
    bind_native_parameters,
    build_warehouse_schema_graph,
    dataframe_records_by_lowercase_column,
    normalize_warehouse_timestamp_utc,
)
from nori_rel.contracts import ColumnSchema, SchemaGraph, TableSchema

ANCHOR = datetime(2026, 1, 1, tzinfo=UTC)
CUTOFF = datetime(2026, 2, 1, tzinfo=UTC)
PINNED_AT = datetime(2026, 10, 12, tzinfo=UTC)
LARGE_ID = 2**53 + 1


def _build_snowflake_config(**options):
    connection = SnowflakeConnection(account="test", database="test", warehouse="test")
    return DatabaseCreateRequest(
        engine="snowflake",
        credential=EnvironmentCredential(variable="WAREHOUSE_TEST_AUTH"),
        connection=connection,
        schema_name="test",
        tables=["users", "orders"],
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
        time_columns={"orders": "event_time"},
        **options,
    )


def _build_databricks_config():
    return DatabaseCreateRequest.model_validate(
        _build_snowflake_config().model_dump()
        | {
            "engine": DatabaseEngine.DATABRICKS,
            "connection": DatabricksConnection(
                server_hostname="example.cloud.databricks.com",
                http_path="/sql/1.0/warehouses/test",
                catalog="test",
            ),
        }
    )


def schema_graph():
    return SchemaGraph(
        database_id="test",
        schema_name="test",
        discovered_at=PINNED_AT,
        tables=[
            TableSchema(
                name="users",
                columns=[
                    ColumnSchema(name="user_id", data_type="BIGINT", nullable=True),
                    ColumnSchema(
                        name="created_at", data_type="TIMESTAMP", nullable=True
                    ),
                    ColumnSchema(name="country", data_type="STRING", nullable=True),
                ],
            ),
            TableSchema(
                name="orders",
                columns=[
                    ColumnSchema(name="order_id", data_type="BIGINT", nullable=True),
                    ColumnSchema(name="user_id", data_type="BIGINT", nullable=True),
                    ColumnSchema(
                        name="event_time", data_type="TIMESTAMP", nullable=True
                    ),
                    ColumnSchema(
                        name="amount", data_type="DECIMAL(12,2)", nullable=True
                    ),
                ],
                time_column="event_time",
            ),
        ],
    )


class SQLiteCursor:
    """Execute warehouse preparation SQL through an explicit syntax adapter."""

    def __init__(self, connection):
        self.connection = connection
        self.cursor = connection.database.cursor()

    @property
    def description(self):
        return self.cursor.description

    def execute(self, sql, params):
        self.connection.calls.append((sql, params))
        sql = re.sub(r"CAST\(\? AS TIMESTAMP(?:_TZ)?\)", "?", sql)
        sql = sql.replace("TIMESTAMPADD(HOUR,", "TIMESTAMPADD('HOUR',")
        params = [
            value[1]
            if isinstance(value, tuple) and value[0] == "TIMESTAMP_TZ"
            else value
            for value in params
        ]
        normalized = [
            value.isoformat(sep=" ") if isinstance(value, datetime) else value
            for value in params
        ]
        self.cursor.execute(sql, normalized)

    def fetchmany(self, size):
        return self.cursor.fetchmany(size)

    def close(self):
        self.cursor.close()


class SQLiteConnection:
    def __init__(self):
        self.database = sqlite3.connect(":memory:")
        self.calls = []
        self.closed = False
        self.cursor_options = []
        self.database.create_function(
            "TIMESTAMPADD",
            3,
            lambda unit, hours, stamp: (
                datetime.fromisoformat(stamp) + timedelta(hours=hours)
            ).isoformat(sep=" "),
        )
        self.database.executescript("""
CREATE TABLE users(user_id INTEGER, created_at TEXT, country TEXT);
CREATE TABLE orders(order_id INTEGER, user_id INTEGER, event_time TEXT, amount NUMERIC);
""")
        self.database.executemany(
            "INSERT INTO users VALUES (?,?,?)",
            [
                (LARGE_ID, "2025-01-01 00:00:00+00:00", "US"),
                (2, "2025-01-01 00:00:00+00:00", "CA"),
            ],
        )
        self.database.executemany(
            "INSERT INTO orders VALUES (?,?,?,?)",
            [
                (1, LARGE_ID, "2026-01-01 00:00:00+00:00", 100),
                (2, LARGE_ID, "2026-01-10 00:00:00+00:00", 1.25),
                (3, LARGE_ID, "2026-01-20 00:00:00+00:00", 2.50),
                (4, LARGE_ID, "2026-02-02 00:00:00+00:00", 10000),
            ],
        )

    def cursor(self, **options):
        self.cursor_options.append(options)
        return SQLiteCursor(self)

    def close(self):
        self.closed = True
        self.database.close()


class PreparedConnector(WarehouseConnector):
    """Exercise the real executor, not a mocked prepare_inputs implementation."""

    def __init__(self, *, engine=DatabaseEngine.SNOWFLAKE, **limits):
        config = (
            _build_databricks_config()
            if engine == DatabaseEngine.DATABRICKS
            else _build_snowflake_config()
        )
        super().__init__(config, RelationalLimits(**limits))
        self.database = SQLiteConnection()
        self.schema = schema_graph()
        self.pins = 0

    def _open_connection(self):
        return self.database

    def _configure_connection(self, connection):
        pass

    def _discover_schema(self, connection, database_id, *, time_columns=None):
        return self.schema.model_copy(deep=True)

    def _pin_tables(self, connection, schema):
        self.pins += 1
        return PINNED_AT, {table.name: f'"{table.name}"' for table in schema.tables}


def prepare(connector, **options):
    return connector.prepare_inputs(
        "test",
        f"PREDICT SUM(orders.amount,0,30,days) FOR users.user_id={LARGE_ID}",
        anchor_times=[ANCHOR],
        anchor_time=CUTOFF,
        observed_through=CUTOFF,
        entity_created_at="created_at",
        **options,
    )


def test_complete_preparation_retains_ids_labels_cutoffs_and_provenance():
    source = PreparedConnector()
    try:
        snapshot, context, counts = prepare(source, query_filter="users.country='US'")
        assert source.pins == 1
        assert dict(zip(context.entity_id, context.label, strict=True)) == {
            2: 0,
            LARGE_ID: 3.75,
        }
        assert counts == {
            "candidate_rows": 2,
            "usable_rows": 2,
            "dropped_unknown_labels": 0,
        }
        assert snapshot.scoring_filter_ids == (LARGE_ID,)
        assert snapshot.tables["users"].user_id.tolist() == [LARGE_ID, 2]
        assert snapshot.tables["orders"].order_id.tolist() == [1, 2, 3]
        assert snapshot.provenance.extraction_timestamp == PINNED_AT
        assert snapshot.provenance.engine == "snowflake"
        assert source.database.cursor_options[0] == {}
    finally:
        source.close()


def test_static_target_reads_labels_at_the_pinned_extraction_time():
    source = PreparedConnector()
    try:
        snapshot, context, counts = source.prepare_inputs(
            "test",
            "PREDICT users.country='US' FOR users.user_id=2",
            anchor_times=None,
            anchor_time=None,
            observed_through=None,
            entity_created_at=None,
            query_filter="users.country='CA'",
        )
        # User 2 is scored, so only the large-ID user remains as context.
        assert context.entity_id.tolist() == [LARGE_ID]
        assert context.label.astype(bool).tolist() == [True]
        assert context.anchor_time.tolist() == [PINNED_AT]
        assert counts["usable_rows"] == 1
        assert snapshot.as_of == PINNED_AT
        assert snapshot.scoring_filter_ids == (2,)
    finally:
        source.close()


def test_databricks_preparation_retains_pinned_inputs_and_bounds_driver_buffers():
    source = PreparedConnector(engine=DatabaseEngine.DATABRICKS)
    try:
        snapshot, context, counts = prepare(source, query_filter="users.country='US'")
        assert source.pins == 1
        assert dict(zip(context.entity_id, context.label, strict=True)) == {
            2: 0,
            LARGE_ID: 3.75,
        }
        assert counts == {
            "candidate_rows": 2,
            "usable_rows": 2,
            "dropped_unknown_labels": 0,
        }
        assert snapshot.scoring_filter_ids == (LARGE_ID,)
        assert snapshot.tables["users"].user_id.tolist() == [LARGE_ID, 2]
        assert snapshot.tables["orders"].order_id.tolist() == [1, 2, 3]
        assert snapshot.provenance.extraction_timestamp == PINNED_AT
        assert snapshot.provenance.engine == "databricks"
        assert all(
            options["arraysize"] <= 1000 and options["buffer_size_bytes"] <= 1_000_000
            for options in source.database.cursor_options
        )
    finally:
        source.close()


@pytest.mark.parametrize(
    "limits",
    [{"max_total_rows": 5}, {"max_rows_per_table": 2}, {"max_snapshot_bytes": 200}],
)
def test_preparation_enforces_combined_budgets(limits):
    source = PreparedConnector(**limits)
    with pytest.raises(SnapshotLimitError):
        prepare(source, max_context_rows=2)
    assert source.database.closed
    source.close()


def test_preparation_rejects_source_schema_changes():
    source = PreparedConnector()
    original = source._discover_schema
    calls = 0

    def discover(*args, **kwargs):
        nonlocal calls
        calls += 1
        graph = original(*args, **kwargs)
        if calls == 2:
            graph.tables[0].columns[0].data_type = "STRING"
        return graph

    source._discover_schema = discover
    with pytest.raises(ConnectorError, match="schema changed"):
        prepare(source)
    assert source.database.closed


def test_preparation_rejects_table_replacement_without_schema_changes():
    source = PreparedConnector()
    calls = 0

    def identities(connection, schema):
        nonlocal calls
        calls += 1
        return {"users": "original", "orders": "original" if calls == 1 else "replaced"}

    source._get_table_identity_by_name = identities
    with pytest.raises(ConnectorError, match="table identity changed"):
        prepare(source)
    assert source.database.closed


@pytest.mark.parametrize(
    "details",
    [
        {"id": ["delta-table-id"], "format": ["delta"]},
        {"id": [None], "format": ["delta"]},
        {"id": ["delta-table-id"], "format": ["parquet"]},
    ],
)
def test_databricks_identity_verification_uses_physical_delta_ids(monkeypatch, details):
    from .test_databricks_drivers import databricks_connector
    from .test_drivers import FakeConnection

    source = databricks_connector(monkeypatch)
    connection = FakeConnection(lambda sql, params: pd.DataFrame(details))
    graph = schema_graph()
    if details["id"][0] and details["format"][0] == "delta":
        assert source._get_table_identity_by_name(connection, graph) == {
            "users": "delta-table-id",
            "orders": "delta-table-id",
        }
        assert all("DESCRIBE DETAIL" in sql for sql, _ in connection.calls)
    else:
        with pytest.raises(ConnectorError, match=r"identity|no longer a Delta"):
            source._get_table_identity_by_name(connection, graph)


@pytest.mark.parametrize(
    "invalid",
    [
        "INSERT INTO users VALUES (2,'2025-01-01','US')",
        "INSERT INTO orders VALUES (5,999,'2026-01-02',1)",
    ],
)
def test_preparation_rejects_invalid_relational_keys(invalid):
    source = PreparedConnector()
    source.database.database.execute(invalid)
    with pytest.raises(ConnectorError, match=r"primary keys|missing parent"):
        prepare(source)
    assert source.database.closed


def test_future_invalid_keys_do_not_affect_cutoff_visible_rows():
    source = PreparedConnector()
    source.database.database.executemany(
        "INSERT INTO orders VALUES (?,?,?,?)",
        [
            (2, LARGE_ID, "2026-02-03 00:00:00+00:00", 1),
            (6, 999, "2026-02-04 00:00:00+00:00", 1),
        ],
    )
    try:
        snapshot, _, _ = prepare(source)
        assert snapshot.tables["orders"].order_id.tolist() == [1, 2, 3]
        key_queries = [
            (sql, params)
            for sql, params in source.database.calls
            if "invalid_key" in sql or "invalid_relationship" in sql
        ]
        assert key_queries
        assert all("event_time" in sql and "<=" in sql for sql, _ in key_queries[1:])
        assert all(
            CUTOFF in [value[1] for value in params] for _, params in key_queries[1:]
        )
    finally:
        source.close()


def test_oversized_metadata_rejected_before_key_scans_or_pinning():
    source = PreparedConnector(max_rows_per_table=2)
    source._get_table_row_counts = lambda connection, schema: {"users": 2, "orders": 4}
    with pytest.raises(SnapshotLimitError, match="table exceeds"):
        prepare(source, max_context_rows=2)
    assert source.pins == 0
    assert source.database.calls == []
    assert source.database.closed


def test_unknown_table_sizes_use_bounded_probes_before_key_scans():
    source = PreparedConnector(max_rows_per_table=2)
    with pytest.raises(SnapshotLimitError, match="table exceeds"):
        prepare(source, max_context_rows=2)
    statements = [sql for sql, _ in source.database.calls]
    assert len(statements) == 2
    assert all("bounded_rows" in sql and "LIMIT 3" in sql for sql in statements)
    assert all(
        "invalid_key" not in sql and "invalid_relationship" not in sql
        for sql in statements
    )


def test_pinned_session_has_named_frozen_fields():
    source = PreparedConnector()
    try:
        with source._session("test") as session:
            assert isinstance(session, WarehouseSession)
            assert session.connection is source.database
            assert session.pinned_at == PINNED_AT
            assert session.table_references == {
                "users": '"users"',
                "orders": '"orders"',
            }
            with pytest.raises(FrozenInstanceError):
                session.pinned_at = ANCHOR
    finally:
        source.close()


def test_snapshot_provenance_is_typed_frozen_and_serializable():
    provenance = SnapshotProvenance(
        engine="snowflake",
        extraction_timestamp=PINNED_AT,
        table_references={"users": '"users"'},
    )
    assert provenance.model_dump(mode="json")["engine"] == "snowflake"
    with pytest.raises(ValueError, match="frozen"):
        provenance.engine = "postgresql"
    with pytest.raises(ValueError, match="Extra inputs"):
        SnapshotProvenance(
            engine="snowflake",
            extraction_timestamp=PINNED_AT,
            table_references={},
            typo="unrecognized",
        )


def test_server_timeout_path_does_not_start_cancellation_thread(monkeypatch):
    source = PreparedConnector(statement_timeout_ms=1)
    monkeypatch.setattr(
        "nori_rel.connectors.warehouse.Timer",
        lambda *args, **kwargs: pytest.fail("server timeout must not start a timer"),
    )
    try:
        source.test()
        assert not source.database.closed
    finally:
        source.close()


def test_supported_query_cancel_callback_never_closes_shared_connection():
    source = PreparedConnector(statement_timeout_ms=1)
    cancelled = Mock()
    source._get_query_cancellation_callback = lambda cursor: cancelled
    original_cursor = source.database.cursor

    def slow_cursor(**options):
        cursor = original_cursor(**options)
        original_execute = cursor.execute

        def slow_execute(sql, params):
            time.sleep(0.03)
            original_execute(sql, params)

        cursor.execute = slow_execute
        return cursor

    source.database.cursor = slow_cursor
    try:
        with pytest.raises(ConnectorError, match="timed out"):
            source._get_query_frame(source.database, "SELECT 1")
        cancelled.assert_called_once_with()
        assert not source.database.closed
    finally:
        source.close()


def test_shared_metadata_helpers_preserve_literal_names_and_utc_instants():
    assert dataframe_records_by_lowercase_column(
        pd.DataFrame({"NAME": ["MixedCase"]})
    ) == [{"name": "MixedCase"}]
    assert normalize_warehouse_timestamp_utc("2026-01-01 01:00:00+01:00") == ANCHOR
    with pytest.raises(ConnectorError, match="snapshot timestamp"):
        normalize_warehouse_timestamp_utc(None)
    graph = build_warehouse_schema_graph(
        _build_snowflake_config(),
        "test",
        ["users"],
        [
            {
                "table_name": "users",
                "column_name": "user_id",
                "data_type": "BIGINT",
                "ordinal_position": 1,
                "is_nullable": "NO",
            }
        ],
        [{"table_name": "users", "column_name": "user_id", "key_sequence": 1}],
        [],
        schema_graph().capabilities,
    )
    assert graph.tables[0].primary_key == ["user_id"]


@pytest.mark.parametrize(
    "statement",
    [
        "DELETE FROM users",
        "WITH x AS (DELETE FROM users) SELECT 1",
        "SELECT 1; DROP TABLE users",
    ],
)
def test_executor_rejects_write_statements(statement):
    source = PreparedConnector()
    with pytest.raises(ConnectorError, match="read statements"):
        source._get_query_frame(source.database, statement)
    assert source.database.calls == []
    source.close()


def test_native_binding_keeps_identifier_percents_and_repeated_values():
    sql, values = bind_native_parameters(
        'SELECT "literal%%(value)s", %(value)s, %(value)s',
        {"value": "';DELETE FROM users"},
    )
    assert sql == 'SELECT "literal%(value)s", ?, ?'
    assert values == ["';DELETE FROM users"] * 2


@pytest.mark.parametrize(
    "value,cast,text",
    [
        (Decimal("1e-18"), "DECIMAL(18, 18)", "0.000000000000000001"),
        (Decimal("1e20"), "DECIMAL(21, 0)", "100000000000000000000"),
        (10**25, "DECIMAL(26, 0)", "10000000000000000000000000"),
    ],
)
@pytest.mark.parametrize("engine", ["snowflake", "databricks"])
def test_warehouse_binding_avoids_driver_numeric_truncation(value, cast, text, engine):
    sql, values = bind_native_parameters(
        "SELECT %(number)s", {"number": value}, engine=engine
    )
    assert sql == f"SELECT CAST(? AS {cast})"
    assert values == [text]


def test_decimal_feature_normalization_preserves_relational_ids():
    table = TableSchema(
        name="orders",
        primary_key=["order_id"],
        columns=[
            ColumnSchema(name="order_id", data_type="NUMBER(38,0)", nullable=False),
            ColumnSchema(name="amount", data_type="NUMBER(12,2)", nullable=True),
        ],
    )
    frame = pd.DataFrame(
        {
            "order_id": [Decimal(LARGE_ID), Decimal(10**25)],
            "amount": [Decimal("1.25"), None],
        }
    )
    normalized = WarehouseConnector._normalize_table_frame(frame, table)
    assert normalized.order_id.tolist() == [Decimal(LARGE_ID), Decimal(10**25)]
    assert str(normalized.amount.dtype) == "Float64"
    assert normalized.amount.iloc[0] == 1.25


def test_public_client_passes_typed_engine_settings_and_relationships():
    runtime = Mock()
    client = NoriRelClient(mode="hosted", api_key="test", _runtime=runtime)
    config = _build_snowflake_config()
    client.connect(
        engine=config.engine,
        connection=config.connection,
        credential=config.credential,
        schema_name=config.schema_name,
        tables=config.tables,
        time_columns=config.time_columns,
        primary_keys=config.primary_keys,
        foreign_keys=config.foreign_keys,
    )
    assert runtime.connect.call_args.args[0] == config


def test_public_client_passes_databricks_settings_and_relationships():
    runtime = Mock()
    client = NoriRelClient(mode="hosted", api_key="test", _runtime=runtime)
    config = _build_databricks_config()
    client.connect(**config.model_dump())
    assert runtime.connect.call_args.args[0] == config


@pytest.mark.parametrize(
    "parameters",
    [
        {"schema_name": None},
        {"connection": None},
        {
            "connection": SnowflakeConnection(
                account="test", database="test", warehouse="test"
            )
        },
        {
            "database_url": "postgresql://user:password@localhost/test",
            "credential": None,
        },
    ],
)
def test_public_client_rejects_mismatched_databricks_settings(parameters):
    runtime = Mock()
    options = _build_databricks_config().model_dump() | parameters
    with pytest.raises(ValueError):
        NoriRelClient(mode="hosted", api_key="test", _runtime=runtime).connect(
            **options
        )
    runtime.connect.assert_not_called()


@pytest.mark.parametrize(
    "overrides",
    [
        {"server_hostname": "https://example.cloud.databricks.com"},
        {"server_hostname": "example.cloud.databricks.com/path"},
        {"http_path": "/sql/1.0/warehouses/test?token=private"},
        {"http_path": "https://example.cloud.databricks.com/sql/1.0/warehouses/test"},
        {"catalog": "\x00"},
    ],
)
def test_databricks_typed_settings_reject_unsafe_endpoints(overrides):
    parameters = _build_databricks_config().connection.model_dump() | overrides
    with pytest.raises(ValueError):
        DatabricksConnection(**parameters)


@pytest.mark.parametrize(
    "parameters",
    [
        {"schema_name": None},
        {"connection": None},
        {
            "database_url": "postgresql://user:password@localhost/test",
            "credential": None,
        },
    ],
)
def test_public_client_rejects_mismatched_warehouse_settings(parameters):
    runtime = Mock()
    options = _build_snowflake_config().model_dump()
    options.pop("primary_keys")
    options.pop("foreign_keys")
    options.update(parameters)
    with pytest.raises(ValueError):
        NoriRelClient(mode="hosted", api_key="test", _runtime=runtime).connect(
            **options
        )
    runtime.connect.assert_not_called()


@pytest.mark.parametrize(
    "secret",
    ["not json", "[]", "{}", '{"token":false}', '{"token":{"value":"secret"}}'],
)
def test_warehouse_secret_json_errors_do_not_expose_values(monkeypatch, secret):
    monkeypatch.setenv("WAREHOUSE_TEST_AUTH", secret)
    with pytest.raises(CredentialResolutionError) as error:
        CredentialResolver().resolve_parameters(
            EnvironmentCredential(variable="WAREHOUSE_TEST_AUTH")
        )
    assert "secret" not in str(error.value)
    assert error.value.__suppress_context__


def test_warehouse_auth_is_resolved_lazily_and_never_dumped(monkeypatch):
    monkeypatch.setenv("WAREHOUSE_TEST_AUTH", json.dumps({"token": "private-value"}))
    config = _build_snowflake_config()
    assert "private-value" not in config.model_dump_json()
    assert CredentialResolver().resolve_parameters(config.credential) == {
        "token": "private-value"
    }


def test_source_cleanup_is_idempotent_and_prevents_reopening():
    source = PreparedConnector()
    source.test()
    source.close()
    source.close()
    with pytest.raises(ConnectorError, match="closed"):
        source.test()


def test_public_import_does_not_load_optional_warehouse_drivers():
    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, nori_rel; "
            "assert 'snowflake.connector' not in sys.modules; "
            "assert 'databricks.sql' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
