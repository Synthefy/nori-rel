"""Databricks official-driver wiring, discovery, and pinned Delta read checks."""

from __future__ import annotations

import json
import sys
import types
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from threading import Event

import pandas as pd
import pytest

from nori_rel.connectors.databricks import DatabricksConnector
from nori_rel.connectors.errors import ConnectorError, SnapshotLimitError
from nori_rel.contracts import (
    DatabaseCreateRequest,
    DatabricksConnection,
    EnvironmentCredential,
)
from nori_rel.limits import RelationalLimits

from .test_drivers import FakeConnection, FakeCursor, snapshot_schema


def databricks_connector(monkeypatch, secret=None, *, tables=None, **limits):
    """Build a Databricks source with credentials resolved from environment JSON."""
    monkeypatch.setenv(
        "NORI_TEST_DATABRICKS_SECRET",
        json.dumps({"access_token": "token"} if secret is None else secret),
    )
    config = DatabaseCreateRequest(
        engine="databricks",
        credential=EnvironmentCredential(variable="NORI_TEST_DATABRICKS_SECRET"),
        connection=DatabricksConnection(
            server_hostname="example.cloud.databricks.com",
            http_path="/sql/1.0/warehouses/abc",
            catalog="sales",
        ),
        schema_name="public",
        tables=tables,
    )
    return DatabricksConnector(config, RelationalLimits(**limits))


def install_databricks_driver(monkeypatch, connection):
    """Install an importable fake Databricks driver and collect connect kwargs."""
    package = types.ModuleType("databricks")
    package.__path__ = []
    driver = types.ModuleType("databricks.sql")
    calls = []

    def connect(**kwargs):
        calls.append(kwargs)
        return connection

    driver.connect = connect
    package.sql = driver
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, driver.__name__, driver)
    return calls


def test_databricks_pat_and_session_config(monkeypatch):
    """PAT authentication keeps native binding and certificate/hostname checks."""
    source = databricks_connector(monkeypatch, {"access_token": "private-token"})
    fake = FakeConnection()
    calls = install_databricks_driver(monkeypatch, fake)
    assert source._open_connection() is fake
    source._configure_connection(fake)
    assert calls[0]["use_inline_params"] is False
    assert calls[0]["use_cloud_fetch"] is False
    assert calls[0]["_tls_no_verify"] is False
    assert calls[0]["_tls_verify_hostname"] is True
    assert calls[0]["access_token"] == "private-token"
    assert fake.calls == [
        ("SET TIME ZONE 'UTC'", None),
        ("SET STATEMENT_TIMEOUT = 300", None),
    ]
    assert fake.cursors[0].closed


def test_databricks_query_limits_direct_result_buffers(monkeypatch):
    """Native cursor buffers and result limits match remaining extraction bounds."""
    source = databricks_connector(monkeypatch)
    fake = FakeConnection(lambda sql, params: pd.DataFrame([[1]], columns=["id"]))
    source._get_query_frame(fake, "SELECT 1 AS id", max_rows=1, max_bytes=200)
    assert fake.cursor_options == [
        {"arraysize": 2, "buffer_size_bytes": 200, "row_limit": 2}
    ]
    source._get_query_frame(fake, "SELECT 1 AS id")
    assert fake.cursor_options[1] == {
        "arraysize": 1000,
        "buffer_size_bytes": 1_000_000,
        "row_limit": 1_000_001,
    }


def test_databricks_oauth_provider_pins_explicit_service_identity(monkeypatch):
    """OAuth uses only the configured HTTPS host and service principal."""
    source = databricks_connector(
        monkeypatch, {"client_id": "id", "client_secret": "private-secret"}
    )
    calls = install_databricks_driver(monkeypatch, FakeConnection())
    sdk = types.ModuleType("databricks.sdk")
    sdk.__path__ = []
    core = types.ModuleType("databricks.sdk.core")
    configs = []
    core.Config = lambda **kwargs: configs.append(kwargs) or kwargs
    core.oauth_service_principal = lambda config: config
    monkeypatch.setitem(sys.modules, "databricks.sdk", sdk)
    monkeypatch.setitem(sys.modules, "databricks.sdk.core", core)
    source._open_connection()
    calls[0]["credentials_provider"]()
    assert configs == [
        {
            "host": "https://example.cloud.databricks.com",
            "client_id": "id",
            "client_secret": "private-secret",
            "auth_type": "oauth-m2m",
        }
    ]


@pytest.mark.parametrize(
    "secret",
    [
        {"access_token": "secret", "_tls_no_verify": "true"},
        {"access_token": "secret", "client_id": "id", "client_secret": "secret"},
        {"client_id": "id"},
    ],
)
def test_databricks_rejects_unsafe_or_incomplete_auth(monkeypatch, secret):
    """TLS flags and partial service-principal credentials cannot reach the driver."""
    source = databricks_connector(monkeypatch, secret)
    with pytest.raises(ConnectorError):
        source._open_connection()


def test_databricks_connection_errors_are_sanitized(monkeypatch):
    """Official driver exceptions cannot expose tokens through exception chaining."""
    source = databricks_connector(monkeypatch, {"access_token": "private-token"})
    install_databricks_driver(monkeypatch, FakeConnection())

    def fail(**kwargs):
        raise RuntimeError("private-token")

    monkeypatch.setattr(sys.modules["databricks.sql"], "connect", fail)
    with pytest.raises(ConnectorError) as error:
        source._open_connection()
    assert "private-token" not in str(error.value)
    assert error.value.__suppress_context__


def test_databricks_session_failure_releases_connection(monkeypatch):
    """Failed session initialization cannot leak secrets or retain the connection."""
    source = databricks_connector(monkeypatch)

    def fail(sql, params):
        raise RuntimeError("private-token")

    fake = FakeConnection(fail)
    install_databricks_driver(monkeypatch, fake)
    with pytest.raises(ConnectorError) as error:
        source._get_connection()
    assert "private-token" not in str(error.value)
    assert error.value.__suppress_context__
    assert fake.closed
    assert fake.cursors[0].closed


def test_databricks_excludes_commits_after_fixed_timestamp(monkeypatch):
    """Concurrent post-pin writes cannot enter a selected table version."""
    source = databricks_connector(monkeypatch)
    instant = datetime(2026, 1, 1, tzinfo=UTC)

    def respond(sql, params):
        if "CURRENT_TIMESTAMP" in sql:
            return pd.DataFrame([[instant]], columns=["timestamp"])
        frame = pd.DataFrame(
            {
                "version": [4, 3, 2],
                "timestamp": [
                    instant + timedelta(seconds=1),
                    instant,
                    instant - timedelta(seconds=1),
                ],
            }
        )
        return frame.head(int(sql.rsplit(" LIMIT ", 1)[1]))

    fake = FakeConnection(respond)
    pinned, references = source._pin_tables(fake, snapshot_schema())
    assert pinned == instant
    assert references == {
        name: f"`sales`.`public`.`{name}` VERSION AS OF 3"
        for name in ("users", "orders")
    }
    assert source._quote_identifier("name%`") == "`name%%```"
    history_calls = [sql for sql, _ in fake.calls if "DESCRIBE HISTORY" in sql]
    assert [int(sql.rsplit(" LIMIT ", 1)[1]) for sql in history_calls] == [1, 16, 1, 16]


@pytest.mark.parametrize(
    "history",
    [
        {
            "version": ["3;DROP TABLE users"],
            "timestamp": [datetime(2026, 1, 1, tzinfo=UTC)],
        },
        {"version": [True], "timestamp": [datetime(2026, 1, 1, tzinfo=UTC)]},
        {"version": [-1], "timestamp": [datetime(2026, 1, 1, tzinfo=UTC)]},
        {"version": [3], "timestamp": [datetime(2026, 2, 1, tzinfo=UTC)]},
        {"version": [], "timestamp": []},
    ],
)
def test_databricks_refuses_invalid_versions_and_missing_history(monkeypatch, history):
    """Version literals cannot carry SQL and missing retained snapshots fail."""
    source = databricks_connector(monkeypatch)
    fake = FakeConnection(
        lambda sql, params: (
            pd.DataFrame([[datetime(2026, 1, 1, tzinfo=UTC)]], columns=["timestamp"])
            if "CURRENT_TIMESTAMP" in sql
            else pd.DataFrame(history)
        )
    )
    with pytest.raises(ConnectorError):
        source._pin_tables(fake, snapshot_schema())


@pytest.mark.parametrize(
    "table_type,source_format",
    [("VIEW", "DELTA"), ("MANAGED", "PARQUET"), ("FOREIGN", "DELTA")],
)
def test_databricks_discovery_rejects_non_delta_and_views(
    monkeypatch, table_type, source_format
):
    """Unsupported relations fail before key discovery or source reads."""
    source = databricks_connector(monkeypatch, tables=["users"])
    fake = FakeConnection(
        lambda sql, params: pd.DataFrame(
            {
                "table_name": ["users"],
                "table_type": [table_type],
                "data_source_format": [source_format],
            }
        )
    )
    with pytest.raises(ConnectorError, match="Unity Catalog Delta"):
        source._discover_schema(fake, "test")


def test_databricks_discovery_retains_composite_keys_and_registered_time(monkeypatch):
    """Declared compound relationships retain their ordinal order."""
    source = databricks_connector(monkeypatch, tables=["users", "orders"])
    source.config.time_columns = {"orders": "event_time"}
    columns = []
    for table, names in {
        "users": ["tenant_id", "user_id"],
        "orders": ["order_id", "user_tenant", "user_id", "event_time"],
    }.items():
        for ordinal, name in enumerate(names, start=1):
            columns.append(
                {
                    "table_name": table,
                    "column_name": name,
                    "data_type": "TIMESTAMP" if name == "event_time" else "BIGINT",
                    "is_nullable": "NO",
                    "ordinal_position": ordinal,
                }
            )
    primary = [
        {"table_name": "users", "column_name": "user_id", "key_sequence": 2},
        {"table_name": "users", "column_name": "tenant_id", "key_sequence": 1},
        {"table_name": "orders", "column_name": "order_id", "key_sequence": 1},
    ]
    foreign = [
        {
            "table_name": "orders",
            "constraint_name": "user_fk",
            "column_name": "user_id",
            "referred_table": "users",
            "referred_column": "user_id",
            "key_sequence": 2,
        },
        {
            "table_name": "orders",
            "constraint_name": "user_fk",
            "column_name": "user_tenant",
            "referred_table": "users",
            "referred_column": "tenant_id",
            "key_sequence": 1,
        },
    ]

    def respond(sql, params):
        if "data_source_format" in sql:
            return pd.DataFrame(
                [
                    {
                        "table_name": name,
                        "table_type": "MANAGED",
                        "data_source_format": "DELTA",
                    }
                    for name in ("users", "orders")
                ]
            )
        if "referential_constraints" in sql:
            return pd.DataFrame(foreign)
        if "table_constraints c" in sql:
            return pd.DataFrame(primary)
        return pd.DataFrame(columns)

    fake = FakeConnection(respond)
    schema = source._discover_schema(fake, "test")
    assert schema.tables[0].primary_key == ["tenant_id", "user_id"]
    assert schema.tables[1].foreign_keys[0].columns == ["user_tenant", "user_id"]
    assert schema.tables[1].foreign_keys[0].referred_columns == ["tenant_id", "user_id"]
    assert schema.tables[1].time_column == "event_time"
    assert schema.capabilities.snapshot_mode == "table_versions"
    assert schema.capabilities.read_only_mode == "role"
    assert not schema.capabilities.server_side_cursor
    assert all(cursor.closed for cursor in fake.cursors)
    assert all("%(schema)s" not in sql for sql, _ in fake.calls)
    assert any(params == ["public"] for _, params in fake.calls)


def test_databricks_native_parameters_preserve_exact_numbers(monkeypatch):
    """Native binds retain decimal scale and IDs that do not fit signed BIGINT."""
    source = databricks_connector(monkeypatch)
    amount = Decimal("21.25")
    entity_id = 2**70
    fake = FakeConnection(
        lambda sql, params: pd.DataFrame(
            [[amount, entity_id]], columns=["amount", "entity_id"], dtype=object
        )
    )
    rows = source._get_query_frame(
        fake,
        "SELECT %(amount)s AS amount,%(entity_id)s AS entity_id",
        {"amount": amount, "entity_id": entity_id},
    )
    statement, parameters = fake.calls[0]
    assert "CAST(? AS DECIMAL(4, 2))" in statement
    assert "CAST(? AS DECIMAL(22, 0))" in statement
    assert parameters == ["21.25", str(entity_id)]
    assert rows.iloc[0, 0] == amount
    assert rows.iloc[0, 1] == entity_id
    assert fake.cursors[0].closed


@pytest.mark.parametrize(
    "options,frame,error_match",
    [
        ({"max_rows": 1}, pd.DataFrame({"id": [1, 2]}), "row limit"),
        (
            {"max_bytes": 100},
            pd.DataFrame({"payload": ["x" * 200]}),
            "memory limit",
        ),
    ],
)
def test_databricks_queries_enforce_row_and_memory_bounds(
    monkeypatch, options, frame, error_match
):
    """Excess rows or bytes fail explicitly and release the result cursor."""
    source = databricks_connector(monkeypatch)
    fake = FakeConnection(lambda sql, params: frame)
    with pytest.raises(SnapshotLimitError, match=error_match):
        source._get_query_frame(fake, "SELECT * FROM users", **options)
    assert fake.cursors[0].closed


@pytest.mark.parametrize(
    "error_code,sql_state,expected_diagnostic",
    [(604, "57014", "code=604, sqlstate=57014"), ("private-token", "secret", None)],
)
def test_databricks_query_errors_keep_only_valid_vendor_codes(
    monkeypatch, error_code, sql_state, expected_diagnostic
):
    """Vendor diagnostics survive while messages and invalid fields stay hidden."""
    source = databricks_connector(monkeypatch)

    def fail(sql, params):
        error = RuntimeError("private-token")
        error.errno = error_code
        error.sqlstate = sql_state
        raise error

    fake = FakeConnection(fail)
    with pytest.raises(ConnectorError) as error:
        source._get_query_frame(fake, "SELECT 1")
    assert "private-token" not in str(error.value)
    assert error.value.__suppress_context__
    if expected_diagnostic is None:
        assert str(error.value) == "warehouse query failed"
    else:
        assert expected_diagnostic in str(error.value)
    assert fake.cursors[0].closed


def test_databricks_query_timeout_cancels_driver_and_closes_cursor(monkeypatch):
    """A hung generated read requests cancellation before releasing its cursor."""
    source = databricks_connector(monkeypatch, statement_timeout_ms=10)
    cancelled = Event()

    class HangingCursor(FakeCursor):
        """Wait for driver cancellation without hitting a real warehouse."""

        def execute(self, sql, params=None):
            """Block until the query deadline invokes cancel."""
            assert cancelled.wait(1)
            raise RuntimeError("private driver error")

        def cancel(self):
            """Release the query as an actual driver's cancel method would."""
            self.cancelled = True
            cancelled.set()

    fake = FakeConnection()
    cursor = HangingCursor(fake)
    fake.cursor = lambda **kwargs: cursor
    with pytest.raises(ConnectorError, match="timed out") as error:
        source._get_query_frame(fake, "SELECT 1")
    assert error.value.__suppress_context__
    assert cursor.cancelled
    assert cursor.closed


@pytest.mark.parametrize(
    "secret,diagnostic",
    [
        ({"access_token": ""}, "access_token: must be a nonempty string"),
        ({"client_id": "private-value"}, "client_secret: required"),
        ({"client_secret": "private-value"}, "client_id: required"),
        (
            {"access_token": "private-value", "client_id": "private-value"},
            "client_id: unsupported",
        ),
        (
            {"access_token": "private-value", "private-value": "private-value"},
            "unexpected credential field",
        ),
        ({"authenticator": "private-value"}, "unsupported authentication mode"),
    ],
)
def test_databricks_typed_auth_reports_safe_field_diagnostics(
    monkeypatch, secret, diagnostic
):
    """Typed modes describe missing or mixed fields without arbitrary input values."""
    source = databricks_connector(monkeypatch, secret)
    monkeypatch.setattr(source, "_get_resolved_auth_fields", lambda: secret)
    with pytest.raises(ConnectorError, match=diagnostic) as error:
        source._open_connection()
    assert "private-value" not in str(error.value)
    assert error.value.__suppress_context__


def test_databricks_history_reads_only_latest_unraced_commit(monkeypatch):
    """A stable table needs one history row regardless of retained log size."""
    source = databricks_connector(monkeypatch)
    instant = datetime(2026, 1, 1, tzinfo=UTC)

    def respond(sql, params):
        if "CURRENT_TIMESTAMP" in sql:
            return pd.DataFrame([[instant]], columns=["timestamp"])
        assert sql.endswith(" LIMIT 1")
        return pd.DataFrame({"version": [10_000], "timestamp": [instant]})

    fake = FakeConnection(respond)
    _, references = source._pin_tables(fake, snapshot_schema())
    assert len(fake.calls) == 3
    assert all(
        reference.endswith("VERSION AS OF 10000") for reference in references.values()
    )


def test_databricks_history_expands_only_raced_table_with_a_bound(monkeypatch):
    """History pages expand geometrically only when commits race the fixed pin."""
    source = databricks_connector(monkeypatch, max_rows_per_table=64)
    instant = datetime(2026, 1, 1, tzinfo=UTC)
    racing_history = pd.DataFrame(
        {
            "version": list(range(40, -1, -1)),
            "timestamp": [instant + timedelta(seconds=1)] * 40 + [instant],
        }
    )

    def respond(sql, params):
        if "CURRENT_TIMESTAMP" in sql:
            return pd.DataFrame([[instant]], columns=["timestamp"])
        limit = int(sql.rsplit(" LIMIT ", 1)[1])
        if "`users`" in sql:
            return pd.DataFrame({"version": [12], "timestamp": [instant]})
        return racing_history.head(limit)

    fake = FakeConnection(respond)
    _, references = source._pin_tables(fake, snapshot_schema())
    assert references["users"].endswith("VERSION AS OF 12")
    assert references["orders"].endswith("VERSION AS OF 0")
    histories = [sql for sql, _ in fake.calls if "DESCRIBE HISTORY" in sql]
    assert [int(sql.rsplit(" LIMIT ", 1)[1]) for sql in histories] == [1, 1, 16, 32, 64]


def test_databricks_history_refuses_to_expand_past_row_budget(monkeypatch):
    """A snapshot fails explicitly when raced history cannot fit the search bound."""
    source = databricks_connector(monkeypatch, max_rows_per_table=8)
    instant = datetime(2026, 1, 1, tzinfo=UTC)

    def respond(sql, params):
        if "CURRENT_TIMESTAMP" in sql:
            return pd.DataFrame([[instant]], columns=["timestamp"])
        limit = int(sql.rsplit(" LIMIT ", 1)[1])
        return pd.DataFrame(
            {
                "version": list(range(limit, 0, -1)),
                "timestamp": [instant + timedelta(seconds=1)] * limit,
            }
        )

    fake = FakeConnection(respond)
    with pytest.raises(SnapshotLimitError, match="history search"):
        source._pin_tables(fake, snapshot_schema())
    assert [
        int(sql.rsplit(" LIMIT ", 1)[1])
        for sql, _ in fake.calls
        if "DESCRIBE HISTORY" in sql
    ] == [1, 8]


def test_databricks_cancellation_hook_uses_query_api_only(monkeypatch):
    """A native cancellation callback does not close the connector connection."""
    source = databricks_connector(monkeypatch)
    fake = FakeConnection()
    cursor = FakeCursor(fake)
    cancelled = []
    cursor.cancel = lambda: cancelled.append(True)
    callback = source._get_query_cancellation_callback(cursor)
    assert callback is not None
    callback()
    assert cancelled == [True]
    assert not fake.closed
