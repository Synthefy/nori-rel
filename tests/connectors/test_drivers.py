"""Official-driver wiring and metadata/snapshot checks without cloud credentials."""

from __future__ import annotations

import json
import sys
import types
from datetime import UTC, datetime
from decimal import Decimal

import pandas as pd
import pytest

from nori_rel.connectors.errors import ConnectorError, SnapshotLimitError
from nori_rel.connectors.snowflake import SnowflakeConnector
from nori_rel.contracts import (
    DatabaseCreateRequest,
    EnvironmentCredential,
    ForeignKeySchema,
    SchemaGraph,
    SnowflakeConnection,
    TableSchema,
)
from nori_rel.limits import RelationalLimits


class FakeCursor:
    """Snowflake-like DBAPI cursor with bounded fetches and no cancel API."""

    def __init__(self, connection):
        """Retain the owning connection and empty result state."""
        self.connection = connection
        self.description = None
        self.rows = []
        self.closed = False

    def execute(self, sql, params=None):
        """Capture native SQL and materialize the configured fake result."""
        self.connection.calls.append((sql, params))
        frame = self.connection.respond(sql, params)
        self.description = [(str(column), None) for column in frame.columns]
        self.rows = list(frame.itertuples(index=False, name=None))
        return self

    def fetchmany(self, size=1000):
        """Return successive bounded result batches."""
        result, self.rows = self.rows[:size], self.rows[size:]
        return result

    def close(self):
        """Record release of result resources."""
        self.closed = True


class FakeConnection:
    """Small DBAPI implementation that reports every cursor and SQL call."""

    def __init__(self, respond=None):
        """Accept an optional query-dependent dataframe responder."""
        self.respond = respond or (lambda sql, params: pd.DataFrame())
        self.calls = []
        self.cursors = []
        self.cursor_options = []
        self.closed = False

    def cursor(self, **kwargs):
        """Create an observable cursor."""
        cursor = FakeCursor(self)
        self.cursors.append(cursor)
        self.cursor_options.append(kwargs)
        return cursor

    def close(self):
        """Record connection release."""
        self.closed = True


def snowflake_connector(monkeypatch, secret, *, tables=None, **limits):
    """Build a Snowflake connector resolving credential JSON from an env ref."""
    monkeypatch.setenv("NORI_TEST_DRIVER_SECRET", json.dumps(secret))
    settings = SnowflakeConnection(
        account="organization-account",
        warehouse="compute",
        database="sales",
        role="reader",
    )
    config = DatabaseCreateRequest(
        engine="snowflake",
        credential=EnvironmentCredential(variable="NORI_TEST_DRIVER_SECRET"),
        connection=settings,
        schema_name="public",
        tables=tables,
    )
    return SnowflakeConnector(config, RelationalLimits(**limits))


def install_snowflake_driver(monkeypatch, connection):
    """Install an importable fake Snowflake driver and collect connect kwargs."""
    package = types.ModuleType("snowflake")
    package.__path__ = []
    driver = types.ModuleType("snowflake.connector")
    calls = []

    def connect(**kwargs):
        calls.append(kwargs)
        return connection

    driver.connect = connect
    package.connector = driver
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, driver.__name__, driver)
    return calls


def test_snowflake_oauth_uses_native_binding_tls_and_bounded_session(monkeypatch):
    """Verify only approved connection kwargs reach the Snowflake driver."""
    source = snowflake_connector(
        monkeypatch, {"authenticator": "oauth", "token": "private-token"}
    )
    fake = FakeConnection()
    calls = install_snowflake_driver(monkeypatch, fake)
    assert source._open_connection() is fake
    source._configure_connection(fake)
    assert calls[0]["paramstyle"] == "qmark"
    assert calls[0]["disable_ocsp_checks"] is False
    assert calls[0]["client_prefetch_threads"] == 1
    assert calls[0]["client_fetch_use_mp"] is False
    assert calls[0]["session_parameters"]["TIMEZONE"] == "UTC"
    assert calls[0]["session_parameters"]["STATEMENT_TIMEOUT_IN_SECONDS"] == 300
    assert fake.calls[0][0] == "USE SECONDARY ROLES NONE"
    assert fake.cursors[0].closed


def test_snowflake_key_file_auth(monkeypatch):
    """Key files use JWT authentication without enabling client-side binding."""
    source = snowflake_connector(
        monkeypatch,
        {"user": "service", "private_key_file": "/tmp/test-key.pem"},
    )
    calls = install_snowflake_driver(monkeypatch, FakeConnection())
    source._open_connection()
    assert calls[0]["authenticator"] == "SNOWFLAKE_JWT"
    assert calls[0]["private_key_file"] == "/tmp/test-key.pem"


def test_snowflake_encrypted_pem_is_decoded_before_driver_connection(monkeypatch):
    """Encrypted PEM becomes the official connector's DER PKCS8 key bytes."""
    serialization = pytest.importorskip("cryptography.hazmat.primitives.serialization")
    rsa = pytest.importorskip("cryptography.hazmat.primitives.asymmetric.rsa")
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = private_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.BestAvailableEncryption(b"private-passphrase"),
    )
    source = snowflake_connector(
        monkeypatch,
        {
            "user": "service",
            "private_key": pem.decode(),
            "private_key_password": "private-passphrase",
        },
    )
    calls = install_snowflake_driver(monkeypatch, FakeConnection())
    source._open_connection()
    assert isinstance(calls[0]["private_key"], bytes)
    assert "private_key_password" not in calls[0]
    decoded = serialization.load_der_private_key(calls[0]["private_key"], password=None)
    assert decoded.private_numbers() == private_key.private_numbers()


@pytest.mark.parametrize(
    "secret",
    [
        {"authenticator": "oauth", "token": "secret", "insecure_mode": "true"},
        {"authenticator": "oauth", "token": "secret", "private_key_file": "key.pem"},
        {
            "authenticator": "OAUTH_CLIENT_CREDENTIALS",
            "oauth_client_id": "id",
            "oauth_client_secret": "secret",
            "oauth_token_request_url": "http://identity.example/token",
        },
    ],
)
def test_snowflake_rejects_tls_overrides_mixed_auth_and_insecure_oauth(
    monkeypatch, secret
):
    """Unsafe or ambiguous credentials fail before any network operation."""
    source = snowflake_connector(monkeypatch, secret)
    with pytest.raises(ConnectorError) as error:
        source._open_connection()
    assert "secret" not in str(error.value)


def test_driver_connection_errors_are_sanitized(monkeypatch):
    """Official driver exceptions cannot expose tokens through exception chaining."""
    source = snowflake_connector(monkeypatch, {"token": "private-token"})
    install_snowflake_driver(monkeypatch, FakeConnection())

    def fail(**kwargs):
        raise RuntimeError("private-token")

    monkeypatch.setattr(
        sys.modules["snowflake.connector"],
        "connect",
        fail,
    )
    with pytest.raises(ConnectorError) as error:
        source._open_connection()
    assert "private-token" not in str(error.value)
    assert error.value.__suppress_context__


def test_session_failure_releases_connection_without_disclosing_driver_error(
    monkeypatch,
):
    """Failed session initialization cannot leak secrets or retain the connection."""
    source = snowflake_connector(monkeypatch, {"token": "token"})

    def fail(sql, params):
        raise RuntimeError("private-token")

    fake = FakeConnection(fail)
    install_snowflake_driver(monkeypatch, fake)
    with pytest.raises(ConnectorError) as error:
        source._get_connection()
    assert "private-token" not in str(error.value)
    assert error.value.__suppress_context__
    assert fake.closed
    assert fake.cursors[0].closed


def snapshot_schema():
    """Use two selected tables to check one shared extraction timestamp."""
    return SchemaGraph(
        database_id="test",
        schema_name="public",
        tables=[
            TableSchema(name="users", columns=[]),
            TableSchema(name="orders", columns=[]),
        ],
        discovered_at=datetime.now(UTC),
    )


def test_snowflake_pins_one_server_timestamp_and_quotes_names(monkeypatch):
    """Every table references the same fixed server timestamp with explicit TZ."""
    source = snowflake_connector(monkeypatch, {"token": "token"})
    instant = datetime(2026, 1, 1, tzinfo=UTC)
    fake = FakeConnection(
        lambda sql, params: pd.DataFrame([[instant]], columns=["CURRENT_TIMESTAMP()"])
    )
    pinned, references = source._pin_tables(fake, snapshot_schema())
    assert pinned == instant
    assert len(fake.calls) == 1
    for name, reference in references.items():
        assert (
            reference == f'"sales"."public"."{name}" AT (TIMESTAMP => '
            "'2026-01-01T00:00:00.000000+00:00'::TIMESTAMP_TZ)"
        )
    assert source._quote_identifier('name%"') == '"name%%"""'


def test_snowflake_native_parameters_preserve_decimals_large_ids_and_time(monkeypatch):
    """Native binds retain exact numbers and explicitly typed aware timestamps."""
    source = snowflake_connector(monkeypatch, {"token": "token"})
    amount = Decimal("21.25")
    entity_id = 2**70
    anchor = datetime(2026, 1, 1, tzinfo=UTC)
    fake = FakeConnection(
        lambda sql, params: pd.DataFrame(
            [[amount, entity_id, anchor]],
            columns=["amount", "entity_id", "anchor_time"],
            dtype=object,
        )
    )
    rows = source._get_query_frame(
        fake,
        "SELECT %(amount)s AS amount,%(entity_id)s AS entity_id,"
        "%(anchor)s AS anchor_time",
        {"amount": amount, "entity_id": entity_id, "anchor": anchor},
    )
    statement, parameters = fake.calls[0]
    assert "CAST(? AS DECIMAL(4, 2))" in statement
    assert "CAST(? AS DECIMAL(22, 0))" in statement
    assert parameters == ["21.25", str(entity_id), ("TIMESTAMP_TZ", anchor)]
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
def test_snowflake_queries_enforce_row_and_memory_bounds(
    monkeypatch, options, frame, error_match
):
    """Excess rows or bytes fail explicitly and release the result cursor."""
    source = snowflake_connector(monkeypatch, {"token": "token"})
    fake = FakeConnection(lambda sql, params: frame)
    with pytest.raises(SnapshotLimitError, match=error_match):
        source._get_query_frame(fake, "SELECT * FROM users", **options)
    assert fake.cursors[0].closed


@pytest.mark.parametrize(
    "error_code,sql_state,expected_diagnostic",
    [(604, "57014", "code=604, sqlstate=57014"), ("private-token", "secret", None)],
)
def test_snowflake_query_errors_keep_only_valid_vendor_codes(
    monkeypatch, error_code, sql_state, expected_diagnostic
):
    """Vendor diagnostics survive while messages and invalid fields stay hidden."""
    source = snowflake_connector(monkeypatch, {"token": "token"})

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


def test_discovery_retains_composite_key_order_and_registered_time(monkeypatch):
    """Declared compound relationships retain their ordinal order."""
    source = snowflake_connector(
        monkeypatch, {"token": "token"}, tables=["users", "orders"]
    )
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
        if "SHOW TABLES" in sql:
            return pd.DataFrame(
                [
                    {
                        "name": name,
                        "kind": "TABLE",
                        "retention_time": 1,
                        "is_hybrid": "N",
                        "is_iceberg": "N",
                    }
                    for name in ("users", "orders")
                ]
            )
        if "SHOW IMPORTED" in sql:
            return pd.DataFrame(
                [
                    {
                        "fk_table_name": row["table_name"],
                        "fk_name": row["constraint_name"],
                        "fk_column_name": row["column_name"],
                        "pk_table_name": row["referred_table"],
                        "pk_column_name": row["referred_column"],
                        "key_sequence": row["key_sequence"],
                        "pk_database_name": "sales",
                        "pk_schema_name": "public",
                    }
                    for row in foreign
                ]
            )
        if "SHOW PRIMARY" in sql:
            return pd.DataFrame(primary)
        return pd.DataFrame(columns)

    fake = FakeConnection(respond)
    schema = source._discover_schema(fake, "test")
    assert schema.tables[0].primary_key == ["tenant_id", "user_id"]
    assert schema.tables[1].foreign_keys[0].columns == ["user_tenant", "user_id"]
    assert schema.tables[1].foreign_keys[0].referred_columns == ["tenant_id", "user_id"]
    assert schema.tables[1].time_column == "event_time"
    assert schema.capabilities.read_only_mode == "role"
    assert not schema.capabilities.server_side_cursor
    assert all(cursor.closed for cursor in fake.cursors)
    assert all("%(schema)s" not in sql for sql, _ in fake.calls)
    assert any(params == ["public"] for _, params in fake.calls)


@pytest.mark.parametrize(
    "flag", ["is_hybrid", "is_iceberg", "is_dynamic", "is_external"]
)
def test_snowflake_discovery_rejects_nonstandard_tables(monkeypatch, flag):
    """Hybrid, Iceberg, dynamic, and external tables are outside the first scope."""
    source = snowflake_connector(monkeypatch, {"token": "token"}, tables=["users"])
    fake = FakeConnection(
        lambda sql, params: pd.DataFrame(
            [{"name": "users", "kind": "TABLE", "retention_time": 1, flag: "Y"}]
        )
    )
    with pytest.raises(ConnectorError, match="standard tables"):
        source._discover_schema(fake, "test")


def test_snowflake_timeout_relies_on_server_without_closing_shared_connection(
    monkeypatch,
):
    """Snowflake has no cursor.cancel; configured server timeout owns cancellation."""
    source = snowflake_connector(
        monkeypatch, {"token": "token"}, statement_timeout_ms=10
    )

    def fail_timer(*args, **kwargs):
        raise AssertionError(
            "Snowflake must not schedule unsupported cursor cancellation"
        )

    monkeypatch.setattr("nori_rel.connectors.warehouse.Timer", fail_timer)

    class ServerTimeoutCursor(FakeCursor):
        """Simulate the official connector's server statement-timeout exception."""

        def execute(self, sql, params=None):
            error = RuntimeError("private server diagnostic")
            error.errno = 604
            error.sqlstate = "57014"
            raise error

    fake = FakeConnection()
    cursor = ServerTimeoutCursor(fake)
    fake.cursor = lambda **kwargs: cursor
    assert not hasattr(cursor, "cancel")
    assert source._get_query_cancellation_callback(cursor) is None
    with pytest.raises(ConnectorError, match="code=604, sqlstate=57014") as error:
        source._get_query_frame(fake, "SELECT 1")
    assert "private server" not in str(error.value)
    assert error.value.__suppress_context__
    assert cursor.closed
    assert not fake.closed


def test_driver_error_codes_survive_without_exposing_secret_messages(monkeypatch):
    class PrivateDriverError(Exception):
        errno = 2003
        sqlstate = "42S02"

    def fail_query(sql, params):
        raise PrivateDriverError("private-value must not appear in the public error")

    source = snowflake_connector(monkeypatch, {"token": "private-value"})
    with pytest.raises(ConnectorError) as error:
        source._get_query_frame(FakeConnection(fail_query), "SELECT 1")
    assert str(error.value) == "warehouse query failed (code=2003, sqlstate=42S02)"
    assert error.value.__suppress_context__


@pytest.mark.parametrize(
    "secret,diagnostic",
    [
        ({"authenticator": "oauth"}, "token: required"),
        ({"token": ""}, "token: must be a nonempty string"),
        (
            {
                "user": "service",
                "private_key_file": "key.pem",
                "private_key": "private-value",
            },
            "exactly one",
        ),
        ({"private_key_file": "key.pem"}, "user: required"),
        (
            {
                "user": "service",
                "private_key_file": "key.pem",
                "private_key_password": "private-value",
            },
            "private_key_password requires private_key",
        ),
        ({"user": "service", "private_key": "private-value"}, "private_key PEM"),
        (
            {
                "authenticator": "OAUTH_CLIENT_CREDENTIALS",
                "oauth_client_id": "private-value",
            },
            "oauth_client_secret: required",
        ),
        (
            {"authenticator": "private-value", "token": "private-value"},
            "unsupported authentication mode",
        ),
        (
            {"token": "private-value", "private-value": "private-value"},
            "unexpected credential field",
        ),
    ],
)
def test_snowflake_typed_auth_reports_safe_field_diagnostics(
    monkeypatch, secret, diagnostic
):
    """Missing, invalid, and mixed fields are precise without printing credentials."""
    source = snowflake_connector(monkeypatch, secret)
    # Validate the union independently of JSON-shape checks in the resolver.
    monkeypatch.setattr(source, "_get_resolved_auth_fields", lambda: secret)
    with pytest.raises(ConnectorError, match=diagnostic) as error:
        source._open_connection()
    assert "private-value" not in str(error.value)
    assert error.value.__suppress_context__


def test_snowflake_oauth_client_credentials_and_key_file_password(monkeypatch):
    """Both documented client-credential and encrypted-file fields reach the driver."""
    source = snowflake_connector(
        monkeypatch,
        {
            "authenticator": "OAUTH_CLIENT_CREDENTIALS",
            "oauth_client_id": "id",
            "oauth_client_secret": "private-value",
            "oauth_token_request_url": "https://identity.example/token",
            "oauth_scope": "session:role:reader",
        },
    )
    calls = install_snowflake_driver(monkeypatch, FakeConnection())
    source._open_connection()
    assert calls[0]["oauth_client_secret"] == "private-value"
    assert calls[0]["oauth_token_request_url"] == "https://identity.example/token"
    source = snowflake_connector(
        monkeypatch,
        {
            "user": "service",
            "private_key_file": "key.pem",
            "private_key_file_pwd": "private-value",
        },
    )
    source._open_connection()
    assert calls[1]["private_key_file_pwd"] == "private-value"


def _snowflake_identifier_metadata(table_columns, *, counts=None):
    """Build a fake standard-table metadata responder with exact physical names."""
    columns = [
        {
            "table_name": table,
            "column_name": column,
            "data_type": "TIMESTAMP_NTZ" if "TIME" in column.upper() else "BIGINT",
            "is_nullable": "NO",
            "ordinal_position": ordinal,
        }
        for table, names in table_columns.items()
        for ordinal, column in enumerate(names, 1)
    ]

    def respond(sql, params):
        if "SHOW TABLES" in sql:
            return pd.DataFrame(
                [
                    {
                        "name": name,
                        "kind": "TABLE",
                        "retention_time": 1,
                        "rows": (counts or {}).get(name),
                    }
                    for name in table_columns
                ],
                dtype=object,
            )
        if "SHOW PRIMARY" in sql or "SHOW IMPORTED" in sql:
            return pd.DataFrame()
        return pd.DataFrame(columns)

    return FakeConnection(respond)


def test_snowflake_resolves_configured_table_time_and_relation_names(monkeypatch):
    """Lowercase portable mappings resolve once to uppercase physical names."""
    source = snowflake_connector(
        monkeypatch, {"token": "token"}, tables=["users", "orders"]
    )
    source.config.time_columns = {"orders": "event_time"}
    source.config.primary_keys = {"users": ["user_id"], "orders": ["order_id"]}
    source.config.foreign_keys = {
        "orders": [
            ForeignKeySchema(
                columns=["user_id"],
                referred_table="users",
                referred_columns=["user_id"],
            )
        ]
    }
    fake = _snowflake_identifier_metadata(
        {
            "USERS": ["USER_ID"],
            "ORDERS": ["ORDER_ID", "USER_ID", "EVENT_TIME", "INGEST_TIME"],
        },
        counts={"USERS": 10, "ORDERS": 20},
    )
    schema = source._discover_schema(
        fake, "test", time_columns={"orders": "ingest_time"}
    )
    assert [table.name for table in schema.tables] == ["USERS", "ORDERS"]
    assert schema.tables[1].time_column == "INGEST_TIME"
    assert schema.capabilities.unquoted_identifier_case == "upper"
    assert source.config.time_columns == {"ORDERS": "EVENT_TIME"}
    assert source.config.primary_keys == {"USERS": ["USER_ID"], "ORDERS": ["ORDER_ID"]}
    relation = source.config.foreign_keys["ORDERS"][0]
    assert relation.columns == ["USER_ID"]
    assert relation.referred_table == "USERS"
    assert relation.referred_columns == ["USER_ID"]
    assert source._get_table_row_counts(fake, schema) == {"USERS": 10, "ORDERS": 20}


def test_snowflake_prefers_exact_quoted_names(monkeypatch):
    """Exact lowercase and camel-case names survive beside uppercase names."""
    source = snowflake_connector(
        monkeypatch, {"token": "token"}, tables=["users", "orderItems"]
    )
    source.config.primary_keys = {"users": ["userId"], "orderItems": ["orderId"]}
    fake = _snowflake_identifier_metadata(
        {"users": ["userId", "USERID"], "USERS": ["USER_ID"], "orderItems": ["orderId"]}
    )
    schema = source._discover_schema(fake, "test")
    assert [table.name for table in schema.tables] == ["users", "orderItems"]
    assert source.config.primary_keys["users"] == ["userId"]
    assert (
        source._get_qualified_table_reference("orderItems")
        == '"sales"."public"."orderItems"'
    )


@pytest.mark.parametrize("kind", ["table", "column"])
def test_snowflake_rejects_ambiguous_case_insensitive_mapping(monkeypatch, kind):
    """Uppercase-equivalent matching is refused when there is no exact winner."""
    source = snowflake_connector(monkeypatch, {"token": "token"}, tables=["UsErS"])
    if kind == "table":
        fake = _snowflake_identifier_metadata({"users": ["id"], "USERS": ["ID"]})
    else:
        source.config.tables = ["users"]
        source.config.primary_keys = {"users": ["UsErId"]}
        fake = _snowflake_identifier_metadata({"users": ["userId", "USERID"]})
    with pytest.raises(ConnectorError, match=f"ambiguous Snowflake {kind}"):
        source._discover_schema(fake, "test")
