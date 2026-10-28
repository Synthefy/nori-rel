"""MySQL 8 against the shared label fixture, plus MySQL-only types and sessions.

Tests run when NORI_REL_TEST_MYSQL_URL names a disposable MySQL 8 server whose
user can create databases, e.g. mysql://root:integration@127.0.0.1:3306/mysql.
Each test creates and drops its own database. NORI_REL_TEST_MYSQL_CA, the
server's CA certificate, also enables the certificate-verification tests.
"""

import os
from datetime import UTC, datetime

import pandas as pd
import pytest
from sqlalchemy.exc import OperationalError

from nori_rel import NoriRelClient, RdsIamToken, RelationalLimits
from nori_rel._credentials import CredentialResolver
from nori_rel.connectors.errors import ConnectorError, SnapshotLimitError

from .conftest import (
    JANUARY_ANCHOR,
    LARGE_ENTITY_ID,
    SCORING_CUTOFF,
    connect_mysql_writer,
    create_mysql_source,
)
from .test_warehouse_live import CASES, _expected_labels, _predict, _prepare

pytestmark = pytest.mark.skipif(
    not os.environ.get("NORI_REL_TEST_MYSQL_URL"),
    reason="MySQL integration database is supplied by CI",
)


@pytest.mark.parity
@pytest.mark.parametrize("operation,expression", CASES)
def test_mysql_shared_fixture_labels(mysql_parity_source, operation, expression):
    """MySQL labels match the expectations every other engine is held to."""
    snapshot, context, counts = _prepare(mysql_parity_source, expression)
    expected = _expected_labels(operation)
    assert context["entity_id"].tolist() == sorted(expected)
    assert context["context_row_id"].tolist() == sorted(expected)
    assert context["anchor_time"].tolist() == [JANUARY_ANCHOR] * len(expected)
    assert context["label"].tolist() == pytest.approx(list(expected.values()))
    assert counts == {
        "candidate_rows": 6,
        "usable_rows": len(expected),
        "dropped_unknown_labels": 6 - len(expected),
    }
    if operation == "binary":
        assert {type(label) for label in context["label"]} == {bool}
    assert snapshot.tables["orders"]["event_time"].max() <= SCORING_CUTOFF
    # BIGINT identities must never pass through a floating-point intermediate.
    assert LARGE_ENTITY_ID in snapshot.tables["users"]["user_id"].tolist()


def _with_url_options(**options) -> str:
    """Return the test server's URL with these query options set."""
    from sqlalchemy.engine import make_url

    url = make_url(os.environ["NORI_REL_TEST_MYSQL_URL"])
    return url.set(query=options).render_as_string(hide_password=False)


def test_mysql_discovery_reports_keys_types_and_a_transaction_snapshot(
    mysql_parity_source,
):
    connector = mysql_parity_source.connector
    assert connector.test() >= 0
    schema = connector.discover("db")
    tables = {table.name: table for table in schema.tables}
    assert [
        (column.name, column.data_type, column.nullable)
        for column in tables["users"].columns
    ] == [
        ("user_id", "BIGINT", False),
        ("created_at", "DATETIME", True),
        ("country", "VARCHAR(8)", True),
    ]
    assert tables["users"].primary_key == ["user_id"]
    assert [
        (key.columns, key.referred_table, key.referred_columns)
        for key in tables["orders"].foreign_keys
    ] == [(["user_id"], "users", ["user_id"])]
    assert tables["orders"].time_column == "event_time"
    assert schema.capabilities.snapshot_mode == "transaction"


TYPED_TABLE = (
    "CREATE TABLE {table} (id BIGINT UNSIGNED PRIMARY KEY, flag BOOLEAN, "
    "small MEDIUMINT UNSIGNED, price DECIMAL(12, 2), "
    "code VARCHAR(16) CHARACTER SET latin1 COLLATE latin1_bin, "
    "kind ENUM('a', 'b'), note MEDIUMTEXT, seen DATETIME(6), "
    "stamped TIMESTAMP(6) NULL, day DATE, valid_to DATE, payload JSON)"
)


def test_mysql_types_reach_features_with_utc_times_and_exact_keys():
    """TIMESTAMP converts through the session zone; DATETIME is read as UTC."""
    with create_mysql_source(tables=["typed"], time_columns={}) as source:
        source.execute_write(TYPED_TABLE.format(table=source.reference("typed")))
        # A writer five hours east stores TIMESTAMP 05:00 as 00:00 UTC, and
        # DATETIME as the literal it was given.
        source.execute_write("SET time_zone = '+05:00'")
        source.execute_write(
            f"INSERT INTO {source.reference('typed')} VALUES "
            "(18446744073709551615, TRUE, 16777215, 12.50, 'x', 'a', 'n', "
            "'2026-01-01 00:00:00', '2026-01-01 05:00:00', '2026-01-02', "
            "'9999-12-31', '{\"k\": 1}'), "
            "(1, FALSE, 0, NULL, NULL, 'b', NULL, NULL, NULL, NULL, NULL, NULL)"
        )
        source.execute_write("SET time_zone = '+00:00'")
        schema = source.connector.discover("db")
        typed = schema.tables[0]
        assert {column.name: column.data_type for column in typed.columns} == {
            "id": "BIGINT UNSIGNED",
            "flag": "BOOLEAN",
            "small": "MEDIUMINT UNSIGNED",
            "price": "DECIMAL(12, 2)",
            "code": "VARCHAR(16)",
            "kind": "ENUM",
            "note": "MEDIUMTEXT",
            "seen": "DATETIME",
            "stamped": "TIMESTAMP",
            "day": "DATE",
            "valid_to": "DATE",
            "payload": "JSON",
        }
        frame = (
            source.connector.snapshot("db", SCORING_CUTOFF)
            .tables["typed"]
            .sort_values("small", ignore_index=True)
        )
    midnight = pd.Timestamp("2026-01-01", tz="UTC")
    assert frame["seen"].tolist()[1] == frame["stamped"].tolist()[1] == midnight
    assert frame["day"].tolist()[1] == pd.Timestamp("2026-01-02", tz="UTC")
    assert frame["id"].tolist() == [1, 18446744073709551615]
    assert frame["flag"].tolist() == [False, True]
    assert str(frame["flag"].dtype) == "boolean"
    assert str(frame["price"].dtype) == "Float64"
    assert frame["price"].tolist()[1] == 12.5
    # A 9999-12-31 sentinel is outside pandas' range; its column stays as read.
    assert str(frame["valid_to"].tolist()[1]) == "9999-12-31"


ACCOUNTS = (
    "CREATE TABLE {table} (account_id BIGINT PRIMARY KEY, is_premium BOOLEAN, "
    "balance DECIMAL(10, 2), tenure INT)"
)


@pytest.mark.parametrize(
    "query,expected",
    [
        (
            "PREDICT accounts.is_premium FOR EACH accounts.account_id",
            [False, True, False, True, False, True, False, True],
        ),
        (
            "PREDICT accounts.balance > 350 FOR EACH accounts.account_id",
            [False, False, False, True, True, True, True],
        ),
        (
            "PREDICT accounts.balance FOR EACH accounts.account_id",
            [100, 200, 300, 400, 500, 600, 700],
        ),
    ],
    ids=["boolean-column", "comparison", "number"],
)
def test_mysql_static_targets_read_stored_values(query, expected):
    with create_mysql_source(tables=["accounts"], time_columns={}) as source:
        source.execute_write(ACCOUNTS.format(table=source.reference("accounts")))
        source.insert_rows(
            "accounts",
            [(i, i % 2 == 0, None if i == 8 else 100 * i, i) for i in range(1, 9)],
        )
        snapshot, context, counts = source.connector.prepare_inputs(
            "db",
            query,
            anchor_times=None,
            anchor_time=None,
            observed_through=None,
            entity_created_at=None,
        )
    labels = context.sort_values("entity_id")["label"].tolist()
    # Account 8 has no balance, so it is no context row for a balance target.
    assert counts["candidate_rows"] == counts["usable_rows"] == len(expected)
    if query.endswith("balance FOR EACH accounts.account_id"):
        assert labels == pytest.approx(expected)
    else:
        assert labels == expected
        assert {type(label) for label in labels} == {bool}
    # The read time is when the snapshot transaction began, in UTC.
    assert snapshot.as_of.tzinfo is not None
    assert (datetime.now(UTC) - snapshot.as_of).total_seconds() < 300


def test_mysql_month_windows_add_calendar_months(mysql_parity_source):
    expression = "COUNT(orders.* WHERE orders.channel='web',0,1,months)"
    _, context, _ = _prepare(mysql_parity_source, expression)
    # (Jan 1, Feb 1] holds the five web rows of (0, 30, days), which ends at
    # Jan 31 00:00, and the 00:00:01 row after it.
    expected = {**dict.fromkeys(range(1, 5), 6), 5: 0, 6: 1}
    assert dict(zip(context["entity_id"], context["label"], strict=True)) == expected


def test_mysql_sessions_are_utc_bounded_and_read_only(mysql_parity_source):
    connector = mysql_parity_source.connector
    engine = connector._engine()
    with connector._read_only_transaction(engine) as connection:
        settings = connection.exec_driver_sql(
            "SELECT @@session.time_zone, @@session.max_execution_time, "
            "@@session.transaction_read_only, @@session.transaction_isolation, "
            "@@session.div_precision_increment",
            {},
        ).one()
        assert tuple(settings) == (
            "+00:00",
            connector.settings.statement_timeout_ms,
            1,
            "REPEATABLE-READ",
            30,
        )
        with pytest.raises(OperationalError) as refused:
            connection.exec_driver_sql(
                f"INSERT INTO {mysql_parity_source.reference('users')} "
                "VALUES (123, NULL, 'XX')",
                {},
            )
    assert refused.value.orig.args[0] == 1792


def test_mysql_labels_and_tables_come_from_one_snapshot(
    mysql_parity_source, monkeypatch
):
    """A write after the snapshot starts reaches neither labels nor tables."""
    source = mysql_parity_source
    snapshot_method = type(source.connector)._snapshot

    def write_then_snapshot(self, connection, *args, **kwargs):
        source.execute_write(
            f"UPDATE {source.reference('orders')} SET "
            f"{source.identifier('amount')} = -1000"
        )
        return snapshot_method(self, connection, *args, **kwargs)

    monkeypatch.setattr(type(source.connector), "_snapshot", write_then_snapshot)
    snapshot, context, _ = _prepare(source, CASES[3][1])
    assert context["label"].tolist() == list(_expected_labels("sum").values())
    assert -1000 not in snapshot.tables["orders"]["amount"].dropna().tolist()


@pytest.mark.parametrize(
    "limits,message",
    [
        (RelationalLimits(max_rows_per_table=10), "row limit"),
        (RelationalLimits(max_total_rows=20), "row limit"),
        (RelationalLimits(max_snapshot_bytes=100), "memory limit"),
    ],
)
def test_mysql_snapshot_limits_fail_explicitly(limits, message):
    with (
        create_mysql_source(limits=limits) as source,
        pytest.raises(SnapshotLimitError, match=message),
    ):
        source.connector.snapshot("db", SCORING_CUTOFF)


def test_mysql_null_foreign_key_keeps_labels_and_large_ids(mysql_parity_source):
    source = mysql_parity_source
    source.insert_rows("orders", [(90001, None, JANUARY_ANCHOR, 42, "web")])
    snapshot, context, _ = _prepare(source, CASES[3][1])
    assert context["label"].tolist() == list(_expected_labels("sum").values())
    identifiers = snapshot.tables["orders"]["user_id"]
    assert identifiers.isna().sum() == 1
    assert LARGE_ENTITY_ID in identifiers.dropna().tolist()


def test_mysql_scoring_filter_leaves_context_unchanged(mysql_parity_source):
    _, plain, _ = _prepare(mysql_parity_source, CASES[3][1])
    snapshot, filtered, _ = _prepare(
        mysql_parity_source, CASES[3][1], query_filter="users.country = 'CA'"
    )
    pd.testing.assert_frame_equal(plain, filtered)
    assert snapshot.scoring_filter_ids == (LARGE_ENTITY_ID,)


def test_mysql_quoted_names_survive_compilation_and_binding():
    """Dots, spaces, percent signs and backticks reach MySQL as literal names."""
    users, orders = "user %(x)s", "or.ders"
    with create_mysql_source(
        tables=[users, orders], time_columns={orders: "event time"}
    ) as source:
        quoted_users = source.reference(users)
        source.execute_write(
            f"CREATE TABLE {quoted_users} (`id%` BIGINT PRIMARY KEY, "
            "`created.at` DATETIME(6))"
        )
        source.execute_write(
            f"CREATE TABLE {source.reference(orders)} (order_id BIGINT PRIMARY KEY, "
            "buyer BIGINT, `event time` DATETIME(6), amount DOUBLE, `note``s` TEXT, "
            f"FOREIGN KEY (buyer) REFERENCES {quoted_users}(`id%`))"
        )
        # The writer sends no parameters, so "%(x)s" stays part of the name.
        source.execute_write(
            f"INSERT INTO {quoted_users} VALUES "
            "(1, '2025-01-01 00:00:00'), (2, '2025-01-01 00:00:00')"
        )
        source.execute_write(
            f"INSERT INTO {source.reference(orders)} VALUES "
            "(1, 1, '2026-01-05', 4, 'a'), (2, 1, '2026-01-06', 6, NULL), "
            "(3, 2, '2026-02-05', 9, 'b')"
        )
        snapshot, context, _ = source.connector.prepare_inputs(
            "db",
            "PREDICT SUM(`or.ders`.amount,0,30,days) FOR EACH `user %(x)s`.`id%`",
            anchor_times=[JANUARY_ANCHOR],
            anchor_time=SCORING_CUTOFF,
            observed_through=SCORING_CUTOFF,
            entity_created_at="created.at",
        )
    assert dict(zip(context["entity_id"], context["label"], strict=True)) == {
        1: 10,
        2: 0,
    }
    assert snapshot.tables[orders]["note`s"].tolist()[:2] == ["a", None]


def test_mysql_hosted_prediction_ignores_rows_after_the_cutoff(
    mysql_parity_source, monkeypatch
):
    calls = []

    class DeterministicTransport:
        def __init__(self, **options):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def predict(self, train, target, query, **options):
            calls.append((train.copy(), target.copy(), query.copy()))
            value = 0.8 if options.get("task") == "classification" else 12.5
            return pd.Series([value] * len(query))

    monkeypatch.setattr("nori_rel._runtime.SynthefyNoriClient", DeterministicTransport)
    source = mysql_parity_source
    with NoriRelClient(
        mode="hosted",
        api_key="mysql-test-only",
        limits=RelationalLimits(min_training_rows=2),
    ) as client:
        database = source.connect(client)
        assert client.test_connection(database).latency_ms >= 0
        before = _predict(client, database, CASES[3][1])
        source.execute_write(
            f"UPDATE {source.reference('orders')} "
            f"SET {source.identifier('amount')}=-999999 "
            f"WHERE {source.identifier('event_time')} > '2026-02-01'"
        )
        after = _predict(client, database, CASES[3][1])
        binary = _predict(client, database, CASES[12][1])
    assert before["predictions"]["entity_id"].tolist() == [LARGE_ENTITY_ID]
    pd.testing.assert_frame_equal(before["predictions"], after["predictions"])
    assert binary["predictions"]["probability"].tolist() == [0.8]
    assert calls[0][1].tolist() == list(_expected_labels("sum").values())
    pd.testing.assert_frame_equal(calls[0][0], calls[1][0])
    pd.testing.assert_frame_equal(calls[0][2], calls[1][2])


@pytest.mark.parity
def test_mysql_matches_postgresql_on_the_shared_fixture(
    mysql_parity_source, postgres_parity_source, monkeypatch
):
    """Same labels, and the same FastDFS training and query matrices."""
    calls = {"mysql": [], "postgresql": []}

    class RecordingTransport:
        engine = None

        def __init__(self, **options):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def predict(self, train, target, query, **options):
            calls[RecordingTransport.engine].append((train, target, query))
            return pd.Series([12.5] * len(query))

    monkeypatch.setattr("nori_rel._runtime.SynthefyNoriClient", RecordingTransport)
    contexts = {}
    for source in (mysql_parity_source, postgres_parity_source):
        RecordingTransport.engine = source.engine
        _, contexts[source.engine], _ = _prepare(source, CASES[9][1])
        with NoriRelClient(
            mode="hosted",
            api_key="mysql-test-only",
            limits=RelationalLimits(min_training_rows=2),
        ) as client:
            for _, expression in CASES[:4]:
                _predict(client, source.connect(client), expression)
    pd.testing.assert_frame_equal(
        contexts["mysql"], contexts["postgresql"], check_dtype=False
    )
    for mysql_call, postgresql_call in zip(
        calls["mysql"], calls["postgresql"], strict=True
    ):
        for left, right in zip(mysql_call, postgresql_call, strict=True):
            if isinstance(left, pd.Series):
                pd.testing.assert_series_equal(left, right, check_dtype=False)
            else:
                pd.testing.assert_frame_equal(left, right, check_dtype=False)


def test_mysql_credential_failures_report_a_code_without_secrets(monkeypatch):
    monkeypatch.setenv(
        "NORI_REL_TEST_MYSQL_WRONG_URL",
        _with_url_options().replace("://root:", "://nori_rel_missing:", 1),
    )
    with (
        create_mysql_source("NORI_REL_TEST_MYSQL_WRONG_URL") as source,
        pytest.raises(
            ConnectorError, match=r"connection failed \(code=1045\)"
        ) as caught,
    ):
        source.connector.test()
    assert "integration" not in str(caught.value)
    assert caught.value.__context__ is None


TLS_USER = "nori_rel_tls_reader"


@pytest.fixture
def tls_only_reader():
    """Create a user the server admits only over TLS, like require_secure_transport."""
    writer = connect_mysql_writer(os.environ["NORI_REL_TEST_MYSQL_URL"])
    try:
        with writer.cursor() as cursor:
            cursor.execute(f"DROP USER IF EXISTS '{TLS_USER}'@'%'")
            cursor.execute(
                f"CREATE USER '{TLS_USER}'@'%' IDENTIFIED BY 'tls-secret' REQUIRE SSL"
            )
            cursor.execute(f"GRANT SELECT ON *.* TO '{TLS_USER}'@'%'")
        yield
    finally:
        with writer.cursor() as cursor:
            cursor.execute(f"DROP USER IF EXISTS '{TLS_USER}'@'%'")
        writer.close()


def _tls_reader_url(**options) -> str:
    return _with_url_options(**options).replace(
        "://root:integration@", f"://{TLS_USER}:tls-secret@", 1
    )


@pytest.mark.parametrize(
    "options,code",
    [
        ({}, None),  # A local host defaults to PREFERRED, which uses TLS.
        ({"ssl_mode": "REQUIRED"}, None),
        ({"ssl_mode": "DISABLED"}, 1045),
    ],
    ids=["preferred", "required", "disabled"],
)
def test_mysql_tls_through_a_local_port(tls_only_reader, monkeypatch, options, code):
    """A tunnel's local end reaches a TLS-only server with the default mode."""
    monkeypatch.setenv("NORI_REL_TEST_MYSQL_TLS_URL", _tls_reader_url(**options))
    with create_mysql_source("NORI_REL_TEST_MYSQL_TLS_URL") as source:
        if code is None:
            assert source.connector.test() >= 0
            engine = source.connector._engine()
            with engine.connect() as connection:
                cipher = connection.exec_driver_sql(
                    "SHOW SESSION STATUS LIKE 'Ssl_cipher'", {}
                ).one()[1]
            assert cipher
        else:
            with pytest.raises(ConnectorError, match=f"code={code}"):
                source.connector.test()


@pytest.mark.skipif(
    not os.environ.get("NORI_REL_TEST_MYSQL_CA"),
    reason="NORI_REL_TEST_MYSQL_CA names the test server's CA certificate",
)
@pytest.mark.parametrize(
    "mode,succeeds", [("VERIFY_CA", True), ("VERIFY_IDENTITY", False)]
)
def test_mysql_certificate_checks_through_a_local_port(monkeypatch, mode, succeeds):
    """A tunnel's 127.0.0.1 never matches the server certificate's name."""
    monkeypatch.setenv(
        "NORI_REL_TEST_MYSQL_TLS_URL",
        _with_url_options(ssl_mode=mode, ssl_ca=os.environ["NORI_REL_TEST_MYSQL_CA"]),
    )
    with create_mysql_source("NORI_REL_TEST_MYSQL_TLS_URL") as source:
        if succeeds:
            assert source.connector.test() >= 0
        else:
            with pytest.raises(ConnectorError, match="code=2003"):
                source.connector.test()


IAM_USER = "nori_rel_iam_reader"
IAM_TOKEN = "stand-in-for-a-signed-token"
IAM_ENDPOINT = "shop.abc123.us-east-1.rds.amazonaws.com:3306"


@pytest.fixture
def iam_stand_in_reader():
    """Create a TLS-only user whose password stands in for an RDS IAM token.

    MySQL itself cannot run RDS's AWSAuthenticationPlugin, so this checks how
    the connector sends a token; tests/test_relational_credentials.py checks
    how it is signed.
    """
    writer = connect_mysql_writer(os.environ["NORI_REL_TEST_MYSQL_URL"])
    try:
        with writer.cursor() as cursor:
            cursor.execute(f"DROP USER IF EXISTS '{IAM_USER}'@'%'")
            cursor.execute(
                f"CREATE USER '{IAM_USER}'@'%' IDENTIFIED BY '{IAM_TOKEN}' REQUIRE SSL"
            )
            cursor.execute(f"GRANT SELECT ON *.* TO '{IAM_USER}'@'%'")
        yield
    finally:
        with writer.cursor() as cursor:
            cursor.execute(f"DROP USER IF EXISTS '{IAM_USER}'@'%'")
        writer.close()


@pytest.mark.parametrize("mode", ["REQUIRED", "VERIFY_CA"])
def test_mysql_rds_iam_tokens_log_in_over_tls_through_a_local_port(
    iam_stand_in_reader, monkeypatch, mode
):
    """Each new pooled connection logs in with a token signed for the endpoint."""
    from sqlalchemy.engine import URL, make_url

    options = {"ssl_mode": mode}
    if mode == "VERIFY_CA":
        if not os.environ.get("NORI_REL_TEST_MYSQL_CA"):
            pytest.skip("NORI_REL_TEST_MYSQL_CA names the test server's CA certificate")
        options["ssl_ca"] = os.environ["NORI_REL_TEST_MYSQL_CA"]
    signed = []

    def get_signer(self, reference, url, *, default_port):
        signed_for = (reference.endpoint, url.username, default_port)
        assert signed_for == (IAM_ENDPOINT, IAM_USER, 3306)

        def sign():
            signed.append(signed_for)
            return IAM_TOKEN

        return sign

    monkeypatch.setattr(CredentialResolver, "get_rds_token_signer", get_signer)
    server = make_url(os.environ["NORI_REL_TEST_MYSQL_URL"])
    url = URL.create(
        "mysql",
        username=IAM_USER,
        host=server.host,
        port=server.port,
        database=server.database,
        query=options,
    )
    credential = RdsIamToken(
        url=url.render_as_string(hide_password=False), endpoint=IAM_ENDPOINT
    )
    with create_mysql_source(credential=credential) as source:
        assert source.connector.test() >= 0
        assert len(signed) == 1
        schema = source.connector.discover("db")
        assert {table.name for table in schema.tables} == {"users", "orders"}
        _, context, counts = _prepare(source, CASES[0][1])
        assert counts["candidate_rows"] == 6
        assert context["label"].tolist() == pytest.approx(
            list(_expected_labels(CASES[0][0]).values())
        )
        engine = source.connector._engine()
        with engine.connect() as first, engine.connect() as second:
            for connection in (first, second):
                cipher = connection.exec_driver_sql(
                    "SHOW SESSION STATUS LIKE 'Ssl_cipher'", {}
                ).one()[1]
                assert cipher
    # The pool holds two connections, and the second one signed its own token.
    assert len(signed) == 2
