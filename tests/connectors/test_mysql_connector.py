"""MySQL connector policy and value handling without a database dependency."""

import ssl
import traceback
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pymysql
import pytest
from sqlalchemy.engine import make_url
from sqlalchemy.exc import OperationalError

from nori_rel import NoriRelClient
from nori_rel.connectors import create_connector
from nori_rel.connectors.errors import ConnectorError
from nori_rel.connectors.mysql import (
    MySQLConnector,
    _get_driver_parameters,
    get_mysql_connect_arguments,
    get_mysql_type_name,
)
from nori_rel.contracts import (
    ColumnSchema,
    DatabaseCreateRequest,
    EnvironmentCredential,
    ForeignKeySchema,
    SnowflakeConnection,
    TableSchema,
)
from nori_rel.limits import RelationalLimits

CREDENTIAL = EnvironmentCredential(variable="TEST_ONLY")


def _mysql_connector():
    return MySQLConnector(
        DatabaseCreateRequest(
            engine="mysql", credential=CREDENTIAL, schema_name="shop"
        ),
        RelationalLimits(),
    )


@pytest.mark.parametrize(
    "url,mode",
    [
        ("mysql+pymysql://u:p@db.example.com/shop", "REQUIRED"),
        ("mysql+pymysql://u:p@db.example.com/shop?ssl_mode=verify_ca", "VERIFY_CA"),
        ("mysql+pymysql://u:p@127.0.0.1:3307/shop", "PREFERRED"),
        ("mysql+pymysql://u:p@localhost/shop?ssl_mode=DISABLED", "DISABLED"),
        ("mysql+pymysql://u:p@[::1]:3307/shop?ssl_mode=REQUIRED", "REQUIRED"),
    ],
)
def test_tls_modes_become_driver_arguments(url, mode):
    stripped, arguments = get_mysql_connect_arguments(make_url(url))
    assert stripped.query == {}
    context = arguments.get("ssl")
    if mode == "PREFERRED":
        # PyMySQL's own default: TLS when the server offers it.
        assert arguments == {}
    elif mode == "DISABLED":
        assert arguments == {"ssl_disabled": True}
    elif mode == "REQUIRED":
        assert context.verify_mode == ssl.CERT_NONE
        assert context.check_hostname is False
    else:
        assert context.verify_mode == ssl.CERT_REQUIRED
        assert context.check_hostname is False


def test_verify_identity_checks_the_host_name_against_the_given_bundle(tmp_path):
    bundle = tmp_path / "bundle.pem"
    bundle.write_text(ssl.DER_cert_to_PEM_cert(b"not a certificate"))
    with pytest.raises(ConnectorError, match="not a readable PEM bundle"):
        get_mysql_connect_arguments(
            make_url(
                "mysql+pymysql://u:p@db.example.com/shop"
                f"?ssl_mode=VERIFY_IDENTITY&ssl_ca={bundle}"
            )
        )
    _, arguments = get_mysql_connect_arguments(
        make_url("mysql+pymysql://u:p@db.example.com/shop?ssl_mode=VERIFY_IDENTITY")
    )
    assert arguments["ssl"].check_hostname is True
    assert arguments["ssl"].verify_mode == ssl.CERT_REQUIRED


@pytest.mark.parametrize(
    "url,message",
    [
        ("mysql+pymysql://u:p@db.example.com/shop?ssl_mode=DISABLED", "require TLS"),
        ("mysql+pymysql://u:p@db.example.com/shop?ssl_mode=PREFERRED", "require TLS"),
        ("mysql+pymysql://u:p@10.0.0.5/shop?ssl_mode=DISABLED", "require TLS"),
        ("mysql+pymysql://u:p@db.example.com/shop?ssl_mode=ON", "must be one of"),
        ("mysql+pymysql://u:p@db.example.com/shop?ssl_ca=/ca.pem", "ssl_ca needs"),
        ("mysql+pymysql://u:p@db.example.com/shop?local_infile=1", "unsupported"),
        ("mysql+pymysql://u:p@db.example.com/shop?init_command=DO+1", "unsupported"),
        (
            "mysql+pymysql://u:p@db.example.com/shop?ssl_mode=REQUIRED"
            "&ssl_mode=DISABLED",
            "must not repeat",
        ),
    ],
)
def test_tls_policy_refuses_unsafe_or_unknown_options(url, message):
    with pytest.raises(ConnectorError, match=message):
        get_mysql_connect_arguments(make_url(url))


@pytest.mark.parametrize(
    "base_type,column_type,expected",
    [
        ("bigint", "bigint unsigned", "BIGINT UNSIGNED"),
        ("bigint", "bigint", "BIGINT"),
        ("int", "int(11)", "INT"),
        ("int", "int(10) unsigned zerofill", "INT UNSIGNED"),
        ("tinyint", "tinyint(1)", "BOOLEAN"),
        ("tinyint", "tinyint(4)", "TINYINT"),
        ("mediumint", "mediumint unsigned", "MEDIUMINT UNSIGNED"),
        ("decimal", "decimal(12,2)", "DECIMAL(12, 2)"),
        ("decimal", "decimal(20,0) unsigned", "DECIMAL(20, 0)"),
        ("varchar", "varchar(16)", "VARCHAR(16)"),
        ("enum", "enum('a','b)c')", "ENUM"),
        ("datetime", "datetime(6)", "DATETIME"),
        ("timestamp", "timestamp(3)", "TIMESTAMP"),
        ("double", "double", "DOUBLE"),
        ("json", "json", "JSON"),
    ],
)
def test_information_schema_types_get_stable_names(base_type, column_type, expected):
    assert get_mysql_type_name(base_type, column_type) == expected


def test_datetimes_bind_as_naive_utc_and_other_values_unchanged():
    east = timezone(timedelta(hours=2))
    parameters = _get_driver_parameters(
        {
            "anchor": datetime(2026, 1, 1, 9, tzinfo=east),
            "naive": datetime(2026, 1, 1, 9),
            "amount": Decimal("1.5"),
        }
    )
    assert parameters == {
        "anchor": datetime(2026, 1, 1, 7),
        "naive": datetime(2026, 1, 1, 9),
        "amount": Decimal("1.5"),
    }


@pytest.mark.parametrize(
    "operation", ["test", "discover", "snapshot", "prepare_context"]
)
def test_vendor_errors_keep_a_code_and_drop_parameters(monkeypatch, operation):
    reader = _mysql_connector()
    engine = Mock()
    secret = "synthetic-sensitive-parameter-123"
    engine.connect.side_effect = OperationalError(
        "SELECT protected_data",
        {"private_value": secret},
        pymysql.err.OperationalError(1045, f"Access denied ({secret})"),
    )
    monkeypatch.setattr(reader, "_engine", lambda: engine)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    with pytest.raises(ConnectorError, match=r"\(code=1045\)") as caught:
        if operation == "test":
            reader.test()
        elif operation == "discover":
            reader.discover("db")
        elif operation == "snapshot":
            reader.snapshot("db", now)
        else:
            reader.prepare_context(
                "db",
                "PREDICT COUNT(orders.*,0,30,days) FOR EACH users.id",
                anchor_times=[now],
                anchor_time=now,
                observed_through=now,
                entity_created_at="created_at",
            )
    rendered = "".join(traceback.format_exception(caught.value))
    assert secret not in rendered
    assert "protected_data" not in rendered
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


@pytest.mark.parametrize(
    "is_mariadb,version,message",
    [(True, (10, 11, 6), "MariaDB"), (False, (5, 7, 44), "8.0 or later")],
)
def test_mariadb_and_mysql_before_8_are_refused(is_mariadb, version, message):
    connection = SimpleNamespace(
        dialect=SimpleNamespace(is_mariadb=is_mariadb, server_version_info=version)
    )
    with pytest.raises(ConnectorError, match=message):
        MySQLConnector._require_supported_server(connection)
    aurora = SimpleNamespace(
        dialect=SimpleNamespace(
            is_mariadb=False, server_version_info=(8, 0, "mysql_aurora", 3, 4, 0)
        )
    )
    MySQLConnector._require_supported_server(aurora)


def _table(*columns, time_column=None):
    return TableSchema(
        name="events",
        columns=[ColumnSchema(name=n, data_type=t, nullable=True) for n, t in columns],
        primary_key=["id"],
        foreign_keys=[
            ForeignKeySchema(
                columns=["ref"], referred_table="parents", referred_columns=["code"]
            )
        ],
        time_column=time_column,
    )


def test_values_get_feature_dtypes_and_keys_stay_exact():
    table = _table(
        ("id", "BIGINT UNSIGNED"),
        ("ref", "DECIMAL(20, 0)"),
        ("count", "INT"),
        ("price", "DECIMAL(12, 2)"),
        ("ratio", "DOUBLE"),
        ("flag", "BOOLEAN"),
        ("level", "BOOLEAN"),
        ("seen", "DATETIME"),
        ("valid_to", "DATE"),
        ("payload", "JSON"),
        time_column="seen",
    )
    frame = pd.DataFrame(
        {
            "id": [2**64 - 1, 1],
            "ref": [Decimal(7), None],
            "count": [3, None],
            "price": [Decimal("12.50"), None],
            "ratio": [0.5, None],
            "flag": [1, None],
            "level": [2, 0],
            "seen": [datetime(2026, 1, 1), None],
            "valid_to": [datetime(9999, 12, 31).date(), None],
            "payload": ['{"k": 1}', None],
        },
        dtype=object,
    )
    frame = MySQLConnector._normalize_table_frame(frame, table)
    assert frame["id"].tolist() == [2**64 - 1, 1]
    assert frame["ref"].tolist()[0] == Decimal(7)
    assert str(frame["count"].dtype) == "Int64"
    assert str(frame["price"].dtype) == str(frame["ratio"].dtype) == "Float64"
    assert frame["price"].tolist()[0] == 12.5
    assert frame["flag"].tolist() == [True, pd.NA]
    # TINYINT(1) holding other numbers is read as an integer, not a flag.
    assert frame["level"].tolist() == [2, 0]
    assert frame["seen"].tolist()[0] == pd.Timestamp("2026-01-01", tz="UTC")
    assert str(frame["valid_to"].tolist()[0]) == "9999-12-31"
    assert frame["payload"].tolist()[0] == '{"k": 1}'


def test_a_time_column_pandas_cannot_hold_fails():
    table = _table(
        ("id", "BIGINT"), ("ref", "INT"), ("seen", "DATE"), time_column="seen"
    )
    frame = pd.DataFrame(
        {"id": [1], "ref": [1], "seen": [datetime(9999, 12, 31).date()]}, dtype=object
    )
    with pytest.raises(ConnectorError, match=r"events\.seen holds a date"):
        MySQLConnector._normalize_table_frame(frame, table)


def test_mysql_sources_name_their_database_and_take_no_warehouse_settings():
    with pytest.raises(ValueError, match="explicit schema_name"):
        DatabaseCreateRequest(engine="mysql", credential=CREDENTIAL)
    with pytest.raises(ValueError, match="require a warehouse engine"):
        DatabaseCreateRequest(
            engine="mysql",
            credential=CREDENTIAL,
            schema_name="shop",
            connection=SnowflakeConnection(account="a", warehouse="w", database="d"),
        )
    with pytest.raises(ValueError, match="require a warehouse engine"):
        DatabaseCreateRequest(
            engine="mysql",
            credential=CREDENTIAL,
            schema_name="shop",
            primary_keys={"users": ["user_id"]},
        )
    config = DatabaseCreateRequest(
        engine="mysql", credential=CREDENTIAL, schema_name="shop"
    )
    # Creating the connector resolves no credential and opens no connection.
    assert isinstance(create_connector(config, RelationalLimits()), MySQLConnector)


def test_client_connect_asks_for_the_mysql_database(fake_checkpoint):
    with (
        NoriRelClient(mode="hosted", api_key="test-only") as client,
        pytest.raises(ValueError, match="MySQL connections require an explicit"),
    ):
        client.connect(engine="mysql", database_url="mysql://u:p@127.0.0.1/shop")
