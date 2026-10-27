from __future__ import annotations

import json

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError

from nori_rel import AwsSecret, EnvironmentCredential
from nori_rel._credentials import CredentialResolutionError, CredentialResolver


class SecretsManager:
    def __init__(self, secret: dict) -> None:
        self.secret = secret

    def get_secret_value(self, *, SecretId: str) -> dict[str, str]:
        assert SecretId
        return {"SecretString": json.dumps(self.secret)}


@pytest.mark.parametrize(
    "host",
    [
        "nori.abc123.us-west-2.rds.amazonaws.com",
        "nori.cluster-abc123.us-west-2.rds.amazonaws.com",
    ],
    ids=["rds-postgresql", "aurora-postgresql"],
)
def test_resolves_aws_postgresql_secrets(monkeypatch, host: str) -> None:
    secret = {
        "engine": "postgres",
        "host": host,
        "port": 5432,
        "dbname": "analytics",
        "username": "nori_readonly",
        "password": "secret",
    }
    monkeypatch.setattr(
        "nori_rel._credentials.boto3.client",
        lambda service, region_name: SecretsManager(secret),
    )

    url = CredentialResolver().resolve(
        AwsSecret(secret_id="nori/database", region_name="us-west-2")
    )

    assert url.drivername == "postgresql+psycopg"
    assert url.host == host
    assert url.database == "analytics"
    assert url.query == {"sslmode": "require"}


def test_a_postgresql_source_rejects_an_aws_mysql_secret(monkeypatch) -> None:
    secret = {
        "engine": "mysql",
        "host": "nori.abc123.us-west-2.rds.amazonaws.com",
        "username": "nori_readonly",
        "password": "secret",
        "dbname": "analytics",
    }
    monkeypatch.setattr(
        "nori_rel._credentials.boto3.client",
        lambda service, region_name: SecretsManager(secret),
    )

    with pytest.raises(CredentialResolutionError, match="PostgreSQL"):
        CredentialResolver().resolve(AwsSecret(secret_id="nori/mysql"))


@pytest.mark.parametrize(
    "engine,host",
    [
        ("mysql", "nori.abc123.us-west-2.rds.amazonaws.com"),
        ("aurora-mysql", "nori.cluster-abc123.us-west-2.rds.amazonaws.com"),
        (None, "nori.abc123.us-west-2.rds.amazonaws.com"),
    ],
    ids=["rds-mysql", "aurora-mysql", "no-engine"],
)
def test_resolves_aws_mysql_secrets(monkeypatch, engine, host) -> None:
    secret = {
        "host": host,
        "username": "nori_readonly",
        "password": "secret",
        "dbname": "analytics",
        "ssl_mode": "VERIFY_IDENTITY",
        "ssl_ca": "/etc/ssl/rds-global-bundle.pem",
    }
    if engine is not None:
        secret["engine"] = engine
    monkeypatch.setattr(
        "nori_rel._credentials.boto3.client",
        lambda service, region_name: SecretsManager(secret),
    )

    url = CredentialResolver().resolve_mysql(
        AwsSecret(secret_id="nori/mysql", region_name="us-west-2")
    )

    assert url.drivername == "mysql+pymysql"
    assert (url.host, url.port, url.database) == (host, 3306, "analytics")
    assert url.query == {
        "ssl_mode": "VERIFY_IDENTITY",
        "ssl_ca": "/etc/ssl/rds-global-bundle.pem",
    }


@pytest.mark.parametrize(
    "raw,expected",
    [
        (
            "mysql://u:p@127.0.0.1:3307/shop",
            "mysql+pymysql://u:***@127.0.0.1:3307/shop",
        ),
        ("mysql+mysqldb://u:p@db/shop", "mysql+pymysql://u:***@db/shop"),
        (
            json.dumps({"host": "db", "username": "u", "password": "p"}),
            "mysql+pymysql://u:***@db:3306",
        ),
    ],
)
def test_mysql_urls_are_read_with_pymysql(monkeypatch, raw, expected) -> None:
    monkeypatch.setenv("NORI_REL_TEST_MYSQL_SECRET", raw)
    url = CredentialResolver().resolve_mysql(
        EnvironmentCredential(variable="NORI_REL_TEST_MYSQL_SECRET")
    )
    assert url.render_as_string(hide_password=True) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "postgresql://u:p@db/shop",
        "mariadb://u:p@db/shop",
        json.dumps(
            {"engine": "mariadb", "host": "db", "username": "u", "password": "p"}
        ),
        json.dumps(
            {"engine": "postgres", "host": "db", "username": "u", "password": "p"}
        ),
        json.dumps({"engine": "mysql", "username": "u", "password": "p"}),
    ],
)
def test_a_mysql_source_rejects_other_engines_and_incomplete_secrets(
    monkeypatch, raw
) -> None:
    monkeypatch.setenv("NORI_REL_TEST_MYSQL_SECRET", raw)
    with pytest.raises(CredentialResolutionError, match="MySQL") as caught:
        CredentialResolver().resolve_mysql(
            EnvironmentCredential(variable="NORI_REL_TEST_MYSQL_SECRET")
        )
    assert "p@" not in str(caught.value)


@pytest.mark.parametrize("code", ["AccessDeniedException", "ResourceNotFoundException"])
def test_aws_secret_errors_retain_safe_vendor_code(monkeypatch, code):
    error = ClientError(
        {"Error": {"Code": code, "Message": "private-password"}}, "GetSecretValue"
    )
    manager = SecretsManager({})
    manager.get_secret_value = lambda **options: (_ for _ in ()).throw(error)
    monkeypatch.setattr(
        "nori_rel._credentials.boto3.client", lambda *args, **kwargs: manager
    )
    with pytest.raises(CredentialResolutionError) as failure:
        CredentialResolver().resolve(AwsSecret(secret_id="private-secret-name"))
    assert f"code={code}" in str(failure.value)
    assert "private" not in str(failure.value)
    assert failure.value.__suppress_context__


def test_aws_secret_invalid_error_code_and_endpoint_are_sanitized(monkeypatch):
    for error in (
        ClientError(
            {"Error": {"Code": "private token\n", "Message": "private"}},
            "GetSecretValue",
        ),
        EndpointConnectionError(endpoint_url="https://private-endpoint"),
    ):
        manager = SecretsManager({})
        manager.get_secret_value = lambda error=error, **options: (_ for _ in ()).throw(
            error
        )
        monkeypatch.setattr(
            "nori_rel._credentials.boto3.client",
            lambda *args, manager=manager, **kwargs: manager,
        )
        with pytest.raises(CredentialResolutionError) as failure:
            CredentialResolver().resolve(AwsSecret(secret_id="private-secret-name"))
        assert str(failure.value) == "database secret could not be resolved"
        assert failure.value.__suppress_context__
