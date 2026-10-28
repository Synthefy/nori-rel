from __future__ import annotations

import json

import boto3
import pytest
from botocore.exceptions import ClientError, EndpointConnectionError
from pydantic import ValidationError

from nori_rel import AwsSecret, EnvironmentCredential, RdsIamToken
from nori_rel._credentials import (
    CredentialResolutionError,
    CredentialResolver,
    get_rds_region,
)


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


@pytest.fixture
def offline_aws(monkeypatch, tmp_path):
    """Hide every AWS identity of this machine; tests set fake keys themselves."""
    for variable in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_SECURITY_TOKEN",
        "AWS_PROFILE",
        "AWS_DEFAULT_PROFILE",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "AWS_ROLE_ARN",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
    ):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "no-config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "no-credentials"))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    # boto3 caches its default session, and with it the first identity found.
    monkeypatch.setattr(boto3, "DEFAULT_SESSION", None)
    return monkeypatch


@pytest.fixture
def fake_aws_keys(offline_aws):
    offline_aws.setenv("AWS_ACCESS_KEY_ID", "AKIDEXAMPLE")
    offline_aws.setenv(
        "AWS_SECRET_ACCESS_KEY", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    )


@pytest.mark.parametrize(
    "host,region",
    [
        ("shop.abc123.us-east-1.rds.amazonaws.com", "us-east-1"),
        ("shop.cluster-abc123.us-west-2.rds.amazonaws.com", "us-west-2"),
        ("shop.cluster-ro-abc123.eu-west-1.rds.amazonaws.com", "eu-west-1"),
        ("shop-proxy.proxy-abc123.eu-central-1.rds.amazonaws.com", "eu-central-1"),
        ("my-db-1.abc123.us-gov-west-1.rds.amazonaws.com", "us-gov-west-1"),
        ("shop.abc123.rds.cn-north-1.amazonaws.com.cn", "cn-north-1"),
        ("SHOP.ABC123.AP-SOUTHEAST-2.RDS.AMAZONAWS.COM.", "ap-southeast-2"),
        ("db.example.com", None),
        ("my-db-1.example.com", None),
    ],
)
def test_rds_region_comes_from_the_endpoint_name(host, region):
    assert get_rds_region(host) == region


def _sign(reference: RdsIamToken) -> str:
    resolver = CredentialResolver()
    url = resolver.resolve_mysql(reference)
    return resolver.get_rds_token_signer(reference, url, default_port=3306)()


@pytest.mark.parametrize(
    "reference,signed_for,region",
    [
        (
            RdsIamToken(url="mysql://nori_reader@shop.abc.us-east-1.rds.amazonaws.com"),
            "shop.abc.us-east-1.rds.amazonaws.com:3306",
            "us-east-1",
        ),
        (
            RdsIamToken(
                url="mysql://nori_reader@127.0.0.1:3307/shop?ssl_mode=VERIFY_CA",
                endpoint="shop.cluster-abc.us-west-2.rds.amazonaws.com:3310",
            ),
            "shop.cluster-abc.us-west-2.rds.amazonaws.com:3310",
            "us-west-2",
        ),
        (
            RdsIamToken(
                url="mysql://nori_reader@127.0.0.1:3307/shop",
                endpoint="shop.abc.rds.cn-north-1.amazonaws.com.cn",
            ),
            "shop.abc.rds.cn-north-1.amazonaws.com.cn:3306",
            "cn-north-1",
        ),
        (
            RdsIamToken(
                url="mysql://nori_reader@db.example.com:3310/shop",
                region_name="eu-west-1",
            ),
            "db.example.com:3310",
            "eu-west-1",
        ),
    ],
    ids=["direct", "tunnel", "default-port", "custom-name"],
)
def test_rds_iam_tokens_are_signed_offline_for_the_endpoint(
    fake_aws_keys, reference, signed_for, region
):
    token = _sign(reference)
    # Signing is local SigV4; the token names the endpoint, user and Region.
    assert token.startswith(f"{signed_for}/?Action=connect&DBUser=nori_reader&")
    assert f"%2F{region}%2Frds-db%2Faws4_request" in token
    assert "X-Amz-Expires=900" in token


def test_rds_iam_signing_uses_the_default_region_for_a_custom_name(
    fake_aws_keys, offline_aws
):
    reference = RdsIamToken(url="mysql://nori_reader@db.example.com/shop")
    with pytest.raises(CredentialResolutionError, match="set region_name"):
        _sign(reference)
    offline_aws.setenv("AWS_DEFAULT_REGION", "ap-south-1")
    assert "%2Fap-south-1%2Frds-db%2F" in _sign(reference)


def test_rds_iam_signing_without_aws_credentials_fails_plainly(offline_aws):
    with pytest.raises(CredentialResolutionError, match="could not be signed") as error:
        _sign(
            RdsIamToken(url="mysql://nori_reader@shop.abc.us-east-1.rds.amazonaws.com")
        )
    assert error.value.__suppress_context__


@pytest.mark.parametrize(
    "url,message",
    [
        ("mysql://nori_reader:private-password@db/shop", "must not hold a password"),
        ("mysql://nori_reader:@db/shop", "must not hold a password"),
        ("mysql://db/shop", "user and host"),
        ("nori_reader@db/shop", "user and host"),
        ("mysql://nori_reader@db:99999/shop", "database URL"),
    ],
)
def test_rds_iam_urls_need_a_user_and_host_and_no_password(url, message):
    with pytest.raises(ValidationError, match=message) as error:
        RdsIamToken(url=url)
    # A refused URL is not repeated, in case it held a password.
    assert "private-password" not in str(error.value)


@pytest.mark.parametrize("endpoint", ["db:0", "db:65536", "https://db", "db:", "-db"])
def test_rds_iam_endpoints_are_a_host_and_optional_port(endpoint):
    with pytest.raises(ValidationError, match="host name or host:port"):
        RdsIamToken(url="mysql://nori_reader@127.0.0.1:3307/shop", endpoint=endpoint)


def test_rds_iam_tokens_resolve_only_for_mysql():
    resolver = CredentialResolver()
    with pytest.raises(CredentialResolutionError, match="MariaDB"):
        resolver.resolve_mysql(RdsIamToken(url="mariadb://nori_reader@db/shop"))
    with pytest.raises(CredentialResolutionError, match="MySQL sources only"):
        resolver.resolve(RdsIamToken(url="postgresql://nori_reader@db/shop"))
    url = resolver.resolve_mysql(
        RdsIamToken(url="mysql://nori_reader@db/shop?ssl_mode=VERIFY_IDENTITY")
    )
    assert url.render_as_string(hide_password=False) == (
        "mysql+pymysql://nori_reader@db/shop?ssl_mode=VERIFY_IDENTITY"
    )
