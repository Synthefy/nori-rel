from __future__ import annotations

import json

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError

from nori_rel import AwsSecret
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


def test_rejects_an_aws_mysql_secret(monkeypatch) -> None:
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
