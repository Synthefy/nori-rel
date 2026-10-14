"""Resolve database credentials only inside the caller's process."""

from __future__ import annotations

import json
import os
import re
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from sqlalchemy.engine import URL, make_url

from nori_rel.contracts import AwsSecret, CredentialReference, DatabaseUrl


class CredentialResolutionError(RuntimeError):
    pass


class CredentialResolver:
    """Resolve configured secret references locally without caching their values."""

    def resolve(self, reference: CredentialReference) -> URL:
        """Resolve a PostgreSQL URL using the existing credential contract."""
        return self._get_postgresql_url(self._get_secret_text(reference))

    def resolve_parameters(self, reference: CredentialReference) -> dict[str, str]:
        """Resolve warehouse auth JSON locally without caching or logging values."""
        if isinstance(reference, DatabaseUrl):
            raise CredentialResolutionError(
                "warehouse auth requires a secret reference"
            )
        try:
            parameters = json.loads(self._get_secret_text(reference))
            if not isinstance(parameters, dict) or not parameters:
                raise ValueError("empty auth")
            if any(
                not isinstance(key, str) or not isinstance(value, str) or not value
                for key, value in parameters.items()
            ):
                raise ValueError("auth fields must be strings")
        except (ValueError, TypeError):
            raise CredentialResolutionError(
                "warehouse auth must be a nonempty JSON object of string fields"
            ) from None
        return parameters

    @staticmethod
    def _get_secret_text(reference: CredentialReference) -> str:
        """Read a secret in the caller's process, without caching it."""
        if isinstance(reference, DatabaseUrl):
            raw = reference.value.get_secret_value()
        elif isinstance(reference, AwsSecret):
            try:
                client = boto3.client(
                    "secretsmanager", region_name=reference.region_name
                )
                response = client.get_secret_value(SecretId=reference.secret_id)
            except ClientError as error:
                code = error.response.get("Error", {}).get("Code")
                diagnostic = (
                    f" (code={code})"
                    if isinstance(code, str)
                    and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", code)
                    else ""
                )
                raise CredentialResolutionError(
                    "database secret could not be resolved" + diagnostic
                ) from None
            except BotoCoreError:
                raise CredentialResolutionError(
                    "database secret could not be resolved"
                ) from None
            raw = response.get("SecretString")
            if not raw:
                raise CredentialResolutionError("database secret has no SecretString")
        else:
            raw = os.getenv(reference.variable)
            if not raw:
                raise CredentialResolutionError(
                    f"database environment variable {reference.variable!r} is not set"
                )
        return raw

    @staticmethod
    def _get_postgresql_url(raw: str) -> URL:
        """Parse a PostgreSQL URL or RDS JSON without retaining raw secret text."""
        if raw.lstrip().startswith("{"):
            try:
                data: dict[str, Any] = json.loads(raw)
                engine = data.get("engine")
                if engine is not None and str(engine).lower() not in {
                    "postgres",
                    "postgresql",
                }:
                    raise ValueError("secret engine is not PostgreSQL")
                url = URL.create(
                    drivername="postgresql+psycopg",
                    username=data["username"],
                    password=data["password"],
                    host=data["host"],
                    port=int(data.get("port", 5432)),
                    database=data.get("dbname") or data["database"],
                    query={"sslmode": data.get("sslmode", "require")},
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                raise CredentialResolutionError(
                    "database secret must be a PostgreSQL URL or RDS credential JSON"
                ) from None
        else:
            try:
                url = make_url(raw)
            except Exception:
                raise CredentialResolutionError("database URL is invalid") from None
        if not url.drivername.startswith("postgresql"):
            raise CredentialResolutionError("only PostgreSQL is supported in version 1")
        if url.drivername != "postgresql+psycopg":
            url = url.set(drivername="postgresql+psycopg")
        return url
