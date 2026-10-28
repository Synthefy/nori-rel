"""Typed configuration models for Nori-Rel."""

from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum
from typing import Literal, get_args
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)

_ENVIRONMENT_VARIABLE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ENDPOINT = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?(?::(?P<port>\d+))?")


class StrictModel(BaseModel):
    """Reject unknown configuration fields."""

    model_config = ConfigDict(extra="forbid")


class DatabaseEngine(StrEnum):
    """Supported database engines."""

    POSTGRESQL = "postgresql"
    SNOWFLAKE = "snowflake"
    DATABRICKS = "databricks"
    MYSQL = "mysql"


class DatabaseStatus(StrEnum):
    """Database connection lifecycle state."""

    PENDING = "pending"
    READY = "ready"
    ERROR = "error"


class PredictionTask(StrEnum):
    """Supported prediction problem types."""

    REGRESSION = "regression"
    CLASSIFICATION = "classification"


class RelationalOutputType(StrEnum):
    """Supported regression output summaries."""

    MEAN = "mean"
    MEDIAN = "median"
    QUANTILES = "quantiles"


#: The errors an agent search can score its candidate programs with, on a
#: regression target; lower is better for each. See ``predict``'s ``metric``.
AgentSearchMetric = Literal[
    "mae", "mse", "rmse", "nmae", "nrmse", "medae", "wape", "mape", "smape"
]
#: Every ``AgentSearchMetric``, in the order above.
AGENT_SEARCH_METRICS: tuple[str, ...] = get_args(AgentSearchMetric)


class AwsSecret(BaseModel):
    """Reference to a database URL or RDS credential JSON in Secrets Manager."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: Literal["aws_secrets_manager"] = "aws_secrets_manager"
    secret_id: str = Field(min_length=1)
    region_name: str | None = None


class EnvironmentCredential(BaseModel):
    """Reference to a database URL or warehouse auth JSON in the environment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: Literal["environment"] = "environment"
    variable: str = Field(min_length=1)

    @field_validator("variable")
    @classmethod
    def validate_variable(cls, value: str) -> str:
        """Require a valid environment-variable name."""
        if not _ENVIRONMENT_VARIABLE.fullmatch(value):
            raise ValueError("variable must be a valid environment-variable name")
        return value


class DatabaseUrl(BaseModel):
    """A database URL retained only in memory by the local client."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: Literal["url"] = "url"
    value: SecretStr


class RdsIamToken(BaseModel):
    """Log in to RDS or Aurora MySQL with IAM database authentication.

    ``url`` is a MySQL URL without a password, e.g.
    ``mysql://nori_reader@shop.abc123.us-east-1.rds.amazonaws.com/shop``,
    with the usual ``ssl_mode`` and ``ssl_ca`` options. Each new connection
    logs in with a token that the caller's AWS identity signs for
    ``endpoint``, the instance or cluster endpoint as ``host`` or
    ``host:port`` (default port 3306). The endpoint defaults to the URL's host
    and port, and is required when the URL names an SSH tunnel's local end.
    ``region_name`` defaults to the Region in the endpoint's name, else to
    boto3's default Region.
    """

    # A rejected URL may hold the password it was refused for.
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    provider: Literal["aws_rds_iam"] = "aws_rds_iam"
    url: str = Field(min_length=1)
    endpoint: str | None = None
    region_name: str | None = None

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        """Require a user and a host, and refuse the password a token replaces."""
        try:
            parts = urlsplit(value)
            # Reading the port raises ValueError for one that is not 1..65535.
            if parts.port == 0:
                raise ValueError("port 0")
        except ValueError:
            raise ValueError("url must be a database URL") from None
        if parts.password is not None:
            raise ValueError("url must not hold a password; IAM tokens replace it")
        if not parts.scheme or not parts.username or not parts.hostname:
            raise ValueError("url must name a database user and host")
        return value

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, value: str | None) -> str | None:
        """Require a host name, optionally followed by a port in 1..65535."""
        if value is None:
            return value
        match = _ENDPOINT.fullmatch(value)
        if match is None or (match["port"] and not 0 < int(match["port"]) < 65536):
            raise ValueError("endpoint must be a host name or host:port")
        return value


CredentialReference = AwsSecret | EnvironmentCredential | DatabaseUrl | RdsIamToken


class SnowflakeConnection(StrictModel):
    """Nonsecret Snowflake account, compute, and database selection."""

    account: str = Field(min_length=1)
    warehouse: str = Field(min_length=1)
    database: str = Field(min_length=1)
    role: str | None = Field(default=None, min_length=1)

    @field_validator("account", "warehouse", "database", "role")
    @classmethod
    def validate_connection_names(cls, value: str | None) -> str | None:
        """Reject blank or NUL-containing settings before connecting."""
        if value is not None and (not value.strip() or "\x00" in value):
            raise ValueError("connection names must not be blank or contain NUL")
        return value


class DatabricksConnection(StrictModel):
    """Nonsecret SQL warehouse endpoint and Unity Catalog selection."""

    server_hostname: str = Field(min_length=1)
    http_path: str = Field(min_length=1)
    catalog: str = Field(min_length=1)

    @field_validator("server_hostname")
    @classmethod
    def validate_server_hostname(cls, value: str) -> str:
        """Require a hostname so credentials cannot change the HTTPS protocol."""
        if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?", value):
            raise ValueError("server_hostname must be a hostname without a URL scheme")
        return value

    @field_validator("http_path")
    @classmethod
    def validate_http_path(cls, value: str) -> str:
        """Require an absolute SQL warehouse path without query parameters."""
        if not re.fullmatch(
            r"/sql/(?:1\.0/warehouses|protocolv1/o)/[A-Za-z0-9_/-]+", value
        ):
            raise ValueError("http_path must identify a SQL warehouse")
        return value

    @field_validator("catalog")
    @classmethod
    def validate_catalog(cls, value: str) -> str:
        """Keep literal catalog names nonblank and NUL-free."""
        if not value.strip() or "\x00" in value:
            raise ValueError("catalog must not be blank or contain NUL")
        return value


WarehouseConnection = SnowflakeConnection | DatabricksConnection


class DatabaseCreateRequest(StrictModel):
    """Configuration retained for a registered database source."""

    engine: DatabaseEngine
    credential: CredentialReference
    schema_name: str = Field(default="public", min_length=1, max_length=128)
    tables: list[str] | None = None
    time_columns: dict[str, str] = Field(default_factory=dict)
    connection: WarehouseConnection | None = None
    primary_keys: dict[str, list[str]] = Field(default_factory=dict)
    foreign_keys: dict[str, list[ForeignKeySchema]] = Field(default_factory=dict)

    @field_validator("schema_name")
    @classmethod
    def strip_names(cls, value: str) -> str:
        """Normalize required source and schema names."""
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("tables")
    @classmethod
    def validate_tables(cls, value: list[str] | None) -> list[str] | None:
        """Normalize and de-duplicate the table allowlist."""
        if value is None:
            return None
        cleaned = [table.strip() for table in value]
        if not cleaned or any(not table for table in cleaned):
            raise ValueError("tables must contain at least one non-blank table")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("tables must not contain duplicates")
        return cleaned

    @field_validator("time_columns")
    @classmethod
    def validate_time_columns(cls, value: dict[str, str]) -> dict[str, str]:
        """Validate explicit table-to-time-column mappings.

        Quoted PostgreSQL names may contain dots, spaces and hyphens, so only
        blank and NUL-bearing names are refused here. Names FastDFS cannot
        carry are aliased internally at the feature boundary.
        """
        if any(
            not table.strip() or not column.strip() or "\x00" in table + column
            for table, column in value.items()
        ):
            raise ValueError("time_columns must map table names to column names")
        return value

    @model_validator(mode="after")
    def validate_table_configuration(self) -> DatabaseCreateRequest:
        """Keep time-column mappings inside the table allowlist."""
        if self.tables is not None:
            if set(self.time_columns).difference(self.tables):
                raise ValueError("time_columns contains a table outside tables")
            configured = (
                set(self.time_columns) | set(self.primary_keys) | set(self.foreign_keys)
            )
            unknown = configured.difference(self.tables)
            if unknown:
                raise ValueError("table configuration contains a table outside tables")
        if (
            isinstance(self.credential, RdsIamToken)
            and self.engine is not DatabaseEngine.MYSQL
        ):
            raise ValueError("RdsIamToken is supported for MySQL sources only")
        if self.engine in {DatabaseEngine.POSTGRESQL, DatabaseEngine.MYSQL}:
            if self.connection is not None:
                raise ValueError("connection settings require a warehouse engine")
            if self.primary_keys or self.foreign_keys:
                raise ValueError(
                    "explicit relationship mappings require a warehouse engine"
                )
            if (
                self.engine is DatabaseEngine.MYSQL
                and "schema_name" not in self.model_fields_set
            ):
                raise ValueError(
                    "MySQL connections require an explicit schema_name, "
                    "the MySQL database to read"
                )
        else:
            expected = (
                SnowflakeConnection
                if self.engine is DatabaseEngine.SNOWFLAKE
                else DatabricksConnection
            )
            if not isinstance(self.connection, expected):
                raise ValueError("connection settings do not match the database engine")
            if isinstance(self.credential, DatabaseUrl):
                raise ValueError(
                    "warehouse credentials require an environment or secret reference"
                )
            if "schema_name" not in self.model_fields_set:
                raise ValueError(
                    "warehouse connections require an explicit schema_name"
                )
        for table, keys in self.primary_keys.items():
            if not table.strip() or not keys or len(keys) != len(set(keys)):
                raise ValueError(
                    "primary_keys must contain nonempty unique column lists"
                )
            if any(not key.strip() or "\x00" in key for key in keys):
                raise ValueError("primary key names must not be blank or contain NUL")
        return self


class Database(StrictModel):
    """Opaque handle for a source registered with one client instance."""

    id: str = Field(min_length=1)
    engine: DatabaseEngine
    schema_name: str
    status: DatabaseStatus
    created_at: datetime
    updated_at: datetime
    error: str | None = None


class ConnectionStatus(StrictModel):
    """Result of a live source connectivity check."""

    database_id: str
    status: DatabaseStatus
    latency_ms: float | None = Field(default=None, ge=0)
    error: str | None = None


class ColumnSchema(StrictModel):
    """Discovered database column metadata."""

    name: str
    data_type: str
    nullable: bool


class ForeignKeySchema(StrictModel):
    """Discovered foreign-key relationship metadata."""

    name: str | None = None
    columns: list[str] = Field(min_length=1)
    referred_table: str
    referred_columns: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def matching_arity(self) -> ForeignKeySchema:
        """Require matching child and parent key widths."""
        if len(self.columns) != len(self.referred_columns):
            raise ValueError("foreign-key columns must have matching arity")
        return self


class TableSchema(StrictModel):
    """Discovered table metadata used for relational traversal."""

    name: str
    columns: list[ColumnSchema]
    primary_key: list[str] = Field(default_factory=list)
    foreign_keys: list[ForeignKeySchema] = Field(default_factory=list)
    time_column: str | None = None


class ConnectorCapabilities(StrictModel):
    """Connector guarantees exposed to the runtime."""

    consistent_snapshot: bool = True
    schemas: bool = True
    server_side_cursor: bool = True
    snapshot_mode: Literal["transaction", "time_travel", "table_versions"] = (
        "transaction"
    )
    read_only_mode: Literal["transaction", "role"] = "transaction"
    unquoted_identifier_case: Literal["exact", "upper"] = "exact"


class SchemaGraph(StrictModel):
    """Tables and foreign-key edges for one registered source."""

    database_id: str
    schema_name: str
    tables: list[TableSchema]
    capabilities: ConnectorCapabilities = Field(default_factory=ConnectorCapabilities)
    discovered_at: datetime


DatabaseCreateRequest.model_rebuild()
