"""Snowflake standard-table discovery and pinned Time Travel reads."""

from __future__ import annotations

import math
from contextlib import suppress
from datetime import datetime
from numbers import Integral
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SecretStr,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from nori_rel.connectors.errors import ConnectorError, SnapshotLimitError
from nori_rel.connectors.warehouse import (
    WarehouseConnector,
    build_warehouse_schema_graph,
    dataframe_records_by_lowercase_column,
    normalize_warehouse_timestamp_utc,
)
from nori_rel.contracts import (
    ConnectorCapabilities,
    SchemaGraph,
    SnowflakeConnection,
)


class _SnowflakeAuthModel(BaseModel):
    """Reject driver overrides and conceal secret inputs in validation errors."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, hide_input_in_errors=True
    )


class SnowflakeKeyPairAuth(_SnowflakeAuthModel):
    """JWT identity using exactly one PEM key or local private-key file."""

    authenticator: Literal["SNOWFLAKE_JWT"] = "SNOWFLAKE_JWT"
    user: str = Field(min_length=1)
    private_key: SecretStr | None = Field(default=None, min_length=1)
    private_key_password: SecretStr | None = Field(default=None, min_length=1)
    private_key_file: SecretStr | None = Field(default=None, min_length=1)
    private_key_file_pwd: SecretStr | None = Field(default=None, min_length=1)
    _private_key_der: bytes | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def validate_private_key(self) -> SnowflakeKeyPairAuth:
        """Validate key selection and decode PEM once without exposing its value."""
        if (self.private_key is None) == (self.private_key_file is None):
            raise ValueError(
                "exactly one of private_key and private_key_file is required"
            )
        if self.private_key is None:
            if self.private_key_password is not None:
                raise ValueError(
                    "private_key_password requires private_key; "
                    "use private_key_file_pwd for a file"
                )
            return self
        if self.private_key_file_pwd is not None:
            raise ValueError(
                "private_key_file_pwd requires private_key_file; "
                "use private_key_password for PEM"
            )
        try:
            from cryptography.hazmat.primitives import serialization

            password = self.private_key_password
            key = serialization.load_pem_private_key(
                self.private_key.get_secret_value().encode(),
                password=password.get_secret_value().encode() if password else None,
            )
            self._private_key_der = key.private_bytes(
                serialization.Encoding.DER,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        except Exception:
            raise ValueError(
                "private_key PEM or private_key_password is invalid"
            ) from None
        return self


class SnowflakeOAuthTokenAuth(_SnowflakeAuthModel):
    """An OAuth access token, optionally scoped to an explicit user."""

    authenticator: Literal["oauth"] = "oauth"
    token: SecretStr = Field(min_length=1)
    user: str | None = Field(default=None, min_length=1)


class SnowflakeOAuthClientCredentialsAuth(_SnowflakeAuthModel):
    """OAuth client credentials with an optional verified HTTPS token endpoint."""

    authenticator: Literal["OAUTH_CLIENT_CREDENTIALS"] = "OAUTH_CLIENT_CREDENTIALS"
    oauth_client_id: SecretStr = Field(min_length=1)
    oauth_client_secret: SecretStr = Field(min_length=1)
    oauth_token_request_url: SecretStr | None = Field(default=None, min_length=1)
    oauth_scope: str | None = Field(default=None, min_length=1)
    user: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_token_endpoint(self) -> SnowflakeOAuthClientCredentialsAuth:
        """Permit only HTTPS endpoints without embedded credentials."""
        if self.oauth_token_request_url is not None:
            try:
                parsed = urlsplit(self.oauth_token_request_url.get_secret_value())
                valid = (
                    parsed.scheme == "https"
                    and parsed.hostname
                    and not parsed.username
                    and not parsed.password
                )
            except ValueError:
                valid = False
            if not valid:
                raise ValueError(
                    "oauth_token_request_url must use HTTPS "
                    "without embedded credentials"
                )
        return self


SnowflakeAuth = Annotated[
    SnowflakeKeyPairAuth
    | SnowflakeOAuthTokenAuth
    | SnowflakeOAuthClientCredentialsAuth,
    Field(discriminator="authenticator"),
]
_SNOWFLAKE_AUTH_ADAPTER = TypeAdapter(SnowflakeAuth)
_SNOWFLAKE_AUTH_FIELDS = frozenset().union(
    SnowflakeKeyPairAuth.model_fields,
    SnowflakeOAuthTokenAuth.model_fields,
    SnowflakeOAuthClientCredentialsAuth.model_fields,
)


def _get_safe_authentication_error(error: ValidationError) -> str:
    """Return useful known field diagnostics without arbitrary inputs or keys."""
    diagnostics = []
    for detail in error.errors(
        include_input=False, include_context=False, include_url=False
    ):
        field = next(
            (
                part
                for part in reversed(detail["loc"])
                if part in _SNOWFLAKE_AUTH_FIELDS
            ),
            None,
        )
        kind = detail["type"]
        if kind == "union_tag_invalid":
            diagnostic = "authenticator: unsupported authentication mode"
        elif kind == "extra_forbidden":
            diagnostic = (
                f"{field}: unsupported for this authentication mode"
                if field
                else "unexpected credential field"
            )
        elif kind == "missing":
            diagnostic = f"{field}: required"
        elif kind == "value_error":
            # Our validators contain fixed messages only; Pydantic context is omitted.
            diagnostic = detail["msg"].removeprefix("Value error, ")
        else:
            diagnostic = f"{field or 'credential field'}: must be a nonempty string"
        if diagnostic not in diagnostics:
            diagnostics.append(diagnostic)
    return "Snowflake authentication invalid: " + "; ".join(diagnostics)


def _get_matching_identifier(name: str, available: list[str], kind: str) -> str:
    """Keep exact names, or resolve one uppercase-equivalent metadata name."""
    if name in available:
        return name
    matches = [
        candidate for candidate in available if candidate.upper() == name.upper()
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ConnectorError(f"ambiguous Snowflake {kind}: {name}")
    raise ConnectorError(f"Snowflake {kind} does not exist: {name}")


class SnowflakeConnector(WarehouseConnector):
    """Read standard Snowflake tables with one timestamp for every extraction."""

    @staticmethod
    def _quote_identifier(name: str) -> str:
        """Quote exact identifiers, preserving compiler percent escaping."""
        if not name or "\x00" in name:
            raise ConnectorError("invalid warehouse identifier")
        return '"' + name.replace('"', '""').replace("%", "%%") + '"'

    def _get_qualified_table_reference(self, name: str) -> str:
        """Address the configured database and schema explicitly."""
        connection = self.config.connection
        if not isinstance(connection, SnowflakeConnection):
            raise ConnectorError("Snowflake connection settings are required")
        return ".".join(
            self._quote_identifier(part)
            for part in (connection.database, self.config.schema_name, name)
        )

    def _get_authentication_parameters(self) -> dict[str, Any]:
        """Validate the typed auth union and expose native fields only to the driver."""
        secret = self._get_resolved_auth_fields()
        inferred = (
            "SNOWFLAKE_JWT"
            if "private_key" in secret or "private_key_file" in secret
            else "oauth"
        )
        try:
            auth = _SNOWFLAKE_AUTH_ADAPTER.validate_python(
                {"authenticator": inferred, **secret}
            )
        except ValidationError as error:
            raise ConnectorError(_get_safe_authentication_error(error)) from None
        result = {
            name: value.get_secret_value() if isinstance(value, SecretStr) else value
            for name, value in auth.__dict__.items()
            if value is not None
        }
        if isinstance(auth, SnowflakeKeyPairAuth) and auth.private_key is not None:
            result["private_key"] = auth._private_key_der
            result.pop("private_key_password", None)
        return result

    def _open_connection(self) -> Any:
        """Import the optional official driver and open a verified TLS connection."""
        connection = self.config.connection
        if not isinstance(connection, SnowflakeConnection):
            raise ConnectorError("Snowflake connection settings are required")
        authentication = self._get_authentication_parameters()
        try:
            import snowflake.connector
        except ImportError:
            raise ConnectorError(
                "install nori-rel[snowflake] to use Snowflake"
            ) from None
        try:
            return snowflake.connector.connect(
                **authentication,
                account=connection.account,
                warehouse=connection.warehouse,
                database=connection.database,
                schema=self.config.schema_name,
                role=connection.role,
                paramstyle="qmark",
                login_timeout=10,
                socket_timeout=10,
                network_timeout=math.ceil(self.settings.statement_timeout_ms / 1000),
                disable_ocsp_checks=False,
                client_session_keep_alive=False,
                client_prefetch_threads=1,
                client_fetch_use_mp=False,
                session_parameters={
                    "TIMEZONE": "UTC",
                    "STATEMENT_TIMEOUT_IN_SECONDS": max(
                        1, self.settings.statement_timeout_ms // 1000
                    ),
                    "QUERY_TAG": "nori-rel",
                },
            )
        except Exception:
            raise ConnectorError("Snowflake connection failed") from None

    def _configure_connection(self, connection: Any) -> None:
        """Disable secondary roles and enforce UTC and a bounded query timeout."""
        cursor = None
        try:
            cursor = connection.cursor()
            cursor.execute("USE SECONDARY ROLES NONE")
            cursor.execute(
                "ALTER SESSION SET TIMEZONE='UTC', STATEMENT_TIMEOUT_IN_SECONDS="
                f"{max(1, self.settings.statement_timeout_ms // 1000)}"
            )
        except Exception:
            raise ConnectorError("Snowflake session configuration failed") from None
        finally:
            if cursor is not None:
                with suppress(Exception):
                    cursor.close()

    def _discover_schema(
        self,
        connection: Any,
        database_id: str,
        *,
        time_columns: dict[str, str] | None = None,
    ) -> SchemaGraph:
        """Discover exact column names and declared keys for standard tables."""
        settings = self.config.connection
        if not isinstance(settings, SnowflakeConnection):
            raise ConnectorError("Snowflake connection settings are required")
        scope = ".".join(
            self._quote_identifier(part)
            for part in (settings.database, self.config.schema_name)
        )
        metadata_limit = self.settings.max_rows_per_table
        tables = dataframe_records_by_lowercase_column(
            self._get_query_frame(
                connection, f"SHOW TABLES IN SCHEMA {scope}", max_rows=metadata_limit
            )
        )
        available = {row["name"]: row for row in tables}
        selected = (
            [
                _get_matching_identifier(name, list(available), "table")
                for name in self.config.tables
            ]
            if self.config.tables
            else sorted(available)
        )
        if len(set(selected)) != len(selected):
            raise ConnectorError(
                "configured Snowflake tables resolve to the same physical name"
            )
        self._row_counts_by_table = {
            name: int(available[name]["rows"])
            for name in selected
            if isinstance(available[name].get("rows"), Integral)
            and not isinstance(available[name]["rows"], bool)
            and available[name]["rows"] >= 0
        }
        if len(selected) > self.settings.max_tables:
            raise SnapshotLimitError("schema exceeds the configured table limit")
        for name in selected:
            row = available[name]
            if row.get("kind") not in {"TABLE", "TRANSIENT"} or any(
                str(row.get(flag, "N")).upper() in {"Y", "YES", "TRUE"}
                for flag in (
                    "is_external",
                    "is_event",
                    "is_hybrid",
                    "is_iceberg",
                    "is_dynamic",
                )
            ):
                raise ConnectorError(f"Snowflake requires standard tables: {name}")
            if int(row.get("retention_time", 0)) < 1:
                raise ConnectorError(
                    f"Snowflake Time Travel retention must be enabled: {name}"
                )
        info = self._quote_identifier(settings.database) + '."INFORMATION_SCHEMA"'
        columns = dataframe_records_by_lowercase_column(
            self._get_query_frame(
                connection,
                "SELECT TABLE_NAME,COLUMN_NAME,DATA_TYPE,IS_NULLABLE,ORDINAL_POSITION "
                f"FROM {info}.COLUMNS WHERE TABLE_SCHEMA=%(schema)s",
                {"schema": self.config.schema_name},
                max_rows=metadata_limit,
            )
        )
        pk = dataframe_records_by_lowercase_column(
            self._get_query_frame(
                connection,
                f"SHOW PRIMARY KEYS IN SCHEMA {scope}",
                max_rows=metadata_limit,
            )
        )
        raw_fk = dataframe_records_by_lowercase_column(
            self._get_query_frame(
                connection,
                f"SHOW IMPORTED KEYS IN SCHEMA {scope}",
                max_rows=metadata_limit,
            )
        )
        fks = [
            {
                "table_name": row["fk_table_name"],
                "constraint_name": row["fk_name"],
                "column_name": row["fk_column_name"],
                "referred_table": row["pk_table_name"],
                "referred_column": row["pk_column_name"],
                "key_sequence": row["key_sequence"],
            }
            for row in raw_fk
            if row["pk_database_name"] == settings.database
            and row["pk_schema_name"] == self.config.schema_name
        ]
        normalized_times = self._normalize_configured_identifiers(
            selected, columns, time_columns
        )
        return build_warehouse_schema_graph(
            self.config,
            database_id,
            selected,
            columns,
            pk,
            fks,
            ConnectorCapabilities(
                server_side_cursor=False,
                snapshot_mode="time_travel",
                read_only_mode="role",
                unquoted_identifier_case="upper",
            ),
            normalized_times,
        )

    def _normalize_configured_identifiers(
        self,
        selected: list[str],
        column_rows: list[dict[str, Any]],
        time_columns: dict[str, str] | None,
    ) -> dict[str, str]:
        """Resolve configured keys and per-call times to physical metadata names."""
        columns_by_table = {
            name: [
                row["column_name"] for row in column_rows if row["table_name"] == name
            ]
            for name in {row["table_name"] for row in column_rows}
        }

        def get_table(name: str) -> str:
            return _get_matching_identifier(name, selected, "table")

        def get_column(table: str, name: str) -> str:
            return _get_matching_identifier(name, columns_by_table[table], "column")

        def get_times(mapping: dict[str, str]) -> dict[str, str]:
            result = {}
            for name, column in mapping.items():
                physical = get_table(name)
                if physical in result:
                    raise ConnectorError(
                        "time mappings resolve to the same Snowflake table"
                    )
                result[physical] = get_column(physical, column)
            return result

        primary_keys = {}
        for name, keys in self.config.primary_keys.items():
            physical = get_table(name)
            if physical in primary_keys:
                raise ConnectorError("primary keys resolve to the same Snowflake table")
            primary_keys[physical] = [get_column(physical, key) for key in keys]
        foreign_keys = {}
        for name, keys in self.config.foreign_keys.items():
            physical = get_table(name)
            if physical in foreign_keys:
                raise ConnectorError("foreign keys resolve to the same Snowflake table")
            normalized = []
            for key in keys:
                visible_parents = list(columns_by_table)
                parent_matches = [
                    name
                    for name in visible_parents
                    if name.upper() == key.referred_table.upper()
                ]
                parent = (
                    _get_matching_identifier(
                        key.referred_table, visible_parents, "table"
                    )
                    if parent_matches
                    else key.referred_table
                )
                normalized.append(
                    key.model_copy(
                        update={
                            "columns": [
                                get_column(physical, column) for column in key.columns
                            ],
                            "referred_table": parent,
                            "referred_columns": [
                                (
                                    get_column(parent, column)
                                    if parent in columns_by_table
                                    else column
                                )
                                for column in key.referred_columns
                            ],
                        }
                    )
                )
            foreign_keys[physical] = normalized
        normalized_times = get_times(time_columns or {})
        self.config = self.config.model_copy(
            update={
                "tables": selected if self.config.tables is not None else None,
                "time_columns": get_times(self.config.time_columns),
                "primary_keys": primary_keys,
                "foreign_keys": foreign_keys,
            }
        )
        return normalized_times

    def _get_table_row_counts(
        self, connection: Any, schema: SchemaGraph
    ) -> dict[str, int]:
        """Reuse cheap SHOW TABLES counts before any whole-source key checks."""
        counts = getattr(self, "_row_counts_by_table", {})
        return {
            table.name: counts[table.name]
            for table in schema.tables
            if table.name in counts
        }

    def _pin_tables(
        self, connection: Any, schema: SchemaGraph
    ) -> tuple[datetime, dict[str, str]]:
        """Use one server timestamp, never a moving OFFSET, for every table read."""
        timestamp = self._get_query_frame(
            connection, "SELECT CURRENT_TIMESTAMP()", max_rows=1
        )
        pinned_at = normalize_warehouse_timestamp_utc(timestamp.iloc[0, 0])
        literal = pinned_at.isoformat(timespec="microseconds")
        references = {
            table.name: self._get_qualified_table_reference(table.name)
            + f" AT (TIMESTAMP => '{literal}'::TIMESTAMP_TZ)"
            for table in schema.tables
        }
        return pinned_at, references
