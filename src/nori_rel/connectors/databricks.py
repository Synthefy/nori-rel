"""Unity Catalog Delta discovery and bounded reads of pinned table versions."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from datetime import datetime
from numbers import Integral
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    TypeAdapter,
    ValidationError,
)

from nori_rel.connectors.errors import ConnectorError, SnapshotLimitError
from nori_rel.connectors.warehouse import (
    WarehouseConnector,
    build_warehouse_schema_graph,
    dataframe_records_by_lowercase_column,
    normalize_warehouse_timestamp_utc,
)
from nori_rel.contracts import ConnectorCapabilities, DatabricksConnection, SchemaGraph


class _DatabricksAuthModel(BaseModel):
    """Forbid driver overrides and conceal credential inputs in diagnostics."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, hide_input_in_errors=True
    )


class DatabricksTokenAuth(_DatabricksAuthModel):
    """An explicit personal access token for the configured workspace."""

    authenticator: Literal["pat"] = "pat"
    access_token: SecretStr = Field(min_length=1)


class DatabricksOAuthM2MAuth(_DatabricksAuthModel):
    """A workspace OAuth machine identity with both client credentials."""

    authenticator: Literal["oauth_m2m"] = "oauth_m2m"
    client_id: SecretStr = Field(min_length=1)
    client_secret: SecretStr = Field(min_length=1)


DatabricksAuth = Annotated[
    DatabricksTokenAuth | DatabricksOAuthM2MAuth,
    Field(discriminator="authenticator"),
]
_DATABRICKS_AUTH_ADAPTER = TypeAdapter(DatabricksAuth)
_DATABRICKS_AUTH_FIELDS = frozenset().union(
    DatabricksTokenAuth.model_fields, DatabricksOAuthM2MAuth.model_fields
)


def _get_safe_authentication_error(error: ValidationError) -> str:
    """Describe missing or mixed fields without credential values or arbitrary keys."""
    diagnostics = []
    for detail in error.errors(
        include_input=False, include_context=False, include_url=False
    ):
        field = next(
            (
                part
                for part in reversed(detail["loc"])
                if part in _DATABRICKS_AUTH_FIELDS
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
        else:
            diagnostic = f"{field or 'credential field'}: must be a nonempty string"
        if diagnostic not in diagnostics:
            diagnostics.append(diagnostic)
    return "Databricks authentication invalid: " + "; ".join(diagnostics)


class DatabricksConnector(WarehouseConnector):
    """Read Unity Catalog Delta tables through a Databricks SQL warehouse."""

    @staticmethod
    def _quote_identifier(name: str) -> str:
        """Quote exact Unity Catalog identifiers and escape compiler percents."""
        if not name or "\x00" in name:
            raise ConnectorError("invalid warehouse identifier")
        return "`" + name.replace("`", "``").replace("%", "%%") + "`"

    def _get_qualified_table_reference(self, name: str) -> str:
        """Use an explicit catalog, schema, and table for every source read."""
        settings = self.config.connection
        if not isinstance(settings, DatabricksConnection):
            raise ConnectorError("Databricks connection settings are required")
        return ".".join(
            self._quote_identifier(part)
            for part in (settings.catalog, self.config.schema_name, name)
        )

    def _open_connection(self) -> Any:
        """Open the optional native connector using PAT or OAuth service identity."""
        settings = self.config.connection
        if not isinstance(settings, DatabricksConnection):
            raise ConnectorError("Databricks connection settings are required")
        secret = self._get_resolved_auth_fields()
        inferred = "pat" if "access_token" in secret else "oauth_m2m"
        try:
            auth = _DATABRICKS_AUTH_ADAPTER.validate_python(
                {"authenticator": inferred, **secret}
            )
        except ValidationError as error:
            raise ConnectorError(_get_safe_authentication_error(error)) from None
        try:
            from databricks import sql
        except ImportError:
            raise ConnectorError(
                "install nori-rel[databricks] to use Databricks"
            ) from None
        authentication: dict[str, Any]
        if isinstance(auth, DatabricksTokenAuth):
            authentication = {"access_token": auth.access_token.get_secret_value()}
        else:
            try:
                from databricks.sdk.core import Config, oauth_service_principal
            except ImportError:
                raise ConnectorError(
                    "install nori-rel[databricks] for OAuth authentication"
                ) from None

            def credential_provider():
                config = Config(
                    host=f"https://{settings.server_hostname}",
                    client_id=auth.client_id.get_secret_value(),
                    client_secret=auth.client_secret.get_secret_value(),
                    auth_type="oauth-m2m",
                )
                return oauth_service_principal(config)

            authentication = {"credentials_provider": credential_provider}
        try:
            return sql.connect(
                server_hostname=settings.server_hostname,
                http_path=settings.http_path,
                catalog=settings.catalog,
                schema=self.config.schema_name,
                user_agent_entry="nori-rel",
                use_inline_params=False,
                use_cloud_fetch=False,
                _tls_no_verify=False,
                _tls_verify_hostname=True,
                _socket_timeout=10,
                _retry_stop_after_attempts_count=2,
                **authentication,
            )
        except Exception:
            raise ConnectorError("Databricks connection failed") from None

    def _get_query_cancellation_callback(
        self, cursor: Any
    ) -> Callable[[], None] | None:
        """Cancel only the active query using the native SQL cursor API."""
        callback = getattr(cursor, "cancel", None)
        return callback if callable(callback) else None

    def _configure_connection(self, connection: Any) -> None:
        """Set UTC and enforce the warehouse's session statement timeout."""
        cursor = None
        try:
            cursor = connection.cursor()
            cursor.execute("SET TIME ZONE 'UTC'")
            timeout = max(1, min(172800, self.settings.statement_timeout_ms // 1000))
            cursor.execute(f"SET STATEMENT_TIMEOUT = {timeout}")
        except Exception:
            raise ConnectorError("Databricks session configuration failed") from None
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
        """Discover Delta tables and declared Unity Catalog primary/foreign keys."""
        settings = self.config.connection
        if not isinstance(settings, DatabricksConnection):
            raise ConnectorError("Databricks connection settings are required")
        info = self._quote_identifier(settings.catalog) + ".information_schema"
        params = {"schema": self.config.schema_name}
        metadata_limit = self.settings.max_rows_per_table
        raw_tables = dataframe_records_by_lowercase_column(
            self._get_query_frame(
                connection,
                f"SELECT table_name,table_type,data_source_format FROM {info}.tables "
                "WHERE table_schema=%(schema)s",
                params,
                max_rows=metadata_limit,
            )
        )
        available = {row["table_name"]: row for row in raw_tables}
        supported = {
            name
            for name, row in available.items()
            if str(row["data_source_format"]).upper() == "DELTA"
            and row["table_type"] in {"MANAGED", "EXTERNAL"}
        }
        selected = self.config.tables or sorted(supported)
        missing = set(selected).difference(supported)
        if missing:
            raise ConnectorError(
                f"Databricks requires Unity Catalog Delta tables: {sorted(missing)}"
            )
        if len(selected) > self.settings.max_tables:
            raise SnapshotLimitError("schema exceeds the configured table limit")
        columns = dataframe_records_by_lowercase_column(
            self._get_query_frame(
                connection,
                "SELECT table_name,column_name,full_data_type AS data_type,"
                "is_nullable,ordinal_position "
                f"FROM {info}.columns WHERE table_schema=%(schema)s",
                params,
                max_rows=metadata_limit,
            )
        )
        pk = dataframe_records_by_lowercase_column(
            self._get_query_frame(
                connection,
                "SELECT k.table_name,k.column_name,k.ordinal_position AS key_sequence "
                f"FROM {info}.table_constraints c JOIN {info}.key_column_usage k "
                "ON c.constraint_catalog=k.constraint_catalog "
                "AND c.constraint_schema=k.constraint_schema "
                "AND c.constraint_name=k.constraint_name "
                "WHERE c.table_schema=%(schema)s AND c.constraint_type='PRIMARY KEY'",
                params,
                max_rows=metadata_limit,
            )
        )
        fks = dataframe_records_by_lowercase_column(
            self._get_query_frame(
                connection,
                "SELECT k.table_name,k.constraint_name,k.column_name,"
                "k.ordinal_position AS key_sequence, "
                "p.table_name AS referred_table,p.column_name AS referred_column "
                f"FROM {info}.referential_constraints r JOIN {info}.key_column_usage k "
                "ON r.constraint_catalog=k.constraint_catalog "
                "AND r.constraint_schema=k.constraint_schema "
                "AND r.constraint_name=k.constraint_name "
                f"JOIN {info}.key_column_usage p "
                "ON r.unique_constraint_catalog=p.constraint_catalog "
                "AND r.unique_constraint_schema=p.constraint_schema "
                "AND r.unique_constraint_name=p.constraint_name "
                "AND k.position_in_unique_constraint=p.ordinal_position "
                "WHERE k.table_schema=%(schema)s AND p.table_schema=%(schema)s",
                params,
                max_rows=metadata_limit,
            )
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
                snapshot_mode="table_versions",
                read_only_mode="role",
            ),
            time_columns,
        )

    def _pin_tables(
        self, connection: Any, schema: SchemaGraph
    ) -> tuple[datetime, dict[str, str]]:
        """Resolve each table's last Delta commit at one fixed server timestamp."""
        timestamp = self._get_query_frame(
            connection, "SELECT CURRENT_TIMESTAMP()", max_rows=1
        )
        pinned_at = normalize_warehouse_timestamp_utc(timestamp.iloc[0, 0])
        references = {}
        for table in schema.tables:
            qualified = self._get_qualified_table_reference(table.name)
            history_limit = 1
            while True:
                history = dataframe_records_by_lowercase_column(
                    self._get_query_frame(
                        connection,
                        f"DESCRIBE HISTORY {qualified} LIMIT {history_limit}",
                        max_rows=history_limit,
                    )
                )
                versions = []
                for row in history:
                    version = row["version"]
                    if (
                        not isinstance(version, Integral)
                        or isinstance(version, bool)
                        or version < 0
                    ):
                        raise ConnectorError(
                            "Databricks returned an invalid Delta version"
                        )
                    if normalize_warehouse_timestamp_utc(row["timestamp"]) <= pinned_at:
                        versions.append(int(version))
                if versions:
                    references[table.name] = (
                        f"{qualified} VERSION AS OF {max(versions)}"
                    )
                    break
                if len(history) < history_limit:
                    raise ConnectorError(
                        "no retained Delta version at extraction timestamp: "
                        f"{table.name}"
                    )
                if history_limit >= self.settings.max_rows_per_table:
                    raise SnapshotLimitError(
                        "Delta history search exceeds the configured row limit"
                    )
                # A racing commit requires more history from this table only.
                history_limit = min(
                    self.settings.max_rows_per_table, max(16, history_limit * 2)
                )
        return pinned_at, references

    def _get_table_identity_by_name(
        self, connection: Any, schema: SchemaGraph
    ) -> dict[str, str]:
        """Detect table replacement that reuses a name and resets Delta versions."""
        identities = {}
        for table in schema.tables:
            qualified = self._get_qualified_table_reference(table.name)
            rows = dataframe_records_by_lowercase_column(
                self._get_query_frame(
                    connection,
                    f"DESCRIBE DETAIL {qualified}",
                    max_rows=1,
                )
            )
            if (
                len(rows) != 1
                or not isinstance(rows[0].get("id"), str)
                or not rows[0]["id"]
            ):
                raise ConnectorError(
                    "Databricks returned an invalid Delta table identity"
                )
            if str(rows[0].get("format", "")).upper() != "DELTA":
                raise ConnectorError("Databricks source is no longer a Delta table")
            identities[table.name] = rows[0]["id"]
        return identities
