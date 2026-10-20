"""Bounded relational execution against warehouse snapshot-pinned tables.

Warehouse roles must have only read privileges. SQL emitted by this executor is
read-only; it does not create temporary tables or use writable transactions.
Extraction versions and event-time cutoffs are deliberately separate concepts.
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal
from threading import Event, RLock, Timer
from typing import Any

import pandas as pd

from nori_rel._credentials import CredentialResolver
from nori_rel.connectors.base import (
    Snapshot,
    SnapshotProvenance,
    get_context_read_row_cap,
)
from nori_rel.connectors.errors import ConnectorError, SnapshotLimitError
from nori_rel.contracts import (
    ColumnSchema,
    ConnectorCapabilities,
    DatabaseCreateRequest,
    DatabaseEngine,
    DatabricksConnection,
    ForeignKeySchema,
    SchemaGraph,
    SnowflakeConnection,
    TableSchema,
)
from nori_rel.limits import RelationalLimits
from nori_rel.pql import parse_pql
from nori_rel.pql.context_execution import compile_prepared_context_sql
from nori_rel.pql.context_sql import DEFAULT_MAX_CONTEXT_ROWS, _utc
from nori_rel.pql.dialects import SQLDialect
from nori_rel.pql.scoring_sql import compile_scoring_filter_sql
from nori_rel.pql.sql_expressions import column_kind

_PYFORMAT_PARAMETER = re.compile(r"%%|%\(([^)]+)\)s")
_SQL_QUOTED_TEXT = re.compile(r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|`(?:``|[^`])*`")
_SQL_WRITE_KEYWORDS = re.compile(
    r"\b(?:INSERT|UPDATE|DELETE|MERGE|CREATE|ALTER|DROP|TRUNCATE|GRANT|REVOKE|"
    r"CALL|COPY|PUT|GET|EXECUTE)\b",
    re.IGNORECASE,
)
_CONTEXT_COLUMNS = ["context_row_id", "entity_id", "anchor_time", "label"]


def bind_native_parameters(
    sql: str,
    parameters: Mapping[str, Any] | None = None,
    *,
    engine: DatabaseEngine | None = None,
) -> tuple[str, list[Any]]:
    """Translate trusted compiler placeholders into qmark native parameters.

    Escaped identifier percents are unescaped exactly once. Values are passed to
    the driver separately, including repeated values, and never interpolated.
    """
    values: list[Any] = []
    parameters = parameters or {}

    def replace_parameter(match: re.Match) -> str:
        if match[0] == "%%":
            return "%"
        name = match[1]
        if name not in parameters:
            raise ValueError("SQL parameter is missing")
        value = parameters[name]
        if engine is not None and (
            isinstance(value, Decimal)
            or (type(value) is int and not -(2**63) <= value < 2**63)
        ):
            if isinstance(value, Decimal) and not value.is_finite():
                raise ValueError("warehouse numeric parameters must be finite")
            # Native Databricks inference misreads exponent-form Decimals and
            # binds every Python int as BIGINT. Bind exact text, then cast in SQL.
            expression = SQLDialect(engine).numeric_cast("?", literal=value)
            values.append(format(Decimal(value), "f"))
            return expression
        values.append(value)
        return "?"

    return _PYFORMAT_PARAMETER.sub(replace_parameter, sql), values


def dataframe_records_by_lowercase_column(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Normalize metadata column names without changing customer identifiers."""
    frame = frame.rename(columns=lambda name: str(name).lower())
    return frame.to_dict(orient="records")


def normalize_warehouse_timestamp_utc(value: Any) -> datetime:
    """Validate a server timestamp and retain its timezone-independent instant."""
    try:
        timestamp = pd.Timestamp(value)
        if pd.isna(timestamp):
            raise ValueError
        timestamp = (
            timestamp.tz_localize("UTC")
            if timestamp.tzinfo is None
            else timestamp.tz_convert("UTC")
        )
        return timestamp.to_pydatetime()
    except (TypeError, ValueError, OverflowError):
        raise ConnectorError(
            "warehouse returned an invalid snapshot timestamp"
        ) from None


def build_warehouse_schema_graph(
    config: DatabaseCreateRequest,
    database_id: str,
    selected_tables: list[str],
    column_rows: list[dict[str, Any]],
    primary_rows: list[dict[str, Any]],
    foreign_rows: list[dict[str, Any]],
    capabilities: ConnectorCapabilities,
    time_columns: dict[str, str] | None = None,
) -> SchemaGraph:
    """Build a graph from normalized warehouse metadata and registered times."""
    time_column_by_table = {**config.time_columns, **(time_columns or {})}
    tables = []
    for table_name in selected_tables:
        raw_columns = sorted(
            (row for row in column_rows if row["table_name"] == table_name),
            key=lambda row: int(row["ordinal_position"]),
        )
        if not raw_columns:
            raise ConnectorError(f"no visible columns for table {table_name}")
        time_column = time_column_by_table.get(table_name)
        if time_column is not None:
            matching = [row for row in raw_columns if row["column_name"] == time_column]
            if not matching:
                raise ConnectorError(
                    f"time column {table_name}.{time_column} does not exist"
                )
            if (
                not str(matching[0]["data_type"])
                .upper()
                .startswith(("DATE", "TIMESTAMP"))
            ):
                raise ConnectorError(
                    f"time column {table_name}.{time_column} is not temporal"
                )
        primary_key_rows = sorted(
            (row for row in primary_rows if row["table_name"] == table_name),
            key=lambda row: int(row["key_sequence"]),
        )
        foreign_rows_by_constraint_and_parent: dict[
            tuple[str, str], list[dict[str, Any]]
        ] = defaultdict(list)
        for row in foreign_rows:
            if row["table_name"] == table_name:
                foreign_rows_by_constraint_and_parent[
                    (row["constraint_name"], row["referred_table"])
                ].append(row)
        table_foreign_keys = []
        for (name, parent), rows in foreign_rows_by_constraint_and_parent.items():
            rows.sort(key=lambda row: int(row["key_sequence"]))
            table_foreign_keys.append(
                ForeignKeySchema(
                    name=name,
                    columns=[row["column_name"] for row in rows],
                    referred_table=parent,
                    referred_columns=[row["referred_column"] for row in rows],
                )
            )
        tables.append(
            TableSchema(
                name=table_name,
                columns=[
                    ColumnSchema(
                        name=row["column_name"],
                        data_type=row["data_type"],
                        nullable=str(row["is_nullable"]).upper() == "YES",
                    )
                    for row in raw_columns
                ],
                primary_key=[row["column_name"] for row in primary_key_rows],
                foreign_keys=table_foreign_keys,
                time_column=time_column,
            )
        )
    return SchemaGraph(
        database_id=database_id,
        schema_name=config.schema_name,
        tables=tables,
        capabilities=capabilities,
        discovered_at=datetime.now(UTC),
    )


@dataclass(frozen=True)
class WarehouseSession:
    """Named handles and source identities for one serialized pinned extraction."""

    connection: Any
    schema: SchemaGraph
    dialect: SQLDialect
    pinned_at: datetime
    table_references: dict[str, str]
    row_counts_by_table: dict[str, int]


class WarehouseConnector:
    """Share discovery validation, bounded reads, PQL preparation, and cleanup.

    A source owns one reusable driver connection. Operations are serialized;
    each preparation pins a fresh set of source references. Driver subclasses
    implement connection, session configuration, metadata, and version pinning.
    """

    def __init__(
        self,
        config: DatabaseCreateRequest,
        settings: RelationalLimits,
        resolver: CredentialResolver | None = None,
    ) -> None:
        """Retain secret references without resolving or caching their values."""
        self.config = config
        self.settings = settings
        self.resolver = resolver or CredentialResolver()
        self._connection: Any = None
        self._operation_lock = RLock()
        self._closed = False

    def _get_resolved_auth_fields(self) -> dict[str, str]:
        """Resolve warehouse secret fields locally without retaining the values."""
        return self.resolver.resolve_parameters(self.config.credential)

    def _quote_identifier(self, name: str) -> str:
        """Quote a literal identifier using the engine's SQL delimiter rules."""
        return SQLDialect(self.config.engine).identifier(name)

    def _get_qualified_table_reference(self, name: str) -> str:
        """Return a fully qualified, escaped reference using literal source names."""
        connection = self.config.connection
        if isinstance(connection, SnowflakeConnection):
            catalog = connection.database
        elif isinstance(connection, DatabricksConnection):
            catalog = connection.catalog
        else:
            raise ConnectorError("warehouse connection settings are missing")
        return SQLDialect(self.config.engine).qualified_name(
            catalog, self.config.schema_name, name
        )

    def _open_connection(self) -> Any:
        """Open the engine's native connection with bounded driver settings."""
        raise NotImplementedError

    def _configure_connection(self, connection: Any) -> None:
        """Configure the reader session, including its server statement timeout."""
        raise NotImplementedError

    def _discover_schema(
        self, connection: Any, database_id: str, *, time_columns=None
    ) -> SchemaGraph:
        """Read literal schema metadata and merge registered event-time columns."""
        raise NotImplementedError

    def _pin_tables(
        self, connection: Any, schema: SchemaGraph
    ) -> tuple[datetime, dict[str, str]]:
        """Return one extraction instant and a pinned SQL reference per table."""
        raise NotImplementedError

    def _get_table_identity_by_name(
        self, connection: Any, schema: SchemaGraph
    ) -> dict[str, str]:
        """Return physical table IDs when version references depend on names."""
        return {}

    def _get_table_row_counts(
        self, connection: Any, schema: SchemaGraph
    ) -> dict[str, int]:
        """Return cheap metadata row counts; missing tables use bounded probes.

        Engines with counts in discovery metadata should override this hook.
        Estimates can reject oversized sources early; they never replace the
        row and memory limits enforced while fetching the pinned source.
        """
        return {}

    def _get_query_cancellation_callback(
        self, cursor: Any
    ) -> Callable[[], None] | None:
        """Return a supported query-only cancellation callback, if available.

        The default relies on the engine's configured server statement timeout.
        A driver may provide a thread-safe query cancellation method; the timer
        must never close the connector's shared connection.
        """
        return None

    def _get_connection(self) -> Any:
        """Open and configure the source connection once, then reuse it serially."""
        if self._closed:
            raise ConnectorError("warehouse connector is closed")
        if self._connection is None:
            connection = self._open_connection()
            try:
                self._configure_connection(connection)
            except Exception:
                with suppress(Exception):
                    connection.close()
                raise
            self._connection = connection
        return self._connection

    def _discard_connection(self) -> None:
        connection, self._connection = self._connection, None
        if connection is not None:
            with suppress(Exception):
                connection.close()

    def close(self) -> None:
        """Release the driver connection after active work; safe to repeat."""
        with self._operation_lock:
            self._closed = True
            self._discard_connection()

    def _get_query_frame(
        self,
        connection: Any,
        sql: str,
        params: Mapping[str, Any] | None = None,
        *,
        max_rows: int | None = None,
        max_bytes: int | None = None,
    ) -> pd.DataFrame:
        """Execute generated read SQL with native bindings and bounded fetching."""
        unquoted = _SQL_QUOTED_TEXT.sub("", sql).strip()
        if (
            not re.match(r"^(SELECT|WITH|SHOW|DESCRIBE|DESC)\b", unquoted, re.I)
            or ";" in unquoted
            or _SQL_WRITE_KEYWORDS.search(unquoted)
        ):
            raise ConnectorError(
                "warehouse executor accepts generated read statements only"
            )
        statement, parameters = bind_native_parameters(
            sql, params, engine=self.config.engine
        )
        if self.config.engine is DatabaseEngine.SNOWFLAKE:
            parameters = [
                ("TIMESTAMP_TZ", value)
                if isinstance(value, datetime) and value.utcoffset() is not None
                else value
                for value in parameters
            ]
        ceiling = self.settings.max_rows_per_table if max_rows is None else max_rows
        byte_limit = (
            self.settings.max_snapshot_bytes if max_bytes is None else max_bytes
        )
        if ceiling < 0 or byte_limit < 0:
            raise SnapshotLimitError(
                "warehouse query exceeds remaining resource limits"
            )
        cursor = None
        timer = None
        timed_out = Event()
        frames, row_count, byte_count = [], 0, 0
        try:
            if self.config.engine is DatabaseEngine.DATABRICKS:
                cursor = connection.cursor(
                    arraysize=min(1000, ceiling + 1),
                    buffer_size_bytes=max(1, min(byte_limit, 1_000_000)),
                    row_limit=ceiling + 1,
                )
            else:
                cursor = connection.cursor()

            cancel_callback = self._get_query_cancellation_callback(cursor)

            def cancel_query() -> None:
                timed_out.set()
                with suppress(Exception):
                    cancel_callback()

            if cancel_callback is not None:
                timer = Timer(self.settings.statement_timeout_ms / 1000, cancel_query)
                timer.daemon = True
                timer.start()
            cursor.execute(statement, parameters)
            columns = [description[0] for description in cursor.description or []]
            while batch := cursor.fetchmany(min(1000, ceiling + 1)):
                if timed_out.is_set():
                    raise ConnectorError("warehouse query timed out")
                row_count += len(batch)
                if row_count > ceiling:
                    raise SnapshotLimitError(
                        "warehouse query exceeds the configured row limit"
                    )
                frame = pd.DataFrame(batch, columns=columns, dtype=object)
                byte_count += int(frame.memory_usage(index=True, deep=True).sum())
                if byte_count > byte_limit:
                    raise SnapshotLimitError(
                        "warehouse query exceeds the configured memory limit"
                    )
                frames.append(frame)
            if timed_out.is_set():
                raise ConnectorError("warehouse query timed out")
            return (
                pd.concat(frames, ignore_index=True)
                if frames
                else pd.DataFrame(columns=columns, dtype=object)
            )
        except ConnectorError:
            raise
        except Exception as driver_error:
            reason = (
                "warehouse query timed out"
                if timed_out.is_set()
                else "warehouse query failed"
            )
            driver_error_code = getattr(driver_error, "errno", None)
            driver_sql_state = getattr(driver_error, "sqlstate", None)
            # Preserve nonsecret vendor codes without exposing driver messages,
            # parameters, connection details, or credential-bearing tracebacks.
            diagnostics = []
            if type(driver_error_code) is int:
                diagnostics.append(f"code={driver_error_code}")
            if isinstance(driver_sql_state, str) and re.fullmatch(
                r"[A-Za-z0-9]{5}", driver_sql_state
            ):
                diagnostics.append(f"sqlstate={driver_sql_state}")
            if diagnostics:
                reason += " (" + ", ".join(diagnostics) + ")"
            raise ConnectorError(reason) from None
        finally:
            if timer is not None:
                timer.cancel()
            if cursor is not None:
                with suppress(Exception):
                    cursor.close()

    def test(self) -> float:
        """Check live connectivity without resolving tables or reading their rows."""
        started = time.monotonic()
        with self._operation_lock:
            try:
                self._get_query_frame(self._get_connection(), "SELECT 1", max_rows=1)
            except Exception:
                self._discard_connection()
                raise
        return (time.monotonic() - started) * 1000

    def _apply_relationships(self, schema: SchemaGraph) -> SchemaGraph:
        """Merge explicit mappings and validate referenced literal column names."""
        tables = {table.name: table for table in schema.tables}
        unknown = (
            set(self.config.primary_keys) | set(self.config.foreign_keys)
        ).difference(tables)
        if unknown:
            raise ConnectorError("relationship mapping names an undiscovered table")
        for table in schema.tables:
            if table.name in self.config.primary_keys:
                table.primary_key = list(self.config.primary_keys[table.name])
            if table.name in self.config.foreign_keys:
                table.foreign_keys = [
                    key.model_copy(deep=True)
                    for key in self.config.foreign_keys[table.name]
                ]
            table.foreign_keys.sort(
                key=lambda key: (
                    key.referred_table,
                    tuple(key.columns),
                    tuple(key.referred_columns),
                    key.name or "",
                )
            )
            columns = {column.name for column in table.columns}
            if not set(table.primary_key).issubset(columns):
                raise ConnectorError("primary key mapping names an unknown column")
            for column in table.columns:
                if column.name in table.primary_key:
                    column.nullable = False
            for key in table.foreign_keys:
                parent = tables.get(key.referred_table)
                if parent is None:
                    # Relations outside the allowlist cannot be traversed.
                    continue
                if not set(key.columns).issubset(columns) or not set(
                    key.referred_columns
                ).issubset(column.name for column in parent.columns):
                    raise ConnectorError("foreign key mapping names an unknown column")
        return schema

    def discover(self, database_id: str) -> SchemaGraph:
        """Read metadata and explicit relationships without downloading tables."""
        with self._operation_lock:
            try:
                return self._apply_relationships(
                    self._discover_schema(self._get_connection(), database_id)
                )
            except Exception:
                self._discard_connection()
                raise

    @contextmanager
    def _session(
        self, database_id: str, *, time_columns=None
    ) -> Iterator[WarehouseSession]:
        """Pin source references after rejecting oversized metadata estimates."""
        with self._operation_lock:
            try:
                connection = self._get_connection()
                schema = self._apply_relationships(
                    self._discover_schema(
                        connection, database_id, time_columns=time_columns
                    )
                )
                row_counts_by_table = self._get_table_row_counts(connection, schema)
                self._check_estimated_table_sizes(schema, row_counts_by_table)
                identities = self._get_table_identity_by_name(connection, schema)
                pinned_at, references = self._pin_tables(connection, schema)
                pinned_at = _utc(pinned_at)
                if set(references) != {table.name for table in schema.tables}:
                    raise ConnectorError(
                        "warehouse snapshot must pin every registered table"
                    )
                dialect = SQLDialect(self.config.engine, table_references=references)
                yield WarehouseSession(
                    connection=connection,
                    schema=schema,
                    dialect=dialect,
                    pinned_at=pinned_at,
                    table_references=references,
                    row_counts_by_table=row_counts_by_table,
                )
                # Time Travel can expose current column definitions. Refuse schema
                # drift before returning results from mismatched metadata.
                current = self._apply_relationships(
                    self._discover_schema(
                        connection, database_id, time_columns=time_columns
                    )
                )
                if [table.model_dump() for table in current.tables] != [
                    table.model_dump() for table in schema.tables
                ]:
                    raise ConnectorError("warehouse schema changed during extraction")
                if self._get_table_identity_by_name(connection, current) != identities:
                    raise ConnectorError(
                        "warehouse table identity changed during extraction"
                    )
            except Exception:
                self._discard_connection()
                raise

    def _check_estimated_table_sizes(
        self, schema: SchemaGraph, row_counts_by_table: Mapping[str, int]
    ) -> None:
        """Reject excessive table counts or cheap size estimates before key scans."""
        if len(schema.tables) > self.settings.max_tables:
            raise SnapshotLimitError("warehouse exceeds the configured table limit")
        counts = [
            row_counts_by_table[table.name]
            for table in schema.tables
            if table.name in row_counts_by_table
        ]
        if any(type(count) is not int or count < 0 for count in counts):
            raise ConnectorError("warehouse returned an invalid table row count")
        if any(count > self.settings.max_rows_per_table for count in counts):
            raise SnapshotLimitError("warehouse table exceeds the configured row limit")
        if sum(counts) > self.settings.max_total_rows:
            raise SnapshotLimitError("warehouse exceeds the configured total row limit")

    @staticmethod
    def _get_visible_table_reference(
        schema: SchemaGraph, dialect: SQLDialect, table: TableSchema
    ) -> str:
        """Restrict registered event tables to the prediction's bound cutoff."""
        reference = dialect.table_reference(schema.schema_name, table.name)
        if table.time_column is None:
            return reference
        return (
            f"(SELECT * FROM {reference} "
            f"WHERE {dialect.identifier(table.time_column)} <= %(cutoff)s)"
        )

    def _validate_visible_table_keys(
        self, session: WarehouseSession, cutoff: datetime
    ) -> None:
        """Bound source sizes first, then check only prediction-visible key rows.

        Unknown metadata sizes use a LIMIT-bounded count probe instead of an
        unbounded COUNT. Key checks still perform GROUP BY/anti-join reads over
        the visible rows; server timeouts bound their execution time.
        """
        counts = dict(session.row_counts_by_table)
        for table in session.schema.tables:
            if table.name in counts:
                continue
            reference = self._get_visible_table_reference(
                session.schema, session.dialect, table
            )
            rows = self._get_query_frame(
                session.connection,
                "SELECT COUNT(*) AS row_count FROM "
                f"(SELECT 1 FROM {reference} AS visible_rows "
                f"LIMIT {self.settings.max_rows_per_table + 1}) AS bounded_rows",
                {"cutoff": cutoff},
                max_rows=1,
            )
            if len(rows) != 1:
                raise ConnectorError("warehouse returned an invalid table row count")
            try:
                count = int(rows.iloc[0, 0])
                if count != rows.iloc[0, 0]:
                    raise ValueError
            except (TypeError, ValueError, OverflowError):
                raise ConnectorError(
                    "warehouse returned an invalid table row count"
                ) from None
            counts[table.name] = count
            self._check_estimated_table_sizes(session.schema, counts)
        self._validate_primary_keys(
            session.connection, session.schema, session.dialect, cutoff
        )
        self._validate_foreign_keys(
            session.connection, session.schema, session.dialect, cutoff
        )

    def _validate_primary_keys(
        self,
        connection: Any,
        schema: SchemaGraph,
        dialect: SQLDialect,
        cutoff: datetime,
    ) -> None:
        """Require unique, non-null primary keys among cutoff-visible source rows."""
        for table in schema.tables:
            if not table.primary_key:
                continue
            reference = self._get_visible_table_reference(schema, dialect, table)
            keys = [dialect.identifier(key) for key in table.primary_key]
            nulls = " OR ".join(f"{key} IS NULL" for key in keys)
            grouped = ", ".join(keys)
            invalid = self._get_query_frame(
                connection,
                f"SELECT 1 AS invalid_key FROM {reference} AS visible_rows "
                f"GROUP BY {grouped} "
                f"HAVING COUNT(*) > 1 OR ({nulls}) LIMIT 1",
                {"cutoff": cutoff},
                max_rows=1,
            )
            if not invalid.empty:
                raise ConnectorError(
                    "warehouse primary keys must be unique and non-null"
                )
            for column in table.columns:
                if column.name in table.primary_key:
                    column.nullable = False

    def _validate_foreign_keys(
        self,
        connection: Any,
        schema: SchemaGraph,
        dialect: SQLDialect,
        cutoff: datetime,
    ) -> None:
        """Require non-null visible foreign keys to identify a visible parent."""
        tables = {table.name: table for table in schema.tables}
        for table in schema.tables:
            child = self._get_visible_table_reference(schema, dialect, table)
            for key in table.foreign_keys:
                if key.referred_table not in tables:
                    continue
                parent = self._get_visible_table_reference(
                    schema, dialect, tables[key.referred_table]
                )
                joined = " AND ".join(
                    f"c.{dialect.identifier(child_key)} = "
                    f"p.{dialect.identifier(parent_key)}"
                    for child_key, parent_key in zip(
                        key.columns, key.referred_columns, strict=True
                    )
                )
                present = " AND ".join(
                    f"c.{dialect.identifier(column)} IS NOT NULL"
                    for column in key.columns
                )
                missing = f"p.{dialect.identifier(key.referred_columns[0])} IS NULL"
                invalid = self._get_query_frame(
                    connection,
                    f"SELECT 1 AS invalid_relationship FROM {child} AS c "
                    f"LEFT JOIN {parent} AS p ON {joined} "
                    f"WHERE {present} AND {missing} LIMIT 1",
                    {"cutoff": cutoff},
                    max_rows=1,
                )
                if not invalid.empty:
                    raise ConnectorError(
                        "warehouse foreign key references a missing parent"
                    )

    def _get_scoring_anchor_time(
        self,
        connection: Any,
        schema: SchemaGraph,
        dialect: SQLDialect,
        anchor_time: datetime | None,
    ) -> datetime:
        """Validate an explicit cutoff or read the latest registered event time."""
        if anchor_time is not None:
            return _utc(anchor_time)
        latest = []
        for table in schema.tables:
            if table.time_column is None:
                continue
            rows = self._get_query_frame(
                connection,
                f"SELECT MAX({dialect.identifier(table.time_column)}) AS latest_time "
                f"FROM {dialect.table_reference(schema.schema_name, table.name)}",
                max_rows=1,
            )
            if not rows.empty and pd.notna(rows.iloc[0, 0]):
                stamp = pd.Timestamp(rows.iloc[0, 0])
                latest.append(
                    stamp.tz_localize("UTC")
                    if stamp.tzinfo is None
                    else stamp.tz_convert("UTC")
                )
        if not latest:
            raise ValueError("no registered event timestamps; provide anchor_time")
        return max(latest).to_pydatetime()

    @staticmethod
    def _normalize_table_frame(frame: pd.DataFrame, table: TableSchema) -> pd.DataFrame:
        """Normalize feature dtypes and UTC times without rounding relational IDs."""
        key_names = set(table.primary_key) | {
            column for key in table.foreign_keys for column in key.columns
        }
        for column in table.columns:
            try:
                kind = column_kind(column)
            except ValueError:
                continue
            values = frame[column.name]
            if kind in {"timestamp", "date"}:
                frame[column.name] = pd.to_datetime(values, utc=True)
            elif kind == "number":
                present = values.dropna()
                if column.name not in key_names and any(
                    isinstance(value, Decimal) for value in present
                ):
                    # Feature backends infer object Decimal columns as categories.
                    # Convert feature values only; relational IDs remain exact.
                    frame[column.name] = pd.array(
                        [
                            float(value) if pd.notna(value) else pd.NA
                            for value in values
                        ],
                        dtype="Float64",
                    )
                    continue
                if (
                    column.name in key_names
                    and len(present)
                    and all(
                        isinstance(value, Decimal)
                        and value == value.to_integral_value()
                        and -(2**63) <= value < 2**63
                        for value in present
                    )
                ):
                    frame[column.name] = pd.array(
                        [int(value) if pd.notna(value) else pd.NA for value in values],
                        dtype="Int64",
                    )
                    continue
                if len(present) and all(
                    type(value) is int and -(2**63) <= value < 2**63
                    for value in present
                ):
                    frame[column.name] = pd.array(values, dtype="Int64")
                elif not len(present) or all(type(value) is float for value in present):
                    frame[column.name] = pd.array(values, dtype="Float64")
            elif kind == "bool":
                frame[column.name] = pd.array(values, dtype="boolean")
        return frame

    def _get_snapshot(
        self,
        connection: Any,
        schema: SchemaGraph,
        dialect: SQLDialect,
        as_of: datetime,
        pinned_at: datetime,
        references: Mapping[str, str],
        *,
        remaining_bytes: int | None = None,
        remaining_rows: int | None = None,
    ) -> Snapshot:
        """Load cutoff-visible source frames within the remaining shared budget."""
        frames: dict[str, pd.DataFrame] = {}
        byte_count, row_count = 0, 0
        byte_limit = (
            self.settings.max_snapshot_bytes
            if remaining_bytes is None
            else remaining_bytes
        )
        row_limit = (
            self.settings.max_total_rows if remaining_rows is None else remaining_rows
        )
        for table in schema.tables:
            reference = dialect.table_reference(schema.schema_name, table.name)
            params = {}
            sql = f"SELECT * FROM {reference}"
            if table.time_column is not None:
                sql += f" WHERE {dialect.identifier(table.time_column)} <= %(cutoff)s"
                params["cutoff"] = as_of
            sql += f" LIMIT {self.settings.max_rows_per_table + 1}"
            frame = self._normalize_table_frame(
                self._get_query_frame(
                    connection,
                    sql,
                    params,
                    max_rows=min(
                        self.settings.max_rows_per_table, row_limit - row_count
                    ),
                    max_bytes=byte_limit - byte_count,
                ),
                table,
            )
            byte_count += int(frame.memory_usage(index=True, deep=True).sum())
            row_count += len(frame)
            if row_count > row_limit or byte_count > byte_limit:
                raise SnapshotLimitError(
                    "warehouse snapshot exceeds aggregate resource limits"
                )
            frames[table.name] = frame
        return Snapshot(
            tables=frames,
            primary_keys={table.name: table.primary_key for table in schema.tables},
            foreign_keys=[
                (
                    table.name,
                    key.columns[0],
                    key.referred_table,
                    key.referred_columns[0],
                )
                for table in schema.tables
                for key in table.foreign_keys
                if len(key.columns) == len(key.referred_columns) == 1
            ],
            time_columns={
                table.name: table.time_column
                for table in schema.tables
                if table.time_column is not None
            },
            schema=schema,
            as_of=as_of,
            approximate_bytes=byte_count,
            provenance=SnapshotProvenance(
                engine=self.config.engine,
                extraction_timestamp=pinned_at,
                table_references=dict(references),
            ),
        )

    def snapshot(
        self, database_id: str, as_of: datetime, *, time_columns=None
    ) -> Snapshot:
        """Read bounded event-time tables at one freshly pinned warehouse state."""
        cutoff = _utc(as_of)
        with self._session(database_id, time_columns=time_columns) as session:
            self._validate_visible_table_keys(session, cutoff)
            snapshot = self._get_snapshot(
                session.connection,
                session.schema,
                session.dialect,
                cutoff,
                session.pinned_at,
                session.table_references,
            )
        return snapshot

    def _get_prepared_inputs(
        self,
        database_id: str,
        query: str,
        *,
        anchor_times: Sequence[datetime],
        anchor_time: datetime | None,
        observed_through: datetime,
        entity_created_at: str,
        max_context_rows: int | None = DEFAULT_MAX_CONTEXT_ROWS,
        query_filter: str | None = None,
        load_source_tables_for_dfs: bool = True,
        candidate_sql: str | None = None,
    ) -> tuple[pd.DataFrame, dict[str, int], Snapshot | None]:
        """Build same-pin historical labels and optional feature-source tables.

        max_context_rows=None reads every candidate that can have a known
        label, for the caller to choose from. A warehouse query needs a row
        cap, so that read uses max_total_rows, the rows one prediction may
        load, which the snapshot below also counts the context rows against.
        candidate_sql is an optional caller SELECT that narrows that read.
        """
        row_cap = get_context_read_row_cap(self.settings, max_context_rows)
        if query_filter is not None and not load_source_tables_for_dfs:
            raise ValueError("query_filter requires prediction input preparation")
        tree = parse_pql(query)
        snapshot = None
        with self._session(database_id) as session:
            connection, schema, dialect = (
                session.connection,
                session.schema,
                session.dialect,
            )
            anchor_time = self._get_scoring_anchor_time(
                connection, schema, dialect, anchor_time
            )
            sql, params = compile_prepared_context_sql(
                tree,
                schema,
                anchor_times=anchor_times,
                anchor_time=anchor_time,
                observed_through=observed_through,
                entity_created_at=entity_created_at,
                max_context_rows=max_context_rows,
                dialect=dialect,
                candidate_sql=candidate_sql,
            )
            self._validate_visible_table_keys(session, anchor_time)
            rows = self._get_query_frame(
                connection,
                sql,
                params,
                max_rows=self.settings.max_total_rows if row_cap is None else row_cap,
            )
            # Unquoted result aliases are uppercased by Snowflake, not by PostgreSQL.
            rows.columns = [str(name).lower() for name in rows.columns]
            if list(rows.columns) != _CONTEXT_COLUMNS:
                raise ConnectorError("warehouse returned invalid context columns")
            rows["anchor_time"] = pd.to_datetime(rows["anchor_time"], utc=True)
            byte_count = int(rows.memory_usage(index=True, deep=True).sum())
            matched, matched_rows = None, 0
            if query_filter is not None:
                scoring_sql, scoring_params = compile_scoring_filter_sql(
                    tree,
                    schema,
                    query_filter,
                    anchor_time=anchor_time,
                    entity_created_at=entity_created_at,
                    dialect=dialect,
                )
                matches = self._get_query_frame(
                    connection,
                    scoring_sql,
                    scoring_params,
                    max_rows=self.settings.max_total_rows - len(rows),
                    max_bytes=self.settings.max_snapshot_bytes - byte_count,
                )
                matched = tuple(matches.iloc[:, 0])
                matched_rows = len(matches)
                byte_count += int(matches.memory_usage(index=True, deep=True).sum())
            if load_source_tables_for_dfs:
                snapshot = self._get_snapshot(
                    connection,
                    schema,
                    dialect,
                    anchor_time,
                    session.pinned_at,
                    session.table_references,
                    remaining_bytes=self.settings.max_snapshot_bytes - byte_count,
                    remaining_rows=self.settings.max_total_rows
                    - len(rows)
                    - matched_rows,
                )
                snapshot = replace(snapshot, scoring_filter_ids=matched)
        usable = rows.loc[rows["label"].notna()].reset_index(drop=True)
        return (
            usable,
            {
                "candidate_rows": len(rows),
                "usable_rows": len(usable),
                "dropped_unknown_labels": len(rows) - len(usable),
            },
            snapshot,
        )

    def prepare_context(
        self,
        database_id: str,
        query: str,
        *,
        anchor_times: Sequence[datetime],
        anchor_time: datetime | None,
        observed_through: datetime,
        entity_created_at: str,
        max_context_rows: int | None = DEFAULT_MAX_CONTEXT_ROWS,
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        """Return historical labels without downloading raw feature tables."""
        rows, counts, _ = self._get_prepared_inputs(
            database_id,
            query,
            anchor_times=anchor_times,
            anchor_time=anchor_time,
            observed_through=observed_through,
            entity_created_at=entity_created_at,
            max_context_rows=max_context_rows,
            load_source_tables_for_dfs=False,
        )
        return rows, counts

    def prepare_inputs(self, database_id: str, query: str, **options):
        """Return snapshot, labeled context, and counts from one pinned source state."""
        rows, counts, snapshot = self._get_prepared_inputs(
            database_id, query, **options
        )
        return snapshot, rows, counts
