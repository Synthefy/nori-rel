"""Bounded, read-only MySQL discovery and snapshot extraction.

The connector reads MySQL 8.0 or later, including Amazon RDS for MySQL and
Aurora MySQL 3. Every read runs in a read-only InnoDB transaction with a
consistent snapshot, in a UTC session. MariaDB and MySQL 5.7 are refused: the
generated SQL needs MySQL 8's common table expressions and window functions.
Metadata comes from information_schema and every statement is quoted here, so
table and column names may hold dots, spaces, percent signs and backticks.
"""

from __future__ import annotations

import math
import re
import ssl
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from threading import Lock
from typing import Any

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import URL, Engine
from sqlalchemy.exc import SQLAlchemyError

from nori_rel._credentials import CredentialResolver
from nori_rel.connectors.base import (
    Snapshot,
    get_context_read_row_cap,
    get_loaded_foreign_keys,
    is_local_host,
)
from nori_rel.connectors.errors import ConnectorError, SnapshotLimitError
from nori_rel.connectors.warehouse import build_warehouse_schema_graph
from nori_rel.contracts import (
    ConnectorCapabilities,
    DatabaseCreateRequest,
    DatabaseEngine,
    SchemaGraph,
    TableSchema,
)
from nori_rel.limits import RelationalLimits
from nori_rel.pql import parse_pql
from nori_rel.pql.binding import bind_columns
from nori_rel.pql.context_execution import compile_prepared_context_sql
from nori_rel.pql.context_sql import DEFAULT_MAX_CONTEXT_ROWS
from nori_rel.pql.dialects import SQLDialect
from nori_rel.pql.scoring_sql import compile_scoring_filter_sql
from nori_rel.pql.sql_expressions import column_kind
from nori_rel.pql.tree_access import node_ids, query_parts

#: ``ssl_mode`` values, named as in the MySQL client's ``--ssl-mode`` option.
SSL_MODES = ("DISABLED", "PREFERRED", "REQUIRED", "VERIFY_CA", "VERIFY_IDENTITY")
_TLS_SSL_MODES = frozenset({"REQUIRED", "VERIFY_CA", "VERIFY_IDENTITY"})
_URL_OPTIONS = frozenset({"ssl_mode", "ssl_ca"})
_INTEGER_TYPES = frozenset({"TINYINT", "SMALLINT", "MEDIUMINT", "INT", "BIGINT"})
_SIZED_TYPES = frozenset({"DECIMAL", "CHAR", "VARCHAR", "BINARY", "VARBINARY"})
#: Socket timeouts exceed the server's statement timeout by this much, so the
#: server reports a slow query before the driver gives up on the connection.
_DRIVER_TIMEOUT_MARGIN_SECONDS = 30
_DIALECT = SQLDialect(DatabaseEngine.MYSQL)
_TABLES_SQL = """SELECT TABLE_NAME AS table_name
FROM information_schema.TABLES
WHERE TABLE_SCHEMA = %(schema_name)s AND TABLE_TYPE = 'BASE TABLE'"""
_COLUMNS_SQL = """SELECT TABLE_NAME AS table_name, COLUMN_NAME AS column_name,
       ORDINAL_POSITION AS ordinal_position, IS_NULLABLE AS is_nullable,
       DATA_TYPE AS base_type, COLUMN_TYPE AS column_type
FROM information_schema.COLUMNS
WHERE TABLE_SCHEMA = %(schema_name)s"""
# MySQL names every primary key PRIMARY; a foreign key names its parent.
_KEYS_SQL = """SELECT TABLE_NAME AS table_name, CONSTRAINT_NAME AS constraint_name,
       COLUMN_NAME AS column_name, ORDINAL_POSITION AS key_sequence,
       REFERENCED_TABLE_SCHEMA AS referred_schema,
       REFERENCED_TABLE_NAME AS referred_table,
       REFERENCED_COLUMN_NAME AS referred_column
FROM information_schema.KEY_COLUMN_USAGE
WHERE TABLE_SCHEMA = %(schema_name)s
  AND (CONSTRAINT_NAME = 'PRIMARY' OR REFERENCED_TABLE_NAME IS NOT NULL)
ORDER BY TABLE_NAME, CONSTRAINT_NAME, ORDINAL_POSITION"""


def get_mysql_type_name(base_type: str, column_type: str) -> str:
    """Name a MySQL column type for ColumnSchema.data_type.

    Inputs are information_schema's DATA_TYPE and COLUMN_TYPE, e.g.
    ("bigint", "bigint unsigned") -> "BIGINT UNSIGNED", ("decimal",
    "decimal(12,2)") -> "DECIMAL(12, 2)", ("datetime", "datetime(6)") ->
    "DATETIME". TINYINT(1), which MySQL defines as BOOLEAN, is "BOOLEAN".
    Character sets, collations, enum values and display widths are dropped.
    """
    name = base_type.upper()
    column_type = column_type.lower()
    if name in _INTEGER_TYPES:
        if column_type.startswith("tinyint(1)"):
            return "BOOLEAN"
        return f"{name} UNSIGNED" if " unsigned" in column_type else name
    sizes = re.match(r"[a-z]+\(([\d,\s]+)\)", column_type)
    if name in _SIZED_TYPES and sizes is not None:
        return f"{name}({', '.join(sizes[1].replace(' ', '').split(','))})"
    return name


def get_mysql_connect_arguments(url: URL) -> tuple[URL, dict[str, Any]]:
    """Apply the TLS policy; return the URL without options and driver arguments.

    ``ssl_mode`` takes the MySQL client's names (SSL_MODES). A remote host
    needs REQUIRED, the default, VERIFY_CA or VERIFY_IDENTITY. A local host,
    such as the local end of an SSH tunnel, defaults to PREFERRED, which uses
    TLS when the server offers it, and may use DISABLED. ``ssl_ca`` is a PEM
    bundle for the VERIFY modes, e.g. the Amazon RDS global bundle; without it
    they trust the system certificates. Other URL options are refused rather
    than passed to the driver, so none can enable LOAD DATA LOCAL or run SQL
    at connect time.

    Raises:
        ConnectorError: An unknown or repeated option, an invalid mode, a
            remote host without TLS, or an unreadable ``ssl_ca`` file.
    """
    options = dict(url.query)
    unknown = sorted(set(options).difference(_URL_OPTIONS))
    if unknown:
        raise ConnectorError(
            f"unsupported MySQL URL options {unknown}; "
            "only ssl_mode and ssl_ca are accepted"
        )
    if any(not isinstance(value, str) for value in options.values()):
        raise ConnectorError("MySQL URL options must not repeat")
    local = is_local_host(url.host)
    mode = options.get("ssl_mode", "PREFERRED" if local else "REQUIRED").upper()
    if mode not in SSL_MODES:
        raise ConnectorError(f"ssl_mode must be one of {', '.join(SSL_MODES)}")
    if not local and mode not in _TLS_SSL_MODES:
        raise ConnectorError(
            "remote MySQL connections require TLS: "
            "ssl_mode=REQUIRED, VERIFY_CA or VERIFY_IDENTITY"
        )
    authority = options.get("ssl_ca")
    if authority is not None and mode not in {"VERIFY_CA", "VERIFY_IDENTITY"}:
        raise ConnectorError("ssl_ca needs ssl_mode=VERIFY_CA or VERIFY_IDENTITY")
    arguments: dict[str, Any] = {}
    if mode == "DISABLED":
        arguments["ssl_disabled"] = True
    elif mode in _TLS_SSL_MODES:
        try:
            context = ssl.create_default_context(cafile=authority)
        except (OSError, ssl.SSLError):
            raise ConnectorError("ssl_ca is not a readable PEM bundle") from None
        # Python 3.13's strict X.509 checks reject the certificates MySQL
        # generates for itself; PyMySQL relaxes the same flag.
        context.verify_flags &= ~ssl.VERIFY_X509_STRICT
        context.check_hostname = mode == "VERIFY_IDENTITY"
        if mode == "REQUIRED":
            context.verify_mode = ssl.CERT_NONE
        arguments["ssl"] = context
    return url.set(query={}), arguments


def _get_error_code(error: SQLAlchemyError) -> int | None:
    """Return the MySQL error number of a driver failure, never its message."""
    arguments = getattr(getattr(error, "orig", None), "args", ())
    code = arguments[0] if arguments else None
    return code if type(code) is int else None


def _get_failure_message(message: str, code: int | None) -> str:
    """Add a MySQL error number, e.g. 1045 or 3159, to a fixed message."""
    return message if code is None else f"{message} (code={code})"


def _get_naive_utc(value: Any) -> Any:
    """Return an aware datetime as naive UTC, and any other value unchanged.

    PyMySQL renders a datetime's wall-clock fields and drops its time zone,
    so 09:00+02:00 would bind as 09:00 in the UTC session instead of 07:00.
    """
    if isinstance(value, datetime) and value.utcoffset() is not None:
        return value.astimezone(UTC).replace(tzinfo=None)
    return value


def _get_driver_parameters(params: Mapping[str, Any]) -> dict[str, Any]:
    """Return compiler parameters with every aware datetime bound as naive UTC."""
    return {name: _get_naive_utc(value) for name, value in params.items()}


def _get_utc_time(value: Any) -> datetime:
    """Read a naive MySQL DATETIME or DATE, in the UTC session, as aware UTC."""
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC").to_pydatetime()


class MySQLConnector:
    """Bounded MySQL connector with read-only, consistent-snapshot transactions."""

    def __init__(
        self,
        config: DatabaseCreateRequest,
        settings: RelationalLimits,
        resolver: CredentialResolver | None = None,
    ) -> None:
        """Store source configuration without resolving its credential."""
        self.config = config
        self.settings = settings
        self.resolver = resolver or CredentialResolver()
        self._pool: Engine | None = None
        self._pool_lock = Lock()
        self._closed = False

    def _engine(self) -> Engine:
        """Lazily create one bounded pool; operations borrow fresh transactions."""
        with self._pool_lock:
            if self._closed:
                raise ConnectorError("MySQL connector is closed")
            if self._pool is None:
                url, tls_arguments = get_mysql_connect_arguments(
                    self.resolver.resolve_mysql(self.config.credential)
                )
                timeout = (
                    math.ceil(self.settings.statement_timeout_ms / 1000)
                    + _DRIVER_TIMEOUT_MARGIN_SECONDS
                )
                self._pool = create_engine(
                    url,
                    pool_pre_ping=True,
                    hide_parameters=True,
                    pool_size=2,
                    max_overflow=0,
                    isolation_level="REPEATABLE READ",
                    connect_args={
                        "connect_timeout": 10,
                        "read_timeout": timeout,
                        "write_timeout": timeout,
                        "program_name": "nori-rel",
                        "local_infile": False,
                        "init_command": self._get_session_settings(),
                        **tls_arguments,
                    },
                )
            return self._pool

    def _get_session_settings(self) -> str:
        """Return the SET statement every pooled session runs once on connect.

        UTC makes TIMESTAMP values and bound datetimes mean one instant;
        max_execution_time bounds each SELECT; division keeps 30 decimal
        places instead of MySQL's default 4; and every transaction is read
        only unless a statement asks otherwise, which none here does.
        """
        return (
            "SET SESSION time_zone = '+00:00', "
            "SESSION max_execution_time = "
            f"{int(self.settings.statement_timeout_ms)}, "
            "SESSION div_precision_increment = 30, "
            "SESSION transaction_read_only = ON"
        )

    def close(self) -> None:
        """Dispose this source's pool once, after its operations have finished.

        The connector cannot be reused after close. Re-register the source in a
        new client to resolve rotated credentials or use a different endpoint.
        """
        with self._pool_lock:
            self._closed = True
            if self._pool is not None:
                self._pool.dispose()
                self._pool = None

    @staticmethod
    def _require_supported_server(connection: Any) -> None:
        """Refuse MariaDB and MySQL before 8.0 before any query runs."""
        dialect = connection.dialect
        if dialect.is_mariadb:
            raise ConnectorError("MariaDB is not supported; use MySQL 8.0 or later")
        if dialect.server_version_info < (8, 0):
            raise ConnectorError("MySQL 8.0 or later is required")

    @contextmanager
    def _read_only_transaction(
        self, engine: Engine, *, consistent_snapshot: bool = False
    ) -> Iterator[Any]:
        """Yield a connection inside a READ ONLY transaction.

        With consistent_snapshot, InnoDB fixes the transaction's read view at
        its start, so labels and feature tables read one database state.
        """
        with engine.connect() as connection, connection.begin():
            self._require_supported_server(connection)
            # The session is already read only. State it again for every
            # transaction that touches customer data, as PostgreSQL does.
            connection.exec_driver_sql(
                "START TRANSACTION WITH CONSISTENT SNAPSHOT, READ ONLY"
                if consistent_snapshot
                else "START TRANSACTION READ ONLY",
                {},
            )
            yield connection

    def _get_query_frame(
        self,
        connection: Any,
        sql: str,
        params: Mapping[str, Any],
        *,
        max_rows: int | None,
        max_bytes: int,
    ) -> pd.DataFrame:
        """Stream a generated SELECT into an object DataFrame within limits.

        SQL uses pyformat placeholders and ``%%`` for a literal percent sign,
        which PyMySQL binds natively. Values keep their driver types, so
        BIGINT keys stay exact Python integers, including UNSIGNED ones.
        max_rows=None reads any number of rows within max_bytes.
        """
        frames, rows, size = [], 0, 0
        with connection.exec_driver_sql(
            sql,
            _get_driver_parameters(params),
            execution_options={"stream_results": True},
        ) as result:
            columns = list(result.keys())
            while batch := result.fetchmany(1000):
                rows += len(batch)
                if max_rows is not None and rows > max_rows:
                    raise SnapshotLimitError(
                        f"table exceeds the remaining row limit of {max_rows}"
                    )
                frame = pd.DataFrame(batch, columns=columns, dtype=object)
                size += int(frame.memory_usage(index=True, deep=True).sum())
                if size > max_bytes:
                    raise SnapshotLimitError(
                        "table exceeds the configured snapshot memory limit"
                    )
                frames.append(frame)
        if not frames:
            return pd.DataFrame(columns=columns, dtype=object)
        return pd.concat(frames, ignore_index=True)

    def test(self) -> float:
        """Verify connectivity and return latency in milliseconds."""
        started = time.monotonic()
        engine = self._engine()
        try:
            with self._read_only_transaction(engine) as connection:
                connection.exec_driver_sql("SELECT 1", {})
        except SQLAlchemyError as error:
            code = _get_error_code(error)
        else:
            return (time.monotonic() - started) * 1000
        # Leave the handler before raising so vendor SQL and parameters cannot
        # remain reachable through the public exception's __context__.
        raise ConnectorError(_get_failure_message("MySQL connection failed", code))

    def discover(self, database_id: str) -> SchemaGraph:
        """Inspect the configured MySQL database without reading rows."""
        engine = self._engine()
        try:
            with self._read_only_transaction(engine) as connection:
                return self._discover(connection, database_id)
        except SQLAlchemyError as error:
            code = _get_error_code(error)
        raise ConnectorError(
            _get_failure_message("MySQL schema discovery failed", code)
        )

    def _discover(
        self,
        connection: Any,
        database_id: str,
        *,
        time_columns: dict[str, str] | None = None,
    ) -> SchemaGraph:
        """Read tables, columns and declared keys from information_schema."""

        def read(sql: str) -> list[dict[str, Any]]:
            rows = connection.exec_driver_sql(sql, parameters).mappings().all()
            return [dict(row) for row in rows]

        parameters = {"schema_name": self.config.schema_name}
        available = {row["table_name"] for row in read(_TABLES_SQL)}
        selected = self.config.tables or sorted(available)
        missing = sorted(set(selected).difference(available))
        if missing:
            raise ConnectorError(f"configured tables do not exist: {missing}")
        if len(selected) > self.settings.max_tables:
            raise SnapshotLimitError(
                f"schema has {len(selected)} selected tables; "
                f"limit is {self.settings.max_tables}"
            )
        columns = [
            {
                **row,
                "data_type": get_mysql_type_name(row["base_type"], row["column_type"]),
            }
            for row in read(_COLUMNS_SQL)
        ]
        keys = read(_KEYS_SQL)
        return build_warehouse_schema_graph(
            self.config,
            database_id,
            selected,
            columns,
            [row for row in keys if row["constraint_name"] == "PRIMARY"],
            # A key to another database cannot be followed from this source.
            [
                row
                for row in keys
                if row["referred_table"] is not None
                and row["referred_schema"] == self.config.schema_name
            ],
            ConnectorCapabilities(),
            time_columns=time_columns,
        )

    def prepare_context(
        self,
        database_id: str,
        query: str,
        *,
        anchor_times: Sequence[datetime] | None,
        anchor_time: datetime | None,
        observed_through: datetime,
        entity_created_at: str,
        max_context_rows: int | None = DEFAULT_MAX_CONTEXT_ROWS,
        entity_time: str | None = None,
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        """Execute temporal PQL context and labels in one read-only snapshot.

        Arguments, returned rows and counts match
        PostgreSQLConnector.prepare_context. A comparison or Boolean label is
        returned as True/False, although MySQL computes it as 1/0, and
        anchor_time is an aware UTC datetime.
        """
        rows, counts, _ = self._prepare(
            database_id,
            query,
            anchor_times=anchor_times,
            anchor_time=anchor_time,
            observed_through=observed_through,
            entity_created_at=entity_created_at,
            max_context_rows=max_context_rows,
            load_source_tables_for_dfs=False,
            entity_time=entity_time,
        )
        return rows, counts

    def prepare_inputs(
        self, database_id: str, query: str, **options
    ) -> tuple[Snapshot, pd.DataFrame, dict[str, int]]:
        """Return (snapshot, labeled_context, counts) from one transaction.

        options are those of PostgreSQLConnector.prepare_inputs. The snapshot
        tables are read in the transaction that computed the labels, from the
        same InnoDB read view. Their DATETIME, TIMESTAMP and DATE columns are
        aware UTC, DECIMAL feature columns are floats and BOOLEAN columns are
        True/False.
        """
        rows, counts, snapshot = self._prepare(
            database_id, query, load_source_tables_for_dfs=True, **options
        )
        return snapshot, rows, counts

    def _read_scoring_filter_ids(self, connection, sql, params):
        """Return (tuple of matching native IDs, estimated bytes) within bounds."""
        ids, size = [], 0
        limit = min(self.settings.max_rows_per_table, self.settings.max_total_rows)
        with connection.exec_driver_sql(
            sql,
            _get_driver_parameters(params),
            execution_options={"stream_results": True},
        ) as result:
            while batch := result.fetchmany(1000):
                values = [row[0] for row in batch]
                size += int(pd.Series(values, dtype=object).memory_usage(deep=True))
                if (
                    len(ids) + len(values) > limit
                    or size > self.settings.max_snapshot_bytes
                ):
                    raise SnapshotLimitError("scoring filter exceeds resource limits")
                ids.extend(values)
        return tuple(ids), size

    @staticmethod
    def _resolve_anchor_time(
        connection: Any,
        schema: SchemaGraph,
        anchor_time: datetime | None,
        *,
        is_static_target: bool = False,
    ) -> datetime:
        """Keep an explicit cutoff, or find the latest registered event time.

        A static target with no cutoff is read at UTC_TIMESTAMP(6), taken
        just after its snapshot transaction began. MySQL returns naive values
        in the UTC session; they are returned as aware UTC datetimes.

        Raises:
            ValueError: No timestamp values exist and no cutoff was supplied.
        """
        if anchor_time is not None:
            return anchor_time
        if is_static_target:
            return _get_utc_time(
                connection.exec_driver_sql("SELECT UTC_TIMESTAMP(6)", {}).scalar_one()
            )
        latest = []
        for table in schema.tables:
            if table.time_column is None:
                continue
            timestamp = connection.exec_driver_sql(
                f"SELECT MAX({_DIALECT.identifier(table.time_column)}) FROM "
                f"{_DIALECT.table_reference(schema.schema_name, table.name)}",
                {},
            ).scalar_one()
            if timestamp is not None:
                latest.append(_get_utc_time(timestamp))
        if not latest:
            raise ValueError("no registered event timestamps; provide anchor_time")
        return max(latest)

    @staticmethod
    def _has_boolean_label(tree: Any, schema: SchemaGraph) -> bool:
        """Whether labels are True/False: a comparison, or a Boolean column."""
        parts = query_parts(tree)
        if parts["comparison"] is not None:
            return True
        if parts["aggregate"] is not None:
            return False
        column = bind_columns(tree, schema)[node_ids(tree)[parts["target_column"]]][1]
        return column_kind(column) == "bool"

    def _prepare(
        self,
        database_id,
        query,
        *,
        anchor_times,
        anchor_time,
        observed_through,
        entity_created_at,
        max_context_rows=DEFAULT_MAX_CONTEXT_ROWS,
        load_source_tables_for_dfs: bool = True,
        query_filter: str | None = None,
        candidate_sql: str | None = None,
        entity_time: str | None = None,
    ):
        """Build historical labels and optionally load tables for feature creation.

        Arguments and results match PostgreSQLConnector._prepare. Labels and
        tables come from one READ ONLY transaction WITH CONSISTENT SNAPSHOT,
        in a UTC session, with max_execution_time on every statement.
        """
        snapshot = None
        row_cap = get_context_read_row_cap(self.settings, max_context_rows)
        tree = parse_pql(query)
        engine = self._engine()
        try:
            with self._read_only_transaction(
                engine, consistent_snapshot=True
            ) as connection:
                schema = self._discover(connection, database_id)
                anchor_time = self._resolve_anchor_time(
                    connection,
                    schema,
                    anchor_time,
                    is_static_target=query_parts(tree)["aggregate"] is None,
                )
                sql, params = compile_prepared_context_sql(
                    tree,
                    schema,
                    dialect=_DIALECT,
                    anchor_times=anchor_times,
                    anchor_time=anchor_time,
                    observed_through=observed_through,
                    entity_created_at=entity_created_at,
                    max_context_rows=max_context_rows,
                    candidate_sql=candidate_sql,
                    entity_time=entity_time,
                    query_filter=query_filter,
                )
                scoring_query = None
                if query_filter is not None:
                    if not load_source_tables_for_dfs:
                        raise ValueError(
                            "query_filter requires prediction input preparation"
                        )
                    scoring_query = compile_scoring_filter_sql(
                        tree,
                        schema,
                        query_filter,
                        anchor_time=anchor_time,
                        entity_created_at=entity_created_at,
                        dialect=_DIALECT,
                    )
                try:
                    # Without a row cap, memory and the statement timeout bound
                    # the read, as in PostgreSQL.
                    rows = self._get_query_frame(
                        connection,
                        sql,
                        params,
                        max_rows=row_cap,
                        max_bytes=self.settings.max_snapshot_bytes,
                    )
                except SnapshotLimitError:
                    raise SnapshotLimitError(
                        "prepared context exceeds resource limits"
                    ) from None
                size = int(rows.memory_usage(index=True, deep=True).sum())
                rows["anchor_time"] = pd.to_datetime(rows["anchor_time"], utc=True)
                if self._has_boolean_label(tree, schema):
                    rows["label"] = pd.Series(
                        [
                            None if pd.isna(value) else bool(value)
                            for value in rows["label"]
                        ],
                        index=rows.index,
                        dtype=object,
                    )
                if load_source_tables_for_dfs:
                    matched = None
                    if scoring_query is not None:
                        scoring_sql, scoring_params = scoring_query
                        matched, match_bytes = self._read_scoring_filter_ids(
                            connection, scoring_sql, scoring_params
                        )
                        size += match_bytes
                    snapshot = self._snapshot(
                        connection,
                        schema,
                        anchor_time,
                        max_bytes=self.settings.max_snapshot_bytes - size,
                    )
                    if matched is not None:
                        snapshot = replace(snapshot, scoring_filter_ids=matched)
                    if (
                        snapshot.approximate_bytes + size
                        > self.settings.max_snapshot_bytes
                    ):
                        raise SnapshotLimitError(
                            "context and snapshot exceed memory limit"
                        )
        except SQLAlchemyError as error:
            code = _get_error_code(error)
        else:
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
        raise ConnectorError(
            _get_failure_message("MySQL context preparation failed", code)
        )

    def snapshot(
        self,
        database_id: str,
        as_of: datetime,
        *,
        time_columns: dict[str, str] | None = None,
    ) -> Snapshot:
        """Read bounded tables in a fresh consistent-snapshot transaction.

        time_columns optionally overrides timestamp mappings for this call,
        e.g. {"orders": "event_time"} for a legacy temporal target. It does not
        mutate the registered source or mappings used by subsequent PQL calls.
        """
        engine = self._engine()
        try:
            with self._read_only_transaction(
                engine, consistent_snapshot=True
            ) as connection:
                schema = self._discover(
                    connection, database_id, time_columns=time_columns
                )
                return self._snapshot(connection, schema, as_of)
        except SQLAlchemyError as error:
            code = _get_error_code(error)
        raise ConnectorError(_get_failure_message("MySQL snapshot failed", code))

    def _snapshot(
        self,
        connection,
        schema: SchemaGraph,
        as_of: datetime,
        *,
        max_bytes: int | None = None,
    ) -> Snapshot:
        """Read tables on the existing snapshot within the remaining byte budget."""
        if max_bytes is None:
            max_bytes = self.settings.max_snapshot_bytes
        if max_bytes < 0:
            raise SnapshotLimitError("context and snapshot exceed memory limit")
        frames: dict[str, pd.DataFrame] = {}
        total_rows = 0
        total_bytes = 0
        for table_schema in schema.tables:
            remaining_rows = min(
                self.settings.max_rows_per_table,
                self.settings.max_total_rows - total_rows,
            )
            sql = (
                "SELECT * FROM "
                f"{_DIALECT.table_reference(schema.schema_name, table_schema.name)}"
            )
            params: dict[str, Any] = {"row_limit": remaining_rows + 1}
            if table_schema.time_column is not None:
                time_column = _DIALECT.identifier(table_schema.time_column)
                sql += f" WHERE {time_column} <= %(cutoff)s"
                params["cutoff"] = as_of
            frame = self._get_query_frame(
                connection,
                sql + " LIMIT %(row_limit)s",
                params,
                max_rows=remaining_rows,
                max_bytes=max_bytes - total_bytes,
            )
            if len(frame) > self.settings.max_rows_per_table:
                raise SnapshotLimitError(
                    f"table {table_schema.name!r} exceeds the row limit "
                    f"of {self.settings.max_rows_per_table}"
                )
            total_rows += len(frame)
            if total_rows > self.settings.max_total_rows:
                raise SnapshotLimitError(
                    f"snapshot exceeds the total row limit of "
                    f"{self.settings.max_total_rows}"
                )
            frame = self._normalize_table_frame(frame, table_schema)
            total_bytes += int(frame.memory_usage(index=True, deep=True).sum())
            if total_bytes > max_bytes:
                raise SnapshotLimitError("snapshot exceeds the configured memory limit")
            frames[table_schema.name] = frame
        primary_keys = {table.name: table.primary_key for table in schema.tables}
        return Snapshot(
            tables=frames,
            primary_keys=primary_keys,
            # Keys to tables the source did not select cannot be followed.
            foreign_keys=get_loaded_foreign_keys(schema, frames),
            time_columns={
                table.name: table.time_column
                for table in schema.tables
                if table.time_column is not None
            },
            schema=schema,
            as_of=as_of,
            approximate_bytes=total_bytes,
        )

    @staticmethod
    def _normalize_table_frame(frame: pd.DataFrame, table: TableSchema) -> pd.DataFrame:
        """Give MySQL values the dtypes the feature backends expect.

        DATETIME, TIMESTAMP and DATE columns become aware UTC: the session
        runs in UTC, so naive values already hold UTC instants. A date pandas
        cannot hold, such as a 9999-12-31 sentinel, leaves its column as read,
        unless the column is the table's registered time. BOOLEAN (TINYINT(1))
        columns holding only 0 and 1 become True/False. Integer columns become
        Int64, except that an UNSIGNED BIGINT above 2**63 - 1 keeps exact
        Python integers. DECIMAL, FLOAT and DOUBLE feature columns become
        Float64, which feature backends read as numbers: they read object
        Decimals as categories. Non-integer key columns stay as read.

        Raises:
            ConnectorError: The registered time column holds such a date.
        """
        key_names = set(table.primary_key) | {
            column for key in table.foreign_keys for column in key.columns
        }
        for column in table.columns:
            try:
                kind = column_kind(column)
            except ValueError:
                continue
            values = frame[column.name]
            present = values.dropna()
            if kind in {"timestamp", "date"}:
                try:
                    frame[column.name] = pd.to_datetime(values, utc=True)
                except (OverflowError, ValueError):
                    if column.name == table.time_column:
                        raise ConnectorError(
                            f"time column {table.name}.{column.name} holds a "
                            "date outside the range pandas can hold"
                        ) from None
            elif kind == "bool" and all(value in (0, 1) for value in present):
                frame[column.name] = pd.array(
                    [pd.NA if pd.isna(value) else bool(value) for value in values],
                    dtype="boolean",
                )
            elif kind in {"bool", "number"}:
                name = column.data_type.partition("(")[0].removesuffix(" UNSIGNED")
                if name in _INTEGER_TYPES or name == "BOOLEAN":
                    if all(-(2**63) <= value < 2**63 for value in present):
                        frame[column.name] = pd.array(values, dtype="Int64")
                elif column.name not in key_names:
                    frame[column.name] = pd.array(
                        [
                            float(value) if pd.notna(value) else pd.NA
                            for value in values
                        ],
                        dtype="Float64",
                    )
        return frame
