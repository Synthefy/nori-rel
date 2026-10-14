"""Bounded, read-only PostgreSQL discovery and snapshot extraction."""

from __future__ import annotations

import ipaddress
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from threading import Lock
from typing import Any

import pandas as pd
from sqlalchemy import MetaData, Table, create_engine, inspect, select, text
from sqlalchemy.engine import URL, Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql.sqltypes import Date, DateTime

from nori_rel._credentials import CredentialResolver
from nori_rel.connectors.base import Snapshot
from nori_rel.connectors.errors import ConnectorError, SnapshotLimitError
from nori_rel.contracts import (
    ColumnSchema,
    ConnectorCapabilities,
    DatabaseCreateRequest,
    ForeignKeySchema,
    SchemaGraph,
    TableSchema,
)
from nori_rel.limits import RelationalLimits
from nori_rel.pql import parse_pql
from nori_rel.pql.context_execution import compile_prepared_context_sql
from nori_rel.pql.context_sql import DEFAULT_MAX_CONTEXT_ROWS
from nori_rel.pql.scoring_sql import compile_scoring_filter_sql
from nori_rel.pql.sql_expressions import identifier


def _is_local_host(host: str | None) -> bool:
    if not host or host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class PostgreSQLConnector:
    """Bounded PostgreSQL connector with enforced read-only transactions."""

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
                raise ConnectorError("PostgreSQL connector is closed")
            if self._pool is None:
                url = self._secure_url(self.resolver.resolve(self.config.credential))
                self._pool = create_engine(
                    url,
                    pool_pre_ping=True,
                    hide_parameters=True,
                    pool_size=2,
                    max_overflow=0,
                    connect_args={
                        "connect_timeout": 10,
                        "application_name": "nori-rel",
                        "options": (
                            "-c statement_timeout="
                            f"{self.settings.statement_timeout_ms} "
                            "-c default_transaction_read_only=on"
                        ),
                    },
                )
            return self._pool

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
    def _secure_url(url: URL) -> URL:
        query = {
            key: value
            for key, value in url.query.items()
            if key not in {"options", "application_name", "connect_timeout"}
        }
        if not _is_local_host(url.host):
            sslmode = query.get("sslmode", "require")
            if sslmode not in {"require", "verify-ca", "verify-full"}:
                raise ConnectorError("remote PostgreSQL connections require TLS")
            query["sslmode"] = sslmode
        return url.set(query=query)

    def _read_bounded(
        self, query: Any, connection: Any, *, max_rows: int, max_bytes: int
    ) -> pd.DataFrame:
        chunks: list[pd.DataFrame] = []
        rows = 0
        size = 0
        # Keep nullable BIGINT keys exact, including values above 2**53.
        for chunk in pd.read_sql(
            query.execution_options(stream_results=True, max_row_buffer=1000),
            connection,
            chunksize=1000,
            dtype_backend="numpy_nullable",
        ):
            rows += len(chunk)
            if rows > max_rows:
                raise SnapshotLimitError(
                    f"table exceeds the remaining row limit of {max_rows}"
                )
            size += int(chunk.memory_usage(index=True, deep=True).sum())
            if size > max_bytes:
                raise SnapshotLimitError(
                    "table exceeds the configured snapshot memory limit"
                )
            chunks.append(chunk)
        if not chunks:
            return pd.DataFrame(
                columns=[column.name for column in query.selected_columns]
            )
        return pd.concat(chunks, ignore_index=True)

    @staticmethod
    @contextmanager
    def _read_only_transaction(engine: Engine) -> Iterator[Any]:
        with engine.connect() as connection, connection.begin():
            # Some managed poolers ignore PostgreSQL startup options. Enforce the
            # invariant again inside every transaction that touches customer data.
            connection.execute(text("SET TRANSACTION READ ONLY"))
            yield connection

    def test(self) -> float:
        """Verify connectivity and return latency in milliseconds."""
        started = time.monotonic()
        engine = self._engine()
        try:
            with self._read_only_transaction(engine) as connection:
                connection.execute(text("SELECT 1"))
        except SQLAlchemyError:
            pass
        else:
            return (time.monotonic() - started) * 1000
        # Leave the handler before raising so vendor SQL and parameters cannot
        # remain reachable through the public exception's __context__.
        raise ConnectorError("PostgreSQL connection failed")

    def discover(self, database_id: str) -> SchemaGraph:
        """Inspect the configured PostgreSQL schema without reading rows."""
        engine = self._engine()
        try:
            with self._read_only_transaction(engine) as connection:
                return self._discover(connection, database_id)
        except SQLAlchemyError:
            pass
        raise ConnectorError("PostgreSQL schema discovery failed")

    def _discover(
        self,
        connection: Any,
        database_id: str,
        *,
        time_columns: dict[str, str] | None = None,
    ) -> SchemaGraph:
        inspector = inspect(connection)
        available = inspector.get_table_names(schema=self.config.schema_name)
        selected = self.config.tables or sorted(available)
        missing = sorted(set(selected).difference(available))
        if missing:
            raise ConnectorError(f"configured tables do not exist: {missing}")
        if len(selected) > self.settings.max_tables:
            raise SnapshotLimitError(
                f"schema has {len(selected)} selected tables; "
                f"limit is {self.settings.max_tables}"
            )

        timestamp_columns = {**self.config.time_columns, **(time_columns or {})}
        tables: list[TableSchema] = []
        for table_name in selected:
            columns_raw = inspector.get_columns(
                table_name, schema=self.config.schema_name
            )
            column_names = {column["name"] for column in columns_raw}
            time_column = timestamp_columns.get(table_name)
            if time_column is not None and time_column not in column_names:
                raise ConnectorError(
                    f"time column {table_name}.{time_column} does not exist"
                )
            if time_column is not None:
                raw = next(c for c in columns_raw if c["name"] == time_column)
                if not isinstance(raw["type"], (Date, DateTime)):
                    raise ConnectorError(
                        f"time column {table_name}.{time_column} is not temporal"
                    )
            pk = (
                inspector.get_pk_constraint(
                    table_name, schema=self.config.schema_name
                ).get("constrained_columns")
                or []
            )
            fks = []
            for fk in inspector.get_foreign_keys(
                table_name, schema=self.config.schema_name
            ):
                referred_schema = fk.get("referred_schema")
                if referred_schema not in (None, self.config.schema_name):
                    continue
                fks.append(
                    ForeignKeySchema(
                        name=fk.get("name"),
                        columns=fk.get("constrained_columns") or [],
                        referred_table=fk["referred_table"],
                        referred_columns=fk.get("referred_columns") or [],
                    )
                )
            tables.append(
                TableSchema(
                    name=table_name,
                    columns=[
                        ColumnSchema(
                            name=column["name"],
                            data_type=str(column["type"]),
                            nullable=bool(column.get("nullable", True)),
                        )
                        for column in columns_raw
                    ],
                    primary_key=list(pk),
                    foreign_keys=fks,
                    time_column=time_column,
                )
            )
        return SchemaGraph(
            database_id=database_id,
            schema_name=self.config.schema_name,
            tables=tables,
            capabilities=ConnectorCapabilities(),
            discovered_at=datetime.now(UTC),
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
        max_context_rows: int = DEFAULT_MAX_CONTEXT_ROWS,
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        """Execute temporal PQL context and labels in one read-only snapshot.

        Args:
            database_id: Registered source identifier, e.g. "sales".
            query: PQL text, e.g. "PREDICT COUNT(orders.*,0,30,days)=0
                FOR users.user_id=42 WHERE users.country='US'".
            anchor_times: Candidate historical aware datetimes.
            anchor_time: Aware scoring cutoff. None resolves to the latest timestamp
                in registered event tables, never wall-clock time. This does not
                establish completeness; observed_through remains explicit.
            observed_through: Verified complete-through time for the event table.
            entity_created_at: Registered entity creation column, e.g. "created_at".
                Caller guarantees its meaning and historical attribute validity.
            max_context_rows: Candidate output cap, default 5,000. Requests above
                connector context/row limits error; no silent reduction.

        Returns:
            (rows, counts). rows is a DataFrame with columns context_row_id,
            entity_id, anchor_time, label. Example: [(1, 7, historical_time, False)].
            Native label values are preserved; unknown labels are removed.
            counts is {"candidate_rows": 2, "usable_rows": 1,
                       "dropped_unknown_labels": 1} for one unknown label.
            IDs survive filtering; empty output retains the four columns.

        Uses schema discovery and one joined SELECT in a UTC repeatable-read
        transaction, with statement timeout, streamed fetches and a memory bound.
        No table writes, raw-table downloads or inference. No refill/minimum-row
        enforcement yet; source completeness and retention are caller guarantees.
        Future feature construction must use a compatible source snapshot.
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
        )
        return rows, counts

    def prepare_inputs(
        self, database_id: str, query: str, **options
    ) -> tuple[Snapshot, pd.DataFrame, dict[str, int]]:
        """Return (snapshot, labeled_context, counts) from one transaction.

        options are prepare_context's anchor_times, anchor_time, observed_through,
        entity_created_at, optional max_context_rows and optional query_filter.
        query_filter is a scalar entity predicate; matching IDs are stored in
        snapshot.scoring_filter_ids for intersection with FOR before features.
        Context columns are
        context_row_id, entity_id, anchor_time, label. Snapshot contains bounded
        registered tables for local FastDFS, read at the same MVCC version as
        labels. All event tables need registered time columns; unmapped tables
        and entity attributes must be immutable. Events must be available at
        their event timestamp. Exceeding table/total/memory limits errors.
        """
        rows, counts, snapshot = self._prepare(
            database_id, query, load_source_tables_for_dfs=True, **options
        )
        return snapshot, rows, counts

    def _read_scoring_filter_ids(self, connection, sql, params):
        """Return (tuple of matching native IDs, estimated bytes) within bounds.

        Inputs: active read-only connection and compile_scoring_filter_sql output.
        IDs are read in batches; neither a new transaction nor a source mutation
        occurs. Empty matches return ((), 0). Limits apply before feature loading.
        """
        ids, size = [], 0
        limit = min(self.settings.max_rows_per_table, self.settings.max_total_rows)
        with connection.exec_driver_sql(
            sql, params, execution_options={"stream_results": True}
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
        connection: Any, schema: SchemaGraph, anchor_time: datetime | None
    ) -> datetime:
        """Keep an explicit cutoff, or find the latest registered event time.

        Args:
            connection: Open SQLAlchemy connection in the caller's read-only
                transaction. This helper never opens a separate connection.
            schema: Registered tables and timestamp mappings, e.g. orders with
                time_column="event_time" in schema "public".
            anchor_time: Explicit datetime returned unchanged, or None to query
                the maximum timestamp across registered event tables.

        Returns:
            A datetime, e.g. datetime(2026, 2, 1, tzinfo=UTC). Inferred times are
            normalized to UTC. Explicit cutoffs are validated by label SQL
            compilation. A latest event time does not establish completeness.

        Raises:
            ValueError: No timestamp values exist and no cutoff was supplied.
        """
        if anchor_time is not None:
            return anchor_time
        latest = []
        for table in schema.tables:
            if table.time_column is None:
                continue
            timestamp = connection.exec_driver_sql(
                f"SELECT MAX({identifier(table.time_column)}) FROM "
                f"{identifier(schema.schema_name)}.{identifier(table.name)}",
                {},
            ).scalar_one()
            if timestamp is not None:
                latest.append(
                    pd.Timestamp(timestamp).tz_localize("UTC")
                    if pd.Timestamp(timestamp).tzinfo is None
                    else pd.Timestamp(timestamp).tz_convert("UTC")
                )
        if not latest:
            raise ValueError("no registered event timestamps; provide anchor_time")
        return max(latest).to_pydatetime()

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
    ):
        """Build historical labels and optionally load tables for feature creation.

        Args:
            database_id: Registered source ID, e.g. "sales"; identifies the
                returned schema and does not select another PostgreSQL database.
            query: Temporal PQL text, e.g. "PREDICT COUNT(orders.*,0,30,days)
                FOR users.user_id=42 WHERE users.country='US'". WHERE selects
                historical context; FOR does not restrict historical entities.
            anchor_times: Candidate historical aware datetimes, e.g.
                [datetime(2026, 1, 1, tzinfo=UTC)]. Each eligible entity/time
                pair becomes a candidate row if its label window is complete.
            anchor_time: Aware scoring cutoff, or None to use the latest
                registered event timestamp. Features cannot use later events.
            observed_through: Caller-verified completeness watermark. Historical
                label windows must end by min(anchor_time, observed_through).
            entity_created_at: Entity creation column, e.g. "created_at".
                Exclude entities not yet created at their historical anchor.
            max_context_rows: Maximum candidate rows before unknown labels are
                removed, default 5,000. Must fit connector resource limits.
            query_filter: Optional scalar entity predicate for scoring only.
                It is validated before data loading, and matching IDs are stored
                on the returned snapshot. Does not change historical context.
            load_source_tables_for_dfs: Defaults to True. Load bounded original
                database tables into Python so FastDFS can calculate features,
                using the same transaction as labels. False returns only labels
                and counts, without loading source tables for feature creation.

        Returns:
            (context, counts, snapshot). context is a DataFrame with columns
            context_row_id, entity_id, anchor_time, label; for example,
            [(1, 7, datetime(2026, 1, 1, tzinfo=UTC), 2)]. Unknown labels are
            dropped without renumbering IDs. counts has integer values, e.g.
            {"candidate_rows": 2, "usable_rows": 1, "dropped_unknown_labels": 1}.
            snapshot is None when load_source_tables_for_dfs=False. Otherwise it
            contains tables={"users": DataFrame, "orders": DataFrame}, keys,
            relationships, timestamp mappings, schema, as_of and byte count.
            These tables are inputs for feature creation, not feature matrices.
        """
        snapshot = None
        ceiling = min(
            self.settings.max_context_rows,
            self.settings.max_rows_per_table,
            self.settings.max_total_rows,
        )
        if type(max_context_rows) is not int or not 0 < max_context_rows <= ceiling:
            raise SnapshotLimitError(
                f"max_context_rows must be an integer in 1..{ceiling}"
            )
        tree = parse_pql(query)
        engine = self._engine()
        try:
            with (
                engine.connect().execution_options(
                    isolation_level="REPEATABLE READ"
                ) as connection,
                connection.begin(),
            ):
                connection.execute(text("SET TRANSACTION READ ONLY"))
                connection.execute(text("SET LOCAL TIME ZONE 'UTC'"))
                connection.execute(
                    text("SELECT set_config('statement_timeout', :timeout, true)"),
                    {"timeout": str(self.settings.statement_timeout_ms)},
                )
                schema = self._discover(connection, database_id)
                anchor_time = self._resolve_anchor_time(connection, schema, anchor_time)
                sql, params = compile_prepared_context_sql(
                    tree,
                    schema,
                    anchor_times=anchor_times,
                    anchor_time=anchor_time,
                    observed_through=observed_through,
                    entity_created_at=entity_created_at,
                    max_context_rows=max_context_rows,
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
                    )
                frames, count, size = [], 0, 0
                with connection.exec_driver_sql(
                    sql, params, execution_options={"stream_results": True}
                ) as result:
                    columns = list(result.keys())
                    while batch := result.fetchmany(1000):
                        frame = pd.DataFrame(batch, columns=columns, dtype=object)
                        count += len(frame)
                        size += int(frame.memory_usage(index=True, deep=True).sum())
                        if (
                            count > max_context_rows
                            or size > self.settings.max_snapshot_bytes
                        ):
                            raise SnapshotLimitError(
                                "prepared context exceeds resource limits"
                            )
                        frames.append(frame)
                rows = (
                    pd.concat(frames, ignore_index=True)
                    if frames
                    else pd.DataFrame(columns=columns, dtype=object)
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
        except SQLAlchemyError:
            pass
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
        raise ConnectorError("PostgreSQL context preparation failed")

    def snapshot(
        self,
        database_id: str,
        as_of: datetime,
        *,
        time_columns: dict[str, str] | None = None,
    ) -> Snapshot:
        """Read bounded tables in a fresh repeatable-read transaction.

        time_columns optionally overrides timestamp mappings for this call,
        e.g. {"orders": "event_time"} for a legacy temporal target. It does not
        mutate the registered source or mappings used by subsequent PQL calls.
        """
        engine = self._engine()
        try:
            with (
                engine.connect().execution_options(
                    isolation_level="REPEATABLE READ"
                ) as connection,
                connection.begin(),
            ):
                connection.execute(text("SET TRANSACTION READ ONLY"))
                schema = self._discover(
                    connection, database_id, time_columns=time_columns
                )
                return self._snapshot(connection, schema, as_of)
        except ConnectorError:
            raise
        except SQLAlchemyError:
            pass
        raise ConnectorError("PostgreSQL snapshot failed")

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
        metadata = MetaData(schema=self.config.schema_name)
        frames: dict[str, pd.DataFrame] = {}
        total_rows = 0
        total_bytes = 0
        for table_schema in schema.tables:
            table = Table(
                table_schema.name,
                metadata,
                autoload_with=connection,
            )
            query = select(table)
            if table_schema.time_column is not None:
                query = query.where(table.c[table_schema.time_column] <= as_of)
            remaining_rows = min(
                self.settings.max_rows_per_table,
                self.settings.max_total_rows - total_rows,
            )
            query = query.limit(remaining_rows + 1)
            frame = self._read_bounded(
                query,
                connection,
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
            total_bytes += int(frame.memory_usage(index=True, deep=True).sum())
            if total_bytes > max_bytes:
                raise SnapshotLimitError("snapshot exceeds the configured memory limit")
            frames[table_schema.name] = frame
        foreign_keys: list[tuple[str, str, str, str]] = []
        primary_keys: dict[str, list[str]] = {}
        for table in schema.tables:
            primary_keys[table.name] = table.primary_key
            for foreign_key in table.foreign_keys:
                if len(foreign_key.columns) == len(foreign_key.referred_columns) == 1:
                    foreign_keys.append(
                        (
                            table.name,
                            foreign_key.columns[0],
                            foreign_key.referred_table,
                            foreign_key.referred_columns[0],
                        )
                    )
        return Snapshot(
            tables=frames,
            primary_keys=primary_keys,
            foreign_keys=foreign_keys,
            time_columns={
                table.name: table.time_column
                for table in schema.tables
                if table.time_column is not None
            },
            schema=schema,
            as_of=as_of,
            approximate_bytes=total_bytes,
        )
