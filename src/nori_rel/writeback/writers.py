"""Write one run's rows into tables of a destination schema, per engine.

Each writer opens its own connection with the destination's writer
credential; no source connector, which only reads, is ever used to write.
Writing a run's rows first removes the rows that run wrote before, so a
retried job leaves one copy:

- PostgreSQL deletes and inserts in one transaction.
- Snowflake deletes and inserts in one explicit transaction. Each insert
  sends its rows as one JSON parameter and expands them with ``FLATTEN``.
- Databricks has no multi-statement transactions, so each table is replaced
  by one ``INSERT INTO … REPLACE WHERE run_id = …`` statement. A write
  bigger than one parameter (the driver caps parameters at 1 MB in all) is
  first loaded into a staging table, which is dropped afterwards.

Values travel as parameters, never as SQL text. Warehouse driver errors are
reported by their codes only, since their messages can echo statements; a
PostgreSQL message is kept, redacted.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, ClassVar, Protocol

from nori_rel.connectors.base import RUN_RECORDS_TABLE, STAGING_TABLE_PREFIX
from nori_rel.connectors.errors import ConnectorError
from nori_rel.contracts import (
    DatabaseCreateRequest,
    DatabaseEngine,
    DatabricksConnection,
    PredictionDestination,
    SnowflakeConnection,
)
from nori_rel.limits import RelationalLimits
from nori_rel.run.redact import redact
from nori_rel.writeback.rows import (
    RUN_RECORD_COLUMNS,
    Column,
    ColumnKind,
    TableRows,
)

logger = logging.getLogger(__name__)

#: Bytes of JSON one Snowflake or Databricks insert carries. Databricks caps
#: all parameter values of one statement at 1 MB.
MAX_PAYLOAD_BYTES = 900_000
#: Run ids are generated (``run_<hex>``); this is checked before one enters
#: a Databricks ``REPLACE WHERE`` clause as a literal.
_RUN_ID = re.compile(r"[A-Za-z0-9_-]+")
#: How every written time is spelled for Snowflake to parse; see ``_get_text``.
_SNOWFLAKE_TIME_FORMAT = "YYYY-MM-DD HH24:MI:SS.FF6TZH:TZM"
#: Databricks starts an error message with its error class and ends it with
#: its SQL state; the text between can quote values, so only these are kept.
_ERROR_CLASS = re.compile(r"^\[([A-Z][A-Z0-9_.]*)\]")
_SQL_STATE = re.compile(r"SQLSTATE: ([0-9A-Z]{5})")


class PredictionWriteError(RuntimeError):
    """A write to the destination failed; the message holds no secret."""


class TableWriter(Protocol):
    """One open writer connection to a destination schema."""

    def get_qualified_name(self, table: str) -> str:
        """Return ``table``'s full name in the destination, for the run record."""
        ...

    def get_column_names(self, table: str) -> list[str] | None:
        """Return ``table``'s physical column names in order; None if missing."""
        ...

    def get_column_name(self, column: str) -> str:
        """Return the physical name a written column has in this destination."""
        ...

    def create_table(self, table: str, columns: Sequence[Column]) -> None:
        """Create ``table`` with ``columns`` unless it exists."""
        ...

    def replace_run_rows(self, writes: Sequence[TableRows]) -> None:
        """Replace each write's run rows in its table with the write's rows."""
        ...

    def read_run_record(self, run_id: str) -> Any:
        """Return the ``record`` of ``run_id`` in the run records table, or None."""
        ...

    def close(self) -> None:
        """Close the connection; safe to repeat."""
        ...


def open_table_writer(
    destination: PredictionDestination, limits: RelationalLimits | None = None
) -> TableWriter:
    """Open a writer for ``destination`` with its own credential."""
    limits = limits or RelationalLimits()
    if destination.engine is DatabaseEngine.POSTGRESQL:
        return PostgreSQLTableWriter(destination, limits)
    if destination.engine is DatabaseEngine.SNOWFLAKE:
        return SnowflakeTableWriter(destination, limits)
    return DatabricksTableWriter(destination, limits)


@contextmanager
def _reporting_errors(
    engine_name: str, *, keep_message: bool = False
) -> Iterator[None]:
    """Turn a driver error into ``PredictionWriteError`` without its secrets.

    Our own refusals (``ValueError``) and ``PredictionWriteError`` pass
    through. A connector's errors are credential-free, so their text is kept.
    """
    try:
        yield
    except (PredictionWriteError, ValueError):
        raise
    except Exception as error:
        from nori_rel._credentials import CredentialResolutionError

        if isinstance(error, ConnectorError | CredentialResolutionError):
            # Both are written without secrets, so their text is kept.
            raise PredictionWriteError(f"{engine_name} write failed: {error}") from None
        if keep_message:
            detail = redact(str(error).split("\n[SQL:")[0]) or "no detail"
            raise PredictionWriteError(
                f"{engine_name} write failed: {detail}"
            ) from None
        raise PredictionWriteError(
            f"{engine_name} write failed{_get_error_codes(error)}"
        ) from None


def _get_error_codes(error: Exception) -> str:
    """Return a driver error's codes, such as `` (code=2003, sqlstate=02000)``.

    Snowflake errors carry ``errno`` and ``sqlstate``; a Databricks message
    starts with its error class and names its SQL state. Nothing else of the
    message is kept, since it can quote statements and values.
    """
    message = str(error)
    error_class = _ERROR_CLASS.search(message)
    sql_state = _SQL_STATE.search(message)
    codes = [
        f"{name}={value}"
        for name, value in (
            ("error", error_class and error_class.group(1)),
            ("code", getattr(error, "errno", None)),
            ("sqlstate", getattr(error, "sqlstate", None)),
            ("sqlstate", sql_state and sql_state.group(1)),
        )
        if isinstance(value, int | str) and re.fullmatch(r"[\w.]{1,64}", str(value))
    ]
    codes = list(dict.fromkeys(codes))
    return f" ({', '.join(codes)})" if codes else ""


class PostgreSQLTableWriter:
    """Write with SQLAlchemy Core, in one transaction per call."""

    def __init__(
        self, destination: PredictionDestination, limits: RelationalLimits
    ) -> None:
        """Resolve the writer credential and create a one-connection engine.

        A remote host needs TLS, as for a PostgreSQL source.
        """
        from sqlalchemy import create_engine

        from nori_rel._credentials import CredentialResolver
        from nori_rel.connectors.postgres import PostgreSQLConnector

        self.schema_name = destination.schema_name
        with _reporting_errors("PostgreSQL", keep_message=True):
            url = PostgreSQLConnector._secure_url(
                CredentialResolver().resolve(destination.credential)
            )
            # UTC, so a table the caller made with timestamp-without-time-zone
            # columns stores UTC wall times, not the server's local ones.
            self._engine = create_engine(
                url,
                hide_parameters=True,
                pool_size=1,
                max_overflow=0,
                connect_args={
                    "connect_timeout": 10,
                    "application_name": "nori-rel-writer",
                    "options": (
                        f"-c statement_timeout={limits.statement_timeout_ms} "
                        "-c timezone=UTC"
                    ),
                },
            )

    def get_qualified_name(self, table: str) -> str:
        """Return ``schema.table``."""
        return f"{self.schema_name}.{table}"

    def get_column_names(self, table: str) -> list[str] | None:
        """Return the table's column names, or None if the schema lacks it."""
        from sqlalchemy import inspect

        with _reporting_errors("PostgreSQL", keep_message=True):
            inspector = inspect(self._engine)
            if not inspector.has_table(table, schema=self.schema_name):
                return None
            columns = inspector.get_columns(table, schema=self.schema_name)
        return [str(column["name"]) for column in columns]

    def get_column_name(self, column: str) -> str:
        """Return ``column``; PostgreSQL keeps the lowercase names as written."""
        return column

    def create_table(self, table: str, columns: Sequence[Column]) -> None:
        """Create the table and an index on ``run_id``, unless they exist.

        Each write deletes by ``run_id``, so the index keeps that from
        scanning every earlier run's rows.
        """
        from sqlalchemy import Index
        from sqlalchemy.schema import CreateIndex, CreateTable

        created = self._get_table(table, columns)
        index_name = (
            f"{table[:40]}_run_id_{hashlib.sha256(table.encode()).hexdigest()[:8]}"
        )
        with (
            _reporting_errors("PostgreSQL", keep_message=True),
            self._engine.begin() as connection,
        ):
            connection.execute(CreateTable(created, if_not_exists=True))
            connection.execute(
                CreateIndex(Index(index_name, created.c.run_id), if_not_exists=True)
            )

    def replace_run_rows(self, writes: Sequence[TableRows]) -> None:
        """Delete and insert every write's rows in one transaction.

        A transaction lock per run, and per pipeline and ``as_of`` when
        earlier runs are replaced, makes two such writes take turns;
        otherwise neither delete sees the other's uncommitted rows and both
        copies stay. Locks are taken in a fixed order.
        """
        from sqlalchemy import and_, delete, insert, or_, text

        with (
            _reporting_errors("PostgreSQL", keep_message=True),
            self._engine.begin() as connection,
        ):
            for key in sorted(_get_lock_keys(self.schema_name, writes)):
                connection.execute(
                    text("SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))"),
                    {"key": key},
                )
            for write in writes:
                table = self._get_table(write.table, write.columns)
                condition = table.c.run_id == write.run_id
                earlier = write.earlier_runs
                if earlier is not None and earlier.as_of_values:
                    condition = or_(
                        condition,
                        and_(
                            table.c.pipeline_id == earlier.pipeline_id,
                            table.c.as_of.in_(earlier.as_of_values),
                        ),
                    )
                connection.execute(delete(table).where(condition))
                if write.rows:
                    names = [column.name for column in write.columns]
                    connection.execute(
                        insert(table),
                        [dict(zip(names, row, strict=True)) for row in write.rows],
                    )

    def read_run_record(self, run_id: str) -> Any:
        """Return the stored record of ``run_id``, or None."""
        from sqlalchemy import select

        runs = self._get_table(RUN_RECORDS_TABLE, RUN_RECORD_COLUMNS)
        with (
            _reporting_errors("PostgreSQL", keep_message=True),
            self._engine.connect() as connection,
        ):
            return connection.execute(
                select(runs.c.record).where(runs.c.run_id == run_id)
            ).scalar_one_or_none()

    def close(self) -> None:
        """Dispose of the engine's connection."""
        self._engine.dispose()

    def _get_table(self, name: str, columns: Sequence[Column]) -> Any:
        """Return a SQLAlchemy table for ``columns`` in the destination schema."""
        import sqlalchemy
        from sqlalchemy.dialects.postgresql import JSONB

        types = {
            "integer": sqlalchemy.BigInteger(),
            "decimal": sqlalchemy.Numeric(38, 0),
            "text": sqlalchemy.Text(),
            "timestamp": sqlalchemy.DateTime(timezone=True),
            "double": sqlalchemy.Double(),
            "boolean": sqlalchemy.Boolean(),
            "json": JSONB(),
        }
        return sqlalchemy.Table(
            name,
            sqlalchemy.MetaData(schema=self.schema_name),
            *(sqlalchemy.Column(column.name, types[column.kind]) for column in columns),
        )


class _WarehouseTableWriter:
    """Run qmark SQL on a writer connection opened like a source's.

    The engine's connector class opens and configures the connection (driver
    options, authentication, UTC, statement timeout); its read-only query
    path is never used.
    """

    engine_name = ""
    types: ClassVar[dict[ColumnKind, str]] = {}

    def __init__(
        self, destination: PredictionDestination, limits: RelationalLimits
    ) -> None:
        """Open the connection with the destination's writer credential."""
        from nori_rel.connectors import create_connector

        self.destination = destination
        config = DatabaseCreateRequest(
            engine=destination.engine,
            credential=destination.credential,
            schema_name=destination.schema_name,
            connection=destination.connection,
        )
        self._connector = create_connector(config, limits)
        with _reporting_errors(self.engine_name):
            self._connection = self._connector._get_connection()

    @property
    def namespace(self) -> str:
        """The Snowflake database or Databricks catalog of the schema."""
        raise NotImplementedError

    def get_qualified_name(self, table: str) -> str:
        """Return ``namespace.schema.table`` as the destination names them."""
        return ".".join(
            (self.namespace, self.destination.schema_name, self.get_table_name(table))
        )

    def get_table_name(self, table: str) -> str:
        """Return the physical name of ``table``."""
        return table

    def get_column_name(self, column: str) -> str:
        """Return the physical name of written ``column``."""
        return column

    def create_table(self, table: str, columns: Sequence[Column]) -> None:
        """Create the table with ``CREATE TABLE IF NOT EXISTS``."""
        definitions = ", ".join(
            f"{self.quote(self.get_column_name(column.name))} {self.types[column.kind]}"
            for column in columns
        )
        self.execute(
            f"CREATE TABLE IF NOT EXISTS {self.get_reference(table)} ({definitions})"
        )

    def read_run_record(self, run_id: str) -> Any:
        """Return the stored record of ``run_id`` as JSON text, or None."""
        record = self.quote(self.get_column_name("record"))
        run_id_column = self.quote(self.get_column_name("run_id"))
        rows = self.fetch_all(
            f"SELECT CAST({record} AS {self.types['text']}) "
            f"FROM {self.get_reference(RUN_RECORDS_TABLE)} WHERE {run_id_column} = ?",
            [run_id],
        )
        return rows[0][0] if rows else None

    def close(self) -> None:
        """Close the writer connection; safe to repeat."""
        self._connector.close()

    def quote(self, name: str) -> str:
        """Return ``name`` as a quoted identifier."""
        raise NotImplementedError

    def get_reference(self, table: str) -> str:
        """Return the quoted ``namespace.schema.table`` reference."""
        return ".".join(
            self.quote(part)
            for part in (
                self.namespace,
                self.destination.schema_name,
                self.get_table_name(table),
            )
        )

    def get_column_list(self, columns: Sequence[Column]) -> str:
        """Return the quoted, comma-separated physical names of ``columns``."""
        return ", ".join(
            self.quote(self.get_column_name(column.name)) for column in columns
        )

    def execute(self, sql: str, parameters: Sequence[Any] = ()) -> None:
        """Run one statement on a fresh cursor."""
        with _reporting_errors(self.engine_name):
            cursor = self._connection.cursor()
            try:
                if parameters:
                    cursor.execute(sql, list(parameters))
                else:
                    cursor.execute(sql)
            finally:
                with suppress(Exception):
                    cursor.close()

    def fetch_all(self, sql: str, parameters: Sequence[Any]) -> list[tuple]:
        """Run one query on a fresh cursor and return its rows."""
        with _reporting_errors(self.engine_name):
            cursor = self._connection.cursor()
            try:
                cursor.execute(sql, list(parameters))
                return [tuple(row) for row in cursor.fetchall()]
            finally:
                with suppress(Exception):
                    cursor.close()


class SnowflakeTableWriter(_WarehouseTableWriter):
    """Write Snowflake tables in one explicit transaction per call.

    Columns are uppercase, so unquoted queries read them, and so is the run
    records table, ``NORI_REL_RUNS``. Other table and schema names are used
    as given, as a Snowflake source uses them.
    """

    engine_name = "Snowflake"
    types: ClassVar[dict[ColumnKind, str]] = {
        "integer": "NUMBER(38,0)",
        "decimal": "NUMBER(38,0)",
        "text": "VARCHAR",
        "timestamp": "TIMESTAMP_TZ",
        "double": "FLOAT",
        "boolean": "BOOLEAN",
        "json": "VARIANT",
    }
    _CASTS: ClassVar[dict[ColumnKind, str]] = {
        "integer": "{value}::NUMBER(38,0)",
        "decimal": "{value}::NUMBER(38,0)",
        "text": "{value}",
        "timestamp": "TO_TIMESTAMP_TZ({value}, '" + _SNOWFLAKE_TIME_FORMAT + "')",
        "double": "{value}::FLOAT",
        "boolean": "{value}::BOOLEAN",
        "json": "PARSE_JSON({value})",
    }

    @property
    def namespace(self) -> str:
        """The Snowflake database of the schema."""
        connection = self.destination.connection
        assert isinstance(connection, SnowflakeConnection)
        return connection.database

    def get_table_name(self, table: str) -> str:
        """Return ``NORI_REL_RUNS`` for the run records table, else ``table``."""
        return table.upper() if table == RUN_RECORDS_TABLE else table

    def get_column_name(self, column: str) -> str:
        """Return the uppercase column name."""
        return column.upper()

    def get_column_names(self, table: str) -> list[str] | None:
        """Return the table's exact column names in order; None if missing.

        Names are compared exactly, so a table with quoted lowercase columns,
        which an uppercase insert cannot reach, is refused.
        """
        rows = self.fetch_all(
            f"SELECT COLUMN_NAME FROM {self.quote(self.namespace)}"
            ".INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = ? AND "
            "TABLE_NAME = ? ORDER BY ORDINAL_POSITION",
            [self.destination.schema_name, self.get_table_name(table)],
        )
        return [str(row[0]) for row in rows] or None

    def replace_run_rows(self, writes: Sequence[TableRows]) -> None:
        """Delete and insert every write's rows in one transaction."""
        self.execute("BEGIN")
        try:
            for write in writes:
                reference = self.get_reference(write.table)
                condition = f"{self.quote(self.get_column_name('run_id'))} = ?"
                parameters: list[Any] = [write.run_id]
                earlier = write.earlier_runs
                if earlier is not None and earlier.as_of_values:
                    times = ", ".join(
                        f"TO_TIMESTAMP_TZ(?, '{_SNOWFLAKE_TIME_FORMAT}')"
                        for _ in earlier.as_of_values
                    )
                    condition += (
                        f" OR ({self.quote(self.get_column_name('pipeline_id'))} = ?"
                        f" AND {self.quote(self.get_column_name('as_of'))}"
                        f" IN ({times}))"
                    )
                    parameters += [
                        earlier.pipeline_id,
                        *(
                            _get_text(value, "timestamp")
                            for value in earlier.as_of_values
                        ),
                    ]
                self.execute(f"DELETE FROM {reference} WHERE {condition}", parameters)
                select_list = ", ".join(
                    self._CASTS[column.kind].format(value=f"value[{index}]::VARCHAR")
                    for index, column in enumerate(write.columns)
                )
                for payload in get_json_payloads(write):
                    self.execute(
                        f"INSERT INTO {reference} "
                        f"({self.get_column_list(write.columns)}) "
                        f"SELECT {select_list} "
                        "FROM TABLE(FLATTEN(INPUT => PARSE_JSON(?)))",
                        [payload],
                    )
            self.execute("COMMIT")
        except BaseException:
            with suppress(Exception):
                self.execute("ROLLBACK")
            raise

    def quote(self, name: str) -> str:
        """Return a double-quoted, case-sensitive identifier."""
        return '"' + name.replace('"', '""') + '"'


class DatabricksTableWriter(_WarehouseTableWriter):
    """Replace Delta rows with one ``INSERT … REPLACE WHERE`` per table.

    ``REPLACE WHERE`` takes no column list, so the rows come in the table's
    column order, which the caller has matched.
    """

    engine_name = "Databricks"
    types: ClassVar[dict[ColumnKind, str]] = {
        "integer": "BIGINT",
        "decimal": "DECIMAL(38,0)",
        "text": "STRING",
        "timestamp": "TIMESTAMP",
        "double": "DOUBLE",
        "boolean": "BOOLEAN",
        "json": "STRING",
    }

    @property
    def namespace(self) -> str:
        """The Unity Catalog catalog of the schema."""
        connection = self.destination.connection
        assert isinstance(connection, DatabricksConnection)
        return connection.catalog

    def get_column_names(self, table: str) -> list[str] | None:
        """Return the table's column names in order; None if it is missing.

        Unity Catalog keeps names in lowercase, so the lookup lowercases too.
        """
        rows = self.fetch_all(
            f"SELECT column_name FROM {self.quote(self.namespace)}"
            ".information_schema.columns WHERE table_schema = lower(?) AND "
            "table_name = lower(?) ORDER BY ordinal_position",
            [self.destination.schema_name, table],
        )
        return [str(row[0]).lower() for row in rows] or None

    def replace_run_rows(self, writes: Sequence[TableRows]) -> None:
        """Replace each table's run rows atomically, one table at a time."""
        for write in writes:
            if not _RUN_ID.fullmatch(write.run_id):
                raise ValueError(f"not a run id: {write.run_id!r}")
            payloads = list(get_json_payloads(write)) or ["[]"]
            replace = (
                f"INSERT INTO {self.get_reference(write.table)} "
                f"REPLACE WHERE {self._get_replaced_rows(write)} "
            )
            if len(payloads) == 1:
                self.execute(
                    replace + self._get_payload_select(write.columns), payloads
                )
                continue
            staging = self.get_reference(f"{STAGING_TABLE_PREFIX}{uuid.uuid4().hex}")
            definitions = ", ".join(
                f"{self.quote(column.name)} {self.types[column.kind]}"
                for column in write.columns
            )
            try:
                self.execute(f"CREATE TABLE {staging} ({definitions})")
                for payload in payloads:
                    self.execute(
                        f"INSERT INTO {staging} "
                        f"{self._get_payload_select(write.columns)}",
                        [payload],
                    )
                self.execute(
                    replace
                    + f"SELECT {self.get_column_list(write.columns)} FROM {staging}"
                )
            finally:
                try:
                    self.execute(f"DROP TABLE IF EXISTS {staging}")
                except PredictionWriteError:
                    logger.warning("could not drop the staging table %s", staging)

    def quote(self, name: str) -> str:
        """Return a backquoted identifier."""
        return "`" + name.replace("`", "``") + "`"

    def _get_replaced_rows(self, write: TableRows) -> str:
        """Return the ``REPLACE WHERE`` condition of the rows ``write`` replaces.

        Literals are safe: the run id and pipeline id are checked against
        their generated shapes, and times are formatted by ``_get_text``.
        """
        condition = f"{self.quote('run_id')} = '{write.run_id}'"
        earlier = write.earlier_runs
        if earlier is None or not earlier.as_of_values:
            return condition
        if not _RUN_ID.fullmatch(earlier.pipeline_id):
            raise ValueError(f"not a pipeline id: {earlier.pipeline_id!r}")
        times = ", ".join(
            f"TIMESTAMP '{_get_text(value, 'timestamp')}'"
            for value in earlier.as_of_values
        )
        return (
            f"{condition} OR ({self.quote('pipeline_id')} = '{earlier.pipeline_id}'"
            f" AND {self.quote('as_of')} IN ({times}))"
        )

    def _get_payload_select(self, columns: Sequence[Column]) -> str:
        """Return a SELECT that reads the rows of one JSON parameter."""
        casts = ", ".join(
            f"CAST(r[{index}] AS {self.types[column.kind]})"
            for index, column in enumerate(columns)
        )
        # FAILFAST: a payload that does not parse fails the statement instead
        # of becoming NULL, which would replace the run's rows with none.
        return (
            f"SELECT {casts} FROM (SELECT explode(from_json(?, "
            "'ARRAY<ARRAY<STRING>>', map('mode', 'FAILFAST'))) AS r)"
        )


def _get_lock_keys(schema_name: str, writes: Sequence[TableRows]) -> set[str]:
    """Return the PostgreSQL lock keys of ``writes``.

    One per run, and one per pipeline and ``as_of`` whose earlier rows a
    write replaces.
    """
    keys = {write.run_id for write in writes}
    for write in writes:
        earlier = write.earlier_runs
        if earlier is not None:
            keys.update(
                f"{schema_name}.{write.table}|{earlier.pipeline_id}|{value.isoformat()}"
                for value in earlier.as_of_values
            )
    return keys


def get_json_payloads(write: TableRows) -> Iterator[str]:
    """Yield the write's rows as JSON arrays of text, at most MAX_PAYLOAD_BYTES each.

    Every value is text (or null), so numbers, wide ids and times keep their
    exact value until the SQL casts them to the column's type.
    """
    batch: list[str] = []
    size = 2
    for row in write.rows:
        encoded = json.dumps(
            [
                _get_text(value, column.kind)
                for value, column in zip(row, write.columns, strict=True)
            ]
        )
        if batch and size + len(encoded) + 1 > MAX_PAYLOAD_BYTES:
            yield "[" + ",".join(batch) + "]"
            batch, size = [], 2
        batch.append(encoded)
        size += len(encoded) + 1
    if batch:
        yield "[" + ",".join(batch) + "]"


def _get_text(value: Any, kind: ColumnKind) -> str | None:
    """Return ``value`` as the text its SQL cast reads back exactly."""
    if value is None:
        return None
    if kind == "timestamp":
        stamp = value.astimezone(UTC) if isinstance(value, datetime) else value
        return stamp.strftime("%Y-%m-%d %H:%M:%S.%f+00:00")
    if kind == "json":
        return json.dumps(value)
    if kind == "boolean":
        return "true" if value else "false"
    if kind == "double":
        return repr(float(value))
    if isinstance(value, Decimal):
        return format(value, "f")
    return str(value)


__all__ = [
    "MAX_PAYLOAD_BYTES",
    "DatabricksTableWriter",
    "PostgreSQLTableWriter",
    "PredictionWriteError",
    "SnowflakeTableWriter",
    "TableWriter",
    "get_json_payloads",
    "open_table_writer",
]
