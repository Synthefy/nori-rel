"""SQL syntax shared by PostgreSQL and warehouse PQL compilation.

Statements retain pyformat placeholders and escaped percent signs. Warehouse
executors must translate those placeholders to native parameters, including
unescaping ``%%`` in identifiers, before passing SQL to their drivers.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal

from nori_rel.contracts import ColumnSchema, DatabaseEngine

_DECIMAL_PRECISION_BY_SQL_INTEGER_TYPE = {
    "BYTE": 3,
    "TINYINT": 3,
    "SMALLINT": 5,
    "SHORT": 5,
    "INT2": 5,
    "SMALLSERIAL": 5,
    "INTEGER": 10,
    "INT": 10,
    "INT4": 10,
    "SERIAL": 10,
    "BIGINT": 19,
    "LONG": 19,
    "INT8": 19,
    "BIGSERIAL": 19,
}


@dataclass(frozen=True)
class SQLDialect:
    """Render trusted syntax and connector-owned, snapshot-pinned references.

    Each entry in table_references is already quoted SQL produced by a connector,
    keyed by the original registered table name. Never populate it from PQL text.
    Warehouse decimals have at most 38 digits; their arithmetic follows native
    engine precision and scale rules rather than PostgreSQL unbounded NUMERIC.
    Sessions must use UTC so naive warehouse timestamps have a single meaning.
    """

    engine: DatabaseEngine = DatabaseEngine.POSTGRESQL
    table_references: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        """Validate engines even when a caller supplies a string."""
        object.__setattr__(self, "engine", DatabaseEngine(self.engine))

    @property
    def is_postgresql(self) -> bool:
        """Whether PostgreSQL's existing casts and aggregates are supported."""
        return self.engine == DatabaseEngine.POSTGRESQL

    def identifier(self, name: str) -> str:
        """Quote a literal identifier without splitting dots in its name."""
        if not name or "\x00" in name:
            raise ValueError("Invalid SQL identifier")
        quote = "`" if self.engine == DatabaseEngine.DATABRICKS else '"'
        return quote + name.replace(quote, quote * 2).replace("%", "%%") + quote

    def qualified_name(self, *parts: str) -> str:
        """Quote each separately supplied catalog/schema/table name."""
        return ".".join(self.identifier(part) for part in parts)

    def table_reference(self, schema_name: str, table_name: str) -> str:
        """Use a pinned source relation or quote the registered schema/table."""
        if self.table_references is not None:
            try:
                return self.table_references[table_name]
            except KeyError:
                raise ValueError(
                    f"No pinned source reference for {table_name}"
                ) from None
        return self.qualified_name(schema_name, table_name)

    def timestamp_cast(self, expression: str) -> str:
        """Cast an aware bound value to the engine's UTC timestamp type."""
        if self.is_postgresql:
            return f"{expression}::timestamptz"
        kind = (
            "TIMESTAMP_TZ" if self.engine == DatabaseEngine.SNOWFLAKE else "TIMESTAMP"
        )
        return f"CAST({expression} AS {kind})"

    def anchor_relation(self, placeholders: Sequence[str]) -> str:
        """Create a typed one-column relation, including the empty-anchor case."""
        if not placeholders:
            return f"SELECT {self.timestamp_cast('NULL')} WHERE FALSE"
        if self.is_postgresql:
            return "VALUES " + ", ".join(
                f"({self.timestamp_cast(value)})" for value in placeholders
            )
        # Bound expressions in a SELECT work with both native warehouse drivers;
        # Snowflake VALUES requires constants for some otherwise valid casts.
        return " UNION ALL ".join(
            f"SELECT {self.timestamp_cast(value)}" for value in placeholders
        )

    def add_days(self, anchor: str, placeholder: str) -> str:
        """Add exact 24-hour periods to an anchor, avoiding calendar-day DST."""
        if self.is_postgresql:
            return f"{anchor} + {placeholder} * INTERVAL '24 hours'"
        return f"TIMESTAMPADD(HOUR, {placeholder} * 24, {anchor})"

    def numeric_cast(
        self,
        expression: str,
        *,
        column: ColumnSchema | None = None,
        literal: int | Decimal | None = None,
        operation: str | None = None,
    ) -> str:
        """Keep fractional division and exact integers without fixed-scale loss.

        PostgreSQL keeps its existing unbounded NUMERIC cast. Warehouses keep
        native decimal/float aggregate types and cast integer operands to decimal.
        Decimal literals are cast at their own scale, rejecting over-38-digit
        values. Native engines can still round arithmetic results or raise on
        overflow according to their documented precision rules.
        """
        if self.is_postgresql:
            return f"{expression}::numeric"
        if literal is not None:
            decimal = Decimal(literal)
            _, digits, exponent = decimal.as_tuple()
            scale = max(0, -exponent)
            integer_digits = max(0, len(digits) + exponent)
            precision = max(1, integer_digits + scale)
            if precision > 38:
                raise ValueError("Warehouse numeric literals support at most 38 digits")
            return f"CAST({expression} AS DECIMAL({precision}, {scale}))"
        if column is not None:
            kind = re.sub(r"\([^)]*\)", "", column.data_type.upper()).strip()
            if kind in {
                "NUMERIC",
                "DECIMAL",
                "NUMBER",
                "REAL",
                "FLOAT",
                "FLOAT4",
                "FLOAT8",
                "DOUBLE",
                "DOUBLE PRECISION",
            }:
                return expression
            # Reserve only the digits the signed source type can hold. Casting
            # all integers to precision 38 makes Spark discard fractional digits
            # when later arithmetic must fit both operands into that precision.
            # SUM widens smaller integer inputs to a signed 64-bit result.
            precision = (
                19
                if operation == "SUM"
                else _DECIMAL_PRECISION_BY_SQL_INTEGER_TYPE[kind]
            )
        else:
            precision = 19  # COUNT/COUNT_DISTINCT are signed 64-bit results.
        return f"CAST({expression} AS DECIMAL({precision}, 0))"

    @property
    def materialized_cte(self) -> str:
        """Return the PostgreSQL CTE optimization fence when supported."""
        return "MATERIALIZED " if self.is_postgresql else ""
