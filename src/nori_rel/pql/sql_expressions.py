"""Typed literals and parameterized predicates for the shared label compiler."""

import json
import re
from collections.abc import Callable
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation

from nori_rel.contracts import ColumnSchema
from nori_rel.pql._generated.PQLGrammarParser import PQLGrammarParser as Grammar
from nori_rel.pql.dialects import SQLDialect

Scalar = str | int | Decimal | bool | date | datetime | None


def identifier(name: str) -> str:
    """Quote one literal identifier; escape quotes and psycopg percent markers.

    Input Order.Events becomes "Order.Events", not a two-part name. Empty names
    and NUL raise ValueError. Call separately for schema/table/column names.
    """
    return SQLDialect().identifier(name)


def column_kind(column: ColumnSchema) -> str:
    """Reduce registered SQL metadata to a supported scalar family.

    Input: ColumnSchema(name="amount", data_type="NUMERIC(12, 2)", nullable=True).
    Output: "number". Other families are text, bool, date and timestamp.
    Unsupported types raise ValueError; no implicit text-to-number conversion.
    """
    name = re.sub(r"\([^)]*\)", "", column.data_type.upper()).strip()
    if name in {
        "SMALLINT",
        "INTEGER",
        "BIGINT",
        "INT",
        "INT2",
        "INT4",
        "INT8",
        "NUMERIC",
        "DECIMAL",
        "NUMBER",
        "BYTE",
        "TINYINT",
        "SHORT",
        "LONG",
        "REAL",
        "FLOAT",
        "FLOAT4",
        "FLOAT8",
        "DOUBLE",
        "DOUBLE PRECISION",
        "SMALLSERIAL",
        "SERIAL",
        "BIGSERIAL",
    }:
        return "number"
    if name in {"BOOL", "BOOLEAN"}:
        return "bool"
    if name in {
        "TEXT",
        "VARCHAR",
        "CHAR",
        "CHARACTER",
        "CHARACTER VARYING",
        "STRING",
    }:
        return "text"
    if name == "DATE":
        return "date"
    if name in {
        "TIMESTAMP",
        "TIMESTAMPTZ",
        "TIMESTAMP WITH TIME ZONE",
        "TIMESTAMP WITHOUT TIME ZONE",
        "TIMESTAMP_NTZ",
        "TIMESTAMP_LTZ",
        "TIMESTAMP_TZ",
    }:
        return "timestamp"
    raise ValueError(f"Unsupported SQL scalar type: {column.data_type}")


def _literal(node: Grammar.ConstantContext) -> Scalar:
    """Decode a scalar grammar node; e.g. 1.25 -> Decimal('1.25'), true -> True.

    Strings support JSON-style escapes with either quote delimiter. Dates use
    ISO YYYY-MM-DD, optionally with an ISO time. Other forms raise ValueError.
    Literal values are returned as Python objects, never inserted into SQL.
    """
    text = node.getText()
    if node.NULL() is not None:
        return None
    if node.BOOL() is not None:
        return text.lower() == "true"
    if node.INT() is not None:
        return int(text)
    if node.DECIMAL() is not None:
        try:
            value = Decimal(text)
        except InvalidOperation:
            raise ValueError("Invalid numeric literal") from None
        if not value.is_finite():
            raise ValueError("Non-finite numeric literal")
        return value
    if node.STR() is not None:
        body = text[1:-1]
        if text[0] == "'":
            body = re.sub(
                r'\\.|"',
                lambda m: "'" if m[0] == "\\'" else '\\"' if m[0] == '"' else m[0],
                body,
            )
        value = json.loads('"' + body + '"', strict=False)
        if "\x00" in value:
            raise ValueError("NUL is not supported in PostgreSQL text")
        return value
    temporal = node.datetime()
    if temporal is not None:
        day = date.fromisoformat(temporal.DATE().getText())
        if temporal.TIME() is None:
            return day
        return datetime.fromisoformat(
            f"{day.isoformat()}T{temporal.TIME().getText()}"
        ).replace(tzinfo=UTC)
    raise ValueError("Unsupported scalar literal")


def _typed_literal(node: Grammar.ConstantContext, kind: str) -> Scalar:
    """Decode and check a literal against number/text/bool/date/timestamp.

    Input: node for '2026-01-01', kind="date". Output: date(2026, 1, 1).
    Mixed-type comparisons raise ValueError rather than relying on SQL coercion.
    NULL stays None, including inside IN lists.
    """
    value = _literal(node)
    if value is None:
        return None
    if kind == "date":
        if isinstance(value, str):
            value = date.fromisoformat(value)
        if type(value) is date:
            return value
    if kind == "timestamp":
        if isinstance(value, str):
            value = datetime.fromisoformat(value)
        if type(value) is date:
            value = datetime.combine(value, datetime.min.time(), tzinfo=UTC)
        if isinstance(value, datetime):
            return value.replace(tzinfo=UTC) if value.tzinfo is None else value
    if (
        (kind == "number" and type(value) in {int, Decimal})
        or (kind == "bool" and type(value) is bool)
        or (kind == "text" and type(value) is str)
    ):
        return value
    raise ValueError(f"Literal does not match {kind} column/expression")


def comparison_sql(
    node: Grammar.ConditionContext,
    left: str,
    kind: str | ColumnSchema,
    params: dict[str, Scalar],
) -> str:
    """Emit a comparison and append bound values to params in place.

    Inputs: condition node for amount>20, trusted left='o."amount"', kind="number",
    params={}. Outputs: '(o."amount" > %(value_0)s)', params={"value_0": 20}.
    Supports = != < <= > >= IN IS NULL/IS NOT NULL. NULL comparisons preserve SQL
    three-valued logic; = NULL is rejected in favor of explicit IS NULL.
    kind may also be ColumnSchema(name, data_type, nullable); its scalar type
    is checked only for literal comparisons. IS NULL works on any column type.
    """
    operator = node.REL_OP().getText().upper()
    constant = node.constant()
    if operator in {"IS", "IS NOT"}:
        if constant.NULL() is None:
            raise ValueError("IS requires NULL")
        return f"({left} {operator} NULL)"
    if operator not in {"=", "!=", "<", "<=", ">", ">=", "IN"}:
        raise ValueError("Unsupported comparison operator")
    if constant.NULL() is not None:
        raise ValueError("Use IS NULL or IS NOT NULL instead of comparing to NULL")
    if isinstance(kind, ColumnSchema):
        kind = column_kind(kind)
    values = constant.array().constant() if operator == "IN" else [constant]
    placeholders = []
    for item in values:
        value = _typed_literal(item, kind)
        name = f"value_{len(params)}"
        params[name] = value
        placeholders.append(f"%({name})s")
    right = ", ".join(placeholders)
    if operator == "IN":
        right = f"({right})"
    return f"({left} {operator} {right})"


def filter_sql(
    node: Grammar.ConditionContext | None,
    column_ref: Callable[[Grammar.ColumnContext], tuple[str, ColumnSchema]],
    params: dict[str, Scalar],
) -> str:
    """Compile an optional AND filter using a schema-bound column resolver.

    column_ref(column_node) returns (quoted SQL reference, ColumnSchema), e.g.
    ('u."country"', ColumnSchema(name='country', data_type='TEXT', nullable=True)).
    Returns SQL and fills params through comparison_sql;
    an absent filter returns ''. The caller chooses JOIN-ON versus WHERE scope.
    """
    if node is None:
        return ""
    pending, clauses = [node], []
    while pending:
        current = pending.pop()
        if current.AND() is not None:
            pending.extend(reversed(current.condition()))
        else:
            left, kind = column_ref(current.column())
            clauses.append(comparison_sql(current, left, kind, params))
    return " AND ".join(clauses)


def namespace_sql(sql: str, params: dict[str, Scalar], prefix: str):
    """Prefix generated parameter names, preserving escaped identifier percents.

    Input: ('x=%(value_0)s', {'value_0': 'US'}, 'context').
    Output: ('x=%(context_value_0)s', {'context_value_0': 'US'}).
    %% escapes are left intact, even inside identifiers resembling placeholders.
    """
    sql = re.sub(
        r"%%|%\(([^)]+)\)s",
        lambda m: f"%({prefix}_{m[1]})s" if m[1] else m[0],
        sql,
    )
    return sql, {f"{prefix}_{name}": value for name, value in params.items()}
