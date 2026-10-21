"""Resolve v0 column references against existing registered-schema metadata."""

import re
from collections import defaultdict

from nori_rel.contracts import ColumnSchema, SchemaGraph, TableSchema
from nori_rel.pql._generated.PQLGrammarParser import PQLGrammarParser as Grammar
from nori_rel.pql.sql_expressions import column_kind
from nori_rel.pql.tree_access import (
    leaf_target,
    node_ids,
    query_parts,
    source_span,
    target_aggregates,
)

# The lexer emits a qualified column as one token. Split only that validated
# token, respecting dots inside backticks; this is not another PQL parser.
_NAME = r"(?:`[^`\r\n]+`|[_a-zA-Z][_a-zA-Z0-9]*)"
_COLUMN = re.compile(rf"({_NAME})\.({_NAME}|\*)")
BoundColumn = tuple[TableSchema, ColumnSchema | None]


class PQLBindingError(ValueError):
    """A column cannot be resolved, with a code and its original source location."""

    def __init__(self, code: str, node: Grammar.ColumnContext) -> None:
        """Create a binding error from a code string and offending ColumnContext.

        Example input: code="UNKNOWN_TABLE", node for missing.age.
        Output attributes: code, one-based line/column, zero-based
        start_offset/end_offset (end excluded). Message contains the code and
        location, not the query text. Also used for relationship binding errors.
        """
        self.code = code
        self.line = node.start.line
        self.column = node.start.column + 1
        self.start_offset, self.end_offset = source_span(node)
        super().__init__(f"{code} at line {self.line}, column {self.column}")


def _names(node: Grammar.ColumnContext) -> tuple[str, str]:
    """Decode one qualified ColumnContext into (table_name, column_name) strings.

    Example: a node containing `Order.Events`.`Total.Amount` returns
    ("Order.Events", "Total.Amount"). Backticks are removed; case/dots are kept.
    Raises PQLBindingError("INVALID_COLUMN_REFERENCE", node) for invalid shape.
    """
    match = _COLUMN.fullmatch(node.getText())
    if match is None:
        raise PQLBindingError("INVALID_COLUMN_REFERENCE", node)
    table, column = match.groups()
    return (
        table[1:-1] if table.startswith("`") else table,
        column[1:-1] if column.startswith("`") else column,
    )


def get_registered_column_matches(
    schema: SchemaGraph,
    table: TableSchema,
    name: str,
    *,
    quoted: bool = False,
) -> list[ColumnSchema]:
    """Match exact column names first, with an engine-declared unquoted fallback.

    Snowflake declares uppercase matching for unquoted names. Backtick-quoted
    PQL references remain exact. Return every matching metadata row so callers
    can refuse ambiguities instead of silently selecting one column.
    """
    exact = [column for column in table.columns if column.name == name]
    if exact or quoted or schema.capabilities.unquoted_identifier_case != "upper":
        return exact
    return [column for column in table.columns if column.name.upper() == name.upper()]


def is_date_or_timestamp_column(column: ColumnSchema) -> bool:
    """Return True when a registered column's SQL type is a date or timestamp.

    Example: ColumnSchema(name="created_at", data_type="TIMESTAMPTZ") is True;
    INTEGER, TEXT and types PQL cannot compare at all, such as JSONB, are False.
    """
    try:
        return column_kind(column) in {"date", "timestamp"}
    except ValueError:  # column_kind rejects JSON, arrays and other types.
        return False


def get_entity_created_at_column(
    schema: SchemaGraph,
    table: TableSchema,
    entity_created_at: str,
    *,
    argument: str = "entity_created_at",
) -> ColumnSchema:
    """Return the entity table's creation-time column named by entity_created_at.

    Args:
        schema: Registered SchemaGraph; its capabilities decide whether an
            unquoted name may match case-insensitively (Snowflake).
        table: The FOR EACH entity table, e.g. users for users.user_id.
        entity_created_at: Column name on that table, e.g. "created_at",
            holding when each entity first existed. Not table-qualified.
        argument: The predict argument that named the column, for the error
            message: "entity_created_at", or "entity_time" for event-time rows.

    Returns:
        The single matching ColumnSchema, whose type is a date or timestamp.

    Raises:
        ValueError: The name matches no column of the entity table, matches
            more than one, or matches a column of another type. The message
            says which, and lists the table's date/timestamp columns or says
            it has none and one must be added.
    """
    matches = get_registered_column_matches(schema, table, entity_created_at)
    if len(matches) == 1 and is_date_or_timestamp_column(matches[0]):
        return matches[0]
    if not matches:
        problem = f"does not match a column of {table.name}"
    elif len(matches) > 1:
        names = ", ".join(column.name for column in matches)
        problem = (
            f"matches more than one column of {table.name} ({names}); "
            "pass the exact name"
        )
    else:
        problem = (
            f"names {table.name}.{matches[0].name}, "
            f"which is {matches[0].data_type}, not a date or timestamp"
        )
    candidates = [
        column.name for column in table.columns if is_date_or_timestamp_column(column)
    ]
    if candidates:
        hint = f"Date/timestamp columns on {table.name}: {', '.join(candidates)}."
    else:
        hint = (
            f"{table.name} has no date or timestamp column: add one, for example "
            "each entity's first event time. See docs/pql.md."
        )
    meaning = (
        "holding when each event happened"
        if argument == "entity_time"
        else "holding when each entity first existed"
    )
    raise ValueError(
        f"{argument}={entity_created_at!r} {problem}. It must name a date "
        f"or timestamp column on the entity table {table.name} {meaning}. {hint}"
    )


def bind_columns(
    tree: Grammar.ProgContext, schema: SchemaGraph
) -> dict[int, BoundColumn]:
    """Resolve every column reference to its original registered metadata.

    Args:
        tree: ProgContext from parse_pql(text); must pass v0 syntax checks.
        schema: SchemaGraph object. Minimal input shape is shown below:
            database_id/schema_name: strings identifying the registered DB.
            discovered_at: timezone-aware datetime (or ISO string when validated).
            tables: list of TableSchema objects, each with name and columns.
            columns: list of ColumnSchema(name, data_type, nullable).
            Optional table fields: primary_key (names), foreign_keys (links),
            time_column (name or None). This function only binds names.

    Returns:
        dict[int, tuple[TableSchema, ColumnSchema | None]] keyed by node_ids(tree).
        Example shape: {column_node_id: (original_table, original_column)}.
        For COUNT(orders.*), the value is (orders_table, None); None means row
        wildcard, never an unresolved column. Returned objects are not copies.

    Example:
        >>> from nori_rel.pql import parse_pql
        >>> schema = SchemaGraph.model_validate({
        ...     "database_id": "db", "schema_name": "public",
        ...     "discovered_at": "2026-01-01T00:00:00Z",
        ...     "tables": [{"name": "users", "columns": [
        ...         {"name": "user_id", "data_type": "INTEGER", "nullable": False},
        ...         {"name": "age", "data_type": "INTEGER", "nullable": True},
        ...     ]}],
        ... })
        >>> tree = parse_pql("PREDICT users.age FOR EACH users.user_id")
        >>> target_id = node_ids(tree)[query_parts(tree)["target_column"]]
        >>> table, column = bind_columns(tree, schema)[target_id]
        >>> table.name, column.model_dump()
        ('users', {'name': 'age', 'data_type': 'INTEGER', 'nullable': True})
        >>> table is schema.tables[0], column is table.columns[1]
        (True, True)

    Raises:
        PQLScopeError: Query form is outside v0.
        PQLBindingError: Unknown/ambiguous table or column, or invalid wildcard.

    Matching is exact first, including quoted dots. A schema may declare an
    uppercase fallback for unquoted names (Snowflake); a nonunique fallback
    fails. Backtick-quoted names are always exact. Makes no data reads or schema
    mutations. Does not validate relationships, types or time windows.
    """
    query_parts(tree)  # Check scope before binding any names.
    tables = defaultdict(list)
    for table in schema.tables:
        tables[table.name].append(table)

    bound = {}
    for node, identifier in node_ids(tree).items():
        if not isinstance(node, Grammar.ColumnContext):
            continue
        table_name, column_name = _names(node)
        matching_tables = tables[table_name]
        table_token, column_token = _COLUMN.fullmatch(node.getText()).groups()
        if (
            not matching_tables
            and not table_token.startswith("`")
            and schema.capabilities.unquoted_identifier_case == "upper"
        ):
            matching_tables = [
                table
                for table in schema.tables
                if table.name.upper() == table_name.upper()
            ]
        if len(matching_tables) != 1:
            code = "UNKNOWN_TABLE" if not matching_tables else "AMBIGUOUS_TABLE"
            raise PQLBindingError(code, node)
        table = matching_tables[0]
        if node.WILDCARD_COLUMN() is not None:
            allowed = any(
                leaf_target(a)[0] is node and a.AGGR().getText().upper() == "COUNT"
                for a in target_aggregates(tree)
            )
            if not allowed:
                raise PQLBindingError("INVALID_WILDCARD", node)
            bound[identifier] = (table, None)
        else:
            matching_columns = get_registered_column_matches(
                schema, table, column_name, quoted=column_token.startswith("`")
            )
            if len(matching_columns) != 1:
                code = "UNKNOWN_COLUMN" if not matching_columns else "AMBIGUOUS_COLUMN"
                raise PQLBindingError(code, node)
            bound[identifier] = (table, matching_columns[0])
    return bound
