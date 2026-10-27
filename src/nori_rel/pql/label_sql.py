"""Compile static/aggregate labels with shared joins and parameterized predicates."""

from functools import partial

from nori_rel.contracts import ColumnSchema, SchemaGraph, TableSchema
from nori_rel.pql._generated.PQLGrammarParser import PQLGrammarParser as Grammar
from nori_rel.pql.binding import PQLBindingError, bind_columns
from nori_rel.pql.dialects import SQLDialect
from nori_rel.pql.relationships import bind_relationship
from nori_rel.pql.sql_expressions import (
    Scalar,
    _literal,
    column_kind,
    comparison_sql,
    filter_sql,
    namespace_sql,
)
from nori_rel.pql.tree_access import (
    node_ids,
    query_parts,
    target_aggregates,
)
from nori_rel.pql.window import target_window


def _count_sql(
    column: ColumnSchema | None,
    foreign_key_sql: str,
    table: TableSchema,
    *,
    dialect: SQLDialect,
) -> str:
    """Emit COUNT(column), or COUNT(matched foreign key) for table.*.

    Input: column=None and foreign_key_sql='o."buyer_id"'. Output:
    'COUNT(o."buyer_id")', excluding the unmatched LEFT JOIN row. A supplied
    ColumnSchema(name, data_type, nullable) counts its non-null values instead.
    """
    operand = (
        foreign_key_sql if column is None else f"o.{dialect.identifier(column.name)}"
    )
    return f"COUNT({operand})"


def _distinct_sql(
    column: ColumnSchema | None,
    foreign_key_sql: str,
    table: TableSchema,
    *,
    dialect: SQLDialect,
) -> str:
    """Count distinct non-null scalar values using the engine's equality/collation.

    Input: ColumnSchema(name="product", data_type="TEXT", nullable=True).
    Output: 'COUNT(DISTINCT o."product")'. Empty/all-null inputs return zero.
    Wildcards and unsupported scalar types are rejected before execution.
    """
    if column is None:
        raise ValueError("COUNT_DISTINCT requires a scalar column")
    column_kind(column)
    return f"COUNT(DISTINCT o.{dialect.identifier(column.name)})"


def _numeric_sql(
    operation: str,
    column: ColumnSchema | None,
    foreign_key_sql: str,
    table: TableSchema,
    *,
    dialect: SQLDialect,
) -> str:
    """Emit a numeric SUM/AVG/MIN/MAX expression from registered column metadata.

    Input: operation='AVG', column with name='amount', data_type='NUMERIC',
    nullable=True. Output: 'AVG(o."amount")'. Non-numeric types raise ValueError.
    SUM returns zero only for no matching rows; all-null values remain NULL.
    AVG/MIN/MAX return NULL for empty/all-null windows. Operation is internal
    and comes from the fixed handler table, never arbitrary query text.
    """
    if column is None or column_kind(column) != "number":
        raise ValueError(f"{operation} requires a numeric column")
    expression = f"{operation}(o.{dialect.identifier(column.name)})"
    if operation == "SUM":
        return f"CASE WHEN COUNT({foreign_key_sql}) = 0 THEN 0 ELSE {expression} END"
    return expression


def _event_order(operation: str, table: TableSchema, dialect: SQLDialect) -> str:
    """Validate the unique event key and order ties consistently with event time."""
    keys = [c for c in table.columns if c.name in table.primary_key]
    if len(table.primary_key) != 1 or len(keys) != 1 or keys[0].nullable:
        raise ValueError(f"{operation} requires one non-null event primary key")
    column_kind(keys[0])  # Reject unsupported ordering types before SQL.
    direction = "ASC" if operation == "FIRST" else "DESC"
    return (
        f"o.{dialect.identifier(table.time_column)} {direction}, "
        f"o.{dialect.identifier(keys[0].name)} {direction}"
    )


def _ordered_sql(
    operation: str,
    column: ColumnSchema | None,
    foreign_key_sql: str,
    table: TableSchema,
    *,
    dialect: SQLDialect,
) -> str:
    """Select an ordered event value, retaining NULL and deterministic key ties.

    PostgreSQL uses its existing ARRAY_AGG, which retains NULL entries.
    Warehouse compilation wraps the raw value in a ranked relation instead,
    because Snowflake ARRAY_AGG drops NULL entries.
    """
    if column is None or column_kind(column) not in {"number", "bool"}:
        raise ValueError(f"{operation} requires a numeric or Boolean column")
    order = _event_order(operation, table, dialect)
    if not dialect.is_postgresql:
        return f"o.{dialect.identifier(column.name)}"
    return (
        f"(ARRAY_AGG(o.{dialect.identifier(column.name)} ORDER BY {order}) "
        f"FILTER (WHERE {foreign_key_sql} IS NOT NULL))[1]"
    )


_AGGREGATE_HANDLERS = {
    "COUNT": _count_sql,
    "COUNT_DISTINCT": _distinct_sql,
    **{name: partial(_ordered_sql, name) for name in ("FIRST", "LAST")},
    **{name: partial(_numeric_sql, name) for name in ("SUM", "AVG", "MIN", "MAX")},
}


def _single_label_sql(
    tree: Grammar.ProgContext,
    schema: SchemaGraph,
    *,
    aggregate_node=None,
    context_schema: str | None = "pg_temp",
    context_table: str = "pql_context",
    dialect: SQLDialect | None = None,
) -> tuple[str, dict[str, Scalar]]:
    """Compile supported v0 label expressions to a parameterized SELECT.

    Args:
        tree: parse_pql(text), e.g. COUNT(orders.*,0,30,days)=0 as the target.
            Supports COUNT(* / column), numeric SUM/AVG/MIN/MAX, static numeric
            or Boolean columns, and scalar comparisons. Filters support column
            comparisons, IN, IS NULL/IS NOT NULL and AND. Context WHERE applies
            to entity rows; inline WHERE applies to events before aggregation.
        schema: SchemaGraph with schema_name and tables containing name,
            columns [{name, data_type, nullable}], primary_key,
            foreign_keys [{columns, referred_table, referred_columns}], and
            time_column. E.g. orders.buyer_id -> users.user_id, event_time mapped
            to a timestamp. Static targets require no event table/time mapping.
        aggregate_node: Optional original aggregate leaf from an arithmetic target.
            Compiles that leaf without applying the final target comparison.
        context_schema: Literal schema name for prepared historical context rows;
            default pg_temp; None refers to an unqualified CTE. Source tables
            use schema.schema_name.
        context_table: Literal table name, default pql_context, with one row
            per unique context_row_id and (entity_id, anchor_time). E.g.
            (1,42,'2026-01-01'), (2,7,'2026-01-01'). These must already be eligible
            context rows with complete future windows and non-null keys/timestamps.
            Use UTC timestamps/session timezone. Entity attributes must be
            immutable or valid for every supplied anchor: versioned attribute
            joins are not supported. Static-target context must exclude scoring labels.
        dialect: Engine SQL syntax and optional snapshot-pinned source relations.
            Defaults to PostgreSQL for existing callers.

    Returns:
        (sql, params): psycopg statement and bound Python scalars. Example:
            sql, params = compile_label_sql(parse_pql(query), schema)
            rows = cursor.execute(sql, params).fetchall()
        Rows have shape (context_row_id, label), e.g. [(1, True), (2, False)] for a
        comparison or [(1, 2), (2, 0)] for COUNT. Order is unspecified. Numeric
        results may be int, Decimal or float depending on PostgreSQL types.
        Missing labels stay None; the later context builder must drop them.

        Parameters have shape {"horizon_days": 30, "value_1": "web", ...}, plus
        "start_days" for a delayed window such as (30,60,days). Values never
        enter SQL text; identifiers are quoted. Event windows are
        (anchor_time + start * 24 hours, anchor_time + end * 24 hours]; for
        (0,30,days) that is (anchor_time, anchor_time + 30 days]. COUNT and
        empty SUM return 0; all-null SUM and empty/all-null AVG/MIN/MAX return
        NULL. Comparisons are applied to observed labels before inference;
        unknown stays NULL.

    Raises:
        PQLScopeError: Syntax outside v0, such as OR or historical aggregates.
        PQLBindingError: Invalid names, relationships, filter scopes or time map.
        ValueError: Invalid types, unsupported literals, a non-positive horizon
            or an interval that does not end after it starts.

    FOR selects scoring entities; it does not narrow the provided historical
    cohort. The executor owns context eligibility, data completeness, static
    label exclusion, bounded reads, features and Nori calls. This compiles SQL
    only; it cannot verify those data guarantees from SchemaGraph metadata.
    """
    dialect = dialect or SQLDialect()
    identifier = dialect.identifier
    parts = query_parts(tree, aggregate=aggregate_node)
    ids = node_ids(tree)
    link = bind_relationship(tree, schema, aggregate=aggregate_node)
    bound = bind_columns(tree, schema)
    target_node = parts["target_column"]
    target, column = bound[ids[target_node]]
    entity, key = bound[ids[parts["entity_column"]]]
    params: dict[str, Scalar] = {}
    context = identifier(context_table)
    if context_schema is not None:
        context = f"{identifier(context_schema)}.{context}"
    users = dialect.table_reference(schema.schema_name, entity.name)
    joins = []
    ranked_order = None

    def column_ref(node, alias):
        """Return ('u."country"', ColumnSchema) for a bound column node and alias."""
        _, metadata = bound[ids[node]]
        return f"{alias}.{identifier(metadata.name)}", metadata

    if link is None or parts["context_filter"] is not None:
        joins.append(f"JOIN {users} AS u ON u.{identifier(key.name)} = e.entity_id")
    aggregate = parts["aggregate"]
    if aggregate is None:
        label, kind = column_ref(target_node, "u")
        if parts["comparison"] is None and column_kind(kind) not in {"number", "bool"}:
            raise ValueError("Static targets require numeric or Boolean columns")
    else:
        window = target_window(aggregate)
        timestamp = next(c for c in target.columns if c.name == target.time_column)
        try:
            temporal = column_kind(timestamp) in {"date", "timestamp"}
        except ValueError:
            temporal = False
        if not temporal:
            raise PQLBindingError("UNSUPPORTED_TIME_TYPE", target_node)
        events = dialect.table_reference(schema.schema_name, target.name)
        foreign_key = f"o.{identifier(link.columns[0])}"
        event_time = f"o.{identifier(target.time_column)}"
        bounds = window.predicate(event_time, "e.anchor_time", params, dialect=dialect)
        target_filter = filter_sql(
            parts["target_filter"], lambda n: column_ref(n, "o"), params
        )
        joins.append(f"""LEFT JOIN {events} AS o
  ON {foreign_key} = e.entity_id
 AND {bounds}""")
        if target_filter:
            joins[-1] += f"\n AND {target_filter}"
        operation = aggregate.AGGR().getText().upper()
        label = _AGGREGATE_HANDLERS[operation](
            column, foreign_key, target, dialect=dialect
        )
        if operation in {"FIRST", "LAST"} and not dialect.is_postgresql:
            ranked_order = _event_order(operation, target, dialect)
        kind = column_kind(column) if operation in {"FIRST", "LAST"} else "number"
        if kind == "bool" and parts["comparison"] is None:
            raise ValueError(
                "Boolean FIRST/LAST targets require an explicit comparison, e.g. =true"
            )
    if parts["comparison"] is not None:
        label = comparison_sql(parts["comparison"], label, kind, params)
    context_filter = filter_sql(
        parts["context_filter"], lambda n: column_ref(n, "u"), params
    )
    select = f"SELECT e.context_row_id, {label} AS label"
    if ranked_order is not None:
        select += (
            ", ROW_NUMBER() OVER (PARTITION BY e.context_row_id "
            f"ORDER BY {ranked_order}) AS __nori_rank"
        )
    lines = [
        select,
        f"FROM {context} AS e",
        *joins,
    ]
    if context_filter:
        lines.append(f"WHERE {context_filter}")
    if aggregate is not None and ranked_order is None:
        lines.append("GROUP BY e.context_row_id")
    sql = "\n".join(lines)
    if ranked_order is not None:
        sql = (
            "SELECT context_row_id, label FROM (\n"
            + sql
            + "\n) AS __nori_ordered WHERE __nori_rank = 1"
        )
    return sql, params


def compile_label_sql(
    tree: Grammar.ProgContext,
    schema: SchemaGraph,
    *,
    context_schema: str | None = "pg_temp",
    context_table: str = "pql_context",
    dialect: SQLDialect | None = None,
) -> tuple[str, dict[str, Scalar]]:
    """Compile a target into a read-only SELECT of (context_row_id, label).

    Input: original parsed PQL, registered SchemaGraph, and a relation containing
    unique context_row_id/entity_id/anchor_time rows. See _single_label_sql for
    source guarantees. Output: (SQL, bound scalar params), executable by psycopg.
    Example: SUM(orders.amount,0,30,days)/COUNT(orders.*,0,30,days) returns one
    numeric ratio per context row. Empty denominators yield NULL, not infinity.

    Every aggregate is reduced separately before joining by context_row_id;
    separate event tables never multiply one another's rows. Arithmetic uses
    PostgreSQL NUMERIC, preserving decimal literals and avoiding integer division.
    Warehouses retain native decimal scales and follow their finite precision
    arithmetic rules. NULLs propagate. Comparisons apply to historical labels
    before inference. Original simple targets retain their existing SQL and types.
    """
    dialect = dialect or SQLDialect()
    expression = tree.target().expression()
    if expression is None:
        return _single_label_sql(
            tree,
            schema,
            context_schema=context_schema,
            context_table=context_table,
            dialect=dialect,
        )
    query_parts(tree)  # Validate the full expression before compiling leaves.
    aggregates = target_aggregates(tree)
    params: dict[str, Scalar] = {}
    ctes = []
    references = {}
    bound, ids = bind_columns(tree, schema), node_ids(tree)
    for i, aggregate in enumerate(aggregates):
        sql, values = _single_label_sql(
            tree,
            schema,
            aggregate_node=aggregate,
            context_schema=context_schema,
            context_table=context_table,
            dialect=dialect,
        )
        sql, values = namespace_sql(sql, values, f"agg_{i}")
        ctes.append(f"__nori_agg_{i} AS ({sql})")
        params.update(values)
        column = bound[ids[query_parts(tree, aggregate=aggregate)["target_column"]]][1]
        operation = aggregate.AGGR().getText().upper()
        if operation in {"COUNT", "COUNT_DISTINCT"}:
            column = None
        # AVG of an integer column is fractional. Keep the warehouse's native
        # AVG result type rather than recasting it to the input's zero scale.
        references[aggregate] = (
            f"a{i}.label"
            if operation == "AVG" and not dialect.is_postgresql
            else dialect.numeric_cast(f"a{i}.label", column=column, operation=operation)
        )

    def emit(node):
        """Reduce an ExpressionContext to trusted SQL; literals remain bound."""
        if node.aggregation() is not None:
            return references[node.aggregation()]
        if node.constant() is not None:
            name = f"literal_{len(params)}"
            params[name] = _literal(node.constant())
            return dialect.numeric_cast(f"%({name})s", literal=params[name])
        children = node.expression()
        if len(children) == 1:
            value = emit(children[0])
            if node.MINUS() is not None:
                return f"(-({value}))"
            return f"({value})"
        left, right = map(emit, children)
        if node.SLASH() is not None:
            return f"(({left}) / NULLIF(({right}), 0))"
        operator = (
            "*" if node.STAR() is not None else "+" if node.PLUS() is not None else "-"
        )
        return f"(({left}) {operator} ({right}))"

    label = emit(expression)
    if tree.target().REL_OP() is not None:
        label = comparison_sql(tree.target(), label, "number", params)
    # Every CTE is built from the same context relation and the same context
    # filter, and a leaf's inline predicate lives in its LEFT JOIN ON clause,
    # so all CTEs emit exactly the same rows and this is equivalent to an inner
    # join. FULL JOIN is defensive: if a leaf ever narrows its own row set at
    # the context level, the row surfaces with a NULL label and is counted in
    # dropped_unknown_labels, rather than vanishing from the output. MySQL has
    # no FULL JOIN; a LEFT JOIN from the first leaf keeps that guarantee for
    # every row the first leaf emits, which is every context row.
    join = "FULL JOIN" if dialect.supports_full_join else "LEFT JOIN"
    joins = "\n".join(
        f"{join} __nori_agg_{i} AS a{i} USING (context_row_id)"
        for i in range(1, len(aggregates))
    )
    sql = (
        "WITH "
        + ",\n".join(ctes)
        + f"\nSELECT context_row_id, {label} AS label\nFROM __nori_agg_0 AS a0\n"
        + joins
    )
    return sql, params
