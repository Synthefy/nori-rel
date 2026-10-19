"""Build the context rows of a static PQL target from one source read.

A static target predicts a column the entity row stores, with no time window,
e.g. PREDICT accounts.balance; a temporal target predicts an aggregate of
future events. The names follow Kumo's static and temporal predictive queries.
"""

from collections.abc import Sequence
from datetime import datetime

from nori_rel.contracts import ColumnSchema, SchemaGraph
from nori_rel.pql._generated.PQLGrammarParser import PQLGrammarParser as Grammar
from nori_rel.pql.binding import bind_columns, get_entity_created_at_column
from nori_rel.pql.context_sql import (
    DEFAULT_MAX_CONTEXT_ROWS,
    _utc,
    get_numbered_context_sql,
    require_valid_context_row_limit,
)
from nori_rel.pql.dialects import SQLDialect
from nori_rel.pql.label_sql import compile_label_sql
from nori_rel.pql.scoring_sql import get_query_filter_predicate_sql
from nori_rel.pql.sql_expressions import (
    Scalar,
    _typed_literal,
    column_kind,
    filter_sql,
)
from nori_rel.pql.tree_access import node_ids, query_parts


def get_static_target_context_sql(
    tree: Grammar.ProgContext,
    schema: SchemaGraph,
    *,
    anchor_time: datetime,
    entity_created_at: str | None = None,
    max_context_rows: int | None = DEFAULT_MAX_CONTEXT_ROWS,
    query_filter: str | None = None,
    anchor_times: Sequence[datetime] | None = None,
    observed_through: datetime | None = None,
    entity_time: str | None = None,
    dialect: SQLDialect | None = None,
    candidate_sql: str | None = None,
) -> tuple[str, dict[str, Scalar]]:
    """Return a read-only SELECT of the context rows for a static target.

    A static target is a stored column of the entity row, e.g.
    ``PREDICT accounts.balance FOR EACH accounts.account_id``. It has no label
    window, so there are no historical anchors: every row comes from the same
    read as the scoring rows.

    Args:
        tree: parse_pql of a static target on the entity table, optionally
            compared with a literal, e.g. ``PREDICT users.income > 50000 FOR
            users.user_id IN (7, 42) WHERE users.country='US'``.
        schema: Registered SchemaGraph of the source.
        anchor_time: Aware datetime of the read, as the connector resolved it
            (PostgreSQL transaction time, warehouse pinned extraction time).
            Without entity_created_at every row's cutoff is this instant. When
            the entity table has a registered time column, rows dated after it
            are excluded, as the feature snapshot excludes them.
        entity_created_at: Optional date/timestamp column on the entity table
            holding each row's own time. When given, each row's cutoff is that
            time, rows with a NULL or later time are excluded, and rows are
            ordered newest first; otherwise they are ordered by primary key.
        max_context_rows: Positive cap on returned rows, or None for every
            row. The cap applies after the known-label, WHERE and scoring
            exclusions, so it counts usable rows only.
        query_filter: Optional scoring predicate, e.g. "users.country='CA'".
            A FOR row is scored only where it is true, so only those FOR rows
            leave the context; a FOR row it rejects, or evaluates to NULL for,
            stays.
        anchor_times: Accepted only so a temporal request fails with a clear
            message; anything but None raises.
        observed_through: Accepted for the same reason; must be None.
        entity_time: Accepted for the same reason; must be None. A static
            target has no label window to anchor at each row's time.
        dialect: Engine SQL syntax and optional snapshot-pinned relations.
        candidate_sql: Optional SELECT over ``candidates(entity_id,
            anchor_time)`` that narrows the rows; needs max_context_rows=None.

    Returns:
        (sql, params) for cursor.execute(sql, params). Rows are
        (context_row_id, entity_id, anchor_time), numbered 1.. in the order
        above. Every row has a known label: the target column is not NULL
        and, for a numeric column, not NaN, which scoring counts as missing.
        No scored row is returned. For ``FOR EACH`` the scored rows are the
        ones whose label is NULL, which the label rule already excludes; for
        ``FOR key = v`` or ``IN (...)`` those IDs are excluded, whether or not
        their label is stored, intersected with query_filter.

    Raises:
        ValueError: An aggregate target, anchor_times or observed_through, a
            target that is the entity's primary key, a naive anchor_time, a
            NULL scoring ID, invalid limits or an invalid creation column.
            PQL scope, binding and type errors propagate from the label
            compiler, including a target on another table.
    """
    dialect = dialect or SQLDialect()
    identifier = dialect.identifier
    parts = query_parts(tree)
    if parts["aggregate"] is not None:
        raise ValueError("static-target context SQL got an aggregate target")
    if anchor_times is not None or observed_through is not None:
        raise ValueError(
            "a static target has no label window: omit anchor_times and "
            "observed_through"
        )
    if entity_time is not None:
        raise ValueError(
            "a static target has no label window to anchor: omit entity_time; "
            "entity_created_at gives each row its own feature cutoff"
        )
    # Validate the target type, its table and the filters before execution.
    compile_label_sql(tree, schema, dialect=dialect)
    require_valid_context_row_limit(max_context_rows, candidate_sql)

    ids, bound = node_ids(tree), bind_columns(tree, schema)
    entity, key = bound[ids[parts["entity_column"]]]
    target_column = bound[ids[parts["target_column"]]][1]
    if target_column.name in entity.primary_key:
        raise ValueError("a static target cannot be the entity's primary key")
    params: dict[str, Scalar] = {"anchor_time": _utc(anchor_time)}
    if max_context_rows is not None:
        params["max_context_rows"] = max_context_rows

    def column_ref(node):
        """Return ('u."country"', ColumnSchema) for a bound entity column."""
        column = bound[ids[node]][1]
        return f"u.{identifier(column.name)}", column

    read_time = dialect.timestamp_cast("%(anchor_time)s")
    target = f"u.{identifier(target_column.name)}"
    conditions = [f"{target} IS NOT NULL"]
    if column_kind(target_column) == "number":
        # A NaN is stored but unknown: scoring counts it as missing, like NULL.
        conditions.append(dialect.get_not_nan_predicate(target))
    if entity.time_column is not None:
        # The feature snapshot keeps only entity rows dated by the read.
        conditions.append(f"u.{identifier(entity.time_column)} <= {read_time}")
    if entity_created_at is None:
        row_time, order_by = read_time, "entity_id"
    else:
        created_column = get_entity_created_at_column(schema, entity, entity_created_at)
        created = f"u.{identifier(created_column.name)}"
        conditions.append(f"{created} <= {read_time}")
        row_time = dialect.timestamp_cast(created)
        order_by = "anchor_time DESC, entity_id"
    context_filter = filter_sql(parts["context_filter"], column_ref, params)
    if context_filter:
        conditions.append(context_filter)
    selection = parts["selection"]
    if selection is not None:
        scoring_ids = ", ".join(get_scoring_id_placeholders(selection, key, params))
        scored = f"u.{identifier(key.name)} IN ({scoring_ids})"
        if query_filter is not None:
            predicate = get_query_filter_predicate_sql(
                tree, schema, query_filter, params, dialect=dialect
            )
            # A NULL scoring predicate does not score the row, so it stays.
            scored += f" AND COALESCE(({predicate}), FALSE)"
        conditions.append(f"NOT ({scored})")

    users = dialect.table_reference(schema.schema_name, entity.name)
    where = "\n      AND ".join(conditions)
    sql = f"""WITH selected AS (
    SELECT u.{identifier(key.name)} AS entity_id, {row_time} AS anchor_time
    FROM {users} AS u
    WHERE {where}"""
    if max_context_rows is not None:
        sql += f"""
    ORDER BY {order_by}
    LIMIT %(max_context_rows)s"""
    sql += "\n)"
    return (
        get_numbered_context_sql(sql, candidate_sql=candidate_sql, order_by=order_by),
        params,
    )


def get_scoring_id_placeholders(
    selection: Grammar.Entity_listContext | Grammar.Filtered_entity_listContext,
    key: ColumnSchema,
    params: dict[str, Scalar],
) -> list[str]:
    """Return one bound placeholder per FOR ID, adding the typed IDs to params.

    ``FOR users.user_id IN (7, 42)`` adds {"scoring_id_0": 7, "scoring_id_1":
    42} and returns ["%(scoring_id_0)s", "%(scoring_id_1)s"]. IDs are checked
    against the key column's type; NULL raises ValueError.
    """
    constant = selection.constant()
    nodes = constant.array().constant() if constant.array() is not None else [constant]
    placeholders = []
    for i, node in enumerate(nodes):
        value = _typed_literal(node, column_kind(key))
        if value is None:
            raise ValueError("scoring IDs cannot be NULL")
        params[f"scoring_id_{i}"] = value
        placeholders.append(f"%(scoring_id_{i})s")
    return placeholders
