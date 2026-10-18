"""Compile a separate entity predicate for scoring, using existing PQL filters."""

from datetime import datetime

from nori_rel.contracts import SchemaGraph
from nori_rel.pql.binding import bind_columns, get_entity_created_at_column
from nori_rel.pql.context_sql import _utc
from nori_rel.pql.dialects import SQLDialect
from nori_rel.pql.parser import parse_pql
from nori_rel.pql.relationships import bind_relationship
from nori_rel.pql.sql_expressions import Scalar, filter_sql
from nori_rel.pql.tree_access import node_ids, query_parts


def compile_scoring_filter_sql(
    tree,
    schema: SchemaGraph,
    query_filter: str,
    *,
    anchor_time: datetime,
    entity_created_at: str,
    dialect: SQLDialect | None = None,
) -> tuple[str, dict[str, Scalar]]:
    """Return SQL/params selecting eligible entity IDs that match query_filter.

    Input: original PQL tree, registered schema and e.g. "users.country='CA'".
    Output: (SELECT ... AS entity_id, {"scoring_anchor": aware datetime,
    "value_1": "CA"}), for the existing read-only transaction. No labels/features.

    Parse the predicate in a synthetic direct-target query using the same grammar,
    scope, binding and SQL predicate compiler. Validate the full input; neither
    arbitrary SQL nor a second filter evaluator is used. The synthetic entity
    reference comes from an already parsed column token. Locations refer to the
    complete synthetic PQL statement. Only immutable entity columns are supported.
    FOR/legacy IDs intersect these matches later, after missing-ID validation.
    """
    dialect = dialect or SQLDialect()
    identifier = dialect.identifier
    if not isinstance(query_filter, str) or not query_filter.strip():
        raise ValueError("query_filter must be a non-empty PQL predicate string")
    entity_ref = query_parts(tree)["entity_column"].getText()
    filter_tree = parse_pql(
        f"PREDICT {entity_ref} FOR EACH {entity_ref} WHERE {query_filter}"
    )
    parts = query_parts(filter_tree)
    bind_relationship(filter_tree, schema)  # Context filter must use this entity.
    ids, bound = node_ids(filter_tree), bind_columns(filter_tree, schema)
    entity, key = bound[ids[parts["entity_column"]]]
    entity_created_at = get_entity_created_at_column(
        schema, entity, entity_created_at
    ).name
    params: dict[str, Scalar] = {"scoring_anchor": _utc(anchor_time)}

    def column_ref(node):
        """Return ('u."country"', ColumnSchema) for a validated entity column."""
        column = bound[ids[node]][1]
        return f"u.{identifier(column.name)}", column

    predicate = filter_sql(parts["context_filter"], column_ref, params)
    table = dialect.table_reference(schema.schema_name, entity.name)
    sql = (
        f"SELECT u.{identifier(key.name)} AS entity_id FROM {table} AS u "
        f"WHERE u.{identifier(entity_created_at)} <= %(scoring_anchor)s "
        f"AND ({predicate}) ORDER BY u.{identifier(key.name)}"
    )
    return sql, params
