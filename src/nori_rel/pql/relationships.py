"""Validate the single direct relationship allowed by the v0 query contract."""

from nori_rel.contracts import ForeignKeySchema, SchemaGraph, TableSchema
from nori_rel.pql._generated.PQLGrammarParser import PQLGrammarParser as Grammar
from nori_rel.pql.binding import PQLBindingError, bind_columns
from nori_rel.pql.tree_access import node_ids, query_parts


def bind_relationship(
    tree: Grammar.ProgContext, schema: SchemaGraph, *, aggregate=None
) -> ForeignKeySchema | None:
    """Resolve the single event-to-entity foreign key required by a v0 query.

    Args:
        tree: ProgContext from parse_pql(text), e.g. the COUNT query below.
        aggregate: Optional aggregate node to bind within a composed target.
            None preserves single-target behavior; each leaf binds independently.
        schema: SchemaGraph containing registered tables, not actual data rows.
            Each table has name, columns [{name, data_type, nullable}],
            primary_key [column names], foreign_keys [link objects], and optional
            time_column. Each link has columns (on this table), referred_table,
            referred_columns (on that table), and optional name.
            The example shows the complete minimal validated input.

    Returns:
        ForeignKeySchema for an aggregate, using the original object from the
        target table. Its structure is:
            name: str | None -- optional constraint name.
            columns: list[str] -- child/event key, e.g. ["user_id"] on orders.
            referred_table: str -- parent/entity table, e.g. "users".
            referred_columns: list[str] -- parent key, e.g. ["user_id"].
        V0 requires one key column on each side. A direct target on the entity
        table (PREDICT users.age ...) returns None because no join is needed.
        None never means an unresolved/missing relationship; that raises.

    Example:
        >>> from nori_rel.pql import parse_pql
        >>> schema = SchemaGraph.model_validate({
        ...     "database_id": "db", "schema_name": "public",
        ...     "discovered_at": "2026-01-01T00:00:00Z",
        ...     "tables": [
        ...         {"name": "users", "primary_key": ["user_id"], "columns": [
        ...             {"name": "user_id", "data_type": "INTEGER", "nullable": False},
        ...         ]},
        ...         {"name": "orders", "time_column": "event_time", "columns": [
        ...             {"name": "user_id", "data_type": "INTEGER", "nullable": False},
        ...             {"name": "event_time", "data_type": "TIMESTAMPTZ",
        ...              "nullable": False},
        ...         ], "foreign_keys": [{
        ...             "columns": ["user_id"], "referred_table": "users",
        ...             "referred_columns": ["user_id"],
        ...         }]},
        ...     ],
        ... })
        >>> tree = parse_pql("PREDICT COUNT(orders.*,0,30,days)=0 FOR users.user_id=42")
        >>> link = bind_relationship(tree, schema)
        >>> from pprint import pprint
        >>> pprint(link.model_dump(), sort_dicts=False)
        {'name': None,
         'columns': ['user_id'],
         'referred_table': 'users',
         'referred_columns': ['user_id']}
        >>> link is schema.tables[1].foreign_keys[0]
        True

    Raises:
        PQLScopeError: Query form is outside v0.
        PQLBindingError: Bad column names, entity key, filter table, foreign key
            or timestamp mapping; missing/ambiguous/multi-hop links also fail.
            Error exposes code, line/column and start_offset/end_offset.

    Reads metadata only. Context filters must use the entity table; target
    filters must use the aggregated table. Operand/timestamp type checking,
    numeric window validation and execution remain separate work.
    """
    parts = query_parts(tree, aggregate=aggregate)
    ids = node_ids(tree)
    columns = bind_columns(tree, schema)
    entity_node, target_node = parts["entity_column"], parts["target_column"]
    entity, key = columns[ids[entity_node]]
    target, _ = columns[ids[target_node]]
    if entity.primary_key != [key.name]:
        raise PQLBindingError("UNSUPPORTED_ENTITY_KEY", entity_node)

    def check_filter(node, expected: TableSchema) -> None:
        """Check a filter ConditionContext against its allowed TableSchema.

        Example: node for users.country='US', expected=users table object.
        None is an absent filter. Returns None or raises INVALID_FILTER_TABLE
        at any column node bound to a different table.
        """
        if node is None:
            return
        for child in node_ids(node):
            if isinstance(child, Grammar.ColumnContext):
                table, _ = columns[ids[child]]
                if table is not expected:
                    raise PQLBindingError("INVALID_FILTER_TABLE", child)

    check_filter(parts["context_filter"], entity)
    check_filter(parts["target_filter"], target)
    if parts["aggregate"] is None:
        if target is not entity:
            raise PQLBindingError("CROSS_TABLE_DIRECT_TARGET", target_node)
        return None

    links = [
        link
        for link in target.foreign_keys
        if link.referred_table == entity.name
        and link.referred_columns == entity.primary_key
        and len(link.columns) == 1
    ]
    if len(links) != 1:
        code = "NO_DIRECT_RELATIONSHIP" if not links else "AMBIGUOUS_RELATIONSHIP"
        raise PQLBindingError(code, target_node)
    link = links[0]
    if sum(column.name == link.columns[0] for column in target.columns) != 1:
        raise PQLBindingError("INVALID_FOREIGN_KEY", target_node)
    if target.time_column is None:
        raise PQLBindingError("MISSING_TIME_COLUMN", target_node)
    if sum(column.name == target.time_column for column in target.columns) != 1:
        raise PQLBindingError("INVALID_TIME_COLUMN", target_node)
    return link
