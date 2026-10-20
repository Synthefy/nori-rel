"""Find the label's required events, so context candidates can start from them.

Some labels are unknown unless an aggregate has at least one event in its
window. A ratio's denominator is the common case: COUNT(paid) / COUNT(settled)
is 0 / 0, so unknown, for every (entity, cutoff) pair with no settled outcome
in the window. AVG, MIN, MAX, FIRST and LAST of an empty window are unknown
too. For such a target, every candidate with a known label has one of those
events, so candidates can be generated from the events inside each cutoff's
window instead of from every entity at every cutoff. In a trial-to-paid task
where each signup's outcome settles once, that is the signups whose outcome
settles in a cutoff's window, a few percent of every signup at every cutoff.

The analysis only has to return a necessary condition: the label SQL still
computes every label exactly and unknown labels are still dropped, so a
candidate set that is too large costs time, never correctness. A target whose
label is always known (COUNT, SUM) has no required event and keeps the
entity-by-cutoff candidates.
"""

from __future__ import annotations

from nori_rel.contracts import SchemaGraph
from nori_rel.pql._generated.PQLGrammarParser import PQLGrammarParser as Grammar
from nori_rel.pql.binding import bind_columns
from nori_rel.pql.dialects import SQLDialect
from nori_rel.pql.relationships import bind_relationship
from nori_rel.pql.sql_expressions import Scalar, filter_sql
from nori_rel.pql.tree_access import node_ids, query_parts, target_aggregates
from nori_rel.pql.window import target_window

#: Aggregates whose value over an empty window is NULL rather than 0.
_UNKNOWN_WHEN_EMPTY_AGGREGATES = frozenset({"AVG", "MIN", "MAX", "FIRST", "LAST"})

AggregateSet = frozenset[Grammar.AggregationContext]


def _get_aggregate_requirements(
    aggregate: Grammar.AggregationContext,
) -> tuple[AggregateSet, AggregateSet]:
    """Return (events needed to know the value, events needed for a nonzero one).

    Every aggregate is 0 or NULL over an empty window, so a nonzero value
    always needs one of its events; a known value needs one only for the
    aggregates that are NULL when empty.
    """
    operation = aggregate.AGGR().getText().upper()
    needs_event = frozenset({aggregate})
    known = needs_event if operation in _UNKNOWN_WHEN_EMPTY_AGGREGATES else frozenset()
    return known, needs_event


def _get_expression_requirements(
    node: Grammar.ExpressionContext,
) -> tuple[AggregateSet, AggregateSet]:
    """Return (aggregates that need an event for a known value, for a nonzero one).

    Each set is a conjunction: every aggregate in it must have an event. The
    rules follow SQL NULL arithmetic and the label SQL's ``NULLIF(right, 0)``
    division:

    - ``x + y`` and ``x - y`` are known when both sides are, and nonzero only
      if one side is, so they need what both sides' nonzero sets share.
    - ``x * y`` is nonzero only when both sides are.
    - ``x / y`` is known only when ``y`` is nonzero.

    A constant or a column needs nothing, even a zero constant: a weaker
    condition is still a necessary one.
    """
    if node.aggregation() is not None:
        return _get_aggregate_requirements(node.aggregation())
    if node.constant() is not None or node.column() is not None:
        return frozenset(), frozenset()
    children = node.expression()
    if len(children) == 1:
        return _get_expression_requirements(children[0])
    left_known, left_nonzero = _get_expression_requirements(children[0])
    right_known, right_nonzero = _get_expression_requirements(children[1])
    known = left_known | right_known
    if node.SLASH() is not None:
        known = known | right_nonzero
        return known, known | left_nonzero
    if node.STAR() is not None:
        return known, known | left_nonzero | right_nonzero
    return known, known | (left_nonzero & right_nonzero)


def get_label_event_aggregates(
    tree: Grammar.ProgContext,
) -> list[Grammar.AggregationContext]:
    """Return the target's aggregates that need an event for a known label.

    The list is in source order and is empty when the label is always known.
    For ``COUNT(paid_outcomes.* WHERE status = 'paid', 6, 7, days) /
    COUNT(paid_outcomes.* WHERE status IN ('paid', 'not_paid'), 6, 7, days)``
    it holds the denominator. A comparison on top of the target does not
    change it: comparing an unknown value gives an unknown label.
    """
    target = tree.target()
    expression = target.expression()
    if expression is not None:
        required, _ = _get_expression_requirements(expression)
    else:
        aggregate = query_parts(tree)["aggregate"]
        required = (
            frozenset()
            if aggregate is None
            else _get_aggregate_requirements(aggregate)[0]
        )
    return [aggregate for aggregate in target_aggregates(tree) if aggregate in required]


def get_label_event_candidates_sql(
    tree: Grammar.ProgContext,
    schema: SchemaGraph,
    aggregate: Grammar.AggregationContext,
    *,
    anchors_relation: str,
    params: dict[str, Scalar],
    dialect: SQLDialect,
) -> str:
    """Return SQL for the distinct (entity_id, anchor_time) pairs with an event.

    The SELECT reads ``aggregate``'s event table joined to ``anchors_relation``
    (one ``anchor_time`` column) and keeps each event inside the aggregate's
    label window, ``(anchor_time + start, anchor_time + end]``, that passes its
    inline WHERE. ``entity_id`` is the event's foreign key to the entity table;
    the caller still checks that the entity exists by the anchor and passes
    the query's WHERE. The window and filter values are added to ``params``.
    """
    identifier = dialect.identifier
    parts = query_parts(tree, aggregate=aggregate)
    ids, bound = node_ids(tree), bind_columns(tree, schema)
    events_table, _ = bound[ids[parts["target_column"]]]
    link = bind_relationship(tree, schema, aggregate=aggregate)
    events = dialect.table_reference(schema.schema_name, events_table.name)
    event_time = f"o.{identifier(events_table.time_column)}"
    window_bounds = target_window(aggregate).predicate(
        event_time, "a.anchor_time", params, dialect=dialect
    )

    def get_event_column_reference(node):
        """Return ('o."status"', ColumnSchema) for a bound event column."""
        column = bound[ids[node]][1]
        return f"o.{identifier(column.name)}", column

    event_filter = filter_sql(
        parts["target_filter"], get_event_column_reference, params
    )
    foreign_key = f"o.{identifier(link.columns[0])}"
    sql = f"""SELECT DISTINCT {foreign_key} AS entity_id, a.anchor_time
    FROM {anchors_relation} AS a
    JOIN {events} AS o
      ON {window_bounds}"""
    if event_filter:
        sql += f"\n     AND {event_filter}"
    return sql


__all__ = ["get_label_event_aggregates", "get_label_event_candidates_sql"]
