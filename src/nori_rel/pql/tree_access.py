"""Named references and source metadata for the original v0 syntax tree."""

from antlr4 import ParserRuleContext

from nori_rel.pql._generated.PQLGrammarParser import PQLGrammarParser as Grammar
from nori_rel.pql.scope import check_v0_syntax


def leaf_target(
    aggregate: Grammar.AggregationContext,
) -> tuple[Grammar.ColumnContext, ParserRuleContext | None]:
    """Return one aggregate's (target column node, inline filter node or None).

    Example: for COUNT(orders.* WHERE orders.channel='web',0,30,days) the
    column node reads 'orders.*' and the filter node reads
    "orders.channel='web'". Pure tree navigation; no scope or schema check.
    """
    filtered = aggregate.filtered_column()
    if filtered is not None:
        return filtered.column(), filtered.condition()
    return aggregate.column(), None


def query_parts(
    tree: Grammar.ProgContext,
    *,
    aggregate: Grammar.AggregationContext | None = None,
) -> dict[str, ParserRuleContext | None]:
    """Return named references to the existing v0 query nodes.

    Args:
        tree: ProgContext from parse_pql(text). Calls check_v0_syntax first.
        aggregate: One leaf of a composed arithmetic target. When given,
            target_column and target_filter describe that leaf and comparison
            is None, so a caller compiling one aggregate of an expression sees
            the same shape as a single-aggregate query.

    Returns:
        dict[str, ParserRuleContext | None] with exactly these keys:
            target: full target expression (TargetContext).
            target_column: predicted/aggregated column (ColumnContext).
            comparison: target predicate, or None. A ConditionContext for a
                single aggregate; the TargetContext itself when arithmetic
                carries a trailing comparison, since REL_OP and constant
                hang off the target there.
            aggregate: aggregate call (AggregationContext), or None.
            entity_column: scoring key (ColumnContext).
            selection: EntityListContext or FilteredEntityListContext for
                explicit scoring IDs; None for FOR EACH.
            context_filter: historical-example ConditionContext, or None.
            target_filter: ConditionContext selecting rows being aggregated,
                e.g. web orders only; None when absent.
        Values are original node objects, not strings or decoded literals.

    Example:
        >>> from nori_rel.pql import parse_pql
        >>> tree = parse_pql("PREDICT COUNT(orders.*,0,30,days)=0 FOR users.user_id=42")
        >>> parts = query_parts(tree)
        >>> parts["target_column"].getText(), parts["entity_column"].getText()
        ('orders.*', 'users.user_id')
        >>> parts["comparison"].REL_OP().getText()
        '='
        >>> parts["selection"].constant().getText()
        '42'
        >>> parts["target_filter"] is parts["context_filter"] is None
        True

    Raises:
        PQLScopeError: Syntax is outside v0. Names/types remain unresolved.
        PQLLimitError: The target exceeds the arithmetic depth or leaf caps.
        ValueError: aggregate is given but is not a leaf of this target.
    """
    check_v0_syntax(tree)
    target = tree.target()
    comparison = None
    if aggregate is not None and aggregate not in target_aggregates(tree):
        raise ValueError("aggregate is not a leaf of this query's target")
    if aggregate is None:
        comparison = target.condition()
        expression = target.expression()
        if expression is not None and target.REL_OP() is not None:
            comparison = target
        aggregate = target.aggregation()
        if expression is not None:
            aggregate = target_aggregates(tree)[0]
        elif comparison is not None:
            aggregate = comparison.aggregation()
    target_filter = None
    if aggregate is not None:
        target_column, target_filter = leaf_target(aggregate)
    else:
        target_column = (
            comparison.column() if comparison is not None else target.column()
        )

    entity = tree.entity()
    selection = entity.entity_list() or entity.filtered_entity_list()
    context = entity.filtered_column() or entity.filtered_entity_list()
    entity_column = (selection or context or entity).column()
    return {
        "target": target,
        "target_column": target_column,
        "comparison": comparison,
        "aggregate": aggregate,
        "entity_column": entity_column,
        "selection": selection,
        "context_filter": context.condition() if context is not None else None,
        "target_filter": target_filter,
    }


def window_text(node: Grammar.AggregationContext) -> tuple[str, str, str]:
    """Read an aggregate's explicit offsets and unit as three strings.

    Args:
        node: AggregationContext, usually query_parts(tree)["aggregate"].
            Check it is not None before calling.

    Returns:
        (start_offset, end_offset, unit), preserving spelling and leading zeros.
        These are raw strings, not validated durations or timestamps.

    Example:
        >>> from nori_rel.pql import parse_pql
        >>> text = "PREDICT COUNT(orders.*,0,030,DAYS) FOR EACH users.user_id"
        >>> tree = parse_pql(text)
        >>> window_text(query_parts(tree)["aggregate"])
        ('0', '030', 'DAYS')

    Raises:
        ValueError: The aggregate lacks two explicit offsets and a unit.
    """
    offsets = node.offset()
    unit = node.TIME_UNIT()
    if len(offsets) != 2 or unit is None:
        raise ValueError("Expected an explicit two-offset window with a unit")
    return offsets[0].getText(), offsets[1].getText(), unit.getText()


def source_span(node: ParserRuleContext) -> tuple[int, int]:
    """Return the node's (start, end) offsets into the original query string.

    Args:
        node: ANTLR ParserRuleContext with start/stop tokens, e.g. tree.target().

    Returns:
        tuple[int, int]: zero-based Unicode character offsets, end excluded.
        Slice text[start:end] to keep original whitespace/comments. Empty
        grammar productions return (start, start), a zero-width span.

    Example:
        >>> from nori_rel.pql import parse_pql
        >>> text = "PREDICT users.age FOR EACH users.user_id"
        >>> span = source_span(parse_pql(text).target())
        >>> span, text[slice(*span)]
        ((8, 17), 'users.age')
    """
    start = node.start.start
    end = node.stop.stop + 1 if node.stop is not None else start
    return start, max(start, end)


def node_ids(tree: ParserRuleContext) -> dict[ParserRuleContext, int]:
    """Map each rule-node object to a sequential integer ID within this tree.

    Args:
        tree: A ProgContext or any ParserRuleContext subtree from parse_pql().

    Returns:
        dict[ParserRuleContext, int]: {root_node: 0, first_child_node: 1, ...}
        in deterministic preorder. Terminal tokens are excluded. Keys are
        objects, not node names; use this map to key separate binder annotations.

    Example:
        >>> from nori_rel.pql import parse_pql
        >>> tree = parse_pql("PREDICT users.age FOR EACH users.user_id")
        >>> ids = node_ids(tree)
        >>> ids[tree], list(ids.values()) == list(range(len(ids)))
        (0, True)

    IDs are local to this parse, not persistent identifiers across query edits.
    """
    ids = {}
    pending = [tree]
    while pending:
        node = pending.pop()
        ids[node] = len(ids)
        pending.extend(
            child
            for child in reversed(node.children or [])
            if isinstance(child, ParserRuleContext)
        )
    return ids


def target_aggregates(tree: Grammar.ProgContext) -> list[Grammar.AggregationContext]:
    """Return target aggregate nodes in source order, excluding context filters.

    Input: parse_pql("PREDICT SUM(o.x,0,30,days)/COUNT(o.*,0,30,days) FOR u.id=1").
    Output: [SUM node, COUNT node], original objects with their source spans.
    No new public expression model is introduced.
    """
    return [
        n for n in node_ids(tree.target()) if isinstance(n, Grammar.AggregationContext)
    ]
