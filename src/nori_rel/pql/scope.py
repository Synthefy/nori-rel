"""Restrict an upstream syntax tree to the agreed first-delivery query forms."""

from antlr4 import ParserRuleContext

from nori_rel.pql._generated.PQLGrammarParser import PQLGrammarParser as Grammar
from nori_rel.pql.limits import MAX_NESTING, MAX_TARGET_AGGREGATES, PQLLimitError

_COMPARISONS = {"=", "!=", "<", "<=", ">", ">="}
_FILTER_OPERATORS = _COMPARISONS | {"IN", "IS", "IS NOT"}
_AGGREGATES = {"COUNT", "SUM", "AVG", "MIN", "MAX", "COUNT_DISTINCT", "FIRST", "LAST"}
# Offsets are checked as text so an absurd literal never reaches int().
_MAX_OFFSET_DIGITS = 9


class PQLScopeError(ValueError):
    """A recognized query form outside v0, with its original source location."""

    code = "UNSUPPORTED_IN_V0"

    def __init__(self, node: ParserRuleContext, feature: str) -> None:
        """Create a located error from a rule node and a short feature label.

        Input: node from parse_pql(), feature="aggregate filter".
        Output error attributes: code, line, column, start_offset, end_offset.
        Coordinates are one-based; the character span is zero-based/end-exclusive.
        """
        self.line = node.start.line
        self.column = node.start.column + 1
        self.start_offset = node.start.start
        self.end_offset = node.stop.stop + 1
        super().__init__(
            f"{self.code}: {feature} at line {self.line}, column {self.column}"
        )


def _require(allowed: bool, node: ParserRuleContext, feature: str) -> None:
    """Return None if allowed is True; otherwise raise a located scope error.

    node is the offending ANTLR rule node; feature is a label such as "modifier".
    """
    if not allowed:
        raise PQLScopeError(node, feature)


def _comparison(node, operators: set[str]) -> None:
    """Validate a comparison node against a set of uppercase operator strings.

    Input node exposes REL_OP() and constant(), e.g. country IN ('US','CA');
    operators might be {"=", "IN"}. Returns None or raises PQLScopeError for
    an unsupported operator/list/null form. Does not decode literal values.
    """
    operator = node.REL_OP().getText().upper()
    _require(operator in operators, node, "operator")
    value = node.constant()
    values = value.array()
    if operator == "IN":
        _require(values is not None, value, "IN requires a list")
        _require(
            all(item.array() is None for item in values.constant()),
            value,
            "nested lists",
        )
    else:
        _require(values is None, value, "list outside IN")
    if operator in {"IS", "IS NOT"}:
        _require(value.NULL() is not None, value, "IS requires NULL")


def _condition(node: Grammar.ConditionContext, *, is_filter: bool) -> None:
    """Check a ConditionContext subtree; return None or raise PQLScopeError.

    is_filter=True allows column conditions joined by AND, e.g. age>=18 AND
    country='US'. False checks a single target comparison, e.g. COUNT(...)>0.
    """
    pending = [node]
    while pending:
        current = pending.pop()
        if is_filter and current.AND() is not None:
            pending.extend(reversed(current.condition()))
            continue
        _require(current.REL_OP() is not None, current, "combined/grouped condition")
        aggregate = current.aggregation()
        _require(not is_filter or aggregate is None, current, "aggregate filter")
        _comparison(current, _FILTER_OPERATORS if is_filter else _COMPARISONS)
        if aggregate is not None:
            _aggregate(aggregate)


def _aggregate(node: Grammar.AggregationContext) -> None:
    """Check an AggregationContext, e.g. COUNT(orders.*,0,30,days).

    Requires a supported operation and an explicit (start, end, days) or
    (start, end, months) window
    whose start is a non-negative integer, so the label interval begins at or
    after the anchor; also checks any inline filter. Returns None or raises
    PQLScopeError. Whether end exceeds start, and database column types, are
    not established by this check.
    """
    _require(node.AGGR().getText().upper() in _AGGREGATES, node, "aggregate function")
    offsets = node.offset()
    unit = node.TIME_UNIT()
    _require(
        len(offsets) == 2
        and unit is not None
        and unit.getText().lower() in ("days", "months"),
        node,
        "window must use (start, end, days) or (start, end, months)",
    )
    start, end = offsets
    # -INF and negative starts are Kumo's lookback form: the interval would
    # straddle the anchor and label a row from events already in its features.
    _require(
        start.NEG_INFTY() is None
        and (start.MINUS() is None or not start.INT().getText().strip("0")),
        start,
        "backward-looking target interval",
    )
    _require(end.NEG_INFTY() is None, end, "unbounded target interval")
    for offset in offsets:
        _require(
            len(offset.INT().getText()) <= _MAX_OFFSET_DIGITS,
            offset,
            f"window offset longer than {_MAX_OFFSET_DIGITS} digits",
        )
    filtered = node.filtered_column()
    if filtered is not None:
        _condition(filtered.condition(), is_filter=True)


def check_v0_syntax(tree: Grammar.ProgContext) -> None:
    """Check supported v0 syntax without changing the tree.

    Args:
        tree: ProgContext returned by parse_pql(text), containing target(),
            entity(), problem_spec() and whatif() rule nodes.

    Returns:
        None on success. The caller keeps the same tree; no replacement model
        or normalized request is created.

    Example:
        >>> from nori_rel.pql import parse_pql
        >>> tree = parse_pql("PREDICT COUNT(orders.*,0,30,days)=0 FOR users.user_id=42")
        >>> check_v0_syntax(tree) is None
        True

    Raises:
        PQLScopeError: Recognized but unsupported syntax, e.g. OR in a filter.
            Attributes: code="UNSUPPORTED_IN_V0", one-based line/column,
            zero-based start_offset/end_offset (end excluded).
        PQLLimitError: An arithmetic target nests deeper than MAX_NESTING or
            holds more than MAX_TARGET_AGGREGATES aggregate leaves.

    This checks query forms, not schema names, relationship paths, operand
    types or whether an interval's end exceeds its start. Success does not
    authorize execution.
    """
    _require(tree.problem_spec().getChildCount() == 0, tree.problem_spec(), "modifier")
    _require(tree.whatif().getChildCount() == 0, tree.whatif(), "ASSUMING")
    target = tree.target()
    if target.expression() is not None:
        _expression(target.expression())
        if target.REL_OP() is not None:
            _comparison(target, _COMPARISONS)
    elif target.condition() is not None:
        _condition(target.condition(), is_filter=False)
    elif target.aggregation() is not None:
        _aggregate(target.aggregation())

    entity = tree.entity()
    selection = entity.entity_list() or entity.filtered_entity_list()
    if selection is not None:
        _comparison(selection, {"=", "IN"})
    filtered = entity.filtered_column() or entity.filtered_entity_list()
    if filtered is not None:
        _condition(filtered.condition(), is_filter=True)


def _expression(node) -> None:
    """Validate numeric expression leaves; preserve ANTLR precedence and spans.

    Input: ExpressionContext for SUM(...)/COUNT(...). Returns None; schema
    types and division behavior belong to SQL compilation, not the parser.
    Direct columns in composed expressions await the direct-target contract.
    """
    pending = [(node, 0)]
    aggregates = 0
    while pending:
        current, depth = pending.pop()
        if depth > MAX_NESTING:
            raise PQLLimitError("expression_depth", MAX_NESTING)
        if current.aggregation() is not None:
            _aggregate(current.aggregation())
            aggregates += 1
        elif current.constant() is not None:
            constant = current.constant()
            _require(
                constant.INT() is not None or constant.DECIMAL() is not None,
                constant,
                "arithmetic requires numeric literals",
            )
        elif current.column() is not None:
            _require(False, current, "direct-column arithmetic is not supported")
        pending.extend((child, depth + 1) for child in current.expression())
    _require(aggregates > 0, node, "arithmetic requires an aggregate")
    if aggregates > MAX_TARGET_AGGREGATES:
        raise PQLLimitError("target_aggregates", MAX_TARGET_AGGREGATES)
