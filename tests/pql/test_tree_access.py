"""Named roles retain identity, filter scope and original source locations."""

import pytest

from nori_rel.pql import PQLScopeError, parse_pql
from nori_rel.pql.tree_access import node_ids, query_parts, source_span, window_text


@pytest.mark.parametrize(
    "entity",
    [
        "FOR EACH users.user_id",
        "FOR users.user_id=42",
        "FOR users.user_id IN (42,43)",
    ],
)
@pytest.mark.parametrize("with_context", [False, True])
def test_scoring_selection_is_separate_from_context(entity, with_context):
    query = f"PREDICT users.age {entity}"
    if with_context:
        query += " WHERE users.country IN ('US','CA')"
    tree = parse_pql(query)
    parts = query_parts(tree)
    assert parts["target"] is tree.target()
    assert parts["target_column"].getText() == "users.age"
    assert parts["entity_column"].getText() == "users.user_id"
    selection = parts["selection"]
    if "EACH" in entity:
        assert selection is None
    else:
        assert selection.constant().getText() in {"42", "(42,43)"}
    context = parts["context_filter"]
    assert (context is not None) == with_context
    if context is not None:
        assert context.column().getText() == "users.country"
    assert parts["aggregate"] is parts["target_filter"] is parts["comparison"] is None


def test_target_filter_and_context_roles_with_original_spelling():
    query = (
        "PREDICT sum(`Order.Events`.`Total.Amount` WHERE "
        "`Order.Events`.channel='café' AND "
        "`Order.Events`.amount>0, 0, 030, DAYS) > 500\n"
        "FOR users.user_id=42 WHERE users.country='CA'"
    )
    tree = parse_pql(query)
    original = tree.toStringTree(recog=tree.parser)
    parts = query_parts(tree)
    assert parts["comparison"] is tree.target().condition()
    assert parts["comparison"].constant().getText() == "500"
    aggregate = parts["aggregate"]
    assert aggregate is parts["comparison"].aggregation()
    assert aggregate.AGGR().getText() == "sum"
    assert parts["target_column"].getText() == "`Order.Events`.`Total.Amount`"
    assert window_text(aggregate) == ("0", "030", "DAYS")
    assert parts["target_filter"] is aggregate.filtered_column().condition()
    assert parts["context_filter"].column().getText() == "users.country"
    assert query[slice(*source_span(parts["context_filter"]))] == "users.country='CA'"
    assert tree.toStringTree(recog=tree.parser) == original


@pytest.mark.parametrize("expression", ["users.age>18", "COUNT(orders.*,0,30,days)"])
def test_direct_comparison_and_uncompared_aggregate(expression):
    parts = query_parts(parse_pql(f"PREDICT {expression} FOR EACH users.user_id"))
    assert (parts["comparison"] is not None) == (expression == "users.age>18")
    assert (parts["aggregate"] is not None) == expression.startswith("COUNT")


def test_spans_keep_comments_and_empty_productions_are_zero_width():
    query = "PREDICT users.age # keep this comment\n > 18 FOR EACH users.user_id"
    tree = parse_pql(query)
    assert (
        query[slice(*source_span(tree.target()))]
        == "users.age # keep this comment\n > 18"
    )
    for empty in (tree.problem_spec(), tree.whatif()):
        start, end = source_span(empty)
        assert 0 <= start == end <= len(query)


def test_ids_are_deterministic_and_refer_to_original_nodes():
    query = "PREDICT COUNT(orders.*,0,30,days)=0 FOR users.user_id=42"
    first, second = parse_pql(query), parse_pql(query)
    ids = node_ids(first)
    assert list(ids.values()) == list(range(len(ids)))
    assert [(n.getRuleIndex(), i) for n, i in ids.items()] == [
        (n.getRuleIndex(), i) for n, i in node_ids(second).items()
    ]
    assert ids[first] == 0
    for node in query_parts(first).values():
        if node is not None:
            assert node in ids


def test_parts_reject_advanced_syntax_instead_of_dropping_it():
    tree = parse_pql("PREDICT users.age>18 OR users.age<10 FOR EACH users.user_id")
    with pytest.raises(PQLScopeError):
        query_parts(tree)


def test_window_reader_requires_explicit_window():
    tree = parse_pql("PREDICT COUNT(orders.*) FOR EACH users.user_id")
    with pytest.raises(ValueError, match="explicit"):
        window_text(tree.target().aggregation())
