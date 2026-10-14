"""V0 scope acceptance, rejection and filter-role boundaries."""

import pytest

from nori_rel.pql import PQLScopeError, check_v0_syntax, parse_pql


@pytest.mark.parametrize(
    "aggregate",
    ["COUNT", "SUM", "AVG", "MIN", "MAX", "COUNT_DISTINCT", "FIRST", "LAST"],
)
@pytest.mark.parametrize("comparison", ["", "=0", "!=0", "<5", "<=5", ">5", ">=5"])
def test_numeric_targets_and_event_filters(aggregate, comparison):
    tree = parse_pql(
        f"PREDICT {aggregate}(orders.amount WHERE orders.channel='web' "
        f"AND orders.amount>0,0,30,days){comparison} "
        "FOR users.user_id IN (42,43) "
        "WHERE users.country IN ('US','CA') AND users.age>=18"
    )
    original = tree.toStringTree(recog=tree.parser)
    assert check_v0_syntax(tree) is None
    assert tree.toStringTree(recog=tree.parser) == original


@pytest.mark.parametrize(
    "query",
    [
        "PREDICT users.age FOR EACH users.user_id",
        "PREDICT users.is_fraud FOR users.user_id='abc'",
        "PREDICT users.age>=18 FOR EACH users.user_id WHERE users.country IS NULL",
        "PREDICT COUNT(orders.*,0,30,days)=0 FOR users.user_id=42 "
        "WHERE users.country IS NOT NULL",
        "predict count(orders.*,0,30,days) for users.user_id=42 "
        "where users.country in ('US','CA') and users.age>=18",
        # Values and schema validity remain binder checks, not scope inference.
        "PREDICT COUNT(unknown.*,0,-1,days) FOR EACH missing.id",
        # A delayed interval starts after the anchor; the form is in scope.
        "PREDICT SUM(orders.amount,30,60,days) FOR EACH users.user_id",
        "PREDICT COUNT(orders.*,1,30,days)>0 FOR users.user_id=42",
        "PREDICT COUNT(orders.*,60,30,days) FOR EACH users.user_id",  # value check
        "PREDICT COUNT(orders.*,-0,30,days) FOR EACH users.user_id",  # zero is zero
        f"PREDICT COUNT(orders.*,{'9' * 9},{'9' * 9},days) FOR EACH users.user_id",
    ],
)
def test_simple_queries_and_unbound_names(query):
    check_v0_syntax(parse_pql(query))


@pytest.mark.parametrize(
    "target",
    [
        "LIST_DISTINCT(orders.amount,0,30,days)",
        "COUNT(orders.*)",
        "COUNT(orders.*,0,30)",
        "COUNT(orders.*,-INF,0,days)",
        "COUNT(orders.*,-INF,60,days)",
        "COUNT(orders.*,-30,60,days)",
        "COUNT(orders.*,0,-INF,days)",
        "COUNT(orders.*,0,2,weeks)",
        "COUNT(orders.*,0,7,days) FORECAST 8 TIMEFRAMES",
        "users.country CLASSIFY",
        "users.country RANK TOP 5",
        "users.age TOP 5",
        "users.age>18 AND users.age<65",
        "users.age>18 OR users.age<65",
        "NOT users.is_fraud=true",
        "(users.age>18)",
        "users.country IN ('US','CA')",
        "COUNT(orders.* WHERE orders.amount>0 OR orders.amount<0,0,30,days)",
    ],
)
def test_advanced_targets(target):
    with pytest.raises(PQLScopeError, match="UNSUPPORTED_IN_V0"):
        check_v0_syntax(parse_pql(f"PREDICT {target} FOR EACH users.user_id"))


@pytest.mark.parametrize(
    "suffix",
    [
        "WHERE COUNT(orders.*,-30,0,days)>0",
        "WHERE users.country='US' OR users.age>18",
        "WHERE NOT users.is_fraud=true",
        "WHERE users.country LIKE 'U%'",
        "WHERE users.country IS IN ('US','CA')",
        "WHERE users.is_fraud IS true",
        "WHERE users.country IN (('US','CA'),'GB')",
        "WHERE users.country=('US','CA')",
        "ASSUMING users.age=25",
    ],
)
def test_advanced_context_and_assumptions(suffix):
    with pytest.raises(PQLScopeError):
        check_v0_syntax(parse_pql(f"PREDICT users.age FOR users.user_id=42 {suffix}"))


def test_error_span_and_no_output(capsys):
    query = (
        "PREDICT users.age\nFOR EACH users.user_id WHERE users.country LIKE 'secret'"
    )
    with pytest.raises(PQLScopeError) as error:
        check_v0_syntax(parse_pql(query))
    failure = error.value
    assert failure.code == "UNSUPPORTED_IN_V0"
    assert (failure.line, failure.column) == (2, 30)
    assert (
        query[failure.start_offset : failure.end_offset]
        == "users.country LIKE 'secret'"
    )
    assert "secret" not in str(failure)
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("selector", ["users.user_id>42", "users.user_id IN 42"])
def test_unsupported_scoring_selection(selector):
    with pytest.raises(PQLScopeError):
        check_v0_syntax(parse_pql(f"PREDICT users.age FOR {selector}"))


@pytest.mark.parametrize(
    "window",
    [f"{'1' * 5000},30", f"0,{'1' * 5000}", f"{'1' * 10},30", f"0,{'1' * 10}"],
)
def test_large_offset_is_a_scope_error_without_integer_conversion(window):
    query = f"PREDICT COUNT(orders.*,{window},days) FOR users.user_id=42"
    with pytest.raises(PQLScopeError, match="digits"):
        check_v0_syntax(parse_pql(query))


@pytest.mark.parametrize("start", ["-30", "-INF"])
def test_backward_looking_start_is_named_as_such(start):
    """A negative start would label a row from events already in its features."""
    query = f"PREDICT SUM(orders.amount,{start},60,days) FOR EACH users.user_id"
    with pytest.raises(PQLScopeError, match="backward-looking") as error:
        check_v0_syntax(parse_pql(query))
    assert query[error.value.start_offset : error.value.end_offset] == start
