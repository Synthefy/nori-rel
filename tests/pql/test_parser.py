"""Contract tests for the first, syntax-only PQL delivery."""

import pytest

from nori_rel.pql import PQLSyntaxError, parse_pql


@pytest.mark.parametrize(
    "query",
    [
        "PREDICT users.age FOR EACH users.user_id",
        "PREDICT users.is_fraud FOR users.user_id=42",
        "PREDICT COUNT(orders.*,0,30,days) FOR users.user_id=42",
        "PREDICT SUM(orders.amount,0,30,days) FOR users.user_id IN (42,43)",
        "PREDICT COUNT(orders.*,0,30,days)=0 FOR users.user_id=42 "
        "WHERE users.country IN ('US','CA') AND users.age>=18",
        "PREDICT users.age FOR EACH users.user_id WHERE users.country IS NOT NULL",
        "PREDICT COUNT(orders.* WHERE orders.channel='web',0,30,days) "
        "FOR users.user_id=42",
        # The v0 allowlist is a separate change; this function parses syntax only.
        "PREDICT SUM(orders.amount,0,7,days) FORECAST 8 TIMEFRAMES "
        "FOR EACH users.user_id",
    ],
)
def test_complete_query(query):
    tree = parse_pql(query)
    assert tree.target() is not None
    assert tree.entity() is not None
    assert tree.EOF() is not None


def test_preserves_target_and_scoring_selection():
    tree = parse_pql("PREDICT users.Age FOR users.user_id=42")
    assert tree.target().column().getText() == "users.Age"
    selection = tree.entity().entity_list()
    assert selection.column().getText() == "users.user_id"
    assert selection.constant().getText() == "42"


@pytest.mark.parametrize(
    "query",
    [
        "",
        "PREDICT users.age FOR",  # Missing token must not be inserted.
        "PREDICT users.age age FOR EACH users.user_id",  # No token deletion.
        "PREDICT users.age FOR EACH users.user_id;",
        "PREDICT users.age FOR EACH users.user_id @",  # Lexer error.
        "PREDICT users.age FOR EACH users.user_id PREDICT users.age",
        "PREDICT users.age FOR users.user_id IN (42)",  # Upstream action raises.
    ],
)
def test_rejects_invalid_input_without_output(query, capsys):
    with pytest.raises(PQLSyntaxError):
        parse_pql(query)
    assert capsys.readouterr() == ("", "")


def test_error_location_and_redaction():
    with pytest.raises(PQLSyntaxError) as error:
        parse_pql("PREDICT users.age\nFOR EACH users.user_id @secret")
    assert (error.value.line, error.value.column) == (2, 24)
    assert "secret" not in str(error.value)


def test_limits_input_bytes():
    with pytest.raises(ValueError, match="query_bytes"):
        parse_pql("é" * 32769)


@pytest.mark.parametrize(
    "target",
    [
        "SUM(orders.amount,0,30,days)+",
        "SUM(orders.amount,0,30,days)**2",
        "SUM(orders.amount,0,30,days) 2",
        "(SUM(orders.amount,0,30,days)",
    ],
)
def test_malformed_arithmetic_has_no_recovery(target):
    with pytest.raises(PQLSyntaxError):
        parse_pql(f"PREDICT {target} FOR users.user_id=42")


def test_deep_arithmetic_rejected_before_sql_compilation():
    from nori_rel.pql import PQLLimitError, check_v0_syntax

    target = "COUNT(orders.*,0,30,days)" + "+1" * 100
    with pytest.raises(PQLLimitError):
        check_v0_syntax(parse_pql(f"PREDICT {target} FOR users.user_id=42"))


def _balanced(aggregate: str, count: int) -> str:
    """Nest `count` aggregates as a balanced tree, keeping the depth low."""
    terms = [aggregate] * count
    while len(terms) > 1:
        terms = [
            f"({terms[i]}+{terms[i + 1]})" if i + 1 < len(terms) else terms[i]
            for i in range(0, len(terms), 2)
        ]
    return terms[0]


@pytest.mark.parametrize("shape", ["flat", "balanced"])
@pytest.mark.parametrize("over", [0, 1])
def test_target_aggregate_count_is_capped_in_both_expression_shapes(shape, over):
    """Each leaf costs one CTE and one scan, so the count needs its own cap.

    MAX_NESTING alone does not provide one: a balanced tree keeps the depth
    low, so before this cap 630 leaves were accepted from 17 KiB of PQL.
    """
    from nori_rel.pql import PQLLimitError, check_v0_syntax
    from nori_rel.pql.limits import MAX_TARGET_AGGREGATES

    count, accepted = MAX_TARGET_AGGREGATES + over, over == 0
    aggregate = "COUNT(orders.*,0,30,days)"
    target = (
        "+".join([aggregate] * count)
        if shape == "flat"
        else _balanced(aggregate, count)
    )
    query = f"PREDICT {target} FOR users.user_id=42"
    if accepted:
        check_v0_syntax(parse_pql(query))
    else:
        with pytest.raises(PQLLimitError, match="target_aggregates"):
            check_v0_syntax(parse_pql(query))


@pytest.mark.parametrize("signs", ["-" * 200, "+-" * 100])
def test_long_unary_sign_chains_are_still_rejected_before_compilation(signs):
    """Sign chains are bounded by the expression depth walk, not a token counter."""
    from nori_rel.pql import PQLLimitError, check_v0_syntax

    with pytest.raises(PQLLimitError, match="expression_depth"):
        check_v0_syntax(
            parse_pql(f"PREDICT {signs}COUNT(orders.*,0,30,days) FOR users.user_id=42")
        )
