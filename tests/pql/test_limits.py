"""Boundary tests ensure resource checks precede generated parsing."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from nori_rel.pql import PQLLimitError, PQLSyntaxError, parse_pql
from nori_rel.pql import parser as parser_module
from nori_rel.pql.limits import MAX_QUERY_BYTES, check_input_size


def query_with_list(size):
    return f"PREDICT users.age FOR users.user_id IN ({','.join(map(str, range(size)))})"


def test_byte_boundary_counts_utf8():
    check_input_size("é" * (MAX_QUERY_BYTES // 2))
    with pytest.raises(PQLLimitError) as error:
        check_input_size("é" * (MAX_QUERY_BYTES // 2) + "x")
    assert error.value.resource == "query_bytes"


@pytest.mark.parametrize("size", [999, 1000])
def test_list_boundary(size):
    parse_pql(query_with_list(size))


@pytest.mark.parametrize("depth", [63, 64])
def test_parenthesis_and_not_boundary(depth):
    parse_pql(f"PREDICT {'(' * depth}users.age>0{')' * depth} FOR EACH users.user_id")
    parse_pql(f"PREDICT {'NOT ' * depth}users.age>0 FOR EACH users.user_id")


@pytest.mark.parametrize(
    ("query", "resource", "limit"),
    [
        (query_with_list(1001), "list_values", 1000),
        (
            "PREDICT "
            + "(" * 65
            + "users.age>0"
            + ")" * 65
            + " FOR EACH users.user_id",
            "nesting",
            64,
        ),
        (
            "PREDICT " + "NOT " * 65 + "users.age>0 FOR EACH users.user_id",
            "nesting",
            64,
        ),
        (
            "PREDICT "
            + "NOT (" * 33
            + "users.age>0"
            + ")" * 33
            + " FOR EACH users.user_id",
            "nesting",
            64,
        ),
        ("x " * 8193, "tokens", 8192),
    ],
    ids=["list-width", "parentheses", "unary-not", "combined-depth", "tokens"],
)
def test_limits_precede_parser(query, resource, limit, monkeypatch, capsys):
    def forbidden_parser(*args):
        pytest.fail("generated parser ran before the limit check")

    monkeypatch.setattr(parser_module, "PQLGrammarParser", forbidden_parser)
    with pytest.raises(PQLLimitError) as error:
        parse_pql(query)
    assert error.value.code == "PQL_LIMIT_EXCEEDED"
    assert (error.value.resource, error.value.limit) == (resource, limit)
    assert capsys.readouterr() == ("", "")


def test_token_boundary_reaches_parser():
    # A short identifier keeps this below the byte cap. Syntax remains invalid.
    with pytest.raises(PQLSyntaxError):
        parse_pql("x " * 8192)


def test_strings_comments_and_separate_lists_do_not_accumulate():
    values = ",".join(map(str, range(1000)))
    parse_pql(
        f"PREDICT users.age FOR users.user_id IN ({values}) "
        f"WHERE users.age IN ({values}) AND users.country='NOT (((,,,)' "
        "# NOT (((,,,\n"
    )


def test_recursion_fallback_has_the_same_error_type(monkeypatch):
    def recurse(self):
        raise RecursionError

    monkeypatch.setattr(parser_module.PQLGrammarParser, "prog", recurse)
    with pytest.raises(PQLLimitError, match="parser_recursion"):
        parse_pql("PREDICT users.age FOR EACH users.user_id")


def test_concurrent_calls_have_independent_limits():
    def check(size):
        try:
            parse_pql(query_with_list(size))
        except PQLLimitError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(check, [1001, 2, 3, 1001])) == [False, True, True, False]
