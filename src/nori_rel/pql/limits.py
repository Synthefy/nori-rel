"""Bound PQL work before invoking the recursive generated parser."""

from antlr4 import Token

from nori_rel.pql._generated.PQLGrammarLexer import PQLGrammarLexer as Lexer

MAX_QUERY_BYTES = 64 * 1024
MAX_TOKENS = 8192
MAX_NESTING = 64
MAX_LIST_VALUES = 1000
# Each aggregate leaf becomes one CTE with its own scan of the event table,
# and MAX_NESTING does not bound leaf count once an expression is balanced
# with parentheses.
MAX_TARGET_AGGREGATES = 8


class PQLLimitError(ValueError):
    """A parser resource limit was exceeded; messages contain no query text."""

    code = "PQL_LIMIT_EXCEEDED"

    def __init__(self, resource: str, limit: int) -> None:
        """Create a limit error from a resource name and maximum allowed integer.

        Example: PQLLimitError("tokens", 8192) exposes resource="tokens",
        limit=8192 and code="PQL_LIMIT_EXCEEDED". Stores no query text.
        """
        self.resource = resource
        self.limit = limit
        super().__init__(f"{self.code}: {resource} exceeds {limit}")


def check_input_size(text: str) -> None:
    """Check the UTF-8 byte length of a query string before lexing.

    Input: str, e.g. "PREDICT users.age FOR EACH users.user_id".
    Output: None when its encoded size is at most MAX_QUERY_BYTES (65,536).
    Raises PQLLimitError(resource="query_bytes", limit=65536) above that cap.
    Characters can occupy multiple bytes; this is not a character-count limit.
    """
    if len(text.encode("utf-8")) > MAX_QUERY_BYTES:
        raise PQLLimitError("query_bytes", MAX_QUERY_BYTES)


def check_tokens(tokens: list[Token]) -> None:
    """Check the resource limits of an already lexed ANTLR token list.

    Args:
        tokens: list[Token] from CommonTokenStream.tokens after fill(). Each
            token exposes .type (integer lexer kind) and .text (source spelling).
            EOF may be present and does not count toward MAX_TOKENS.

    Returns:
        None within the caps: 8,192 tokens, 64 combined parentheses/NOT depth,
        and 1,000 values in each parenthesized sequence (including call args).

    Example:
        >>> from antlr4 import CommonTokenStream, InputStream
        >>> text = "PREDICT users.age FOR EACH users.user_id"
        >>> stream = CommonTokenStream(Lexer(InputStream(text)))
        >>> stream.fill()
        >>> check_tokens(stream.tokens) is None
        True

    Raises:
        PQLLimitError: resource is "tokens", "nesting" or "list_values";
            limit is the corresponding maximum allowed integer.

    Commas count only at their own parenthesis level; literals/comments cannot
    inflate nesting or list width. Syntax validation remains the parser's job.
    """
    count = 0
    depth = 0
    unary = [0]
    commas = []
    boundaries = {
        Lexer.AND,
        Lexer.OR,
        Lexer.FOR,
        Lexer.FOR_EACH,
        Lexer.WHERE,
        Lexer.ASSUMING,
    }
    for token in tokens:
        if token.type == Token.EOF:
            continue
        count += 1
        if count > MAX_TOKENS:
            raise PQLLimitError("tokens", MAX_TOKENS)
        if token.text == "(":
            unary.append(0)
            commas.append(0)
            depth += 1
        elif token.text == ")" and commas:
            depth -= 1 + unary.pop()
            commas.pop()
        elif token.type == Lexer.NOT:
            unary[-1] += 1
            depth += 1
        if token.type in boundaries or token.text == ",":
            depth -= unary[-1]
            unary[-1] = 0
        if token.text == "," and commas:
            commas[-1] += 1
            if commas[-1] + 1 > MAX_LIST_VALUES:
                raise PQLLimitError("list_values", MAX_LIST_VALUES)
        if depth > MAX_NESTING:
            raise PQLLimitError("nesting", MAX_NESTING)
