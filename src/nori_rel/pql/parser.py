"""Parse complete PQL statements without recovery or database/model calls."""

from sys import getrecursionlimit

from antlr4 import CommonTokenStream, InputStream
from antlr4.error.ErrorListener import ErrorListener
from antlr4.error.ErrorStrategy import BailErrorStrategy

from nori_rel.pql._generated.PQLGrammarLexer import PQLGrammarLexer
from nori_rel.pql._generated.PQLGrammarParser import PQLGrammarParser
from nori_rel.pql.limits import PQLLimitError, check_input_size, check_tokens


class PQLSyntaxError(ValueError):
    """Invalid PQL with a one-based line and column, without query contents."""

    def __init__(self, line: int, column: int) -> None:
        """Build an error from one-based integer source coordinates.

        Example: PQLSyntaxError(2, 7) has line=2, column=7 and message
        "Invalid PQL at line 2, column 7". No query text is stored.
        """
        self.line = line
        self.column = column
        super().__init__(f"Invalid PQL at line {line}, column {column}")


class _RaisingListener(ErrorListener):
    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        """Translate ANTLR's zero-based column to PQLSyntaxError(line, column+1).

        ANTLR supplies the recognizer, offending token, integer line/column,
        diagnostic string and exception. Returns nothing; always raises.
        """
        raise PQLSyntaxError(line, column + 1)


class _FailFastStrategy(BailErrorStrategy):
    def reportError(self, recognizer, e):
        # Bypass ANTLR's stdout diagnostic for bare RecognitionException actions.
        """Send a parser/recognition-exception pair to the raising listener.

        Uses e.offendingToken for location. Produces PQLSyntaxError rather
        than ANTLR's printed diagnostic; no successful return value.
        """
        recognizer.notifyErrorListeners("Invalid PQL", e.offendingToken, e)

    def recoverInline(self, recognizer):
        """Reject missing/unexpected tokens through the parser's listener.

        recognizer is the active ANTLR parser. Raises PQLSyntaxError at its
        current token instead of returning an inserted/recovered token.
        """
        recognizer.notifyErrorListeners("Invalid PQL")


def parse_pql(text: str) -> PQLGrammarParser.ProgContext:
    """Parse one complete PQL string into the upstream syntax tree.

    Args:
        text: Query text, e.g. "PREDICT users.age FOR EACH users.user_id".
            The entire string must parse; trailing invalid input is an error.

    Returns:
        PQLGrammarParser.ProgContext: ANTLR rule-node object, not a dict or
        prediction. Its target() and entity() methods return child rule nodes;
        column().getText() reads a qualified column name from either child.
        getText() omits whitespace/comments; source token offsets retain them.

    Example:
        >>> tree = parse_pql("PREDICT users.age FOR EACH users.user_id")
        >>> tree.target().column().getText()
        'users.age'
        >>> tree.entity().column().getText()
        'users.user_id'

    Raises:
        PQLSyntaxError: Malformed input; line and column are one-based.
        PQLLimitError: Byte/token/nesting/list or parser recursion cap exceeded.

    Accepts advanced upstream syntax too. Nori v0 scope, registered-schema
    validation and prediction execution are separate steps.
    """
    check_input_size(text)
    lexer = PQLGrammarLexer(InputStream(text))
    lexer.removeErrorListeners()
    lexer.addErrorListener(_RaisingListener())
    tokens = CommonTokenStream(lexer)
    tokens.fill()  # Validate all characters, including trailing invalid input.
    check_tokens(tokens.tokens)
    parser = PQLGrammarParser(tokens)
    parser.removeErrorListeners()
    parser.addErrorListener(_RaisingListener())
    parser._errHandler = _FailFastStrategy()
    try:
        return parser.prog()  # The upstream entry rule requires EOF.
    except RecursionError:
        raise PQLLimitError("parser_recursion", getrecursionlimit()) from None
