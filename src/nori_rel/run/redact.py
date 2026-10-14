"""Strip credentials and row values from error text before it leaves a run.

Table, column and constraint names, generated feature names and file paths are
metadata and are kept unless the caller asks to hide them. Values are not:
single-quoted literals (``E'…'`` included), double-quoted values after a colon,
``string`` or ``value`` (PostgreSQL's ``invalid input syntax for type …: "…"``
and ``value "…" is out of range``, Python's and pandas' reprs), PostgreSQL's
``Failing row contains (…)`` and ``Key (col)=(value)`` details, and the SQL
quoted after ``LINE n:`` are replaced.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Collection

MAX_LENGTH = 300
#: Error text is cut to this many characters before the rules run, so a huge
#: message cannot stall the run's thread. The result keeps at most MAX_LENGTH.
MAX_SCANNED_CHARS = 4_000

#: Names whose value is a credential: ``password=…``, ``"api_key": "…"``,
#: and prefixed names such as ``PGPASSWORD`` or ``OPENAI_API_KEY``.
_CREDENTIAL_NAME_PATTERN = r"api[_-]?key|token|secret|password|passwd|pwd"


def _get_opaque_token_replacement(match: re.Match[str]) -> str:
    """Return ``***`` for an opaque token, or the matched text to keep it.

    Identifiers (with an ``_``, such as generated feature names) and paths
    (more than two ``/``) are kept. Base64 secrets such as AWS secret keys
    have neither.
    """
    token = match.group(0)
    is_identifier = "_" in token
    is_path = token.count("/") > 2
    return token if is_identifier or is_path else "***"


_RULES: tuple[tuple[re.Pattern[str], str | Callable[[re.Match[str]], str]], ...] = (
    # Credentials embedded in a URL: scheme://user:password@host. The scheme
    # must start a word; otherwise long alphanumeric text is rescanned from
    # every position.
    (
        re.compile(r"(?<![A-Za-z0-9+.-])([A-Za-z][A-Za-z0-9+.-]*://)[^\s/@]+@"),
        r"\1***@",
    ),
    (re.compile(r"(?i)\bbearer\s+\S+"), "Bearer ***"),
    (
        re.compile(
            rf"(?i)\b(\w*(?:{_CREDENTIAL_NAME_PATTERN}))\b"
            r"([\"']?\s*[=:]\s*[\"']?)[^\s\"',;}]+"
        ),
        r"\1\2***",
    ),
    # Provider key shapes: Anthropic, OpenAI, Stripe and Hugging Face.
    (re.compile(r"\b(?:sk-ant-|sk-proj-|sk-)[A-Za-z0-9_-]{20,}"), "***"),
    (re.compile(r"\b(?:sk|pk|rk)_(?:live|test)_[A-Za-z0-9]{16,}"), "***"),
    (re.compile(r"\bhf_[A-Za-z0-9]{30,}"), "***"),
    (re.compile(r"\b[A-Za-z0-9+/_-]{40,}={0,2}"), _get_opaque_token_replacement),
    # Row values. The SQL that PostgreSQL and DuckDB quote after "LINE n:",
    # with its caret, goes first: it can hold any of the values below.
    (re.compile(r"(?s)\bLINE \d+:.*"), "(SQL omitted)"),
    (
        re.compile(r"(?i)\bFailing row contains \((?:[^()]|\([^()]*\))*\)"),
        "Failing row contains (?)",
    ),
    (re.compile(r"=\((?:[^()]|\([^()]*\))*\)"), "=(?)"),
    # Double-quoted values, before single-quoted literals, so "O'Brien" goes
    # whole: `type integer: "x"`, `string "x1"`, `value "9999999999" is out
    # of range`.
    (re.compile(r'(?i)(: |\bstring |\bvalue )"[^"]*"'), r'\1"?"'),
    # SQL literals, with an optional one-letter prefix such as E'…'.
    (re.compile(r"(?<![A-Za-z])[A-Za-z]?'(?:[^']|'')*'"), "'?'"),
)


def redact(
    text: str, *, hide_names: Collection[str] = (), max_length: int = MAX_LENGTH
) -> str:
    """Return ``text`` without credentials or row values, at most ``max_length``.

    ``hide_names`` lists table and column names to replace with ``<name>``, for
    callers that treat their schema as sensitive.
    """
    text = text[:MAX_SCANNED_CHARS]
    for pattern, replacement in _RULES:
        text = pattern.sub(replacement, text)
    for name in sorted(hide_names, key=len, reverse=True):
        text = re.sub(rf"(?<!\w){re.escape(name)}(?!\w)", "<name>", text)
    text = " ".join(text.split())
    if len(text) > max_length:
        text = text[: max_length - 1] + "…"
    return text


__all__ = ["MAX_LENGTH", "MAX_SCANNED_CHARS", "redact"]
