"""Internal aliases for database names FastDFS cannot carry unchanged."""

from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import blake2b

from nori_rel.feature.dfs import RelationalData

# A name FastDFS can carry unchanged: a letter or underscore, then letters,
# digits and underscores. Used with fullmatch, so "orders" and "_x9" pass while
# "order.items", "unit-price", "user id" and "2024_events" do not.
_SAFE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# Every character a safe name may not contain. Substituted with "_" to build
# the readable half of an alias: "order.items" becomes "order_items".
_UNSAFE_CHARACTER = re.compile(r"[^A-Za-z0-9_]")

# Two or more underscores in a row, collapsed to one so that substituting the
# characters above can never rebuild a FastDFS sentinel: "sales - target - 2026"
# would otherwise become "sales___target___2026", which contains "__target__".
_UNDERSCORE_RUN = re.compile(r"_{2,}")

# How much of the original name survives into the alias. Readability only; the
# digest below is what makes an alias unique, so truncation here is harmless.
_SLUG_LIMIT = 40

# blake2b digest length in BYTES. An alias spells it as twice as many hex
# characters, so 5 here means a 10-character suffix: "order_items_xbdf01312cd".
_DIGEST_BYTES = 5

# A name that already looks like one of our aliases: anything, then "_x", then
# exactly _DIGEST_BYTES * 2 hex characters. Used with fullmatch, so
# "a_b_xdeadbeef01" matches while "orders" and "a_b_xZZZZ" do not. Such names
# are aliased too, so a real table named like an alias can never collide with
# the alias generated for some other name.
#
# Matched case-insensitively because DuckDB folds identifiers: "A_B_XAB1F84B1E7"
# left alone would collide there with "a_b_xab1f84b1e7", the alias of "a.b", and
# one column would silently return the other's values.
_ALIAS_SHAPE = re.compile(
    rf"[A-Za-z0-9_]*_x[0-9a-f]{{{_DIGEST_BYTES * 2}}}", re.IGNORECASE
)

# Column names FastDFS and this package reserve for their own bookkeeping.
_RESERVED = frozenset(
    {
        "__nori_pql_row_id",
        "__nori_rel_cutoff_time",
        "__target_index__",
        "__index__",
        "__time_for_cutoff__",
        "__cutoff_time__",
    }
)


def _needs_alias(name: str) -> bool:
    """Report whether FastDFS would misread this name, or we would.

    A real name already shaped like one of our aliases is aliased too, so it
    can never collide with the alias generated for some other name.
    """
    return (
        not _SAFE.fullmatch(name)
        or name in _RESERVED
        or _ALIAS_SHAPE.fullmatch(name) is not None
    )


def _alias(name: str) -> str:
    """Build one deterministic alias matching [A-Za-z_][A-Za-z0-9_]*.

    Example: _alias("order.items") is "order_items_x" plus a ten-character
    digest of the original name. The slug keeps the alias readable in
    generated SQL; the digest keeps distinct originals distinct, including
    "a.b" against a real table already named "a_b".

    Runs of unsafe characters collapse to one underscore so a slug can never
    reproduce a FastDFS sentinel such as __target__, whose features FastDFS
    drops on a substring match. Distinctness is unaffected: the digest hashes
    the original name, not the slug.
    """
    slug = _UNDERSCORE_RUN.sub("_", _UNSAFE_CHARACTER.sub("_", name))[:_SLUG_LIMIT]
    if not slug[:1].isalpha() and not slug.startswith("_"):
        slug = "_" + slug
    digest = blake2b(name.encode("utf-8"), digest_size=_DIGEST_BYTES).hexdigest()
    return f"{slug}_x{digest}"


def _display(name: str) -> str:
    """Quote a restored name only when it contains FastDFS's separator."""
    return f"`{name}`" if "." in name else name


@dataclass(frozen=True)
class NameAliases:
    """Two-way map between database names and FastDFS-safe aliases.

    Only names FastDFS cannot carry appear in the map. Everything else is
    absent and passes through untouched, which is why an ordinary schema
    produces an empty instance and unchanged feature names. Build one with
    build_aliases(); the fields are shown here to make the shape concrete.

    Example:
        A schema holding a table "order.items" with a column "profile.score",
        alongside an ordinary table "users":

        >>> import re
        >>> aliases = NameAliases(
        ...     to_alias={
        ...         "order.items": "order_items_xbdf01312cd",
        ...         "profile.score": "profile_score_xc9a20690f3",
        ...     },
        ...     to_original={
        ...         "order_items_xbdf01312cd": "order.items",
        ...         "profile_score_xc9a20690f3": "profile.score",
        ...     },
        ...     pattern=re.compile(
        ...         "(?:profile_score_xc9a20690f3|order_items_xbdf01312cd)"
        ...         "(?![A-Za-z0-9_])"
        ...     ),
        ... )
        >>> aliases.apply("order.items")
        'order_items_xbdf01312cd'
        >>> aliases.apply("users")
        'users'
        >>> aliases.restore(
        ...     "d2__users.MEAN(order_items_xbdf01312cd.profile_score_xc9a20690f3)"
        ... )
        'd2__users.MEAN(`order.items`.`profile.score`)'
    """

    to_alias: dict[str, str]
    to_original: dict[str, str]
    pattern: re.Pattern[str] | None

    def apply(self, name: str) -> str:
        """Return this name's alias, or the name itself when none is needed."""
        return self.to_alias.get(name, name)

    def restore(self, text: str) -> str:
        """Rewrite every alias inside one generated feature name.

        Example: "d2__order_items_xbdf01312cd.COUNT(...)" becomes
        "d2__`order.items`.COUNT(...)". Names that were never aliased, and
        FastDFS's own primitive and depth prefixes, are returned untouched.
        """
        if self.pattern is None:
            return text
        return self.pattern.sub(
            lambda match: _display(self.to_original[match.group(0)]), text
        )


def build_aliases(snapshot: RelationalData) -> NameAliases:
    """Map the names in one snapshot that FastDFS cannot carry unchanged.

    Args:
        snapshot: Registered tables whose names and columns cross the FastDFS
            boundary. Only the table keys and each frame's column labels are
            read; no row is touched.

    Returns:
        NameAliases: Empty when every name already matches
        [A-Za-z_][A-Za-z0-9_]* and avoids the reserved columns, which is the
        ordinary case and leaves generated feature names byte-identical.

    Raises:
        RuntimeError: Two names resolve to one alias, refused rather than
            silently dropping a table from the feature search. The reserved
            columns cannot be reached, since every alias ends in a digest, but
            the guard is kept because that is a property of _alias, not of
            this function.
    """
    names = set(snapshot.tables)
    for frame in snapshot.tables.values():
        names.update(str(column) for column in frame.columns)
    to_alias = {name: _alias(name) for name in sorted(names) if _needs_alias(name)}
    resolved = {to_alias.get(name, name) for name in names}
    if len(resolved) != len(names) or resolved & _RESERVED:
        raise RuntimeError("internal feature aliases collide for this schema")
    pattern = None
    if to_alias:
        # Longest-first resolves one alias containing another. The trailing
        # lookahead stops an alias matching the prefix of an ordinary name that
        # merely starts with the same characters. There is no leading boundary
        # because FastDFS's own "d2__" depth prefix ends in an underscore.
        ordered = sorted(to_alias.values(), key=len, reverse=True)
        pattern = re.compile(
            "(?:"
            + "|".join(re.escape(alias) for alias in ordered)
            + ")(?![A-Za-z0-9_])"
        )
    return NameAliases(
        to_alias=to_alias,
        to_original={alias: name for name, alias in to_alias.items()},
        pattern=pattern,
    )
