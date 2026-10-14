"""Prove internal FastDFS aliases are deterministic, reversible and collision-free."""

from datetime import UTC, datetime

import pandas as pd
import pytest

from nori_rel._names import _alias, build_aliases
from nori_rel.connectors.base import Snapshot
from nori_rel.contracts import (
    ColumnSchema,
    DatabaseCreateRequest,
    EnvironmentCredential,
    SchemaGraph,
    TableSchema,
)


def snapshot(tables: dict[str, list[str]]) -> Snapshot:
    """Build the smallest Snapshot build_aliases reads: names and columns only."""
    frames = {
        name: pd.DataFrame({column: pd.Series(dtype="int64") for column in columns})
        for name, columns in tables.items()
    }
    return Snapshot(
        tables=frames,
        primary_keys={},
        foreign_keys=[],
        time_columns={},
        schema=SchemaGraph(
            database_id="test",
            schema_name="public",
            tables=[
                TableSchema(
                    name=name,
                    columns=[
                        ColumnSchema(name=column, data_type="bigint", nullable=True)
                        for column in columns
                    ],
                )
                for name, columns in tables.items()
            ],
            discovered_at=datetime(2026, 1, 1, tzinfo=UTC),
        ),
        as_of=datetime(2026, 1, 1, tzinfo=UTC),
        approximate_bytes=0,
    )


def test_ordinary_names_are_never_renamed():
    aliases = build_aliases(snapshot({"users": ["user_id", "_private", "x9"]}))
    assert aliases.to_alias == {}
    assert aliases.apply("users") == "users"
    assert aliases.restore("d2__users.MEAN(orders.amount)") == (
        "d2__users.MEAN(orders.amount)"
    )


@pytest.mark.parametrize(
    "name",
    ["order.items", "unit-price", "user id", "2024_events", "a$b", "__index__"],
)
def test_names_fastdfs_cannot_carry_are_aliased(name):
    aliases = build_aliases(snapshot({"users": ["user_id"], name: [name]}))
    alias = aliases.apply(name)
    assert alias != name
    assert alias.replace("_", "a").isalnum() and not alias[0].isdigit()
    assert aliases.restore(alias) == (f"`{name}`" if "." in name else name)


def test_dot_and_underscore_names_do_not_collide():
    aliases = build_aliases(snapshot({"a.b": ["value"], "a_b": ["value"]}))
    assert aliases.apply("a.b") != aliases.apply("a_b") == "a_b"
    assert aliases.restore(aliases.apply("a.b")) == "`a.b`"


def test_a_name_shaped_like_an_alias_gets_its_own_alias():
    lookalike = _alias("a.b")
    aliases = build_aliases(snapshot({"a.b": ["value"], lookalike: ["value"]}))
    assert aliases.apply("a.b") == lookalike
    assert aliases.apply(lookalike) not in {lookalike, aliases.apply("a.b")}
    assert aliases.restore(aliases.apply(lookalike)) == lookalike


def test_an_uppercase_alias_lookalike_cannot_shadow_an_alias():
    # DuckDB folds identifiers, so an untouched "A_B_XAB1F84B1E7" would resolve
    # to the same column as the alias of "a.b" and return its values.
    lookalike = _alias("a.b").upper()
    aliases = build_aliases(snapshot({"a.b": ["value"], lookalike: ["value"]}))
    assert aliases.apply(lookalike) != lookalike
    resolved = {aliases.apply("a.b").lower(), aliases.apply(lookalike).lower()}
    assert len(resolved) == 2


def test_an_alias_collision_is_refused_rather_than_dropping_a_table(monkeypatch):
    # A real digest collision is not constructible; force one to prove the
    # guard fires instead of silently losing a table from the feature search.
    monkeypatch.setattr("nori_rel._names._alias", lambda name: "same_x0123456789")
    with pytest.raises(RuntimeError, match="collide"):
        build_aliases(snapshot({"a.b": ["value"], "c.d": ["value"]}))


def test_repeated_column_names_alias_consistently_across_tables():
    aliases = build_aliases(snapshot({"orders": ["a.b"], "refunds": ["a.b"]}))
    restored = aliases.restore(
        f"d2__users.MEAN({aliases.apply('orders')}.{aliases.apply('a.b')})"
    )
    assert restored == "d2__users.MEAN(orders.`a.b`)"


def test_aliases_are_deterministic_across_builds():
    tables = {"order.items": ["profile.score", "amount"], "users": ["user.id"]}
    first = build_aliases(snapshot(tables))
    second = build_aliases(snapshot(dict(reversed(list(tables.items())))))
    assert first.to_alias == second.to_alias


def test_restore_rewrites_a_whole_feature_name():
    aliases = build_aliases(
        snapshot({"order.items": ["profile.score"], "user.accounts": ["user.id"]})
    )
    generated = "d2__{}.MAX({}.{})".format(
        aliases.apply("user.accounts"),
        aliases.apply("order.items"),
        aliases.apply("profile.score"),
    )
    assert aliases.restore(generated) == (
        "d2__`user.accounts`.MAX(`order.items`.`profile.score`)"
    )


@pytest.mark.parametrize(
    "name", ["sales - target - 2026", "a__target__b", "x  index  y", "p..target..q"]
)
def test_an_alias_never_reproduces_a_fastdfs_sentinel(name):
    # FastDFS drops every feature whose name merely CONTAINS __target__, so a
    # slug that rebuilt one would lose that column's aggregates in silence.
    alias = _alias(name)
    assert "__" not in alias.removesuffix(alias[alias.rindex("_x") :])
    assert "__target__" not in alias and "__index__" not in alias


def test_restore_leaves_a_longer_name_that_merely_starts_with_an_alias():
    aliases = build_aliases(snapshot({"a.b": ["value"]}))
    alias = aliases.apply("a.b")
    assert aliases.restore(f"d2__users.MEAN({alias}_extra)") == (
        f"d2__users.MEAN({alias}_extra)"
    )
    assert aliases.restore(f"d2__users.MEAN({alias})") == "d2__users.MEAN(`a.b`)"


def _request(time_columns):
    return DatabaseCreateRequest(
        engine="postgresql",
        credential=EnvironmentCredential(variable="NORI_REL_TEST_DATABASE_URL"),
        schema_name="public",
        time_columns=time_columns,
    )


@pytest.mark.parametrize(
    "time_columns",
    [
        {"order.items": "event_time"},
        {"orders": "event.time"},
        {"order items": "event-time"},
    ],
)
def test_quoted_names_can_be_registered_as_time_columns(time_columns):
    assert _request(time_columns).time_columns == time_columns


@pytest.mark.parametrize("time_columns", [{"": "t"}, {"orders": " "}, {"a\x00b": "t"}])
def test_blank_and_nul_time_column_names_are_still_refused(time_columns):
    with pytest.raises(ValueError, match="time_columns"):
        _request(time_columns)
