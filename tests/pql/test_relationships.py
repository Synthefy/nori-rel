"""V0 cannot silently follow multiple links or widen the filter table scope."""

from datetime import UTC, datetime

import pytest

from nori_rel.contracts import ColumnSchema, ForeignKeySchema, SchemaGraph, TableSchema
from nori_rel.pql import parse_pql
from nori_rel.pql.binding import PQLBindingError
from nori_rel.pql.relationships import bind_relationship

QUERY = "PREDICT COUNT(orders.*,0,30,days)=0 FOR users.user_id=42"


@pytest.fixture
def schema():
    def column(name):
        return ColumnSchema(name=name, data_type="INTEGER", nullable=False)

    return SchemaGraph(
        database_id="db",
        schema_name="public",
        discovered_at=datetime(2026, 1, 1, tzinfo=UTC),
        tables=[
            TableSchema(
                name="users",
                primary_key=["user_id"],
                columns=[column("user_id"), column("age"), column("country")],
            ),
            TableSchema(
                name="orders",
                columns=[
                    column("user_id"),
                    column("amount"),
                    column("channel"),
                    ColumnSchema(
                        name="event_time", data_type="TIMESTAMPTZ", nullable=False
                    ),
                ],
                time_column="event_time",
                foreign_keys=[
                    ForeignKeySchema(
                        columns=["user_id"],
                        referred_table="users",
                        referred_columns=["user_id"],
                    )
                ],
            ),
        ],
    )


def test_returns_registered_link_without_mutating_metadata(schema):
    original = schema.model_dump()
    query = "PREDICT SUM(orders.amount WHERE orders.channel='web',0,30,days) "
    query += "FOR users.user_id=42 WHERE users.country='US'"
    assert (
        bind_relationship(parse_pql(query), schema) is schema.tables[1].foreign_keys[0]
    )
    assert schema.model_dump() == original


def test_direct_target_needs_no_join_or_timestamp(schema):
    assert (
        bind_relationship(parse_pql("PREDICT users.age FOR EACH users.user_id"), schema)
        is None
    )


@pytest.mark.parametrize("primary_key", [[], ["age"], ["user_id", "age"]])
def test_requires_single_selected_entity_key(schema, primary_key):
    schema.tables[0].primary_key = primary_key
    with pytest.raises(PQLBindingError, match="UNSUPPORTED_ENTITY_KEY"):
        bind_relationship(parse_pql(QUERY), schema)


def test_no_foreign_key_is_not_guessed_from_matching_column_names(schema):
    schema.tables[1].foreign_keys = []
    with pytest.raises(PQLBindingError, match="NO_DIRECT_RELATIONSHIP"):
        bind_relationship(parse_pql(QUERY), schema)


def test_multihop_is_rejected(schema):
    schema.tables[1].foreign_keys[0].referred_table = "accounts"
    schema.tables.append(
        TableSchema(
            name="accounts",
            primary_key=["user_id"],
            columns=[schema.tables[0].columns[0]],
            foreign_keys=[
                ForeignKeySchema(
                    columns=["user_id"],
                    referred_table="users",
                    referred_columns=["user_id"],
                )
            ],
        )
    )
    with pytest.raises(PQLBindingError, match="NO_DIRECT_RELATIONSHIP"):
        bind_relationship(parse_pql(QUERY), schema)


def test_ambiguous_link_is_not_silently_selected(schema):
    schema.tables[1].foreign_keys.append(schema.tables[1].foreign_keys[0].model_copy())
    with pytest.raises(PQLBindingError, match="AMBIGUOUS_RELATIONSHIP"):
        bind_relationship(parse_pql(QUERY), schema)


def test_foreign_key_column_must_exist(schema):
    schema.tables[1].foreign_keys[0].columns = ["missing"]
    with pytest.raises(PQLBindingError, match="INVALID_FOREIGN_KEY"):
        bind_relationship(parse_pql(QUERY), schema)


@pytest.mark.parametrize("timestamp", [None, "missing"])
def test_timestamp_mapping_is_required(schema, timestamp):
    schema.tables[1].time_column = timestamp
    with pytest.raises(PQLBindingError, match="TIME_COLUMN"):
        bind_relationship(parse_pql(QUERY), schema)


@pytest.mark.parametrize(
    "query",
    [
        QUERY + " WHERE orders.channel='web'",
        "PREDICT COUNT(orders.* WHERE users.age>18,0,30,days) FOR users.user_id=42",
    ],
)
def test_filters_cannot_reference_the_other_table(schema, query):
    with pytest.raises(PQLBindingError, match="INVALID_FILTER_TABLE"):
        bind_relationship(parse_pql(query), schema)


def test_direct_target_cannot_cross_tables(schema):
    with pytest.raises(PQLBindingError, match="CROSS_TABLE_DIRECT_TARGET"):
        bind_relationship(
            parse_pql("PREDICT orders.amount FOR users.user_id=42"), schema
        )
