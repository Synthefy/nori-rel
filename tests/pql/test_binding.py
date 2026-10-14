"""Resolve every reference without schema mutation, SQL or model calls."""

from datetime import UTC, datetime

import pytest

from nori_rel.contracts import ColumnSchema, SchemaGraph, TableSchema
from nori_rel.pql import PQLScopeError, parse_pql
from nori_rel.pql.binding import PQLBindingError, bind_columns
from nori_rel.pql.tree_access import node_ids, query_parts


@pytest.fixture
def schema():
    def table(name, names):
        return TableSchema(
            name=name,
            columns=[
                ColumnSchema(name=n, data_type="INTEGER", nullable=False) for n in names
            ],
        )

    return SchemaGraph(
        database_id="registered-db",
        schema_name="public",
        discovered_at=datetime(2026, 1, 1, tzinfo=UTC),
        tables=[
            table("users", ["user_id", "age", "country"]),
            table("orders", ["amount", "channel"]),
        ],
    )


def test_binds_target_selector_and_both_filters_to_existing_metadata(schema):
    tree = parse_pql(
        "PREDICT SUM(orders.amount WHERE orders.channel='web',0,30,days)>500 "
        "FOR users.user_id=42 WHERE users.country='US' AND users.age>=18"
    )
    original = schema.model_dump()
    bound = bind_columns(tree, schema)
    ids, parts = node_ids(tree), query_parts(tree)
    table, column = bound[ids[parts["target_column"]]]
    assert table is schema.tables[1]
    assert column is table.columns[0]
    assert column.data_type == "INTEGER"
    assert bound[ids[parts["entity_column"]]][1].name == "user_id"
    assert {c.name for _, c in bound.values()} == {
        "amount",
        "channel",
        "user_id",
        "country",
        "age",
    }
    assert len(bound) == 5
    assert schema.model_dump() == original


def test_count_rows_and_count_values_are_distinct(schema):
    for column_name in ("*", "amount"):
        tree = parse_pql(
            f"PREDICT COUNT(orders.{column_name},0,30,days) FOR users.user_id=42"
        )
        target = query_parts(tree)["target_column"]
        table, column = bind_columns(tree, schema)[node_ids(tree)[target]]
        assert table.name == "orders"
        assert (column is None) == (column_name == "*")


@pytest.mark.parametrize(
    ("query", "code"),
    [
        ("PREDICT missing.age FOR EACH users.user_id", "UNKNOWN_TABLE"),
        ("PREDICT users.missing FOR EACH users.user_id", "UNKNOWN_COLUMN"),
        ("PREDICT users.age FOR users.missing=42", "UNKNOWN_COLUMN"),
        (
            "PREDICT users.age FOR EACH users.user_id WHERE users.missing=0",
            "UNKNOWN_COLUMN",
        ),
        (
            "PREDICT COUNT(orders.* WHERE orders.missing=0,0,30,days) "
            "FOR users.user_id=42",
            "UNKNOWN_COLUMN",
        ),
        ("PREDICT SUM(orders.*,0,30,days) FOR users.user_id=42", "INVALID_WILDCARD"),
        ("PREDICT users.* FOR EACH users.user_id", "INVALID_WILDCARD"),
        ("PREDICT users.age FOR EACH users.*", "INVALID_WILDCARD"),
        (
            "PREDICT users.age FOR EACH users.user_id WHERE users.*=0",
            "INVALID_WILDCARD",
        ),
        ("PREDICT USERS.age FOR EACH users.user_id", "UNKNOWN_TABLE"),
    ],
)
def test_rejects_unresolved_or_invalid_references(schema, query, code):
    with pytest.raises(PQLBindingError) as error:
        bind_columns(parse_pql(query), schema)
    assert error.value.code == code
    assert query[error.value.start_offset : error.value.end_offset]


def test_ambiguous_metadata_is_not_silently_overwritten(schema):
    query = parse_pql("PREDICT users.age FOR EACH users.user_id")
    schema.tables[0].columns.append(schema.tables[0].columns[1])
    with pytest.raises(PQLBindingError, match="AMBIGUOUS_COLUMN"):
        bind_columns(query, schema)
    schema.tables.append(schema.tables[0])
    with pytest.raises(PQLBindingError, match="AMBIGUOUS_TABLE"):
        bind_columns(query, schema)


def test_quoted_dots_case_and_literal_star_column(schema):
    schema.tables[1].name = "Shop.Orders"
    schema.tables[1].columns[0].name = "Total.Amount"
    schema.tables[1].columns[1].name = "*"
    for name in ("Total.Amount", "*"):
        tree = parse_pql(
            f"PREDICT SUM(`Shop.Orders`.`{name}`,0,30,days) FOR users.user_id=42"
        )
        target = query_parts(tree)["target_column"]
        table, column = bind_columns(tree, schema)[node_ids(tree)[target]]
        assert table.name == "Shop.Orders"
        assert column.name == name


def test_error_location_points_to_reference(schema):
    query = (
        "PREDICT users.age\nFOR EACH users.user_id WHERE users.secret='private-value'"
    )
    with pytest.raises(PQLBindingError) as error:
        bind_columns(parse_pql(query), schema)
    assert (error.value.line, error.value.column) == (2, 30)
    assert query[error.value.start_offset : error.value.end_offset] == "users.secret"
    assert "private-value" not in str(error.value)


def test_scope_errors_precede_name_binding(schema):
    with pytest.raises(PQLScopeError):
        bind_columns(
            parse_pql("PREDICT missing.age CLASSIFY FOR EACH missing.id"), schema
        )


def test_snowflake_unquoted_names_bind_original_uppercase_metadata(schema):
    schema.capabilities.unquoted_identifier_case = "upper"
    for table in schema.tables:
        table.name = table.name.upper()
        for column in table.columns:
            column.name = column.name.upper()
    original = schema.model_dump()
    tree = parse_pql(
        "PREDICT COUNT(orders.* WHERE orders.channel='web',0,30,days) "
        "FOR users.user_id=42 WHERE users.age>18"
    )
    bound = bind_columns(tree, schema)
    entity_table, entity_key = bound[node_ids(tree)[query_parts(tree)["entity_column"]]]
    assert entity_table.name == "USERS"
    assert entity_key.name == "USER_ID"
    assert schema.model_dump() == original


def test_snowflake_exact_and_backtick_names_keep_literal_case(schema):
    schema.capabilities.unquoted_identifier_case = "upper"
    schema.tables[0].name = "Users"
    schema.tables[0].columns[1].name = "Age"
    tree = parse_pql("PREDICT `Users`.`Age` FOR EACH `Users`.user_id")
    target = query_parts(tree)["target_column"]
    assert bind_columns(tree, schema)[node_ids(tree)[target]][1].name == "Age"
    for query in (
        "PREDICT `users`.Age FOR EACH Users.user_id",
        "PREDICT Users.`age` FOR EACH Users.user_id",
    ):
        with pytest.raises(PQLBindingError, match="UNKNOWN"):
            bind_columns(parse_pql(query), schema)
    schema.tables.append(
        schema.tables[0].model_copy(update={"name": "USERS"}, deep=True)
    )
    # An exact match wins even if the fallback name set would be ambiguous.
    assert bind_columns(tree, schema)[node_ids(tree)[target]][0].name == "Users"


@pytest.mark.parametrize("ambiguous_kind", ["table", "column"])
def test_snowflake_unquoted_fallback_rejects_ambiguous_names(schema, ambiguous_kind):
    schema.capabilities.unquoted_identifier_case = "upper"
    schema.tables[0].name = "USERS"
    schema.tables[0].columns[1].name = "AGE"
    if ambiguous_kind == "table":
        schema.tables.append(
            schema.tables[0].model_copy(update={"name": "Users"}, deep=True)
        )
    else:
        schema.tables[0].columns.append(
            schema.tables[0].columns[1].model_copy(update={"name": "Age"})
        )
    with pytest.raises(PQLBindingError, match=f"AMBIGUOUS_{ambiguous_kind.upper()}"):
        bind_columns(parse_pql("PREDICT users.age FOR EACH users.user_id"), schema)
