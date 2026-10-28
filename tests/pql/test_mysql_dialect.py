"""MySQL 8 spelling of the shared PQL compilers.

Execution against a real MySQL server is covered by
tests/connectors/test_mysql_live.py; these tests pin the generated syntax.
"""

from datetime import UTC, datetime

import pytest

from nori_rel.contracts import ColumnSchema, DatabaseEngine, SchemaGraph
from nori_rel.pql import parse_pql
from nori_rel.pql.context_execution import compile_prepared_context_sql
from nori_rel.pql.dialects import SQLDialect
from nori_rel.pql.label_sql import compile_label_sql
from nori_rel.pql.sql_expressions import column_kind

MYSQL = SQLDialect(DatabaseEngine.MYSQL)
JAN = datetime(2026, 1, 1, tzinfo=UTC)
FEB = datetime(2026, 2, 1, tzinfo=UTC)


@pytest.fixture
def schema():
    def column(name, kind, nullable=True):
        return {"name": name, "data_type": kind, "nullable": nullable}

    return SchemaGraph.model_validate(
        {
            "database_id": "shop",
            "schema_name": "shop",
            "discovered_at": FEB,
            "tables": [
                {
                    "name": "users",
                    "primary_key": ["user_id"],
                    "columns": [
                        column("user_id", "BIGINT UNSIGNED", False),
                        column("created_at", "DATETIME"),
                        column("country", "VARCHAR(8)"),
                    ],
                },
                {
                    "name": "orders",
                    "primary_key": ["order_id"],
                    "time_column": "event_time",
                    "foreign_keys": [
                        {
                            "columns": ["user_id"],
                            "referred_table": "users",
                            "referred_columns": ["user_id"],
                        }
                    ],
                    "columns": [
                        column("order_id", "BIGINT", False),
                        column("user_id", "BIGINT UNSIGNED"),
                        column("event_time", "TIMESTAMP"),
                        column("quantity", "MEDIUMINT UNSIGNED"),
                        column("amount", "DECIMAL(12, 2)"),
                    ],
                },
            ],
        }
    )


def test_identifiers_use_backticks_and_escape_percent_markers():
    assert MYSQL.identifier("order`s %(x)s") == "`order``s %%(x)s`"
    assert MYSQL.qualified_name("shop", "or.ders") == "`shop`.`or.ders`"


def test_times_are_utc_datetimes_and_an_empty_anchor_list_is_valid_sql():
    assert MYSQL.timestamp_cast("%(anchor_0)s") == "CAST(%(anchor_0)s AS DATETIME(6))"
    assert (
        MYSQL.anchor_relation([])
        == "SELECT CAST(NULL AS DATETIME(6)) FROM DUAL WHERE FALSE"
    )
    assert MYSQL.anchor_relation(["%(a)s", "%(b)s"]) == (
        "SELECT CAST(%(a)s AS DATETIME(6)) UNION ALL SELECT CAST(%(b)s AS DATETIME(6))"
    )
    assert MYSQL.add_days("e.anchor_time", "%(d)s") == (
        "TIMESTAMPADD(HOUR, %(d)s * 24, e.anchor_time)"
    )
    assert MYSQL.add_months("e.anchor_time", "%(m)s") == (
        "TIMESTAMPADD(MONTH, %(m)s, e.anchor_time)"
    )


def test_nan_predicate_keeps_zero_because_mysql_stores_no_nan():
    # CAST('NaN' AS DOUBLE) is 0 in MySQL, so the warehouse predicate would
    # drop every zero.
    assert MYSQL.get_not_nan_predicate("u.`balance`") == "TRUE"


@pytest.mark.parametrize(
    "data_type,operation,expected",
    [
        ("TINYINT", None, "DECIMAL(3, 0)"),
        ("MEDIUMINT", None, "DECIMAL(7, 0)"),
        ("MEDIUMINT UNSIGNED", None, "DECIMAL(8, 0)"),
        ("INT UNSIGNED", None, "DECIMAL(10, 0)"),
        ("BIGINT", None, "DECIMAL(19, 0)"),
        ("BIGINT UNSIGNED", None, "DECIMAL(20, 0)"),
        ("BIGINT UNSIGNED", "SUM", "DECIMAL(38, 0)"),
        ("SMALLINT", "SUM", "DECIMAL(38, 0)"),
    ],
)
def test_integer_operands_keep_every_digit_their_type_holds(
    data_type, operation, expected
):
    column = ColumnSchema(name="x", data_type=data_type, nullable=True)
    assert MYSQL.numeric_cast("a.label", column=column, operation=operation) == (
        f"CAST(a.label AS {expected})"
    )


def test_decimals_and_floats_keep_their_native_type():
    for data_type in ("DECIMAL(12, 2)", "DOUBLE", "FLOAT"):
        column = ColumnSchema(name="x", data_type=data_type, nullable=True)
        assert MYSQL.numeric_cast("a.label", column=column) == "a.label"


@pytest.mark.parametrize(
    "data_type,family",
    [
        ("BIGINT UNSIGNED", "number"),
        ("MEDIUMINT", "number"),
        ("DECIMAL(12, 2)", "number"),
        ("BOOLEAN", "bool"),
        ("MEDIUMTEXT", "text"),
        ("ENUM", "text"),
        ("DATETIME", "timestamp"),
        ("TIMESTAMP", "timestamp"),
        ("DATE", "date"),
    ],
)
def test_mysql_types_have_scalar_families(data_type, family):
    column = ColumnSchema(name="x", data_type=data_type, nullable=True)
    assert column_kind(column) == family


@pytest.mark.parametrize("data_type", ["JSON", "SET", "YEAR", "TIME", "BLOB"])
def test_other_mysql_types_are_not_scalar_targets(data_type):
    with pytest.raises(ValueError, match="Unsupported SQL scalar type"):
        column_kind(ColumnSchema(name="x", data_type=data_type, nullable=True))


ARITHMETIC = (
    "PREDICT SUM(orders.amount,0,30,days) + COUNT(orders.*,0,30,days) "
    "FOR EACH users.user_id"
)


def test_arithmetic_joins_its_leaves_without_full_join(schema):
    sql, _ = compile_label_sql(parse_pql(ARITHMETIC), schema, dialect=MYSQL)
    assert "FULL JOIN" not in sql
    assert "LEFT JOIN __nori_agg_1 AS a1 USING (context_row_id)" in sql
    for engine in (DatabaseEngine.POSTGRESQL, DatabaseEngine.SNOWFLAKE):
        other, _ = compile_label_sql(
            parse_pql(ARITHMETIC), schema, dialect=SQLDialect(engine)
        )
        assert "FULL JOIN __nori_agg_1 AS a1 USING (context_row_id)" in other


def test_prepared_context_quotes_its_cte_and_never_materializes(schema):
    sql, params = compile_prepared_context_sql(
        parse_pql(
            "PREDICT LAST(orders.quantity,0,1,months) FOR EACH users.user_id "
            "WHERE users.country='US'"
        ),
        schema,
        dialect=MYSQL,
        anchor_times=[JAN],
        anchor_time=FEB,
        observed_through=FEB,
        entity_created_at="created_at",
    )
    assert sql.startswith("WITH `__nori_context` AS (")
    assert "MATERIALIZED" not in sql
    assert "::" not in sql
    assert "ROW_NUMBER() OVER (PARTITION BY e.context_row_id" in sql
    assert "TIMESTAMPADD(MONTH" in sql
    assert "LEFT JOIN `shop`.`orders` AS o" in sql
    assert params["context_anchor_0"] == JAN
