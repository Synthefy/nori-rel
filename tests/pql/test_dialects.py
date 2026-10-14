"""Warehouse SQL compilation and portable ordered-label semantics.

SQLite exercises the common joins and ranked selections; adapting placeholders
and the timestamp-add spelling does not certify real warehouse SQL execution.
Warehouse integration tests provide that certification separately.
"""

import re
import sqlite3
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pandas as pd
import pytest

from nori_rel._pql_features import build_prediction_rows
from nori_rel.connectors.base import Snapshot
from nori_rel.contracts import ColumnSchema, DatabaseEngine, SchemaGraph
from nori_rel.pql import parse_pql
from nori_rel.pql.context_execution import compile_prepared_context_sql
from nori_rel.pql.context_sql import compile_context_sql
from nori_rel.pql.dialects import SQLDialect
from nori_rel.pql.label_sql import compile_label_sql
from nori_rel.pql.scoring_sql import compile_scoring_filter_sql
from nori_rel.pql.sql_expressions import column_kind, namespace_sql

WAREHOUSES = [DatabaseEngine.SNOWFLAKE, DatabaseEngine.DATABRICKS]
JAN = datetime(2026, 1, 1, tzinfo=UTC)
APR = datetime(2026, 4, 1, tzinfo=UTC)
ENTITY_ID = 2**53 + 1


@pytest.fixture(params=WAREHOUSES)
def dialect(request):
    return SQLDialect(request.param)


@pytest.fixture
def schema():
    def column(name, kind, nullable=True):
        return {"name": name, "data_type": kind, "nullable": nullable}

    return SchemaGraph.model_validate(
        {
            "database_id": "warehouse",
            "schema_name": "main",
            "discovered_at": APR,
            "tables": [
                {
                    "name": "users",
                    "primary_key": ["user_id"],
                    "columns": [
                        column("user_id", "BIGINT", False),
                        column("created_at", "TIMESTAMP_NTZ"),
                        column("country", "STRING"),
                    ],
                },
                {
                    "name": "orders",
                    "primary_key": ["event_id"],
                    "time_column": "event_time",
                    "columns": [
                        column("event_id", "BIGINT", False),
                        column("buyer_id", "BIGINT"),
                        column("event_time", "TIMESTAMP_NTZ"),
                        column("amount", "NUMBER(38, 18)"),
                        column("approved", "BOOLEAN"),
                        column("channel", "STRING"),
                    ],
                    "foreign_keys": [
                        {
                            "columns": ["buyer_id"],
                            "referred_table": "users",
                            "referred_columns": ["user_id"],
                        }
                    ],
                },
            ],
        }
    )


def options():
    return {
        "anchor_times": [JAN],
        "anchor_time": APR,
        "observed_through": APR,
        "entity_created_at": "created_at",
    }


def query(target="COUNT(orders.*,0,30,days)", suffix=""):
    return parse_pql(f"PREDICT {target} FOR EACH users.user_id {suffix}")


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("NUMBER(38,18)", "number"),
        ("DECIMAL(12,2)", "number"),
        ("BYTE", "number"),
        ("TINYINT", "number"),
        ("SHORT", "number"),
        ("LONG", "number"),
        ("STRING", "text"),
        ("TIMESTAMP_NTZ(9)", "timestamp"),
        ("TIMESTAMP_LTZ", "timestamp"),
        ("TIMESTAMP_TZ", "timestamp"),
    ],
)
def test_warehouse_scalar_families(kind, expected):
    assert (
        column_kind(ColumnSchema(name="value", data_type=kind, nullable=True))
        == expected
    )


@pytest.mark.parametrize(
    "kind", ["VARIANT", "ARRAY", "OBJECT", "MAP", "STRUCT", "BINARY"]
)
def test_complex_warehouse_types_are_not_scalar_targets(kind):
    with pytest.raises(ValueError, match="Unsupported SQL scalar"):
        column_kind(ColumnSchema(name="value", data_type=kind, nullable=True))


def test_quoting_escapes_delimiters_and_percent_markers(dialect):
    delimiter = "`" if dialect.engine == DatabaseEngine.DATABRICKS else '"'
    name = f"odd{delimiter}.%(value_0)s"
    quoted = dialect.identifier(name)
    assert quoted == f"{delimiter}odd{delimiter}{delimiter}.%%(value_0)s{delimiter}"
    renamed, params = namespace_sql(quoted + "=%(value_0)s", {"value_0": 5}, "leaf")
    assert renamed == quoted + "=%(leaf_value_0)s"
    assert params == {"leaf_value_0": 5}
    assert dialect.qualified_name("cat", "sch", "table.name").count(".") == 3


def test_pinned_references_reach_context_labels_and_scoring(dialect, schema):
    refs = {
        "users": '"db"."main"."users" AT (TIMESTAMP => pinned)',
        "orders": '"db"."main"."orders" AT (TIMESTAMP => pinned)',
    }
    if dialect.engine == DatabaseEngine.DATABRICKS:
        refs = {name: f"`cat`.`main`.`{name}` VERSION AS OF 8" for name in refs}
    pinned = SQLDialect(dialect.engine, refs)
    tree = query(suffix="WHERE users.country='CA'")
    sql, params = compile_prepared_context_sql(
        tree, schema, dialect=pinned, **options()
    )
    assert refs["orders"] in sql
    assert sql.count(refs["users"]) == 2
    assert "MATERIALIZED" not in sql
    assert params["context_anchor_0"] == JAN
    assert params["label_horizon_days"] == 30
    assert params["context_value_1"] == "CA"
    scoring_sql, params = compile_scoring_filter_sql(
        tree,
        schema,
        f"users.user_id={ENTITY_ID} AND users.country='CA'",
        dialect=pinned,
        anchor_time=APR,
        entity_created_at="created_at",
    )
    assert refs["users"] in scoring_sql
    assert ENTITY_ID in params.values()
    assert str(ENTITY_ID) not in scoring_sql


def test_incomplete_pinned_references_fail_before_execution(dialect, schema):
    pinned = SQLDialect(
        dialect.engine, {"users": dialect.qualified_name("main", "users")}
    )
    with pytest.raises(ValueError, match="No pinned source reference for orders"):
        compile_label_sql(query(), schema, dialect=pinned)


@pytest.mark.parametrize("anchors", [[], [JAN]])
def test_anchors_are_bound_and_cast_for_the_warehouse(dialect, schema, anchors):
    settings = options() | {"anchor_times": anchors}
    sql, params = compile_context_sql(query(), schema, dialect=dialect, **settings)
    expected_type = (
        "TIMESTAMP_TZ" if dialect.engine == DatabaseEngine.SNOWFLAKE else "TIMESTAMP"
    )
    assert f"AS {expected_type})" in sql
    assert "::timestamptz" not in sql
    assert "2026" not in sql
    assert ("anchor_0" in params) == bool(anchors)
    if not anchors:
        assert "WHERE FALSE" in sql


def test_delayed_windows_use_exact_hours_and_bound_offsets(dialect, schema):
    sql, params = compile_label_sql(
        query("SUM(orders.amount,30,60,days)"), schema, dialect=dialect
    )
    assert "TIMESTAMPADD(HOUR, %(start_days)s * 24, e.anchor_time)" in sql
    assert "TIMESTAMPADD(HOUR, %(horizon_days)s * 24, e.anchor_time)" in sql
    assert params == {"horizon_days": 60, "start_days": 30}


def test_arithmetic_keeps_decimal_scale_and_exact_large_literals(dialect, schema):
    tree = query(f"SUM(orders.amount,0,30,days)/COUNT(orders.*,0,30,days)+{ENTITY_ID}")
    sql, params = compile_label_sql(tree, schema, dialect=dialect)
    assert "a0.label::numeric" not in sql
    assert "CAST(a0.label" not in sql  # Preserve the source's scale of 18.
    assert "CAST(a1.label AS DECIMAL(19, 0))" in sql
    assert "NULLIF" in sql
    assert "DECIMAL(16, 0)" in sql
    assert ENTITY_ID in params.values()
    decimal_sql, params = compile_label_sql(
        query("COUNT(orders.*,0,30,days)+1e-18"), schema, dialect=dialect
    )
    assert "DECIMAL(18, 18)" in decimal_sql
    assert Decimal("1e-18") in params.values()


def test_arithmetic_average_keeps_fractional_integer_input(dialect, schema):
    sql, _ = compile_label_sql(
        query("AVG(orders.buyer_id,0,30,days)+COUNT(orders.*,0,30,days)"),
        schema,
        dialect=dialect,
    )
    assert "CAST(a0.label" not in sql
    assert "CAST(a1.label AS DECIMAL(19, 0))" in sql


@pytest.mark.parametrize(
    "kind,precision",
    [("TINYINT", 3), ("BYTE", 3), ("SMALLINT", 5), ("INTEGER", 10), ("BIGINT", 19)],
)
def test_integer_arithmetic_uses_the_declared_signed_width(
    dialect, schema, kind, precision
):
    schema.tables[1].columns[3].data_type = kind
    sql, _ = compile_label_sql(
        query("FIRST(orders.amount,0,30,days)+0.01"), schema, dialect=dialect
    )
    assert f"CAST(a0.label AS DECIMAL({precision}, 0))" in sql


def test_count_plus_small_decimal_fits_spark_without_scale_reduction(schema):
    sql, _ = compile_label_sql(
        query("COUNT(orders.*,0,30,days)+1e-18"),
        schema,
        dialect=SQLDialect(DatabaseEngine.DATABRICKS),
    )
    types = [
        tuple(map(int, match))
        for match in re.findall(r"AS DECIMAL\((\d+), (\d+)\)", sql)
    ]
    assert len(types) == 2
    (left_precision, left_scale), (right_precision, right_scale) = types
    # Spark adds one carry digit to the largest integer part and retains the
    # larger fractional part. Staying within 38 prevents engine scale reduction.
    result_scale = max(left_scale, right_scale)
    result_precision = (
        result_scale
        + max(left_precision - left_scale, right_precision - right_scale)
        + 1
    )
    assert result_scale == 18
    assert result_precision <= 38


def test_sum_cast_uses_promoted_width_instead_of_small_input_width(dialect, schema):
    schema.tables[1].columns[3].data_type = "TINYINT"
    sql, _ = compile_label_sql(
        query("SUM(orders.amount,0,30,days)+0.01"), schema, dialect=dialect
    )
    assert "CAST(a0.label AS DECIMAL(19, 0))" in sql


@pytest.mark.parametrize("literal", ["1e-39", "1e39"])
def test_unrepresentable_decimal_literals_fail_instead_of_rounding(
    dialect, schema, literal
):
    with pytest.raises(ValueError, match="at most 38 digits"):
        compile_label_sql(
            query(f"COUNT(orders.*,0,30,days)+{literal}"), schema, dialect=dialect
        )


def test_postgresql_defaults_preserve_existing_syntax(schema):
    default_sql, default_params = compile_prepared_context_sql(
        query(), schema, **options()
    )
    explicit_sql, explicit_params = compile_prepared_context_sql(
        query(), schema, dialect=SQLDialect(), **options()
    )
    assert (default_sql, default_params) == (explicit_sql, explicit_params)
    assert "AS MATERIALIZED" in default_sql
    assert "::timestamptz" in default_sql
    assert "INTERVAL '24 hours'" in default_sql
    ordered, _ = compile_label_sql(query("FIRST(orders.amount,0,30,days)"), schema)
    assert "ARRAY_AGG" in ordered
    assert "FILTER (WHERE" in ordered


@pytest.mark.parametrize(
    "target",
    [
        "COUNT(orders.*,0,30,days)",
        "FIRST(orders.amount,0,30,days)",
        "LAST(orders.amount,0,30,days)",
        "SUM(orders.amount,0,30,days)+COUNT(orders.*,0,30,days)",
        "FIRST(orders.amount,0,30,days)/LAST(orders.amount,0,30,days)",
    ],
)
def test_snowflake_cte_references_resolve_with_case_sensitive_identifiers(
    schema, target
):
    sql, _ = compile_prepared_context_sql(
        query(target), schema, dialect=SQLDialect(DatabaseEngine.SNOWFLAKE), **options()
    )
    internal_name_pattern = r'(?P<quote>["`]?)(?P<name>__nori_\w+)(?P=quote)'
    cte_definition_pattern = re.compile(
        internal_name_pattern + r"\s+AS\s+(?:MATERIALIZED\s+)?\("
    )
    relation_reference_pattern = re.compile(r"(?:FROM|JOIN)\s+" + internal_name_pattern)
    # Snowflake preserves quoted names and uppercases unquoted names. SQLite
    # treats both forms case-insensitively and cannot catch a quoted CTE mismatch.
    declared_cte_names = {
        match["name"] if match["quote"] else match["name"].upper()
        for match in cte_definition_pattern.finditer(sql)
    }
    referenced_cte_names = {
        match["name"] if match["quote"] else match["name"].upper()
        for match in relation_reference_pattern.finditer(sql)
    }
    assert declared_cte_names
    assert referenced_cte_names <= declared_cte_names
    assert 'WITH "__nori_context" AS (' in sql
    assert 'FROM "__nori_context" AS c' in sql


def test_postgresql_context_cte_keeps_its_existing_unquoted_definition(schema):
    sql, _ = compile_prepared_context_sql(query(), schema, **options())
    assert sql.startswith("WITH __nori_context AS MATERIALIZED (")
    assert "FROM __nori_context AS c" in sql


@pytest.fixture
def portable_db():
    connection = sqlite3.connect(":memory:")

    def timestampadd(unit, hours, anchor):
        assert unit == "HOUR"
        return (datetime.fromisoformat(anchor) + timedelta(hours=hours)).isoformat()

    connection.create_function("TIMESTAMPADD", 3, timestampadd)
    connection.executescript(
        "CREATE TABLE users (user_id BIGINT PRIMARY KEY, "
        "created_at TEXT, country TEXT);"
        "CREATE TABLE orders (event_id BIGINT PRIMARY KEY, "
        "buyer_id BIGINT, event_time TEXT, "
        "amount NUMERIC, approved BOOLEAN, channel TEXT);"
        "CREATE TABLE pql_context (context_row_id INTEGER PRIMARY KEY, "
        "entity_id BIGINT, anchor_time TEXT);"
    )
    connection.executemany(
        "INSERT INTO users VALUES (?, ?, ?)",
        [(ENTITY_ID, JAN.isoformat(), "CA"), (7, JAN.isoformat(), "CA")],
    )
    connection.executemany(
        "INSERT INTO pql_context VALUES (?, ?, ?)",
        [(1, ENTITY_ID, JAN.isoformat()), (2, 7, JAN.isoformat())],
    )
    connection.executemany(
        "INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?)",
        [
            (0, ENTITY_ID, JAN.isoformat(), 500, True, "web"),  # Lower bound excluded.
            (1, ENTITY_ID, (JAN + timedelta(days=1)).isoformat(), None, None, "web"),
            (2, ENTITY_ID, (JAN + timedelta(days=1)).isoformat(), 5, True, "web"),
            (3, ENTITY_ID, (JAN + timedelta(days=30)).isoformat(), 7, False, "store"),
            (4, ENTITY_ID, (JAN + timedelta(days=30)).isoformat(), None, None, "store"),
            (
                5,
                ENTITY_ID,
                (JAN + timedelta(days=30, microseconds=1)).isoformat(),
                900,
                True,
                "web",
            ),
        ],
    )
    try:
        yield connection
    finally:
        connection.close()


@pytest.mark.parametrize(
    "target,expected",
    [
        ("COUNT(orders.*,0,30,days)", [4, 0]),
        ("COUNT(orders.amount,0,30,days)", [2, 0]),
        ("COUNT_DISTINCT(orders.channel,0,30,days)", [2, 0]),
        ("SUM(orders.amount,0,30,days)", [12, 0]),
        ("AVG(orders.amount,0,30,days)", [6, None]),
        ("MIN(orders.amount,0,30,days)", [5, None]),
        ("MAX(orders.amount,0,30,days)", [7, None]),
        ("FIRST(orders.amount,0,30,days)", [None, None]),
        ("LAST(orders.amount,0,30,days)", [None, None]),
        ("FIRST(orders.amount WHERE orders.amount IS NOT NULL,0,30,days)", [5, None]),
        ("LAST(orders.amount WHERE orders.amount IS NOT NULL,0,30,days)", [7, None]),
        ("SUM(orders.amount WHERE orders.channel='missing',0,30,days)", [0, 0]),
        ("SUM(orders.amount WHERE orders.amount IS NULL,0,30,days)", [None, 0]),
        (
            "FIRST(orders.approved WHERE orders.approved IS NOT NULL,0,30,days)=true",
            [1, None],
        ),
    ],
)
def test_portable_warehouse_labels_keep_nulls_ties_and_boundaries(
    portable_db, dialect, schema, target, expected
):
    sql, params = compile_label_sql(
        query(target), schema, dialect=dialect, context_schema=None
    )
    sql = re.sub(
        r"%%|%\(([^)]+)\)s", lambda match: ":" + match[1] if match[1] else "%", sql
    )
    sql = sql.replace("TIMESTAMPADD(HOUR,", "TIMESTAMPADD('HOUR',")
    rows = portable_db.execute(sql, params).fetchall()
    assert sorted(rows) == list(enumerate(expected, 1))


def test_declared_case_matching_preserves_physical_sql_and_feature_names(
    dialect, schema
):
    schema.capabilities.unquoted_identifier_case = "upper"
    for table in schema.tables:
        table.name = table.name.upper()
        table.primary_key = [name.upper() for name in table.primary_key]
        if table.time_column is not None:
            table.time_column = table.time_column.upper()
        for column in table.columns:
            column.name = column.name.upper()
        for key in table.foreign_keys:
            key.columns = [name.upper() for name in key.columns]
            key.referred_table = key.referred_table.upper()
            key.referred_columns = [name.upper() for name in key.referred_columns]
    tree = query("SUM(orders.amount,0,30,days)")
    sql, _ = compile_prepared_context_sql(tree, schema, dialect=dialect, **options())
    assert dialect.identifier("ORDERS") in sql and dialect.identifier("AMOUNT") in sql
    assert (
        dialect.identifier("USERS") in sql and dialect.identifier("CREATED_AT") in sql
    )
    assert dialect.identifier("orders") not in sql
    assert dialect.identifier("created_at") not in sql
    sql, _ = compile_scoring_filter_sql(
        tree,
        schema,
        "users.country='CA'",
        dialect=dialect,
        anchor_time=APR,
        entity_created_at="created_at",
    )
    assert dialect.identifier("COUNTRY") in sql
    assert dialect.identifier("CREATED_AT") in sql
    snapshot = Snapshot(
        tables={"USERS": pd.DataFrame({"USER_ID": [ENTITY_ID], "CREATED_AT": [JAN]})},
        primary_keys={"USERS": ["USER_ID"]},
        foreign_keys=[],
        time_columns={},
        schema=schema,
        as_of=APR,
        approximate_bytes=0,
    )
    scoring = build_prediction_rows(
        snapshot,
        "PREDICT COUNT(orders.*,0,30,days) FOR EACH users.user_id",
        "created_at",
    )
    assert scoring.entity_id.tolist() == [ENTITY_ID]


def test_month_windows_add_calendar_months_on_the_warehouse(dialect, schema):
    sql, params = compile_label_sql(
        query("SUM(orders.amount,1,3,months)"), schema, dialect=dialect
    )
    assert "TIMESTAMPADD(MONTH, %(start_months)s, e.anchor_time)" in sql
    assert "TIMESTAMPADD(MONTH, %(horizon_months)s, e.anchor_time)" in sql
    assert params == {"horizon_months": 3, "start_months": 1}
