"""Execute compiled v0 labels against PostgreSQL, with explicit expected answers."""

import os
from datetime import UTC, datetime
from decimal import Decimal

import psycopg
import pytest

from nori_rel.contracts import ColumnSchema, SchemaGraph
from nori_rel.pql import PQLScopeError, parse_pql
from nori_rel.pql.binding import PQLBindingError
from nori_rel.pql.label_sql import compile_label_sql

QUERY = "PREDICT COUNT(orders.*,0,30,days) FOR users.user_id=42"


@pytest.fixture
def schema():
    def columns(**types):
        return [
            dict(name=name, data_type=kind, nullable=True)
            for name, kind in types.items()
        ]

    return SchemaGraph.model_validate(
        dict(
            database_id="test",
            schema_name="pg_temp",
            discovered_at=datetime(2026, 1, 1, tzinfo=UTC),
            tables=[
                dict(
                    name="users",
                    primary_key=["user_id"],
                    columns=columns(
                        user_id="INTEGER",
                        age="INTEGER",
                        country="TEXT",
                        active="BOOLEAN",
                    ),
                ),
                dict(
                    name="orders",
                    time_column="event_time",
                    columns=columns(
                        buyer_id="INTEGER",
                        event_time="TIMESTAMPTZ",
                        amount="NUMERIC(12, 2)",
                        channel="TEXT",
                    ),
                    foreign_keys=[
                        dict(
                            columns=["buyer_id"],
                            referred_table="users",
                            referred_columns=["user_id"],
                        )
                    ],
                ),
            ],
        )
    )


def compile_target(target, schema, suffix=""):
    # FOR is intentionally independent of the prepared historical cohort.
    return compile_label_sql(
        parse_pql(f"PREDICT {target} FOR users.user_id=500 {suffix}"), schema
    )


@pytest.mark.parametrize(
    "target",
    [
        "COUNT(orders.*,0,30,days)",
        "COUNT(orders.amount,0,30,days)",
        "SUM(orders.amount,0,30,days)",
        "AVG(orders.amount,0,30,days)",
        "MIN(orders.amount,0,30,days)",
        "MAX(orders.amount,0,30,days)",
        "COUNT(orders.*,0,30,days)=0",
        "users.age",
        "users.active",
        "users.country='US'",
    ],
)
def test_supported_targets_have_parameterized_sql(schema, target):
    sql, params = compile_target(target, schema)
    assert "500" not in sql
    assert params.get("horizon_days") == (30 if "orders" in target else None)


@pytest.mark.parametrize(
    "target",
    [
        "SUM(orders.channel,0,30,days)",
        "AVG(orders.channel,0,30,days)",
        "MIN(orders.channel,0,30,days)",
        "MAX(orders.channel,0,30,days)",
        "users.country",
        "users.age='25'",
        "users.age=true",
        "users.active=1",
        "COUNT(orders.*,0,30,days)='0'",
        "users.age=NULL",
        "users.age=--1.25",
    ],
)
def test_type_errors_fail_before_execution(schema, target):
    with pytest.raises(ValueError):
        compile_target(target, schema)


@pytest.mark.parametrize(
    "suffix",
    [
        "WHERE users.age IN (20,'bad')",
        "WHERE users.country=123",
        "WHERE users.age = NULL",
        "WHERE users.country='\\u0000'",
    ],
)
def test_invalid_filter_values_fail_before_execution(schema, suffix):
    with pytest.raises(ValueError):
        compile_target("COUNT(orders.*,0,30,days)", schema, suffix)


@pytest.mark.parametrize(
    "target,suffix",
    [
        ("COUNT(orders.*,-30,0,days)", ""),
        ("COUNT(orders.*,0,1,months)", ""),
        ("COUNT(orders.*,0,30,days)", "WHERE users.age>18 OR users.active=true"),
        ("COUNT(orders.*,0,30,days)", "WHERE COUNT(orders.*,-30,0,days)>0"),
    ],
)
def test_advanced_syntax_remains_unsupported(schema, target, suffix):
    with pytest.raises(PQLScopeError):
        compile_target(target, schema, suffix)


@pytest.mark.parametrize("days", ["0", "-1"])
def test_requires_positive_horizon(schema, days):
    with pytest.raises(ValueError, match="positive"):
        compile_label_sql(parse_pql(QUERY.replace(",30,", f",{days},")), schema)


@pytest.mark.parametrize("window", ["60,30", "30,30"])
def test_interval_must_end_after_it_starts(schema, window):
    """Reversed and empty intervals are value errors, not scope errors."""
    with pytest.raises(ValueError, match="end after it starts"):
        compile_target(f"COUNT(orders.*,{window},days)", schema)


def test_delayed_window_binds_its_start_and_immediate_windows_are_unchanged(schema):
    sql, params = compile_target("SUM(orders.amount,30,60,days)", schema)
    assert params["start_days"] == 30 and params["horizon_days"] == 60
    lower = "e.anchor_time + %(start_days)s * INTERVAL '24 hours'"
    upper = "e.anchor_time + %(horizon_days)s * INTERVAL '24 hours'"
    assert f'\n AND o."event_time" > {lower}\n AND o."event_time" <= {upper}' in sql
    sql, params = compile_target("SUM(orders.amount,0,60,days)", schema)
    assert "start_days" not in params and "start_days" not in sql
    assert '\n AND o."event_time" > e.anchor_time\n' in sql


def test_rejects_bad_mapping_and_wrong_filter_table(schema):
    schema.tables[1].columns[1].data_type = "TEXT"
    with pytest.raises(PQLBindingError, match="UNSUPPORTED_TIME_TYPE"):
        compile_label_sql(parse_pql(QUERY), schema)
    schema.tables[1].foreign_keys = []
    with pytest.raises(PQLBindingError, match="NO_DIRECT_RELATIONSHIP"):
        compile_label_sql(parse_pql(QUERY), schema)


@pytest.mark.parametrize(
    "target,suffix",
    [
        ("COUNT(orders.* WHERE users.age>18,0,30,days)", ""),
        ("COUNT(orders.*,0,30,days)", "WHERE orders.channel='web'"),
    ],
)
def test_filter_scopes_are_checked(schema, target, suffix):
    with pytest.raises(PQLBindingError, match="INVALID_FILTER_TABLE"):
        compile_target(target, schema, suffix)


@pytest.fixture
def db():
    if "NORI_REL_TEST_DATABASE_URL" not in os.environ:
        pytest.skip("PostgreSQL integration database is supplied by CI")
    # Only session-local fixtures; never create/drop persistent user tables.
    with psycopg.connect(os.environ["NORI_REL_TEST_DATABASE_URL"]) as conn:
        conn.execute("SET LOCAL TIME ZONE 'UTC'")
        conn.execute(
            "CREATE TEMP TABLE users (user_id integer PRIMARY KEY, "
            "age integer, country text, active boolean)"
        )
        conn.execute(
            "INSERT INTO users VALUES "
            "(42,25,'US',true), (7,NULL,'CA',false), (99,50,NULL,NULL)"
        )
        conn.execute(
            "CREATE TEMP TABLE pql_context (context_row_id integer PRIMARY KEY, "
            "entity_id integer NOT NULL, anchor_time timestamptz NOT NULL)"
        )
        conn.execute(
            "INSERT INTO pql_context VALUES "
            "(1,42,'2026-01-01'), (2,7,'2026-01-01'), "
            "(3,99,'2026-01-01'), (4,42,'2026-02-01')"
        )
        conn.execute(
            "CREATE TEMP TABLE orders (buyer_id integer, "
            "event_time timestamptz, amount numeric(12,2), channel text)"
        )
        conn.execute(
            "INSERT INTO orders VALUES "
            "(42,'2025-12-31',100,'web'), (42,'2026-01-01',100,'web'), "
            "(42,'2026-01-01 00:00:00.000001+00',10,'web'), "
            "(42,'2026-01-02',NULL,'web'), (42,'2026-01-31',20,'store'), "
            "(42,'2026-01-31 00:00:00.000001+00',100,'web'), "
            "(7,'2026-01-02',NULL,'web'), "
            "(42,'2026-02-02',5,'web'), (42,'2026-02-02',5,'web'), "
            "(NULL,'2026-01-02',100,'web'), (42,NULL,100,'web')"
        )
        yield conn
        conn.rollback()


def unique_labels(rows):
    result = dict(rows)
    assert len(rows) == len(result), "Duplicate context_row_id in label output"
    return result


def test_duplicate_output_rows_fail_even_when_labels_match():
    with pytest.raises(AssertionError, match="Duplicate context_row_id"):
        unique_labels([(1, 25), (1, 25), (2, None)])


def labels(db, schema, target, suffix=""):
    sql, params = compile_target(target, schema, suffix)
    return unique_labels(db.execute(sql, params).fetchall())


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("COUNT(orders.*,0,30,days)", [3, 1, 0, 2]),
        ("COUNT(orders.amount,0,30,days)", [2, 0, 0, 2]),
        ("COUNT_DISTINCT(orders.amount,0,30,days)", [2, 0, 0, 1]),
        ("COUNT_DISTINCT(orders.channel,0,30,days)", [2, 1, 0, 1]),
        ("SUM(orders.amount,0,30,days)", [30, None, 0, 10]),
        ("AVG(orders.amount,0,30,days)", [15, None, None, 5]),
        ("MIN(orders.amount,0,30,days)", [10, None, None, 5]),
        ("MAX(orders.amount,0,30,days)", [20, None, None, 5]),
        # Delayed intervals: (anchor + start, anchor + end], both bounds in days.
        # Row 1 has orders at exactly +1 day and +30 days from its anchor.
        ("COUNT(orders.*,1,30,days)", [1, 0, 0, 0]),
        ("COUNT(orders.*,30,60,days)", [3, 0, 0, 0]),
        ("SUM(orders.amount,30,60,days)", [110, 0, 0, 0]),
        ("MAX(orders.amount,30,60,days)", [100, None, None, None]),
        ("COUNT(orders.*,30,60,days)>0", [True, False, False, False]),
        ("users.age", [25, None, 50, 25]),
        ("users.active", [True, False, None, True]),
        ("users.country='US'", [True, False, None, True]),
        ("users.age>18", [True, None, True, True]),
    ],
)
def test_labels_against_known_answers(db, schema, expression, expected):
    assert labels(db, schema, expression) == dict(enumerate(expected, 1))


@pytest.mark.parametrize(
    "operator,expected",
    [
        ("=", [False, False, False, True]),
        ("!=", [True, True, True, False]),
        (">", [True, False, False, False]),
        (">=", [True, False, False, True]),
        ("<", [False, True, True, False]),
        ("<=", [False, True, True, True]),
    ],
)
def test_all_target_comparisons(db, schema, operator, expected):
    assert labels(db, schema, f"COUNT(orders.*,0,30,days){operator}2") == dict(
        enumerate(expected, 1)
    )


def test_comparison_unknown_stays_null_and_decimals_are_exact(db, schema):
    assert labels(db, schema, "AVG(orders.amount,0,30,days)>5.25") == {
        1: True,
        2: None,
        3: None,
        4: False,
    }
    _, params = compile_target("SUM(orders.amount,0,30,days)>1.25", schema)
    assert Decimal("1.25") in params.values()
    assert labels(db, schema, "SUM(orders.amount,0,30,days)=0") == {
        1: False,
        2: None,
        3: True,
        4: False,
    }


def test_target_filter_does_not_remove_zero_event_context_rows(db, schema):
    assert labels(
        db,
        schema,
        "COUNT(orders.* WHERE orders.channel='web' AND orders.amount>=10,0,30,days)=0",
    ) == {1: False, 2: True, 3: True, 4: True}
    assert labels(
        db, schema, "SUM(orders.amount WHERE orders.channel='missing',0,30,days)"
    ) == {1: 0, 2: 0, 3: 0, 4: 0}


def test_context_filter_is_separate_from_target_and_scoring(db, schema):
    assert labels(
        db,
        schema,
        "COUNT(orders.* WHERE orders.channel IN ('web','mobile'),0,30,days)",
        "WHERE users.country IN ('US','CA')",
    ) == {1: 2, 2: 1, 4: 2}
    assert labels(
        db,
        schema,
        "users.age",
        "WHERE users.country IN ('US','CA') AND users.age>=18",
    ) == {1: 25, 4: 25}


@pytest.mark.parametrize(
    "suffix,expected_ids",
    [
        ("WHERE users.country IS NULL", {3}),
        ("WHERE users.country IS NOT NULL", {1, 2, 4}),
        ("WHERE users.country IN ('US',NULL)", {1, 4}),
        ("WHERE users.active=false", {2}),
        ("WHERE users.country='missing'", set()),
    ],
)
def test_null_boolean_and_empty_context_filters(db, schema, suffix, expected_ids):
    assert set(labels(db, schema, "COUNT(orders.*,0,30,days)", suffix)) == expected_ids


@pytest.mark.parametrize(
    "literal,value",
    [
        (r"'O\'Reilly'", "O'Reilly"),
        (r"'slash\\path'", "slash\\path"),
        (r"'caf\u00e9'", "café"),
        ("'a\"b'", 'a"b'),
        ('"x\'); DROP TABLE users;--"', "x'); DROP TABLE users;--"),
    ],
)
def test_string_literals_are_bound_not_interpolated(db, schema, literal, value):
    db.execute("UPDATE orders SET channel=%s", (value,))
    sql, params = compile_target(
        f"COUNT(orders.* WHERE orders.channel={literal},0,30,days)", schema
    )
    assert value in params.values()
    assert value not in sql
    assert unique_labels(db.execute(sql, params).fetchall()) == {1: 3, 2: 1, 3: 0, 4: 2}
    assert db.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 3


def test_temporal_filter_and_quoted_identifier(db, schema):
    assert labels(
        db, schema, "COUNT(orders.* WHERE orders.event_time<'2026-01-10',0,30,days)"
    ) == {1: 2, 2: 1, 3: 0, 4: 0}
    name = 'Order.Events"%s'
    db.execute(
        psycopg.sql.SQL("ALTER TABLE orders RENAME TO {}").format(
            psycopg.sql.Identifier(name)
        )
    )
    schema.tables[1].name = name
    assert labels(db, schema, f"COUNT(`{name}`.*,0,30,days)") == {
        1: 3,
        2: 1,
        3: 0,
        4: 2,
    }


def test_empty_context_table_and_direct_target_without_time_mapping(db, schema):
    schema.tables = schema.tables[:1]
    assert labels(db, schema, "users.age") == {1: 25, 2: None, 3: 50, 4: 25}
    db.execute("DELETE FROM pql_context")
    assert labels(db, schema, "users.age") == {}


def test_inline_null_filter_preserves_context_rows_with_only_non_null_events(
    db, schema
):
    assert labels(
        db, schema, "COUNT(orders.* WHERE orders.amount IS NULL,0,30,days)"
    ) == {1: 1, 2: 1, 3: 0, 4: 0}


@pytest.mark.parametrize("operator", ["IS NULL", "IS NOT NULL"])
def test_jsonb_null_checks_in_both_filter_scopes(db, schema, operator):
    for table in schema.tables:
        table.columns.append(
            ColumnSchema(name="metadata", data_type="JSONB", nullable=True)
        )
        db.execute(f"ALTER TABLE {table.name} ADD COLUMN metadata jsonb")
    db.execute("UPDATE users SET metadata='{}' WHERE user_id=42")
    db.execute("UPDATE orders SET metadata='{}' WHERE channel='store'")
    is_null = operator == "IS NULL"
    assert labels(
        db, schema, f"COUNT(orders.* WHERE orders.metadata {operator},0,30,days)"
    ) == ({1: 2, 2: 1, 3: 0, 4: 2} if is_null else {1: 1, 2: 0, 3: 0, 4: 0})
    assert labels(
        db,
        schema,
        "COUNT(orders.*,0,30,days)",
        f"WHERE users.metadata {operator}",
    ) == ({2: 1, 3: 0} if is_null else {1: 3, 4: 2})


@pytest.mark.parametrize(
    "target,suffix",
    [
        ("COUNT(orders.* WHERE orders.metadata='{}',0,30,days)", ""),
        ("COUNT(orders.*,0,30,days)", "WHERE users.metadata='{}'"),
        ("users.metadata='{}'", ""),
    ],
)
def test_jsonb_literal_comparisons_still_rejected(schema, target, suffix):
    for table in schema.tables:
        table.columns.append(
            ColumnSchema(name="metadata", data_type="JSONB", nullable=True)
        )
    with pytest.raises(ValueError, match="Unsupported SQL scalar type: JSONB"):
        compile_target(target, schema, suffix)


def test_distinct_filters_nulls_and_comparison(db, schema):
    assert labels(
        db,
        schema,
        "COUNT_DISTINCT(orders.amount WHERE orders.channel='web',0,30,days)=1",
    ) == {1: True, 2: False, 3: False, 4: True}


@pytest.mark.parametrize("column", ["*", "metadata"])
def test_distinct_rejects_wildcard_and_non_scalar(schema, column):
    schema.tables[1].columns.append(
        ColumnSchema(name="metadata", data_type="JSONB", nullable=True)
    )
    with pytest.raises(ValueError):
        compile_target(f"COUNT_DISTINCT(orders.{column},0,30,days)", schema)


@pytest.fixture
def ordered_events(db, schema):
    db.execute(
        "ALTER TABLE orders ADD COLUMN event_id integer "
        "GENERATED ALWAYS AS IDENTITY PRIMARY KEY"
    )
    schema.tables[1].primary_key = ["event_id"]
    schema.tables[1].columns.append(
        ColumnSchema(name="event_id", data_type="INTEGER", nullable=False)
    )
    return db, schema


@pytest.mark.parametrize(
    "operation,expected",
    [
        ("FIRST", {1: 10, 2: None, 3: None, 4: 5}),
        ("LAST", {1: 20, 2: None, 3: None, 4: 5}),
    ],
)
def test_ordered_labels_exact_boundaries(ordered_events, operation, expected):
    db, schema = ordered_events
    assert labels(db, schema, f"{operation}(orders.amount,0,30,days)") == expected
    # Tied timestamps use the primary key in the same direction as time.
    db.execute(
        "INSERT INTO orders (buyer_id,event_time,amount,channel) "
        "VALUES (42,'2026-02-02',8,'web')"
    )
    result = labels(db, schema, f"{operation}(orders.amount,0,30,days)")
    assert result[4] == (5 if operation == "FIRST" else 8)


def test_ordered_selected_null_is_not_skipped_and_filters_apply_first(ordered_events):
    db, schema = ordered_events
    assert labels(
        db, schema, "LAST(orders.amount WHERE orders.channel='web',0,30,days)"
    ) == {1: None, 2: None, 3: None, 4: 5}
    assert labels(
        db,
        schema,
        "LAST(orders.amount WHERE orders.amount IS NOT NULL "
        "AND orders.channel='web',0,30,days)=5",
    ) == {1: False, 2: None, 3: None, 4: True}


@pytest.mark.parametrize("operation", ["FIRST", "LAST"])
def test_ordered_requires_unique_event_key(schema, operation):
    with pytest.raises(ValueError, match="primary key"):
        compile_target(f"{operation}(orders.amount,0,30,days)", schema)


def test_ordered_boolean_requires_explicit_predicate(ordered_events):
    db, schema = ordered_events
    db.execute("ALTER TABLE orders ADD COLUMN approved boolean DEFAULT true")
    schema.tables[1].columns.append(
        ColumnSchema(name="approved", data_type="BOOLEAN", nullable=True)
    )
    assert labels(db, schema, "FIRST(orders.approved,0,30,days)=true") == {
        1: True,
        2: True,
        3: None,
        4: True,
    }
    with pytest.raises(ValueError, match="explicit comparison"):
        compile_target("FIRST(orders.approved,0,30,days)", schema)
    with pytest.raises(ValueError, match="numeric or Boolean"):
        compile_target("FIRST(orders.channel,0,30,days)", schema)


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("SUM(orders.amount,0,30,days)-1", [29, None, -1, 9]),
        # The most leaves allowed: eight CTEs joined seven times, run for real.
        ("+".join(["COUNT(orders.*,0,30,days)"] * 8), [24, 8, 0, 16]),
        ("SUM(orders.amount,0,30,days)/COUNT(orders.*,0,30,days)", [10, None, None, 5]),
        ("1+SUM(orders.amount,0,30,days)*2", [61, None, 1, 21]),
        ("(1+SUM(orders.amount,0,30,days))*2", [62, None, 2, 22]),
        (
            "-(SUM(orders.amount,0,30,days)-1)/2",
            [Decimal("-14.5"), None, Decimal("0.5"), Decimal("-4.5")],
        ),
        ("COUNT(orders.*,0,30,days)/2", [Decimal("1.5"), Decimal("0.5"), 0, 1]),
        ("COUNT(orders.*,0,30,days)+1-2-3", [-1, -3, -4, -2]),
        ("COUNT(orders.*,0,30,days)--2", [5, 3, 2, 4]),
        (
            "1e-2+COUNT(orders.*,0,30,days)",
            [Decimal("3.01"), Decimal("1.01"), Decimal("0.01"), Decimal("2.01")],
        ),
        ("SUM(orders.amount,0,30,days)/0", [None] * 4),
        (
            # Leaves keep their own windows: (30,60] minus (0,30] per row.
            "SUM(orders.amount,30,60,days)-SUM(orders.amount,0,30,days)",
            [80, None, 0, -10],
        ),
        (
            # Two delayed leaves: adjacent windows, separate agg_i_start_days.
            # Row 1 is the Jan 31 web order (100) minus the two Feb 2 ones (5+5).
            "SUM(orders.amount,30,31,days)-SUM(orders.amount,31,60,days)",
            [90, 0, 0, 0],
        ),
        (
            "SUM(orders.amount,0,30,days)/COUNT(orders.*,0,30,days)>5",
            [True, None, None, False],
        ),
    ],
)
def test_arithmetic_exact_results(db, schema, expression, expected):
    assert labels(db, schema, expression) == dict(enumerate(expected, 1))


def test_delayed_interval_bounds_are_open_below_and_closed_above(db, schema):
    """Probe both ends of (anchor + 30, anchor + 60] to the microsecond.

    Row 1's anchor is January 1. Its order at exactly January 31 00:00:00 is
    excluded from (30,60] and included in (0,30]; the one a microsecond later
    is included. An order at exactly March 2 00:00:00 is the last one inside
    (30,60]; a microsecond later it belongs to (60,90]. Adjacent intervals
    therefore tile time: every order lands in exactly one of them. Both probe
    orders also sit inside row 4's immediate window, (Feb 1, Mar 3].
    """
    db.execute(
        "INSERT INTO orders VALUES (42,'2026-03-02',7,'web'), "
        "(42,'2026-03-02 00:00:00.000001+00',1000,'web')"
    )
    assert labels(db, schema, "COUNT(orders.*,0,30,days)") == {1: 3, 2: 1, 3: 0, 4: 4}
    assert labels(db, schema, "SUM(orders.amount,30,60,days)") == {
        1: 117,
        2: 0,
        3: 0,
        4: 0,
    }
    assert labels(db, schema, "SUM(orders.amount,60,90,days)") == {
        1: 1000,
        2: 0,
        3: 0,
        4: 0,
    }
    assert labels(db, schema, "COUNT(orders.*,30,90,days)") == {1: 5, 2: 0, 3: 0, 4: 0}


def test_arithmetic_separate_tables_do_not_multiply_labels(db, schema):
    db.execute(
        "CREATE TEMP TABLE refunds (buyer_id integer, event_time timestamptz, "
        "amount numeric, channel text)"
    )
    db.execute(
        "INSERT INTO refunds VALUES (42,'2026-01-02',2,'web'), "
        "(42,'2026-01-03',3,'web')"
    )
    table = schema.tables[1].model_copy(deep=True)
    table.name = "refunds"
    schema.tables.append(table)
    assert labels(
        db,
        schema,
        "SUM(orders.amount WHERE orders.channel='web',0,30,days)"
        "-SUM(refunds.amount WHERE refunds.channel='web',0,30,days)",
        "WHERE users.country IN ('US','CA')",
    ) == {1: 5, 2: None, 4: 10}


def test_empty_window_semantics_follow_the_aggregate_on_that_side(db, schema):
    """SUM of nothing is 0; AVG of nothing is undefined. Arithmetic inherits both.

    Only user 42 has refunds, and none in February. Rows 3 and 4 therefore
    carry no refund data, and whether they survive depends on which aggregate
    was chosen for the refund side: SUM yields 0 and keeps the row, AVG yields
    NULL and the row is later excluded as an unknown label. That is correct,
    since an average of no values has no value, and this pins it so a future
    normalisation is a deliberate change rather than a drift.
    """
    db.execute(
        "CREATE TEMP TABLE refunds (buyer_id integer, event_time timestamptz, "
        "amount numeric, channel text)"
    )
    db.execute(
        "INSERT INTO refunds VALUES (42,'2026-01-02',2,'web'), "
        "(42,'2026-01-03',3,'web')"
    )
    table = schema.tables[1].model_copy(deep=True)
    table.name = "refunds"
    schema.tables.append(table)
    orders = "SUM(orders.amount,0,30,days)"
    assert labels(db, schema, f"{orders}-SUM(refunds.amount,0,30,days)") == {
        1: 25,
        2: None,
        3: 0,
        4: 10,
    }
    assert labels(db, schema, f"{orders}-AVG(refunds.amount,0,30,days)") == {
        1: Decimal("27.5"),
        2: None,
        3: None,
        4: None,
    }


@pytest.mark.parametrize(
    "expression",
    [
        "SUM(orders.channel,0,30,days)+1",
        "users.age+1",
        "1+2",
        "COUNT(orders.*,0,30,days)+'1'",
        "COUNT(orders.*,0,30,days)+true",
        "COUNT(orders.*,0,30,days)+SUM(orders.missing,0,30,days)",
        "COUNT(orders.*,0,30,days)+SUM(orders.amount WHERE users.age>0,0,30,days)",
    ],
)
def test_invalid_arithmetic_fails_before_sql(schema, expression):
    with pytest.raises(ValueError):
        compile_target(expression, schema)


def test_arithmetic_ctes_are_combined_with_a_full_join(schema):
    """Pin the outer join choice.

    Every per-aggregate CTE shares the context relation and context filter,
    and inline leaf predicates sit in the LEFT JOIN ON clause, so all CTEs
    emit the same rows and an inner join would behave identically today. The
    FULL JOIN exists so a future per-leaf narrowing at the context level
    surfaces as a NULL label instead of a vanished row. Only hand-splicing a
    predicate into one CTE could exercise that, so this asserts the emitted
    shape; the eight-leaf row in test_arithmetic_exact_results executes the
    chained joins for real.
    """
    sql, _ = compile_label_sql(
        parse_pql(
            "PREDICT SUM(orders.amount,0,30,days)-COUNT(orders.*,0,30,days) "
            "FOR users.user_id=42"
        ),
        schema,
    )
    assert "FULL JOIN __nori_agg_1 AS a1 USING (context_row_id)" in sql
    assert "\nJOIN __nori_agg_" not in sql
