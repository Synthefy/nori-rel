"""Check the context rows of static targets, in SQL and in PostgreSQL."""

import os
from datetime import UTC, datetime

import psycopg
import pytest

from nori_rel.contracts import SchemaGraph
from nori_rel.pql import parse_pql
from nori_rel.pql.binding import PQLBindingError
from nori_rel.pql.context_execution import compile_prepared_context_sql
from nori_rel.pql.static_context_sql import get_static_target_context_sql

READ = datetime(2026, 4, 1, tzinfo=UTC)
DEC, JAN, FEB, MAR = (
    datetime(2025, 12, 1, tzinfo=UTC),
    *(datetime(2026, m, 1, tzinfo=UTC) for m in (1, 2, 3)),
)
INCOME = "PREDICT users.income FOR EACH users.user_id"


@pytest.fixture
def schema():
    def column(name, kind):
        return dict(name=name, data_type=kind, nullable=True)

    return SchemaGraph.model_validate(
        dict(
            database_id="static",
            schema_name="pg_temp",
            discovered_at=READ,
            tables=[
                dict(
                    name="users",
                    primary_key=["user_id"],
                    columns=[
                        column("user_id", "INTEGER"),
                        column("created_at", "TIMESTAMPTZ"),
                        column("country", "TEXT"),
                        column("income", "NUMERIC"),
                        column("is_vip", "BOOLEAN"),
                    ],
                ),
                dict(
                    name="orders",
                    columns=[
                        column("buyer_id", "INTEGER"),
                        column("event_time", "TIMESTAMPTZ"),
                        column("amount", "NUMERIC"),
                    ],
                    time_column="event_time",
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


def compile_static(schema, query=INCOME, **options):
    return get_static_target_context_sql(
        parse_pql(query), schema, **({"anchor_time": READ} | options)
    )


@pytest.mark.parametrize(
    "options", [dict(anchor_times=[JAN]), dict(observed_through=READ)]
)
def test_label_window_arguments_are_rejected(schema, options):
    with pytest.raises(ValueError, match="no label window"):
        compile_static(schema, **options)


def test_aggregate_targets_keep_the_temporal_compiler(schema):
    with pytest.raises(ValueError, match="got an aggregate target"):
        compile_static(
            schema, query="PREDICT COUNT(orders.*,0,30,days) FOR EACH users.user_id"
        )


def test_naive_read_time_is_rejected(schema):
    with pytest.raises(ValueError, match="timezone-aware"):
        compile_static(schema, anchor_time=datetime(2026, 4, 1))


def test_text_target_without_comparison_is_rejected(schema):
    with pytest.raises(ValueError, match="numeric or Boolean"):
        compile_static(schema, query="PREDICT users.country FOR EACH users.user_id")


def test_target_on_another_table_is_rejected(schema):
    with pytest.raises(PQLBindingError) as error:
        compile_static(schema, query="PREDICT orders.amount FOR EACH users.user_id")
    assert error.value.code == "CROSS_TABLE_DIRECT_TARGET"


def test_null_scoring_id_is_rejected(schema):
    with pytest.raises(ValueError, match="scoring IDs cannot be NULL"):
        compile_static(
            schema, query="PREDICT users.income FOR users.user_id IN (7, NULL)"
        )


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (dict(max_context_rows=0), "positive integer or None"),
        (
            dict(candidate_sql="SELECT * FROM candidates", max_context_rows=10),
            "max_context_rows=None",
        ),
        (dict(entity_created_at="country"), "not a date or timestamp"),
    ],
)
def test_invalid_limits_and_creation_columns_are_rejected(schema, options, message):
    with pytest.raises(ValueError, match=message):
        compile_static(schema, **options)


def test_ids_filters_and_times_are_bound(schema):
    sql, params = compile_static(
        schema,
        query="PREDICT users.income FOR users.user_id=500 WHERE users.country='GB'",
        query_filter="users.country='US'",
        entity_created_at="created_at",
    )
    assert "500" not in sql and "2026" not in sql
    assert "'US'" not in sql and "'GB'" not in sql
    assert params["scoring_id_0"] == 500
    assert params["anchor_time"] == READ
    assert {"GB", "US"} <= set(params.values())
    assert params["max_context_rows"] == 5000


@pytest.fixture
def db():
    if "NORI_REL_TEST_DATABASE_URL" not in os.environ:
        pytest.skip("PostgreSQL integration database is supplied by CI")
    with psycopg.connect(os.environ["NORI_REL_TEST_DATABASE_URL"]) as conn:
        conn.execute("SET LOCAL TIME ZONE 'UTC'")
        conn.execute(
            "CREATE TEMP TABLE users (user_id integer PRIMARY KEY, "
            "created_at timestamptz, country text, income numeric, is_vip boolean)"
        )
        conn.execute(
            "INSERT INTO users VALUES "
            "(7, '2026-01-01', 'CA', 50, true), "
            "(42, '2026-01-15', 'US', NULL, NULL), "
            "(99, '2025-12-01', 'GB', 80, false), "
            "(123, NULL, 'US', 30, true), "
            "(500, '2026-03-01', 'US', 70, false), "
            "(600, '2026-05-01', 'US', 90, true), "
            "(800, '2026-02-01', NULL, 10, false)"
        )
        yield conn
        conn.rollback()


def context_rows(db, schema, query=INCOME, **options):
    """Return the labelled (entity_id, anchor_time, label) rows in ID order."""
    sql, params = compile_prepared_context_sql(
        parse_pql(query), schema, **({"anchor_time": READ} | options)
    )
    rows = db.execute(sql, params).fetchall()
    assert [row[0] for row in rows] == list(range(1, len(rows) + 1))
    return [
        (entity, anchor, label if isinstance(label, bool) else float(label))
        for _, entity, anchor, label in rows
    ]


def test_for_each_context_is_every_row_with_a_known_label(db, schema):
    # 42's income is unknown, so it is scored and never context. Without a
    # row time every row, even 600 created after the read, uses the read time.
    assert context_rows(db, schema) == [
        (7, READ, 50.0),
        (99, READ, 80.0),
        (123, READ, 30.0),
        (500, READ, 70.0),
        (600, READ, 90.0),
        (800, READ, 10.0),
    ]


def test_for_ids_leave_the_context_whether_or_not_their_label_is_known(db, schema):
    query = "PREDICT users.income FOR users.user_id IN (7, 42, 500)"
    assert [row[0] for row in context_rows(db, schema, query)] == [99, 123, 600, 800]


def test_query_filter_keeps_unscored_for_rows_in_the_context(db, schema):
    # Only 500 is scored: 7 is CA, and 800's NULL country does not match.
    query = "PREDICT users.income FOR users.user_id IN (7, 500, 800)"
    rows = context_rows(db, schema, query, query_filter="users.country='US'")
    assert [row[0] for row in rows] == [7, 99, 123, 600, 800]


def test_row_time_sets_each_cutoff_and_drops_unknown_or_later_rows(db, schema):
    # 123 has no creation time and 600 was created after the read.
    assert context_rows(db, schema, entity_created_at="created_at") == [
        (500, MAR, 70.0),
        (800, FEB, 10.0),
        (7, JAN, 50.0),
        (99, DEC, 80.0),
    ]


def test_cap_counts_usable_rows_after_where_and_scoring(db, schema):
    # 42 is the first US row by key, but its unknown label never takes a slot.
    query = INCOME + " WHERE users.country='US'"
    assert [r[0] for r in context_rows(db, schema, query, max_context_rows=1)] == [123]
    query = "PREDICT users.income FOR users.user_id=123 WHERE users.country='US'"
    assert [r[0] for r in context_rows(db, schema, query, max_context_rows=1)] == [500]
    rows = context_rows(db, schema, entity_created_at="created_at", max_context_rows=2)
    assert [row[0] for row in rows] == [500, 800]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        (
            "PREDICT users.is_vip FOR EACH users.user_id",
            [True, False, True, False, True, False],
        ),
        (
            "PREDICT users.income > 60 FOR EACH users.user_id",
            [False, True, False, True, True, False],
        ),
    ],
)
def test_binary_labels_are_read_from_the_row(db, schema, query, expected):
    rows = context_rows(db, schema, query)
    assert [row[0] for row in rows] == [7, 99, 123, 500, 600, 800]
    assert [row[2] for row in rows] == expected


def test_candidate_sql_narrows_the_static_context(db, schema):
    rows = context_rows(
        db,
        schema,
        max_context_rows=None,
        candidate_sql="SELECT entity_id, anchor_time FROM candidates "
        "WHERE entity_id < 200 UNION ALL SELECT 42, anchor_time FROM candidates",
    )
    assert [row[0] for row in rows] == [7, 99, 123]


@pytest.mark.parametrize("new_income", ["1e9", "NULL"])
def test_changing_a_scored_label_changes_no_context_row(db, schema, new_income):
    query = "PREDICT users.income FOR users.user_id IN (7, 500)"
    before = context_rows(db, schema, query)
    db.execute(f"UPDATE users SET income = {new_income} WHERE user_id IN (7, 500)")
    assert context_rows(db, schema, query) == before


def test_nan_check_applies_to_numeric_targets_only(schema):
    sql, _ = compile_static(schema)
    assert "CAST(u.\"income\" AS DOUBLE PRECISION) <> CAST('NaN'" in sql
    sql, _ = compile_static(schema, query="PREDICT users.is_vip FOR EACH users.user_id")
    assert "NaN" not in sql


@pytest.mark.parametrize(
    "query", [INCOME, "PREDICT users.income > 60 FOR EACH users.user_id"]
)
def test_a_nan_value_is_unknown_and_never_context(db, schema, query):
    # PostgreSQL ranks NaN above every number, so NaN > 60 would be True.
    db.execute("INSERT INTO users VALUES (900, '2026-02-15', 'US', 'NaN', false)")
    assert 900 not in [row[0] for row in context_rows(db, schema, query)]


def test_a_registered_entity_time_column_bounds_the_context(db, schema):
    # The snapshot keeps users dated by the read, so the context must too:
    # 600 is dated after it and 123 has no date.
    users = schema.tables[0].model_copy(update={"time_column": "created_at"})
    timed = schema.model_copy(update={"tables": [users, *schema.tables[1:]]})
    assert [row[0] for row in context_rows(db, timed)] == [7, 99, 500, 800]
