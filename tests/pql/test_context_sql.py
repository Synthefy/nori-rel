"""Check exact historical candidates and their labels in PostgreSQL."""

import os
import warnings
from datetime import UTC, datetime, timedelta, timezone

import psycopg
import pytest

from nori_rel.contracts import SchemaGraph
from nori_rel.pql import parse_pql
from nori_rel.pql.context_sql import compile_context_sql
from nori_rel.pql.label_sql import compile_label_sql

QUERY = (
    "PREDICT COUNT(orders.*,0,30,days) FOR users.user_id=500 "
    "WHERE users.country IN ('US','CA')"
)
JAN, FEB, MAR = (datetime(2026, m, 1, tzinfo=UTC) for m in (1, 2, 3))


@pytest.fixture
def schema():
    def column(name, kind):
        return dict(name=name, data_type=kind, nullable=True)

    return SchemaGraph.model_validate(
        dict(
            database_id="context",
            schema_name="pg_temp",
            discovered_at=MAR,
            tables=[
                dict(
                    name="users",
                    primary_key=["user_id"],
                    columns=[
                        column("user_id", "INTEGER"),
                        column("created_at", "TIMESTAMPTZ"),
                        column("country", "TEXT"),
                    ],
                ),
                dict(
                    name="orders",
                    columns=[
                        column("buyer_id", "INTEGER"),
                        column("event_time", "TIMESTAMPTZ"),
                        column("amount", "NUMERIC"),
                        column("channel", "TEXT"),
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


def compile_context(schema, query=QUERY, **overrides):
    options = dict(
        anchor_times=[JAN, FEB, MAR],
        anchor_time=datetime(2026, 4, 1, tzinfo=UTC),
        observed_through=datetime(2026, 3, 3, tzinfo=UTC),
        entity_created_at="created_at",
    )
    options.update(overrides)
    return compile_context_sql(parse_pql(query), schema, **options)


@pytest.mark.parametrize("limit", [0, -1, 1.5, True, "5000"])
def test_invalid_candidate_budget(schema, limit):
    with pytest.raises(ValueError, match="positive integer"):
        compile_context(schema, max_context_rows=limit)


@pytest.mark.parametrize(
    "options",
    [
        dict(anchor_time=datetime(2026, 4, 1)),
        dict(observed_through=datetime(2026, 3, 3)),
        dict(anchor_times=[datetime(2026, 1, 1)]),
        dict(anchor_times=["2026-01-01"]),
    ],
)
def test_naive_or_string_times_are_rejected(schema, options):
    with pytest.raises(ValueError, match="timezone-aware"):
        compile_context(schema, **options)


@pytest.mark.parametrize("column", ["missing", "country", "orders.event_time"])
def test_creation_column_must_be_registered_temporal_entity_column(schema, column):
    with pytest.raises(ValueError):
        compile_context(schema, entity_created_at=column)


def test_direct_targets_need_separate_context_policy(schema):
    with pytest.raises(ValueError, match="temporal aggregate"):
        compile_context(schema, query="PREDICT users.user_id FOR EACH users.user_id")


def test_invalid_target_is_rejected_before_generating_candidates(schema):
    with pytest.raises(ValueError, match="numeric"):
        compile_context(
            schema, query=QUERY.replace("COUNT(orders.*", "SUM(orders.channel")
        )


def test_candidate_times_are_deduplicated_normalized_and_bound(schema):
    same_jan = JAN.astimezone(timezone(timedelta(hours=-6)))
    sql, params = compile_context(schema, anchor_times=[JAN, same_jan, FEB, MAR])
    assert [v for k, v in params.items() if k.startswith("anchor_")] == [JAN, FEB]
    assert "2026" not in sql
    assert "5000" not in sql
    assert params["max_context_rows"] == 5000


@pytest.fixture
def db():
    if "NORI_REL_TEST_DATABASE_URL" not in os.environ:
        pytest.skip("PostgreSQL integration database is supplied by CI")
    with psycopg.connect(os.environ["NORI_REL_TEST_DATABASE_URL"]) as conn:
        conn.execute("SET LOCAL TIME ZONE 'UTC'")
        conn.execute(
            "CREATE TEMP TABLE users (user_id integer PRIMARY KEY, "
            "created_at timestamptz, country text)"
        )
        conn.execute(
            "INSERT INTO users VALUES (7,'2026-01-01','CA'), "
            "(42,'2026-01-15','US'), (99,'2025-12-01','GB'), "
            "(123,NULL,'US'), (500,'2026-03-01','US')"
        )
        conn.execute(
            "CREATE TEMP TABLE orders (buyer_id integer, event_time timestamptz, "
            "amount numeric, channel text)"
        )
        conn.execute(
            "INSERT INTO orders VALUES (7,'2026-01-02',2,'web'), "
            "(7,'2026-01-31',3,'store'), (7,'2026-02-01',4,'web'), "
            "(42,'2026-02-02',10,'web'), (99,'2026-01-02',20,'web')"
        )
        yield conn
        conn.rollback()


def candidates(db, schema, **options):
    sql, params = compile_context(schema, **options)
    rows = db.execute(sql, params).fetchall()
    assert len(rows) == len({r[0] for r in rows})
    assert len(rows) == len({r[1:] for r in rows})
    return rows


def test_complete_windows_and_entity_existence_across_the_cohort(db, schema):
    # March is incomplete; user 42 did not exist in January; 123 has unknown
    # creation; GB is filtered out; scoring user 500 does not narrow context.
    assert candidates(db, schema) == [(1, 7, FEB), (2, 42, FEB), (3, 7, JAN)]


@pytest.mark.parametrize("cutoff_field", ["anchor_time", "observed_through"])
def test_both_scoring_cutoff_and_completeness_bound_the_horizon(
    db, schema, cutoff_field
):
    # Feb 1 + 30 days == Mar 3; one microsecond short is still incomplete.
    short = datetime(2026, 3, 3, tzinfo=UTC) - timedelta(microseconds=1)
    assert candidates(db, schema, **{cutoff_field: short}) == [(1, 7, JAN)]


def test_context_filter_is_applied_before_limit(db, schema):
    query = QUERY.replace("IN ('US','CA')", "='US'")
    assert candidates(db, schema, query=query, max_context_rows=1) == [(1, 42, FEB)]
    assert candidates(db, schema, max_context_rows=2) == [(1, 7, FEB), (2, 42, FEB)]
    assert candidates(db, schema, anchor_times=[JAN, FEB, JAN, FEB]) == [
        (1, 7, FEB),
        (2, 42, FEB),
        (3, 7, JAN),
    ]


@pytest.mark.parametrize("anchors", [[], [MAR]])
def test_empty_and_unobserved_anchor_sets_return_no_rows(db, schema, anchors):
    assert candidates(db, schema, anchor_times=anchors) == []


@pytest.mark.parametrize(
    "target,expected",
    [
        ("COUNT(orders.*,0,30,days)", [(1, 0), (2, 1), (3, 2)]),
        (
            "COUNT(orders.* WHERE orders.channel='web',0,30,days)=0",
            [(1, True), (2, False), (3, False)],
        ),
        ("SUM(orders.amount,0,30,days)", [(1, 0), (2, 10), (3, 5)]),
        ("AVG(orders.amount,0,30,days)", [(1, None), (2, 10), (3, 2.5)]),
        ("MIN(orders.amount,0,30,days)", [(1, None), (2, 10), (3, 2)]),
        ("MAX(orders.amount,0,30,days)", [(1, None), (2, 10), (3, 3)]),
        # January's month (Jan 1, Feb 1] also holds the order at Feb 1 00:00.
        ("COUNT(orders.*,0,1,months)", [(1, 0), (2, 1), (3, 3)]),
    ],
)
def test_pql_to_context_to_exact_sql_labels(db, schema, target, expected):
    query = QUERY.replace("COUNT(orders.*,0,30,days)", target)
    sql, params = compile_context(schema, query=query)
    # Test-only temporary materialization exercises #34's actual relation input.
    db.execute("CREATE TEMP TABLE pql_context AS " + sql, params)
    label_sql, label_params = compile_label_sql(parse_pql(query), schema)
    rows = db.execute(label_sql, label_params).fetchall()
    assert sorted(rows) == expected  # Duplicates cannot be hidden by dict().
    assert db.execute(
        "SELECT context_row_id, entity_id, anchor_time "
        "FROM pql_context ORDER BY context_row_id"
    ).fetchall() == [(1, 7, FEB), (2, 42, FEB), (3, 7, JAN)]


def test_budget_can_span_thousands_of_other_entities(db, schema):
    db.execute(
        "INSERT INTO users SELECT id, '2025-12-01'::timestamptz, 'US' "
        "FROM generate_series(10000,15999) AS ids(id)"
    )
    rows = candidates(db, schema, anchor_times=[JAN], max_context_rows=5000)
    assert rows == [(1, 7, JAN)] + [(i + 2, 10000 + i, JAN) for i in range(4999)]


@pytest.mark.parametrize(
    "target",
    [
        "COUNT(orders.*,30,60,days)",
        "SUM(orders.amount,30,60,days)+COUNT(orders.*,0,30,days)",
    ],
)
def test_delayed_interval_eligibility_keys_on_the_end_not_the_length(schema, target):
    """With the cutoff on March 3, only January's (Jan 31, Mar 2] is observed.

    February's interval (Mar 3, Apr 2] lies entirely after the cutoff. The
    plausible wrong rule, cutoff minus the interval's 30-day length, gives
    February 1 and would admit it, labelling rows from data that has not
    arrived; the correct rule, cutoff minus the 60-day end, gives January 2.
    """
    _, params = compile_context(
        schema, query=QUERY.replace("COUNT(orders.*,0,30,days)", target)
    )
    assert [v for k, v in params.items() if k.startswith("anchor_")] == [JAN]


def test_nine_digit_end_beyond_the_datetime_range_is_a_value_error(schema):
    """Scope admits nine digits; the eligibility subtraction is the next gate."""
    with pytest.raises(ValueError, match="supported datetime range"):
        compile_context(schema, query=QUERY.replace("0,30,days", "0,999999999,days"))


def test_arithmetic_waits_for_longest_complete_label_interval(schema):
    query = QUERY.replace(
        "COUNT(orders.*,0,30,days)",
        "COUNT(orders.*,0,30,days)+COUNT(orders.*,0,60,days)",
    )
    _, params = compile_context(schema, query=query)
    assert [v for k, v in params.items() if k.startswith("anchor_")] == [JAN]


def _get_bound_anchors(params):
    return [v for k, v in params.items() if k.startswith("anchor_")]


def test_a_dropped_anchor_is_reported_with_a_month_window_hint(schema):
    options = dict(anchor_times=[JAN, FEB], anchor_time=MAR, observed_through=MAR)
    with pytest.warns(UserWarning, match=r"1 of 2 anchor_times.*2026-02-01.*months"):
        _, params = compile_context(schema, **options)
    assert _get_bound_anchors(params) == [JAN]


def test_a_month_window_keeps_february_without_a_warning(schema):
    query = QUERY.replace("0,30,days", "0,1,months")
    options = dict(anchor_times=[JAN, FEB], anchor_time=MAR, observed_through=MAR)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, params = compile_context(schema, query=query, **options)
    assert _get_bound_anchors(params) == [JAN, FEB]
    with pytest.warns(UserWarning) as record:
        compile_context(
            schema,
            query=query,
            anchor_times=[JAN, FEB, MAR],
            **{k: v for k, v in options.items() if k != "anchor_times"},
        )
    assert "2026-03-01" in str(record[0].message)
    assert "(0, 1, months)" not in str(record[0].message)
