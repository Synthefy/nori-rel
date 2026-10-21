"""Event-time rows (entity_time): one example per entity row, anchored at its own time.

Real PostgreSQL -> PQL -> context, labels, scoring rows -> FastDFS / NoriDFS, with the
model transport stubbed. The schema is a small event table: page views by devices,
events attached to a view, and signups credited to a view.
"""

import os
from datetime import UTC, datetime
from uuid import uuid4

import pandas as pd
import psycopg
import pytest

from nori_rel import ClassBalance, ContextPolicy, NoriRelClient
from nori_rel._runtime import LocalRelationalRuntime
from nori_rel.connectors.postgres import PostgreSQLConnector
from nori_rel.contracts import DatabaseCreateRequest, EnvironmentCredential
from nori_rel.feature.dfs.noridfs import NativeFeatureConfig, NoriDFS, SiblingPathConfig
from nori_rel.limits import RelationalLimits
from nori_rel.pql import parse_pql
from nori_rel.pql.context_sql import compile_context_sql

RUN = datetime(2026, 3, 5, tzinfo=UTC)
QUERY = "PREDICT COUNT(signups.*, 0, 1, days) FOR EACH views.view_id"
OPTIONS = {
    "anchor_times": None,
    "anchor_time": RUN,
    "observed_through": RUN,
    "entity_created_at": "view_time",
    "entity_time": "view_time",
    "query_filter": "views.view_time >= 2026-03-04",
}

VIEWS = [
    # view_id, device_id, view_time, price
    (1, 1, "2026-03-01 10:00", 5.0),  # signup 5 minutes later: label 1
    # no signup: label 0. Fractional seconds, as in real event logs: the context
    # anchors then mix time formats, which FastDFS rejects unless they are a
    # datetime column.
    (2, 1, "2026-03-01 12:00:00.25", 6.0),
    (3, 2, "2026-03-02 09:00", 7.0),  # signup 25 hours later, outside the window: 0
    (4, 2, "2026-03-02 09:00", 8.0),  # same instant as view 3: label 0
    (5, 3, "2026-03-03 23:30", 9.0),  # signup after midnight, inside the window: 1
    (6, 3, "2026-03-04 08:00", 10.0),  # scored; its window is still open at the run
    (
        7,
        1,
        "2026-03-04 20:00",
        11.0,
    ),  # scored; its signup at 20:30 must not be a feature
    (8, 2, "2026-03-05 01:00", 12.0),  # after the run: neither context nor scored
]
VIEW_EVENTS = [
    # event_id, view_id, event_time, value
    (1, 6, "2026-03-04 07:59", 1.0),  # before view 6: visible to its features
    (2, 6, "2026-03-04 08:00", 10.0),  # at view 6's own time: not before it
    (3, 6, "2026-03-04 08:01", 100.0),  # after view 6
    (4, 1, "2026-03-01 09:00", 2.0),  # before view 1
]
SIGNUPS = [
    # signup_id, view_id, signup_time
    (1, 1, "2026-03-01 10:05"),
    (2, 3, "2026-03-03 10:00"),
    (3, 5, "2026-03-04 00:10"),
    (4, 7, "2026-03-04 20:30"),
]


@pytest.fixture(params=["timestamptz", "timestamp"])
def event_source(request):
    """Yield (connector, connection) for the event schema, with its times stored
    as ``timestamptz`` or as ``timestamp`` without a time zone (read as UTC)."""
    time_type = request.param
    if not os.environ.get("NORI_REL_TEST_DATABASE_URL"):
        pytest.skip("PostgreSQL integration database is supplied by CI")
    name = "pql_event_time_" + uuid4().hex
    url = os.environ["NORI_REL_TEST_DATABASE_URL"]
    with psycopg.connect(url, autocommit=True) as db:
        db.execute(
            psycopg.sql.SQL("CREATE SCHEMA {}").format(psycopg.sql.Identifier(name))
        )
        try:
            db.execute("SET TIME ZONE 'UTC'")
            db.execute(
                psycopg.sql.SQL("SET search_path TO {}").format(
                    psycopg.sql.Identifier(name)
                )
            )
            db.execute(
                "CREATE TABLE devices (device_id bigint PRIMARY KEY, "
                f"created_at {time_type})"
            )
            db.execute(
                "CREATE TABLE views (view_id bigint PRIMARY KEY, device_id bigint "
                f"REFERENCES devices(device_id), view_time {time_type}, price numeric)"
            )
            db.execute(
                "CREATE TABLE view_events (event_id bigint PRIMARY KEY, view_id bigint "
                f"REFERENCES views(view_id), event_time {time_type}, value numeric)"
            )
            db.execute(
                "CREATE TABLE signups (signup_id bigint PRIMARY KEY, view_id bigint "
                f"REFERENCES views(view_id), signup_time {time_type})"
            )
            with db.cursor() as cursor:
                cursor.executemany(
                    "INSERT INTO devices VALUES (%s, '2026-01-01')", [(1,), (2,), (3,)]
                )
                cursor.executemany("INSERT INTO views VALUES (%s,%s,%s,%s)", VIEWS)
                cursor.executemany(
                    "INSERT INTO view_events VALUES (%s,%s,%s,%s)", VIEW_EVENTS
                )
                cursor.executemany("INSERT INTO signups VALUES (%s,%s,%s)", SIGNUPS)
            config = DatabaseCreateRequest(
                engine="postgresql",
                credential=EnvironmentCredential(variable="NORI_REL_TEST_DATABASE_URL"),
                schema_name=name,
                tables=["devices", "views", "view_events", "signups"],
                time_columns={
                    "views": "view_time",
                    "view_events": "event_time",
                    "signups": "signup_time",
                },
            )
            connector = PostgreSQLConnector(config, RelationalLimits())
            try:
                yield connector, db
            finally:
                connector.close()
        finally:
            db.execute(
                psycopg.sql.SQL("DROP SCHEMA {} CASCADE").format(
                    psycopg.sql.Identifier(name)
                )
            )


def _runtime(event_source, monkeypatch, features):
    """Return (runtime, database, calls); the model transport records its inputs."""
    calls = []

    class Transport:
        def __init__(self, **options):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def predict(self, train, target, query, **options):
            calls.append((train.copy(), target.copy(), query.copy(), options))
            return pd.Series([0.25] * len(query))

    monkeypatch.setattr("nori_rel._runtime.SynthefyNoriClient", Transport)
    runtime = LocalRelationalRuntime(
        api_key="test-only",
        model="nori-30m",
        classification_model=None,
        timeout=10,
        max_retries=0,
        limits=RelationalLimits(min_training_rows=2),
        features=features,
    )
    return runtime, runtime.connect(event_source[0].config), calls


def _ts(text):
    return pd.Timestamp(text, tz="UTC")


def test_each_view_is_one_candidate_labelled_from_its_own_time(event_source):
    connector, _ = event_source
    _, context, counts = connector.prepare_inputs(
        connector.config.schema_name,
        QUERY,
        max_context_rows=None,
        **{key: value for key, value in OPTIONS.items() if key != "query_filter"},
    )
    rows = context.assign(anchor_time=pd.to_datetime(context["anchor_time"], utc=True))
    by_view = rows.set_index("entity_id").sort_index()
    # Eligible: view time + 1 day <= the run, so views 1-5; views 6-8 are unsettled.
    assert by_view.index.tolist() == [1, 2, 3, 4, 5]
    assert by_view["anchor_time"].tolist() == [
        _ts("2026-03-01 10:00"),
        _ts("2026-03-01 12:00:00.25"),
        _ts("2026-03-02 09:00"),
        _ts("2026-03-02 09:00"),
        _ts("2026-03-03 23:30"),
    ]
    # Signups in (t, t + 1 day]: view 1 at +5 min and view 5 across midnight count;
    # view 3's, 25 hours later, does not.
    assert by_view["label"].astype(int).tolist() == [1, 0, 0, 0, 1]
    assert counts == {
        "candidate_rows": 5,
        "usable_rows": 5,
        "dropped_unknown_labels": 0,
    }


def test_context_sql_rejects_mixed_or_missing_anchor_arguments(event_source):
    connector, _ = event_source
    schema = connector.discover(connector.config.schema_name)
    tree = parse_pql(QUERY)
    common = {
        "anchor_time": RUN,
        "observed_through": RUN,
        "entity_created_at": "view_time",
    }
    with pytest.raises(ValueError, match="anchor_times=None"):
        compile_context_sql(
            tree, schema, anchor_times=[RUN], entity_time="view_time", **common
        )
    with pytest.raises(ValueError, match="anchor_times is required"):
        compile_context_sql(tree, schema, anchor_times=None, **common)
    with pytest.raises(ValueError, match=r"entity_time='price' names views\.price"):
        compile_context_sql(
            tree, schema, anchor_times=None, entity_time="price", **common
        )


def test_a_capped_read_keeps_the_newest_events_first(event_source):
    connector, _ = event_source
    _, context, _ = connector.prepare_inputs(
        connector.config.schema_name,
        QUERY,
        max_context_rows=2,
        **{key: value for key, value in OPTIONS.items() if key != "query_filter"},
    )
    assert sorted(context["entity_id"]) == [3, 5] or sorted(context["entity_id"]) == [
        4,
        5,
    ]


@pytest.mark.parametrize("features", ["fastdfs", "noridfs"])
def test_scored_views_use_their_own_time_and_read_only_earlier_rows(
    event_source, monkeypatch, features
):
    runtime, database, calls = _runtime(event_source, monkeypatch, features)
    try:
        result = runtime.predict_pql(database, QUERY, **OPTIONS)
    finally:
        runtime.close()
    # Views 6 and 7 exist by the run and pass the filter; view 8 comes after the run.
    assert result["entity_id"].tolist() == [6, 7]
    assert pd.to_datetime(result["anchor_time"], utc=True).tolist() == [
        _ts("2026-03-04 08:00"),
        _ts("2026-03-04 20:00"),
    ]
    assert result.attrs["entity_time"] == "view_time"
    assert result.attrs["pipeline_id"] is None
    (train, target, query, _) = calls[0]
    assert len(train) == 5 and len(query) == 2
    assert sorted(target.tolist()) == [0.0, 0.0, 0.0, 1.0, 1.0]
    price = [
        column
        for column in query.columns
        if "price" in column and "view_events" not in column
    ]
    assert price, f"the view's own price is a feature: {query.columns.tolist()}"
    # Each scored view reads its own columns at its own time.
    assert query[price[0]].astype(float).tolist() == [10.0, 11.0]
    event_count = [
        column
        for column in query.columns
        if "view_events" in column and "count" in column.lower()
    ]
    assert event_count, (
        f"a count of the view's events is a feature: {query.columns.tolist()}"
    )
    # View 6 had three events: 07:59 (before), 08:00 (its own instant), 08:01 (after).
    # NoriDFS counts rows strictly before the view; FastDFS keeps its rule of rows up
    # to and including the cutoff, so it also counts the same-instant event. Neither
    # sees 08:01.
    # View 7 has no events.
    same_instant_counted = 1.0 if features == "fastdfs" else 0.0
    assert query[event_count[0]].fillna(0).astype(float).tolist() == [
        1.0 + same_instant_counted,
        0.0,
    ]
    # View 7's signup at 20:30 is after its 20:00 anchor: no signup feature sees it.
    for column in query.columns:
        if "signups" in column and "count" in column.lower():
            assert query[column].fillna(0).astype(float).tolist() == [0.0, 0.0]


def test_the_report_counts_event_time_rows_by_day(event_source, monkeypatch):
    runtime, database, _ = _runtime(event_source, monkeypatch, "fastdfs")
    try:
        result = runtime.predict_pql(
            database,
            QUERY,
            **OPTIONS,
            context_policy=ContextPolicy(
                balance=ClassBalance(positives="all", negatives_per_positive=1),
                time_sampling="equal_per_cutoff",
                seed=3,
                min_positives=1,
            ),
            output_type="mean",
        )
    finally:
        runtime.close()
    cutoffs = result.attrs["context_policy"]["cutoffs"]
    days = {pd.Timestamp(c["anchor_time"]).strftime("%m-%d"): c for c in cutoffs}
    assert sorted(days) == ["03-01", "03-02", "03-03"]
    assert (
        days["03-01"]["labelled_rows"] == 2 and days["03-01"]["labelled_positives"] == 1
    )
    assert (
        days["03-02"]["labelled_rows"] == 2 and days["03-02"]["labelled_positives"] == 0
    )
    assert (
        days["03-03"]["labelled_rows"] == 1 and days["03-03"]["labelled_positives"] == 1
    )
    # Negatives follow the positives by day: 1 Mar keeps its positive and its
    # one negative; 3 Mar keeps its positive but has no negative, and the
    # shortfall is not taken from 2 Mar, which has no positive.
    assert [
        (day, days[day]["kept_rows"], days[day]["kept_positives"])
        for day in sorted(days)
    ] == [("03-01", 2, 1), ("03-02", 0, 0), ("03-03", 1, 1)]


def test_the_client_checks_entity_time_arguments(event_source):
    connector, _ = event_source
    with NoriRelClient(mode="hosted", api_key="fixture-only", max_retries=0) as client:
        database = client.connect(
            engine="postgresql",
            credential=connector.config.credential,
            schema_name=connector.config.schema_name,
            tables=["devices", "views", "view_events", "signups"],
            time_columns={
                "views": "view_time",
                "view_events": "event_time",
                "signups": "signup_time",
            },
        )
        base = {
            "source": database,
            "query": QUERY,
            "anchor_time": RUN,
            "observed_through": RUN,
        }
        with pytest.raises(ValueError, match="omit anchor_times"):
            client.predict(**base, anchor_times=[RUN], entity_time="view_time")
        with pytest.raises(ValueError, match="omit entity_created_at"):
            client.predict(
                **base, entity_created_at="created_at", entity_time="view_time"
            )
        with pytest.raises(ValueError, match="requires observed_through"):
            client.predict(
                source=database, query=QUERY, anchor_time=RUN, entity_time="view_time"
            )
        with pytest.raises(ValueError, match="requires anchor_times"):
            client.predict(**base, entity_created_at="view_time")


def test_device_siblings_are_the_devices_views_before_each_view(
    event_source, monkeypatch
):
    features = NoriDFS(
        NativeFeatureConfig(max_features=128, max_direct_features=8),
        horizon="1D",
        sibling_paths=SiblingPathConfig(parent_tables=("devices",)),
    )
    runtime, database, calls = _runtime(event_source, monkeypatch, features)
    try:
        runtime.predict_pql(database, QUERY, **OPTIONS)
    finally:
        runtime.close()
    (train, _, query, _) = calls[0]
    prefix = (
        "nr|path|views%7CS%3Aviews.device_id-%3Edevices.device_id%7Cviews.device_id|"
    )
    count = query[prefix + "rows|count|all"].tolist()
    day = query[prefix + "rows|count|86400000000000ns"].tolist()
    recency = query[prefix + "rows|recency|all"].tolist()
    # View 6 (device 3, 4 Mar 08:00): only view 5, 3 Mar 23:30, is earlier.
    # View 7 (device 1, 4 Mar 20:00): views 1 and 2 on 1 Mar; none in the last day.
    assert count == [1.0, 2.0]
    assert day == [1.0, 0.0]
    assert recency == pytest.approx([8.5 / 24, 3 + 8 / 24])
    # Context rows read their device's history at their own time too: view 2 sees
    # view 1, two hours earlier; view 4 does not see view 3, logged at the same instant.
    assert sorted(train[prefix + "rows|count|all"].tolist()) == [
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
