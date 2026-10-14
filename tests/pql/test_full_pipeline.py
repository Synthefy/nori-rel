"""Real PostgreSQL -> PQL -> FastDFS -> real SDK; only HTTP is simulated."""

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import ANY

import httpx
import numpy as np
import pandas as pd
import pytest

from nori_rel import NoriRelClient
from nori_rel._runtime import HostedNoriPredictor
from nori_rel.connectors.postgres import PostgreSQLConnector

DEC = datetime(2025, 12, 16, tzinfo=UTC)
JAN = datetime(2026, 1, 1, tzinfo=UTC)
FEB = datetime(2026, 2, 1, tzinfo=UTC)
CUTOFF = datetime(2026, 3, 3, tzinfo=UTC)


@pytest.fixture
def full_pipeline(source, monkeypatch):
    connector, db = source
    # The source fixture owns this isolated schema and drops it afterward.
    db.execute("DELETE FROM orders")
    db.execute("DELETE FROM users")
    users = [(i, "2025-01-01", "US") for i in range(1, 19)] + [
        (50, "2026-01-15", "US"),  # Eligible at February, not January.
        (60, "2026-02-15", "US"),  # Born after both historical anchors.
        (900, "2025-01-01", "CA"),
        (901, "2025-01-01", "CA"),
    ]
    with db.cursor() as cursor:
        cursor.executemany("INSERT INTO users VALUES (%s,%s,%s)", users)
    orders = []
    for i in range(1, 17):
        for timestamp, amount, channel in [
            ("2025-12-15", i, "store"),
            ("2026-01-01", 10 * i, "web"),  # Lower label boundary: excluded.
            ("2026-01-10", 100 + i, "web"),
            ("2026-01-20", 9000, "store"),  # Feature input, not a web label.
            ("2026-01-31", 200 + i, "web"),  # Upper January boundary: included.
            ("2026-01-31 00:00:01", 300 + i, "web"),
            ("2026-02-01", 400 + i, "web"),  # Past for Feb features, not label.
            ("2026-02-10", 500 + i, "web"),
            ("2026-03-03", 600 + i, "web"),  # Upper February boundary.
            ("2026-03-03 00:00:01", 999999, "web"),  # Beyond scoring cutoff.
        ]:
            orders.append((len(orders) + 1, i, timestamp, amount, channel))
    for i, timestamp, amount in [
        (17, "2025-12-15", 17),
        (18, "2025-12-15", 18),
        (50, "2026-01-25", 50),
        (50, "2026-02-15", None),
        (900, "2025-12-15", 90),
        (900, "2026-02-20", 900),
        (901, "2025-12-15", 91),
        (901, "2026-02-20", 901),
    ]:
        orders.append((len(orders) + 1, i, timestamp, amount, "web"))
    with db.cursor() as cursor:
        cursor.executemany("INSERT INTO orders VALUES (%s,%s,%s,%s,%s)", orders)
    captured = {"prepared": [], "arrays": [], "http": []}
    prepare = PostgreSQLConnector.prepare_inputs
    predict = HostedNoriPredictor.predict_arrays

    def observe_context(self, *args, **kwargs):
        snapshot, context, counts = prepare(self, *args, **kwargs)
        captured["prepared"].append((snapshot, context.copy(), counts.copy()))
        # Perturb row order without changing identities, labels or source data.
        return snapshot, context.iloc[::-1].copy(), counts

    def observe_arrays(self, train, target, query, **options):
        captured["arrays"].append((train.copy(), target.copy(), query.copy()))
        return predict(self, train, target, query, **options)

    def reply(self, request, **kwargs):
        assert str(request.url) == "https://inference.baseten.co/predict"
        assert request.method == "POST"
        payload = json.loads(request.content)
        captured["http"].append(payload)
        result = {
            "task": "regression",
            "output_type": payload.get("output_type", "median"),
            "predictions": [12.5, 97.5][: len(payload["X_test"])],
        }
        if payload.get("task") == "classification":
            result = {
                "task": "classification",
                "predictions": [0.2, 0.8][: len(payload["X_test"])],
            }
        elif payload.get("output_type") == "quantiles":
            result.update(
                output_type="quantiles",
                quantiles=[[1, 2, 3], [10, 20, 30]],
                taus=[0.1, 0.5, 0.9],
            )
        return httpx.Response(200, json=result, request=request)

    monkeypatch.setattr(PostgreSQLConnector, "prepare_inputs", observe_context)
    monkeypatch.setattr(HostedNoriPredictor, "predict_arrays", observe_arrays)
    monkeypatch.setattr(httpx.Client, "send", reply)
    with NoriRelClient(
        mode="hosted",
        api_key="fixture-only",
        model="synthefy/nori-30m-dev",
        max_retries=0,
    ) as client:
        database = client.connect(
            engine="postgresql",
            credential=connector.config.credential,
            schema_name=connector.config.schema_name,
            tables=["users", "orders"],
            time_columns={"orders": "event_time"},
        )
        yield client, database, db, captured


def run(client, database, expression, **options):
    return client.predict(
        source=database,
        query=f"PREDICT {expression} FOR users.user_id IN (901,900) "
        "WHERE users.country='US'",
        anchor_times=[JAN, FEB, FEB, datetime(2026, 2, 2, tzinfo=UTC)],
        anchor_time=CUTOFF,
        observed_through=CUTOFF,
        entity_created_at="created_at",
        return_details=True,
        **options,
    )


def expected_context(operation):
    """Hand-computed (row_id, entity_id, anchor_time, label), before shuffling.

    January web outcomes are 100+i and 200+i; February's are 500+i and
    600+i. Users 17/18 have no outcomes; user 50 has one all-null outcome.
    Row IDs retain gaps after unknown labels are removed.
    """
    rows = []
    candidates = [(FEB, i) for i in [*range(1, 19), 50]] + [
        (JAN, i) for i in range(1, 19)
    ]
    for row_id, (anchor, entity) in enumerate(candidates, 1):
        if entity <= 16:
            low = (100 if anchor == JAN else 500) + entity
            label = {
                "COUNT(*)": 2,
                "COUNT(amount)": 2,
                "COUNT_DISTINCT": 1,
                "SUM": 2 * low + 100,
                "AVG": low + 50,
                "MIN": low,
                "FIRST": low,
                "LAST": low + 100,
                "MAX": low + 100,
                "binary": False,
            }[operation]
        elif operation == "COUNT(*)" or operation == "COUNT_DISTINCT":
            label = int(entity == 50)
        elif operation == "COUNT(amount)":
            label = 0
        elif operation == "binary":
            label = entity != 50
        elif operation == "SUM" and entity != 50:
            label = 0
        else:
            continue  # Unknown numeric outcome, not a fabricated zero.
        rows.append((row_id, entity, anchor, label))
    return rows


def expected_features(identities):
    """Return all seven expected FastDFS columns, calculated independently.

    Input: (entity_id, anchor_time) pairs in model row order. Output: a feature
    DataFrame in that same order. All past channels count, including events
    exactly at the cutoff; label-window events must not enter earlier features.
    """
    rows = []
    for entity, anchor in identities:
        mode = "web"
        if entity <= 16 and anchor == DEC:
            amounts = [entity]  # Only the December 15 store order has happened.
            mode = "store"
        elif entity <= 16:
            amounts = [entity, 10 * entity]
            if anchor == JAN:
                mode = "store"  # Tied modes choose alphabetically first.
            else:
                amounts += [
                    100 + entity,
                    9000,
                    200 + entity,
                    300 + entity,
                    400 + entity,
                ]
        elif entity in (17, 18, 50):
            amounts = [entity]
        else:
            amounts = [entity - 810, entity]  # Scoring users 900 and 901.
        rows.append(
            [
                "US" if entity < 900 else "CA",
                len(amounts),
                max(amounts),
                sum(amounts) / len(amounts),
                min(amounts),
                mode,
                np.std(amounts, ddof=1) if len(amounts) > 1 else np.nan,
            ]
        )
    return pd.DataFrame(
        rows,
        columns=[
            "d1__users.country",
            "d2__users.COUNT(orders)",
            "d2__users.MAX(orders.amount)",
            "d2__users.MEAN(orders.amount)",
            "d2__users.MIN(orders.amount)",
            "d2__users.MODE(orders.channel)",
            "d2__users.STD(orders.amount)",
        ],
    )


def wire_features(frame):
    """Expected SDK encoding: categories -> numbers, float32, NaN -> JSON null."""
    encoded = frame.copy()
    encoded.iloc[:, 0] = frame.iloc[:, 0].map({"US": 0, "CA": 1})
    encoded.iloc[:, 5] = frame.iloc[:, 5].map({"store": 0, "web": 1})
    return [
        [None if np.isnan(v) else float(v) for v in row]
        for row in encoded.to_numpy(dtype=np.float32)
    ]


def delayed_label(kind, anchor, entity):
    """Hand-computed label over (30,60] days for the delayed-interval tests.

    January's interval is (Jan 31 00:00:00, Mar 2]: the web orders at Jan 31
    00:00:01, Feb 1 and Feb 10. December 16's is (Jan 15, Feb 14]: Jan 31,
    Jan 31 00:00:01, Feb 1 and Feb 10. The immediate (0,30] windows hold
    Jan 10 and Jan 31 00:00:00 for January, and Jan 1 and Jan 10 for December.
    Users 17 and 18 only ever ordered on December 15.
    """
    if entity > 16:
        return {"SUM": 0, "FIRST": None, "SUM-SUM": 0}[kind]
    delayed = 1200 + 3 * entity if anchor == JAN else 1400 + 4 * entity
    immediate = 300 + 2 * entity if anchor == JAN else 100 + 11 * entity
    first = 300 + entity if anchor == JAN else 200 + entity
    return {"SUM": delayed, "FIRST": first, "SUM-SUM": delayed - immediate}[kind]


@pytest.mark.parametrize(
    "kind,expression",
    [
        ("SUM", "SUM(orders.amount WHERE orders.channel='web',30,60,days)"),
        ("FIRST", "FIRST(orders.amount WHERE orders.channel='web',30,60,days)"),
        (
            "SUM-SUM",
            "SUM(orders.amount WHERE orders.channel='web',30,60,days)"
            "-SUM(orders.amount WHERE orders.channel='web',0,30,days)",
        ),
    ],
)
def test_delayed_interval_labels_the_gap_but_features_stop_at_the_anchor(
    full_pipeline, kind, expression
):
    """Over (30,60] days: eligibility keys on the end, features on the anchor.

    December 16 and January anchors are eligible; a February anchor's interval
    closes on April 2, after the March 3 cutoff, even though the interval is 30
    days long (the wrong rule, cutoff minus the length, would admit it).
    Jan 31 00:00:00 is January's open lower bound and Mar 3 is past its closed
    upper bound. FIRST picks a different event than it would over (0,30], and
    an empty delayed window leaves it unknown. The arithmetic row keeps each
    leaf's own window. Features stop at each anchor, exactly as for an
    immediate window.
    """
    client, database, _, captured = full_pipeline
    client.predict(
        source=database,
        query=f"PREDICT {expression} "
        "FOR users.user_id IN (901,900) WHERE users.country='US'",
        anchor_times=[DEC, JAN, FEB, FEB, datetime(2026, 2, 2, tzinfo=UTC)],
        anchor_time=CUTOFF,
        observed_through=CUTOFF,
        entity_created_at="created_at",
        return_details=True,
    )
    candidates = [(JAN, i) for i in range(1, 19)] + [(DEC, i) for i in range(1, 19)]
    expected = [
        (row_id, entity, anchor, label)
        for row_id, (anchor, entity) in enumerate(candidates, 1)
        if (label := delayed_label(kind, anchor, entity)) is not None
    ]
    snapshot, context, counts = captured["prepared"][0]
    assert list(context.itertuples(index=False, name=None)) == expected
    assert counts == {
        "candidate_rows": 36,
        "usable_rows": len(expected),
        "dropped_unknown_labels": 36 - len(expected),
    }
    # The snapshot does hold the gap and the interval; features must ignore both.
    assert snapshot.tables["orders"]["event_time"].max() == CUTOFF
    ordered = expected[::-1]
    train, target, _ = captured["arrays"][0]
    pd.testing.assert_frame_equal(
        train,
        expected_features([(r[1], r[2]) for r in ordered]),
        check_dtype=False,
        check_categorical=False,
        rtol=1e-12,
        atol=1e-12,
    )
    assert target.tolist() == [float(r[3]) for r in ordered]
    assert captured["http"][0]["y_train"] == target.tolist()


@pytest.mark.parametrize(
    "operation,expression,options",
    [
        ("COUNT(*)", "COUNT(orders.* WHERE orders.channel='web',0,30,days)", {}),
        (
            "AVG",
            "SUM(orders.amount WHERE orders.channel='web',0,30,days)/"
            "COUNT(orders.* WHERE orders.channel='web',0,30,days)",
            {},
        ),
        (
            "binary",
            "(COUNT(orders.* WHERE orders.channel='web',0,30,days)+1)=1",
            {"decision_threshold": 0.9},
        ),
        (
            "COUNT_DISTINCT",
            "COUNT_DISTINCT(orders.channel WHERE orders.channel='web',0,30,days)",
            {},
        ),
        (
            "COUNT(amount)",
            "COUNT(orders.amount WHERE orders.channel='web',0,30,days)",
            {},
        ),
        *[
            (op, f"{op}(orders.amount WHERE orders.channel='web',0,30,days)", {})
            for op in ("SUM", "AVG", "MIN", "MAX", "FIRST", "LAST")
        ],
        (
            "binary",
            "COUNT(orders.* WHERE orders.channel='web',0,30,days)=0",
            {"decision_threshold": 0.9},
        ),
        (
            "SUM",
            "SUM(orders.amount WHERE orders.channel='web',0,30,days)",
            {"output_type": "quantiles", "quantiles": [0.1, 0.5, 0.9]},
        ),
    ],
)
def test_pql_to_http_exact_context_features_and_results(
    full_pipeline, operation, expression, options
):
    client, database, _, captured = full_pipeline
    result = run(client, database, expression, **options)
    expected = expected_context(operation)
    snapshot, context, counts = captured["prepared"][0]
    assert list(context.itertuples(index=False, name=None)) == expected
    assert counts == {
        "candidate_rows": 37,
        "usable_rows": len(expected),
        "dropped_unknown_labels": 37 - len(expected),
    }
    assert snapshot.tables["orders"]["event_time"].max() == CUTOFF

    # The fixture reverses context rows. Features AND labels must follow IDs.
    ordered = expected[::-1]
    train_expected = expected_features([(r[1], r[2]) for r in ordered])
    query_expected = expected_features([(900, CUTOFF), (901, CUTOFF)])
    train, target, query = captured["arrays"][0]
    pd.testing.assert_frame_equal(
        train,
        train_expected,
        check_dtype=False,
        check_categorical=False,
        rtol=1e-12,
        atol=1e-12,
    )
    pd.testing.assert_frame_equal(
        query,
        query_expected,
        check_dtype=False,
        check_categorical=False,
        rtol=1e-12,
        atol=1e-12,
    )
    assert target.tolist() == [float(r[3]) for r in ordered]

    task = "classification" if operation == "binary" else "regression"
    # Exact HTTP keys also exclude credentials, database tables and entity IDs.
    expected_payload = {
        "X_train": wire_features(train_expected),
        "y_train": target.tolist(),
        "X_test": wire_features(query_expected),
        "task": task,
        "model": "synthefy/nori-30m-dev",
    }
    if task == "regression":
        expected_payload["output_type"] = options.get("output_type", "median")
        if "quantiles" in options:
            expected_payload["quantiles"] = options["quantiles"]
    assert captured["http"] == [expected_payload]

    frame = result["predictions"]
    assert frame["entity_id"].tolist() == [900, 901]  # FOR order was 901,900.
    assert frame["anchor_time"].tolist() == [CUTOFF, CUTOFF]
    if operation == "binary":
        assert list(frame.columns) == [
            "entity_id",
            "anchor_time",
            "prediction",
            "probability",
        ]
        assert frame["probability"].tolist() == [0.2, 0.8]
        assert frame["prediction"].tolist() == [False, False]  # Explicit 0.9 threshold.
    elif "quantiles" in options:
        assert list(frame.columns) == [
            "entity_id",
            "anchor_time",
            "prediction",
            "quantiles",
        ]
        assert frame["prediction"].tolist() == [2.0, 20.0]
        assert frame["quantiles"].tolist() == [[1, 2, 3], [10, 20, 30]]
    else:
        assert list(frame.columns) == ["entity_id", "anchor_time", "prediction"]
        assert frame["prediction"].tolist() == [12.5, 97.5]
    assert result["metadata"] == {
        "task": task,
        "model": "synthefy/nori-30m-dev",
        "features": "fastdfs",
        **counts,
        "feature_count": 7,
        "pipeline_id": result["pipeline"].pipeline_id,
        "context_rows_sha256": ANY,
        "feature_matrix_sha256": ANY,
    }
    assert result["pipeline"].manifest.features.kind == "fastdfs"
    assert result["pipeline"].feature_names == tuple(captured["arrays"][-1][0].columns)


def test_future_database_changes_leave_model_inputs_unchanged(full_pipeline):
    client, database, db, captured = full_pipeline
    expression = "SUM(orders.amount WHERE orders.channel='web',0,30,days)"
    first = run(client, database, expression)
    db.execute("UPDATE orders SET amount=-123456 WHERE event_time > %s", (CUTOFF,))
    db.execute("INSERT INTO orders VALUES (10000,900,'2026-03-04',765432,'web')")
    second = run(client, database, expression)
    pd.testing.assert_frame_equal(
        captured["prepared"][0][1], captured["prepared"][1][1]
    )
    for before, after in zip(captured["arrays"][0], captured["arrays"][1], strict=True):
        if isinstance(before, pd.Series):
            pd.testing.assert_series_equal(before, after)
        else:
            pd.testing.assert_frame_equal(before, after)
    assert captured["http"][0] == captured["http"][1]
    pd.testing.assert_frame_equal(first["predictions"], second["predictions"])
    assert first["metadata"] == second["metadata"]


def test_context_cap_and_unknown_labels_fail_before_inference(full_pipeline):
    client, database, _, captured = full_pipeline
    # Only February's 19 candidates fit; AVG loses 17/18 (empty) and 50 (NULL).
    with pytest.raises(
        ValueError, match="only 16 usable context rows; at least 32 required"
    ):
        run(
            client,
            database,
            "AVG(orders.amount WHERE orders.channel='web',0,30,days)",
            max_context_rows=19,
        )
    _, context, counts = captured["prepared"][0]
    assert (
        list(context.itertuples(index=False, name=None)) == expected_context("AVG")[:16]
    )
    assert counts == {
        "candidate_rows": 19,
        "usable_rows": 16,
        "dropped_unknown_labels": 3,
    }
    assert not captured["arrays"]
    assert not captured["http"]


@pytest.mark.parametrize(
    "operation,aggregate",
    [
        ("count", "COUNT"),
        ("total", "SUM"),
        ("average", "AVG"),
        ("minimum", "MIN"),
        ("maximum", "MAX"),
    ],
)
def test_legacy_adapter_matches_pql_end_to_end(full_pipeline, operation, aggregate):
    """Both inputs must produce identical context, features, HTTP and results."""
    client, database, _, captured = full_pipeline
    options = dict(
        anchor_times=[JAN, FEB],
        observed_through=CUTOFF,
        entity_created_at="created_at",
        return_details=True,
    )
    direct = client.predict(
        source=database,
        query=f"PREDICT {aggregate}(orders.amount,0,30,days) "
        "FOR users.user_id IN (901,900)",
        anchor_time=CUTOFF,
        **options,
    )
    with pytest.warns(FutureWarning, match="scoring IDs do not restrict"):
        adapted = client.predict(
            source=database,
            entity="users",
            target="orders.amount",
            target_time="orders.event_time",
            operation=operation,
            lookahead="30 days",
            entity_ids=[901, 900],
            as_of=CUTOFF,
            **options,
        )
    pd.testing.assert_frame_equal(
        captured["prepared"][0][1], captured["prepared"][1][1]
    )
    assert captured["prepared"][0][2] == captured["prepared"][1][2]
    for before, after in zip(captured["arrays"][0], captured["arrays"][1], strict=True):
        if isinstance(before, pd.Series):
            pd.testing.assert_series_equal(before, after)
        else:
            pd.testing.assert_frame_equal(before, after)
    assert captured["http"][0] == captured["http"][1]
    pd.testing.assert_frame_equal(direct["predictions"], adapted["predictions"])
    # The adapter writes its own PQL text, so only the pipeline IDs differ.
    assert direct["metadata"].pop("pipeline_id") != adapted["metadata"].pop(
        "pipeline_id"
    )
    assert direct["metadata"] == adapted["metadata"]


@pytest.mark.parametrize(
    "expression,normal_count,null_only_count",
    [
        ("COUNT(orders.*", 4, 1),
        ("COUNT(orders.amount", 3, 0),
        ("COUNT_DISTINCT(orders.amount", 2, 0),
    ],
)
def test_distinct_cardinality_reaches_sdk(
    full_pipeline, expression, normal_count, null_only_count
):
    """Check real SQL labels and SDK y_train for duplicates, NULLs and exclusions.

    Most users have [10, 10, 20, NULL]: row count=4, non-null count=3,
    distinct count=2. User 1 has one distinct value and user 2 has three.
    Users 17/18 have no events; user 50 has only a NULL-valued event.
    """
    client, database, db, captured = full_pipeline
    db.execute(
        "DELETE FROM orders WHERE buyer_id BETWEEN 1 AND 16 AND event_time > %s", (JAN,)
    )
    events = []
    for anchor in (JAN, FEB):
        for entity in range(1, 17):
            values = [10, 10, 10 if entity == 1 else 20, None]
            if entity == 2:
                values[1] = 15
            for day, value in zip((1, 2, 30, 3), values, strict=True):
                events.append((entity, anchor + timedelta(days=day), value, "web"))
            # Unique values outside the target filter/window must not count.
            events.extend(
                [
                    (entity, anchor, 700, "web"),
                    (entity, anchor + timedelta(days=30, seconds=1), 800, "web"),
                    (entity, anchor + timedelta(days=2), 900, "store"),
                ]
            )
    with db.cursor() as cursor:
        cursor.executemany(
            "INSERT INTO orders VALUES (%s,%s,%s,%s,%s)",
            [(10000 + i, *event) for i, event in enumerate(events)],
        )
    run(client, database, expression + " WHERE orders.channel='web',0,30,days)")
    expected = []
    for row_id, entity, anchor, _ in expected_context("COUNT(*)"):
        label = normal_count if entity <= 16 else null_only_count if entity == 50 else 0
        if expression.startswith("COUNT_DISTINCT") and entity in (1, 2):
            label = {1: 1, 2: 3}[entity]
        expected.append((row_id, entity, anchor, label))
    _, context, counts = captured["prepared"][0]
    assert list(context.itertuples(index=False, name=None)) == expected
    assert counts == {
        "candidate_rows": 37,
        "usable_rows": 37,
        "dropped_unknown_labels": 0,
    }
    # The fixture reverses context rows before feature construction/inference.
    labels = [float(row[3]) for row in expected[::-1]]
    assert captured["arrays"][0][1].tolist() == labels
    assert captured["http"][0]["y_train"] == labels


def test_scoring_filter_changes_only_scored_rows(full_pipeline):
    client, database, _, captured = full_pipeline
    expression = "SUM(orders.amount WHERE orders.channel='web',0,30,days)"
    run(client, database, expression)
    selected = run(
        client,
        database,
        expression,
        query_filter="users.country='CA' AND users.user_id=900",
    )
    assert selected["predictions"]["entity_id"].tolist() == [900]
    assert selected["predictions"]["prediction"].tolist() == [12.5]
    pd.testing.assert_frame_equal(
        captured["prepared"][0][1], captured["prepared"][1][1]
    )
    assert captured["prepared"][0][2] == captured["prepared"][1][2]
    before, after = captured["http"]
    assert before["X_train"] == after["X_train"]
    assert before["y_train"] == after["y_train"]
    assert after["X_test"] == before["X_test"][:1]
    assert len(captured["arrays"][1][2]) == 1


@pytest.mark.parametrize("predicate", ["users.country='US'", "users.country IS NULL"])
def test_empty_scoring_filter_skips_features_and_model(
    full_pipeline, monkeypatch, predicate
):
    client, database, _, captured = full_pipeline

    def forbidden(*args, **kwargs):
        pytest.fail("empty scoring must not build features")

    monkeypatch.setattr("nori_rel._pql_runtime.build_pql_features", forbidden)
    result = run(client, database, "COUNT(orders.*,0,30,days)", query_filter=predicate)
    assert result["predictions"].empty
    assert list(result["predictions"].columns) == [
        "entity_id",
        "anchor_time",
        "prediction",
    ]
    assert not captured["arrays"] and not captured["http"]


@pytest.mark.parametrize(
    "predicate,expected",
    [
        ("users.user_id IN (900,901) AND users.country IS NOT NULL", [900, 901]),
        ("users.country IN ('CA',NULL)", [900, 901]),
        ("users.user_id >= 901", [901]),
    ],
)
def test_scoring_filter_for_each(full_pipeline, predicate, expected):
    client, database, _, _ = full_pipeline
    result = client.predict(
        source=database,
        query="PREDICT COUNT(orders.*,0,30,days) FOR EACH users.user_id "
        "WHERE users.country='US'",
        query_filter=predicate,
        anchor_times=[JAN, FEB],
        anchor_time=CUTOFF,
        observed_through=CUTOFF,
        entity_created_at="created_at",
    )
    assert result["entity_id"].tolist() == expected


def test_scoring_filter_does_not_hide_missing_requested_id(full_pipeline):
    client, database, _, captured = full_pipeline
    with pytest.raises(RuntimeError, match="missing"):
        client.predict(
            source=database,
            query="PREDICT COUNT(orders.*,0,30,days) FOR users.user_id IN (900,9999)",
            query_filter="users.user_id=900",
            anchor_times=[JAN, FEB],
            anchor_time=CUTOFF,
            observed_through=CUTOFF,
            entity_created_at="created_at",
        )
    assert not captured["http"]


@pytest.mark.parametrize(
    "predicate",
    [
        "",
        "users.country=1",
        "orders.amount>0",
        "users.missing=1",
        "users.country='US' OR users.country='CA'",
        "COUNT(orders.*,0,30,days)>0",
        "users.country='CA'; DROP TABLE users",
        "users.country='CA' ASSUMING users.user_id=1",
    ],
)
def test_invalid_scoring_filter_fails_before_data_loading(
    full_pipeline, monkeypatch, predicate
):
    client, database, _, captured = full_pipeline

    def forbidden(*args, **kwargs):
        pytest.fail("invalid filter must fail before source-table reads")

    monkeypatch.setattr(PostgreSQLConnector, "_read_bounded", forbidden)
    with pytest.raises(ValueError):
        run(client, database, "COUNT(orders.*,0,30,days)", query_filter=predicate)
    assert not captured["http"]


def test_legacy_ids_intersect_scoring_filter(full_pipeline):
    client, database, _, _ = full_pipeline
    with pytest.warns(FutureWarning):
        result = client.predict(
            source=database,
            entity="users",
            target="orders.amount",
            target_time="orders.event_time",
            operation="count",
            lookahead="30 days",
            entity_ids=[900, 901],
            query_filter="users.user_id=901",
            anchor_times=[JAN, FEB],
            anchor_time=CUTOFF,
            observed_through=CUTOFF,
            entity_created_at="created_at",
        )
    assert result["entity_id"].tolist() == [901]
