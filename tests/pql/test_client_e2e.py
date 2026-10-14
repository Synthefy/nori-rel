"""Public PQL API through real DB/features; hosted responses stubbed by default."""

import os
from datetime import UTC, datetime

import numpy as np
import pytest

from nori_rel.client import NoriRelClient

QUERY = (
    "PREDICT COUNT(orders.*,0,30,days) FOR users.user_id=99 WHERE users.country='US'"
)
OPTIONS = dict(
    anchor_times=[datetime(2026, 1, 1, tzinfo=UTC)],
    anchor_time=datetime(2026, 2, 1, tzinfo=UTC),
    observed_through=datetime(2026, 2, 1, tzinfo=UTC),
    entity_created_at="created_at",
)


def test_public_pql_e2e_with_details_and_info(runtime_source, caplog):
    runtime, database, calls = runtime_source
    client = NoriRelClient(mode="hosted", api_key="test-only", _runtime=runtime)
    with caplog.at_level("INFO", logger="nori_rel._pql_runtime"):
        result = client.predict(
            source=database, query=QUERY, **OPTIONS, return_details=True
        )
    frame = result["predictions"]
    assert list(frame.itertuples(index=False, name=None)) == [
        (99, OPTIONS["anchor_time"], 12.5)
    ]
    assert result["metadata"]["usable_rows"] == 2
    assert result["metadata"]["task"] == "regression"
    assert "PQL task=regression" in caplog.text
    assert "context=2 usable/2 candidates" in caplog.text
    assert calls[0][1].tolist() == [2.0, 0.0]


def test_omitted_anchor_uses_latest_event_not_wall_clock(runtime_source, source):
    runtime, database, _ = runtime_source
    source[1].execute("INSERT INTO orders VALUES (4,99,'2026-02-01',10,'web')")
    client = NoriRelClient(mode="hosted", api_key="test-only", _runtime=runtime)
    options = {k: v for k, v in OPTIONS.items() if k != "anchor_time"}
    frame = client.predict(source=database, query=QUERY, **options)
    assert frame["anchor_time"].tolist() == [datetime(2026, 2, 1, tzinfo=UTC)]


def test_binary_public_api(runtime_source):
    runtime, database, _ = runtime_source
    client = NoriRelClient(mode="hosted", api_key="test-only", _runtime=runtime)
    result = client.predict(
        source=database, query=QUERY.replace(") FOR", ")>0 FOR"), **OPTIONS
    )
    assert result["entity_id"].tolist() == [99]
    assert result["prediction"].tolist() == [True]
    assert result["probability"].tolist() == [0.8]


@pytest.mark.parametrize(
    "extra",
    [
        dict(entity="users"),
        dict(target="orders.amount"),
        dict(entity_ids=[99]),
        dict(as_of=OPTIONS["anchor_time"]),
    ],
)
def test_mixed_legacy_and_pql_arguments_error(runtime_source, extra):
    runtime, database, calls = runtime_source
    client = NoriRelClient(mode="hosted", api_key="test-only", _runtime=runtime)
    with pytest.raises(ValueError, match="not both"):
        client.predict(source=database, query=QUERY, **OPTIONS, **extra)
    assert not calls


@pytest.mark.parametrize(
    "missing", ["anchor_times", "observed_through", "entity_created_at"]
)
def test_missing_temporal_guarantees_error(runtime_source, missing):
    runtime, database, calls = runtime_source
    client = NoriRelClient(mode="hosted", api_key="test-only", _runtime=runtime)
    with pytest.raises(ValueError, match="requires"):
        client.predict(
            source=database,
            query=QUERY,
            **{k: v for k, v in OPTIONS.items() if k != missing},
        )
    assert not calls


LIVE_TARGETS = [
    pytest.param("COUNT(orders.*,0,30,days)", 42, id="count"),
    pytest.param(
        "SUM(orders.amount,0,30,days) + COUNT(orders.*,0,30,days)",
        42,
        id="addition",
    ),
    pytest.param(
        "SUM(orders.amount,0,30,days) / COUNT(orders.*,0,30,days)",
        36,
        id="division",
    ),
]


@pytest.mark.live_nori
@pytest.mark.parametrize(("target", "usable_rows"), LIVE_TARGETS)
def test_live_hosted_pql_smoke(hosted_api_key, source, target, usable_rows):
    """Score one entity through the public gateway from 42 candidate context rows.

    December orders provide varying past features; January orders provide
    future labels at the January 1 context anchor. Five of the forty added
    users have no January order, so the division target has a zero
    denominator for them and for fixture user 42: those six rows leave the
    context as unknown labels, and the 36 that remain stay above the runtime's
    32-row context floor. No hosted response is stubbed.
    NORI_REL_TEST_HOSTED_MODEL selects the gateway slug; the default is
    synthefy/nori-30m.
    """
    connector, db = source
    db.execute(
        "INSERT INTO users SELECT id, '2025-01-01'::timestamptz, 'US' "
        "FROM generate_series(1000,1039) ids(id)"
    )
    db.execute(
        "INSERT INTO orders SELECT 1000+id, id, '2025-12-15'::timestamptz, "
        "id-999, 'web' FROM generate_series(1000,1039) ids(id)"
    )
    db.execute(
        "INSERT INTO orders SELECT 2000+id, id, '2026-01-05'::timestamptz, 1, 'web' "
        "FROM generate_series(1000,1034) ids(id)"
    )
    with NoriRelClient(
        mode="hosted",
        api_key=hosted_api_key,
        model=os.getenv("NORI_REL_TEST_HOSTED_MODEL", "synthefy/nori-30m"),
    ) as client:
        database = client.connect(
            engine="postgresql",
            credential=connector.config.credential,
            schema_name=connector.config.schema_name,
            tables=["users", "orders"],
            time_columns={"orders": "event_time"},
        )
        result = client.predict(
            source=database,
            query=f"PREDICT {target} FOR users.user_id=99 WHERE users.country='US'",
            **OPTIONS,
        )
        assert result["entity_id"].tolist() == [99]
        assert np.isfinite(result["prediction"]).all()
        assert result.attrs["candidate_rows"] == 42
        assert result.attrs["usable_rows"] == usable_rows
        assert result.attrs["dropped_unknown_labels"] == 42 - usable_rows


def test_dotted_names_predict_through_the_public_api(runtime_source, source):
    """Quoted dotted names survive discovery, SQL, FastDFS and the SDK payload."""
    runtime, database, calls = runtime_source
    connector, db = source
    db.execute('ALTER TABLE users RENAME TO "User.Table"')
    db.execute('ALTER TABLE "User.Table" RENAME COLUMN user_id TO "user.id"')
    db.execute('ALTER TABLE "User.Table" ADD COLUMN "profile.score" integer')
    db.execute('UPDATE "User.Table" SET "profile.score" = "user.id"')
    connector.config.tables = ["User.Table", "orders"]
    query = (
        "PREDICT COUNT(orders.*,0,30,days) "
        "FOR `User.Table`.`user.id`=99 WHERE `User.Table`.country='US'"
    )
    client = NoriRelClient(mode="hosted", api_key="test-only", _runtime=runtime)
    frame = client.predict(source=database, query=query, **OPTIONS)

    assert list(frame.itertuples(index=False, name=None)) == [
        (99, OPTIONS["anchor_time"], 12.5)
    ]
    train, target, scored, _ = calls[0]
    assert target.tolist() == [2.0, 0.0]
    assert list(train.columns) == list(scored.columns)
    assert any("`profile.score`" in str(name) for name in train.columns)
    assert not any("_x" in str(name) for name in train.columns)
