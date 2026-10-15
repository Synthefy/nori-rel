"""Save a pipeline, then predict with it in a new process against PostgreSQL."""

import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pipeline_fakes
import psycopg
import pytest

from nori_rel import (
    EnvironmentCredential,
    NoriRelClient,
    RelationalLimits,
)
from nori_rel._feature_program import program_record
from nori_rel.feature.agentic.contracts import FeatureProgram, FeatureQuery
from nori_rel.pipeline import (
    PipelineContextError,
    PipelineModelError,
    PipelineSchemaError,
    PipelineTaskError,
)

QUERY = "PREDICT SUM(orders.amount,0,30,days) FOR EACH users.user_id"
ANCHORS = [datetime(2025, month, 1, tzinfo=UTC) for month in (9, 10, 11, 12)]
CUTOFF = datetime(2026, 1, 1, tzinfo=UTC)
LATER = datetime(2026, 2, 1, tzinfo=UTC)
CHILD = Path(__file__).with_name("pipeline_child.py")
LIMITS = RelationalLimits(min_training_rows=2)


@pytest.fixture
def shop(monkeypatch, tmp_path):
    """A schema with 12 users and their orders from August to December 2025."""
    if "NORI_REL_TEST_DATABASE_URL" not in os.environ:
        pytest.skip("PostgreSQL integration database is supplied by CI")
    name = "pipeline_" + uuid4().hex
    url = os.environ["NORI_REL_TEST_DATABASE_URL"]
    with psycopg.connect(url, autocommit=True) as db:
        db.execute(
            psycopg.sql.SQL("CREATE SCHEMA {}").format(psycopg.sql.Identifier(name))
        )
        try:
            db.execute(
                psycopg.sql.SQL("SET search_path TO {}").format(
                    psycopg.sql.Identifier(name)
                )
            )
            db.execute(
                "CREATE TABLE users (user_id bigint PRIMARY KEY, "
                "created_at timestamptz, country text)"
            )
            db.execute(
                "CREATE TABLE orders (id integer PRIMARY KEY, buyer_id bigint "
                "REFERENCES users(user_id), event_time timestamptz, amount numeric, "
                "channel text)"
            )
            for user in range(1, 13):
                db.execute(
                    "INSERT INTO users VALUES (%s, '2025-01-01', %s)",
                    (user, "US" if user % 3 else "CA"),
                )
            order = 0
            for user in range(1, 13):
                for month in range(8, 13):
                    for day in range(1, 1 + (user + month) % 4):
                        order += 1
                        db.execute(
                            "INSERT INTO orders VALUES (%s, %s, %s, %s, %s)",
                            (
                                order,
                                user,
                                datetime(2025, month, 2 + 7 * day, tzinfo=UTC),
                                user * 1.5 + month + day,
                                "web" if (user + day) % 2 else "store",
                            ),
                        )
            log = tmp_path / "inputs.log"
            monkeypatch.setenv(pipeline_fakes.INPUTS_LOG_ENV, str(log))
            yield name, db
        finally:
            db.execute(
                psycopg.sql.SQL("DROP SCHEMA {} CASCADE").format(
                    psycopg.sql.Identifier(name)
                )
            )


@pytest.fixture
def checkpoint(fake_checkpoint, monkeypatch):
    monkeypatch.setattr(
        "synthefy_nori.NoriRegressor", pipeline_fakes.InputHashRegressor
    )
    return fake_checkpoint.path


def _connect(client, schema):
    return client.connect(
        engine="postgresql",
        credential=EnvironmentCredential(variable="NORI_REL_TEST_DATABASE_URL"),
        schema_name=schema,
        tables=["users", "orders"],
        time_columns={"orders": "event_time"},
    )


def _first_prediction(schema, features="noridfs", **options):
    with NoriRelClient(limits=LIMITS, device="cpu") as client:
        return client.predict(
            source=_connect(client, schema),
            query=QUERY,
            feature_strategy=features,
            anchor_times=ANCHORS,
            anchor_time=CUTOFF,
            observed_through=CUTOFF,
            entity_created_at="created_at",
            return_details=True,
            **options,
        )


def _child_prediction(path, checkpoint, schema, anchor_time, tmp_path):
    output = tmp_path / f"child-{uuid4().hex}.json"
    subprocess.run(
        [
            sys.executable,
            str(CHILD),
            str(path),
            str(checkpoint),
            schema,
            anchor_time.isoformat(),
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(CHILD.parent)},
    )
    return json.loads(output.read_text())


def _program(baseline_columns):
    sql = (
        "SELECT e.row_id, COUNT(o.id) AS web_orders FROM eval_table e "
        "LEFT JOIN orders o ON o.buyer_id = e.user_id AND o.channel = 'web' "
        "AND o.event_time < e.cutoff GROUP BY e.row_id"
    )
    return FeatureProgram((FeatureQuery("web_orders", sql),)), tuple(baseline_columns)


@pytest.mark.parametrize("kind", ["noridfs", "fastdfs", "agentic"])
def test_a_saved_pipeline_predicts_the_same_in_a_new_process(
    shop, checkpoint, tmp_path, kind
):
    schema, _ = shop
    first = _first_prediction(
        schema, features="fastdfs" if kind == "fastdfs" else "noridfs"
    )
    if kind == "agentic":
        program, baseline = _program(first["pipeline"].feature_names)
        with NoriRelClient(limits=LIMITS, device="cpu") as client:
            source = _connect(client, schema)
            record = program_record(
                program,
                query=QUERY,
                entity_created_at="created_at",
                schema=client.discover(source),
                baseline_columns=baseline,
            )
        first = _first_prediction(schema, feature_program=record)
    pipeline = first["pipeline"]
    path = pipeline.save(tmp_path / "pipeline.json")
    reference = first["predictions"]
    reference_inputs = pipeline_fakes.last_inputs_hash()

    child = _child_prediction(path, checkpoint, schema, CUTOFF, tmp_path)

    assert pipeline.manifest.features.kind == kind
    assert child["inputs"] == reference_inputs
    assert child["entity_id"] == [int(value) for value in reference["entity_id"]]
    assert child["prediction"] == reference["prediction"].tolist()
    strategy = {"noridfs": "noridfs", "fastdfs": "fastdfs", "agentic": "saved_program"}
    assert child["metadata"]["features"] == strategy[kind]
    assert (
        child["metadata"]["context_rows_sha256"]
        == (first["metadata"]["context_rows_sha256"])
    )
    assert (
        child["metadata"]["feature_matrix_sha256"]
        == (first["metadata"]["feature_matrix_sha256"])
    )
    assert child["metadata"]["pipeline_id"] == pipeline.pipeline_id
    if kind == "agentic":
        assert pipeline.feature_names[-1] == "web_orders__web_orders"


def _later_prediction(schema, pipeline, **options):
    with NoriRelClient(limits=LIMITS, device="cpu") as client:
        return client.predict(
            source=_connect(client, schema),
            pipeline=pipeline,
            as_of=options.pop("as_of", LATER),
            observed_through=options.pop("observed_through", LATER),
            return_details=True,
            **options,
        )


def test_a_later_run_builds_new_context_with_the_saved_rules(shop, checkpoint):
    schema, db = shop
    first = _first_prediction(schema)
    pipeline = first["pipeline"]
    db.execute(
        "INSERT INTO orders VALUES (9001, 3, '2026-01-10', 500, 'web'), "
        "(9002, 3, '2026-01-20', 700, 'web')"
    )

    later = _later_prediction(
        schema,
        pipeline,
        query="PREDICT SUM(orders.amount,0,30,days) FOR users.user_id IN (3, 5)",
    )

    assert later["predictions"]["entity_id"].tolist() == [3, 5]
    assert later["pipeline"] == pipeline
    assert later["metadata"]["features"] == "noridfs"
    # The anchors moved with as_of, so the context rows are not the first run's.
    assert (
        later["metadata"]["context_rows_sha256"]
        != first["metadata"]["context_rows_sha256"]
    )
    assert later["metadata"]["feature_matrix_sha256"]


def test_a_changed_history_does_not_fail_the_run(shop, checkpoint):
    schema, db = shop
    first = _first_prediction(schema)
    db.execute("INSERT INTO orders VALUES (9100, 1, '2025-10-15', 99, 'web')")

    again = _later_prediction(
        schema, first["pipeline"], as_of=CUTOFF, observed_through=CUTOFF
    )

    assert (
        again["metadata"]["context_rows_sha256"]
        != first["metadata"]["context_rows_sha256"]
    )


def test_a_past_run_can_be_reproduced_by_its_context_hash(shop, checkpoint):
    schema, db = shop
    first = _first_prediction(schema)
    expected = first["metadata"]["context_rows_sha256"]

    same = _later_prediction(
        schema,
        first["pipeline"],
        as_of=CUTOFF,
        observed_through=CUTOFF,
        expected_context_sha256=expected,
    )
    assert same["predictions"]["prediction"].tolist() == (
        first["predictions"]["prediction"].tolist()
    )
    with pytest.raises(PipelineContextError, match="not the expected"):
        _later_prediction(schema, first["pipeline"], expected_context_sha256=expected)
    db.execute("INSERT INTO orders VALUES (9100, 1, '2025-10-15', 99, 'web')")
    with pytest.raises(PipelineContextError, match="not the expected"):
        _later_prediction(
            schema,
            first["pipeline"],
            as_of=CUTOFF,
            observed_through=CUTOFF,
            expected_context_sha256=expected,
        )


def test_a_pipeline_run_reports_its_id_and_hashes_to_run_control(shop, checkpoint):
    schema, _ = shop
    first = _first_prediction(schema)
    pipeline = first["pipeline"]
    events = []

    with NoriRelClient(limits=LIMITS, device="cpu") as client:
        run = client.start_predict(
            source=_connect(client, schema),
            pipeline=pipeline,
            as_of=LATER,
            observed_through=LATER,
            on_event=events.append,
        )
        frame = run.result(timeout=120)

    summary = run.progress
    assert summary.status == "completed"
    assert summary.pipeline_id == pipeline.pipeline_id
    assert summary.strategy_used == "noridfs"
    assert summary.context_sha256 == frame.attrs["context_rows_sha256"]
    assert summary.features_sha256 == frame.attrs["feature_matrix_sha256"]
    assert frame.attrs["run"]["pipeline_id"] == pipeline.pipeline_id
    assert events[-1].data["pipeline_id"] == pipeline.pipeline_id


def test_a_pipeline_rejects_changed_schemas_tasks_and_arguments(shop, checkpoint):
    schema, db = shop
    pipeline = _first_prediction(schema)["pipeline"]

    def predict(**options):
        client_options = {"device": "cpu", **options.pop("client", {})}
        with NoriRelClient(limits=LIMITS, **client_options) as client:
            return client.predict(
                source=_connect(client, schema),
                pipeline=pipeline,
                as_of=CUTOFF,
                observed_through=CUTOFF,
                **options,
            )

    with pytest.raises(PipelineTaskError, match="only the FOR clause may change"):
        predict(query=QUERY.replace("SUM", "AVG"))
    with pytest.raises(ValueError, match="anchor_times, entity_created_at cannot"):
        predict(anchor_times=ANCHORS, entity_created_at="created_at")
    with pytest.raises(ValueError, match="anchor_time cannot be passed"):
        predict(anchor_time=CUTOFF)
    with pytest.raises(ValueError, match="metric cannot be passed"):
        predict(metric="wape")
    with pytest.raises(PipelineModelError, match="uses local Nori"):
        predict(client={"mode": "hosted", "api_key": "test-only", "device": None})
    with pytest.raises(PipelineModelError, match="model 'nori-30m'"):
        predict(client={"model": "nori-6m"})

    db.execute("ALTER TABLE users ADD COLUMN tier text")
    with pytest.raises(PipelineSchemaError, match=r"column users\.tier is new"):
        predict()


def test_a_tampered_checkpoint_is_refused(shop, checkpoint):
    schema, _ = shop
    pipeline = _first_prediction(schema)["pipeline"]
    checkpoint.write_bytes(b"other weights")

    with (
        NoriRelClient(limits=LIMITS, device="cpu") as client,
        pytest.raises(PipelineModelError, match="SHA-256"),
    ):
        client.predict(
            source=_connect(client, schema),
            pipeline=pipeline,
            as_of=CUTOFF,
            observed_through=CUTOFF,
        )
    assert (
        pipeline.manifest.model.sha256
        == hashlib.sha256(b"fake nori checkpoint").hexdigest()
    )
