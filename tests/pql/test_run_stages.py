"""A PQL prediction reports its stages to the run it belongs to."""

from datetime import UTC, datetime

import pytest

from nori_rel import NoriRelClient
from nori_rel.run import RunContext, RunEvent, progress_from_events

QUERY = "PREDICT COUNT(orders.*,0,30,days) FOR users.user_id=99"
OPTIONS = dict(
    anchor_times=[datetime(2026, 1, 1, tzinfo=UTC)],
    anchor_time=datetime(2026, 2, 1, tzinfo=UTC),
    observed_through=datetime(2026, 2, 1, tzinfo=UTC),
    entity_created_at="created_at",
)


def _stage_events(events: list[RunEvent]) -> list[tuple[str, str]]:
    return [(event.data["kind"], event.data["stage"]) for event in events]


def test_prediction_reports_each_stage_in_order(runtime_source) -> None:
    runtime, database, _ = runtime_source
    events: list[RunEvent] = []
    run_context = RunContext("run_1", event_sink=events.append)

    frame = runtime.predict_pql(database, QUERY, **OPTIONS, run_context=run_context)

    assert frame["entity_id"].tolist() == [99]
    assert _stage_events(events) == [
        ("stage_started", "materialize"),
        ("stage_finished", "materialize"),
        ("stage_started", "plan_search"),
        ("stage_finished", "plan_search"),
        ("stage_started", "predict_batch"),
        ("stage_finished", "predict_batch"),
    ]
    assert events[1].message.startswith("Finished reading the database snapshot in")
    assert progress_from_events("run_1", events).stage is None


def test_a_failed_stage_is_left_open(runtime_source, monkeypatch) -> None:
    runtime, database, _ = runtime_source
    events: list[RunEvent] = []

    def failing_predict(*args, **kwargs):
        raise RuntimeError("inference failed")

    monkeypatch.setattr(runtime.predictor, "predict_arrays", failing_predict)
    with pytest.raises(RuntimeError, match="inference failed"):
        runtime.predict_pql(
            database,
            QUERY,
            **OPTIONS,
            run_context=RunContext("run_1", event_sink=events.append),
        )

    assert _stage_events(events)[-1] == ("stage_started", "predict_batch")
    assert progress_from_events("run_1", events).stage == "predict_batch"


def test_prediction_without_a_run_reports_nothing(runtime_source) -> None:
    runtime, database, _ = runtime_source

    frame = runtime.predict_pql(database, QUERY, **OPTIONS)

    assert frame["entity_id"].tolist() == [99]


def test_a_background_prediction_streams_its_stages(runtime_source) -> None:
    runtime, database, _ = runtime_source
    client = NoriRelClient(mode="hosted", api_key="test-only", _runtime=runtime)

    run = client.start_predict(source=database, query=QUERY, **OPTIONS)
    events = list(run.events())
    frame = run.result(timeout=30)

    assert frame["entity_id"].tolist() == [99]
    assert [event.event_id for event in events] == list(range(1, len(events) + 1))
    assert events[0].message == "Run started: fastdfs search."
    assert _stage_events(events[1:-1]) == [
        ("stage_started", "materialize"),
        ("stage_finished", "materialize"),
        ("stage_started", "plan_search"),
        ("stage_finished", "plan_search"),
        ("stage_started", "predict_batch"),
        ("stage_finished", "predict_batch"),
    ]
    assert frame.attrs["run"] == run.progress.model_dump(mode="json")
    assert run.progress.status == "completed"
