import copy
import pickle
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from nori_rel.run import (
    EVENT_TYPE_BY_KIND,
    EventKind,
    PredictionCancelled,
    PredictionFailed,
    RunEvent,
    RunProgress,
    RunStatusResponse,
    RunTimeout,
    apply_event_to_progress,
    progress_from_events,
    render,
)

START = datetime(2026, 10, 6, 12, 0, tzinfo=UTC)


class EventLog:
    """Build a run's events with increasing ids and caller-chosen times."""

    def __init__(self, run_id: str = "run_1") -> None:
        self.run_id = run_id
        self.events: list[RunEvent] = []

    def add(self, at_second: float, kind: EventKind, **data) -> RunEvent:
        event = RunEvent(
            run_id=self.run_id,
            event_id=len(self.events) + 1,
            timestamp=START + timedelta(seconds=at_second),
            type=EVENT_TYPE_BY_KIND[kind],
            message=render(kind, data),
            data={"kind": kind.value, **data},
        )
        self.events.append(event)
        return event

    def progress(self) -> RunProgress:
        return progress_from_events(self.run_id, self.events)


def _feature_run() -> EventLog:
    log = EventLog()
    log.add(
        0,
        EventKind.STATUS_CHANGED,
        **{"from": "queued", "to": "running"},
        strategy="fastdfs",
    )
    log.add(1, EventKind.STAGE_STARTED, stage="plan_search")
    log.add(9, EventKind.STAGE_FINISHED, stage="plan_search", seconds=8.0)
    return log


def test_runtime_counts_up_to_every_event_and_never_backwards() -> None:
    log = EventLog()
    log.add(
        0,
        EventKind.STATUS_CHANGED,
        **{"from": "queued", "to": "running"},
        strategy="dfs",
    )
    log.add(2, EventKind.STAGE_STARTED, stage="materialize")
    assert log.progress().active_runtime_seconds == 2.0

    log.add(1, EventKind.STAGE_FINISHED, stage="materialize", seconds=1.0)
    log.add(5, EventKind.STAGE_STARTED, stage="plan_search")

    assert log.progress().active_runtime_seconds == 5.0


def test_out_of_order_ids_are_ignored() -> None:
    log = _feature_run()
    late = log.events[-2]
    progress = log.progress()

    assert apply_event_to_progress(progress, late) == progress


def test_a_nested_stage_returns_to_its_outer_stage() -> None:
    log = _feature_run()
    log.add(20, EventKind.STAGE_STARTED, stage="plan_search")
    log.add(21, EventKind.STAGE_STARTED, stage="predict_batch")
    log.add(
        22,
        EventKind.STAGE_FINISHED,
        stage="predict_batch",
        seconds=1.0,
        outer_stage="plan_search",
    )

    assert log.progress().stage == "plan_search"


def test_replayed_events_change_nothing() -> None:
    log = _feature_run()
    progress = log.progress()

    for event in log.events:
        assert apply_event_to_progress(progress, event) == progress


def test_events_for_another_run_are_rejected() -> None:
    event = EventLog("run_2").add(0, EventKind.STAGE_STARTED, stage="materialize")

    with pytest.raises(ValueError, match="run_2"):
        apply_event_to_progress(RunProgress(run_id="run_1"), event)


def test_failed_run_reports_its_error() -> None:
    log = EventLog()
    log.add(
        0,
        EventKind.STATUS_CHANGED,
        **{"from": "queued", "to": "running"},
        strategy="dfs",
    )
    log.add(
        1,
        EventKind.ERROR,
        error_kind="permission_denied",
        message="Permission denied when reading sales.orders.",
    )
    log.add(1, EventKind.STATUS_CHANGED, **{"from": "running", "to": "failed"})
    progress = log.progress()

    response = RunStatusResponse.from_progress(progress)

    assert response.status == "failed"
    assert response.error == "Permission denied when reading sales.orders."
    error = PredictionFailed("run failed", progress=progress)
    assert (error.run_id, error.error_kind) == ("run_1", "permission_denied")


def test_an_error_without_a_kind_and_a_failure_without_an_error() -> None:
    log = EventLog()
    log.add(
        0,
        EventKind.STATUS_CHANGED,
        **{"from": "queued", "to": "running"},
        strategy="dfs",
    )
    log.add(1, EventKind.ERROR, message="Database unavailable.")
    assert log.progress().error_kind is None

    bare = EventLog()
    bare.add(
        0,
        EventKind.STATUS_CHANGED,
        **{"from": "queued", "to": "running"},
        strategy="dfs",
    )
    bare.add(1, EventKind.STATUS_CHANGED, **{"from": "running", "to": "failed"})

    assert RunStatusResponse.from_progress(bare.progress()).error == "Run failed."


@pytest.mark.parametrize("error_type", [PredictionCancelled, PredictionFailed])
def test_run_errors_survive_pickle_and_copy(error_type) -> None:
    error = error_type("run ended", progress=_feature_run().progress())

    for clone in (pickle.loads(pickle.dumps(error)), copy.copy(error)):
        assert type(clone) is error_type
        assert (str(clone), clone.progress) == ("run ended", error.progress)


def test_status_response_requires_error_exactly_when_failed() -> None:
    running = RunStatusResponse.from_progress(_feature_run().progress())

    assert running.error is None
    with pytest.raises(ValidationError, match="error is required"):
        RunStatusResponse(run_id="run_1", status="failed")
    with pytest.raises(ValidationError, match="error is required"):
        RunStatusResponse(run_id="run_1", status="running", error="boom")


def test_progress_round_trips_through_json() -> None:
    progress = _feature_run().progress()

    assert RunProgress.model_validate_json(progress.model_dump_json()) == progress
    assert RunTimeout("still running", progress=progress).progress is progress
