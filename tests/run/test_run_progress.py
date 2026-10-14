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


def _agent_run() -> EventLog:
    log = EventLog()
    log.add(
        0,
        EventKind.STATUS_CHANGED,
        **{"from": "queued", "to": "running"},
        strategy="agent",
        max_trials=24,
        max_spend_usd=5.0,
    )
    log.add(1, EventKind.STAGE_STARTED, stage="plan_search")
    log.add(5, EventKind.BASELINE_SCORED, metric="mae", score=3.8, feature_count=48)
    log.add(9, EventKind.STAGE_FINISHED, stage="plan_search", seconds=8.0)
    for trial, (score, accepted) in enumerate([(3.91, False), (3.74, True)], 1):
        log.add(
            10 + trial,
            EventKind.TRIAL_SCORED,
            trial=trial,
            max_trials=24,
            metric="mae",
            score=score,
            best_score=3.74 if accepted else None,
            baseline_score=3.8,
            accepted=accepted,
        )
    log.add(13, EventKind.SPEND_UPDATED, spend_usd=0.42, llm_calls=8)
    return log


def test_progress_tracks_search_results_and_spend() -> None:
    progress = _agent_run().progress()

    assert progress.status == "running"
    assert progress.stage is None
    assert (progress.strategy, progress.metric) == ("agent", "mae")
    assert (progress.baseline_score, progress.best_score) == (3.8, 3.74)
    assert (progress.trials_attempted, progress.trials_accepted) == (2, 1)
    assert (progress.max_trials, progress.max_spend_usd) == (24, 5.0)
    assert progress.spend_usd == 0.42
    assert progress.message == "Agent spend: $0.42 (8 LLM calls)."
    assert progress.last_event_id == 7


def test_runtime_excludes_paused_time_and_summary_is_final() -> None:
    log = _agent_run()
    log.add(20, EventKind.STATUS_CHANGED, **{"from": "running", "to": "paused"})
    log.add(80, EventKind.STATUS_CHANGED, **{"from": "paused", "to": "running"})
    log.add(81, EventKind.LIMIT_REACHED, limit="max_spend_usd")
    log.add(
        82,
        EventKind.FALLBACK_USED,
        reason="stopped_before_confirmation",
        to_strategy="noridfs",
    )
    log.add(
        90,
        EventKind.STATUS_CHANGED,
        **{"from": "running", "to": "completed"},
        strategy_used="noridfs",
        search_interrupted=True,
        pipeline_id="pipe_1",
    )

    summary = log.progress()

    assert summary.status == "completed"
    assert summary.active_runtime_seconds == 30.0
    assert summary.running_since is None
    assert summary.search_interrupted
    assert summary.fallback_reason == "stopped_before_confirmation"
    assert (summary.strategy_used, summary.pipeline_id) == ("noridfs", "pipe_1")


def test_a_null_best_after_a_best_keeps_the_best() -> None:
    log = _agent_run()
    log.add(
        14,
        EventKind.TRIAL_SCORED,
        trial=3,
        metric="mae",
        score=3.95,
        best_score=None,
        accepted=False,
    )

    assert log.progress().best_score == 3.74


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


def test_only_search_stopped_marks_the_search_interrupted() -> None:
    log = _agent_run()
    log.add(20, EventKind.LIMIT_REACHED, limit="max_spend_usd")

    assert not log.progress().search_interrupted

    log.add(21, EventKind.SEARCH_STOPPED, trials=3)

    assert log.progress().search_interrupted


def test_the_final_status_carries_the_fallback_reason() -> None:
    log = _agent_run()
    log.add(
        20,
        EventKind.STATUS_CHANGED,
        **{"from": "running", "to": "completed"},
        fallback_reason="no_program_confirmed",
    )

    assert log.progress().fallback_reason == "no_program_confirmed"


def test_out_of_order_ids_are_ignored() -> None:
    log = _agent_run()
    late = log.events[-2]
    progress = log.progress()

    assert apply_event_to_progress(progress, late) == progress


def test_a_nested_stage_returns_to_its_outer_stage() -> None:
    log = _agent_run()
    log.add(20, EventKind.STAGE_STARTED, stage="plan_search")
    log.add(21, EventKind.STAGE_STARTED, stage="select_pipeline")
    log.add(
        22,
        EventKind.STAGE_FINISHED,
        stage="select_pipeline",
        seconds=1.0,
        outer_stage="plan_search",
    )

    assert log.progress().stage == "plan_search"


def test_replayed_events_change_nothing() -> None:
    log = _agent_run()
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
    error = error_type("run ended", progress=_agent_run().progress())

    for clone in (pickle.loads(pickle.dumps(error)), copy.copy(error)):
        assert type(clone) is error_type
        assert (str(clone), clone.progress) == ("run ended", error.progress)


def test_status_response_requires_error_exactly_when_failed() -> None:
    running = RunStatusResponse.from_progress(_agent_run().progress())

    assert running.error is None
    with pytest.raises(ValidationError, match="error is required"):
        RunStatusResponse(run_id="run_1", status="failed")
    with pytest.raises(ValidationError, match="error is required"):
        RunStatusResponse(run_id="run_1", status="running", error="boom")


def test_progress_round_trips_through_json() -> None:
    progress = _agent_run().progress()

    assert RunProgress.model_validate_json(progress.model_dump_json()) == progress
    assert RunTimeout("still running", progress=progress).progress is progress
