import gc
import itertools
import threading
import time
import weakref
from collections import Counter

import pandas as pd
import pytest

from nori_rel.run import (
    FINAL_STATUSES,
    EventKind,
    FallbackReason,
    LocalRun,
    PredictionCancelled,
    PredictionFailed,
    RunContext,
    RunEvent,
    RunLimits,
    RunTimeout,
    Stage,
    new_run_id,
    progress_from_events,
)

#: Long enough for a worker thread to reach a point a test waits for.
WAIT_SECONDS = 5.0


def _event_kinds(events: list[RunEvent]) -> list[str]:
    return [event.data["kind"] for event in events]


def _command_outcomes(events: list[RunEvent]) -> list[tuple[str, str, str | None]]:
    """Return each command event's (command, status, rejection reason)."""
    return [
        (event.data["command"], event.data["status"], event.data.get("reason"))
        for event in events
        if event.data["kind"] == "command"
    ]


class GatedPrediction:
    """A prediction that reports two stages and waits for a test between them."""

    def __init__(self) -> None:
        self.reached_gate = threading.Event()
        self.open_gate = threading.Event()

    def __call__(self, run_context: RunContext) -> pd.DataFrame:
        with run_context.report_stage(Stage.MATERIALIZE):
            self.reached_gate.set()
            assert self.open_gate.wait(WAIT_SECONDS)
        with run_context.report_stage(Stage.PREDICT_BATCH):
            return pd.DataFrame({"entity_id": [7], "prediction": [3.5]})


def _execution_options(**overrides) -> dict:
    """Return LocalRun._execute_prediction options for a plain run, overridden."""
    return {
        "strategy": "fastdfs",
        "max_trials": None,
        "summary_in_result_attrs": True,
        "reraise_failure": False,
    } | overrides


def _start_run(prediction, *, on_event=None, limits=None, **overrides) -> LocalRun:
    """Execute prediction as a new run on a daemon thread, as an executor does."""
    run = LocalRun(new_run_id(), limits=limits, on_event=on_event)
    threading.Thread(
        target=run._execute_prediction,
        args=(prediction,),
        kwargs=_execution_options(**overrides),
        daemon=True,
    ).start()
    return run


def _run_here(prediction, *, on_event=None, limits=None, **overrides) -> LocalRun:
    """Execute prediction as a new run on this thread, re-raising a failure."""
    run = LocalRun(new_run_id(), limits=limits, on_event=on_event)
    run._execute_prediction(
        prediction, **_execution_options(reraise_failure=True, **overrides)
    )
    return run


def test_a_started_run_streams_events_and_returns_its_frame() -> None:
    prediction = GatedPrediction()
    received: list[RunEvent] = []
    run = _start_run(prediction, strategy="fastdfs", on_event=received.append)

    assert prediction.reached_gate.wait(WAIT_SECONDS)
    assert (run.progress.status, run.progress.stage) == ("running", "materialize")
    with pytest.raises(RunTimeout) as timeout:
        run.result(timeout=0.01)
    assert timeout.value.progress.status == "running"
    prediction.open_gate.set()
    frame = run.result(timeout=WAIT_SECONDS)

    assert frame["prediction"].tolist() == [3.5]
    assert frame.attrs["run"] == run.progress.model_dump(mode="json")
    assert frame.attrs["run"]["strategy_used"] == "fastdfs"
    assert _event_kinds(received) == [
        "status_changed",
        "stage_started",
        "stage_finished",
        "stage_started",
        "stage_finished",
        "status_changed",
    ]
    assert received[0].message == "Run started: fastdfs search."
    assert received[-1].data == {
        "kind": "status_changed",
        "from": "running",
        "to": "completed",
        "strategy_used": "fastdfs",
        "search_interrupted": False,
        "fallback_reason": None,
        "pipeline_id": None,
    }
    assert list(run.events()) == received
    assert list(run.events(after=4)) == received[4:]
    assert list(run.events(after=-3)) == received
    assert run.status().status == "completed"


def _read_events_on_a_thread(run) -> tuple[list[RunEvent], threading.Thread]:
    """Start a thread that collects ``run.events()``; join it with a timeout."""
    collected: list[RunEvent] = []
    reader = threading.Thread(
        target=lambda: collected.extend(run.events()), daemon=True
    )
    reader.start()
    return collected, reader


def _wait_until(condition, seconds: float = WAIT_SECONDS) -> bool:
    """Poll ``condition`` until it holds or ``seconds`` pass; return whether it held."""
    deadline = time.monotonic() + seconds
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.005)
    return True


def test_events_can_be_read_while_the_run_is_going() -> None:
    prediction = GatedPrediction()
    run = _start_run(prediction, strategy="agentic", max_trials=3)
    collected, reader = _read_events_on_a_thread(run)

    assert prediction.reached_gate.wait(WAIT_SECONDS)
    assert _wait_until(lambda: len(collected) == 2)
    assert collected[0].message == "Run started: agentic search, up to 3 trials."
    assert reader.is_alive()
    prediction.open_gate.set()
    reader.join(WAIT_SECONDS)

    assert not reader.is_alive()
    assert [event.event_id for event in collected] == [1, 2, 3, 4, 5, 6]


def test_on_event_sees_progress_that_includes_the_event() -> None:
    progress_event_ids: list[tuple[int, int]] = []

    def record_progress(event: RunEvent) -> None:
        progress_event_ids.append((event.event_id, run.progress.last_event_id))

    run = LocalRun("run_1", on_event=record_progress)
    run._execute_prediction(
        lambda run_context: "done", **_execution_options(reraise_failure=True)
    )

    assert progress_event_ids == [(1, 1), (2, 2)]


def test_a_failure_is_redacted_and_raised_from_the_original() -> None:
    def failing_prediction(run_context: RunContext) -> pd.DataFrame:
        with run_context.report_stage(Stage.PREDICT_BATCH):
            raise ConnectionError("postgresql://reader:hunter2@db:5432 refused")

    run = _start_run(failing_prediction, strategy="fastdfs")

    with pytest.raises(PredictionFailed) as failure:
        run.result(timeout=WAIT_SECONDS)
    assert isinstance(failure.value.__cause__, ConnectionError)
    assert failure.value.error_kind == "connection_error"
    assert "hunter2" not in str(failure.value)
    assert failure.value.progress.stage == "predict_batch"
    assert run.status().error == "postgresql://***@db:5432 refused"
    assert run.progress.strategy_used is None
    assert _event_kinds(list(run.events()))[-2:] == ["error", "status_changed"]


def test_a_failed_background_run_releases_its_frames() -> None:
    class LargeBuffer:
        pass

    buffer_references = []

    def failing_prediction(run_context: RunContext) -> None:
        buffer = LargeBuffer()
        buffer_references.append(weakref.ref(buffer))
        raise OSError("disk full")

    run = _start_run(failing_prediction, strategy="fastdfs")
    with pytest.raises(PredictionFailed) as failure:
        run.result(timeout=WAIT_SECONDS)
    gc.collect()

    assert failure.value.error_kind == "os_error"
    assert buffer_references[0]() is None


def test_a_failure_can_be_reraised_unchanged() -> None:
    def failing_prediction(run_context: RunContext) -> pd.DataFrame:
        raise ValueError("bad request")

    with pytest.raises(ValueError, match="bad request"):
        _run_here(failing_prediction)


def test_a_run_that_cannot_start_still_ends() -> None:
    run = _start_run(
        lambda run_context: "done", strategy="fastdfs", max_trials=object()
    )

    with pytest.raises(PredictionFailed) as failure:
        run.result(timeout=WAIT_SECONDS)
    assert failure.value.progress.status == "failed"


def test_an_interrupted_run_ends_cancelled_and_reraises() -> None:
    def interrupted_prediction(run_context: RunContext) -> None:
        raise KeyboardInterrupt

    events: list[RunEvent] = []
    with pytest.raises(KeyboardInterrupt):
        _run_here(interrupted_prediction, on_event=events.append)

    assert events[-1].data["to"] == "cancelled"


def test_a_cancelled_run_raises_and_a_later_command_gets_a_rejected_receipt() -> None:
    prediction = GatedPrediction()
    run = _start_run(prediction, strategy="fastdfs")
    assert prediction.reached_gate.wait(WAIT_SECONDS)

    assert run.cancel().command_id.startswith("cmd_")
    run._queue_command_for_next_boundary("pause")
    prediction.open_gate.set()

    with pytest.raises(PredictionCancelled):
        run.result(timeout=WAIT_SECONDS)
    events_at_the_end = list(run.events())
    receipt = run._queue_command_for_next_boundary("resume")

    assert (receipt.status, receipt.reason) == ("rejected", "run_finished")
    assert run.progress.status == "cancelled"
    assert list(run.events()) == events_at_the_end
    assert events_at_the_end[-1].data["to"] == "cancelled"
    assert _command_outcomes(events_at_the_end) == [
        ("cancel", "applied", None),
        ("pause", "rejected", "run_finished"),
    ]


def test_a_second_cancel_during_cleanup_is_rejected_before_the_final_status() -> None:
    cleaning_up = threading.Event()
    second_cancel_sent = threading.Event()

    def run_until_cancelled(run_context: RunContext) -> None:
        try:
            while True:
                run_context.apply_pending_commands()
                time.sleep(0.001)
        finally:
            cleaning_up.set()
            second_cancel_sent.wait(WAIT_SECONDS)

    run = _start_run(run_until_cancelled, strategy="agentic")
    run._queue_command_for_next_boundary("cancel")
    assert cleaning_up.wait(WAIT_SECONDS)
    second_cancel = run._queue_command_for_next_boundary("cancel")
    second_cancel_sent.set()

    with pytest.raises(PredictionCancelled):
        run.result(timeout=WAIT_SECONDS)
    events = list(run.events())
    assert second_cancel.status == "accepted"
    assert _command_outcomes(events) == [
        ("cancel", "applied", None),
        ("cancel", "rejected", "no_op"),
    ]
    assert events[-2].data["command_id"] == second_cancel.command_id
    assert events[-1].data["to"] == "cancelled"


def test_a_repeated_stop_found_at_the_end_is_rejected_as_a_no_op() -> None:
    run = LocalRun("run_1")

    def stop_again_after_the_stop_applied(run_context: RunContext) -> str:
        run_context.apply_pending_commands()
        run.stop()
        run.cancel()
        return "done"

    run.stop()
    run._execute_prediction(stop_again_after_the_stop_applied, **_execution_options())

    events = list(run.events())
    assert _command_outcomes(events) == [
        ("stop", "applied", None),
        ("stop", "rejected", "no_op"),
        ("cancel", "rejected", "run_finished"),
    ]
    assert events[-1].data["to"] == "completed"


def test_a_command_the_run_never_applied_is_rejected_before_the_final_status() -> None:
    prediction = GatedPrediction()
    run = _start_run(prediction, strategy="fastdfs")
    assert prediction.reached_gate.wait(WAIT_SECONDS)
    receipts = []

    def stop_during_the_last_stage(event: RunEvent) -> None:
        if event.data == {"kind": "stage_started", "stage": "predict_batch"}:
            receipts.append(run.stop())

    run._on_event = stop_during_the_last_stage
    prediction.open_gate.set()
    frame = run.result(timeout=WAIT_SECONDS)

    events = list(run.events())
    assert [receipt.status for receipt in receipts] == ["accepted"]
    assert _command_outcomes(events) == [("stop", "rejected", "run_finished")]
    assert events[-2].data["command_id"] == receipts[0].command_id
    assert events[-1].data["to"] == "completed"
    assert run.progress.last_event_id == events[-1].event_id
    assert frame.attrs["run"]["last_event_id"] == events[-1].event_id
    assert run.progress == progress_from_events(run.run_id, events)


def test_a_command_after_the_end_gets_a_rejected_receipt_and_no_event() -> None:
    received: list[RunEvent] = []
    run = _run_here(lambda run_context: "done", on_event=received.append)
    events_at_the_end = list(run.events())

    receipts = [
        run._queue_command_for_next_boundary("cancel"),
        run._queue_command_for_next_boundary("stop"),
        run._queue_command_for_next_boundary("pause"),
        run._queue_command_for_next_boundary("resume"),
        run._queue_command_for_next_boundary("steer", "Use order history."),
    ]

    assert [(receipt.status, receipt.reason) for receipt in receipts] == [
        ("rejected", "run_finished")
    ] * 5
    assert {receipt.run_id for receipt in receipts} == {run.run_id}
    assert len({receipt.command_id for receipt in receipts}) == 5
    assert list(run.events()) == events_at_the_end == received
    assert events_at_the_end[-1].data["to"] == "completed"
    assert run.progress.last_event_id == events_at_the_end[-1].event_id
    with pytest.raises(ValueError, match="blank"):
        run._queue_command_for_next_boundary("steer", "  ")


def test_a_command_sent_from_the_callback_of_the_final_status_is_rejected() -> None:
    receipts = []

    def send_commands_around_the_end(event: RunEvent) -> None:
        if event.data.get("to") == "completed":
            receipts.append(run._queue_command_for_next_boundary("stop"))

    run = LocalRun("run_1", on_event=send_commands_around_the_end)
    executing = threading.Thread(
        target=run._execute_prediction,
        args=(lambda run_context: "done",),
        kwargs=_execution_options(),
        daemon=True,
    )
    executing.start()
    executing.join(WAIT_SECONDS)

    assert not executing.is_alive(), "sending a command from on_event deadlocked"
    assert run.result() == "done"
    assert [(receipt.status, receipt.reason) for receipt in receipts] == [
        ("rejected", "run_finished")
    ]
    events = list(run.events())
    assert _command_outcomes(events) == []
    assert events[-1].data["to"] == "completed"


def test_a_command_sent_while_the_run_rejects_the_queued_ones_is_rejected() -> None:
    receipts = []

    def send_a_command_after_a_rejection(event: RunEvent) -> None:
        if event.data.get("command") == "stop":
            # The inbox is already closed, and this callback runs on the worker
            # thread inside the event lock.
            receipts.append(run._queue_command_for_next_boundary("pause"))

    def stop_during_the_last_stage(run_context: RunContext) -> str:
        run._queue_command_for_next_boundary("stop")
        return "done"

    run = LocalRun("run_1", on_event=send_a_command_after_a_rejection)
    executing = threading.Thread(
        target=run._execute_prediction,
        args=(stop_during_the_last_stage,),
        kwargs=_execution_options(),
        daemon=True,
    )
    executing.start()
    executing.join(WAIT_SECONDS)

    assert not executing.is_alive(), "sending a command from on_event deadlocked"
    assert [receipt.status for receipt in receipts] == ["rejected"]
    events = list(run.events())
    assert _command_outcomes(events) == [("stop", "rejected", "run_finished")]
    assert events[-1].data["to"] == "completed"


def test_a_failed_run_rejects_its_queued_commands_before_the_final_status() -> None:
    def failing_prediction(run_context: RunContext) -> None:
        run._queue_command_for_next_boundary("pause")
        raise OSError("disk full")

    run = LocalRun("run_1")
    run._execute_prediction(failing_prediction, **_execution_options())

    events = list(run.events())
    assert _event_kinds(events)[-3:] == ["error", "command", "status_changed"]
    assert _command_outcomes(events) == [("pause", "rejected", "run_finished")]
    assert events[-1].data["to"] == "failed"


def _start_repeating_command_senders(
    run: LocalRun,
    commands: tuple[str, ...],
    senders_should_stop: threading.Event,
    receipts: list,
) -> list[threading.Thread]:
    """Start a thread per command that sends it repeatedly until told to stop."""

    def send_repeatedly(command: str) -> None:
        steering_text = "Use order history." if command == "steer" else None
        while not senders_should_stop.is_set():
            receipts.append(
                run._queue_command_for_next_boundary(command, steering_text)
            )
            time.sleep(0.0002)

    threads = [
        threading.Thread(target=send_repeatedly, args=(command,), daemon=True)
        for command in commands
    ]
    for thread in threads:
        thread.start()
    return threads


def test_commands_racing_the_end_never_follow_the_final_status_or_get_lost() -> None:
    def prediction(run_context: RunContext) -> str:
        for stage in (Stage.MATERIALIZE, Stage.PREDICT_BATCH):
            with run_context.report_stage(stage):
                time.sleep(0.001)
        time.sleep(0.002)  # no boundary is left: commands now go unapplied
        return "done"

    total_accepted = total_rejected = 0
    for run_number in range(100):
        # A cancel ends some runs early; every run still ends under spam.
        commands = ("stop", "pause", "resume", "steer")
        if run_number % 2:
            commands += ("cancel",)
        streamed: list[RunEvent] = []
        run = LocalRun(new_run_id(), on_event=streamed.append)
        senders_should_stop = threading.Event()
        receipts: list = []
        spammers = _start_repeating_command_senders(
            run, commands, senders_should_stop, receipts
        )
        collected, reader = _read_events_on_a_thread(run)
        executing = threading.Thread(
            target=run._execute_prediction,
            args=(prediction,),
            kwargs=_execution_options(),
            daemon=True,
        )
        executing.start()

        assert run._finished.wait(WAIT_SECONDS), f"run {run_number} did not end"
        receipts.append(run._queue_command_for_next_boundary("stop"))
        senders_should_stop.set()
        for thread in (*spammers, reader, executing):
            thread.join(WAIT_SECONDS)
            assert not thread.is_alive(), f"run {run_number} left a thread running"

        events = list(run.events())
        assert streamed == collected == events
        assert events[-1].data["kind"] == "status_changed"
        assert events[-1].data["to"] in FINAL_STATUSES
        assert all(event.data.get("to") not in FINAL_STATUSES for event in events[:-1])
        outcome_counts = Counter(
            event.data["command_id"]
            for event in events
            if event.data["kind"] == "command"
        )
        accepted_ids = {
            receipt.command_id for receipt in receipts if receipt.status == "accepted"
        }
        rejected_ids = {
            receipt.command_id for receipt in receipts if receipt.status == "rejected"
        }
        assert set(outcome_counts) == accepted_ids
        assert set(outcome_counts.values()) <= {1}
        assert not accepted_ids & rejected_ids
        assert run.progress.last_event_id == events[-1].event_id
        total_accepted += len(accepted_ids)
        total_rejected += len(rejected_ids)

    assert total_accepted > 0
    assert total_rejected >= 100


def test_a_failing_event_callback_does_not_stop_the_run() -> None:
    def broken_callback(event: RunEvent) -> None:
        raise RuntimeError("display closed")

    run = _run_here(lambda run_context: "done", on_event=broken_callback)

    assert run.result() == "done"


def test_the_summary_is_put_in_attrs_only_when_asked() -> None:
    def prediction(run_context: RunContext) -> pd.DataFrame:
        return pd.DataFrame({"prediction": [1.0]})

    assert (
        "run" not in _run_here(prediction, summary_in_result_attrs=False).result().attrs
    )
    assert "run" in _run_here(prediction).result().attrs


def test_a_fallback_is_the_strategy_used() -> None:
    def falling_back(run_context: RunContext) -> str:
        run_context.emit_event(
            EventKind.FALLBACK_USED,
            reason=FallbackReason.NO_PROGRAM_CONFIRMED.value,
            to_strategy="noridfs",
        )
        return "done"

    run = _run_here(falling_back, strategy="agentic")

    final_event = list(run.events())[-1]
    assert final_event.data["strategy_used"] == "noridfs"
    assert final_event.data["fallback_reason"] == "no_program_confirmed"
    assert run.progress.strategy_used == "noridfs"


def test_a_reader_can_resume_mid_run_and_two_readers_see_the_same_events() -> None:
    prediction = GatedPrediction()
    run = _start_run(prediction, strategy="fastdfs")
    assert prediction.reached_gate.wait(WAIT_SECONDS)
    seen_before_gate = list(itertools.islice(run.events(), 2))
    read: dict[str, list[int]] = {}

    def read_events(name: str, after: int) -> None:
        read[name] = [event.event_id for event in run.events(after=after)]

    readers = [
        threading.Thread(target=read_events, args=("resumed", 2)),
        threading.Thread(target=read_events, args=("from_start", 0)),
    ]
    for reader in readers:
        reader.start()
    prediction.open_gate.set()
    for reader in readers:
        reader.join(WAIT_SECONDS)

    assert [event.event_id for event in seen_before_gate] == [1, 2]
    assert read["resumed"] == read["from_start"][2:]
    assert read["from_start"] == list(range(1, len(read["from_start"]) + 1))
    assert list(run.events(after=10_000)) == []


def test_an_event_after_the_final_status_is_dropped() -> None:
    run = _run_here(lambda run_context: "done")
    journal_length = len(list(run.events()))

    run._context.emit_event(EventKind.STAGE_STARTED, stage="materialize")

    events = list(run.events())
    assert len(events) == journal_length
    assert events[-1].data["to"] == "completed"


def test_a_fold_that_raises_still_journals_the_event_and_wakes_readers(
    monkeypatch,
) -> None:
    import nori_rel.run.local_run as local_run_module

    real_fold = local_run_module.apply_event_to_progress

    def fold_that_fails_on_stages(progress, event):
        if event.data["kind"] == "stage_started":
            raise ValueError("bad stage event")
        return real_fold(progress, event)

    monkeypatch.setattr(
        local_run_module, "apply_event_to_progress", fold_that_fails_on_stages
    )

    def prediction(run_context: RunContext) -> str:
        with run_context.report_stage(Stage.MATERIALIZE):
            return "done"

    run = _start_run(prediction, strategy="fastdfs")

    assert run.result(timeout=WAIT_SECONDS) == "done"
    assert "stage_started" in _event_kinds(list(run.events()))
    assert run.progress.status == "completed"


def test_result_waits_briefly_for_a_run_whose_final_status_is_out() -> None:
    final_status_seen = threading.Event()
    release_worker = threading.Event()

    def hold_after_final_status(event: RunEvent) -> None:
        if event.data.get("to") == "completed":
            final_status_seen.set()
            release_worker.wait(WAIT_SECONDS)

    run = _start_run(
        lambda run_context: "done",
        on_event=hold_after_final_status,
        strategy="fastdfs",
    )
    assert final_status_seen.wait(WAIT_SECONDS)
    outcome: list[object] = []
    reader = threading.Thread(target=lambda: outcome.append(run.result(timeout=0)))
    reader.start()
    release_worker.set()
    reader.join(WAIT_SECONDS)

    assert outcome == ["done"]


def test_on_event_cannot_wait_for_its_own_runs_result() -> None:
    errors: list[BaseException] = []
    run = LocalRun(new_run_id())

    def wait_from_callback(event: RunEvent) -> None:
        if event.data.get("to") == "completed":
            try:
                run.result(timeout=0)
            except BaseException as error:
                errors.append(error)

    run._on_event = wait_from_callback
    run._execute_prediction(lambda run_context: "done", **_execution_options())

    assert run.result(timeout=WAIT_SECONDS) == "done"
    assert [type(error) for error in errors] == [RuntimeError]


def test_the_start_event_reports_the_trial_cap_and_the_spend_limit() -> None:
    run = _run_here(
        lambda run_context: "done",
        strategy="agentic",
        max_trials=3,
        limits=RunLimits(max_spend_usd=0.5),
    )

    start = next(iter(run.events()))
    assert (start.data["max_trials"], start.data["max_spend_usd"]) == (3, 0.5)
    assert (run.progress.max_trials, run.progress.max_spend_usd) == (3, 0.5)


def test_a_run_without_a_spend_limit_reports_none() -> None:
    run = _run_here(lambda run_context: "done")

    assert next(iter(run.events())).data["max_spend_usd"] is None
    assert run.progress.max_spend_usd is None


def test_a_paused_run_waits_at_its_next_boundary_until_resumed() -> None:
    prediction = GatedPrediction()
    run = _start_run(prediction)
    assert prediction.reached_gate.wait(WAIT_SECONDS)

    run.pause()
    prediction.open_gate.set()
    assert _wait_until(lambda: run.progress.status == "paused")
    with pytest.raises(RunTimeout):
        run.result(timeout=0.05)
    run.resume()

    assert run.result(timeout=WAIT_SECONDS)["prediction"].tolist() == [3.5]
    statuses = [event.data["to"] for event in run.events() if "to" in event.data]
    assert statuses == ["running", "paused", "running", "completed"]
    assert run.events() is not None and run.status().message == "Run completed."


def test_a_stop_ends_a_pause_and_the_run_completes_without_a_resume() -> None:
    prediction = GatedPrediction()
    run = _start_run(prediction)
    assert prediction.reached_gate.wait(WAIT_SECONDS)

    run.pause()
    prediction.open_gate.set()
    assert _wait_until(lambda: run.progress.status == "paused")
    run.stop()

    assert run.result(timeout=WAIT_SECONDS)["prediction"].tolist() == [3.5]
    events = list(run.events())
    statuses = [event.data["to"] for event in events if "to" in event.data]
    assert statuses == ["running", "paused", "running", "completed"]
    assert _command_outcomes(events) == [
        ("pause", "applied", None),
        ("stop", "applied", None),
    ]


def test_a_run_without_an_agent_search_rejects_steering() -> None:
    prediction = GatedPrediction()
    run = _start_run(prediction)
    assert prediction.reached_gate.wait(WAIT_SECONDS)

    receipt = run.steer("Use order history.")
    prediction.open_gate.set()
    run.result(timeout=WAIT_SECONDS)

    command_events = [e for e in run.events() if e.data["kind"] == "command"]
    assert [
        (e.data["command_id"], e.data["status"], e.data["reason"])
        for e in command_events
    ] == [(receipt.command_id, "rejected", "not_applicable")]


class GatedAgentSearchPrediction(GatedPrediction):
    """A gated prediction that sets up its run the way an agent search does."""

    def __call__(self, run_context: RunContext) -> pd.DataFrame:
        run_context.steering_enabled = True
        return super().__call__(run_context)


def test_an_undelivered_steer_is_rejected_before_the_final_status() -> None:
    prediction = GatedAgentSearchPrediction()
    run = _start_run(prediction, strategy="agentic")
    assert prediction.reached_gate.wait(WAIT_SECONDS)

    receipt = run.steer("Use order history.")
    prediction.open_gate.set()
    run.result(timeout=WAIT_SECONDS)

    events = list(run.events())
    command_events = [e for e in events if e.data["kind"] == "command"]
    assert [
        (e.data["command_id"], e.data["status"], e.data["reason"])
        for e in command_events
    ] == [(receipt.command_id, "rejected", "run_finished")]
    assert "text" not in command_events[0].data
    assert events[-1].data["kind"] == "status_changed"
    assert events[-1].data["to"] == "completed"
    assert events.index(command_events[0]) == len(events) - 2


class ThreeStagePrediction:
    """A prediction that reports three stages and waits for a test in the first two."""

    def __init__(self) -> None:
        self.in_first_stage = threading.Event()
        self.leave_first_stage = threading.Event()
        self.in_second_stage = threading.Event()
        self.leave_second_stage = threading.Event()

    def __call__(self, run_context: RunContext) -> pd.DataFrame:
        run_context.steering_enabled = True
        with run_context.report_stage(Stage.MATERIALIZE):
            self.in_first_stage.set()
            assert self.leave_first_stage.wait(WAIT_SECONDS)
        with run_context.report_stage(Stage.PLAN_SEARCH):
            self.in_second_stage.set()
            assert self.leave_second_stage.wait(WAIT_SECONDS)
        with run_context.report_stage(Stage.PREDICT_BATCH):
            return pd.DataFrame({"entity_id": [7], "prediction": [3.5]})


def test_a_cancel_rejects_a_held_steer_as_run_finished() -> None:
    prediction = ThreeStagePrediction()
    run = _start_run(prediction, strategy="agentic")
    assert prediction.in_first_stage.wait(WAIT_SECONDS)
    steer_receipt = run.steer("Use order history.")
    prediction.leave_first_stage.set()
    # The steer was taken at this stage's boundary and is held from here on.
    assert prediction.in_second_stage.wait(WAIT_SECONDS)

    cancel_receipt = run.cancel()
    prediction.leave_second_stage.set()
    with pytest.raises(PredictionCancelled):
        run.result(timeout=WAIT_SECONDS)

    events = list(run.events())
    command_events = [e for e in events if e.data["kind"] == "command"]
    assert [
        (e.data["command_id"], e.data["status"], e.data.get("reason"))
        for e in command_events
    ] == [
        (cancel_receipt.command_id, "applied", None),
        (steer_receipt.command_id, "rejected", "run_finished"),
    ]
    assert events[-1].data["to"] == "cancelled"
    assert events.index(command_events[-1]) == len(events) - 2


def test_a_pipeline_run_reports_its_id_and_audit_hashes_in_the_summary() -> None:
    def prediction(run_context: RunContext) -> pd.DataFrame:
        run_context.report_pipeline(
            "pipe_0123456789abcdef", context_sha256="a" * 64, features_sha256="b" * 64
        )
        return pd.DataFrame({"entity_id": [7], "prediction": [3.5]})

    received: list[RunEvent] = []
    run = _run_here(prediction, strategy="noridfs", on_event=received.append)
    frame = run.result()

    assert received[-1].data["pipeline_id"] == "pipe_0123456789abcdef"
    summary = run.progress
    assert (summary.pipeline_id, summary.context_sha256, summary.features_sha256) == (
        "pipe_0123456789abcdef",
        "a" * 64,
        "b" * 64,
    )
    assert frame.attrs["run"]["pipeline_id"] == "pipe_0123456789abcdef"
