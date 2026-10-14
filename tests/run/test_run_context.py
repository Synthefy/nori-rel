import threading
import time

import pytest
from run_control_fakes import FakeClock

from nori_rel.run import (
    CommandInbox,
    EventKind,
    RunCancelled,
    RunContext,
    RunEvent,
    RunLimits,
    Stage,
    progress_from_events,
)


def _running_context(clock=None) -> tuple[RunContext, list[RunEvent]]:
    events: list[RunEvent] = []
    context = RunContext("run_1", event_sink=events.append, clock=clock or FakeClock())
    context.change_status("running", strategy="fastdfs")
    return context, events


def _commands(events: list[RunEvent]) -> list[tuple[str, str, str | None]]:
    return [
        (event.data["command"], event.data["status"], event.data.get("reason"))
        for event in events
        if event.data["kind"] == "command"
    ]


def test_events_are_numbered_typed_and_rendered() -> None:
    clock = FakeClock()
    context, events = _running_context(clock)

    with context.report_stage(Stage.MATERIALIZE):
        clock.advance(2.5)

    assert [event.event_id for event in events] == [1, 2, 3]
    assert [event.type for event in events] == ["status", "progress", "progress"]
    assert events[2].message == "Finished reading the database snapshot in 2.5 s."
    assert progress_from_events("run_1", events).status == "running"


def test_failed_stage_reports_no_finish() -> None:
    context, events = _running_context()

    with pytest.raises(RuntimeError), context.report_stage(Stage.PREDICT_BATCH):
        raise RuntimeError("inference failed")

    assert events[-1].data["kind"] == "stage_started"


def _pausable_context() -> tuple[RunContext, list[RunEvent], threading.Event]:
    """Return a running context, its events, and an event set once it pauses."""
    events: list[RunEvent] = []
    paused = threading.Event()

    def record(event: RunEvent) -> None:
        events.append(event)
        if event.data.get("to") == "paused":
            paused.set()

    context = RunContext("run_1", event_sink=record, clock=FakeClock())
    context.change_status("running", strategy="fastdfs")
    return context, events, paused


def _apply_commands_on_a_thread(
    context: RunContext,
) -> tuple[threading.Thread, list[BaseException]]:
    """Run the boundary on a worker thread; return it and what it raised."""
    raised: list[BaseException] = []

    def apply() -> None:
        try:
            context.apply_pending_commands()
        except BaseException as error:
            raised.append(error)

    thread = threading.Thread(target=apply, daemon=True)
    thread.start()
    return thread, raised


def _resolve_pause(context: RunContext, paused: threading.Event, command: str):
    """Pause the run on a worker, check it blocks, then send ``command``."""
    context.commands.submit_command("pause")
    thread, raised = _apply_commands_on_a_thread(context)
    assert paused.wait(timeout=5)
    thread.join(timeout=0.05)
    assert thread.is_alive(), "the paused boundary returned without a command"
    context.commands.submit_command(command)
    thread.join(timeout=5)
    assert not thread.is_alive()
    return raised


def _statuses(events: list[RunEvent]) -> list[str]:
    return [event.data["to"] for event in events if event.type == "status"]


def test_pause_blocks_until_resume() -> None:
    context, events, paused = _pausable_context()

    assert _resolve_pause(context, paused, "resume") == []
    assert context.status == "running"
    assert _statuses(events) == ["running", "paused", "running"]


def test_cancel_while_paused_raises() -> None:
    context, _, paused = _pausable_context()

    raised = _resolve_pause(context, paused, "cancel")

    assert [type(error) for error in raised] == [RunCancelled]


def test_cancel_applies_before_earlier_commands_in_its_batch() -> None:
    context, events = _running_context()
    context.commands.submit_command("pause")
    context.commands.submit_command("cancel")

    with pytest.raises(RunCancelled):
        context.apply_pending_commands()

    assert _commands(events) == [
        ("cancel", "applied", None),
        ("pause", "rejected", "run_finished"),
    ]
    assert _statuses(events) == ["running"]


def test_failing_event_sink_does_not_break_the_run() -> None:
    def broken_sink(event: RunEvent) -> None:
        raise OSError("stream closed")

    context = RunContext("run_1", event_sink=broken_sink)

    event = context.emit_event(EventKind.STAGE_STARTED, stage="materialize")

    assert event.event_id == 1


def test_events_from_several_threads_reach_the_sink_in_order() -> None:
    delivered_ids: list[int] = []

    def slow_sink(event: RunEvent) -> None:
        if event.event_id % 2:
            time.sleep(0.001)
        delivered_ids.append(event.event_id)

    context = RunContext("run_1", event_sink=slow_sink)

    def emit_stage_events() -> None:
        for _ in range(20):
            context.emit_event(EventKind.STAGE_STARTED, stage="materialize")

    threads = [threading.Thread(target=emit_stage_events) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert delivered_ids == list(range(1, 81))


def test_inbox_wait_times_out_without_commands() -> None:
    assert not CommandInbox("run_1").wait_for_command(timeout_seconds=0.01)


def test_cancel_keeps_raising_at_every_later_boundary() -> None:
    context, _ = _running_context()
    context.commands.submit_command("cancel")

    for _ in range(2):
        with pytest.raises(RunCancelled):
            context.apply_pending_commands()

    assert not issubclass(RunCancelled, Exception)


def _limited_context(limits: RunLimits) -> tuple[RunContext, list[RunEvent], FakeClock]:
    clock = FakeClock()
    events: list[RunEvent] = []
    context = RunContext("run_1", event_sink=events.append, limits=limits, clock=clock)
    context.change_status("running", strategy="fastdfs")
    return context, events, clock


def test_active_runtime_leaves_out_paused_time() -> None:
    clock = FakeClock()
    context, _ = _running_context(clock)
    clock.advance(5)
    context.change_status("paused")
    clock.advance(10)
    context.change_status("running")
    clock.advance(2)

    assert context.get_active_runtime_seconds() == 7


def test_a_wall_clock_step_does_not_change_durations() -> None:
    wall_clock = FakeClock()
    seconds: list[float] = [100.0]
    context = RunContext(
        "run_1",
        event_sink=lambda event: None,
        clock=wall_clock,
        seconds_clock=lambda: seconds[0],
    )
    context.change_status("running", strategy="fastdfs")
    wall_clock.advance(-3600)
    seconds[0] += 2

    assert context.get_active_runtime_seconds() == 2


def test_runtime_limit_cancels_at_a_boundary_and_stays_cancelled():
    context, events, clock = _limited_context(RunLimits(max_runtime_seconds=2))
    clock.advance(3)
    for _ in range(2):
        with pytest.raises(RunCancelled):
            context.apply_pending_commands()
    assert [event.data["kind"] for event in events].count("limit_reached") == 1


def test_research_commands_are_rejected_before_entering_the_inbox():
    from pydantic import ValidationError

    inbox = CommandInbox("run_1")
    for command in ("stop", "steer"):
        with pytest.raises(ValidationError):
            inbox.submit_command(command)
    assert inbox.take_all_commands() == []


def test_every_command_is_returned_by_the_close_or_rejected() -> None:
    inbox = CommandInbox("run_1")
    receipts: list = []
    start = threading.Barrier(5)

    def send_commands() -> None:
        start.wait()
        for _ in range(200):
            receipts.append(inbox.submit_command("pause"))

    senders = [threading.Thread(target=send_commands, daemon=True) for _ in range(4)]
    for sender in senders:
        sender.start()
    start.wait()
    time.sleep(0.002)
    queued = inbox.close_and_take_all_commands()
    for sender in senders:
        sender.join(5)

    assert len(receipts) == 800
    assert {item.command_id for item in queued} == {
        receipt.command_id for receipt in receipts if receipt.status == "accepted"
    }
    assert inbox.take_all_commands() == []


def test_closing_the_inbox_returns_queued_commands_and_rejects_later_ones():
    inbox = CommandInbox("run_1")
    accepted = inbox.submit_command("pause")
    inbox.submit_command("resume")
    queued = inbox.close_and_take_all_commands()
    rejected = inbox.submit_command("cancel")
    assert [item.command for item in queued] == ["pause", "resume"]
    assert queued[0].command_id == accepted.command_id
    assert (rejected.status, rejected.reason) == ("rejected", "run_finished")
    assert inbox.take_all_commands() == []
    assert inbox.close_and_take_all_commands() == []
