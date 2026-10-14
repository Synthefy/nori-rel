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
    context.change_status("running", strategy="agent")
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


def test_a_repeated_stop_is_rejected_as_a_no_op() -> None:
    context, events = _running_context()
    context.commands.submit_command("stop")
    context.commands.submit_command("stop")

    context.apply_pending_commands()

    assert context.stop_applied
    assert _commands(events) == [
        ("stop", "applied", None),
        ("stop", "rejected", "no_op"),
    ]


def test_a_steer_is_held_and_reported_applied_when_it_is_delivered() -> None:
    context, events = _running_context()
    context.steering_enabled = True
    receipt = context.commands.submit_command(
        "steer", steering_text="Use order history."
    )

    context.apply_pending_commands()

    assert _commands(events) == []
    assert context.deliver_held_steering_texts() == ["Use order history."]
    assert _commands(events) == [("steer", "applied", None)]
    assert events[-1].data["command_id"] == receipt.command_id
    assert events[-1].message == 'Steer applied: "Use order history."'
    assert context.deliver_held_steering_texts() == []
    assert len(_commands(events)) == 1


def test_held_steers_are_delivered_oldest_first_each_reported_applied() -> None:
    context, events = _running_context()
    context.steering_enabled = True
    first = context.commands.submit_command("steer", steering_text="First.")
    context.apply_pending_commands()
    second = context.commands.submit_command("steer", steering_text="Second.")
    context.apply_pending_commands()

    assert context.deliver_held_steering_texts() == ["First.", "Second."]

    assert [
        (event.data["command_id"], event.data["text"])
        for event in events
        if event.data["kind"] == "command"
    ] == [(first.command_id, "First."), (second.command_id, "Second.")]


def test_a_steer_applied_beside_a_stop_stays_held() -> None:
    context, events = _running_context()
    context.steering_enabled = True
    context.commands.submit_command("steer", steering_text="Use order history.")
    context.commands.submit_command("stop")

    context.apply_pending_commands()

    assert context.stop_applied
    assert _commands(events) == [("stop", "applied", None)]
    context.reject_undelivered_steering_texts("not_applicable")
    assert _commands(events) == [
        ("stop", "applied", None),
        ("steer", "rejected", "not_applicable"),
    ]
    assert context.deliver_held_steering_texts() == []


def test_undelivered_steers_are_rejected_without_their_text_and_steering_ends() -> None:
    context, events = _running_context()
    context.steering_enabled = True
    context.commands.submit_command("steer", steering_text="Use order history.")
    context.apply_pending_commands()

    context.reject_undelivered_steering_texts("not_applicable")
    context.commands.submit_command("steer", steering_text="Too late.")
    context.apply_pending_commands()

    assert not context.steering_enabled
    assert _commands(events) == [
        ("steer", "rejected", "not_applicable"),
        ("steer", "rejected", "not_applicable"),
    ]
    assert all("text" not in event.data for event in events if "reason" in event.data)
    assert context.deliver_held_steering_texts() == []


def test_a_cancel_rejects_held_steers_as_run_finished() -> None:
    context, events = _running_context()
    context.steering_enabled = True
    held = context.commands.submit_command("steer", steering_text="Held.")
    context.apply_pending_commands()
    context.commands.submit_command("cancel")
    batched = context.commands.submit_command("steer", steering_text="Batched.")

    with pytest.raises(RunCancelled):
        context.apply_pending_commands()

    assert _commands(events) == [
        ("cancel", "applied", None),
        ("steer", "rejected", "run_finished"),
        ("steer", "rejected", "run_finished"),
    ]
    assert [event.data["command_id"] for event in events[-2:]] == [
        held.command_id,
        batched.command_id,
    ]
    assert context.deliver_held_steering_texts() == []


def test_inapplicable_commands_are_rejected() -> None:
    context, events = _running_context()
    context.commands.submit_command("steer", steering_text="Use order history.")
    context.commands.submit_command("resume")

    context.apply_pending_commands()

    assert _commands(events) == [
        ("steer", "rejected", "not_applicable"),
        ("resume", "rejected", "no_op"),
    ]


def test_cancel_raises_and_rejects_later_commands() -> None:
    context, events = _running_context()
    receipt = context.commands.submit_command("cancel")
    context.commands.submit_command("stop")

    with pytest.raises(RunCancelled):
        context.apply_pending_commands()

    assert _commands(events) == [
        ("cancel", "applied", None),
        ("stop", "rejected", "run_finished"),
    ]
    assert events[-2].data["command_id"] == receipt.command_id


def _pausable_context() -> tuple[RunContext, list[RunEvent], threading.Event]:
    """Return a running context, its events, and an event set once it pauses."""
    events: list[RunEvent] = []
    paused = threading.Event()

    def record(event: RunEvent) -> None:
        events.append(event)
        if event.data.get("to") == "paused":
            paused.set()

    context = RunContext("run_1", event_sink=record, clock=FakeClock())
    context.change_status("running", strategy="agent")
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


def test_stop_while_paused_ends_the_pause() -> None:
    context, events, paused = _pausable_context()

    assert _resolve_pause(context, paused, "stop") == []
    assert context.stop_applied
    assert _statuses(events) == ["running", "paused", "running"]


def test_pause_after_stop_is_rejected() -> None:
    context, events = _running_context()
    context.commands.submit_command("stop")
    context.commands.submit_command("pause")

    context.apply_pending_commands()

    assert context.status == "running"
    assert _commands(events) == [
        ("stop", "applied", None),
        ("pause", "rejected", "not_applicable"),
    ]


@pytest.mark.parametrize(
    "before,expected",
    [
        ("pause", ("pause", "rejected", "run_finished")),
        ("cancel", ("cancel", "rejected", "no_op")),
    ],
)
def test_a_cancel_overrides_the_commands_in_its_batch(before, expected) -> None:
    context, events = _running_context()
    context.steering_enabled = True
    context.commands.submit_command("cancel")
    context.commands.submit_command(before)
    context.commands.submit_command("steer", steering_text="Use order history.")

    with pytest.raises(RunCancelled):
        context.apply_pending_commands()

    assert _commands(events) == [
        ("cancel", "applied", None),
        expected,
        ("steer", "rejected", "run_finished"),
    ]
    assert not context.has_held_steering()
    assert _statuses(events) == ["running"]
    with pytest.raises(RunCancelled):
        context.apply_pending_commands()


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


def test_unobserved_context_runs_without_a_listener() -> None:
    context = RunContext.unobserved()

    with context.report_stage(Stage.PLAN_SEARCH):
        pass

    assert not context.stop_applied


@pytest.mark.parametrize(
    "command,steering_text,match",
    [
        ("steer", None, "steer needs"),
        ("pause", "text", "steer needs"),
        ("steer", "  ", "blank"),
    ],
)
def test_inbox_validates_steering_text(command, steering_text, match) -> None:
    with pytest.raises(ValueError, match=match):
        CommandInbox("run_1").submit_command(command, steering_text)


def test_closing_the_inbox_returns_the_queued_commands_and_rejects_later_ones() -> None:
    inbox = CommandInbox("run_1")
    accepted = inbox.submit_command("stop")
    inbox.submit_command("steer", steering_text="Use order history.")

    queued = inbox.close_and_take_all_commands()
    rejected = inbox.submit_command("pause")

    assert [(item.command, item.steering_text) for item in queued] == [
        ("stop", None),
        ("steer", "Use order history."),
    ]
    assert queued[0].command_id == accepted.command_id
    assert (rejected.run_id, rejected.status, rejected.reason) == (
        "run_1",
        "rejected",
        "run_finished",
    )
    assert rejected.command_id not in {item.command_id for item in queued}
    assert inbox.take_all_commands() == []
    assert inbox.close_and_take_all_commands() == []
    assert not inbox.wait_for_command(timeout_seconds=0.01)


def test_a_closed_inbox_still_validates_steering_text() -> None:
    inbox = CommandInbox("run_1")
    inbox.close_and_take_all_commands()

    with pytest.raises(ValueError, match="blank"):
        inbox.submit_command("steer", steering_text="  ")
    with pytest.raises(ValueError, match="steer needs"):
        inbox.submit_command("pause", steering_text="text")


def test_every_command_is_returned_by_the_close_or_rejected() -> None:
    inbox = CommandInbox("run_1")
    receipts: list = []
    start = threading.Barrier(5)

    def send_commands() -> None:
        start.wait()
        for _ in range(200):
            receipts.append(inbox.submit_command("stop"))

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


def test_inbox_wait_times_out_without_commands() -> None:
    assert not CommandInbox("run_1").wait_for_command(timeout_seconds=0.01)


def test_a_repeated_cancel_or_stop_in_a_cancels_batch_is_a_no_op() -> None:
    context, events = _running_context()
    context.commands.submit_command("stop")
    context.apply_pending_commands()
    for command in ("cancel", "cancel", "stop", "pause"):
        context.commands.submit_command(command)

    with pytest.raises(RunCancelled):
        context.apply_pending_commands()

    assert _commands(events) == [
        ("stop", "applied", None),
        ("cancel", "applied", None),
        ("cancel", "rejected", "no_op"),
        ("stop", "rejected", "no_op"),
        ("pause", "rejected", "run_finished"),
    ]


def test_a_command_left_in_the_inbox_by_a_cancel_is_rejected_by_what_it_repeats() -> (
    None
):
    context, _ = _running_context()
    submitted = [
        context.commands.submit_command(command)
        for command in ("cancel", "stop", "pause")
    ]
    queued = context.commands.take_all_commands()

    assert [
        context.rejection_reason_once_the_run_ends(command) for command in queued
    ] == ["run_finished"] * 3
    with pytest.raises(RunCancelled):
        context._apply_commands(queued[:1])

    assert [
        context.rejection_reason_once_the_run_ends(command) for command in queued
    ] == ["no_op", "run_finished", "run_finished"]
    assert len({receipt.command_id for receipt in submitted}) == 3


def test_cancel_keeps_raising_at_every_later_boundary() -> None:
    context, _ = _running_context()
    context.commands.submit_command("cancel")

    for _ in range(2):
        with pytest.raises(RunCancelled):
            context.apply_pending_commands()

    assert not issubclass(RunCancelled, Exception)


def test_a_stop_ends_a_pause_and_paused_time_is_not_running_time() -> None:
    clock = FakeClock()
    context, _, paused = _pausable_context()
    context._clock = clock
    context._seconds_clock = lambda: clock().timestamp()
    context.change_status("running", strategy="agentic")
    clock.advance(5)

    def advance_while_paused(command: str) -> None:
        clock.advance(45)
        context.commands.submit_command(command)

    context.commands.submit_command("pause")
    thread, raised = _apply_commands_on_a_thread(context)
    assert paused.wait(timeout=5)
    advance_while_paused("stop")
    thread.join(timeout=5)
    clock.advance(2)

    assert raised == [] and (context.status, context.stop_applied) == (
        "running",
        True,
    )
    assert context.get_active_runtime_seconds() == 7


def test_a_resume_after_a_stop_released_a_pause_is_a_no_op() -> None:
    context, events, paused = _pausable_context()

    assert _resolve_pause(context, paused, "stop") == []
    context.commands.submit_command("resume")
    context.apply_pending_commands()

    assert _commands(events)[-1] == ("resume", "rejected", "no_op")
    assert context.status == "running"


def test_a_pause_and_a_stop_in_one_batch_end_running() -> None:
    context, events = _running_context()
    context.commands.submit_command("pause")
    context.commands.submit_command("stop")

    context.apply_pending_commands()

    assert context.status == "running"
    assert _commands(events) == [
        ("pause", "applied", None),
        ("stop", "applied", None),
    ]


def _limited_context(limits: RunLimits) -> tuple[RunContext, list[RunEvent], FakeClock]:
    clock = FakeClock()
    events: list[RunEvent] = []
    context = RunContext("run_1", event_sink=events.append, limits=limits, clock=clock)
    context.change_status("running", strategy="agentic")
    return context, events, clock


def test_a_pause_after_a_limit_stop_is_rejected_as_not_applicable() -> None:
    context, events, clock = _limited_context(RunLimits(max_runtime_seconds=3))
    clock.advance(4)
    context.apply_pending_commands(check_limits=True)
    context.commands.submit_command("pause")

    context.apply_pending_commands()

    assert context.stop_applied
    assert context.status == "running"
    assert _commands(events) == [("pause", "rejected", "not_applicable")]


def test_active_runtime_leaves_out_paused_time() -> None:
    clock = FakeClock()
    context, _ = _running_context(clock)
    clock.advance(5)
    context.change_status("paused")
    clock.advance(10)
    context.change_status("running")
    clock.advance(2)

    assert context.get_active_runtime_seconds() == 7


@pytest.mark.parametrize(
    "limits,spend_usd,reached_limit",
    [
        (RunLimits(max_runtime_seconds=3), None, "max_runtime_seconds"),
        (RunLimits(max_spend_usd=0.5), 0.6, "max_spend_usd"),
    ],
)
def test_a_reached_limit_stops_the_run_once(limits, spend_usd, reached_limit) -> None:
    context, events, clock = _limited_context(limits)
    clock.advance(4)
    if spend_usd is not None:
        context.record_spend(spend_usd)

    context.apply_pending_commands(check_limits=True)
    context.apply_pending_commands(check_limits=True)

    assert context.stop_applied
    limit_events = [event for event in events if event.data["kind"] == "limit_reached"]
    assert [event.data["limit"] for event in limit_events] == [reached_limit]
    # Only the backend's search_stopped marks the search interrupted.
    assert not progress_from_events("run_1", events).search_interrupted


def test_limits_are_checked_only_where_asked() -> None:
    context, events, clock = _limited_context(RunLimits(max_runtime_seconds=3))
    clock.advance(4)

    context.apply_pending_commands()
    with context.report_stage(Stage.PREDICT_BATCH):
        pass

    assert not context.stop_applied
    assert "limit_reached" not in [event.data["kind"] for event in events]


def test_a_wall_clock_step_does_not_change_durations() -> None:
    wall_clock = FakeClock()
    seconds: list[float] = [100.0]
    context = RunContext(
        "run_1",
        event_sink=lambda event: None,
        clock=wall_clock,
        seconds_clock=lambda: seconds[0],
    )
    context.change_status("running", strategy="agentic")
    wall_clock.advance(-3600)
    seconds[0] += 2

    assert context.get_active_runtime_seconds() == 2


def test_steering_text_over_4000_characters_is_refused_at_any_time() -> None:
    inbox = CommandInbox("run_1")
    longest_steering_text = "x" * 4_000

    accepted = inbox.submit_command("steer", longest_steering_text)
    with pytest.raises(ValueError, match="at most 4000 characters"):
        inbox.submit_command("steer", longest_steering_text + "x")
    inbox.close_and_take_all_commands()
    with pytest.raises(ValueError, match="at most 4000 characters"):
        inbox.submit_command("steer", longest_steering_text + "x")

    assert accepted.status == "accepted"
