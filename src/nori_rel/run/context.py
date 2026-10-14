"""What the prediction pipeline sees of its run: events out, commands in.

The pipeline calls :meth:`RunContext.apply_pending_commands` between stages,
and before each LLM call, trial, Nori call and the final prediction. Code that
is not part of a watched run uses :meth:`RunContext.unobserved`, which nobody
watches or commands.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

from nori_rel.run.commands import CommandInbox, QueuedCommand
from nori_rel.run.messages import render
from nori_rel.run.models import (
    EVENT_TYPE_BY_KIND,
    EventKind,
    RunEvent,
    RunLimits,
    RunStatus,
    Stage,
)

logger = logging.getLogger(__name__)


class RunCancelled(BaseException):
    """Raised at every boundary after a cancel; the executor ends the run cancelled.

    A ``BaseException``, like ``asyncio.CancelledError``, so code that handles
    ``Exception`` cannot swallow it: tool dispatch and the session's refusal
    handling catch ``Exception`` and carry on.
    """


def _utc_now() -> datetime:
    return datetime.now(UTC)


class RunContext:
    """One run's event sink, command inbox, limits and command state."""

    def __init__(
        self,
        run_id: str,
        *,
        event_sink: Callable[[RunEvent], None],
        limits: RunLimits | None = None,
        clock: Callable[[], datetime] = _utc_now,
        seconds_clock: Callable[[], float] | None = None,
    ) -> None:
        """Send events to ``event_sink``; read time from ``clock`` (UTC).

        Durations (stage times, running time for ``max_runtime_seconds``) come
        from ``seconds_clock``: ``time.monotonic`` by default, so a step of
        the wall clock cannot change them. A test that passes its own
        ``clock`` gets durations from that clock unless it passes both.
        """
        self.run_id = run_id
        self.limits = limits or RunLimits()
        self.commands = CommandInbox(run_id)
        self.status: RunStatus = "queued"
        self.stop_applied = False
        self.cancel_applied = False
        #: The agent's spend so far in dollars, for ``limits.max_spend_usd``.
        self.spend_usd: float | None = None
        #: Whether steering is accepted. An agent search turns it on from the
        #: run's first boundary and the search's end turns it off; steering is
        #: held until the agent delivers it.
        self.steering_enabled = False
        #: What the final status reports about a pipeline run: its id and the
        #: audit hashes of its inputs. Empty for a run without a pipeline.
        self.pipeline_summary: dict[str, str] = {}
        self._event_sink = event_sink
        self._clock = clock
        if seconds_clock is None:
            seconds_clock = (
                time.monotonic if clock is _utc_now else lambda: clock().timestamp()
            )
        self._seconds_clock = seconds_clock
        self._finished_running_seconds = 0.0
        self._running_since_seconds: float | None = None
        # Reentrant so that a sink which emits again on the same thread, such
        # as one that reports its own failure as an event, cannot deadlock the
        # run. No sink in this package does that today.
        self._event_lock = threading.RLock()
        self._last_event_id = 0
        #: Stages under way, outermost first: confirming the best pipeline
        #: runs inside planning features.
        self._open_stages: list[Stage] = []
        #: Steering accepted but not yet sent to the model. Only the run's
        #: worker thread touches it, so it needs no lock.
        self._held_steers: list[QueuedCommand] = []

    @classmethod
    def unobserved(cls) -> RunContext:
        """Return a context whose events go nowhere and that gets no commands."""
        return cls("unobserved", event_sink=lambda event: None)

    def emit_event(self, kind: EventKind, **data: Any) -> RunEvent:
        """Send the next event of ``kind``; a failing sink is logged, not raised.

        Events reach the sink in id order even when several threads emit, since
        a reader ignores an event older than one it has applied. The sink runs
        under the lock, so it must be quick; the lock is reentrant, so a sink
        that emits on the same thread does not deadlock.
        """
        message = render(kind, data)
        with self._event_lock:
            self._last_event_id += 1
            event = RunEvent(
                run_id=self.run_id,
                event_id=self._last_event_id,
                timestamp=self._clock(),
                type=EVENT_TYPE_BY_KIND[kind],
                message=message,
                data={"kind": kind.value, **data},
            )
            try:
                self._event_sink(event)
            except Exception:
                logger.exception("run %s: event sink failed", self.run_id)
        return event

    def report_pipeline(
        self, pipeline_id: str, *, context_sha256: str, features_sha256: str
    ) -> None:
        """Keep a pipeline run's id and audit hashes for its final status.

        The hashes describe this run's inputs and change from run to run; the
        run summary records them so a past prediction can be traced.
        """
        self.pipeline_summary = {
            "pipeline_id": pipeline_id,
            "context_sha256": context_sha256,
            "features_sha256": features_sha256,
        }

    def change_status(self, to: RunStatus, **data: Any) -> None:
        """Move the run to status ``to`` and report it, counting running time."""
        now_seconds = self._seconds_clock()
        if self._running_since_seconds is not None:
            elapsed = max(now_seconds - self._running_since_seconds, 0.0)
            self._finished_running_seconds += elapsed
        self._running_since_seconds = now_seconds if to == "running" else None
        previous, self.status = self.status, to
        self.emit_event(
            EventKind.STATUS_CHANGED, **{"from": previous, "to": to}, **data
        )

    @contextmanager
    def report_stage(self, stage: Stage) -> Iterator[None]:
        """Apply commands, then report ``stage`` starting and, on success, ending.

        The end names ``outer_stage``, the stage still under way when a nested
        stage ends, so progress returns to it.
        """
        self.apply_pending_commands()
        started_at_seconds = self._seconds_clock()
        self.emit_event(EventKind.STAGE_STARTED, stage=stage.value)
        self._open_stages.append(stage)
        try:
            yield
        finally:
            self._open_stages.pop()
        seconds = max(self._seconds_clock() - started_at_seconds, 0.0)
        outer_stage = self._open_stages[-1].value if self._open_stages else None
        self.emit_event(
            EventKind.STAGE_FINISHED,
            stage=stage.value,
            seconds=seconds,
            outer_stage=outer_stage,
        )

    def apply_pending_commands(self, *, check_limits: bool = False) -> None:
        """Apply queued commands, wait until a pause ends, and raise on cancel.

        A pause ends with a resume, a stop or a cancel; only a command wakes a
        paused run, so whoever shuts a run down sends one. Once a cancel has
        been applied, every later call raises again, so a cancel caught and
        dropped somewhere still ends the run at the next boundary.

        With ``check_limits``, a reached ``max_runtime_seconds`` or
        ``max_spend_usd`` then stops the run as a stop command would. Only the
        agent search's request boundary asks for it: a limit ends the search
        before its next paid request, not a proposal already paid for, and it
        has nothing to stop outside the search.
        """
        if self.cancel_applied:
            raise RunCancelled(self.run_id)
        self._apply_commands(self.commands.take_all_commands())
        while self.status == "paused":
            self.commands.wait_for_command()
            self._apply_commands(self.commands.take_all_commands())
        # A limit stop stays after the pause loop: a stopped run is never
        # paused, so paused time never counts against anything after a stop.
        if check_limits:
            self._stop_at_reached_limit()

    def get_active_runtime_seconds(self) -> float:
        """Return the seconds spent running, excluding queued and paused time."""
        if self._running_since_seconds is None:
            return self._finished_running_seconds
        current = max(self._seconds_clock() - self._running_since_seconds, 0.0)
        return self._finished_running_seconds + current

    def record_spend(self, spend_usd: float) -> None:
        """Keep the agent's spend so far; the next request boundary checks it."""
        self.spend_usd = spend_usd

    def _stop_at_reached_limit(self) -> None:
        """Stop the run, as the stop command does, once a limit is reached."""
        if self.stop_applied:
            return
        limits = self.limits
        if (
            limits.max_runtime_seconds is not None
            and self.get_active_runtime_seconds() >= limits.max_runtime_seconds
        ):
            reached_limit = "max_runtime_seconds"
        elif (
            limits.max_spend_usd is not None
            and self.spend_usd is not None
            and self.spend_usd >= limits.max_spend_usd
        ):
            reached_limit = "max_spend_usd"
        else:
            return
        self.stop_applied = True
        self.emit_event(EventKind.LIMIT_REACHED, limit=reached_limit)

    def has_held_steering(self) -> bool:
        """Return whether steering is waiting for the agent's next request."""
        return bool(self._held_steers)

    def deliver_held_steering_texts(self) -> list[str]:
        """Return the held steering texts, oldest first, and report each applied.

        Call it as the last step before building the model request that carries
        the texts: a steer is applied only once its text is on its way to the
        model, however long it was held. Worker thread only.
        """
        delivered, self._held_steers = self._held_steers, []
        for queued in delivered:
            self._report_command(queued, "applied", text=queued.steering_text)
        return [queued.steering_text or "" for queued in delivered]

    def reject_undelivered_steering_texts(self, reason: str) -> None:
        """Reject every held steer with ``reason`` and stop accepting steering.

        Steering that arrives later is rejected as ``not_applicable`` at the
        next boundary. Call it wherever the agent search can end, so no steer
        is left held, and reported neither applied nor rejected, after that.
        Worker thread only.
        """
        self.steering_enabled = False
        undelivered, self._held_steers = self._held_steers, []
        for queued in undelivered:
            self._report_command(queued, "rejected", reason=reason)

    def _rejection_reason(self, queued: QueuedCommand) -> str | None:
        """Return why ``queued`` cannot apply now, or ``None`` if it can."""
        match queued.command:
            case "stop" if self.stop_applied:
                return "no_op"
            case "pause" if self.status == "paused":
                return "no_op"
            case "pause" if self.stop_applied:
                return "not_applicable"
            case "resume" if self.status != "paused":
                return "no_op"
            case "steer" if not self.steering_enabled:
                return "not_applicable"
        return None

    def rejection_reason_once_the_run_ends(self, queued: QueuedCommand) -> str:
        """Return why ``queued``, still unapplied as the run ends, is rejected.

        A cancel or a stop that repeats one already applied is ``no_op``, as
        in :meth:`_rejection_reason`. Any other command had no boundary left
        to be applied at, so it is ``run_finished``.
        """
        repeats_applied_command = (
            queued.command == "cancel" and self.cancel_applied
        ) or (queued.command == "stop" and self.stop_applied)
        return "no_op" if repeats_applied_command else "run_finished"

    def _report_command(self, queued: QueuedCommand, status: str, **data: Any) -> None:
        self.emit_event(
            EventKind.COMMAND,
            command_id=queued.command_id,
            command=queued.command,
            status=status,
            **data,
        )

    def _apply_commands(self, queued_commands: list[QueuedCommand]) -> None:
        """Apply commands in arrival order; a cancel overrides every other one.

        With a cancel among them, the cancel applies and the other commands,
        before or after it, are rejected as if the run had ended: a repeated
        cancel or stop is ``no_op``, any other command ``run_finished``.
        """
        cancel = next(
            (queued for queued in queued_commands if queued.command == "cancel"), None
        )
        if cancel is not None:
            self._report_command(cancel, "applied")
            self.cancel_applied = True
            self.reject_undelivered_steering_texts("run_finished")
            for other in queued_commands:
                if other is not cancel:
                    self._report_command(
                        other,
                        "rejected",
                        reason=self.rejection_reason_once_the_run_ends(other),
                    )
            raise RunCancelled(self.run_id)
        for queued in queued_commands:
            reason = self._rejection_reason(queued)
            if reason is not None:
                self._report_command(queued, "rejected", reason=reason)
                continue
            if queued.command == "steer":
                # Held, not reported, until the agent delivers it to the model.
                self._held_steers.append(queued)
                continue
            self._report_command(queued, "applied")
            match queued.command:
                case "stop":
                    self.stop_applied = True
                    # A stop ends a pause: the run finishes without a resume.
                    if self.status == "paused":
                        self.change_status("running")
                case "pause":
                    self.change_status("paused")
                case "resume":
                    self.change_status("running")


__all__ = ["RunCancelled", "RunContext"]
