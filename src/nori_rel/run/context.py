"""What the prediction pipeline sees of its run: events out, commands in.

The pipeline calls :meth:`RunContext.apply_pending_commands` between stages.
Code outside a watched run uses :meth:`RunContext.unobserved`.
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
    ``Exception`` cannot swallow cancellation.
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
        self.cancel_applied = False
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
        #: Stages under way, outermost first: nested stages report their
        #: enclosing stage when they finish.
        self._open_stages: list[Stage] = []

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

    def apply_pending_commands(self) -> None:
        """Apply commands between stages, waiting while paused and raising on cancel.

        Cancellation is cooperative: an in-flight database/model operation uses
        its own timeout, then the next stage boundary observes the command.
        """
        if self.cancel_applied:
            raise RunCancelled(self.run_id)
        self._apply_commands(self.commands.take_all_commands())
        while self.status == "paused":
            self.commands.wait_for_command()
            self._apply_commands(self.commands.take_all_commands())
        limit = self.limits.max_runtime_seconds
        if limit is not None and self.get_active_runtime_seconds() >= limit:
            self.cancel_applied = True
            self.emit_event(EventKind.LIMIT_REACHED, limit="max_runtime_seconds")
            raise RunCancelled(self.run_id)

    def get_active_runtime_seconds(self) -> float:
        """Return the seconds spent running, excluding queued and paused time."""
        if self._running_since_seconds is None:
            return self._finished_running_seconds
        current = max(self._seconds_clock() - self._running_since_seconds, 0.0)
        return self._finished_running_seconds + current

    def _rejection_reason(self, queued: QueuedCommand) -> str | None:
        """Return why ``queued`` cannot apply now, or ``None`` if it can."""
        match queued.command:
            case "pause" if self.status == "paused":
                return "no_op"
            case "resume" if self.status != "paused":
                return "no_op"
        return None

    def rejection_reason_once_the_run_ends(self, queued: QueuedCommand) -> str:
        """Return why ``queued``, still unapplied as the run ends, is rejected.

        A cancel that repeats one already applied is ``no_op``, as
        in :meth:`_rejection_reason`. Any other command had no boundary left
        to be applied at, so it is ``run_finished``.
        """
        repeated_cancel = queued.command == "cancel" and self.cancel_applied
        return "no_op" if repeated_cancel else "run_finished"

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
        cancel is ``no_op``, any other command ``run_finished``.
        """
        cancel = next(
            (queued for queued in queued_commands if queued.command == "cancel"), None
        )
        if cancel is not None:
            self._report_command(cancel, "applied")
            self.cancel_applied = True
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
            self._report_command(queued, "applied")
            match queued.command:
                case "pause":
                    self.change_status("paused")
                case "resume":
                    self.change_status("running")


__all__ = ["RunCancelled", "RunContext"]
