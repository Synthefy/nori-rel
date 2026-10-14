"""Local runs: a prediction executed in this process and watched through events.

``PredictionRun`` is the handle a caller holds: the run's progress, its
replayable event journal and its result. ``LocalRun`` implements it in this
process. See
``docs/run-control.md``.
"""

from __future__ import annotations

import logging
import re
import threading
import traceback
import uuid
from collections.abc import Callable, Iterator
from functools import partial
from typing import Any, Protocol

from nori_rel.run.commands import QueuedCommand
from nori_rel.run.context import RunCancelled, RunContext
from nori_rel.run.errors import PredictionCancelled, PredictionFailed, RunTimeout
from nori_rel.run.models import (
    FINAL_STATUSES,
    Command,
    CommandReceipt,
    EventKind,
    RunEvent,
    RunLimits,
    RunStatus,
)
from nori_rel.run.progress import (
    RunProgress,
    RunStatusResponse,
    apply_event_to_progress,
)
from nori_rel.run.redact import redact

logger = logging.getLogger(__name__)

#: Where words meet in a class name, e.g. ``OS|Error`` or ``Value|Error``.
_CLASS_NAME_WORD_BOUNDARY = re.compile(
    r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])"
)


def new_run_id() -> str:
    """Return a new run id, e.g. ``run_3f2b9c0d1e4a5b6c``."""
    return f"run_{uuid.uuid4().hex[:16]}"


def _error_kind_from_exception(error: BaseException) -> str:
    """Return the error's class name in snake case, e.g. ``os_error``."""
    return _CLASS_NAME_WORD_BOUNDARY.sub("_", type(error).__name__).lower()


class PredictionRun(Protocol):
    """A prediction run as its caller sees it, wherever it executes."""

    run_id: str

    @property
    def progress(self) -> RunProgress:
        """The run's progress as of its latest event; its summary once final."""
        ...

    def status(self) -> RunStatusResponse:
        """Return the run's status, latest message and, if it failed, its error."""
        ...

    def events(self, after: int = 0) -> Iterator[RunEvent]:
        """Yield the run's events with ids above ``after`` until it has ended."""
        ...

    def result(self, timeout: float | None = None) -> Any:
        """Return the prediction, waiting at most ``timeout`` seconds if given."""
        ...

    def cancel(self) -> CommandReceipt:
        """Abort the run at its next boundary; ``result()`` then raises.

        After the run has ended this returns a rejected receipt and emits no
        event; it does not raise.
        """
        ...

    def pause(self) -> CommandReceipt:
        """Hold the run at its next boundary until :meth:`resume` or cancel.

        After the run has ended this returns a rejected receipt and emits no
        event; it does not raise.
        """
        ...

    def resume(self) -> CommandReceipt:
        """Continue a paused run.

        After the run has ended this returns a rejected receipt and emits no
        event; it does not raise.
        """
        ...


class LocalRun:
    """One prediction run in this process: its progress, events and result.

    Events are kept for the run's lifetime, so :meth:`events` can replay them
    from any id. ``on_event`` receives each event in order on the thread that
    emits it. It should return quickly and must not wait for the run through
    ``result()`` or ``events()``; an exception it raises is logged.
    """

    def __init__(
        self,
        run_id: str,
        *,
        limits: RunLimits | None = None,
        on_event: Callable[[RunEvent], None] | None = None,
    ) -> None:
        """Create a queued run; an executor runs it with ``_execute_prediction``."""
        self.run_id = run_id
        self._on_event = on_event
        self._journal: list[RunEvent] = []
        self._progress = RunProgress(run_id=run_id)
        self._run_state_changed = threading.Condition()
        self._finished = threading.Event()
        self._prediction: Any = None
        self._failure: Exception | None = None
        #: Set once the final status is journaled; later events are dropped.
        self._final_status_recorded = False
        #: The thread that ends the run, so ``result()`` called from its own
        #: ``on_event`` fails instead of waiting for itself.
        self._finishing_thread_id: int | None = None
        self._context = RunContext(run_id, event_sink=self._record_event, limits=limits)

    @property
    def progress(self) -> RunProgress:
        """The run's progress as of its latest event; its summary once final."""
        with self._run_state_changed:
            return self._progress

    @property
    def is_finished(self) -> bool:
        """Whether the run has ended and its result is available."""
        return self._finished.is_set()

    def status(self) -> RunStatusResponse:
        """Return the run's status, latest message and, if it failed, its error."""
        return RunStatusResponse.from_progress(self.progress)

    def wait_until_finished(self, timeout: float | None = None) -> bool:
        """Wait at most ``timeout`` seconds, if given, for the run to end.

        Returns whether it has ended. Unlike ``result()``, it never raises.
        """
        return self._finished.wait(timeout)

    def events(self, after: int = 0) -> Iterator[RunEvent]:
        """Yield the run's events with ids above ``after``, waiting for new ones.

        Ends once the run has finished and every event has been yielded; the
        final status is the last event. Each call replays independently, so a
        reader that reconnects passes the last event id it saw. Event ids run
        1, 2, 3, … so id ``n`` is journal entry ``n - 1``.
        """
        next_index = max(after, 0)
        while True:
            with self._run_state_changed:
                self._run_state_changed.wait_for(
                    partial(self._has_events_after_index_or_finished, next_index)
                )
                new_events = self._journal[next_index:]
            if not new_events:
                return
            yield from new_events
            next_index += len(new_events)

    def result(self, timeout: float | None = None) -> Any:
        """Return the prediction, waiting at most ``timeout`` seconds if given.

        Raises ``RunTimeout`` if the wait ends first, and the run keeps going;
        ``PredictionCancelled`` after a cancel; ``PredictionFailed``, chained to
        the original exception, after a failure. The final status event comes a
        moment before the result is ready, so a reader that stops on that event
        and calls ``result(timeout=0)`` still gets the result, not a timeout.
        """
        if not self._finished.wait(timeout):
            with self._run_state_changed:
                final_status_recorded = self._final_status_recorded
            if not final_status_recorded:
                progress = self.progress
                raise RunTimeout(
                    f"Run {self.run_id} is still {progress.status} after {timeout} s.",
                    progress=progress,
                )
            if threading.get_ident() == self._finishing_thread_id:
                raise RuntimeError(
                    "on_event cannot wait for its own run's result; "
                    "read run.result() after the callback returns"
                )
            # Only bookkeeping is left once the final status is out.
            self._finished.wait()
        progress = self.progress
        if progress.status == "cancelled":
            raise PredictionCancelled(
                f"Run {self.run_id} was cancelled.", progress=progress
            )
        if progress.status == "failed":
            raise PredictionFailed(
                progress.error_message or "Run failed.", progress=progress
            ) from self._failure
        return self._prediction

    def _execute_prediction(
        self,
        prediction: Callable[[RunContext], Any],
        *,
        strategy: str,
        summary_in_result_attrs: bool,
        reraise_failure: bool,
    ) -> None:
        """Run ``prediction`` with this run's context on the calling thread.

        Its outcome becomes its final status, read with :meth:`result`. With
        ``reraise_failure`` a failure is also raised here unchanged; an
        interruption such as ``KeyboardInterrupt`` always is, after the run
        ends cancelled. Runs execute independently.
        """
        final_status: RunStatus = "failed"
        try:
            self._context.change_status(
                "running",
                strategy=strategy,
            )
            self._prediction = prediction(self._context)
            final_status = "completed"
        except RunCancelled:
            final_status = "cancelled"
        except Exception as error:
            self._record_failure(error, keep_frames=reraise_failure)
            if reraise_failure:
                raise
        except BaseException:
            final_status = "cancelled"
            raise
        finally:
            self._finish_run(final_status, strategy, summary_in_result_attrs)

    def pause(self) -> CommandReceipt:
        """Hold the run at its next boundary until :meth:`resume` or cancel.

        While paused the run reports ``paused`` and its active runtime, which
        ``max_runtime_seconds`` limits, does not grow. After the run has ended
        this returns a rejected
        receipt and emits no event; it does not raise.
        """
        return self._queue_command_for_next_boundary("pause")

    def resume(self) -> CommandReceipt:
        """Continue a paused run.

        After the run has ended this returns a rejected receipt and emits no
        event; it does not raise.
        """
        return self._queue_command_for_next_boundary("resume")

    def cancel(self) -> CommandReceipt:
        """Abort the run at its next boundary; ``result()`` then raises.

        The work in flight (a database or model call) finishes
        first; then only cleanup runs and the run ends ``cancelled``. After the
        run has ended this returns a rejected receipt and emits no event; it does
        not raise.
        """
        return self._queue_command_for_next_boundary("cancel")

    def _has_events_after_index_or_finished(self, journal_index: int) -> bool:
        return len(self._journal) > journal_index or self._finished.is_set()

    def _queue_command_for_next_boundary(self, command: Command) -> CommandReceipt:
        """Queue ``command`` for the next boundary, or reject it once the run ends.

        A command queued before the end is applied or rejected by an event. After
        the end the receipt is rejected and no event reports it, so the final
        status stays the run's last event. This takes no run lock and emits
        nothing, so it is safe to call from an ``on_event`` callback.
        """
        return self._context.commands.submit_command(command)

    def _record_event(self, event: RunEvent) -> None:
        """Journal ``event``, fold it into the progress, and pass it to ``on_event``.

        The final status is always the last event: one emitted after it is
        logged and dropped, since a hosted reader stops on the final status.
        A fold that raises is logged; the event stays journaled and waiting
        readers are still woken.
        """
        with self._run_state_changed:
            if self._final_status_recorded:
                logger.error(
                    "run %s: dropped a %s event after the final status",
                    self.run_id,
                    event.data.get("kind"),
                )
                return
            self._journal.append(event)
            try:
                self._progress = apply_event_to_progress(self._progress, event)
            except Exception:
                logger.exception("run %s: could not fold an event", self.run_id)
            finally:
                if event.data.get("kind") == EventKind.STATUS_CHANGED and (
                    event.data.get("to") in FINAL_STATUSES
                ):
                    self._final_status_recorded = True
                self._run_state_changed.notify_all()
        if self._on_event is not None:
            try:
                self._on_event(event)
            except Exception:
                logger.exception("run %s: on_event callback failed", self.run_id)

    def _record_failure(self, error: Exception, *, keep_frames: bool) -> None:
        """Keep ``error`` for ``result()`` and report it, redacted, as an event.

        Unless the caller re-raises it, the frames' local variables are
        released; the traceback still shows where the prediction failed. The
        first frame is this run's own, still running, so it is dropped rather
        than cleared: it holds the prediction closure (client, source).
        """
        self._failure = error
        if not keep_frames and error.__traceback__ is not None:
            error.__traceback__ = error.__traceback__.tb_next
            traceback.clear_frames(error.__traceback__)
        self._context.emit_event(
            EventKind.ERROR,
            error_kind=_error_kind_from_exception(error),
            message=redact(str(error)) or "Run failed.",
        )

    def _finish_run(
        self, status: RunStatus, strategy: str, summary_in_result_attrs: bool
    ) -> None:
        """Reject the commands the run never applied, then end it with ``status``.

        The inbox is closed first, so later commands get a rejected receipt and
        no event: the final status, which carries the summary fields, is the run's
        last event. With ``summary_in_result_attrs``, the prediction's
        ``attrs["run"]`` also holds the run's summary.
        """
        self._finishing_thread_id = threading.get_ident()
        try:
            for queued in self._context.commands.close_and_take_all_commands():
                self._reject_command_after_run_finished(queued)
            progress = self.progress
            completed_strategy = strategy if status == "completed" else None
            self._context.change_status(
                status,
                strategy_used=progress.strategy_used or completed_strategy,
                **{
                    "pipeline_id": progress.pipeline_id,
                    **self._context.pipeline_summary,
                },
            )
            attrs = getattr(self._prediction, "attrs", None)
            if summary_in_result_attrs and isinstance(attrs, dict):
                attrs["run"] = self.progress.model_dump(mode="json")
        finally:
            with self._run_state_changed:
                self._finished.set()
                self._run_state_changed.notify_all()

    def _reject_command_after_run_finished(self, queued: QueuedCommand) -> None:
        """Report that the run ended before ``queued`` could be applied.

        A cancel that repeats one the run already applied is ``no_op``;
        any other command is ``run_finished``.
        """
        self._context.emit_event(
            EventKind.COMMAND,
            command_id=queued.command_id,
            command=queued.command,
            status="rejected",
            reason=self._context.rejection_reason_once_the_run_ends(queued),
        )


__all__ = ["LocalRun", "PredictionRun", "new_run_id"]
