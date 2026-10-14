"""LocalExecutor: starts local runs and finds them again by id.

A :class:`~nori_rel.run.local_run.LocalRun` holds a run's state and executes
its prediction; the executor only chooses the thread it executes on, a new
worker or the caller's, and keeps background runs for ``get_run``. See
``docs/run-control.md``.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from nori_rel.run.context import RunContext
from nori_rel.run.local_run import LocalRun, new_run_id
from nori_rel.run.models import RunEvent, RunLimits

logger = logging.getLogger(__name__)

#: Finished runs an executor keeps for ``get_run``, with their predictions and
#: journals. The oldest by start order are forgotten when the next run starts.
_MAX_FINISHED_RUNS_KEPT = 20
#: How often waiting for runs looks for one that has paused.
_PAUSED_RUN_CHECK_SECONDS = 0.1


class LocalExecutor:
    """Executes local runs and finds the background ones again by id."""

    def __init__(self) -> None:
        """Start with no runs, accepting new ones."""
        self._runs: dict[str, LocalRun] = {}
        self._runs_lock = threading.Lock()
        self._closed = False

    def start_run(
        self,
        prediction: Callable[[RunContext], Any],
        *,
        strategy: str,
        max_trials: int | None = None,
        limits: RunLimits | None = None,
        on_event: Callable[[RunEvent], None] | None = None,
    ) -> LocalRun:
        """Start ``prediction`` on a new worker thread and return its run at once.

        Its prediction carries the run's summary in ``attrs["run"]``. The thread
        is a daemon: a run still going when the process exits is lost. Raises
        ``RuntimeError`` once the executor is closed.
        """
        run = LocalRun(new_run_id(), limits=limits, on_event=on_event)
        with self._runs_lock:
            if self._closed:
                raise RuntimeError(
                    "NoriRelClient is closed; create a new client to start a run"
                )
            finished_run_ids = [
                run_id for run_id, known in self._runs.items() if known.is_finished
            ]
            excess_run_count = len(finished_run_ids) - _MAX_FINISHED_RUNS_KEPT
            for run_id in finished_run_ids[: max(excess_run_count, 0)]:
                del self._runs[run_id]
            self._runs[run.run_id] = run
        worker = threading.Thread(
            target=_execute_run_on_worker,
            args=(run, prediction, strategy, max_trials),
            name=f"nori-rel-{run.run_id}",
            daemon=True,
        )
        try:
            worker.start()
        except BaseException:
            # A run that never started must not wait in the registry for ever.
            with self._runs_lock:
                self._runs.pop(run.run_id, None)
            raise
        return run

    def run_on_this_thread(
        self,
        prediction: Callable[[RunContext], Any],
        *,
        strategy: str,
        max_trials: int | None = None,
        limits: RunLimits | None = None,
        on_event: Callable[[RunEvent], None] | None = None,
        summary_in_result_attrs: bool = True,
    ) -> LocalRun:
        """Execute ``prediction`` as a run on the calling thread and return it.

        A failure is raised unchanged, as a plain call would raise it. The run
        is not kept for ``get_run``.
        """
        run = LocalRun(new_run_id(), limits=limits, on_event=on_event)
        run._execute_prediction(
            prediction,
            strategy=strategy,
            max_trials=max_trials,
            summary_in_result_attrs=summary_in_result_attrs,
            reraise_failure=True,
        )
        return run

    def get_run(self, run_id: str) -> LocalRun:
        """Return the background run with ``run_id``; ``KeyError`` if unknown."""
        with self._runs_lock:
            if run_id not in self._runs:
                raise KeyError(
                    f"unknown run {run_id!r}: not started here, or forgotten after "
                    f"{_MAX_FINISHED_RUNS_KEPT} later runs finished"
                )
            return self._runs[run_id]

    def cancel_runs(self, runs: list[LocalRun]) -> None:
        """Send each run a cancel; it applies at the run's next boundary."""
        for run in runs:
            run.cancel()

    def close(self) -> list[LocalRun]:
        """Refuse new runs; return the background runs that have not finished."""
        with self._runs_lock:
            self._closed = True
            return [run for run in self._runs.values() if not run.is_finished]

    def wait_for_unfinished_runs(self, timeout: float | None = None) -> list[LocalRun]:
        """Wait for the background runs to finish; return those still going.

        A paused run would wait for a resume that may never come, so it is
        cancelled instead of waited for. ``timeout`` bounds the whole wait, in
        seconds; without it, the wait lasts until every run has ended.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._runs_lock:
            runs = [run for run in self._runs.values() if not run.is_finished]
        cancelled_run_ids: set[str] = set()
        for run in runs:
            while not run.wait_until_finished(_PAUSED_RUN_CHECK_SECONDS):
                if run.run_id not in cancelled_run_ids and (
                    run.progress.status == "paused"
                ):
                    run.cancel()
                    cancelled_run_ids.add(run.run_id)
                if deadline is not None and time.monotonic() >= deadline:
                    break
        return [run for run in runs if not run.is_finished]


def _execute_run_on_worker(
    run: LocalRun,
    prediction: Callable[[RunContext], Any],
    strategy: str,
    max_trials: int | None,
) -> None:
    """Execute ``run`` on its worker thread; log anything that escapes it.

    Without this, an error leaving the worker would reach only
    ``threading.excepthook``, with no run id.
    """
    try:
        run._execute_prediction(
            prediction,
            strategy=strategy,
            max_trials=max_trials,
            summary_in_result_attrs=True,
            reraise_failure=False,
        )
    except BaseException:
        logger.exception("run %s: the worker thread failed", run.run_id)


__all__ = ["LocalExecutor"]
