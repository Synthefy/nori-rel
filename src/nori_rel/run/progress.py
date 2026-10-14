"""A run's progress, rebuilt from its events.

Progress is never sent on its own: local and remote callers fold the same event
stream with :func:`apply_event_to_progress`, so they see the same numbers. The
progress after a run's final event is its summary.
"""

from __future__ import annotations

from typing import Any

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from nori_rel.run.models import (
    EventKind,
    RunEvent,
    RunStatus,
    Stage,
)


class RunProgress(BaseModel):
    """What a run has done so far, as of ``last_event_id``.

    ``active_runtime_seconds`` counts only time spent ``running``, up to the
    latest event; an event timestamped before an earlier one adds nothing. The
    final strategy, pipeline identifier and input hashes describe the result.
    The hashes change as the database context or feature matrix changes.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=1)
    last_event_id: int = 0
    status: RunStatus = "queued"
    stage: Stage | None = None
    message: str | None = None
    active_runtime_seconds: float = 0.0
    running_since: AwareDatetime | None = None
    strategy: str | None = None
    strategy_used: str | None = None
    pipeline_id: str | None = None
    context_sha256: str | None = None
    features_sha256: str | None = None
    error_kind: str | None = None
    error_message: str | None = None


def _get_runtime_update(progress: RunProgress, event: RunEvent) -> dict[str, Any]:
    """Return the running time ``event`` adds while the run is ``running``.

    Time is counted up to every event, not only status changes, and never
    backwards: an out-of-order timestamp adds nothing.
    """
    if progress.running_since is None:
        return {}
    elapsed = (event.timestamp - progress.running_since).total_seconds()
    return {
        "active_runtime_seconds": progress.active_runtime_seconds + max(elapsed, 0.0),
        "running_since": max(event.timestamp, progress.running_since),
    }


def _status_change_update(event: RunEvent) -> dict[str, Any]:
    """Return the fields a status change sets, and when running time restarts."""
    data = event.data
    update: dict[str, Any] = {"status": data["to"]}
    update["running_since"] = event.timestamp if data["to"] == "running" else None
    for field in (
        "strategy",
        "strategy_used",
        "pipeline_id",
        "context_sha256",
        "features_sha256",
    ):
        if field in data:
            update[field] = data[field]
    return update


def apply_event_to_progress(progress: RunProgress, event: RunEvent) -> RunProgress:
    """Return ``progress`` updated by ``event``.

    An event already applied (``event_id`` not above ``last_event_id``) changes
    nothing, so a replayed or reconnected stream can be folded again safely.
    """
    if event.run_id != progress.run_id:
        raise ValueError(f"event for run {event.run_id!r}, not {progress.run_id!r}")
    if event.event_id <= progress.last_event_id:
        return progress
    data = event.data
    update: dict[str, Any] = {"last_event_id": event.event_id, "message": event.message}
    update |= _get_runtime_update(progress, event)
    match EventKind(data["kind"]):
        case EventKind.STATUS_CHANGED:
            update |= _status_change_update(event)
        case EventKind.STAGE_STARTED:
            update["stage"] = data["stage"]
        case EventKind.STAGE_FINISHED:
            # A nested stage returns to the stage it ran inside.
            update["stage"] = data.get("outer_stage")
        case EventKind.ERROR:
            update |= {
                "error_kind": data.get("error_kind"),
                "error_message": event.message,
            }
        case EventKind.COMMAND | EventKind.LIMIT_REACHED:
            pass
    return RunProgress.model_validate(progress.model_dump() | update)


def progress_from_events(run_id: str, events: list[RunEvent]) -> RunProgress:
    """Return the progress after applying ``events`` in order to a new run."""
    progress = RunProgress(run_id=run_id)
    for event in events:
        progress = apply_event_to_progress(progress, event)
    return progress


class RunStatusResponse(BaseModel):
    """Status, latest message, and the sanitized error if a run failed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=1)
    status: RunStatus
    message: str | None = None
    error: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _error_iff_failed(self) -> RunStatusResponse:
        if (self.status == "failed") != (self.error is not None):
            raise ValueError(
                "an error is required for failed runs and absent otherwise"
            )
        return self

    @classmethod
    def from_progress(cls, progress: RunProgress) -> RunStatusResponse:
        """Report ``progress``; a failed run's error is its redacted message."""
        failed = progress.status == "failed"
        return cls(
            run_id=progress.run_id,
            status=progress.status,
            message=progress.message,
            error=(progress.error_message or "Run failed.") if failed else None,
        )


__all__ = [
    "RunProgress",
    "RunStatusResponse",
    "apply_event_to_progress",
    "progress_from_events",
]
