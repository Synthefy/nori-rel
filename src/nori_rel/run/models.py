"""Run status, events, commands and limits shared by local and hosted runs.

See ``docs/run-control.md`` for the public lifecycle contract.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final, Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    model_validator,
)

RunStatus = Literal["queued", "running", "paused", "completed", "failed", "cancelled"]
FINAL_STATUSES: Final = frozenset({"completed", "failed", "cancelled"})

EventType = Literal["status", "progress", "action", "metric", "warning", "error"]
Command = Literal["cancel", "stop", "pause", "resume", "steer"]


class Stage(StrEnum):
    """The stages of relational prediction."""

    MATERIALIZE = "materialize"
    PLAN_SEARCH = "plan_search"
    RUN_TRIAL = "run_trial"
    EVALUATE_TRIAL = "evaluate_trial"
    SELECT_PIPELINE = "select_pipeline"
    PREDICT_BATCH = "predict_batch"


class EventKind(StrEnum):
    """What an event reports; carried as ``RunEvent.data["kind"]``."""

    STATUS_CHANGED = "status_changed"
    STAGE_STARTED = "stage_started"
    STAGE_FINISHED = "stage_finished"
    BASELINE_SCORED = "baseline_scored"
    TRIAL_SCORED = "trial_scored"
    SPEND_UPDATED = "spend_updated"
    COMMAND = "command"
    FALLBACK_USED = "fallback_used"
    SEARCH_STOPPED = "search_stopped"
    LIMIT_REACHED = "limit_reached"
    ERROR = "error"


class FallbackReason(StrEnum):
    """Why a run fell back from its search to the DFS baseline."""

    NO_PROGRAM_FOUND = "no_program_found"
    NO_PROGRAM_CONFIRMED = "no_program_confirmed"
    STOPPED_BEFORE_CONFIRMATION = "stopped_before_confirmation"
    STOPPED_WITHOUT_PROGRAM = "stopped_without_program"
    SEARCH_FAILED = "search_failed"


class TrialOutcome(StrEnum):
    """What the search decided about one scored trial.

    ``NEW_BEST`` and ``BEST_BELOW_BASELINE`` accept the trial as the best so
    far; the other two reject it. Significance is checked before the
    point-in-time check, so ``NOT_SIGNIFICANT_GAIN`` does not certify that the
    trial passed the check.
    """

    NEW_BEST = "new_best"
    BEST_BELOW_BASELINE = "best_below_baseline"
    NOT_SIGNIFICANT_GAIN = "not_significant_gain"
    FAILED_POINT_IN_TIME_CHECK = "failed_point_in_time_check"


#: The event type each kind is reported as.
EVENT_TYPE_BY_KIND: Final[dict[EventKind, EventType]] = {
    EventKind.STATUS_CHANGED: "status",
    EventKind.STAGE_STARTED: "progress",
    EventKind.STAGE_FINISHED: "progress",
    EventKind.BASELINE_SCORED: "metric",
    EventKind.TRIAL_SCORED: "metric",
    EventKind.SPEND_UPDATED: "metric",
    EventKind.COMMAND: "action",
    EventKind.FALLBACK_USED: "warning",
    EventKind.SEARCH_STOPPED: "warning",
    EventKind.LIMIT_REACHED: "warning",
    EventKind.ERROR: "error",
}


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RunEvent(_Frozen):
    """One update in a run's event stream .

    ``event_id`` increases within a run and is stable when the stream is
    replayed. ``data`` always names its ``kind``, and ``type`` must be the type
    that kind is reported as. The model is strict because it checks the events
    this package produces; a hosted client must parse the service's events
    leniently, since a newer service can add kinds and fields.
    """

    run_id: str = Field(min_length=1)
    event_id: int = Field(ge=1)
    timestamp: AwareDatetime
    type: EventType
    message: str = Field(min_length=1)
    data: dict[str, JsonValue] = Field(default_factory=dict)

    def __hash__(self) -> int:
        """Hash by run and event id, which identify an event; ``data`` is a dict."""
        return hash((self.run_id, self.event_id))

    @model_validator(mode="after")
    def _kind_matches_type(self) -> RunEvent:
        kind = self.data.get("kind")
        if not isinstance(kind, str) or kind not in set(EventKind):
            raise ValueError(f"data.kind must be one of {[k.value for k in EventKind]}")
        expected = EVENT_TYPE_BY_KIND[EventKind(kind)]
        if expected != self.type:
            raise ValueError(f"{kind} events have type {expected!r}")
        return self


class CommandAcceptedResponse(_Frozen):
    """Receipt for a recorded command. It is applied at a later boundary.

    Exactly one ``command`` event, applied or rejected, later carries its
    ``command_id``.
    """

    run_id: str = Field(min_length=1)
    command_id: str = Field(min_length=1)
    status: Literal["accepted"] = "accepted"


class CommandRejectedResponse(_Frozen):
    """Receipt for a command sent after the run had ended.

    Nothing was recorded and no event reports it: the run's final status is
    already its last event. ``command_id`` is only for the caller's own logs.
    """

    run_id: str = Field(min_length=1)
    command_id: str = Field(min_length=1)
    status: Literal["rejected"] = "rejected"
    reason: Literal["run_finished"] = "run_finished"


CommandReceipt = CommandAcceptedResponse | CommandRejectedResponse


class RunLimits(_Frozen):
    """Bounds on a run.

    The ``max_*`` limits end the search gracefully, as ``stop`` does: the run
    still completes with the best result so far. ``max_runtime_seconds``
    counts only running time; it takes fractions of a second, where a hosted
    request takes whole seconds. ``stop_grace_s`` budgets the work
    after a stop that ended the search: the best program is confirmed only if
    the estimated confirmation fits in ``confirm_budget_fraction`` of it,
    leaving the rest for running the selected program. The trial cap is not a
    run limit: it is ``AgenticFeatureConfig.max_trials``, which counts scored
    trials.
    """

    max_runtime_seconds: float | None = Field(default=None, gt=0)
    max_spend_usd: float | None = Field(default=None, gt=0)
    stop_grace_s: float = Field(default=60.0, gt=0)
    confirm_budget_fraction: float = Field(default=0.5, gt=0, le=1)


__all__ = [
    "EVENT_TYPE_BY_KIND",
    "FINAL_STATUSES",
    "Command",
    "CommandAcceptedResponse",
    "CommandReceipt",
    "CommandRejectedResponse",
    "EventKind",
    "EventType",
    "FallbackReason",
    "RunEvent",
    "RunLimits",
    "RunStatus",
    "Stage",
    "TrialOutcome",
]
