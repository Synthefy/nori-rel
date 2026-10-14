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
Command = Literal["cancel", "pause", "resume"]


class Stage(StrEnum):
    """The stages of relational prediction."""

    MATERIALIZE = "materialize"
    PLAN_SEARCH = "plan_search"
    PREDICT_BATCH = "predict_batch"


class EventKind(StrEnum):
    """What an event reports; carried as ``RunEvent.data["kind"]``."""

    STATUS_CHANGED = "status_changed"
    STAGE_STARTED = "stage_started"
    STAGE_FINISHED = "stage_finished"
    COMMAND = "command"
    LIMIT_REACHED = "limit_reached"
    ERROR = "error"


#: The event type each kind is reported as.
EVENT_TYPE_BY_KIND: Final[dict[EventKind, EventType]] = {
    EventKind.STATUS_CHANGED: "status",
    EventKind.STAGE_STARTED: "progress",
    EventKind.STAGE_FINISHED: "progress",
    EventKind.COMMAND: "action",
    EventKind.LIMIT_REACHED: "warning",
    EventKind.ERROR: "error",
}


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RunEvent(_Frozen):
    """One update in a run's event stream.

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
    """Receipt for a command sent after the run ended.

    Nothing was recorded and no event reports it: the run's final status is
    already its last event. ``command_id`` is only for the caller's own logs.
    """

    run_id: str = Field(min_length=1)
    command_id: str = Field(min_length=1)
    status: Literal["rejected"] = "rejected"
    reason: Literal["run_finished"] = "run_finished"


CommandReceipt = CommandAcceptedResponse | CommandRejectedResponse


class RunLimits(_Frozen):
    """Cooperative runtime limit, checked between prediction stages.

    Database and model calls retain their own timeouts. Paused time does not
    count towards this limit; reaching it cancels the run at the next boundary.
    """

    max_runtime_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)


__all__ = [
    "EVENT_TYPE_BY_KIND",
    "FINAL_STATUSES",
    "Command",
    "CommandAcceptedResponse",
    "CommandReceipt",
    "CommandRejectedResponse",
    "EventKind",
    "EventType",
    "RunEvent",
    "RunLimits",
    "RunStatus",
    "Stage",
]
