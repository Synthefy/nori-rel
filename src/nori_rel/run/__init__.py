"""Run progress, commands and cancellation; see ``docs/run-control.md``.

This package depends only on pydantic, so it imports without the database,
agent or Nori extras.
"""

from nori_rel.run.commands import CommandInbox, QueuedCommand
from nori_rel.run.context import RunCancelled, RunContext
from nori_rel.run.errors import (
    PredictionCancelled,
    PredictionFailed,
    RunError,
    RunTimeout,
)
from nori_rel.run.executor import LocalExecutor
from nori_rel.run.local_run import LocalRun, PredictionRun, new_run_id
from nori_rel.run.messages import render
from nori_rel.run.models import (
    EVENT_TYPE_BY_KIND,
    FINAL_STATUSES,
    Command,
    CommandAcceptedResponse,
    CommandReceipt,
    CommandRejectedResponse,
    EventKind,
    EventType,
    FallbackReason,
    RunEvent,
    RunLimits,
    RunStatus,
    Stage,
    TrialOutcome,
)
from nori_rel.run.progress import (
    RunProgress,
    RunStatusResponse,
    apply_event_to_progress,
    progress_from_events,
)
from nori_rel.run.redact import redact

__all__ = [
    "EVENT_TYPE_BY_KIND",
    "FINAL_STATUSES",
    "Command",
    "CommandAcceptedResponse",
    "CommandInbox",
    "CommandReceipt",
    "CommandRejectedResponse",
    "EventKind",
    "EventType",
    "FallbackReason",
    "LocalExecutor",
    "LocalRun",
    "PredictionCancelled",
    "PredictionFailed",
    "PredictionRun",
    "QueuedCommand",
    "RunCancelled",
    "RunContext",
    "RunError",
    "RunEvent",
    "RunLimits",
    "RunProgress",
    "RunStatus",
    "RunStatusResponse",
    "RunTimeout",
    "Stage",
    "TrialOutcome",
    "apply_event_to_progress",
    "new_run_id",
    "progress_from_events",
    "redact",
    "render",
]
