"""User-facing event messages with sanitized failure details."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Final

from nori_rel.run.models import EventKind, Stage
from nori_rel.run.redact import redact

STAGE_LABELS: Final[dict[Stage, str]] = {
    Stage.MATERIALIZE: "reading the database snapshot",
    Stage.PLAN_SEARCH: "planning features",
    Stage.PREDICT_BATCH: "predicting",
}


def _status_changed(data: Mapping[str, Any]) -> str:
    if data["to"] == "running" and data.get("from") == "queued":
        return f"Run started: {data['strategy']} search."
    if data["to"] == "running" and data.get("from") == "paused":
        return "Run resumed."
    return f"Run {data['to']}."


def _command(data: Mapping[str, Any]) -> str:
    name = data["command"].capitalize()
    if data["status"] == "rejected":
        return f"{name} rejected ({data['reason']})."
    return f"{name} applied."


_MESSAGE_RENDERERS_BY_KIND: Final[
    dict[EventKind, Callable[[Mapping[str, Any]], str]]
] = {
    EventKind.STATUS_CHANGED: _status_changed,
    EventKind.STAGE_STARTED: lambda data: f"Started {STAGE_LABELS[data['stage']]}.",
    EventKind.STAGE_FINISHED: lambda data: (
        f"Finished {STAGE_LABELS[data['stage']]} in {data['seconds']:.1f} s."
    ),
    EventKind.COMMAND: _command,
    EventKind.LIMIT_REACHED: lambda data: (
        f"Reached {data['limit']}; cancelling the run."
    ),
    EventKind.ERROR: lambda data: redact(data["message"]) or "Run failed.",
}


def render(kind: EventKind | str, data: Mapping[str, Any]) -> str:
    """Render a known event, rejecting unknown kinds and missing fields."""
    return _MESSAGE_RENDERERS_BY_KIND[EventKind(kind)](data)


__all__ = ["STAGE_LABELS", "render"]
