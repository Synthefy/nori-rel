"""User-facing event messages, rendered only from fixed templates.

A message never includes anything an agent wrote, so it can be shown to anyone
who may watch the run. Two kinds of event carry outside text: steering echoes
the caller's own input on purpose, and an error's text goes through ``redact``
here, whatever its producer did.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Final

from nori_rel.run.models import EventKind, FallbackReason, Stage, TrialOutcome
from nori_rel.run.redact import redact

STAGE_LABELS: Final[dict[Stage, str]] = {
    Stage.MATERIALIZE: "reading the database snapshot",
    Stage.PLAN_SEARCH: "planning features",
    Stage.RUN_TRIAL: "running a trial",
    Stage.EVALUATE_TRIAL: "scoring a trial",
    Stage.SELECT_PIPELINE: "confirming the best pipeline",
    Stage.PREDICT_BATCH: "predicting",
}

FALLBACK_MESSAGES: Final[dict[FallbackReason, str]] = {
    FallbackReason.NO_PROGRAM_FOUND: "The agent found no program",
    FallbackReason.NO_PROGRAM_CONFIRMED: "No program beat the baseline",
    FallbackReason.STOPPED_BEFORE_CONFIRMATION: (
        "Stopped before the best program was confirmed"
    ),
    FallbackReason.STOPPED_WITHOUT_PROGRAM: "Stopped before any program was found",
    FallbackReason.SEARCH_FAILED: "The feature search failed",
}

#: What a trial_scored message says about each outcome. A rejected score may be
#: inflated by a leak, so the failed check is stated as a rejection.
TRIAL_OUTCOME_MESSAGES: Final[dict[TrialOutcome, str]] = {
    TrialOutcome.NEW_BEST: "new best",
    TrialOutcome.BEST_BELOW_BASELINE: "best so far, but below the baseline",
    TrialOutcome.NOT_SIGNIFICANT_GAIN: "not a significant gain",
    TrialOutcome.FAILED_POINT_IN_TIME_CHECK: "rejected, failed the point-in-time check",
}

#: Steering text longer than this is shortened in its message.
_MAX_STEERING_TEXT_CHARS = 500
#: Scores at or above this magnitude are shown in full, not in e-notation.
_MIN_GROUPED_SCORE_MAGNITUDE = 1e4


def _format_score_for_message(data: Mapping[str, Any], key: str) -> str:
    """Return the score under ``key`` as text: "3.91", or "12,345.6" when large.

    Raises ``ValueError`` when the score is not a number, such as ``None``.
    """
    score = data[key]
    if isinstance(score, bool) or not isinstance(score, int | float):
        raise ValueError(f"{key} must be a number, not {score!r}")
    if abs(score) >= _MIN_GROUPED_SCORE_MAGNITUDE:
        return f"{score:,.1f}"
    return f"{score:.4g}"


def _format_metric_score(data: Mapping[str, Any], key: str) -> str:
    """Return the metric and the score under ``key``, e.g. "MAE 3.91"."""
    return f"{data['metric'].upper()} {_format_score_for_message(data, key)}"


def _status_changed(data: Mapping[str, Any]) -> str:
    if data["to"] == "running" and data.get("from") == "queued":
        trials = data.get("max_trials")
        limit = "" if trials is None else f", up to {trials} trials"
        return f"Run started: {data['strategy']} search{limit}."
    if data["to"] == "running" and data.get("from") == "paused":
        return "Run resumed."
    return f"Run {data['to']}."


def _trial_outcome_suffix(data: Mapping[str, Any]) -> str:
    """Return "; <outcome>" for a known ``data["outcome"]``, else nothing.

    Hosted or older events carry no outcome, and a newer producer may send one
    this version does not know; both render as before.
    """
    try:
        outcome = TrialOutcome(data.get("outcome"))
    except ValueError:
        return ""
    return f"; {TRIAL_OUTCOME_MESSAGES[outcome]}"


def _trial_scored(data: Mapping[str, Any]) -> str:
    of = "" if data.get("max_trials") is None else f"/{data['max_trials']}"
    context = [
        f"{name} {_format_score_for_message(data, key)}"
        for name, key in (("best", "best_score"), ("baseline", "baseline_score"))
        if data.get(key) is not None
    ]
    detail = f" ({', '.join(context)})" if context else ""
    score = _format_metric_score(data, "score")
    return f"Trial {data['trial']}{of}: {score}{detail}{_trial_outcome_suffix(data)}."


def _command(data: Mapping[str, Any]) -> str:
    name = data["command"].capitalize()
    if data["status"] == "rejected":
        return f"{name} rejected ({data['reason']})."
    text = data.get("text")
    if text is None:
        return f"{name} applied."
    if len(text) > _MAX_STEERING_TEXT_CHARS:
        text = text[: _MAX_STEERING_TEXT_CHARS - 1] + "…"
    return f'{name} applied: "{text}"'


_TEMPLATES: Final[dict[EventKind, Callable[[Mapping[str, Any]], str]]] = {
    EventKind.STATUS_CHANGED: _status_changed,
    EventKind.STAGE_STARTED: lambda data: f"Started {STAGE_LABELS[data['stage']]}.",
    EventKind.STAGE_FINISHED: lambda data: (
        f"Finished {STAGE_LABELS[data['stage']]} in {data['seconds']:.1f} s."
    ),
    EventKind.BASELINE_SCORED: lambda data: (
        f"Baseline: {_format_metric_score(data, 'score')} "
        f"({data['feature_count']} features)."
    ),
    EventKind.TRIAL_SCORED: _trial_scored,
    EventKind.SPEND_UPDATED: lambda data: (
        f"Agent spend: ${data['spend_usd']:.2f} ({data['llm_calls']} LLM calls)."
    ),
    EventKind.COMMAND: _command,
    EventKind.FALLBACK_USED: lambda data: (
        f"{FALLBACK_MESSAGES[data['reason']]}; using {data['to_strategy']}."
    ),
    EventKind.SEARCH_STOPPED: lambda data: (
        f"Search stopped after {data['trials']} trials; finishing with the best "
        "result so far."
    ),
    EventKind.LIMIT_REACHED: lambda data: (
        f"Reached {data['limit']}; finishing with the best result so far."
    ),
    EventKind.ERROR: lambda data: redact(data["message"]) or "Run failed.",
}


def render(kind: EventKind | str, data: Mapping[str, Any]) -> str:
    """Return the message for an event of ``kind`` with ``data``.

    Raises ``ValueError`` for an unknown kind or a score that is not a number,
    and ``KeyError`` for a missing field, so a malformed event fails where it is
    built rather than reaching a caller.
    """
    return _TEMPLATES[EventKind(kind)](data)


__all__ = ["FALLBACK_MESSAGES", "STAGE_LABELS", "TRIAL_OUTCOME_MESSAGES", "render"]
