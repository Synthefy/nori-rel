"""Report an agent search's baseline, trials, spend and fallback to its run.

The search scores higher-is-better (negative MAE, WAPE or MAPE; ROC-AUC); a run
reports the metric people read, so a negative error is shown as the error.
"""

from __future__ import annotations

from typing import Any

from nori_rel.run import EventKind, FallbackReason, RunContext

from .contracts import ScoreReport, Trial
from .session import PlanningSession

_NEGATED_METRICS = {"neg_mae": "mae", "neg_wape": "wape", "neg_mape": "mape"}


def displayed_metric_score(metric: str, score: float) -> tuple[str, float]:
    """Return the metric name and score as a person reads them.

    The score is a plain ``float``: a custom scorer may return a NumPy scalar
    such as ``np.float32``, which an event's JSON data does not accept.
    """
    if metric in _NEGATED_METRICS:
        return _NEGATED_METRICS[metric], -float(score)
    return metric, float(score)


def _displayed_report_score(report: ScoreReport) -> float:
    return displayed_metric_score(report.metric, report.score)[1]


class AgenticSearchReporter:
    """Reports one agentic feature search's progress to its run.

    The best trial and baseline are read from ``session`` when reported, so the
    reporter keeps no copy of the search's state.
    """

    def __init__(
        self, run_context: RunContext, session: PlanningSession, *, max_trials: int
    ) -> None:
        """Report ``session``'s progress, out of ``max_trials``, to ``run_context``."""
        self.run_context = run_context
        self.session = session
        self.max_trials = max_trials
        #: Model responses so far in this search.
        self.llm_calls = 0

    def report_baseline_score(self, feature_count: int) -> None:
        """Report the DFS baseline the search must beat, if it was scored."""
        baseline = self.session.baseline
        if baseline is None:
            return
        metric, score = displayed_metric_score(baseline.metric, baseline.score)
        self.run_context.emit_event(
            EventKind.BASELINE_SCORED,
            metric=metric,
            score=score,
            feature_count=feature_count,
        )

    def report_scored_trial(self, trial: Trial) -> None:
        """Report one scored trial with the best and baseline scores so far.

        The event says what the search decided (``outcome``) but never why in
        words: the trial's note names the agent's queries and anchor rows.
        """
        metric, score = displayed_metric_score(trial.report.metric, trial.report.score)
        data: dict[str, Any] = {
            "trial": trial.index,
            "max_trials": self.max_trials,
            "metric": metric,
            "score": score,
            "accepted": trial.accepted,
            "outcome": trial.outcome.value,
            "feature_count": trial.feature_count,
        }
        if self.session.best is not None:
            data["best_score"] = _displayed_report_score(self.session.best.report)
        if self.session.baseline is not None:
            data["baseline_score"] = _displayed_report_score(self.session.baseline)
        self.run_context.emit_event(EventKind.TRIAL_SCORED, **data)

    def report_model_response(self, spend_usd: float | None) -> None:
        """Count one model response and report the search's spend so far.

        An agent without prices reports ``None``, and no spend event is sent.
        """
        self.llm_calls += 1
        if spend_usd is not None:
            self.run_context.record_spend(float(spend_usd))
            self.run_context.emit_event(
                EventKind.SPEND_UPDATED,
                spend_usd=float(spend_usd),
                llm_calls=self.llm_calls,
            )


def report_search_stopped(run_context: RunContext, trials: int) -> None:
    """Report that a stop ended the search after ``trials`` scored trials."""
    run_context.emit_event(EventKind.SEARCH_STOPPED, trials=trials)


def report_fallback_to_baseline(
    run_context: RunContext, reason: FallbackReason
) -> None:
    """Report that the prediction uses NoriDFS instead of an agent program."""
    run_context.emit_event(
        EventKind.FALLBACK_USED, reason=reason.value, to_strategy="noridfs"
    )


__all__ = [
    "AgenticSearchReporter",
    "displayed_metric_score",
    "report_fallback_to_baseline",
    "report_search_stopped",
]
