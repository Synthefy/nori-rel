"""An agent search reports its baseline, trials, spend and outcome to its run."""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import pytest

from nori_rel.feature.agentic import (
    AgenticFeatureConfig,
    AgenticFeatures,
    FeatureProgram,
    FeatureQuery,
    PlanningSession,
    ScoreReport,
)
from nori_rel.feature.agentic.run_reporting import displayed_metric_score
from nori_rel.feature.dfs.noridfs import NoriDFS
from nori_rel.run import RunCancelled, RunContext, RunEvent, Stage

EXTEND = AgenticFeatureConfig(
    max_trials=4,
    max_features=6,
    holdout_fraction=0.3,
    confirmation_fraction=0.2,
    leak_check_anchors=2,
    extend_baseline=True,
)
PROGRAM_COLUMN = "hist__n_events"


class _FixedScorer:
    """MAE 1.0 for the baseline; ``program_mae`` once the program's columns join."""

    def __init__(self, program_mae: float, score_type=float) -> None:
        self.program_mae = program_mae
        self.score_type = score_type

    def __call__(self, context, target, holdout, holdout_target, **kwargs):
        mae = self.program_mae if PROGRAM_COLUMN in context.columns else 1.0
        return ScoreReport(
            metric="neg_mae",
            score=self.score_type(-mae),
            noise=0.0,
            predictions=np.zeros(len(holdout)),
        )


class _SpendingAgent:
    """Reports one model response through the run, then submits one program.

    It keeps no state between runs, like the real agents: the spend reaches
    the run only through the session.
    """

    def __init__(self, program: FeatureProgram, response_cost_usd=0.25) -> None:
        self.program = program
        self.response_cost_usd = response_cost_usd
        self.during_run = None

    def run(self, session: PlanningSession, task_hint: str) -> None:
        session.report_model_response(self.response_cost_usd)
        session.validate_program(self.program)
        if self.during_run is not None:
            self.during_run()


def _search_backend(
    sql,
    events: list[RunEvent],
    program_mae: float,
    program: str = "safe",
    **options,
) -> tuple[AgenticFeatures, _SpendingAgent]:
    agent = _SpendingAgent(
        FeatureProgram((FeatureQuery("hist", sql[program]),)),
        options.pop("response_cost_usd", 0.25),
    )
    backend = AgenticFeatures(
        agent=agent,
        scorer=_FixedScorer(program_mae, options.pop("score_type", float)),
        train_target=pd.Series([float(i) for i in range(10)]),
        fallback=options.pop("fallback", NoriDFS()),
        config=options.pop("config", EXTEND),
        run_context=options.pop(
            "run_context", RunContext("run_1", event_sink=events.append)
        ),
    )
    return backend, agent


def _compute(backend: AgenticFeatures, anchors, snapshot) -> None:
    train = anchors(10, start="2020-01-03").assign(row_id=[f"c{i}" for i in range(10)])
    query = anchors(2, start="2020-01-14").assign(row_id=["q1", "q2"])
    backend.compute(snapshot, train, query, "accounts", row_id_column="row_id")


def _search(snapshot, anchors, sql, program_mae: float, program="safe", **options):
    events: list[RunEvent] = []
    backend, agent = _search_backend(sql, events, program_mae, program, **options)
    _compute(backend, anchors, snapshot)
    return events, agent


def test_a_winning_search_reports_baseline_trial_spend_and_selection(
    snapshot, anchors, sql
) -> None:
    events, agent = _search(snapshot, anchors, sql, program_mae=0.5)

    kinds = [event.data["kind"] for event in events]
    assert kinds == [
        "baseline_scored",
        "spend_updated",
        "trial_scored",
        "stage_started",
        "stage_finished",
    ]
    baseline, spend, trial = events[:3]
    assert (baseline.data["metric"], baseline.data["score"]) == ("mae", 1.0)
    assert spend.message == "Agent spend: $0.25 (1 LLM calls)."
    assert (trial.data["score"], trial.data["best_score"]) == (0.5, 0.5)
    assert trial.data["accepted"] and trial.data["max_trials"] == 4
    assert trial.data["outcome"] == "new_best"
    assert trial.message == "Trial 1/4: MAE 0.5 (best 0.5, baseline 1); new best."
    assert events[3].data["stage"] == "select_pipeline"
    assert not hasattr(agent, "on_usage")


def test_a_losing_search_reports_the_fallback(snapshot, anchors, sql) -> None:
    events, _ = _search(snapshot, anchors, sql, program_mae=2.0)

    # The first program is the best so far even below the baseline; the
    # baseline wins only at confirmation, where the fallback is reported.
    trial = next(event for event in events if event.data["kind"] == "trial_scored")
    assert (trial.data["score"], trial.data["baseline_score"]) == (2.0, 1.0)
    assert trial.data["accepted"]
    assert trial.data["outcome"] == "best_below_baseline"
    assert trial.message == (
        "Trial 1/4: MAE 2 (best 2, baseline 1); best so far, but below the baseline."
    )
    assert events[-1].data == {
        "kind": "fallback_used",
        "reason": "no_program_confirmed",
        "to_strategy": "noridfs",
    }
    assert events[-1].message == (
        "The best program did not beat the baseline on the confirmation rows; "
        "using noridfs."
    )


def test_a_trial_that_fails_the_point_in_time_check_is_reported_as_rejected(
    snapshot, anchors, sql
) -> None:
    events, _ = _search(snapshot, anchors, sql, program_mae=0.5, program="hidden")

    trial = next(event for event in events if event.data["kind"] == "trial_scored")
    assert not trial.data["accepted"]
    assert trial.data["outcome"] == "failed_point_in_time_check"
    assert "best_score" not in trial.data
    assert trial.message == (
        "Trial 1/4: MAE 1 (baseline 1); rejected, failed the point-in-time check."
    )
    # The event gives the code, never the agent's note, queries or feature names.
    assert "total_value" not in str(trial.data) + trial.message
    assert "leakage" not in str(trial.data) + trial.message
    assert events[-1].data["kind"] == "fallback_used"


def test_an_agent_without_prices_reports_no_spend(snapshot, anchors, sql) -> None:
    events, _ = _search(snapshot, anchors, sql, program_mae=0.5, response_cost_usd=None)

    assert "spend_updated" not in [event.data["kind"] for event in events]


def test_numpy_scores_are_reported_as_floats(snapshot, anchors, sql) -> None:
    events, _ = _search(snapshot, anchors, sql, program_mae=0.5, score_type=np.float32)

    baseline = events[0]
    assert baseline.data["kind"] == "baseline_scored"
    assert type(baseline.data["score"]) is float


def test_a_search_without_a_baseline_reports_no_baseline(
    snapshot, anchors, sql
) -> None:
    events, _ = _search(
        snapshot,
        anchors,
        sql,
        program_mae=0.5,
        fallback=None,
        config=AgenticFeatureConfig(
            max_trials=4,
            max_features=6,
            holdout_fraction=0.3,
            confirmation_fraction=0.2,
            leak_check_anchors=2,
        ),
    )

    kinds = [event.data["kind"] for event in events]
    assert "baseline_scored" not in kinds
    trial = events[kinds.index("trial_scored")]
    assert "baseline_score" not in trial.data


def test_a_cancel_at_the_select_pipeline_boundary_closes_the_session(
    snapshot, anchors, sql
) -> None:
    events: list[RunEvent] = []
    run_context = RunContext("run_1", event_sink=events.append)
    backend, agent = _search_backend(
        sql, events, program_mae=0.5, run_context=run_context
    )
    agent.during_run = lambda: run_context.commands.submit_command("cancel")

    with pytest.raises(RunCancelled):
        _compute(backend, anchors, snapshot)

    assert backend.last_session is not None
    with pytest.raises(duckdb.ConnectionException):
        backend.last_session._executor._connection.execute("SELECT 1")


def test_select_pipeline_reports_its_outer_stage(snapshot, anchors, sql) -> None:
    events: list[RunEvent] = []
    run_context = RunContext("run_1", event_sink=events.append)
    backend, _ = _search_backend(sql, events, program_mae=0.5, run_context=run_context)

    with run_context.report_stage(Stage.PLAN_SEARCH):
        _compute(backend, anchors, snapshot)

    stages = [
        (event.data["kind"], event.data["stage"], event.data.get("outer_stage"))
        for event in events
        if event.data["kind"].startswith("stage_")
    ]
    assert stages == [
        ("stage_started", "plan_search", None),
        ("stage_started", "select_pipeline", None),
        ("stage_finished", "select_pipeline", "plan_search"),
        ("stage_finished", "plan_search", None),
    ]


def test_scores_are_shown_as_people_read_them() -> None:
    assert displayed_metric_score("neg_mae", -3.8) == ("mae", 3.8)
    assert displayed_metric_score("roc_auc", 0.71) == ("roc_auc", 0.71)
