"""The session scores, guards, and accepts; the backend replays and falls back."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nori_rel.feature import CUTOFF_COLUMN
from nori_rel.feature.agentic import (
    AgenticFeatureConfig,
    AgenticFeatures,
    FeatureProgram,
    FeatureQuery,
    PlanningSession,
    ProgramRejected,
    RidgeScorer,
    ScoreReport,
    Trial,
)
from nori_rel.feature.dfs.noridfs import AnchorDefinition, DatabaseView, NoriDFS
from nori_rel.run import TrialOutcome

CONFIG = AgenticFeatureConfig(
    max_trials=4,
    max_features=6,
    holdout_fraction=0.3,
    confirmation_fraction=0.2,
    leak_check_anchors=2,
)


def _session(snapshot, anchors, target, **kwargs) -> PlanningSession:
    view = DatabaseView.from_parts(
        tables=snapshot.tables,
        primary_keys=snapshot.primary_keys,
        foreign_keys=snapshot.foreign_keys,
        time_columns=snapshot.time_columns,
    )
    return PlanningSession(
        database=view,
        anchors=AnchorDefinition.create(
            entity_table="accounts",
            entity_column="account_id",
            cutoff_column=CUTOFF_COLUMN,
        ),
        config=kwargs.pop("config", CONFIG),
        train_anchors=anchors(10, start="2020-01-03"),
        train_target=target,
        scorer=kwargs.pop("scorer", RidgeScorer(max_context_rows=4)),
        **kwargs,
    )


def _target() -> pd.Series:
    return pd.Series([float(i) for i in range(10)])


def test_schema_profile_and_exploration(snapshot, anchors, sql) -> None:
    session = _session(snapshot, anchors, _target())
    try:
        schema = session.describe_schema()
        assert schema["eval_columns"] == ["row_id", "account_id", "cutoff"]
        assert schema["tables"]["events"]["time_column"] == "timestamp"
        assert schema["tables"]["accounts"]["columns"]["region"] == "categorical"

        profile = session.profile_column("events", "value")
        assert profile["kind"] == "numeric"
        assert "quantiles" in profile
        assert session.profile_column("accounts", "region")["top_values"]["west"] == 2

        assert len(session.run_sql("SELECT * FROM eval_table")) == 10
        with pytest.raises(ProgramRejected, match="blocked"):
            session.run_sql("DROP TABLE events")
        with pytest.raises(KeyError):
            session.profile_column("accounts", "nope")
    finally:
        session.close()


def test_the_schema_tells_the_agent_which_columns_are_prose(snapshot, anchors) -> None:
    """A description has to profile as text, not as a category with no levels.

    The agent only learns a column exists through the schema, so a prose column
    reported as categorical is one it will either skip or bucket, and the 16
    columns of width it costs can only be judged against its mean length.
    """
    snapshot.tables["accounts"] = snapshot.tables["accounts"].assign(
        notes=[
            "this account was reviewed in the most recent compliance cycle",
            "no findings were recorded against this account during the period",
            "the account changed owner twice and carries an open remediation",
        ]
    )
    session = _session(snapshot, anchors, _target())
    try:
        profile = session.profile_column("accounts", "notes")
        region = session.profile_column("accounts", "region")
    finally:
        session.close()
    assert profile["kind"] == "text"
    assert profile["mean_length"] > 20
    assert profile["sample"].startswith("this account was reviewed")
    assert region["kind"] == "categorical"


def test_a_hidden_leak_never_becomes_the_best(snapshot, anchors, sql) -> None:
    """With no incumbent the margin gate is skipped, so the leak check runs."""
    session = _session(snapshot, anchors, _target())
    try:
        leaky = session.validate_program(
            FeatureProgram((FeatureQuery("hidden", sql["hidden"]),))
        )
        assert not leaky.accepted
        assert leaky.outcome is TrialOutcome.FAILED_POINT_IN_TIME_CHECK
        assert "leakage" in leaky.note
        assert session.best is None
        assert session.best_program() is None
        assert not session.confirm()
    finally:
        session.close()


def test_accepts_scores_and_reports_coverage(snapshot, anchors, sql) -> None:
    session = _session(snapshot, anchors, _target())
    try:
        first = session.validate_program(
            FeatureProgram((FeatureQuery("hist", sql["safe"]),))
        )
        assert first.accepted
        assert first.outcome is TrialOutcome.NEW_BEST
        assert first.feature_count == 2
        assert first.report.metric == "neg_mae"
        assert first.coverage["hist"] == 1.0
        assert session.best is first

        with pytest.raises(ProgramRejected, match="forward-looking"):
            session.validate_program(
                FeatureProgram((FeatureQuery("leak", sql["leaky"]),))
            )

        flat = session.validate_program(
            FeatureProgram((FeatureQuery("flat", sql["flat"]),))
        )
        assert not flat.accepted
        assert flat.outcome is TrialOutcome.NOT_SIGNIFICANT_GAIN
        assert "not a significant gain" in flat.note
        assert session.best is first
        assert session.best_program() is first.program
    finally:
        session.close()


class _ScriptedScorer:
    """Return exact scores so acceptance tests do not depend on model noise."""

    def __init__(self, scores: list[float], noise: float = 0.0) -> None:
        self.scores = iter(scores)
        self.noise = noise

    def __call__(self, *args, **kwargs) -> ScoreReport:
        return ScoreReport("neg_mae", next(self.scores), self.noise, predictions=None)


@pytest.mark.parametrize("baseline_score", [-1.0, 1.0])
@pytest.mark.parametrize("gain, accepted", [(1e-6, False), (0.02, True)])
def test_relative_gain_gates_trials_and_confirmation(
    snapshot, anchors, sql, baseline_score, gain, accepted
) -> None:
    candidate_score = baseline_score + gain
    session = _session(
        snapshot,
        anchors,
        _target(),
        config=AgenticFeatureConfig(min_relative_gain=0.01, leak_check_anchors=0),
        baseline_features=pd.DataFrame({"base": range(10)}),
        scorer=_ScriptedScorer(
            [baseline_score, candidate_score, candidate_score, baseline_score]
        ),
    )
    try:
        trial = session.validate_program(
            FeatureProgram((FeatureQuery("candidate", sql["flat"]),))
        )
        assert trial.accepted is accepted
        assert session.confirm() is accepted
        if not accepted:
            assert "baseline" in trial.note
            assert session.best is None
    finally:
        session.close()


def test_relative_gain_also_gates_incumbent_replacement(snapshot, anchors, sql) -> None:
    session = _session(
        snapshot,
        anchors,
        _target(),
        config=AgenticFeatureConfig(min_relative_gain=0.01, leak_check_anchors=0),
        baseline_features=pd.DataFrame({"base": range(10)}),
        scorer=_ScriptedScorer([-1.0, -0.9, -0.9 + 1e-6, -0.88]),
    )
    try:
        first = session.validate_program(
            FeatureProgram((FeatureQuery("first", sql["flat"]),))
        )
        tiny_gain = session.validate_program(
            FeatureProgram((FeatureQuery("tiny_gain", sql["flat"]),))
        )
        assert first.accepted
        assert not tiny_gain.accepted
        assert session.best is first
        meaningful_gain = session.validate_program(
            FeatureProgram((FeatureQuery("meaningful_gain", sql["flat"]),))
        )
        assert meaningful_gain.accepted
        assert session.best is meaningful_gain
    finally:
        session.close()


@pytest.mark.parametrize("baseline_score", [-1.0, 1.0])
def test_confirmation_rejects_tiny_gain_after_a_meaningful_trial(
    snapshot, anchors, sql, baseline_score
) -> None:
    session = _session(
        snapshot,
        anchors,
        _target(),
        config=AgenticFeatureConfig(min_relative_gain=0.01, leak_check_anchors=0),
        baseline_features=pd.DataFrame({"base": range(10)}),
        scorer=_ScriptedScorer(
            [
                baseline_score,
                baseline_score + 0.1,
                baseline_score + 1e-6,
                baseline_score,
            ]
        ),
    )
    try:
        trial = session.validate_program(
            FeatureProgram((FeatureQuery("candidate", sql["flat"]),))
        )
        assert trial.accepted
        assert not session.confirm()
    finally:
        session.close()


@pytest.mark.parametrize("gain, accepted", [(0.02, False), (0.04, True)])
def test_score_gain_uses_the_larger_noise_or_relative_margin(
    snapshot, anchors, sql, gain, accepted
) -> None:
    session = _session(
        snapshot,
        anchors,
        _target(),
        config=AgenticFeatureConfig(min_relative_gain=0.02, leak_check_anchors=0),
        baseline_features=pd.DataFrame({"base": range(10)}),
        scorer=_ScriptedScorer([-1.0, -1.0 + gain, -1.0 + gain, -1.0], noise=0.015),
    )
    try:
        trial = session.validate_program(
            FeatureProgram((FeatureQuery("candidate", sql["flat"]),))
        )
        assert trial.accepted is accepted
        assert session.confirm() is accepted
    finally:
        session.close()


@pytest.mark.parametrize(
    "scores, trial_accepted",
    [([0.0, 0.5], False), ([-1.0, -0.9, 0.5, 0.0], True)],
)
def test_relative_gain_refuses_zero_reference_scores(
    snapshot, anchors, sql, scores, trial_accepted
) -> None:
    session = _session(
        snapshot,
        anchors,
        _target(),
        config=AgenticFeatureConfig(min_relative_gain=0.01, leak_check_anchors=0),
        baseline_features=pd.DataFrame({"base": range(10)}),
        scorer=_ScriptedScorer(scores),
    )
    try:
        trial = session.validate_program(
            FeatureProgram((FeatureQuery("candidate", sql["flat"]),))
        )
        assert trial.accepted is trial_accepted
        assert not session.confirm()
    finally:
        session.close()


def test_zero_relative_gain_preserves_the_historical_acceptance_protocol(
    snapshot, anchors, sql
) -> None:
    session = _session(
        snapshot,
        anchors,
        _target(),
        baseline_features=pd.DataFrame({"base": range(10)}),
        scorer=_ScriptedScorer([-1.0, -2.0, -2.0 + 1e-6, -1.0 + 1e-6, -1.0]),
    )
    try:
        below_baseline = session.validate_program(
            FeatureProgram((FeatureQuery("first", sql["flat"]),))
        )
        tiny_gain = session.validate_program(
            FeatureProgram((FeatureQuery("tiny_gain", sql["flat"]),))
        )
        assert below_baseline.accepted
        assert "below the baseline" in below_baseline.note
        assert tiny_gain.accepted
        assert session.confirm()
    finally:
        session.close()


@pytest.mark.parametrize("min_relative_gain", [-0.01, 1.0, float("inf"), float("nan")])
def test_relative_gain_must_be_finite_and_in_range(min_relative_gain) -> None:
    with pytest.raises(ValueError, match="min_relative_gain"):
        AgenticFeatureConfig(min_relative_gain=min_relative_gain)


def test_on_trial_scored_receives_each_trials_outcome(snapshot, anchors, sql) -> None:
    scored: list[Trial] = []
    session = _session(snapshot, anchors, _target(), on_trial_scored=scored.append)
    try:
        for name in ("hidden", "safe", "flat"):
            session.validate_program(FeatureProgram((FeatureQuery(name, sql[name]),)))
    finally:
        session.close()

    assert [(trial.outcome, trial.accepted) for trial in scored] == [
        (TrialOutcome.FAILED_POINT_IN_TIME_CHECK, False),
        (TrialOutcome.NEW_BEST, True),
        (TrialOutcome.NOT_SIGNIFICANT_GAIN, False),
    ]
    assert scored == session.trials


def test_a_failing_trial_callback_does_not_refuse_the_trial(
    snapshot, anchors, sql
) -> None:
    def broken_callback(trial: Trial) -> None:
        raise RuntimeError("event stream closed")

    session = _session(snapshot, anchors, _target(), on_trial_scored=broken_callback)
    try:
        trial = session.validate_program(
            FeatureProgram((FeatureQuery("safe", sql["safe"]),))
        )
    finally:
        session.close()

    assert trial.accepted
    assert session.refusals == []


class _FixedMaeScorer:
    """MAE 1.0 on the baseline's columns and ``program_mae`` on any others."""

    def __init__(self, program_mae: float) -> None:
        self.program_mae = program_mae

    def __call__(self, context, target, holdout, holdout_target, **kwargs):
        mae = 1.0 if "base" in context.columns else self.program_mae
        return ScoreReport(
            metric="neg_mae", score=-mae, noise=0.0, predictions=np.zeros(len(holdout))
        )


def test_a_trial_below_the_baseline_is_reported_as_best_below_baseline(
    snapshot, anchors, sql
) -> None:
    scored: list[Trial] = []
    session = _session(
        snapshot,
        anchors,
        _target(),
        baseline_features=pd.DataFrame({"base": [1.0] * 10}),
        scorer=_FixedMaeScorer(program_mae=2.0),
        on_trial_scored=scored.append,
    )
    try:
        trial = session.validate_program(
            FeatureProgram((FeatureQuery("hist", sql["safe"]),))
        )
        assert session.baseline is not None and session.baseline.score == -1.0
        assert trial.accepted
        assert trial.outcome is TrialOutcome.BEST_BELOW_BASELINE
        assert session.best is trial
        assert not session.confirm()
    finally:
        session.close()
    assert scored == [trial]


def test_budgets_are_enforced(snapshot, anchors, sql) -> None:
    session = _session(
        snapshot,
        anchors,
        _target(),
        config=AgenticFeatureConfig(max_trials=1, max_queries=1, max_features=1),
    )
    try:
        with pytest.raises(ProgramRejected, match="1 queries"):
            session.validate_program(
                FeatureProgram(
                    (
                        FeatureQuery("a", sql["flat"]),
                        FeatureQuery("b", sql["flat"].replace("one", "two")),
                    )
                )
            )
        with pytest.raises(ProgramRejected, match="budget is 1"):
            session.validate_program(
                FeatureProgram((FeatureQuery("hist", sql["safe"]),))
            )
        session.validate_program(FeatureProgram((FeatureQuery("f", sql["flat"]),)))
        with pytest.raises(ProgramRejected, match="trial budget"):
            session.validate_program(FeatureProgram((FeatureQuery("g", sql["flat"]),)))
    finally:
        session.close()


class _ScriptedAgent:
    """Submits one program, like a model that got it right the first time."""

    def __init__(self, program: FeatureProgram) -> None:
        self.program = program
        self.hints: list[str] = []

    def run(self, session: PlanningSession, task_hint: str) -> None:
        self.hints.append(task_hint)
        session.validate_program(self.program)


def test_backend_replays_a_fixed_program(snapshot, anchors, sql) -> None:
    train = anchors(6, start="2020-01-03").assign(row_id=[f"c{i}" for i in range(6)])
    query = anchors(2, start="2020-01-10").assign(row_id=["q1", "q2"])
    backend = AgenticFeatures(
        program=FeatureProgram((FeatureQuery("hist", sql["safe"]),))
    )
    matrices = backend.compute(
        snapshot, train, query, "accounts", row_id_column="row_id"
    )
    assert matrices.feature_names == ["hist__n_events", "hist__mean_value"]
    assert len(matrices.train) == 6
    assert len(matrices.query) == 2


def test_backend_searches_and_beats_the_baseline(snapshot, anchors, sql) -> None:
    train = anchors(10, start="2020-01-03").assign(row_id=[f"c{i}" for i in range(10)])
    query = anchors(2, start="2020-01-14").assign(row_id=["q1", "q2"])
    agent = _ScriptedAgent(FeatureProgram((FeatureQuery("hist", sql["safe"]),)))
    backend = AgenticFeatures(
        agent=agent,
        scorer=RidgeScorer(max_context_rows=4),
        train_target=_target(),
        config=CONFIG,
        task_hint="predict activity",
    )
    matrices = backend.compute(
        snapshot, train, query, "accounts", row_id_column="row_id"
    )
    assert agent.hints == ["predict activity"]
    assert matrices.feature_names == ["hist__n_events", "hist__mean_value"]
    assert backend.last_program is agent.program


def test_a_refused_program_is_recorded_not_only_returned(
    snapshot, anchors, sql
) -> None:
    """The agent is told why; nothing else was keeping it.

    A search that scores nothing otherwise reports only "no accepted
    program", which cannot tell a model that never called the tool from one
    whose every program hit the guard, and those need opposite fixes.
    """
    session = _session(snapshot, anchors, _target())
    leaky = FeatureQuery(
        "leak",
        "SELECT e.row_id, COUNT(*) AS n FROM eval_table e JOIN events ev "
        f"ON ev.account_id = e.account_id AND ev.timestamp >= e.{CUTOFF_COLUMN} "
        "GROUP BY e.row_id",
    )
    try:
        with pytest.raises(ProgramRejected):
            session.validate_program(FeatureProgram((leaky,)))
        assert session.trials == []
        assert len(session.refusals) == 1
        assert "ProgramRejected" in session.refusals[0]
    finally:
        session.close()


def test_the_backend_reports_the_kinds_its_matrices_cannot_carry(
    snapshot, anchors, sql
) -> None:
    """A caller has to be told which columns are prose.

    `FeatureMatrices` carries frames and names only, and dtypes cannot
    separate prose from a category, so fitting Nori from the matrices alone
    would score a description as a categorical with a level per row.
    """
    features = AgenticFeatures(
        program=FeatureProgram((FeatureQuery("hist", sql["safe"]),))
    )
    rows = anchors(6, start="2020-01-05")
    matrices = features.compute(
        snapshot,
        rows.iloc[:4].assign(row_id=range(4)),
        rows.iloc[4:].reset_index(drop=True).assign(row_id=[4, 5]),
        "accounts",
        row_id_column="row_id",
    )

    assert list(matrices.train.columns) == ["hist__n_events", "hist__mean_value"]
    assert features.last_categorical == ()
    assert features.last_text == ()


def test_backend_falls_back_when_confirmation_fails(snapshot, anchors, sql) -> None:
    train = anchors(10, start="2020-01-03").assign(row_id=[f"c{i}" for i in range(10)])
    query = anchors(2, start="2020-01-14").assign(row_id=["q1", "q2"])
    agent = _ScriptedAgent(FeatureProgram((FeatureQuery("flat", sql["flat"]),)))
    fallback = NoriDFS()
    backend = AgenticFeatures(
        agent=agent,
        scorer=RidgeScorer(max_context_rows=4),
        train_target=_target(),
        fallback=fallback,
        config=CONFIG,
    )
    matrices = backend.compute(
        snapshot, train, query, "accounts", row_id_column="row_id"
    )
    expected = fallback.compute(
        snapshot, train, query, "accounts", row_id_column="row_id"
    )
    assert matrices.feature_names == expected.feature_names
    assert backend.last_program is None
