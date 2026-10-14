"""An extending search scores DFS plus the program, and returns both."""

from __future__ import annotations

import json
from unittest.mock import Mock

import pandas as pd
import pytest

from nori_rel.feature import CUTOFF_COLUMN, FeatureBackend, FeatureMatrices
from nori_rel.feature.agentic import (
    AgenticFeatureConfig,
    AgenticFeatures,
    FeatureProgram,
    FeatureQuery,
    PlanningSession,
    ProgramAgentBase,
    ProgramRejected,
    RidgeScorer,
)
from nori_rel.feature.agentic._agent_base import parse_program
from nori_rel.feature.agentic.baseline import describe_plan, unread_columns
from nori_rel.feature.dfs.noridfs import (
    AnchorDefinition,
    DatabaseView,
    FeaturePlan,
    FeatureSpec,
    ForeignKey,
    NativeFeatureConfig,
    NoriDFS,
    RelationPath,
)

CONFIG = AgenticFeatureConfig(
    max_trials=4,
    max_features=6,
    holdout_fraction=0.3,
    confirmation_fraction=0.2,
    leak_check_anchors=2,
)
EXTEND = AgenticFeatureConfig(
    max_trials=4,
    max_features=6,
    holdout_fraction=0.3,
    confirmation_fraction=0.2,
    leak_check_anchors=2,
    extend_baseline=True,
)


class _RecordingScorer:
    """Ridge that remembers which columns each call was scored on."""

    def __init__(self) -> None:
        self.inner = RidgeScorer(max_context_rows=4)
        self.columns: list[list[str]] = []
        self.categorical: list[tuple[str, ...]] = []

    def __call__(self, context, target, holdout, holdout_target, **kwargs):
        self.columns.append(list(context.columns))
        self.categorical.append(tuple(kwargs["categorical_columns"]))
        return self.inner(context, target, holdout, holdout_target, **kwargs)


class _ScriptedAgent:
    """Submits one program, like a model that got it right the first time."""

    def __init__(self, program: FeatureProgram) -> None:
        self.program = program

    def run(self, session: PlanningSession, task_hint: str) -> None:
        session.validate_program(self.program)


class _UnusedBackend(NoriDFS):
    """A fallback that fails the test if anything computes it."""

    def compute(self, *args, **kwargs):
        raise AssertionError("the fallback should not have been computed")


def _baseline() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dfs_rate": [float(i % 3) for i in range(10)],
            "dfs_region": ["east", "west"] * 5,
        }
    )


def _view(snapshot) -> DatabaseView:
    return DatabaseView.from_parts(
        tables=snapshot.tables,
        primary_keys=snapshot.primary_keys,
        foreign_keys=snapshot.foreign_keys,
        time_columns=snapshot.time_columns,
    )


_DAY_NS = 24 * 60 * 60 * 1_000_000_000


def _plan(snapshot) -> FeaturePlan:
    """Two columns: a row count over all history, a 7-day mean of value."""
    path = RelationPath(
        "accounts", (ForeignKey("events", "account_id", "accounts", "account_id"),)
    )
    return FeaturePlan(
        schema_digest="test",
        anchors=AnchorDefinition.create(
            entity_table="accounts",
            entity_column="account_id",
            cutoff_column=CUTOFF_COLUMN,
        ),
        config=NativeFeatureConfig(),
        paths=(path,),
        features=(
            FeatureSpec.aggregate(path, "count"),
            FeatureSpec.aggregate(path, "mean", column="value", window_ns=7 * _DAY_NS),
        ),
    )


def _session(
    snapshot, anchors, *, config, baseline, scorer=None, plan=None
) -> PlanningSession:
    view = _view(snapshot)
    return PlanningSession(
        database=view,
        anchors=AnchorDefinition.create(
            entity_table="accounts",
            entity_column="account_id",
            cutoff_column=CUTOFF_COLUMN,
        ),
        config=config,
        train_anchors=anchors(10, start="2020-01-03"),
        train_target=pd.Series([float(i) for i in range(10)]),
        scorer=scorer or RidgeScorer(max_context_rows=4),
        baseline_features=baseline,
        baseline_categorical=() if baseline is None else ("dfs_region",),
        baseline_plan=plan,
    )


def test_a_program_is_scored_on_top_of_the_baseline(snapshot, anchors, sql) -> None:
    scorer = _RecordingScorer()
    session = _session(
        snapshot, anchors, config=EXTEND, baseline=_baseline(), scorer=scorer
    )
    try:
        trial = session.validate_program(
            FeatureProgram((FeatureQuery("hist", sql["safe"]),))
        )
        # The first call scored the baseline alone; the second, the program.
        assert scorer.columns[0] == ["dfs_rate", "dfs_region"]
        assert scorer.columns[1] == [
            "dfs_rate",
            "dfs_region",
            "hist__n_events",
            "hist__mean_value",
        ]
        assert scorer.categorical[1] == ("dfs_region",)
        # Width is what Nori reads, baseline included.
        assert trial.feature_count == 4
    finally:
        session.close()


def test_a_plain_search_still_scores_the_program_alone(snapshot, anchors, sql) -> None:
    scorer = _RecordingScorer()
    session = _session(
        snapshot, anchors, config=CONFIG, baseline=_baseline(), scorer=scorer
    )
    try:
        trial = session.validate_program(
            FeatureProgram((FeatureQuery("hist", sql["safe"]),))
        )
        assert scorer.columns[1] == ["hist__n_events", "hist__mean_value"]
        assert trial.feature_count == 2
        assert "baseline_features" not in session.describe_schema()
    finally:
        session.close()


def test_the_schema_names_the_columns_every_program_gets(snapshot, anchors) -> None:
    session = _session(snapshot, anchors, config=EXTEND, baseline=_baseline())
    try:
        schema = session.describe_schema()
        assert schema["baseline_features"] == [
            {"name": "dfs_rate", "kind": "numeric", "null_fraction": 0.0},
            {"name": "dfs_region", "kind": "categorical", "null_fraction": 0.0},
        ]
        # Without a NoriDFS plan there is nothing to describe or diff.
        assert "baseline_never_reads" not in schema
        prompt = ProgramAgentBase.user_prompt(None, session, "")
        assert "baseline_never_reads" in prompt
        assert "drop_baseline" in prompt
        assert "2 deterministic DFS columns" in prompt
    finally:
        session.close()


def test_a_program_that_reuses_a_baseline_name_is_refused(
    snapshot, anchors, sql
) -> None:
    baseline = _baseline().rename(columns={"dfs_rate": "hist__n_events"})
    session = _session(snapshot, anchors, config=EXTEND, baseline=baseline)
    try:
        with pytest.raises(ProgramRejected, match="already exist in the baseline"):
            session.validate_program(
                FeatureProgram((FeatureQuery("hist", sql["safe"]),))
            )
        assert "already exist" in session.refusals[0]
    finally:
        session.close()


def test_extending_needs_something_to_extend(snapshot, anchors) -> None:
    with pytest.raises(ValueError, match="baseline features"):
        _session(snapshot, anchors, config=EXTEND, baseline=None)
    with pytest.raises(ValueError, match="fallback backend"):
        AgenticFeatures(
            program=FeatureProgram((FeatureQuery("a", "SELECT 1"),)), config=EXTEND
        )


def test_the_backend_returns_dfs_then_program_columns(snapshot, anchors, sql) -> None:
    train = anchors(10, start="2020-01-03").assign(row_id=[f"c{i}" for i in range(10)])
    query = anchors(2, start="2020-01-14").assign(row_id=["q1", "q2"])
    fallback = NoriDFS()
    expected = fallback.compute(
        snapshot, train, query, "accounts", row_id_column="row_id"
    )
    program = FeatureProgram((FeatureQuery("hist", sql["safe"]),))
    backend = AgenticFeatures(program=program, fallback=fallback, config=EXTEND)
    matrices = backend.compute(
        snapshot, train, query, "accounts", row_id_column="row_id"
    )
    assert matrices.feature_names == [
        *expected.feature_names,
        "hist__n_events",
        "hist__mean_value",
    ]
    assert len(matrices.train) == 10
    assert len(matrices.query) == 2
    pd.testing.assert_frame_equal(
        matrices.query[expected.feature_names], expected.query
    )
    dfs_categorical = fallback.categorical_columns()
    assert backend.last_categorical[: len(dfs_categorical)] == dfs_categorical
    assert backend.categorical_columns() == backend.last_categorical


def test_a_plain_replay_never_computes_the_fallback(snapshot, anchors, sql) -> None:
    train = anchors(4, start="2020-01-03").assign(row_id=range(4))
    query = anchors(2, start="2020-01-10").assign(row_id=[4, 5])
    backend = AgenticFeatures(
        program=FeatureProgram((FeatureQuery("hist", sql["safe"]),)),
        fallback=_UnusedBackend(),
    )
    matrices = backend.compute(
        snapshot, train, query, "accounts", row_id_column="row_id"
    )
    assert matrices.feature_names == ["hist__n_events", "hist__mean_value"]


def test_an_extending_search_keeps_dfs_in_the_answer(snapshot, anchors, sql) -> None:
    train = anchors(10, start="2020-01-03").assign(row_id=[f"c{i}" for i in range(10)])
    query = anchors(2, start="2020-01-14").assign(row_id=["q1", "q2"])
    fallback = NoriDFS()
    expected = fallback.compute(
        snapshot, train, query, "accounts", row_id_column="row_id"
    )
    backend = AgenticFeatures(
        agent=_ScriptedAgent(FeatureProgram((FeatureQuery("hist", sql["safe"]),))),
        scorer=RidgeScorer(max_context_rows=4),
        train_target=pd.Series([float(i) for i in range(10)]),
        fallback=fallback,
        config=EXTEND,
    )
    matrices = backend.compute(
        snapshot, train, query, "accounts", row_id_column="row_id"
    )
    # Whether or not the program survived confirmation, DFS is in the answer.
    assert matrices.feature_names[: len(expected.feature_names)] == (
        expected.feature_names
    )
    assert backend.last_run_result is None


@pytest.mark.parametrize("declaration", [("category_code",), (), None])
def test_backend_keeps_baseline_category_declarations(
    snapshot, anchors, sql, declaration
) -> None:
    class DeclaredBackend(FeatureBackend):
        def categorical_columns(self):
            return declaration

        def compute(self, snapshot, train_rows, query_rows, entity_table, **kwargs):
            frame = pd.DataFrame(
                {
                    "category_code": [index % 2 for index in range(12)],
                    "numeric_text": ["1", "2"] * 6,
                }
            )
            return FeatureMatrices(frame.iloc[:10], frame.iloc[10:], list(frame))

    train = anchors(10, start="2020-01-03").assign(row_id=range(10))
    query = anchors(2, start="2020-01-14").assign(row_id=[10, 11])
    scorer = _RecordingScorer()
    backend = AgenticFeatures(
        agent=_ScriptedAgent(FeatureProgram((FeatureQuery("hist", sql["safe"]),))),
        scorer=scorer,
        train_target=pd.Series(range(10), dtype=float),
        fallback=DeclaredBackend(),
        config=EXTEND,
    )
    backend.compute(snapshot, train, query, "accounts", row_id_column="row_id")
    categories = ("numeric_text",) if declaration is None else declaration
    assert scorer.categorical[0] == categories
    assert scorer.categorical[1] == categories
    assert backend.categorical_columns() == (
        declaration if backend.last_program is None else categories
    )
    backend.program = FeatureProgram((FeatureQuery("hist", sql["safe"]),))
    backend.compute(snapshot, train, query, "accounts", row_id_column="row_id")
    assert backend.categorical_columns() == categories


def test_backend_stores_each_agent_run_result(snapshot, anchors, sql) -> None:
    from nori_rel.feature.agentic.usage import AgentRunResult, TokenUsage

    result = AgentRunResult(
        model="test", usage=TokenUsage(calls=1), cost_usd=0.25, stop_reason="finished"
    )

    class ReportingAgent(_ScriptedAgent):
        def run(self, session, task_hint):
            super().run(session, task_hint)
            return result

    train = anchors(10, start="2020-01-03").assign(row_id=range(10))
    query = anchors(2, start="2020-01-14").assign(row_id=[10, 11])
    program = FeatureProgram((FeatureQuery("hist", sql["safe"]),))
    backend = AgenticFeatures(
        agent=ReportingAgent(program),
        scorer=RidgeScorer(max_context_rows=4),
        train_target=pd.Series(range(10), dtype=float),
        fallback=NoriDFS(),
        config=EXTEND,
    )
    backend.compute(snapshot, train, query, "accounts", row_id_column="row_id")
    assert backend.last_run_result is result
    backend.agent = _ScriptedAgent(program)
    backend.compute(snapshot, train, query, "accounts", row_id_column="row_id")
    assert backend.last_run_result is None


def test_each_dfs_column_is_described_in_words(snapshot) -> None:
    columns = describe_plan(_plan(snapshot))
    assert [(c.reads, c.description) for c in columns] == [
        (
            "events",
            "number of events rows, over the entity's events (by account_id) "
            "rows, all history",
        ),
        (
            "events.value",
            "mean of events.value, over the entity's events (by account_id) "
            "rows, last 7 days",
        ),
    ]
    assert columns[0].name.startswith("nr|path|")


def test_the_gap_list_names_value_columns_dfs_never_reads(snapshot) -> None:
    # region is never aggregated; value is; keys and time are structure.
    assert unread_columns(_plan(snapshot), _view(snapshot)) == {"accounts": ["region"]}
    snapshot.tables["events"] = snapshot.tables["events"].assign(
        channel=["web", "app"] * 6, score=range(12)
    )
    assert unread_columns(_plan(snapshot), _view(snapshot)) == {
        "accounts": ["region"],
        "events": ["channel", "score"],
    }


def test_the_schema_describes_a_noridfs_baseline(snapshot, anchors) -> None:
    plan = _plan(snapshot)
    names = [feature.name for feature in plan.features]
    baseline = pd.DataFrame({names[0]: [1.0] * 10, names[1]: [None] * 5 + [2.0] * 5})
    session = _session(snapshot, anchors, config=EXTEND, baseline=baseline, plan=plan)
    try:
        schema = session.describe_schema()
        first, second = schema["baseline_features"]
        assert first["reads"] == "events"
        assert second["description"].startswith("mean of events.value")
        assert second["null_fraction"] == 0.5
        assert schema["baseline_never_reads"] == {"accounts": ["region"]}
    finally:
        session.close()


def test_a_program_can_drop_redundant_baseline_columns(snapshot, anchors, sql) -> None:
    scorer = _RecordingScorer()
    session = _session(
        snapshot, anchors, config=EXTEND, baseline=_baseline(), scorer=scorer
    )
    try:
        trial = session.validate_program(
            FeatureProgram(
                (FeatureQuery("hist", sql["safe"]),), drop_baseline=("dfs_region",)
            )
        )
        assert scorer.columns[1] == ["dfs_rate", "hist__n_events", "hist__mean_value"]
        # A dropped categorical column is no longer passed as categorical.
        assert scorer.categorical[1] == ()
        assert trial.feature_count == 3
        with pytest.raises(ProgramRejected, match="unknown columns"):
            session.validate_program(
                FeatureProgram(
                    (FeatureQuery("hist", sql["safe"]),), drop_baseline=("nope",)
                )
            )
    finally:
        session.close()


def test_dropping_needs_a_search_that_extends(snapshot, anchors, sql) -> None:
    session = _session(snapshot, anchors, config=CONFIG, baseline=_baseline())
    try:
        with pytest.raises(ProgramRejected, match="extends DFS"):
            session.validate_program(
                FeatureProgram(
                    (FeatureQuery("hist", sql["safe"]),), drop_baseline=("dfs_rate",)
                )
            )
    finally:
        session.close()


def test_the_backend_leaves_dropped_columns_out(snapshot, anchors, sql) -> None:
    train = anchors(10, start="2020-01-03").assign(row_id=[f"c{i}" for i in range(10)])
    query = anchors(2, start="2020-01-14").assign(row_id=["q1", "q2"])
    fallback = NoriDFS()
    expected = fallback.compute(
        snapshot, train, query, "accounts", row_id_column="row_id"
    )
    dropped = expected.feature_names[0]
    program = FeatureProgram(
        (FeatureQuery("hist", sql["safe"]),), drop_baseline=(dropped,)
    )
    backend = AgenticFeatures(
        program=program,
        fallback=fallback,
        config=EXTEND,
        expected_baseline_columns=tuple(expected.feature_names),
    )
    matrices = backend.compute(
        snapshot, train, query, "accounts", row_id_column="row_id"
    )
    assert dropped not in matrices.feature_names
    assert matrices.feature_names == [
        *expected.feature_names[1:],
        "hist__n_events",
        "hist__mean_value",
    ]
    assert dropped not in backend.last_categorical
    assert backend.last_baseline_columns == tuple(expected.feature_names)


def test_replay_rejects_missing_baseline_drops_before_sql(
    snapshot, anchors, sql, monkeypatch
) -> None:
    train = anchors(4, start="2020-01-03").assign(row_id=range(4))
    query = anchors(2, start="2020-01-10").assign(row_id=[4, 5])
    program = FeatureProgram(
        (FeatureQuery("hist", sql["safe"]),),
        drop_baseline=("removed_numeric", "removed_category"),
    )
    backend = AgenticFeatures(program=program, fallback=NoriDFS(), config=EXTEND)
    executor = Mock()
    monkeypatch.setattr("nori_rel.feature.agentic.backend.ProgramExecutor", executor)
    with pytest.raises(
        ValueError,
        match=(
            r"saved program no longer matches data: "
            r".*removed_category.*removed_numeric"
        ),
    ):
        backend.compute(snapshot, train, query, "accounts", row_id_column="row_id")
    executor.assert_not_called()


def test_replay_rejects_changed_baseline_order_before_sql(
    snapshot, anchors, sql, monkeypatch
) -> None:
    train = anchors(4, start="2020-01-03").assign(row_id=range(4))
    query = anchors(2, start="2020-01-10").assign(row_id=[4, 5])
    fallback = NoriDFS()
    baseline = fallback.compute(
        snapshot, train, query, "accounts", row_id_column="row_id"
    )
    backend = AgenticFeatures(
        program=FeatureProgram((FeatureQuery("hist", sql["safe"]),)),
        fallback=fallback,
        config=EXTEND,
        expected_baseline_columns=tuple(reversed(baseline.feature_names)),
    )
    executor = Mock()
    monkeypatch.setattr("nori_rel.feature.agentic.backend.ProgramExecutor", executor)
    with pytest.raises(ValueError, match="NoriDFS columns changed; run a new search"):
        backend.compute(snapshot, train, query, "accounts", row_id_column="row_id")
    executor.assert_not_called()


def test_an_extending_search_sees_the_noridfs_plan(snapshot, anchors, sql) -> None:
    class _Looking(_ScriptedAgent):
        def run(self, session, task_hint):
            self.schema = session.describe_schema()
            super().run(session, task_hint)

    train = anchors(10, start="2020-01-03").assign(row_id=[f"c{i}" for i in range(10)])
    query = anchors(2, start="2020-01-14").assign(row_id=["q1", "q2"])
    agent = _Looking(FeatureProgram((FeatureQuery("hist", sql["safe"]),)))
    AgenticFeatures(
        agent=agent,
        scorer=RidgeScorer(max_context_rows=4),
        train_target=pd.Series([float(i) for i in range(10)]),
        fallback=NoriDFS(),
        config=EXTEND,
    ).compute(snapshot, train, query, "accounts", row_id_column="row_id")
    assert all("description" in entry for entry in agent.schema["baseline_features"])
    assert "baseline_never_reads" in agent.schema


def test_the_tool_passes_drop_baseline_through(snapshot, anchors, sql) -> None:
    program = parse_program(
        json.dumps([{"name": "hist", "sql": sql["safe"]}]), ["dfs_region"]
    )
    assert program.drop_baseline == ("dfs_region",)
    with pytest.raises(ValueError, match="list of column names"):
        parse_program("[]", "dfs_region")
    scorer = _RecordingScorer()
    session = _session(
        snapshot, anchors, config=EXTEND, baseline=_baseline(), scorer=scorer
    )
    try:
        tools = {spec.name: spec for spec in ProgramAgentBase.tools(None, session)}
        report = json.loads(
            tools["validate_program"].handler(
                json.dumps([{"name": "hist", "sql": sql["safe"]}]),
                drop_baseline=["dfs_region"],
            )
        )
        assert report["features"] == 3
        assert "dfs_region" not in scorer.columns[-1]
    finally:
        session.close()
