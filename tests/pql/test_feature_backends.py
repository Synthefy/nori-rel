"""PostgreSQL + NoriDFS or agent search + hosted-client transport, offline."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pandas as pd
import pytest

from nori_rel import AgenticSearch
from nori_rel._feature_search import program_from_record
from nori_rel._runtime import LocalRelationalRuntime
from nori_rel.feature.agentic import FeatureProgram, FeatureQuery
from nori_rel.feature.agentic.usage import AgentRunResult, TokenUsage
from nori_rel.limits import RelationalLimits
from nori_rel.run import RunContext, RunEvent

QUERY = "PREDICT COUNT(orders.*,0,30,days) FOR EACH users.user_id"
OPTIONS = dict(
    anchor_times=[
        datetime(2025, 9, 1, tzinfo=UTC),
        datetime(2025, 10, 1, tzinfo=UTC),
        datetime(2025, 11, 1, tzinfo=UTC),
        datetime(2025, 12, 1, tzinfo=UTC),
        datetime(2026, 1, 1, tzinfo=UTC),
    ],
    anchor_time=datetime(2026, 2, 1, tzinfo=UTC),
    observed_through=datetime(2026, 2, 1, tzinfo=UTC),
    entity_created_at="created_at",
)
PROGRAM = FeatureProgram(
    (
        FeatureQuery(
            "history",
            "SELECT e.row_id, COUNT(o.id) AS n_orders FROM eval_table e "
            "LEFT JOIN orders o ON o.buyer_id = e.user_id "
            "AND o.event_time < e.cutoff GROUP BY e.row_id",
        ),
    )
)


def _agentic():
    """Skip unless the agentic extra, which runs the agent's SQL, is installed."""
    pytest.importorskip("duckdb")


class _ScriptedAgent:
    """Submits one program, like a model that got it right the first time."""

    def __init__(self) -> None:
        self.hints: list[str] = []
        self.instructions: list[str | None] = []

    def run(self, session, task_hint: str) -> AgentRunResult:
        self.hints.append(task_hint)
        self.instructions.append(session.agent_instructions)
        session.validate_program(PROGRAM)
        return AgentRunResult(
            model="scripted",
            usage=TokenUsage(input_tokens=100, output_tokens=20),
            cost_usd=0.02,
            stop_reason="end_turn",
        )


def _runtime(source, monkeypatch, features, answer):
    calls = []

    class Transport:
        def __init__(self, **options):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def predict(self, train, target, query, **options):
            calls.append((train.copy(), target.copy(), query.copy(), options))
            return pd.Series([answer(train)] * len(query))

    monkeypatch.setattr("nori_rel._runtime.SynthefyNoriClient", Transport)
    runtime = LocalRelationalRuntime(
        api_key="test-only",
        model="nori-30m",
        classification_model=None,
        timeout=10,
        max_retries=0,
        limits=RelationalLimits(min_training_rows=2),
        features=features,
    )
    return runtime, runtime.connect(source[0].config), calls


def test_noridfs_sends_declared_categoricals_to_hosted_nori(source, monkeypatch):
    runtime, database, calls = _runtime(source, monkeypatch, "noridfs", lambda _: 1.0)
    try:
        result = runtime.predict_pql(database, QUERY, **OPTIONS)
    finally:
        runtime.close()
    assert sorted(result["entity_id"]) == [7, 42, 99, 500]
    assert result.attrs["features"] == "noridfs"
    (train, target, query, options) = calls[0]
    assert train.columns.tolist() == query.columns.tolist()
    assert any(column.startswith("nr|path|") for column in train.columns)
    assert len(target) == 15
    declared = options["categorical_columns"]
    assert all(column in train for column in declared)
    assert "nr|anchor|entity_key" in declared
    assert result.attrs["feature_count"] == train.shape[1]


def test_a_prediction_overrides_the_runtime_feature_search(source, monkeypatch):
    runtime, database, calls = _runtime(source, monkeypatch, "fastdfs", lambda _: 1.0)
    try:
        result = runtime.predict_pql(
            database, QUERY, feature_strategy="noridfs", **OPTIONS
        )
    finally:
        runtime.close()
    assert result.attrs["features"] == "noridfs"
    assert any(column.startswith("nr|path|") for column in calls[0][0].columns)
    assert runtime.features == "fastdfs"


def _with_program(train: pd.DataFrame) -> float:
    """Predict the context labels exactly once the program's column is there."""
    return 0.0 if "history__n_orders" in train else 50.0


@pytest.fixture
def saved_program(source, monkeypatch):
    _agentic()
    search = AgenticSearch(agent=_ScriptedAgent())
    runtime, database, _ = _runtime(source, monkeypatch, search, _with_program)
    try:
        found = runtime.predict_pql(database, QUERY, **OPTIONS)
    finally:
        runtime.close()
    return program_from_record(json.loads(json.dumps(found.attrs["agentic_program"])))


def test_an_agent_program_that_beats_noridfs_is_kept(source, monkeypatch):
    _agentic()
    agent = _ScriptedAgent()
    search = AgenticSearch(agent=agent, task_hint="order counts")
    runtime, database, calls = _runtime(source, monkeypatch, search, _with_program)
    try:
        result = runtime.predict_pql(database, QUERY, **OPTIONS)
    finally:
        runtime.close()
    assert agent.hints == ["order counts"]
    assert result.attrs["features"] == "agentic"
    assert result.attrs["agentic_trials"] == 1
    assert result.attrs["agentic_queries"] == ["history"]
    assert result.attrs["agentic_model"] == "scripted"
    assert result.attrs["agentic_cost_usd"] == 0.02
    assert (
        result.attrs["agentic_usage"]
        == TokenUsage(input_tokens=100, output_tokens=20).as_dict()
    )
    assert result.attrs["agentic_stop_reason"] == "end_turn"
    saved = program_from_record(result.attrs["agentic_program"])
    assert saved.to_program() == PROGRAM
    assert saved.pql_query == QUERY
    assert saved.entity_created_at == OPTIONS["entity_created_at"]
    # Baseline and trial on the holdout, then both on confirmation, then the
    # prediction itself.
    assert len(calls) == 5
    final_train, _, final_query, options = calls[-1]
    assert "history__n_orders" in final_train
    assert final_train.columns.tolist() == final_query.columns.tolist()
    assert "nr|anchor|entity_key" in options["categorical_columns"]
    assert "history__n_orders" not in options["categorical_columns"]
    assert result["prediction"].tolist() == [0.0] * 4
    # The context report records how the agent split the rows it scored on.
    split = result.attrs["context_policy"]["agent_search_split"]
    assert split["requested_folds"] == len(split["folds"]) == 1
    fold = split["folds"][0]
    assert fold["context_rows"] + fold["validation_rows"] + split[
        "confirmation_rows"
    ] == len(final_train)


def test_agent_instructions_reach_the_agent_and_the_report(source, monkeypatch):
    _agentic()
    agent = _ScriptedAgent()
    search = AgenticSearch(agent=agent)
    runtime, database, _ = _runtime(source, monkeypatch, search, _with_program)
    instructions = "Prefer order history over account attributes."
    try:
        result = runtime.predict_pql(
            database, QUERY, **OPTIONS, agent_instructions=instructions
        )
    finally:
        runtime.close()
    assert agent.instructions == [instructions]
    report = result.attrs["context_policy"]
    assert report["agent_instructions"] == instructions
    assert report["agent_instruction_stages"] == ["feature_search"]


def test_a_program_that_does_not_beat_noridfs_falls_back(source, monkeypatch):
    _agentic()
    search = AgenticSearch(agent=_ScriptedAgent())
    runtime, database, calls = _runtime(source, monkeypatch, search, lambda _: 1.0)
    try:
        result = runtime.predict_pql(database, QUERY, **OPTIONS)
    finally:
        runtime.close()
    assert result.attrs["agentic_trials"] == 1
    assert result.attrs["agentic_queries"] is None
    assert result.attrs["agentic_program"] is None
    assert result.attrs["agentic_cost_usd"] == 0.02
    assert result.attrs["agentic_stop_reason"] == "end_turn"
    final_train = calls[-1][0]
    assert not any(column.startswith("history__") for column in final_train)
    assert any(column.startswith("nr|path|") for column in final_train)


def test_an_agent_search_needs_five_anchor_times(source, monkeypatch):
    _agentic()
    search = AgenticSearch(agent=_ScriptedAgent())
    runtime, database, calls = _runtime(source, monkeypatch, search, lambda _: 1.0)
    options = {**OPTIONS, "anchor_times": OPTIONS["anchor_times"][1:]}
    try:
        with pytest.raises(ValueError, match="cutoff"):
            runtime.predict_pql(database, QUERY, **options)
    finally:
        runtime.close()
    assert calls == []


def test_a_saved_program_replays_without_a_search(source, monkeypatch, saved_program):
    # A FastDFS client replays it, on two anchor times, which a search refuses.
    runtime, database, calls = _runtime(source, monkeypatch, "fastdfs", _with_program)
    options = {**OPTIONS, "anchor_times": OPTIONS["anchor_times"][-2:]}
    try:
        result = runtime.predict_pql(
            database, QUERY, **options, feature_program=saved_program
        )
    finally:
        runtime.close()
    # Only the prediction itself: no baseline, trials, or confirmation fits.
    assert len(calls) == 1
    train, _, query, call_options = calls[0]
    assert "history__n_orders" in train
    assert any(column.startswith("nr|path|") for column in train)
    assert train.columns.tolist() == query.columns.tolist()
    assert "nr|anchor|entity_key" in call_options["categorical_columns"]
    assert result.attrs["features"] == "saved_program"
    assert result.attrs["agentic_trials"] == 0
    assert result.attrs["agentic_cost_usd"] == 0.0
    assert result.attrs["agentic_usage"] == TokenUsage().as_dict()
    assert result.attrs["agentic_stop_reason"] == "replay"
    assert result.attrs["agentic_program"] == saved_program.model_dump(mode="json")
    assert result["prediction"].tolist() == [0.0] * 4
    # A replay runs no search, so there is no agent split to report.
    assert result.attrs["context_policy"]["agent_search_split"] is None


@pytest.mark.parametrize(
    "query,entity_created_at,match",
    [
        (QUERY.replace("COUNT(orders.*", "SUM(orders.amount"), "created_at", "PQL"),
        (QUERY.replace("0,30", "0,7"), "created_at", "PQL"),
        (QUERY.replace("users.user_id", "orders.id"), "created_at", "PQL"),
        (QUERY, "registered_at", "entity_created_at"),
    ],
    ids=["target", "window", "entity", "creation-column"],
)
def test_replay_rejects_changed_task_before_reading(
    source, monkeypatch, saved_program, query, entity_created_at, match
):
    runtime, database, calls = _runtime(source, monkeypatch, "fastdfs", _with_program)

    def refuse_read(*args, **kwargs):
        pytest.fail("A mismatched replay must not read the database")

    monkeypatch.setattr(runtime._connector(database), "prepare_inputs", refuse_read)
    try:
        with pytest.raises(ValueError, match=match):
            runtime.predict_pql(
                database,
                query,
                **{**OPTIONS, "entity_created_at": entity_created_at},
                feature_program=saved_program,
            )
    finally:
        runtime.close()
    assert calls == []


@pytest.mark.parametrize("query_filter", [None, "users.country='absent'"])
def test_replay_rejects_schema_drift_before_features(
    source, monkeypatch, saved_program, query_filter
):
    source[1].execute("ALTER TABLE orders ADD COLUMN campaign text")
    runtime, database, calls = _runtime(source, monkeypatch, "fastdfs", _with_program)

    def refuse_features(*args, **kwargs):
        pytest.fail("A mismatched replay must not compute features")

    monkeypatch.setattr(runtime, "feature_backend", refuse_features)
    try:
        with pytest.raises(ValueError, match="database schema"):
            runtime.predict_pql(
                database,
                QUERY,
                **OPTIONS,
                query_filter=query_filter,
                feature_program=saved_program,
            )
    finally:
        runtime.close()
    assert calls == []


def test_replay_rejects_category_growth_before_saved_sql(
    source, monkeypatch, saved_program
):
    assert any("|channel|mode|" in name for name in saved_program.baseline_columns)
    source[1].execute(
        "INSERT INTO orders (id, buyer_id, event_time, amount, channel) "
        "SELECT 100 + value, 7, '2026-01-15'::timestamptz, 1, "
        "'channel-' || value FROM generate_series(1, 33) AS value"
    )
    runtime, database, calls = _runtime(source, monkeypatch, "fastdfs", _with_program)

    def refuse_saved_sql(*args, **kwargs):
        pytest.fail("A changed NoriDFS baseline must fail before saved SQL runs")

    monkeypatch.setattr(
        "nori_rel.feature.agentic.backend.ProgramExecutor", refuse_saved_sql
    )
    try:
        with pytest.raises(ValueError, match="NoriDFS columns changed"):
            runtime.predict_pql(
                database, QUERY, **OPTIONS, feature_program=saved_program
            )
    finally:
        runtime.close()
    assert calls == []


STEERING_TEXT = "Use order history."


class _SteeringReadingAgent:
    """Reads the steering the session holds when it starts, then submits a program."""

    def __init__(self) -> None:
        self.steering: list[str | None] = []

    def run(self, session, task_hint: str) -> AgentRunResult:
        session.check_for_interrupt()
        self.steering.append(session.deliver_steering_text())
        session.validate_program(PROGRAM)
        return AgentRunResult(
            model="scripted", usage=TokenUsage(), cost_usd=0.0, stop_reason="end_turn"
        )


def _get_command_outcomes(events: list[RunEvent]) -> list[tuple[str, str, str | None]]:
    """Return each command event as (command, status, rejection reason)."""
    return [
        (event.data["command"], event.data["status"], event.data.get("reason"))
        for event in events
        if event.data["kind"] == "command"
    ]


def _get_observed_context_with_steer() -> tuple[RunContext, list[RunEvent]]:
    """Return an observed run context with one steer already sent."""
    events: list[RunEvent] = []
    run_context = RunContext("run_1", event_sink=events.append)
    run_context.commands.submit_command("steer", STEERING_TEXT)
    return run_context, events


def test_an_agent_search_holds_a_steer_sent_before_its_agent_starts(
    source, monkeypatch
):
    _agentic()
    agent = _SteeringReadingAgent()
    runtime, database, _ = _runtime(
        source, monkeypatch, AgenticSearch(agent=agent), _with_program
    )
    run_context, events = _get_observed_context_with_steer()
    try:
        runtime.predict_pql(database, QUERY, **OPTIONS, run_context=run_context)
    finally:
        runtime.close()
    assert agent.steering == [f"Steering from the user: {STEERING_TEXT}"]
    assert _get_command_outcomes(events) == [("steer", "applied", None)]
    kinds = [event.data["kind"] for event in events]
    assert kinds.index("baseline_scored") < kinds.index("command")
    assert kinds.index("command") < kinds.index("trial_scored")


@pytest.mark.parametrize("features", ["fastdfs", "noridfs"])
def test_a_dfs_run_rejects_steering_at_its_first_boundary(
    source, monkeypatch, features
):
    runtime, database, _ = _runtime(source, monkeypatch, features, lambda _: 1.0)
    run_context, events = _get_observed_context_with_steer()
    try:
        runtime.predict_pql(database, QUERY, **OPTIONS, run_context=run_context)
    finally:
        runtime.close()
    assert _get_command_outcomes(events) == [("steer", "rejected", "not_applicable")]
    assert events[0].data["kind"] == "command"


def test_a_saved_program_replay_rejects_steering_at_its_first_boundary(
    source, monkeypatch, saved_program
):
    agent = _SteeringReadingAgent()
    runtime, database, _ = _runtime(
        source, monkeypatch, AgenticSearch(agent=agent), _with_program
    )
    run_context, events = _get_observed_context_with_steer()
    try:
        runtime.predict_pql(
            database,
            QUERY,
            **OPTIONS,
            feature_program=saved_program,
            run_context=run_context,
        )
    finally:
        runtime.close()
    assert agent.steering == []
    assert _get_command_outcomes(events) == [("steer", "rejected", "not_applicable")]
    assert events[0].data["kind"] == "command"


def test_a_stop_before_the_agent_starts_rejects_a_held_steer_at_once(
    source, monkeypatch
):
    _agentic()
    agent = _SteeringReadingAgent()
    runtime, database, _ = _runtime(
        source, monkeypatch, AgenticSearch(agent=agent), _with_program
    )
    run_context, events = _get_observed_context_with_steer()
    run_context.commands.submit_command("stop")
    try:
        runtime.predict_pql(database, QUERY, **OPTIONS, run_context=run_context)
    finally:
        runtime.close()
    assert agent.steering == []
    assert _get_command_outcomes(events) == [
        ("stop", "applied", None),
        ("steer", "rejected", "not_applicable"),
    ]
    stages = [
        event.data["stage"] for event in events if event.data["kind"] == "stage_started"
    ]
    assert stages == ["materialize", "plan_search", "predict_batch"]
    rejection = next(e for e in events if e.data.get("reason") == "not_applicable")
    later_stage = next(e for e in events if e.data.get("stage") == "predict_batch")
    assert rejection.event_id < later_stage.event_id


@pytest.fixture
def source_with_unselected_parent(source):
    """The users/orders source, with users.region_id keyed to a regions table
    that the source does not select."""
    _, db = source
    db.execute("CREATE TABLE regions (region_id integer PRIMARY KEY, name text)")
    db.execute("INSERT INTO regions VALUES (1, 'west')")
    db.execute(
        "ALTER TABLE users ADD COLUMN region_id integer REFERENCES regions(region_id)"
    )
    db.execute("UPDATE users SET region_id = 1")
    return source


@pytest.mark.parametrize("features", ["fastdfs", "noridfs", "agentic"])
def test_a_key_to_an_unselected_table_is_dropped_and_reported(
    source_with_unselected_parent, monkeypatch, features
):
    if features == "agentic":
        _agentic()
        features = AgenticSearch(agent=_ScriptedAgent())
    runtime, database, calls = _runtime(
        source_with_unselected_parent, monkeypatch, features, _with_program
    )
    try:
        result = runtime.predict_pql(database, QUERY, **OPTIONS)
    finally:
        runtime.close()
    assert result.attrs["dropped_foreign_keys"] == [
        "users.region_id -> regions.region_id"
    ]
    train = calls[-1][0]
    # No feature follows the dropped key into the unselected table.
    assert not any("regions" in column for column in train.columns)
    assert len(result) == 4
