"""The agent base builds one tool surface; each subclass only drives its SDK."""

from __future__ import annotations

import copy
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from nori_rel._feature_search import DEFAULT_SEARCH_CONFIG
from nori_rel.feature import CUTOFF_COLUMN
from nori_rel.feature.agentic import (
    AgenticFeatureConfig,
    FeatureProgram,
    PlanningSession,
    ProgramAgentBase,
    RidgeScorer,
    ScoreReport,
)
from nori_rel.feature.agentic._agent_base import ToolSpec, parse_program
from nori_rel.feature.agentic.agent_claude import ClaudeProgramAgent
from nori_rel.feature.agentic.agent_openai import OpenAIProgramAgent
from nori_rel.feature.dfs.noridfs import AnchorDefinition, DatabaseView


def _session(
    snapshot, anchors, *, baseline=None, min_relative_gain=0, noise_margin=2.0
) -> PlanningSession:
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
        config=AgenticFeatureConfig(
            max_trials=4,
            leak_check_anchors=2,
            min_relative_gain=min_relative_gain,
            noise_margin=noise_margin,
        ),
        train_anchors=anchors(10, start="2020-01-03"),
        train_target=pd.Series([float(i) for i in range(10)]),
        scorer=RidgeScorer(max_context_rows=4),
        baseline_features=baseline,
    )


def test_parse_program_reads_the_tool_argument() -> None:
    program = parse_program('[{"name": "a", "sql": "SELECT 1"}]')
    assert program.queries[0].name == "a"
    with pytest.raises(ValueError, match="JSON list"):
        parse_program('{"name": "a"}')
    with pytest.raises(ValueError, match="identifier"):
        parse_program('[{"name": "not an identifier", "sql": "SELECT 1"}]')


def test_prompt_states_the_relative_acceptance_floor(snapshot, anchors):
    session = _session(snapshot, anchors, min_relative_gain=0.02)
    try:
        prompt = ProgramAgentBase.user_prompt(None, session, "")
        assert "2.00%" in prompt and "baseline and current best" in prompt
        assert "Confirmation" in prompt
    finally:
        session.close()


def test_the_client_search_counts_any_higher_score_as_a_gain(snapshot, anchors):
    """With the client's settings, a trial or confirmation needs no margin."""
    session = _session(
        snapshot,
        anchors,
        min_relative_gain=DEFAULT_SEARCH_CONFIG.min_relative_gain,
        noise_margin=DEFAULT_SEARCH_CONFIG.noise_margin,
    )
    try:
        best = ScoreReport(
            metric="mae", score=-2.478, noise=0.2, predictions=np.zeros(1)
        )
        trial = ScoreReport(
            metric="mae", score=-2.463, noise=0.2, predictions=np.zeros(1)
        )
        assert session._minimum_score_gain(trial, best) == 0.0
    finally:
        session.close()


def test_tools_carry_schemas_and_return_rejections_as_text(
    snapshot, anchors, sql
) -> None:
    session = _session(snapshot, anchors)
    try:
        tools = {spec.name: spec for spec in ProgramAgentBase.tools(None, session)}
        assert set(tools) == {
            "describe_schema",
            "profile_column",
            "run_sql",
            "validate_program",
        }
        for spec in tools.values():
            assert spec.parameters["type"] == "object"

        schema = json.loads(tools["describe_schema"].handler())
        assert schema["eval_table"] == "eval_table"

        # A bad call comes back as text so the model can repair it.
        assert (
            tools["profile_column"].handler("accounts", "nope").startswith("rejected")
        )
        assert tools["run_sql"].handler("DROP TABLE events").startswith("rejected")

        report = json.loads(
            tools["validate_program"].handler(
                json.dumps([{"name": "hist", "sql": sql["safe"]}])
            )
        )
        assert report["accepted"] is True
        assert report["features"] == 2
        assert report["coverage"]["hist"] == 1.0
    finally:
        session.close()


class _StubCompletions:
    """Two turns: one tool call, then a final answer with no tool calls."""

    def __init__(self, program_json: str) -> None:
        self.program_json = program_json
        self.calls: list[dict] = []

    def create(self, **kwargs):
        # The agent reuses one message list, so snapshot it per call.
        self.calls.append({**kwargs, "messages": copy.deepcopy(kwargs["messages"])})
        if len(self.calls) == 1:
            call = SimpleNamespace(
                id="call_1",
                function=SimpleNamespace(
                    name="validate_program",
                    arguments=json.dumps({"queries_json": self.program_json}),
                ),
            )
            message = SimpleNamespace(content=None, tool_calls=[call])
        else:
            message = SimpleNamespace(content="done", tool_calls=None)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=message,
                    finish_reason="tool_calls" if message.tool_calls else "stop",
                )
            ]
        )


def test_openai_agent_runs_tools_and_stops(snapshot, anchors, sql) -> None:
    session = _session(snapshot, anchors)
    completions = _StubCompletions(json.dumps([{"name": "hist", "sql": sql["safe"]}]))
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    try:
        result = OpenAIProgramAgent(model="qwen3-32b", client=client).run(
            session, "a hint"
        )
        assert result.model == "qwen3-32b"
        assert result.stop_reason == "model_finished"
        assert result.usage is None and result.cost_usd is None
        assert result.response_count == 2
        assert isinstance(session.best_program(), FeatureProgram)
        assert len(completions.calls) == 2
        messages = completions.calls[1]["messages"]
        assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool"]
        assert "a hint" in messages[1]["content"]
        assert json.loads(messages[3]["content"])["accepted"] is True
    finally:
        session.close()


@pytest.mark.parametrize("has_baseline", [False, True])
def test_openai_agent_returns_baseline_when_no_program_is_accepted(
    snapshot, anchors, has_baseline
) -> None:
    session = _session(
        snapshot,
        anchors,
        baseline=pd.DataFrame({"baseline": range(10)}) if has_baseline else None,
    )

    class _BadCall(_StubCompletions):
        def create(self, **kwargs):
            self.calls.append({**kwargs, "messages": copy.deepcopy(kwargs["messages"])})
            if len(self.calls) == 1:
                call = SimpleNamespace(
                    id="c", function=SimpleNamespace(name="nope", arguments="{oops")
                )
                message = SimpleNamespace(content=None, tool_calls=[call])
            else:
                message = SimpleNamespace(content="giving up", tool_calls=None)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=message,
                        finish_reason="tool_calls" if message.tool_calls else "stop",
                    )
                ]
            )

    completions = _BadCall("")
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    try:
        agent = OpenAIProgramAgent(model="m", client=client)
        if has_baseline:
            assert agent.run(session, "").stop_reason == "model_finished"
            assert session.best is None
        else:
            with pytest.raises(RuntimeError, match="no accepted program"):
                agent.run(session, "")
        assert "unknown tool" in completions.calls[1]["messages"][-1]["content"]
    finally:
        session.close()


def test_openai_agent_sends_top_p_only_when_set_and_echoes_reasoning(
    snapshot, anchors, sql
) -> None:
    """A reasoning model's ``reasoning_content`` travels back with its call."""
    program_json = json.dumps([{"name": "hist", "sql": sql["safe"]}])

    class _Reasoning(_StubCompletions):
        def create(self, **kwargs):
            response = super().create(**kwargs)
            response.choices[0].message.reasoning_content = "think first"
            return response

    session = _session(snapshot, anchors)
    completions = _Reasoning(program_json)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    try:
        OpenAIProgramAgent(model="moonshotai/Kimi-K3", top_p=0.95, client=client).run(
            session, ""
        )
        first, second = completions.calls
        assert first["top_p"] == 0.95
        assert second["messages"][2]["reasoning_content"] == "think first"
    finally:
        session.close()

    session = _session(snapshot, anchors)
    completions = _StubCompletions(program_json)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    try:
        OpenAIProgramAgent(model="qwen3-32b", client=client).run(session, "")
        first, second = completions.calls
        assert "top_p" not in first
        assert "reasoning_content" not in second["messages"][2]
    finally:
        session.close()


@pytest.mark.parametrize(
    "field,other",
    [("max_tokens", "max_completion_tokens"), ("max_completion_tokens", "max_tokens")],
)
def test_openai_agent_sends_its_cap_in_the_configured_field(
    snapshot, anchors, sql, field, other
) -> None:
    """OpenAI's reasoning models refuse ``max_tokens``; other servers may need it."""
    session = _session(snapshot, anchors)
    completions = _StubCompletions(json.dumps([{"name": "hist", "sql": sql["safe"]}]))
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    try:
        OpenAIProgramAgent(
            model="m", max_tokens=1234, token_limit_field=field, client=client
        ).run(session, "")
        assert all(call[field] == 1234 for call in completions.calls)
        assert all(other not in call for call in completions.calls)
    finally:
        session.close()


def test_openai_agent_refuses_an_unknown_token_limit_field() -> None:
    with pytest.raises(ValueError, match="token_limit_field"):
        OpenAIProgramAgent(model="m", token_limit_field="max_output_tokens")


def test_a_program_that_cannot_be_parsed_is_recorded_as_a_refusal(
    snapshot, anchors
) -> None:
    session = _session(snapshot, anchors)
    try:
        tools = {spec.name: spec for spec in ProgramAgentBase.tools(None, session)}
        validate = tools["validate_program"].handler
        assert validate("{not json").startswith("rejected")
        assert validate('[{"sql": "SELECT 1"}]').startswith("rejected")
        assert len(session.refusals) == 2
        assert session.refusals[0].startswith("JSONDecodeError")
        assert session.refusals[1].startswith("KeyError")
    finally:
        session.close()


@pytest.mark.parametrize("reason", ["length", "content_filter", "stop", "refusal"])
def test_openai_terminal_responses_never_execute_tools(reason):
    handler = Mock(return_value="must not run")
    continuation = Mock(return_value="must not continue")
    message = SimpleNamespace(
        content=None,
        refusal="declined" if reason == "refusal" else None,
        tool_calls=[
            SimpleNamespace(
                id="call", function=SimpleNamespace(name="probe", arguments="{}")
            )
        ],
    )
    completions = SimpleNamespace(
        create=Mock(
            return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=message, finish_reason=reason)]
            )
        )
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    result = OpenAIProgramAgent(model="test", client=client)._run_loop(
        "system", "prompt", (ToolSpec("probe", "", {}, handler),), continuation
    )
    handler.assert_not_called()
    continuation.assert_not_called()
    assert result.stop_reason == ("model_finished" if reason == "stop" else reason)


def test_openai_turn_cap_prevents_the_final_response_tool():
    completions = _StubCompletions("[]")
    handler = Mock(return_value="must not run")
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    result = OpenAIProgramAgent(model="test", max_turns=1, client=client)._run_loop(
        "system", "prompt", (ToolSpec("validate_program", "", {}, handler),)
    )
    handler.assert_not_called()
    assert result.stop_reason == "max_turns" and result.response_count == 1


@pytest.mark.parametrize("agent_class", [ClaudeProgramAgent, OpenAIProgramAgent])
@pytest.mark.parametrize(
    "options",
    [
        {"max_tokens": 0},
        {"max_tokens": -1},
        {"max_turns": 0},
        {"max_turns": None},
        {"min_trials": -1},
        {"max_continuation_requests": -1},
        {"max_turns": True},
    ],
)
def test_provider_budgets_are_validated(agent_class, options):
    with pytest.raises(ValueError):
        agent_class(model="test", **options)


@pytest.mark.parametrize("agent_class", [ClaudeProgramAgent, OpenAIProgramAgent])
def test_one_agent_can_run_two_searches_without_shared_counters(agent_class):
    barrier = Barrier(2)

    def create_response(**kwargs):
        claude = "system" in kwargs
        first_request_length = 1 if claude else 2
        history = kwargs["messages"]
        prompt = history[0 if claude else 1]["content"]
        if len(history) == first_request_length:
            barrier.wait(timeout=5)
        if claude:
            return SimpleNamespace(
                stop_reason="end_turn",
                content=[{"type": "text", "text": "done"}],
                usage=SimpleNamespace(
                    input_tokens=10 if prompt.endswith("first") else 20,
                    output_tokens=0,
                    cache_read_input_tokens=0,
                    cache_creation_input_tokens=0,
                    cache_creation=None,
                ),
            )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content="done", tool_calls=[]),
                )
            ]
        )

    replies = SimpleNamespace(create=create_response)
    client = SimpleNamespace(
        messages=replies, chat=SimpleNamespace(completions=replies)
    )
    agent = agent_class(
        model="test", client=client, min_trials=1, max_continuation_requests=1
    )
    original_attributes = vars(agent).copy()

    def run_search(hint):
        session = SimpleNamespace(
            best=object(),
            baseline=None,
            trials=[],
            report_model_response=lambda cost_usd: None,
            check_for_interrupt=lambda: False,
            has_held_steering=lambda: False,
            deliver_steering_text=lambda: None,
            config=SimpleNamespace(
                max_trials=1, allow_target_transform=False, min_relative_gain=0
            ),
        )
        return agent.run(session, hint)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = list(pool.map(run_search, ["first", "second"]))
    assert first.continuation_requests == second.continuation_requests == 1
    assert first.response_count == second.response_count == 2
    assert first.stop_reason == second.stop_reason == "model_finished"
    if agent_class is ClaudeProgramAgent:
        assert first.usage.input_tokens == 20
        assert second.usage.input_tokens == 40
    else:
        assert first.usage is None and second.usage is None
    assert vars(agent) == original_attributes
