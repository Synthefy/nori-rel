"""One Claude history, run-local accounting, and checks before tool execution."""

from __future__ import annotations

import copy
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nori_rel.feature.agentic._agent_base import ToolSpec
from nori_rel.feature.agentic.agent_claude import ClaudeProgramAgent
from nori_rel.feature.agentic.usage import TokenPrices, TokenUsage

PRICES = TokenPrices(
    input=2.0, output=10.0, cache_read=0.2, cache_write_5m=2.5, cache_write_1h=4.0
)
PROBE = ToolSpec("probe", "Return a test value.", {"type": "object"}, lambda: "ok")


def _usage(*, fresh=0, read=0, write_1h=0, output=0):
    return SimpleNamespace(
        input_tokens=fresh,
        cache_read_input_tokens=read,
        cache_creation_input_tokens=write_1h,
        cache_creation=SimpleNamespace(
            ephemeral_1h_input_tokens=write_1h, ephemeral_5m_input_tokens=0
        ),
        output_tokens=output,
    )


def _response(usage=None, stop_reason="tool_use", content=None):
    if content is None:
        content = (
            [{"type": "tool_use", "id": "call", "name": "probe", "input": {}}]
            if stop_reason == "tool_use"
            else [{"type": "text", "text": "done"}]
        )
    return SimpleNamespace(
        usage=usage or _usage(), stop_reason=stop_reason, content=content
    )


class _StubMessages:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        return next(self.responses)


def _client(responses):
    messages = _StubMessages(responses)
    return SimpleNamespace(messages=messages), messages


def test_caching_usage_and_one_history_are_recorded():
    client, messages = _client(
        [
            _response(_usage(fresh=10, write_1h=5_000, output=400)),
            _response(_usage(fresh=10, read=5_000, write_1h=900, output=300)),
            _response(_usage(fresh=10, read=5_900, output=200), "end_turn"),
        ]
    )
    costs = []
    agent = ClaudeProgramAgent(
        model="claude-sonnet-5-5",
        cache_ttl="1h",
        prices=PRICES,
        client=client,
        on_usage=costs.append,
    )
    result = agent._run_loop("system", "prompt", (PROBE,))
    request = messages.calls[-1]
    assert request["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert request["tools"][0]["input_schema"] == PROBE.parameters
    assert request["model"] == result.model == "claude-sonnet-5-5"
    assert [entry["role"] for entry in request["messages"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert request["messages"][2]["content"] == [
        {"type": "tool_result", "tool_use_id": "call", "content": "ok"}
    ]
    assert result.usage.calls == result.response_count == 3
    assert result.usage.cache_read_tokens == 10_900
    assert result.usage.cache_write_1h_tokens == 5_900
    assert result.stop_reason == "model_finished"
    expected = 1e-6 * (30 * 2.0 + 900 * 10.0 + 10_900 * 0.2 + 5_900 * 4.0)
    assert result.cost_usd == costs[-1] == pytest.approx(expected)
    assert len(costs) == 3
    assert not hasattr(agent, "usage") and not hasattr(agent, "stop_reason")


@pytest.mark.parametrize(
    "reason",
    [
        "max_tokens",
        "refusal",
        "model_context_window_exceeded",
        "stop_sequence",
        "pause_turn",
        "end_turn",
    ],
)
def test_terminal_messages_cannot_execute_tools_or_continue(reason):
    handler = Mock(return_value="must not run")
    continuation = Mock(return_value="must not continue")
    response = _response()
    response.stop_reason = reason
    client, messages = _client([response])
    result = ClaudeProgramAgent(client=client)._run_loop(
        "system", "prompt", (ToolSpec("probe", "", {}, handler),), continuation
    )
    handler.assert_not_called()
    continuation.assert_not_called()
    assert len(messages.calls) == result.response_count == 1
    assert result.stop_reason == ("model_finished" if reason == "end_turn" else reason)


@pytest.mark.parametrize(
    "options,reason",
    [
        ({"max_turns": 2}, "max_turns"),
        ({"max_cost_usd": 0.3, "prices": PRICES}, "max_cost"),
    ],
)
def test_response_caps_stop_before_running_its_tools(options, reason):
    client, messages = _client([_response(_usage(fresh=100_000)) for _ in range(3)])
    handler = Mock(return_value="ok")
    result = ClaudeProgramAgent(client=client, **options)._run_loop(
        "system", "prompt", (ToolSpec("probe", "", {}, handler),)
    )
    assert len(messages.calls) == 2
    assert handler.call_count == 1
    assert result.stop_reason == reason
    if reason == "max_cost":
        assert result.cost_usd == pytest.approx(0.4)


def test_direct_agent_default_stops_an_endless_tool_loop():
    client, messages = _client([_response() for _ in range(81)])
    handler = Mock(return_value="ok")
    result = ClaudeProgramAgent(client=client)._run_loop(
        "system", "prompt", (ToolSpec("probe", "", {}, handler),)
    )
    assert len(messages.calls) == result.response_count == 80
    assert handler.call_count == 79
    assert result.stop_reason == "max_turns"


def test_continuation_resumes_one_history_after_an_empty_turn():
    client, messages = _client(
        [
            _response(),
            _response(stop_reason="end_turn", content=[]),
            _response(stop_reason="end_turn"),
        ]
    )
    requests = iter(["keep going", None])
    result = ClaudeProgramAgent(client=client)._run_loop(
        "system", "prompt", (PROBE,), lambda count: next(requests)
    )
    history = messages.calls[-1]["messages"]
    assert [entry["role"] for entry in history] == ["user", "assistant", "user"]
    assert history[-1]["content"][-1] == {"type": "text", "text": "keep going"}
    assert history[-1]["content"][0]["type"] == "tool_result"
    assert result.continuation_requests == 1
    assert "cache_control" not in messages.calls[0]


def test_trial_floor_uses_run_local_continuation_count():
    session = SimpleNamespace(
        config=SimpleNamespace(max_trials=24, allow_target_transform=True),
        trials=[object()] * 3,
    )
    agent = ClaudeProgramAgent(min_trials=12, max_continuation_requests=2)
    message = agent.continuation(session, 0)
    assert "3 of the 24 trials" in message and "at least 12" in message
    assert "target_transform" in message
    assert agent.continuation(session, 2) is None
    session.trials = [object()] * 12
    assert agent.continuation(session, 0) is None


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_nonpositive_or_nonfinite_costs_and_prices_are_refused(value):
    with pytest.raises(ValueError, match="max_cost_usd"):
        ClaudeProgramAgent(max_cost_usd=value, prices=PRICES)
    with pytest.raises(ValueError, match="prices"):
        TokenPrices(**{**asdict(PRICES), "input": value})
    with pytest.raises(ValueError, match="request_timeout_seconds"):
        ClaudeProgramAgent(request_timeout_seconds=value)


def test_cost_requires_prices_and_cache_ttl_is_checked():
    with pytest.raises(ValueError, match="needs prices"):
        ClaudeProgramAgent(max_cost_usd=1.0)
    with pytest.raises(ValueError, match="cache_ttl"):
        ClaudeProgramAgent(cache_ttl="10m")
    with pytest.raises(ValueError, match="exactly"):
        TokenPrices.from_mapping({"input": 1.0})


def test_unsplit_cache_writes_use_the_five_minute_price():
    usage = TokenUsage()
    response_usage = _usage()
    response_usage.cache_creation = None
    response_usage.cache_creation_input_tokens = 1_000
    usage.add_anthropic(response_usage)
    assert usage.cache_write_5m_tokens == 1_000
    assert usage.cost(PRICES) == pytest.approx(0.0025)


def test_default_token_budget_passes_sdk_nonstreaming_validation(monkeypatch):
    anthropic = pytest.importorskip("anthropic")
    response = anthropic.types.Message(
        id="test",
        model="claude-sonnet-5-5",
        role="assistant",
        type="message",
        stop_reason="end_turn",
        content=[],
        usage=anthropic.types.Usage(input_tokens=1, output_tokens=1),
    )
    with anthropic.Anthropic(api_key="test-only") as client:
        send_request = Mock(return_value=response)
        monkeypatch.setattr(client.messages, "_post", send_request)
        result = ClaudeProgramAgent(client=client)._run_loop("system", "prompt", ())
    assert result.response_count == 1
    assert send_request.call_args.kwargs["body"]["max_tokens"] == 32768
