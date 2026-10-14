"""Claude program search with one conversation and response-level budget checks."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

from ._agent_base import (
    DEFAULT_MAX_AGENT_TURNS,
    DEFAULT_MAX_CONTINUATION_REQUESTS,
    DEFAULT_MAX_RESPONSE_TOKENS,
    ProgramAgentBase,
    ToolSpec,
    get_stop_flag_and_steering_text,
    is_steering_waiting,
)
from .contracts import SearchRunControl
from .usage import AgentRunResult, AgentStopReason, TokenPrices, TokenUsage

DEFAULT_REQUEST_TIMEOUT_SECONDS = 1800


class ClaudeProgramAgent(ProgramAgentBase):
    """Call ``messages.create`` without an SDK-managed second conversation.

    Each run returns its own usage and stop reason. A response that reaches a
    cost or turn cap cannot execute tools. The dollar cap is checked after a
    response, so one response can exceed it. ``cache_ttl`` enables automatic
    prompt caching; prices must be supplied to calculate a cost.
    """

    def __init__(
        self,
        *,
        model: str = "claude-opus-5",
        effort: str = "high",
        max_tokens: int = DEFAULT_MAX_RESPONSE_TOKENS,
        max_turns: int = DEFAULT_MAX_AGENT_TURNS,
        cache_ttl: str | None = None,
        prices: TokenPrices | None = None,
        max_cost_usd: float | None = None,
        min_trials: int = 0,
        max_continuation_requests: int = DEFAULT_MAX_CONTINUATION_REQUESTS,
        request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        client: Any | None = None,
        on_usage: Callable[[float | None], None] | None = None,
    ) -> None:
        """Set finite budgets; ``on_usage`` receives this run's cost per response.

        The explicit timeout permits non-streaming responses above the SDK's
        default ten-minute estimate. Choose a token cap supported by the model.
        """
        super().__init__(
            model=model,
            max_tokens=max_tokens,
            max_turns=max_turns,
            min_trials=min_trials,
            max_continuation_requests=max_continuation_requests,
        )
        if cache_ttl not in (None, "5m", "1h"):
            raise ValueError(f"cache_ttl must be '5m', '1h' or None, not {cache_ttl!r}")
        if max_cost_usd is not None:
            if not math.isfinite(max_cost_usd) or max_cost_usd <= 0:
                raise ValueError("max_cost_usd must be finite and positive")
            if prices is None:
                raise ValueError("max_cost_usd needs prices to compute the cost")
        if not math.isfinite(request_timeout_seconds) or request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be finite and positive")
        self.effort = effort
        self.cache_ttl = cache_ttl
        self.prices = prices
        self.max_cost_usd = max_cost_usd
        self.request_timeout_seconds = request_timeout_seconds
        self.client = client
        self.on_usage = on_usage

    def _run_loop(
        self,
        system: str,
        prompt: str,
        tools: tuple[ToolSpec, ...],
        continuation: Callable[[int], str | None] | None = None,
        *,
        run_control: SearchRunControl | None = None,
    ) -> AgentRunResult:
        """Execute tools only on complete tool-use responses below both caps.

        ``run_control`` is asked before every request whether the run wants the
        search to end, and hears about every response. Its held steering is
        delivered as part of the next user turn, and keeps a search whose
        model has finished going for one more request.
        """
        client = self.client
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        caching: dict[str, Any] = {}
        if self.cache_ttl is not None:
            caching["cache_control"] = {"type": "ephemeral", "ttl": self.cache_ttl}
        tool_schema = [_anthropic_tool_schema(spec) for spec in tools]
        handlers = {spec.name: spec.handler for spec in tools}
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        usage = TokenUsage()
        continuation_requests = 0
        stop_reason = AgentStopReason.MODEL_FINISHED
        response_count = 0
        while response_count < self.max_turns:
            should_stop, steering = get_stop_flag_and_steering_text(run_control)
            if should_stop:
                stop_reason = AgentStopReason.STOPPED
                break
            if steering is not None:
                _append_user_text(messages, steering)
            message = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                timeout=self.request_timeout_seconds,
                system=system,
                output_config={"effort": self.effort},
                tools=tool_schema,
                messages=messages,
                **caching,
            )
            response_count += 1
            usage.add_anthropic(message.usage)
            cost = None if self.prices is None else usage.cost(self.prices)
            if self.on_usage is not None:
                self.on_usage(cost)
            if run_control is not None:
                run_control.report_model_response(cost)
            if self.max_cost_usd is not None and cost >= self.max_cost_usd:
                stop_reason = AgentStopReason.MAX_COST
                break
            if response_count >= self.max_turns:
                stop_reason = AgentStopReason.MAX_TURNS
                break
            if message.stop_reason not in {"tool_use", "end_turn"}:
                stop_reason = message.stop_reason or AgentStopReason.INVALID_RESPONSE
                break
            content = [
                block.model_dump(mode="json", exclude_none=True)
                if hasattr(block, "model_dump")
                else block
                for block in message.content
            ]
            calls = [block for block in content if block["type"] == "tool_use"]
            if message.stop_reason == "tool_use":
                if not calls:
                    stop_reason = AgentStopReason.INVALID_RESPONSE
                    break
                messages.append({"role": "assistant", "content": content})
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": call["id"],
                                "content": _dispatch_tool_call(handlers, call),
                            }
                            for call in calls
                        ],
                    }
                )
                continue
            # Do not retain terminal tool blocks as unanswered calls.
            if calls:
                break
            if content:
                messages.append({"role": "assistant", "content": content})
            request = (
                None if continuation is None else continuation(continuation_requests)
            )
            if request is not None:
                continuation_requests += 1
                _append_user_text(messages, request)
            elif not is_steering_waiting(run_control):
                break
        return AgentRunResult(
            model=self.model,
            usage=usage,
            cost_usd=None if self.prices is None else usage.cost(self.prices),
            stop_reason=stop_reason,
            continuation_requests=continuation_requests,
            response_count=response_count,
        )


def _append_user_text(messages: list[dict[str, Any]], text: str) -> None:
    """Add ``text`` as the next user turn, joining a trailing user turn.

    An empty final assistant turn is not retained, so the text may need to
    join the preceding tool-results turn instead.
    """
    if messages and messages[-1]["role"] == "user":
        last = messages[-1]
        content = last["content"]
        blocks = (
            list(content)
            if isinstance(content, list)
            else [{"type": "text", "text": content}]
        )
        messages[-1] = {**last, "content": [*blocks, {"type": "text", "text": text}]}
        return
    messages.append({"role": "user", "content": text})


def _anthropic_tool_schema(spec: ToolSpec) -> dict[str, Any]:
    """Translate one session tool into a Messages API tool definition."""
    return {
        "name": spec.name,
        "description": spec.description,
        "input_schema": dict(spec.parameters),
    }


def _dispatch_tool_call(handlers: Mapping[str, Callable[..., str]], call: dict) -> str:
    """Return a tool result or a recoverable argument error."""
    handler = handlers.get(call["name"])
    if handler is None:
        return f"rejected: unknown tool {call['name']!r}"
    arguments = call["input"]
    if not isinstance(arguments, dict):
        return "rejected: arguments must be a JSON object"
    try:
        return handler(**arguments)
    except TypeError as error:
        return f"rejected: {error}"


__all__ = ["ClaudeProgramAgent"]
