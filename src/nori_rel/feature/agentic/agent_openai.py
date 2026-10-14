"""An OpenAI-compatible model as the program agent, through chat completions.

The same class reaches OpenAI, a hosted open-weight provider, or a local
vLLM server: they all speak the chat-completions tool protocol, so only the
base URL and key differ.
"""

from __future__ import annotations

import json
from collections.abc import Callable
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
from .usage import AgentRunResult, AgentStopReason

#: Request fields that can carry the per-response token cap. OpenAI's own API
#: takes ``max_completion_tokens``; its reasoning models refuse ``max_tokens``,
#: which many OpenAI-compatible servers still read instead.
TOKEN_LIMIT_FIELDS = ("max_tokens", "max_completion_tokens")


class OpenAIProgramAgent(ProgramAgentBase):
    """Run the loop with function tools on ``client.chat.completions``.

    The loop appends every tool result and re-sends until the model answers
    without tool calls or ``max_turns`` is reached. A tool the model names
    but the session does not define comes back as an error string rather
    than raising, so an open-weight model can recover from a bad call.
    """

    def __init__(
        self,
        *,
        model: str,
        max_tokens: int = DEFAULT_MAX_RESPONSE_TOKENS,
        max_turns: int = DEFAULT_MAX_AGENT_TURNS,
        temperature: float = 1.0,
        top_p: float | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        client: Any | None = None,
        min_trials: int = 0,
        max_continuation_requests: int = DEFAULT_MAX_CONTINUATION_REQUESTS,
        token_limit_field: str = "max_tokens",
    ) -> None:
        """Keep the model id, budgets, sampling, and endpoint or a prebuilt client.

        ``top_p`` is only sent when given, so a run that never set it keeps
        the server's default and stays comparable with earlier runs.
        ``min_trials`` is the search floor described on ``ProgramAgentBase``.
        ``token_limit_field`` is the request field that carries ``max_tokens``,
        one of ``TOKEN_LIMIT_FIELDS``; the default suits compatible servers.
        """
        super().__init__(
            model=model,
            max_tokens=max_tokens,
            max_turns=max_turns,
            min_trials=min_trials,
            max_continuation_requests=max_continuation_requests,
        )
        if token_limit_field not in TOKEN_LIMIT_FIELDS:
            raise ValueError(
                f"token_limit_field must be one of {TOKEN_LIMIT_FIELDS}, "
                f"not {token_limit_field!r}"
            )
        self.token_limit_field = token_limit_field
        self.temperature = temperature
        self.top_p = top_p
        self.base_url = base_url
        self.api_key = api_key
        self.client = client

    def _run_loop(
        self,
        system: str,
        prompt: str,
        tools: tuple[ToolSpec, ...],
        continuation: Callable[[int], str | None] | None = None,
        *,
        run_control: SearchRunControl | None = None,
    ) -> AgentRunResult:
        """Alternate model responses and tool outputs until the model stops.

        A turn without tool calls ends the loop unless ``continuation``
        returns a message, which becomes the next user turn. ``run_control`` is
        asked before every request whether the run wants the search to end,
        and hears about every response. Its held steering becomes the next
        user turn, and keeps a search whose model has finished going for one
        more request.
        """
        client = self.client or self._create_client()
        handlers = {spec.name: spec.handler for spec in tools}
        schema = [_openai_tool_schema(spec) for spec in tools]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        sampling: dict[str, Any] = {"temperature": self.temperature}
        if self.top_p is not None:
            sampling["top_p"] = self.top_p
        continuation_requests = 0
        stop_reason = AgentStopReason.MODEL_FINISHED
        response_count = 0
        while response_count < self.max_turns:
            should_stop, steering = get_stop_flag_and_steering_text(run_control)
            if should_stop:
                stop_reason = AgentStopReason.STOPPED
                break
            if steering is not None:
                messages.append({"role": "user", "content": steering})
            response = client.chat.completions.create(
                model=self.model,
                tools=schema,
                messages=messages,
                **{self.token_limit_field: self.max_tokens},
                **sampling,
            )
            response_count += 1
            if run_control is not None:
                run_control.report_model_response(None)
            if response_count >= self.max_turns:
                stop_reason = AgentStopReason.MAX_TURNS
                break
            choice = response.choices[0]
            message = choice.message
            if getattr(message, "refusal", None):
                stop_reason = "refusal"
                break
            if choice.finish_reason not in {"tool_calls", "stop"}:
                stop_reason = choice.finish_reason or AgentStopReason.INVALID_RESPONSE
                break
            calls = list(message.tool_calls or ())
            if choice.finish_reason == "stop" and calls:
                break
            if choice.finish_reason == "tool_calls" and not calls:
                stop_reason = AgentStopReason.INVALID_RESPONSE
                break
            # A reasoning model (Kimi, DeepSeek, Qwen thinking) returns its
            # reasoning beside the content and expects it back with the
            # tool results, or the next turn starts from a shorter context
            # than the one that chose the call.
            reasoning = getattr(message, "reasoning_content", None)
            messages.append(
                {
                    "role": "assistant",
                    "content": message.content or "",
                    **({"reasoning_content": reasoning} if reasoning else {}),
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.function.name,
                                "arguments": call.function.arguments,
                            },
                        }
                        for call in calls
                    ],
                }
            )
            if not calls:
                request = (
                    None
                    if continuation is None
                    else continuation(continuation_requests)
                )
                if request is not None:
                    continuation_requests += 1
                    messages.append({"role": "user", "content": request})
                elif not is_steering_waiting(run_control):
                    break
                continue
            for call in calls:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": _dispatch_tool_call(handlers, call),
                    }
                )
        return AgentRunResult(
            model=self.model,
            usage=None,
            cost_usd=None,
            stop_reason=stop_reason,
            continuation_requests=continuation_requests,
            response_count=response_count,
        )

    def _create_client(self) -> Any:
        from openai import OpenAI

        kwargs: dict[str, Any] = {}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.api_key:
            kwargs["api_key"] = self.api_key
        return OpenAI(**kwargs)


def _dispatch_tool_call(handlers: dict[str, Any], call: Any) -> str:
    """Run one tool call, returning any failure as text for the model."""
    handler = handlers.get(call.function.name)
    if handler is None:
        return f"rejected: unknown tool {call.function.name!r}"
    try:
        arguments = json.loads(call.function.arguments or "{}")
    except json.JSONDecodeError as error:
        return f"rejected: arguments are not valid JSON ({error})"
    if not isinstance(arguments, dict):
        return "rejected: arguments must be a JSON object"
    try:
        return handler(**arguments)
    except TypeError as error:
        return f"rejected: {error}"


def _openai_tool_schema(spec: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters,
        },
    }


__all__ = ["TOKEN_LIMIT_FIELDS", "OpenAIProgramAgent"]
