"""A steer is held until the agent sends it to the model, and applied only then."""

from __future__ import annotations

import copy
import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nori_rel.feature.agentic import (
    AgenticFeatureConfig,
    AgenticFeatures,
    PlanningSession,
    ScoreReport,
)
from nori_rel.feature.agentic._agent_base import ToolSpec
from nori_rel.feature.agentic.agent_claude import ClaudeProgramAgent
from nori_rel.feature.agentic.agent_openai import OpenAIProgramAgent
from nori_rel.feature.dfs.noridfs import NoriDFS
from nori_rel.run import RunCancelled, RunContext, RunEvent

STEERING_PREFIX = "Steering from the user"
STEERING = f"{STEERING_PREFIX}: Use order history."
EXTEND = AgenticFeatureConfig(
    max_trials=4,
    max_features=6,
    holdout_fraction=0.3,
    confirmation_fraction=0.2,
    leak_check_anchors=2,
    extend_baseline=True,
)
AGENT_LOOPS = ["claude", "openai"]
#: A scripted reply in which the model refuses to go on.
REFUSE = "refuse"
#: The kinds of event that show where a steer's outcome falls in a search.
_OUTLINED_KINDS = {"baseline_scored", "trial_scored", "command", "search_stopped"}
#: Long enough for a helper thread to reach or leave a point a test waits for.
WAIT_SECONDS = 5.0
PROBE = ToolSpec("probe", "probe", {"type": "object", "properties": {}}, lambda: "ok")


class _HeldSteering:
    """A run control that holds ``STEERING`` from when ``is_due()`` is true.

    It never asks the search to end. ``order`` records each interrupt check and
    each request for steering, so a test can see what came between them.
    """

    def __init__(self, is_due, order: list[str] | None = None) -> None:
        self.is_due = is_due
        self.order = [] if order is None else order
        self.delivered = False

    def check_for_interrupt(self) -> bool:
        """Record the check; return that the run does not want the search ended."""
        self.order.append("check")
        return False

    def report_model_response(self, cost_usd: float | None) -> None:
        """Ignore the response's spend."""

    def has_held_steering(self) -> bool:
        """Return whether ``STEERING`` is due and not yet delivered."""
        return self.is_due() and not self.delivered

    def deliver_steering_text(self) -> str | None:
        """Return ``STEERING`` the first time it is due, and ``None`` otherwise."""
        if not self.has_held_steering():
            self.order.append("no steering")
            return None
        self.delivered = True
        self.order.append("steering")
        return STEERING


def _get_texts_in_last_user_turn(messages: list[dict]) -> list[str]:
    """Return the text of every text block in the last message, a user turn."""
    content = messages[-1]["content"]
    if isinstance(content, str):
        return [content]
    return [block["text"] for block in content if block.get("type") == "text"]


def _has_steering_in_messages(messages: list[dict]) -> bool:
    """Return whether any message in ``messages`` carries steering text."""
    for message in messages:
        content = message["content"]
        texts = (
            [content]
            if isinstance(content, str)
            else [block.get("text", "") for block in content if isinstance(block, dict)]
        )
        if any(STEERING_PREFIX in text for text in texts):
            return True
    return False


def _claude_usage() -> SimpleNamespace:
    return SimpleNamespace(
        input_tokens=1,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
        cache_creation=None,
        output_tokens=1,
    )


def _get_claude_response(number: int, reply) -> SimpleNamespace:
    """Return the Anthropic response for a scripted ``reply`` to request ``number``.

    A reply is ``None`` to end the turn, ``REFUSE`` to refuse, or
    ``(tool, arguments)`` to call a tool.
    """
    if reply is None:
        stop_reason, content = "end_turn", [{"type": "text", "text": "done"}]
    elif reply == REFUSE:
        stop_reason, content = "refusal", []
    else:
        stop_reason = "tool_use"
        content = [
            {
                "type": "tool_use",
                "id": f"t{number}",
                "name": reply[0],
                "input": reply[1],
            }
        ]
    return SimpleNamespace(
        usage=_claude_usage(), stop_reason=stop_reason, content=content
    )


def _get_openai_response(number: int, reply) -> SimpleNamespace:
    """Return the OpenAI response for a scripted ``reply`` to request ``number``."""
    calls = None
    refusal = None
    if reply == REFUSE:
        refusal = "I can't help with that."
    elif reply is not None:
        calls = [
            SimpleNamespace(
                id=f"c{number}",
                function=SimpleNamespace(name=reply[0], arguments=json.dumps(reply[1])),
            )
        ]
    message = SimpleNamespace(
        content=None if calls else "done", tool_calls=calls, refusal=refusal
    )
    finish_reason = "tool_calls" if calls else "stop"
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)]
    )


def test_the_claude_loop_sends_steering_with_the_tool_results() -> None:
    requests: list[dict] = []
    replies = [("probe", {}), None]

    def create(**params):
        requests.append(copy.deepcopy(params))
        return _get_claude_response(len(requests), replies[len(requests) - 1])

    agent = ClaudeProgramAgent(
        client=SimpleNamespace(messages=SimpleNamespace(create=create))
    )

    agent._run_loop(
        "system",
        "prompt",
        (PROBE,),
        run_control=_HeldSteering(lambda: len(requests) >= 1),
    )

    assert len(requests) == 2
    second_request = requests[1]["messages"]
    assert STEERING in _get_texts_in_last_user_turn(second_request)
    assert any(
        block.get("type") == "tool_result" for block in second_request[-1]["content"]
    )


def test_the_openai_loop_sends_steering_instead_of_finishing() -> None:
    requests: list[list[dict]] = []

    def create(**params):
        requests.append([dict(message) for message in params["messages"]])
        return _get_openai_response(len(requests), None)

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    OpenAIProgramAgent(model="m", client=client)._run_loop(
        "system",
        "prompt",
        (),
        run_control=_HeldSteering(lambda: len(requests) >= 1),
    )

    assert len(requests) == 2
    assert requests[1][-1] == {"role": "user", "content": STEERING}


class _ScriptedModel:
    """The replies a model gives in turn, and what each request carried.

    A reply is ``None`` to end the turn, ``REFUSE`` to refuse, or
    ``(tool, arguments)`` to call a tool. ``on_request`` runs with the
    request's number (from 1) while that request is in flight, as a user
    acting during a model call would.
    """

    def __init__(self, replies: list, on_request=None) -> None:
        self.replies = replies
        self.on_request = on_request
        self.requests: list[list[dict]] = []

    @property
    def steering_in_requests(self) -> list[bool]:
        """Return, for each request so far, whether it carried steering."""
        return [_has_steering_in_messages(messages) for messages in self.requests]

    def get_next_reply(self, messages: list[dict]):
        """Record ``messages``, then return the next scripted reply."""
        self.requests.append(copy.deepcopy(messages))
        number = len(self.requests)
        assert number <= len(self.replies), "the agent made an unscripted request"
        if self.on_request is not None:
            self.on_request(number)
        return self.replies[number - 1]


def _get_claude_agent(model: _ScriptedModel, **options) -> ClaudeProgramAgent:
    """Return a Claude agent whose model answers from ``model``."""

    def create(**params):
        reply = model.get_next_reply(params["messages"])
        return _get_claude_response(len(model.requests), reply)

    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    return ClaudeProgramAgent(client=client, **options)


def _get_openai_agent(model: _ScriptedModel, **options) -> OpenAIProgramAgent:
    """Return an OpenAI agent whose model answers from ``model``."""

    def create(**params):
        reply = model.get_next_reply(params["messages"])
        return _get_openai_response(len(model.requests), reply)

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    return OpenAIProgramAgent(model="scripted", client=client, **options)


@pytest.fixture(params=AGENT_LOOPS)
def build_agent(request):
    """Return a builder of the Claude or the OpenAI agent over a scripted model."""
    return {"claude": _get_claude_agent, "openai": _get_openai_agent}[request.param]


def _get_program_proposal(sql_text: str):
    """Return the reply that calls ``validate_program`` with one query."""
    queries = [{"name": "hist", "sql": sql_text}]
    return ("validate_program", {"queries_json": json.dumps(queries)})


def test_no_interrupt_check_comes_between_taking_steering_and_the_request(
    build_agent,
) -> None:
    order: list[str] = []
    model = _ScriptedModel(
        [None, None], on_request=lambda number: order.append(f"request {number}")
    )
    run_control = _HeldSteering(lambda: len(model.requests) >= 1, order)

    build_agent(model)._run_loop("system", "prompt", (PROBE,), run_control=run_control)

    assert model.steering_in_requests == [False, True]
    assert order[order.index("steering") + 1] == "request 2"


class _ConstantScorer:
    def __call__(self, context, target, holdout, holdout_target, **kwargs):
        return ScoreReport(
            metric="neg_mae", score=-1.0, noise=0.0, predictions=np.zeros(len(holdout))
        )


def _get_command_outcomes(events: list[RunEvent]) -> list[tuple[str, str, str | None]]:
    """Return each command event as (command, status, rejection reason)."""
    return [
        (event.data["command"], event.data["status"], event.data.get("reason"))
        for event in events
        if event.data["kind"] == "command"
    ]


def _get_search_outline(events: list[RunEvent]) -> list[str]:
    """Return baseline, trial, command and stop events as short labels, in order."""
    outline = []
    for event in events:
        kind = event.data["kind"]
        if kind not in _OUTLINED_KINDS:
            continue
        if kind != "command":
            outline.append(kind)
            continue
        label = f"{event.data['command']}:{event.data['status']}"
        if "reason" in event.data:
            label += f":{event.data['reason']}"
        outline.append(label)
    return outline


def _get_fallback_reasons(events: list[RunEvent]) -> list[str]:
    """Return the reason of every fallback the search reported, in order."""
    return [
        event.data["reason"]
        for event in events
        if event.data["kind"] == "fallback_used"
    ]


def _get_observed_agent_search_context() -> tuple[RunContext, list[RunEvent]]:
    """Return an observed context set up as an agent search sets up its run."""
    events: list[RunEvent] = []
    run_context = RunContext("run_1", event_sink=events.append)
    run_context.steering_enabled = True
    return run_context, events


def _get_steer_during_request(run_context: RunContext, request_number: int, *texts):
    """Return an ``on_request`` that steers while request ``request_number`` runs."""

    def steer_if_this_is_the_request(number: int) -> None:
        if number == request_number:
            for text in texts or ("Use order history.",):
                run_context.commands.submit_command("steer", text)

    return steer_if_this_is_the_request


def _get_backend(agent, run_context: RunContext) -> AgenticFeatures:
    return AgenticFeatures(
        agent=agent,
        scorer=_ConstantScorer(),
        train_target=pd.Series([float(i) for i in range(10)]),
        fallback=NoriDFS(),
        config=EXTEND,
        run_context=run_context,
        fall_back_on_search_failure=True,
    )


def _search(backend: AgenticFeatures, snapshot, anchors) -> None:
    train = anchors(10, start="2020-01-03").assign(row_id=[f"c{i}" for i in range(10)])
    query = anchors(2, start="2020-01-14").assign(row_id=["q1", "q2"])
    backend.compute(snapshot, train, query, "accounts", row_id_column="row_id")


def test_steering_is_held_for_the_agent_and_rejected_once_the_search_ends(
    snapshot, anchors
) -> None:
    run_context, events = _get_observed_agent_search_context()
    received: list[str | None] = []
    outcomes_while_held: list[list] = []

    class _SteeredAgent:
        def run(self, session: PlanningSession, task_hint: str) -> None:
            run_context.commands.submit_command("steer", "Use order history.")
            session.check_for_interrupt()
            outcomes_while_held.append(_get_command_outcomes(events))
            received.append(session.deliver_steering_text())
            received.append(session.deliver_steering_text())

    _search(_get_backend(_SteeredAgent(), run_context), snapshot, anchors)
    run_context.commands.submit_command("steer", "Too late.")
    run_context.apply_pending_commands()

    assert outcomes_while_held == [[]]
    assert received == [STEERING, None]
    assert _get_command_outcomes(events) == [
        ("steer", "applied", None),
        ("steer", "rejected", "not_applicable"),
    ]


def test_a_steer_sent_before_the_agent_starts_goes_out_in_its_first_request(
    snapshot, anchors, build_agent
) -> None:
    run_context, events = _get_observed_agent_search_context()
    run_context.commands.submit_command("steer", "Use order history.")
    model = _ScriptedModel([None])

    _search(_get_backend(build_agent(model), run_context), snapshot, anchors)

    assert model.steering_in_requests == [True]
    assert _get_search_outline(events) == ["baseline_scored", "steer:applied"]


def test_two_held_steers_go_out_joined_in_one_request(
    snapshot, anchors, build_agent
) -> None:
    run_context, events = _get_observed_agent_search_context()
    model = _ScriptedModel(
        [None, None],
        on_request=_get_steer_during_request(
            run_context, 1, "Use order history.", "Skip the refunds table."
        ),
    )

    _search(_get_backend(build_agent(model), run_context), snapshot, anchors)

    assert model.steering_in_requests == [False, True]
    assert _get_texts_in_last_user_turn(model.requests[1])[-1] == (
        f"{STEERING}\n\n{STEERING_PREFIX}: Skip the refunds table."
    )
    assert _get_search_outline(events) == [
        "baseline_scored",
        "steer:applied",
        "steer:applied",
    ]


def test_a_steer_is_applied_after_the_trial_of_a_program_proposed_before_it(
    snapshot, anchors, sql, build_agent
) -> None:
    run_context, events = _get_observed_agent_search_context()
    model = _ScriptedModel(
        [_get_program_proposal(sql["safe"]), None],
        on_request=_get_steer_during_request(run_context, 1),
    )

    _search(_get_backend(build_agent(model), run_context), snapshot, anchors)

    assert model.steering_in_requests == [False, True]
    assert _get_search_outline(events) == [
        "baseline_scored",
        "trial_scored",
        "steer:applied",
    ]


@pytest.mark.parametrize("stop_comes_first", [False, True])
def test_a_steer_sent_with_a_stop_is_rejected_and_never_sent(
    snapshot, anchors, sql, build_agent, stop_comes_first
) -> None:
    run_context, events = _get_observed_agent_search_context()

    def send_steer_and_stop_during_the_first_request(number: int) -> None:
        commands = ["stop", "steer"] if stop_comes_first else ["steer", "stop"]
        for command in commands:
            text = "Use order history." if command == "steer" else None
            run_context.commands.submit_command(command, text)

    model = _ScriptedModel(
        [_get_program_proposal(sql["safe"]), None],
        on_request=send_steer_and_stop_during_the_first_request,
    )

    _search(_get_backend(build_agent(model), run_context), snapshot, anchors)

    assert model.steering_in_requests == [False]
    assert _get_search_outline(events) == [
        "baseline_scored",
        "stop:applied",
        "steer:rejected:not_applicable",
        "search_stopped",
    ]
    rejection = next(e for e in events if e.data.get("reason") == "not_applicable")
    assert "text" not in rejection.data
    assert "Use order history" not in rejection.message


def test_a_steer_is_not_applied_when_the_agent_ends_at_its_turn_limit(
    snapshot, anchors, build_agent
) -> None:
    run_context, events = _get_observed_agent_search_context()
    model = _ScriptedModel([None], on_request=_get_steer_during_request(run_context, 1))

    _search(
        _get_backend(build_agent(model, max_turns=1), run_context), snapshot, anchors
    )

    assert model.steering_in_requests == [False]
    assert _get_search_outline(events) == [
        "baseline_scored",
        "steer:rejected:not_applicable",
    ]


def test_a_stop_during_the_final_response_finds_the_search_already_over(
    snapshot, anchors, sql, build_agent
) -> None:
    run_context, events = _get_observed_agent_search_context()

    def stop_during_the_final_request(number: int) -> None:
        if number == 2:
            run_context.commands.submit_command("stop")

    model = _ScriptedModel(
        [_get_program_proposal(sql["safe"]), None],
        on_request=stop_during_the_final_request,
    )

    _search(_get_backend(build_agent(model), run_context), snapshot, anchors)

    # The model finished on its own, so the stop ended nothing early.
    assert _get_search_outline(events) == [
        "baseline_scored",
        "trial_scored",
        "stop:applied",
    ]


def test_a_refusal_ends_the_search_and_rejects_the_steer_it_held(
    snapshot, anchors, build_agent
) -> None:
    run_context, events = _get_observed_agent_search_context()
    model = _ScriptedModel(
        [REFUSE], on_request=_get_steer_during_request(run_context, 1)
    )

    _search(_get_backend(build_agent(model), run_context), snapshot, anchors)

    assert model.steering_in_requests == [False]
    assert _get_search_outline(events) == [
        "baseline_scored",
        "steer:rejected:not_applicable",
    ]
    assert _get_fallback_reasons(events) == ["no_program_found"]


def test_a_steer_sent_during_a_pause_goes_out_after_the_resume(
    snapshot, anchors, build_agent
) -> None:
    run_context, events = _get_observed_agent_search_context()
    paused_in_time: list[bool] = []

    def steer_while_paused_then_resume() -> None:
        deadline = time.monotonic() + WAIT_SECONDS
        while run_context.status != "paused" and time.monotonic() < deadline:
            time.sleep(0.005)
        paused_in_time.append(run_context.status == "paused")
        run_context.commands.submit_command("steer", "Use order history.")
        # Resume even if the pause never came, so a regression fails, not hangs.
        run_context.commands.submit_command("resume")

    helper = threading.Thread(target=steer_while_paused_then_resume, daemon=True)

    def pause_during_the_first_request(number: int) -> None:
        if number == 1:
            run_context.commands.submit_command("pause")
            helper.start()

    model = _ScriptedModel([None, None], on_request=pause_during_the_first_request)

    _search(_get_backend(build_agent(model), run_context), snapshot, anchors)
    helper.join(WAIT_SECONDS)

    assert paused_in_time == [True]
    assert model.steering_in_requests == [False, True]
    assert _get_search_outline(events) == [
        "baseline_scored",
        "pause:applied",
        "resume:applied",
        "steer:applied",
    ]


def test_a_stop_before_the_agent_starts_rejects_a_held_steer_at_once(
    snapshot, anchors
) -> None:
    run_context, events = _get_observed_agent_search_context()
    run_context.commands.submit_command("steer", "Use order history.")
    run_context.commands.submit_command("stop")
    agent_started: list[bool] = []

    class _Agent:
        def run(self, session: PlanningSession, task_hint: str) -> None:
            agent_started.append(True)

    _search(_get_backend(_Agent(), run_context), snapshot, anchors)

    assert agent_started == []
    assert not run_context.steering_enabled
    assert _get_search_outline(events) == [
        "stop:applied",
        "steer:rejected:not_applicable",
        "search_stopped",
    ]


def test_a_cancel_after_a_steer_was_held_rejects_it_as_run_finished(
    snapshot, anchors
) -> None:
    run_context, events = _get_observed_agent_search_context()

    class _CancelledAgent:
        def run(self, session: PlanningSession, task_hint: str) -> None:
            run_context.commands.submit_command("steer", "Use order history.")
            session.check_for_interrupt()
            run_context.commands.submit_command("cancel")
            session.check_for_interrupt()

    with pytest.raises(RunCancelled):
        _search(_get_backend(_CancelledAgent(), run_context), snapshot, anchors)

    assert not run_context.steering_enabled
    assert _get_command_outcomes(events) == [
        ("cancel", "applied", None),
        ("steer", "rejected", "run_finished"),
    ]
