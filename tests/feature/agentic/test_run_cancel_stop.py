"""Cancel and stop take effect inside an agent search (run-control PR 6)."""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from run_control_fakes import FakeClock

from nori_rel.feature.agentic import (
    AgenticFeatureConfig,
    AgenticFeatures,
    FeatureProgram,
    FeatureQuery,
    PlanningSession,
    ScoreReport,
)
from nori_rel.feature.agentic._agent_base import ToolSpec
from nori_rel.feature.agentic.agent_claude import ClaudeProgramAgent
from nori_rel.feature.agentic.agent_openai import OpenAIProgramAgent
from nori_rel.feature.agentic.contracts import NoAcceptedProgram
from nori_rel.feature.agentic.session import ProgramRejected
from nori_rel.feature.agentic.usage import TokenPrices
from nori_rel.feature.dfs.noridfs import NoriDFS
from nori_rel.run import (
    LocalRun,
    RunCancelled,
    RunContext,
    RunEvent,
    RunLimits,
    progress_from_events,
)

EXTEND = AgenticFeatureConfig(
    max_trials=4,
    max_features=6,
    holdout_fraction=0.3,
    confirmation_fraction=0.2,
    leak_check_anchors=2,
    extend_baseline=True,
)
PROGRAM_COLUMN = "hist__n_events"


class _WinningScorer:
    """MAE 1.0 for the baseline and 0.5 once the program's columns join."""

    def __call__(self, context, target, holdout, holdout_target, **kwargs):
        mae = 0.5 if PROGRAM_COLUMN in context.columns else 1.0
        return ScoreReport(
            metric="neg_mae", score=-mae, noise=0.0, predictions=np.zeros(len(holdout))
        )


class _SlowScorer(_WinningScorer):
    """Moves ``clock`` forward by ``seconds_per_call`` during every scoring call."""

    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.seconds_per_call = 0.0

    def __call__(self, *args, **kwargs):
        self.clock.advance(self.seconds_per_call)
        return super().__call__(*args, **kwargs)


class _ScriptedAgent:
    """Runs ``steps`` against the session, with no boundaries of its own."""

    def __init__(self, steps) -> None:
        self.steps = steps

    def run(self, session: PlanningSession, task_hint: str) -> None:
        for step in self.steps:
            step(session)


def _observed_context(**options) -> tuple[RunContext, list[RunEvent]]:
    events: list[RunEvent] = []
    return RunContext("run_1", event_sink=events.append, **options), events


def _backend(agent, run_context: RunContext, *, fall_back: bool = True, scorer=None):
    return AgenticFeatures(
        agent=agent,
        scorer=scorer or _WinningScorer(),
        train_target=pd.Series([float(i) for i in range(10)]),
        fallback=NoriDFS(),
        config=EXTEND,
        run_context=run_context,
        fall_back_on_search_failure=fall_back,
    )


def _compute(backend: AgenticFeatures, snapshot, anchors):
    train = anchors(10, start="2020-01-03").assign(row_id=[f"c{i}" for i in range(10)])
    query = anchors(2, start="2020-01-14").assign(row_id=["q1", "q2"])
    return backend.compute(snapshot, train, query, "accounts", row_id_column="row_id")


def _event_kinds(events: list[RunEvent]) -> list[str]:
    return [event.data["kind"] for event in events]


def _fallback_reasons(events: list[RunEvent]) -> list[str]:
    return [
        event.data["reason"]
        for event in events
        if event.data["kind"] == "fallback_used"
    ]


def _submit(program: FeatureProgram):
    return lambda session: session.validate_program(program)


def _send(run_context: RunContext, command: str):
    return lambda session: run_context.commands.submit_command(command)


def _send_and_reach_a_boundary(run_context: RunContext, command: str):
    """Send ``command``, then reach the next request boundary, as agents do."""

    def send(session: PlanningSession) -> None:
        run_context.commands.submit_command(command)
        session.check_for_interrupt()

    return send


def _submit_expecting_stop_refusal(program: FeatureProgram):
    def submit(session: PlanningSession) -> None:
        with pytest.raises(ProgramRejected, match="the search was stopped"):
            session.validate_program(program)

    return submit


def test_a_stop_after_a_trial_keeps_its_program_when_confirmation_fits(
    snapshot, anchors, sql
) -> None:
    program = FeatureProgram((FeatureQuery("hist", sql["safe"]),))
    run_context, events = _observed_context(clock=FakeClock())
    backend = _backend(
        _ScriptedAgent(
            [
                _submit(program),
                _send(run_context, "stop"),
                _submit_expecting_stop_refusal(program),
            ]
        ),
        run_context,
    )

    matrices = _compute(backend, snapshot, anchors)

    assert PROGRAM_COLUMN in matrices.feature_names
    assert len(backend.last_session.trials) == 1
    assert backend.last_session.refusals == []
    stopped = next(e for e in events if e.data["kind"] == "search_stopped")
    assert stopped.data["trials"] == 1
    assert _fallback_reasons(events) == []
    assert progress_from_events("run_1", events).search_interrupted


def test_a_stop_refusal_an_agent_does_not_catch_still_ends_the_search_cleanly(
    snapshot, anchors, sql
) -> None:
    program = FeatureProgram((FeatureQuery("hist", sql["safe"]),))
    run_context, events = _observed_context(clock=FakeClock())
    backend = _backend(
        _ScriptedAgent(
            [_submit(program), _send(run_context, "stop"), _submit(program)]
        ),
        run_context,
    )

    matrices = _compute(backend, snapshot, anchors)

    assert PROGRAM_COLUMN in matrices.feature_names
    assert _fallback_reasons(events) == []


def test_a_stop_falls_back_when_confirmation_would_not_fit(
    snapshot, anchors, sql
) -> None:
    clock = FakeClock()
    program = FeatureProgram((FeatureQuery("hist", sql["safe"]),))
    run_context, events = _observed_context(
        clock=clock, limits=RunLimits(stop_grace_s=10.0)
    )

    def stop_with_a_confirmation_too_slow_for_the_budget(
        session: PlanningSession,
    ) -> None:
        run_context.commands.submit_command("stop")
        assert session.check_for_interrupt()
        # Two scoring calls of 6 s do not fit in half of the 10 s grace.
        session.latest_scoring_seconds = 6.0

    backend = _backend(
        _ScriptedAgent(
            [_submit(program), stop_with_a_confirmation_too_slow_for_the_budget]
        ),
        run_context,
    )

    matrices = _compute(backend, snapshot, anchors)

    assert PROGRAM_COLUMN not in matrices.feature_names
    assert _fallback_reasons(events) == ["stopped_before_confirmation"]
    assert progress_from_events("run_1", events).search_interrupted


def test_a_stop_before_any_trial_falls_back_without_a_program(
    snapshot, anchors, sql
) -> None:
    run_context, events = _observed_context()
    backend = _backend(
        _ScriptedAgent([_send_and_reach_a_boundary(run_context, "stop")]), run_context
    )

    matrices = _compute(backend, snapshot, anchors)

    assert PROGRAM_COLUMN not in matrices.feature_names
    assert backend.last_session.trials == []
    assert _fallback_reasons(events) == ["stopped_without_program"]
    assert progress_from_events("run_1", events).strategy_used == "noridfs"


def test_a_stop_before_the_search_skips_it(snapshot, anchors) -> None:
    run_context, events = _observed_context()
    run_context.commands.submit_command("stop")
    backend = _backend(_ScriptedAgent([]), run_context)

    _compute(backend, snapshot, anchors)

    assert backend.last_session is None
    assert _event_kinds(events)[-3:] == ["command", "search_stopped", "fallback_used"]
    assert _fallback_reasons(events) == ["stopped_without_program"]


def test_a_cancel_during_the_search_ends_it_without_a_fallback(
    snapshot, anchors, sql
) -> None:
    program = FeatureProgram((FeatureQuery("hist", sql["safe"]),))
    run_context, events = _observed_context()
    backend = _backend(
        _ScriptedAgent(
            [_submit(program), _send(run_context, "cancel"), _submit(program)]
        ),
        run_context,
    )

    with pytest.raises(RunCancelled):
        _compute(backend, snapshot, anchors)

    assert len(backend.last_session.trials) == 1
    assert backend.last_session.refusals == []
    assert _fallback_reasons(events) == []
    with pytest.raises(RunCancelled):
        run_context.apply_pending_commands()


class _ProviderRateLimited(Exception):
    """An SDK error with the HTTP status of a rate limit."""

    status_code = 429


@pytest.mark.parametrize(
    "failure,reason",
    [
        (_ProviderRateLimited("rate limited"), "search_failed"),
        (NoAcceptedProgram("no program"), "no_program_found"),
    ],
)
def test_a_failed_search_falls_back_only_when_asked(
    snapshot, anchors, failure, reason
) -> None:
    def fail(session: PlanningSession) -> None:
        raise failure

    run_context, events = _observed_context()
    backend = _backend(_ScriptedAgent([fail]), run_context)
    _compute(backend, snapshot, anchors)
    assert _fallback_reasons(events) == [reason]
    assert backend.last_fallback_reason == reason

    benchmark_backend = _backend(
        _ScriptedAgent([fail]), RunContext.unobserved(), fall_back=False
    )
    with pytest.raises(type(failure)):
        _compute(benchmark_backend, snapshot, anchors)


@pytest.mark.parametrize(
    "failure",
    [PermissionError("invalid x-api-key"), ImportError("no SDK"), TypeError("bug")],
)
def test_a_configuration_error_is_raised_even_with_the_fallback_on(
    snapshot, anchors, failure
) -> None:
    def fail(session: PlanningSession) -> None:
        raise failure

    run_context, events = _observed_context()

    with pytest.raises(type(failure)):
        _compute(_backend(_ScriptedAgent([fail]), run_context), snapshot, anchors)
    assert _fallback_reasons(events) == []


def test_a_cancel_pending_when_the_agent_raises_wins(snapshot, anchors) -> None:
    run_context, _ = _observed_context()

    def cancel_then_fail(session: PlanningSession) -> None:
        run_context.commands.submit_command("cancel")
        raise _ProviderRateLimited("rate limited")

    with pytest.raises(RunCancelled):
        _compute(
            _backend(_ScriptedAgent([cancel_then_fail]), run_context),
            snapshot,
            anchors,
        )


def test_a_stop_after_the_agent_finished_keeps_its_program(
    snapshot, anchors, sql
) -> None:
    program = FeatureProgram((FeatureQuery("hist", sql["safe"]),))
    run_context, events = _observed_context()
    # The agent finishes on its own; the stop arrives after its last boundary.
    backend = _backend(
        _ScriptedAgent([_submit(program), _send(run_context, "stop")]), run_context
    )

    matrices = _compute(backend, snapshot, anchors)

    assert PROGRAM_COLUMN in matrices.feature_names
    assert "search_stopped" not in _event_kinds(events)
    assert not progress_from_events("run_1", events).search_interrupted


def test_a_stop_mid_trial_does_not_use_up_the_confirmation_budget(
    snapshot, anchors, sql
) -> None:
    clock = FakeClock()
    program = FeatureProgram((FeatureQuery("hist", sql["safe"]),))
    run_context, events = _observed_context(
        clock=clock, limits=RunLimits(stop_grace_s=10.0)
    )

    def stop_then_finish_a_slow_trial(session: PlanningSession) -> None:
        run_context.commands.submit_command("stop")
        assert session.check_for_interrupt()
        clock.advance(45.0)

    backend = _backend(
        _ScriptedAgent([_submit(program), stop_then_finish_a_slow_trial]), run_context
    )

    matrices = _compute(backend, snapshot, anchors)

    assert PROGRAM_COLUMN in matrices.feature_names
    assert _fallback_reasons(events) == []


def test_a_cancel_during_confirmation_ends_the_run(snapshot, anchors, sql) -> None:
    program = FeatureProgram((FeatureQuery("hist", sql["safe"]),))
    run_context, _ = _observed_context()

    def cancel_at_the_next_scoring_call(session: PlanningSession) -> None:
        scorer = session.scorer

        def score_then_cancel(*args, **kwargs):
            run_context.commands.submit_command("cancel")
            return scorer(*args, **kwargs)

        session.scorer = score_then_cancel

    backend = _backend(
        _ScriptedAgent([_submit(program), cancel_at_the_next_scoring_call]),
        run_context,
    )

    with pytest.raises(RunCancelled):
        _compute(backend, snapshot, anchors)


def test_run_stop_through_a_local_run_ends_the_search_early(
    snapshot, anchors, sql
) -> None:
    program = FeatureProgram((FeatureQuery("hist", sql["safe"]),))
    run = LocalRun("run_1")

    def stop_the_run(session: PlanningSession) -> None:
        run.stop()
        session.check_for_interrupt()

    def predict(run_context: RunContext):
        backend = _backend(
            _ScriptedAgent([_submit(program), stop_the_run]), run_context
        )
        return _compute(backend, snapshot, anchors)

    run._execute_prediction(
        predict,
        strategy="agentic",
        max_trials=4,
        summary_in_result_attrs=False,
        reraise_failure=True,
    )

    assert PROGRAM_COLUMN in run.result().feature_names
    assert run.progress.status == "completed"
    assert run.progress.search_interrupted


def _anthropic_tool_use_message(message_id: str):
    anthropic = pytest.importorskip("anthropic")
    return anthropic.types.Message.model_validate(
        {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "model": "scripted",
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "content": [
                {
                    "type": "tool_use",
                    "id": f"{message_id}_tool",
                    "name": "probe",
                    "input": {},
                }
            ],
        }
    )


def _scripted_anthropic_client(model_calls: list[dict]):
    """Return a real Anthropic client whose requests each return a tool call."""
    anthropic = pytest.importorskip("anthropic")
    client = anthropic.Anthropic(api_key="scripted-not-a-key")

    def scripted_create(**params):
        model_calls.append(params)
        return _anthropic_tool_use_message(f"m{len(model_calls)}")

    client.messages.create = scripted_create
    return client


def _run_control(check_for_interrupt) -> SimpleNamespace:
    return SimpleNamespace(
        check_for_interrupt=check_for_interrupt,
        report_model_response=lambda cost_usd: None,
        has_held_steering=lambda: False,
        deliver_steering_text=lambda: None,
    )


def test_a_cancel_raised_by_a_tool_escapes_the_claude_loop() -> None:
    model_calls: list[dict] = []
    client = _scripted_anthropic_client(model_calls)

    def cancelled_tool() -> str:
        raise RunCancelled("run_1")

    agent = ClaudeProgramAgent(client=client)
    tool = ToolSpec(
        "probe", "probe", {"type": "object", "properties": {}}, cancelled_tool
    )

    with pytest.raises(RunCancelled):
        agent._run_loop("system", "prompt", (tool,))

    assert len(model_calls) == 1


def test_the_claude_loop_ends_before_a_request_once_the_run_asks() -> None:
    model_calls: list[dict] = []
    client = _scripted_anthropic_client(model_calls)
    agent = ClaudeProgramAgent(client=client)
    tool = ToolSpec(
        "probe", "probe", {"type": "object", "properties": {}}, lambda: "ok"
    )

    result = agent._run_loop(
        "system",
        "prompt",
        (tool,),
        run_control=_run_control(lambda: len(model_calls) >= 2),
    )

    assert len(model_calls) == result.response_count == 2
    assert result.stop_reason == "stopped"


def test_the_openai_loop_ends_before_a_request_once_the_run_asks() -> None:
    model_calls: list[dict] = []

    def create(**params):
        model_calls.append(params)
        call = SimpleNamespace(
            id=f"c{len(model_calls)}",
            function=SimpleNamespace(name="probe", arguments="{}"),
        )
        message = SimpleNamespace(content=None, tool_calls=[call], refusal=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="tool_calls")]
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    agent = OpenAIProgramAgent(model="m", client=client)
    tool = ToolSpec(
        "probe", "probe", {"type": "object", "properties": {}}, lambda: "ok"
    )

    result = agent._run_loop(
        "system",
        "prompt",
        (tool,),
        run_control=_run_control(lambda: len(model_calls) >= 3),
    )

    assert len(model_calls) == result.response_count == 3
    assert result.stop_reason == "stopped"


#: Long enough for a helper thread to reach or leave a point a test waits for.
WAIT_SECONDS = 5.0
PRICES = TokenPrices(
    input=3.0, output=15.0, cache_read=0.3, cache_write_5m=3.75, cache_write_1h=6.0
)


def _becomes_true_within_wait(condition) -> bool:
    deadline = time.monotonic() + WAIT_SECONDS
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.005)
    return True


def _reach_a_boundary_expecting_the_search_to_end(session: PlanningSession) -> None:
    assert session.check_for_interrupt()


def test_a_stop_sent_while_paused_ends_the_pause_and_keeps_the_best_program(
    snapshot, anchors, sql
) -> None:
    clock = FakeClock()
    program = FeatureProgram((FeatureQuery("hist", sql["safe"]),))
    run_context, events = _observed_context(clock=clock)
    released_by_the_stop: list[bool] = []

    def stop_after_a_pause_of_45_seconds() -> None:
        assert _becomes_true_within_wait(lambda: run_context.status == "paused")
        clock.advance(45.0)
        run_context.commands.submit_command("stop")
        released_by_the_stop.append(
            _becomes_true_within_wait(lambda: run_context.status == "running")
        )
        if not released_by_the_stop[0]:
            # Let a regression fail the test instead of holding it forever.
            run_context.commands.submit_command("resume")

    def pause_at_the_next_boundary(session: PlanningSession) -> None:
        run_context.commands.submit_command("pause")
        threading.Thread(target=stop_after_a_pause_of_45_seconds, daemon=True).start()
        assert session.check_for_interrupt()

    backend = _backend(
        _ScriptedAgent([_submit(program), pause_at_the_next_boundary]), run_context
    )

    matrices = _compute(backend, snapshot, anchors)

    assert released_by_the_stop == [True]
    assert PROGRAM_COLUMN in matrices.feature_names
    assert _fallback_reasons(events) == []
    statuses = [e.data["to"] for e in events if e.data["kind"] == "status_changed"]
    assert statuses == ["paused", "running"]
    assert progress_from_events("run_1", events).search_interrupted


def test_a_pause_sent_with_a_stop_is_rejected(snapshot, anchors, sql) -> None:
    program = FeatureProgram((FeatureQuery("hist", sql["safe"]),))
    run_context, events = _observed_context()

    def stop_and_pause_together(session: PlanningSession) -> None:
        run_context.commands.submit_command("stop")
        run_context.commands.submit_command("pause")
        assert session.check_for_interrupt()

    backend = _backend(
        _ScriptedAgent([_submit(program), stop_and_pause_together]), run_context
    )

    matrices = _compute(backend, snapshot, anchors)

    assert PROGRAM_COLUMN in matrices.feature_names
    outcomes = [
        (e.data["command"], e.data["status"], e.data.get("reason"))
        for e in events
        if e.data["kind"] == "command"
    ]
    assert outcomes == [
        ("stop", "applied", None),
        ("pause", "rejected", "not_applicable"),
    ]
    assert not any(e.data.get("to") == "paused" for e in events)


def test_reaching_the_spend_limit_ends_the_search_at_its_next_request(
    snapshot, anchors, sql
) -> None:
    program = FeatureProgram((FeatureQuery("hist", sql["safe"]),))
    run_context, events = _observed_context(limits=RunLimits(max_spend_usd=0.5))

    def spend_past_the_limit(session: PlanningSession) -> None:
        run_context.record_spend(0.75)

    backend = _backend(
        _ScriptedAgent(
            [
                _submit(program),
                spend_past_the_limit,
                _reach_a_boundary_expecting_the_search_to_end,
                _submit_expecting_stop_refusal(program),
            ]
        ),
        run_context,
    )

    matrices = _compute(backend, snapshot, anchors)

    assert PROGRAM_COLUMN in matrices.feature_names
    assert len(backend.last_session.trials) == 1
    kinds = _event_kinds(events)
    assert kinds.index("limit_reached") < kinds.index("search_stopped")
    assert progress_from_events("run_1", events).search_interrupted


def test_the_proposal_from_the_response_that_crossed_the_limit_is_scored(
    snapshot, anchors, sql
) -> None:
    program = FeatureProgram((FeatureQuery("hist", sql["safe"]),))
    run_context, _ = _observed_context(limits=RunLimits(max_spend_usd=0.5))

    def spend_past_the_limit(session: PlanningSession) -> None:
        run_context.record_spend(0.75)

    backend = _backend(
        _ScriptedAgent(
            [
                spend_past_the_limit,
                _submit(program),
                _reach_a_boundary_expecting_the_search_to_end,
            ]
        ),
        run_context,
    )

    matrices = _compute(backend, snapshot, anchors)

    assert len(backend.last_session.trials) == 1
    assert PROGRAM_COLUMN in matrices.feature_names


def test_a_limit_is_not_checked_after_the_search_ended(snapshot, anchors, sql) -> None:
    clock = FakeClock()
    program = FeatureProgram((FeatureQuery("hist", sql["safe"]),))
    run_context, events = _observed_context(
        clock=clock, limits=RunLimits(max_runtime_seconds=10)
    )
    run_context.change_status("running", strategy="agentic")
    scorer = _SlowScorer(clock)

    def finish_the_search_then_score_slowly(session: PlanningSession) -> None:
        scorer.seconds_per_call = 11.0

    backend = _backend(
        _ScriptedAgent([_submit(program), finish_the_search_then_score_slowly]),
        run_context,
        scorer=scorer,
    )

    matrices = _compute(backend, snapshot, anchors)

    kinds = _event_kinds(events)
    assert "limit_reached" not in kinds and "search_stopped" not in kinds
    assert PROGRAM_COLUMN in matrices.feature_names
    assert not progress_from_events("run_1", events).search_interrupted


def test_the_agents_reported_spend_reaches_the_spend_limit(
    snapshot, anchors, sql
) -> None:
    program = FeatureProgram((FeatureQuery("hist", sql["safe"]),))
    run_context, events = _observed_context(limits=RunLimits(max_spend_usd=0.5))

    def report_a_costly_response(session: PlanningSession) -> None:
        session.report_model_response(0.75)

    backend = _backend(
        _ScriptedAgent(
            [
                _submit(program),
                report_a_costly_response,
                _reach_a_boundary_expecting_the_search_to_end,
            ]
        ),
        run_context,
    )

    _compute(backend, snapshot, anchors)

    limit = next(e for e in events if e.data["kind"] == "limit_reached")
    assert limit.data["limit"] == "max_spend_usd"


def _validate_program_message(message_id: str, sql_text: str):
    """Return a 1000-in, 1000-out token response that submits one program."""
    anthropic = pytest.importorskip("anthropic")
    return anthropic.types.Message.model_validate(
        {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "model": "scripted",
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "usage": {"input_tokens": 1000, "output_tokens": 1000},
            "content": [
                {
                    "type": "tool_use",
                    "id": f"{message_id}_tool",
                    "name": "validate_program",
                    "input": {
                        "queries_json": json.dumps([{"name": "hist", "sql": sql_text}])
                    },
                }
            ],
        }
    )


def test_a_claude_agent_with_prices_stops_at_the_spend_limit(
    snapshot, anchors, sql
) -> None:
    anthropic = pytest.importorskip("anthropic")
    client = anthropic.Anthropic(api_key="scripted-not-a-key")
    model_calls: list[dict] = []

    def scripted_create(**params):
        model_calls.append(params)
        return _validate_program_message(f"m{len(model_calls)}", sql["safe"])

    client.messages.create = scripted_create
    # One response costs 1000 * $3/M + 1000 * $15/M = $0.018 at these prices.
    run_context, events = _observed_context(limits=RunLimits(max_spend_usd=0.01))
    agent = ClaudeProgramAgent(client=client, prices=PRICES, min_trials=0)

    _compute(_backend(agent, run_context), snapshot, anchors)

    assert len(model_calls) == 1
    limit = next(e for e in events if e.data["kind"] == "limit_reached")
    assert limit.data["limit"] == "max_spend_usd"
    spend = next(e for e in events if e.data["kind"] == "spend_updated")
    assert spend.data["spend_usd"] == pytest.approx(0.018)
