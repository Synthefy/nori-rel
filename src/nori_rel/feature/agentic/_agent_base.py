"""Provider-neutral base for program agents.

The base class owns everything that does not depend on an SDK: the tool
surface over ``PlanningSession``, the prompts, and the JSON program parser.
A search must return an accepted program or leave the baseline in place.
A provider subclass translates ``ToolSpec`` entries and runs its request loop.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from .contracts import (
    CUTOFF_ALIAS,
    EVAL_TABLE,
    ROW_COLUMN,
    TARGET_TRANSFORMS,
    FeatureProgram,
    FeatureQuery,
    NoAcceptedProgram,
    SearchRunControl,
)
from .session import PlanningSession, ProgramRejected
from .usage import AgentRunResult

_MAX_EXPLORATION_ROWS = 50
DEFAULT_MAX_RESPONSE_TOKENS = 32768
DEFAULT_MAX_AGENT_TURNS = 80
DEFAULT_MAX_CONTINUATION_REQUESTS = 24

SYSTEM_PROMPT = f"""\
You are the data scientist for a relational prediction task. You write SQL
feature queries; frozen Nori, an in-context tabular foundation model, scores
them. Nori is not trained: it reads a sample of labelled rows as context and
predicts each holdout row in one forward pass. There are no hyperparameters,
so every gain has to come from the feature table.

Rules for every query:
- Anchor on `{EVAL_TABLE} e` and return `e.{ROW_COLUMN}` exactly once per row,
  plus feature columns that are numeric, low-cardinality text, or prose.
- Only rows strictly before the anchor are known: filter every temporal table
  with `time_col < e.{CUTOFF_ALIAS}`. Never compare with `>=` or `>`, and never
  add an interval to the cutoff. A trailing window is
  `time_col >= e.{CUTOFF_ALIAS} - INTERVAL 'N days' AND time_col < e.{CUTOFF_ALIAS}`.
- No labels exist in the database; Nori conditions on them itself.

What Nori rewards:
- A narrow table. Context rows scale as budget / (features + 1), so 40 columns
  see far more context than 200. Aim for 20 to 60 informative columns.
- Missing values, which carry signal. Do not COALESCE to 0 unless 0 is the
  true value. Leave NULL where an entity has no history.
- Bounded values. Use NULLIF(count, 0) before dividing and clip ratios.
- Rates, ratios, and within-entity deviations over raw totals, plus one column
  that restores absolute scale such as log(1 + total).
- Coverage columns, such as event counts and history span in days, so the
  model can tell a quiet entity from a new one.
Redundant near-duplicate aggregates cost context rows and add nothing.

Prose is a feature. A description, title, body or summary column is encoded
and reduced to 16 dimensions; describe_column reports `kind: text` with a mean
length and a sample so you can judge one. The whole text path costs 16 columns
of width however many prose columns you return, so one is nearly as cheap as
three, and those 16 columns are context rows you do not get. Take it where the
prose says something the relational columns cannot, and skip it for a short
label you could bucket into a category instead. Prose obeys the cutoff like
everything else: join the entity's point-in-time row, never its latest.

Work in this order: describe_schema, profile the columns that look relevant,
run_sql to check assumptions, then validate_program. Read the score, its
noise, and the per-query coverage, then revise. A change is real only when it
beats the best by more than the noise margin. Stop when the score plateaus or
the trial budget is spent, and finish with a short rationale.
"""


@dataclass(frozen=True)
class ToolSpec:
    """One tool the model may call, in SDK-neutral form.

    ``parameters`` is a JSON Schema object. ``handler`` takes the parsed
    arguments and returns text; it never raises for a bad program, it
    returns the rejection so the model can repair the query.
    """

    name: str
    description: str
    parameters: Mapping[str, object]
    handler: Callable[..., str]


class ProgramAgentBase(ABC):
    """Shared tool surface and prompts; subclasses drive one SDK's loop.

    ``min_trials`` keeps a search going: when the model ends its turn with
    fewer scored trials than that, it is told how many it has used and asked
    for another program, at most ``max_continuation_requests`` times. A zero
    ``min_trials``, the default, lets the model stop whenever it likes.
    """

    def __init__(
        self,
        *,
        model: str,
        max_tokens: int = DEFAULT_MAX_RESPONSE_TOKENS,
        max_turns: int = DEFAULT_MAX_AGENT_TURNS,
        min_trials: int = 0,
        max_continuation_requests: int = DEFAULT_MAX_CONTINUATION_REQUESTS,
    ) -> None:
        """Keep the model id, the per-response token cap, and the search floor."""
        for name, value, minimum in (
            ("max_tokens", max_tokens, 1),
            ("max_turns", max_turns, 1),
            ("min_trials", min_trials, 0),
            ("max_continuation_requests", max_continuation_requests, 0),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        self.model = model
        self.max_tokens = max_tokens
        self.max_turns = max_turns
        self.min_trials = min_trials
        self.max_continuation_requests = max_continuation_requests

    def run(self, session: PlanningSession, task_hint: str) -> AgentRunResult:
        """Run one search, allowing the baseline when no program improves it.

        The loop ends before its next request once the session says the run
        wants the search to end; a search ended that way may have no program.
        Without a baseline and without a stop, no program is an error.
        """
        result = self._run_loop(
            SYSTEM_PROMPT,
            self.user_prompt(session, task_hint),
            self.tools(session),
            lambda count: self.continuation(session, count),
            run_control=session,
        )
        # A stop sent during the last request is applied here. Limits are not:
        # the agent has already finished on its own.
        if (
            session.best is None
            and session.baseline is None
            and not session.check_for_interrupt(check_limits=False)
        ):
            raise NoAcceptedProgram("the agent produced no accepted program")
        return result

    def continuation(
        self, session: PlanningSession, continuation_requests: int = 0
    ) -> str | None:
        """Return the message that asks for another program, or ``None`` to stop.

        The floor is ``min_trials`` capped at the trial budget, and only
        scored trials count: a refused program does not move the search on.
        """
        budget = session.config.max_trials
        target = min(self.min_trials, budget)
        scored = len(session.trials)
        if scored >= target or continuation_requests >= self.max_continuation_requests:
            return None
        transform = (
            ", or a different target_transform"
            if session.config.allow_target_transform
            else ""
        )
        return (
            f"You have scored {scored} of the {budget} trials this search allows, "
            f"and it should score at least {target} before it stops. Propose "
            "another program with validate_program: a table, time window, "
            f"ratio, or interaction you have not tried{transform}. If the last "
            "idea did not help, change direction rather than make a small variant."
        )

    def tools(self, session: PlanningSession) -> tuple[ToolSpec, ...]:
        """Return describe_schema, profile_column, run_sql, and validate_program.

        ``validate_program`` takes ``queries_json``, a JSON list of
        ``{"name", "sql"}`` objects, and returns the trial's score, noise,
        feature count, coverage per query, and acceptance note. When the search
        allows it, it also takes ``target_transform``.
        """
        validate_properties: dict[str, object] = {
            "queries_json": {
                "type": "string",
                "description": (
                    'JSON list of {"name", "sql"} objects. Every query '
                    f"must return {ROW_COLUMN} from {EVAL_TABLE}."
                ),
            },
            "drop_baseline": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Baseline column names to leave out, when the "
                    "search extends DFS. Drop redundant ones to "
                    "free context rows."
                ),
            },
        }
        if session.config.allow_target_transform:
            validate_properties["target_transform"] = {
                "type": "string",
                "enum": list(TARGET_TRANSFORMS),
                "description": (
                    "How Nori sees the target: auto keeps the default rule (a "
                    "log scale only for extremely skewed targets), identity "
                    "never rescales, log1p always reads log(1 + y) and needs a "
                    "non-negative target. Predictions are returned to the "
                    "label scale before scoring."
                ),
            }

        def validate(
            queries_json: str,
            drop_baseline: Sequence[str] = (),
            target_transform: str = "auto",
        ) -> str:
            """Parse, validate, and report one program; a rejection is text.

            A program that cannot be parsed never reaches the session, so its
            reason is recorded here; otherwise a search whose every program
            was malformed would look like one that never called the tool.
            """
            try:
                program = parse_program(queries_json, drop_baseline, target_transform)
            except (ValueError, KeyError, TypeError) as error:
                session.refusals.append(f"{type(error).__name__}: {error}")
                return f"rejected: {error}"
            return _guard(lambda: _report(session.validate_program(program)))

        return (
            ToolSpec(
                name="describe_schema",
                description="Tables, columns with kinds, keys, and time columns.",
                parameters={"type": "object", "properties": {}, "required": []},
                handler=lambda: _dump(session.describe_schema()),
            ),
            ToolSpec(
                name="profile_column",
                description="Aggregate statistics for one column; never row values.",
                parameters={
                    "type": "object",
                    "properties": {
                        "table": {"type": "string", "description": "Table name."},
                        "column": {"type": "string", "description": "Column name."},
                    },
                    "required": ["table", "column"],
                },
                handler=lambda table, column: _guard(
                    lambda: _dump(session.profile_column(table, column))
                ),
            ),
            ToolSpec(
                name="run_sql",
                description=(
                    f"Run one read-only SELECT over the snapshot and {EVAL_TABLE}. "
                    f"At most {_MAX_EXPLORATION_ROWS} rows come back."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "sql": {"type": "string", "description": "A single SELECT."}
                    },
                    "required": ["sql"],
                },
                handler=lambda sql: _guard(
                    lambda: session.run_sql(sql).to_string(index=False)
                ),
            ),
            ToolSpec(
                name="validate_program",
                description=(
                    "Execute and score a full feature program on the holdout. "
                    "Returns score, noise, feature count, coverage, and whether "
                    "it became the new best."
                ),
                parameters={
                    "type": "object",
                    "properties": validate_properties,
                    "required": ["queries_json"],
                },
                handler=validate,
            ),
        )

    def system_prompt(self) -> str:
        """State the cutoff rules, that SQL sees no labels, and Nori's preferences."""
        return SYSTEM_PROMPT

    def user_prompt(self, session: PlanningSession, task_hint: str) -> str:
        """Ask for the best program and name the baseline score to beat.

        When programs extend the baseline, say so: the DFS columns are already
        in every scored table, so the useful program is the one that adds what
        they lack.
        """
        prompt = "Find the best feature program for this database."
        if session.baseline is not None and session.extends_baseline:
            assert session.baseline_features is not None
            prompt += (
                f"\nEvery program is scored on top of "
                f"{session.baseline_features.shape[1]} deterministic DFS columns, "
                "listed under baseline_features in describe_schema with what "
                "each one computes. Alone they score "
                f"{session.baseline.score:.4f} ({session.baseline.metric}). "
                "Do not recompute them. Start from baseline_never_reads, the "
                "value columns DFS never aggregates, and add what the baseline "
                "lacks: aggregates of those columns, ratios across tables, "
                "conditional counts, trends, and domain quantities. Every "
                "column costs context rows, so pass drop_baseline for DFS "
                "columns that repeat each other, such as counts of the same "
                "rows reached by two paths, or that are mostly missing."
            )
        elif session.baseline is not None:
            prompt += (
                f"\nThe deterministic DFS baseline scores "
                f"{session.baseline.score:.4f} ({session.baseline.metric}); beat it."
            )
        if session.config.min_relative_gain > 0:
            prompt += (
                "\nA candidate must beat both the baseline and current best, when "
                "present, by more than "
                f"{session.config.min_relative_gain:.2%} of each reference's "
                "absolute score and the noise margin. Confirmation must also "
                "beat the baseline by this margin."
            )
        if session.config.allow_target_transform:
            prompt += (
                "\nvalidate_program also takes target_transform, which changes "
                "how Nori reads the label: auto (the default rule), identity, or "
                "log1p for a non-negative, heavy-tailed target. A program's "
                "score includes its transform, so compare transforms the same "
                "way you compare columns."
            )
        if task_hint:
            prompt += f"\nTask context: {task_hint}"
        return prompt

    @abstractmethod
    def _run_loop(
        self,
        system: str,
        prompt: str,
        tools: tuple[ToolSpec, ...],
        continuation: Callable[[int], str | None] | None = None,
        *,
        run_control: SearchRunControl | None = None,
    ) -> AgentRunResult:
        """Run the SDK loop until the model stops calling tools or the budget ends.

        Every tool call goes through the matching ``ToolSpec.handler`` and its
        text result goes back to the model. When the model ends its turn,
        ``continuation`` is asked for a message; one is sent as the next user
        turn and the loop resumes, ``None`` ends the loop. Its argument is the
        number of continuation requests already sent in this run.
        ``run_control``, when given, hears about every model response.
        """
        ...


def get_stop_flag_and_steering_text(
    run_control: SearchRunControl | None,
) -> tuple[bool, str | None]:
    """Return whether to stop before the next request, and steering to send in it.

    Every request boundary of a ``_run_loop`` goes through here, so the order
    lives in one place: interrupts first, steering last, because delivering
    steering reports it applied. A loop calls this as the last step before
    building the request and sends the text in that request.
    """
    if run_control is None:
        return False, None
    if run_control.check_for_interrupt():
        return True, None
    return False, run_control.deliver_steering_text()


def is_steering_waiting(run_control: SearchRunControl | None) -> bool:
    """Return whether a finished model turn should still make one more request.

    Held steering keeps the search going, to be delivered at the next request.
    There is no interrupt check here: the model has finished, so a stop or a
    limit arriving now does not count as ending the search early.
    """
    return run_control is not None and run_control.has_held_steering()


def parse_program(
    queries_json: str,
    drop_baseline: Sequence[str] = (),
    target_transform: str = "auto",
) -> FeatureProgram:
    """Turn the ``validate_program`` arguments into a ``FeatureProgram``."""
    payload = json.loads(queries_json)
    if not isinstance(payload, list):
        raise ValueError("queries_json must be a JSON list")
    if isinstance(drop_baseline, str):
        raise ValueError("drop_baseline must be a list of column names")
    return FeatureProgram(
        tuple(
            FeatureQuery(name=str(item["name"]), sql=str(item["sql"]))
            for item in payload
        ),
        drop_baseline=tuple(str(name) for name in drop_baseline),
        target_transform=str(target_transform),
    )


def _report(trial) -> str:
    return _dump(
        {
            "trial": trial.index,
            "metric": trial.report.metric,
            "score": trial.report.score,
            "noise": trial.report.noise,
            "features": trial.feature_count,
            "coverage": trial.coverage,
            "accepted": trial.accepted,
            "note": trial.note,
        }
    )


def _guard(action: Callable[[], str]) -> str:
    try:
        return action()
    except (ProgramRejected, KeyError, ValueError, TimeoutError) as error:
        return f"rejected: {error}"


def _dump(value: object) -> str:
    return json.dumps(value, default=str)


__all__ = ["SYSTEM_PROMPT", "ProgramAgentBase", "ToolSpec", "parse_program"]
