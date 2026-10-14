"""Contracts shared by the agent, its session, the scorer, and the backend."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np
import pandas as pd

from nori_rel.run.models import TrialOutcome

if TYPE_CHECKING:
    from .session import PlanningSession
    from .usage import AgentRunResult

TaskKind = Literal["regression", "binary"]

ROW_COLUMN = "row_id"
CUTOFF_ALIAS = "cutoff"
EVAL_TABLE = "eval_table"
#: How the scorer may rescale the regression target before Nori reads it.
#: ``auto`` keeps the board's own rule (a log scale only for extremely skewed
#: targets), ``identity`` never rescales, and ``log1p`` always reads
#: log(1 + y), for non-negative targets. Predictions return to the label scale
#: before they are scored.
TARGET_TRANSFORMS = ("auto", "identity", "log1p")


@dataclass(frozen=True)
class FeatureQuery:
    """One named SQL block anchored on ``eval_table``."""

    name: str
    sql: str

    def __post_init__(self) -> None:
        """Require an identifier-like name and a non-empty statement."""
        if not self.name.isidentifier():
            raise ValueError(f"query name {self.name!r} must be an identifier")
        if not self.sql.strip():
            raise ValueError(f"query {self.name!r} has an empty statement")


@dataclass(frozen=True)
class FeatureProgram:
    """An ordered set of feature queries; what the agent proposes and we replay.

    ``drop_baseline`` names baseline columns the program leaves out when it
    extends a baseline, so a redundant DFS column stops costing context rows.
    ``target_transform`` is one of ``TARGET_TRANSFORMS``.
    """

    queries: tuple[FeatureQuery, ...]
    drop_baseline: tuple[str, ...] = ()
    target_transform: str = "auto"

    def __post_init__(self) -> None:
        """Require at least one query, unique names, and a known transform."""
        if not self.queries:
            raise ValueError("a feature program needs at least one query")
        names = [query.name for query in self.queries]
        if len(names) != len(set(names)):
            raise ValueError("feature query names must be unique")
        if len(self.drop_baseline) != len(set(self.drop_baseline)):
            raise ValueError("drop_baseline must not repeat a column")
        if self.target_transform not in TARGET_TRANSFORMS:
            raise ValueError(
                f"target_transform must be one of {TARGET_TRANSFORMS}, "
                f"not {self.target_transform!r}"
            )

    @property
    def digest(self) -> str:
        """Return a stable identity for caches and logs."""
        payload: list[object] = [(query.name, query.sql) for query in self.queries]
        if self.drop_baseline:
            payload.append(sorted(self.drop_baseline))
        if self.target_transform != "auto":
            payload.append(self.target_transform)
        encoded = json.dumps(payload, separators=(",", ":")).encode()
        return hashlib.blake2s(encoded, digest_size=16).hexdigest()


@dataclass(frozen=True)
class AgenticFeatureConfig:
    """Budgets for one planning episode."""

    max_queries: int = 12
    max_features: int = 64
    max_trials: int = 24
    holdout_fraction: float = 0.25
    confirmation_fraction: float = 0.15
    leak_check_anchors: int = 8
    noise_margin: float = 2.0
    sql_timeout_seconds: float = 60.0
    #: Off refuses prose the way this backend did before it had a text path,
    #: so the path can be ablated within one implementation.
    allow_text: bool = True
    #: On scores every program as the baseline's columns plus its own, so DFS
    #: is the starting point the agent adds to rather than a bar it has to
    #: clear by rebuilding the same aggregates in SQL.
    extend_baseline: bool = False
    #: Rolling-cutoff validation windows, each ``holdout_fraction`` of the
    #: anchors, laid back to back before the confirmation slice. Each window is
    #: scored with only the anchors before it as context, and a trial's score
    #: is their mean. One window is the single holdout this session always had.
    n_folds: int = 1
    #: On lets a program choose its ``target_transform``; the scorer must then
    #: accept a ``target_transform`` keyword. Off keeps every program on
    #: ``auto``, as before.
    allow_target_transform: bool = False
    #: On cuts every validation window and the confirmation slice at a change
    #: of cutoff, so no scored row shares a cutoff with a context row. Off cuts
    #: at fixed row counts, as before. Rows that share a cutoff share a label
    #: window, so letting them straddle a boundary scores same-period
    #: interpolation rather than the forecast the test split asks for.
    align_to_cutoffs: bool = False
    #: Additional gain required over each reference score, combined with the
    #: noise margin by taking the larger threshold. Zero preserves the
    #: historical search protocol; a positive value also gates trials against
    #: the baseline and refuses relative comparisons to a zero reference.
    min_relative_gain: float = 0.0
    #: Minimum distinct context cutoffs in every aligned validation fold.
    #: Unaligned, row-based splits do not use this floor.
    min_context_cutoffs: int = 1

    def __post_init__(self) -> None:
        """Validate the episode budget."""
        if min(self.max_queries, self.max_features, self.max_trials) < 1:
            raise ValueError("query, feature, and trial budgets must be positive")
        if self.n_folds < 1:
            raise ValueError("n_folds must be at least 1")
        validated = self.n_folds * self.holdout_fraction + self.confirmation_fraction
        if not 0.0 < validated < 1.0:
            raise ValueError("holdout and confirmation fractions must leave context")
        if self.leak_check_anchors < 0 or self.noise_margin < 0.0:
            raise ValueError("leak_check_anchors and noise_margin must be non-negative")
        if self.sql_timeout_seconds <= 0.0:
            raise ValueError("sql_timeout_seconds must be positive")
        if not math.isfinite(self.min_relative_gain) or not (
            0.0 <= self.min_relative_gain < 1.0
        ):
            raise ValueError("min_relative_gain must be finite and in [0, 1)")
        if (
            not isinstance(self.min_context_cutoffs, int)
            or isinstance(self.min_context_cutoffs, bool)
            or self.min_context_cutoffs < 1
        ):
            raise ValueError("min_context_cutoffs must be a positive integer")


@dataclass(frozen=True)
class QueryDiagnostics:
    """Execution facts about one query the agent can act on."""

    columns: tuple[str, ...]
    coverage: float
    missing_fraction: float
    seconds: float


@dataclass(frozen=True)
class ExecutionResult:
    """Aligned features for the supplied anchors plus per-query diagnostics."""

    frame: pd.DataFrame
    categorical_columns: tuple[str, ...]
    text_columns: tuple[str, ...]
    diagnostics: Mapping[str, QueryDiagnostics]


@dataclass(frozen=True)
class ScoreReport:
    """One scorer run: a higher-is-better score and its seed-to-seed noise."""

    metric: str
    score: float
    noise: float
    predictions: np.ndarray = field(repr=False, compare=False)


@dataclass(frozen=True)
class Trial:
    """One validated program and what the agent was told about it.

    ``outcome`` says what the search decided; ``note`` is the agent-facing text
    behind it and is never reported to a run.
    """

    index: int
    program: FeatureProgram
    report: ScoreReport
    feature_count: int
    coverage: dict[str, float]
    outcome: TrialOutcome
    note: str

    @property
    def accepted(self) -> bool:
        """Return whether the search kept this trial as its best so far."""
        return self.outcome in (TrialOutcome.NEW_BEST, TrialOutcome.BEST_BELOW_BASELINE)


class NoAcceptedProgram(RuntimeError):
    """The agent finished a search that it was not asked to stop with no program."""


class ProgramScorer(Protocol):
    """Fit on context features, predict a holdout, and report the task metric."""

    def __call__(
        self,
        context_features: pd.DataFrame,
        context_target: pd.Series,
        holdout_features: pd.DataFrame,
        holdout_target: pd.Series,
        *,
        categorical_columns: Sequence[str],
        text_columns: Sequence[str] = (),
    ) -> ScoreReport:
        """Return the holdout score; larger must mean better."""
        ...


class SearchRunControl(Protocol):
    """The run a search belongs to, as an agent's request loop sees it.

    ``PlanningSession`` implements it. A loop calls ``report_model_response``
    after every model response, with this run's cost so far in dollars, or
    ``None`` when the agent has no prices. Per-run reporting replaces any
    callback on the agent object, which several searches may share.
    """

    def report_model_response(self, cost_usd: float | None) -> None:
        """Report one model response and the run's cost so far."""
        ...

    def check_for_interrupt(self) -> bool:
        """Apply the run's commands; return whether the search must end now.

        A loop asks before every request. A cancel raises ``RunCancelled``,
        a ``BaseException``; a pause waits here until it ends.
        """
        ...

    def has_held_steering(self) -> bool:
        """Return whether steering waits for the next request, without taking it.

        A loop whose model has finished asks this instead of
        ``check_for_interrupt``: held steering keeps the search going for one
        more request, and a stop or limit arriving once the model has finished
        does not count as ending the search early.
        """
        ...

    def deliver_steering_text(self) -> str | None:
        """Return the held steering as one user message, or ``None``.

        Delivering reports each steer applied, so a loop asks only as the last
        step before the request that carries the text, and sends it there.
        """
        ...


class ProgramAgent(Protocol):
    """Drive a session to a best program; the session owns validity and scores."""

    def run(self, session: PlanningSession, task_hint: str) -> AgentRunResult | None:
        """Call the session's public methods until satisfied or out of budget."""
        ...


__all__ = [
    "CUTOFF_ALIAS",
    "EVAL_TABLE",
    "ROW_COLUMN",
    "TARGET_TRANSFORMS",
    "AgenticFeatureConfig",
    "ExecutionResult",
    "FeatureProgram",
    "FeatureQuery",
    "NoAcceptedProgram",
    "ProgramAgent",
    "ProgramScorer",
    "QueryDiagnostics",
    "ScoreReport",
    "SearchRunControl",
    "TaskKind",
    "Trial",
]
