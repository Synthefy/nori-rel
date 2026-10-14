"""One planning episode: SQL programs in, a confirmed best program out.

The session owns the DuckDB copy of the snapshot, the anchor splits, both
leak checks, and acceptance. An agent only calls the public methods.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nori_rel.run.models import TrialOutcome

from ..dfs.noridfs.contracts import (
    AnchorDefinition,
    ColumnKind,
    DatabaseView,
    FeaturePlan,
)
from .baseline import catalog, unread_columns
from .contracts import (
    CUTOFF_ALIAS,
    EVAL_TABLE,
    ROW_COLUMN,
    AgenticFeatureConfig,
    ExecutionResult,
    FeatureProgram,
    ProgramScorer,
    ScoreReport,
    Trial,
)
from .executor import ProgramExecutor, is_prose
from .guard import (
    LeakFinding,
    blocked_findings,
    point_in_time_findings,
    static_findings,
)

_EXPLORATION_ROWS = 50
_TOP_VALUES = 10
_QUANTILES = (0.01, 0.25, 0.5, 0.75, 0.99)


class ProgramRejected(ValueError):
    """A program failed validation; the message tells the agent why."""

    @classmethod
    def from_findings(cls, findings: tuple[LeakFinding, ...]) -> ProgramRejected:
        """Flatten guard findings into one agent-readable message."""
        return cls("; ".join(f"{f.query}: {f.reason}" for f in findings))


logger = logging.getLogger(__name__)


def _call_reporting_callback(callback: Callable[..., None] | None, value) -> None:
    """Call a progress callback with ``value``, logging instead of raising."""
    if callback is None:
        return
    try:
        callback(value)
    except Exception:
        logger.exception("a search progress callback failed")


@dataclass
class PlanningSession:
    """Everything an agent may do while searching for a feature program.

    Snapshot tables load into an in-memory DuckDB with external access off.
    ``eval_table`` exposes ``row_id``, the entity key, and ``cutoff`` for the
    train anchors, which split by cutoff time into context, holdout, and
    confirmation slices. ``train_target`` is read only by the scorer.
    """

    database: DatabaseView
    anchors: AnchorDefinition
    config: AgenticFeatureConfig
    train_anchors: pd.DataFrame
    train_target: pd.Series
    scorer: ProgramScorer
    baseline_features: pd.DataFrame | None = None
    baseline_categorical: tuple[str, ...] = ()
    #: The plan behind ``baseline_features`` when it came from NoriDFS. It is
    #: what lets the schema say what each baseline column computes.
    baseline_plan: FeaturePlan | None = None
    #: Called with every scored trial, after the session records it.
    on_trial_scored: Callable[[Trial], None] | None = None
    #: Called after every model response with the run's cost so far in
    #: dollars, or ``None`` without prices. See ``SearchRunControl``.
    on_model_response: Callable[[float | None], None] | None = None
    #: Applies the run's queued commands, raising on cancel and waiting while
    #: paused, and returns whether the search must end. It runs before each
    #: model call, trial and scoring call, the baseline's included.
    apply_run_commands: Callable[..., bool] | None = None
    #: Hands over the steering the caller sent and the run held: returns
    #: its texts, clears them, and reports each steer applied.
    deliver_run_steering_texts: Callable[[], list[str]] | None = None
    #: Says whether the run holds steering, without taking it.
    has_run_steering: Callable[[], bool] | None = None
    #: Whether the run asked the search to end; it stays ended once it has.
    interrupted: bool = field(init=False, repr=False, default=False)
    #: The latest scoring call and the best program's SQL, in seconds; they
    #: estimate what ``confirm()`` costs. The latest call, not the slowest,
    #: so a first call that loads the model does not inflate the estimate.
    latest_scoring_seconds: float = field(init=False, repr=False, default=0.0)
    best_program_sql_seconds: float = field(init=False, repr=False, default=0.0)
    trials: list[Trial] = field(init=False, repr=False, default_factory=list)
    best: Trial | None = field(init=False, repr=False, default=None)
    #: Why programs were turned away, in order. The agent is told each reason
    #: and nothing else records it, so a search that scores nothing otherwise
    #: reports only "no accepted program" -- which cannot distinguish a model
    #: that never called the tool from one whose every program hit the guard.
    refusals: list[str] = field(init=False, repr=False, default_factory=list)
    baseline: ScoreReport | None = field(init=False, repr=False, default=None)
    _executor: ProgramExecutor = field(init=False, repr=False)
    _context: np.ndarray = field(init=False, repr=False)
    _holdout: np.ndarray = field(init=False, repr=False)
    _confirmation: np.ndarray = field(init=False, repr=False)
    #: (context, validation) pairs, oldest window first; the last pair is
    #: (``_context``, ``_holdout``).
    _folds: list[tuple[np.ndarray, np.ndarray]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Load the snapshot, split the anchors, and score the DFS baseline."""
        if self.anchors.cutoff_column is None:
            raise ValueError("agentic planning requires anchors with a cutoff column")
        expected = [self.anchors.entity_column, self.anchors.cutoff_column]
        if list(self.train_anchors.columns) != expected:
            raise ValueError(f"train_anchors must hold exactly {expected}")
        if len(self.train_target) != len(self.train_anchors):
            raise ValueError("train_target must align with train_anchors")
        self.train_anchors = self.train_anchors.reset_index(drop=True)
        self.train_target = self.train_target.reset_index(drop=True)
        if self.config.align_to_cutoffs:
            self._folds, self._confirmation = self._aligned_folds()
            self._context, self._holdout = self._folds[-1]
        else:
            self._context, self._holdout, self._confirmation = self._split()
            self._folds = self._rolling_folds()
        self._executor = ProgramExecutor(
            self.database,
            entity_column=self.anchors.entity_column,
            timeout_seconds=self.config.sql_timeout_seconds,
            allow_text=self.config.allow_text,
        )
        try:
            self._bind_anchors_and_score_baseline()
        except BaseException:
            self.close()
            raise

    def _bind_anchors_and_score_baseline(self) -> None:
        """Load the anchors into the executor and score the baseline on the folds."""
        self._executor.bind_anchors(self.train_anchors)
        if self.config.extend_baseline and self.baseline_features is None:
            raise ValueError("extend_baseline needs baseline features to extend")
        if self.baseline_features is not None:
            if len(self.baseline_features) != len(self.train_anchors):
                raise ValueError("baseline_features must align with train_anchors")
            self.baseline_features = self.baseline_features.reset_index(drop=True)
            self.baseline = self._score_folds(
                self.baseline_features, self.baseline_categorical
            )

    @property
    def extends_baseline(self) -> bool:
        """Return whether programs are scored on top of the baseline columns."""
        return self.config.extend_baseline and self.baseline_features is not None

    def close(self) -> None:
        """Release the SQL connection."""
        self._executor.close()

    def describe_schema(self) -> dict[str, object]:
        """Return tables, columns with kinds, keys, and time columns.

        When programs extend the baseline, ``baseline_features`` lists the
        columns every program already gets, each described when the plan is
        known, and ``baseline_never_reads`` lists per table the value columns
        none of them aggregate. The agent adds what the baseline lacks
        instead of recomputing it.
        """
        schema: dict[str, object] = {
            "entity_table": self.anchors.entity_table,
            "entity_column": self.anchors.entity_column,
            "eval_table": EVAL_TABLE,
            "eval_columns": [ROW_COLUMN, self.anchors.entity_column, CUTOFF_ALIAS],
            "tables": {
                name: {
                    "rows": len(frame),
                    "primary_key": self.database.primary_keys.get(name),
                    "time_column": self.database.time_columns.get(name),
                    "columns": {
                        column: _column_kind(frame[column]) for column in frame
                    },
                }
                for name, frame in self.database.tables.items()
            },
            "foreign_keys": [r.identity for r in self.database.foreign_keys],
        }
        if self.extends_baseline:
            assert self.baseline_features is not None
            frame = self.baseline_features
            schema["baseline_features"] = catalog(
                self.baseline_plan,
                frame,
                {column: _column_kind(frame[column]) for column in frame},
            )
            if self.baseline_plan is not None:
                schema["baseline_never_reads"] = unread_columns(
                    self.baseline_plan, self.database
                )
        return schema

    def profile_column(self, table: str, column: str) -> dict[str, object]:
        """Return aggregate statistics for one column, never row values."""
        if table not in self.database.tables:
            raise KeyError(f"unknown table {table!r}")
        frame = self.database.tables[table]
        if column not in frame:
            raise KeyError(f"unknown column {table}.{column}")
        series = frame[column]
        kind = _column_kind(series, self.config.allow_text)
        present = series.dropna()
        profile: dict[str, object] = {
            "kind": kind,
            "null_fraction": float(series.isna().mean()) if len(series) else 0.0,
            "unique_count": int(present.nunique()),
        }
        if kind == "numeric" and len(present):
            levels = present.astype(float).quantile(list(_QUANTILES))
            profile["quantiles"] = {str(q): float(v) for q, v in levels.items()}
        elif kind == "categorical":
            counts = present.value_counts().head(_TOP_VALUES)
            profile["top_values"] = {str(k): int(v) for k, v in counts.items()}
        elif kind == "text" and len(present):
            # Enough to judge whether the prose is worth 16 columns of width,
            # without pasting a paragraph per column into the schema.
            profile["mean_length"] = float(present.astype(str).str.len().mean())
            profile["sample"] = str(present.iloc[0])[:200]
        elif kind == "datetime" and len(present):
            profile["earliest"] = str(present.min())
            profile["latest"] = str(present.max())
        return profile

    def run_sql(self, sql: str) -> pd.DataFrame:
        """Run one read-only SELECT over the snapshot and eval_table, row-capped."""
        findings = blocked_findings("exploration", sql)
        if findings:
            raise ProgramRejected.from_findings(findings)
        return self._executor.sql(sql, limit=_EXPLORATION_ROWS)

    def check_for_interrupt(self, *, check_limits: bool = True) -> bool:
        """Apply the run's commands; return whether the search must end now.

        Agents call it before each model request, where the run's runtime and
        spend limits are checked too. A cancel raises; a pause waits for a
        resume, a stop or a cancel. Once the search must end, it stays ended.
        """
        if self.apply_run_commands is not None and self.apply_run_commands(
            check_limits=check_limits
        ):
            self.interrupted = True
        return self.interrupted

    def has_held_steering(self) -> bool:
        """Apply the run's commands; return whether steering waits for a request.

        A loop whose model has finished asks this, and held steering keeps the
        search going for one more request. It checks no limit and never ends
        the search: the model has finished, so a stop or limit arriving now
        finds the search over, as when an agent finishes on its own. A stop
        applied here returns ``False``; the search's end rejects its steer.
        """
        if self.has_run_steering is None:
            return False
        if self.apply_run_commands is not None and self.apply_run_commands(
            check_limits=False
        ):
            return False
        return self.has_run_steering()

    def deliver_steering_text(self) -> str | None:
        """Return the caller's held steering as one message, or ``None``.

        Delivering reports each steer applied, so an agent asks only as the
        last step before the model request that sends the text as a user turn.
        """
        if self.deliver_run_steering_texts is None:
            return None
        texts = self.deliver_run_steering_texts()
        if not texts:
            return None
        return "\n\n".join(f"Steering from the user: {text}" for text in texts)

    def estimated_confirm_seconds(self) -> float:
        """Estimate ``confirm()``: the best program's SQL and two scoring calls."""
        return self.best_program_sql_seconds + 2 * self.latest_scoring_seconds

    def validate_program(self, program: FeatureProgram) -> Trial:
        """Record why a program was turned away, then re-raise for the agent.

        A scored trial goes to ``on_trial_scored`` only after it is recorded and
        outside the refusal handling, so a failing callback is logged and never
        turns an accepted trial into a refusal. Once the search must end,
        programs are refused before any work, and that refusal is not
        recorded: the program was never judged.
        """
        # Commands only: a limit reached by the response that proposed this
        # program is checked before the next request, so a paid-for
        # proposal is still scored.
        if self.check_for_interrupt(check_limits=False):
            raise ProgramRejected(
                "the search was stopped; propose no more programs and finish"
            )
        try:
            trial = self._validate_program(program)
        except Exception as error:
            self.refusals.append(f"{type(error).__name__}: {error}")
            raise
        _call_reporting_callback(self.on_trial_scored, trial)
        return trial

    def report_model_response(self, cost_usd: float | None) -> None:
        """Pass one model response's running cost to ``on_model_response``.

        A failing callback is logged and never raised: reporting progress must
        not end a search.
        """
        _call_reporting_callback(self.on_model_response, cost_usd)

    def _validate_program(self, program: FeatureProgram) -> Trial:
        """Execute, guard, and score one program; accept it when it is the best.

        Execution runs every query with a preflight row count, an interrupt
        timeout, and alignment on ``row_id``; a query that returns more than
        one row per anchor is rejected, a missing anchor becomes a missing
        value. A regex layer rejects ``>=``, ``>``, or ``+ INTERVAL`` against
        the cutoff and temporal tables read without a ``< cutoff`` filter.
        The scorer fits on the context slice and predicts the holdout. The
        program becomes the best only when it beats the incumbent by
        ``noise_margin`` times the noise and, on ``leak_check_anchors``
        sampled anchors, yields identical features against a snapshot
        truncated before each anchor's cutoff.
        """
        if len(self.trials) >= self.config.max_trials:
            raise ProgramRejected("trial budget is exhausted; stop and keep the best")
        if len(program.queries) > self.config.max_queries:
            raise ProgramRejected(
                f"programs may hold {self.config.max_queries} queries"
            )
        if program.target_transform != "auto":
            if not self.config.allow_target_transform:
                raise ProgramRejected(
                    "this search does not take target_transform; leave it at auto"
                )
            if program.target_transform == "log1p" and self.train_target.min() < 0:
                raise ProgramRejected(
                    "log1p needs a non-negative target, and this one goes below 0"
                )
        findings = tuple(
            finding
            for query in program.queries
            for finding in static_findings(query, self.database)
        )
        if findings:
            raise ProgramRejected.from_findings(findings)
        result = self._executor.run(program, self.train_anchors)
        width = result.frame.shape[1]
        if width == 0:
            raise ProgramRejected("program yields no feature columns")
        if width > self.config.max_features:
            raise ProgramRejected(
                f"program yields {width} features; the budget is "
                f"{self.config.max_features}"
            )
        features, categorical = self._scored(result, program.drop_baseline)
        report = self._score_folds(
            features, categorical, result.text_columns, program.target_transform
        )
        outcome, note = self._consider(program, result, report)
        trial = Trial(
            index=len(self.trials) + 1,
            program=program,
            report=report,
            feature_count=features.shape[1],
            coverage={
                name: diagnostics.coverage
                for name, diagnostics in result.diagnostics.items()
            },
            outcome=outcome,
            note=note,
        )
        self.trials.append(trial)
        if trial.accepted:
            self.best = trial
            self.best_program_sql_seconds = sum(
                diagnostics.seconds for diagnostics in result.diagnostics.values()
            )
        return trial

    def best_program(self) -> FeatureProgram | None:
        """Return the accepted program with the highest holdout score."""
        return None if self.best is None else self.best.program

    def confirm(self) -> bool:
        """Re-score the best against the baseline on the confirmation slice."""
        if self.best is None:
            return False
        result = self._executor.run(self.best.program, self.train_anchors)
        features, categorical = self._scored(result, self.best.program.drop_baseline)
        report = self._score(
            features,
            categorical,
            self._confirmation,
            result.text_columns,
            target_transform=self.best.program.target_transform,
        )
        if self.baseline is None:
            return True
        baseline = self._score(
            self.baseline_features, self.baseline_categorical, self._confirmation
        )
        margin = self._minimum_score_gain(report, baseline)
        return report.score - baseline.score > margin

    def _scored(
        self, result: ExecutionResult, drop: tuple[str, ...]
    ) -> tuple[pd.DataFrame, tuple[str, ...]]:
        """Return the columns a program is scored on and which are categorical.

        Extending puts the kept baseline columns first and refuses a program
        column that reuses a baseline name, since one of the two would be
        dropped, as well as a drop that names no baseline column.
        """
        if not self.extends_baseline:
            if drop:
                raise ProgramRejected("drop_baseline needs a search that extends DFS")
            return result.frame, result.categorical_columns
        assert self.baseline_features is not None
        unknown = sorted(set(drop) - set(self.baseline_features))
        if unknown:
            raise ProgramRejected(f"drop_baseline names unknown columns {unknown}")
        kept = self.baseline_features.drop(columns=list(drop))
        clashes = sorted(set(result.frame.columns) & set(kept))
        if clashes:
            raise ProgramRejected(
                f"columns {clashes} already exist in the baseline; rename them"
            )
        features = pd.concat([kept, result.frame.reset_index(drop=True)], axis=1)
        categorical = tuple(
            column for column in self.baseline_categorical if column in kept
        )
        return features, categorical + result.categorical_columns

    def _consider(
        self, program: FeatureProgram, result: ExecutionResult, report: ScoreReport
    ) -> tuple[TrialOutcome, str]:
        """Return the outcome of a scored program and the note the agent reads.

        The significance gate runs before the point-in-time check, so a program
        that is not a significant gain is never checked.
        """
        if self.best is not None:
            margin = self._minimum_score_gain(report, self.best.report)
            if not report.score - self.best.report.score > margin:
                return TrialOutcome.NOT_SIGNIFICANT_GAIN, (
                    f"not a significant gain over {self.best.report.score:.4f} "
                    f"(margin {margin:.4f})"
                )
        if self.baseline is not None and self.config.min_relative_gain > 0.0:
            margin = self._minimum_score_gain(report, self.baseline)
            if not report.score - self.baseline.score > margin:
                return TrialOutcome.NOT_SIGNIFICANT_GAIN, (
                    f"not a significant gain over the baseline "
                    f"{self.baseline.score:.4f} (margin {margin:.4f})"
                )
        findings = point_in_time_findings(
            program,
            self.database,
            self.train_anchors,
            result.frame,
            entity_column=self.anchors.entity_column,
            sample=self._leak_sample(),
            timeout_seconds=self.config.sql_timeout_seconds,
            allow_text=self.config.allow_text,
        )
        if findings:
            return (
                TrialOutcome.FAILED_POINT_IN_TIME_CHECK,
                f"rejected for leakage: {ProgramRejected.from_findings(findings)}",
            )
        if self.baseline is not None and report.score <= self.baseline.score:
            return (
                TrialOutcome.BEST_BELOW_BASELINE,
                f"accepted but below the baseline {self.baseline.score:.4f}",
            )
        return TrialOutcome.NEW_BEST, "accepted as the new best"

    def _minimum_score_gain(self, report: ScoreReport, reference: ScoreReport) -> float:
        """Require both the noise margin and the configured relative gain.

        Relative gain has no scale at a zero reference score, so a positive
        relative requirement cannot be satisfied there.
        """
        if self.config.min_relative_gain > 0.0 and reference.score == 0.0:
            return float("inf")
        return max(
            self.config.noise_margin * max(report.noise, reference.noise),
            abs(reference.score) * self.config.min_relative_gain,
        )

    def _score(
        self,
        features: pd.DataFrame | None,
        categorical: tuple[str, ...],
        holdout: np.ndarray,
        text: tuple[str, ...] = (),
        context: np.ndarray | None = None,
        target_transform: str = "auto",
    ) -> ScoreReport:
        """Fit on ``context`` (the last fold's by default) and score ``holdout``.

        ``target_transform`` reaches the scorer only when it is not ``auto``,
        so a scorer that has no transforms keeps its existing signature.
        """
        if features is None:
            raise ValueError("no features to score")
        context = self._context if context is None else context
        transform = (
            {} if target_transform == "auto" else {"target_transform": target_transform}
        )
        # Commands apply before each scoring call; a stop does not end the
        # scoring already under way, such as a confirmation chosen after it.
        self.check_for_interrupt(check_limits=False)
        started_at = time.perf_counter()
        report = self.scorer(
            features.iloc[context],
            self.train_target.iloc[context],
            features.iloc[holdout],
            self.train_target.iloc[holdout],
            categorical_columns=categorical,
            text_columns=text,
            **transform,
        )
        self.latest_scoring_seconds = time.perf_counter() - started_at
        return report

    def _score_folds(
        self,
        features: pd.DataFrame | None,
        categorical: tuple[str, ...],
        text: tuple[str, ...] = (),
        target_transform: str = "auto",
    ) -> ScoreReport:
        """Score every rolling window and report the mean.

        With one window this is exactly the single-holdout score. With more,
        the noise is the mean of the windows' own seed spread, so the
        acceptance margin means what it meant with one window; the spread
        between windows is mostly shared by every program and is not noise in
        a comparison between two of them.
        """
        reports = [
            self._score(
                features,
                categorical,
                holdout,
                text,
                context=context,
                target_transform=target_transform,
            )
            for context, holdout in self._folds
        ]
        if len(reports) == 1:
            return reports[0]
        return ScoreReport(
            metric=reports[0].metric,
            score=float(np.mean([report.score for report in reports])),
            noise=float(np.mean([report.noise for report in reports])),
            predictions=reports[-1].predictions,
        )

    def _rolling_folds(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Lay ``n_folds`` holdout-sized windows back to back before confirmation.

        The newest window is the session's holdout. Each earlier window sits
        directly before the next, and every window's context is all anchors
        before it, so no window is scored with a later row in its context.
        """
        order = np.concatenate([self._context, self._holdout])
        size = len(self._holdout)
        end = len(self._context)
        folds = []
        for back in range(self.config.n_folds - 1, -1, -1):
            start = end - back * size
            if start < 1:
                raise ValueError(
                    f"{self.config.n_folds} folds of {size} anchors leave the "
                    "oldest window no context"
                )
            folds.append((order[:start], order[start : start + size]))
        return folds

    def _aligned_folds(
        self,
    ) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
        """Cut the windows and the confirmation slice at changes of cutoff.

        Walking back from the newest cutoff, confirmation takes the fewest
        whole cutoff groups that hold ``confirmation_fraction`` of the anchors,
        then each window the fewest whole groups that hold
        ``holdout_fraction``. Every window reserves ``min_context_cutoffs``
        earlier groups as context, so a task with few cutoffs gets fewer windows than
        ``n_folds``; the session's ``_folds`` says how many it got.
        """
        count = len(self.train_anchors)
        cutoffs = self.train_anchors[self.anchors.cutoff_column].to_numpy()
        order = np.argsort(cutoffs, kind="stable")
        ordered = cutoffs[order]
        starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
        bounds = np.r_[starts, count]
        groups = len(starts)

        context_floor = self.config.min_context_cutoffs
        if groups < context_floor + 2:
            raise ValueError(
                f"{groups} cutoff(s) leave no room for {context_floor} context "
                "cutoffs, a holdout window and a confirmation slice"
            )

        def first_group(end: int, rows: int, floor: int) -> int:
            """Return the earliest group a slice ending before ``end`` needs.

            The slice keeps at least one group and never starts before
            ``floor``, which leaves room for what has to come before it.
            """
            start = end - 1
            while start > floor and bounds[end] - bounds[start] < rows:
                start -= 1
            return start

        confirmation_start = first_group(
            groups,
            max(1, round(count * self.config.confirmation_fraction)),
            context_floor + 1,
        )
        window_rows = max(1, round(count * self.config.holdout_fraction))
        folds: list[tuple[np.ndarray, np.ndarray]] = []
        end = confirmation_start
        for _ in range(self.config.n_folds):
            if end < context_floor + 1:
                break
            start = first_group(end, window_rows, context_floor)
            folds.append((order[: bounds[start]], order[bounds[start] : bounds[end]]))
            end = start
        folds.reverse()
        return folds, order[bounds[confirmation_start] :]

    def _split(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        count = len(self.train_anchors)
        if count < 3:
            raise ValueError("planning requires at least three train anchors")
        cutoffs = self.train_anchors[self.anchors.cutoff_column].to_numpy()
        order = np.argsort(cutoffs, kind="stable")
        confirmation = max(1, round(count * self.config.confirmation_fraction))
        holdout = max(1, round(count * self.config.holdout_fraction))
        if count - holdout - confirmation < 1:
            raise ValueError("splits leave no context anchors")
        context = count - holdout - confirmation
        return (
            order[:context],
            order[context : context + holdout],
            order[context + holdout :],
        )

    def _leak_sample(self) -> np.ndarray:
        """Draw check anchors from the earliest cutoffs.

        A leak is only observable at an anchor that has rows after its cutoff,
        and the earliest anchors have the most. Sampling the holdout instead
        would silently miss leaks, because those anchors sit at the end of the
        history where there is little future left to read.
        """
        size = min(self.config.leak_check_anchors, len(self._context))
        if size == 0:
            return np.array([], dtype=int)
        rng = np.random.default_rng(0)
        return np.sort(rng.choice(self._context, size, replace=False))


def _column_kind(series: pd.Series, allow_text: bool = True) -> ColumnKind | str | None:
    if pd.api.types.is_bool_dtype(series):
        return "categorical"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_string_dtype(series) or isinstance(
        series.dtype, pd.CategoricalDtype
    ):
        present = series.dropna()
        if allow_text and is_prose(present):
            return "text"
        return "categorical"
    return None


__all__ = ["PlanningSession", "ProgramRejected"]
