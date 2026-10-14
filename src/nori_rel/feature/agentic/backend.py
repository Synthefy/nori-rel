"""Snapshot-to-matrix interface around an agent-written SQL feature program."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from nori_rel.run import FallbackReason, RunContext, Stage

from .._contract import CUTOFF_COLUMN, FeatureBackend, FeatureMatrices, RelationalData
from ..dfs.noridfs import NoriDFS
from ..dfs.noridfs.contracts import AnchorDefinition, DatabaseView
from .contracts import (
    AgenticFeatureConfig,
    FeatureProgram,
    NoAcceptedProgram,
    ProgramAgent,
    ProgramScorer,
)
from .executor import ProgramExecutor
from .provider_errors import is_transient_provider_error
from .run_reporting import (
    AgenticSearchReporter,
    report_fallback_to_baseline,
    report_search_stopped,
)
from .session import PlanningSession, ProgramRejected

if TYPE_CHECKING:
    from .usage import AgentRunResult

logger = logging.getLogger(__name__)


class AgenticFeatures(FeatureBackend):
    """Search a SQL feature program with an agent, or replay a saved one.

    With ``program`` set the backend only executes. Otherwise the agent
    searches against ``scorer`` with ``fallback`` scored once as the
    baseline; the winner must beat the baseline on the confirmation slice,
    or the fallback's own matrices are returned. With
    ``config.extend_baseline`` the fallback's columns are part of every
    program, in the search and in the returned matrices. The shared
    interface carries no labels, so ``train_target`` arrives here for
    scoring only.
    """

    def __init__(
        self,
        *,
        program: FeatureProgram | None = None,
        agent: ProgramAgent | None = None,
        scorer: ProgramScorer | None = None,
        train_target: pd.Series | None = None,
        fallback: FeatureBackend | None = None,
        config: AgenticFeatureConfig | None = None,
        task_hint: str = "",
        expected_baseline_columns: tuple[str, ...] | None = None,
        run_context: RunContext | None = None,
        fall_back_on_search_failure: bool = False,
    ) -> None:
        """Keep either a fixed program or the pieces of a search.

        ``run_context`` receives the search's baseline, trials, spend and
        fallback, and its commands apply inside the search; without one,
        nothing is reported. Spend comes from each model response of this
        search, never from the shared agent object. With
        ``fall_back_on_search_failure``, a search whose model provider could
        not answer, or whose agent found no program, returns the fallback's
        matrices; any other error, such as a bad API key, is raised. Without
        it, every failed search raises.
        """
        searching = (
            agent is not None and scorer is not None and train_target is not None
        )
        if program is None and not searching:
            raise ValueError("searching needs an agent, a scorer, and a train target")
        if config is not None and config.extend_baseline and fallback is None:
            raise ValueError("extend_baseline needs a fallback backend to extend")
        self.program = program
        self.agent = agent
        self.scorer = scorer
        self.train_target = train_target
        self.fallback = fallback
        self.config = config or AgenticFeatureConfig()
        self.task_hint = task_hint
        self.expected_baseline_columns = expected_baseline_columns
        self.run_context = run_context or RunContext.unobserved()
        self.fall_back_on_search_failure = fall_back_on_search_failure
        self.last_program = program
        self.last_session: PlanningSession | None = None
        self.last_run_result: AgentRunResult | None = None
        #: Why the last compute used NoriDFS instead of a program, if it did.
        self.last_fallback_reason: FallbackReason | None = None
        self.last_baseline_columns: tuple[str, ...] = ()
        #: Column kinds from the last execution. `FeatureMatrices` carries only
        #: frames and names, and a caller cannot recover these from dtypes: a
        #: prose column and a category are both object-typed, and fitting Nori
        #: on prose as a category is what this backend exists to avoid.
        self.last_categorical: tuple[str, ...] | None = ()
        self.last_text: tuple[str, ...] = ()

    def categorical_columns(self) -> tuple[str, ...] | None:
        """Return the category declaration from the last executed result."""
        return self.last_categorical

    def compute(
        self,
        snapshot: RelationalData,
        train_rows: pd.DataFrame,
        query_rows: pd.DataFrame,
        entity_table: str,
        *,
        row_id_column: str,
    ) -> FeatureMatrices:
        """Search or replay, then return features in the supplied row order."""
        self.last_run_result = None
        self.last_fallback_reason = None
        database = DatabaseView.from_parts(
            tables=snapshot.tables,
            primary_keys=snapshot.primary_keys,
            foreign_keys=snapshot.foreign_keys,
            time_columns=snapshot.time_columns,
        )
        if entity_table not in database.primary_keys:
            raise ValueError("AgenticFeatures requires a single-column entity key")
        entity_column = database.primary_keys[entity_table]
        combined = pd.concat([train_rows, query_rows], ignore_index=True)
        if row_id_column not in combined or combined[row_id_column].isna().any():
            raise ValueError("feature rows require non-null row identifiers")
        if combined[row_id_column].duplicated().any():
            raise ValueError("feature rows require unique row identifiers")
        keys = [entity_column, CUTOFF_COLUMN]

        # Computed once: the search scores it as the baseline, and it is the
        # answer when no program survives or the base every program extends.
        # A plain replay uses neither, so it skips the DFS pass.
        needs_baseline = self.program is None or self.config.extend_baseline
        baseline = (
            None
            if self.fallback is None or not needs_baseline
            else self.fallback.compute(
                snapshot,
                train_rows,
                query_rows,
                entity_table,
                row_id_column=row_id_column,
            )
        )
        self.last_baseline_columns = (
            () if baseline is None else tuple(baseline.feature_names)
        )
        declared_baseline_categorical = (
            () if baseline is None else self.fallback.categorical_columns()
        )
        # The search scorer needs explicit columns. Only infer them when the
        # fallback leaves category detection automatic.
        baseline_categorical = (
            tuple(baseline.train.select_dtypes(exclude="number").columns)
            if declared_baseline_categorical is None
            else declared_baseline_categorical
        )
        program = self.program
        if program is None:
            program, fallback_reason = self._search(
                database,
                train_rows,
                keys,
                entity_table,
                baseline=baseline,
                baseline_categorical=baseline_categorical,
            )
            if program is None:
                if baseline is None:
                    raise RuntimeError("no program survived confirmation")
                assert fallback_reason is not None
                self.last_fallback_reason = fallback_reason
                report_fallback_to_baseline(self.run_context, fallback_reason)
                self.last_categorical = declared_baseline_categorical
                self.last_text = ()
                return baseline
        missing_columns = sorted(
            set(program.drop_baseline) - set(self.last_baseline_columns)
        )
        if missing_columns:
            raise ValueError(
                "saved program no longer matches data: "
                f"missing baseline columns {missing_columns}"
            )
        if (
            self.program is not None
            and self.expected_baseline_columns is not None
            and self.last_baseline_columns != self.expected_baseline_columns
        ):
            raise ValueError(
                "saved feature program no longer matches this data: "
                "NoriDFS columns changed; run a new search"
            )
        self.last_program = program
        # Running the program over every row can be the longest SQL of the
        # run, so a cancel or pause sent during confirmation applies first.
        self.run_context.apply_pending_commands()
        executor = ProgramExecutor(
            database,
            entity_column=entity_column,
            timeout_seconds=self.config.sql_timeout_seconds,
            allow_text=self.config.allow_text,
        )
        try:
            executed = executor.run(program, combined[keys])
        finally:
            executor.close()
        frame = executed.frame
        self.last_categorical = executed.categorical_columns
        self.last_text = executed.text_columns
        if self.config.extend_baseline:
            assert baseline is not None
            clashes = sorted(set(frame.columns) & set(baseline.feature_names))
            if clashes:
                raise ValueError(
                    f"columns {clashes} exist in both the program and the baseline"
                )
            base = pd.concat([baseline.train, baseline.query], ignore_index=True)
            base = base.drop(columns=list(program.drop_baseline))
            frame = pd.concat([base, frame.reset_index(drop=True)], axis=1)
            self.last_categorical = (
                tuple(c for c in baseline_categorical if c in base)
                + self.last_categorical
            )
        split = len(train_rows)
        return FeatureMatrices(
            train=frame.iloc[:split].reset_index(drop=True),
            query=frame.iloc[split:].reset_index(drop=True),
            feature_names=list(frame.columns),
        )

    def _search(
        self,
        database: DatabaseView,
        train_rows: pd.DataFrame,
        keys: list[str],
        entity_table: str,
        *,
        baseline: FeatureMatrices | None,
        baseline_categorical: tuple[str, ...],
    ) -> tuple[FeatureProgram | None, FallbackReason | None]:
        """Return the confirmed program, or ``None`` and why NoriDFS is used."""
        assert self.agent is not None and self.scorer is not None
        assert self.train_target is not None
        if len(self.train_target) != len(train_rows):
            raise ValueError("train_target must align with train_rows")
        # The search's first request boundary, so limits are checked too.
        if self._apply_run_commands(check_limits=True):
            # The agent never starts, so it can deliver no held steering.
            self.run_context.reject_undelivered_steering_texts("not_applicable")
            report_search_stopped(self.run_context, trials=0)
            return None, FallbackReason.STOPPED_WITHOUT_PROGRAM
        session = PlanningSession(
            database=database,
            anchors=AnchorDefinition.create(
                entity_table=entity_table,
                entity_column=keys[0],
                cutoff_column=CUTOFF_COLUMN,
            ),
            config=self.config,
            train_anchors=train_rows[keys],
            train_target=self.train_target,
            scorer=self.scorer,
            baseline_features=None if baseline is None else baseline.train,
            baseline_categorical=baseline_categorical,
            baseline_plan=(
                self.fallback.last_plan if isinstance(self.fallback, NoriDFS) else None
            ),
            apply_run_commands=self._apply_run_commands,
            deliver_run_steering_texts=self.run_context.deliver_held_steering_texts,
            has_run_steering=self.run_context.has_held_steering,
        )
        self.last_session = session
        try:
            # Inside the try, so a reporting failure still closes the session.
            reporter = AgenticSearchReporter(
                self.run_context, session, max_trials=self.config.max_trials
            )
            if baseline is not None:
                reporter.report_baseline_score(len(baseline.feature_names))
            session.on_trial_scored = reporter.report_scored_trial
            session.on_model_response = reporter.report_model_response
            try:
                self.last_run_result = self.agent.run(session, self.task_hint)
            except Exception as error:
                # A cancel sent meanwhile wins over the error.
                self.run_context.apply_pending_commands()
                self.run_context.reject_undelivered_steering_texts("not_applicable")
                # An agent that lets the stop refusal escape still ended the
                # search the way a stop asks, so selection goes ahead.
                is_stop_refusal = session.interrupted and isinstance(
                    error, ProgramRejected
                )
                if not is_stop_refusal:
                    fallback_reason = self._get_fallback_reason_for_search_error(error)
                    if fallback_reason is None:
                        raise
                    return None, fallback_reason
            # The agent delivers steering while it searches; what it never
            # delivered, such as a steer sent with a stop, is rejected here.
            self.run_context.reject_undelivered_steering_texts("not_applicable")
            with self.run_context.report_stage(Stage.SELECT_PIPELINE):
                return self._select_pipeline(session)
        finally:
            session.close()

    def _apply_run_commands(self, *, check_limits: bool = False) -> bool:
        """Apply the run's commands; return whether a stop ends the search.

        With ``check_limits``, a reached runtime or spend limit stops it too.
        """
        self.run_context.apply_pending_commands(check_limits=check_limits)
        return self.run_context.stop_applied

    def _get_fallback_reason_for_search_error(
        self, error: Exception
    ) -> FallbackReason | None:
        """Return why NoriDFS is used after ``error``, or ``None`` to raise it.

        Only with ``fall_back_on_search_failure``: an agent that found no
        program, and a model provider that could not answer, fall back. Any
        other error, such as a bad API key, is raised.
        """
        if not self.fall_back_on_search_failure:
            return None
        if isinstance(error, NoAcceptedProgram):
            return FallbackReason.NO_PROGRAM_FOUND
        if is_transient_provider_error(error):
            logger.warning(
                "the model provider failed; using the baseline", exc_info=error
            )
            return FallbackReason.SEARCH_FAILED
        return None

    def _select_pipeline(
        self, session: PlanningSession
    ) -> tuple[FeatureProgram | None, FallbackReason | None]:
        """Confirm the best program, or return why NoriDFS is used instead.

        A stop the search saw (``session.interrupted``) ended it early: the
        best program is then confirmed only if the estimated confirmation fits
        in ``confirm_budget_fraction`` of ``stop_grace_s``. A stop or limit
        applied after the agent finished on its own changes nothing here.
        """
        if session.interrupted:
            report_search_stopped(self.run_context, trials=len(session.trials))
            if session.best is None:
                return None, FallbackReason.STOPPED_WITHOUT_PROGRAM
            limits = self.run_context.limits
            budget_seconds = limits.stop_grace_s * limits.confirm_budget_fraction
            if session.estimated_confirm_seconds() > budget_seconds:
                return None, FallbackReason.STOPPED_BEFORE_CONFIRMATION
        elif session.best is None:
            return None, FallbackReason.NO_PROGRAM_FOUND
        try:
            confirmed = session.confirm()
        except Exception as error:
            if not (
                self.fall_back_on_search_failure and is_transient_provider_error(error)
            ):
                raise
            logger.warning(
                "confirming the best program failed; using the baseline",
                exc_info=error,
            )
            return None, FallbackReason.SEARCH_FAILED
        if not confirmed:
            return None, FallbackReason.NO_PROGRAM_CONFIRMED
        return session.best_program(), None


__all__ = ["AgenticFeatures"]
