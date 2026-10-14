"""Feature backends the database client can build before Nori predicts.

``"fastdfs"`` is the default. ``"noridfs"`` plans the NoriDFS relationship
search on the snapshot. ``AgenticSearch`` starts from that NoriDFS
table and lets an LLM agent add SQL feature columns, scoring each candidate
program with the client's own Nori predictor, local or hosted, on a slice of
the historical context.

Importing this module needs only the core dependencies; the agent's model
provider and DuckDB load when an ``AgenticSearch`` runs.
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from nori_rel._feature_program import (
    PROGRAM_RECORD_FORMAT,
    PROGRAM_RECORD_VERSION,
    program_from_record,
    program_record,
)
from nori_rel.contracts import PredictionTask, RelationalOutputType, SchemaGraph
from nori_rel.feature._contract import FEATURE_BACKENDS, FeatureBackendName
from nori_rel.feature.agentic import (
    AgenticFeatureConfig,
    AgenticFeatures,
    ProgramAgent,
    ScoreReport,
    TaskKind,
)
from nori_rel.feature.agentic._agent_base import (
    DEFAULT_MAX_AGENT_TURNS as MAX_MODEL_RESPONSES,
)
from nori_rel.feature.agentic.scoring import metric_name, task_score
from nori_rel.feature.agentic.usage import TokenUsage
from nori_rel.feature.dfs.noridfs import NoriDFS
from nori_rel.feature.dfs.noridfs.contracts import NativeFeatureConfig
from nori_rel.run import RunLimits

if TYPE_CHECKING:
    from nori_rel._runtime import Predictor
    from nori_rel.feature.agentic.usage import TokenPrices
    from nori_rel.run import RunContext

#: The NoriDFS relationship search every ``noridfs`` and agent-search run
#: starts from. Changing it changes the features of a new run, so a test pins
#: it.
NORIDFS_CONFIG = NativeFeatureConfig(
    max_depth=2,
    max_paths=24,
    max_features=48,
    max_direct_features=1,
    max_categorical_cardinality=32,
    window_multipliers=(1, 4, 16),
)

#: The search settings this agent was developed with, with client text
#: disabled. The 1% gain floor and three context cutoffs are safeguards that
#: have not been benchmarked.
DEFAULT_SEARCH_CONFIG = AgenticFeatureConfig(
    max_trials=24,
    holdout_fraction=0.15,
    confirmation_fraction=0.15,
    extend_baseline=True,
    align_to_cutoffs=True,
    allow_text=False,
    min_relative_gain=0.01,
    min_context_cutoffs=3,
)

#: The agent settings it was developed with: a floor of 12 scored trials, at most 80
#: model responses, and a one-hour prompt cache.
DEFAULT_CLAUDE_MODEL = "claude-sonnet-5-5"
MIN_SCORED_TRIALS = 12

AGENTIC_INSTALL_HINT = 'pip install "nori-rel[database,agentic]"'


def require_agentic_modules(*modules: str) -> None:
    """Raise ``ImportError`` with the install command unless every module exists.

    An agent search imports DuckDB and its provider's SDK only once the first
    prediction is under way, after the snapshot read and the NoriDFS pass.
    Checking when the search is configured turns a missing ``agentic`` extra
    into an error at client construction instead.
    """
    missing = [name for name in modules if importlib.util.find_spec(name) is None]
    if missing:
        raise ImportError(
            f"AgenticSearch needs {', '.join(missing)}; run `{AGENTIC_INSTALL_HINT}`"
        )


def noridfs_backend(horizon_days: int) -> NoriDFS:
    """Return the NoriDFS search with windows scaled to the target's.

    Aggregation windows are 1x, 4x and 16x ``horizon_days``, the length of the
    PQL label interval.
    """
    return NoriDFS(NORIDFS_CONFIG, horizon=pd.Timedelta(days=horizon_days))


@dataclass(frozen=True)
class AgenticSearch:
    """Let an LLM agent add SQL features to NoriDFS for each prediction.

    The agent writes SQL over the local snapshot. Every candidate program is
    executed locally, appended to the NoriDFS table, and scored with one call
    to the client's Nori predictor on a holdout slice of the historical
    context. A program is kept only if it also beats NoriDFS alone on a
    separate confirmation slice; otherwise the prediction uses NoriDFS alone.

    The agent's tools send the agent's model provider the schema, column
    profiles (which include up to 10 raw values of each profiled categorical
    column), the results of its exploratory SELECTs (at most 50 rows each),
    and the error messages of failed queries, which can quote values. Do not
    enable this on data that may not leave your environment for that provider.
    In hosted mode, Nori still receives only feature matrices and labels.

    ``config`` extends NoriDFS with at most 24 trials, a 1% minimum relative
    gain, and at least three training cutoff groups. A search must leave text
    and target transforms off, which the client's prediction
    path does not carry. Use ``AgenticSearch.claude(model=...)`` for the default
    agent settings, or ``AgenticSearch.openai(model=...)`` for a model behind
    an OpenAI-compatible endpoint.
    """

    agent: ProgramAgent
    config: AgenticFeatureConfig = DEFAULT_SEARCH_CONFIG
    task_hint: str = ""
    #: Bounds on each prediction's run: ``max_runtime_seconds`` and
    #: ``max_spend_usd`` end the search gracefully before its next model
    #: request, as a stop does, and ``stop_grace_s`` budgets the work after a
    #: stop. ``max_spend_usd`` needs an agent that reports dollars, which the
    #: Claude agent does only with ``prices``; the search refuses it otherwise.
    #: The trial cap is ``config.max_trials``.
    limits: RunLimits = field(default_factory=RunLimits)

    def __post_init__(self) -> None:
        """Require DuckDB and refuse settings the client or its agent cannot honour."""
        require_agentic_modules("duckdb")
        if not self.config.extend_baseline:
            raise ValueError("the client's agent search extends NoriDFS")
        if self.config.allow_text:
            raise ValueError("the client sends no text columns; set allow_text=False")
        if self.config.allow_target_transform:
            raise ValueError("the client's Nori path takes no target transform")
        has_spend_limit = self.limits.max_spend_usd is not None
        if has_spend_limit and not is_agent_reporting_spend_in_dollars(self.agent):
            raise ValueError(
                "limits.max_spend_usd needs an agent that reports its spend in "
                "dollars: pass prices=TokenPrices(...) to AgenticSearch.claude. "
                "The OpenAI agent reports no spend; bound it with "
                "limits.max_runtime_seconds"
            )

    @classmethod
    def claude(
        cls,
        *,
        model: str = DEFAULT_CLAUDE_MODEL,
        effort: str = "high",
        api_key: str | None = None,
        prices: TokenPrices | None = None,
        max_cost_usd: float | None = None,
        task_hint: str = "",
    ) -> AgenticSearch:
        """Build a capped Claude agent; needs ``nori-rel[agentic]``.

        ``model`` is any Anthropic model id; the search was developed with the
        default. ``api_key`` defaults to ``ANTHROPIC_API_KEY``; without either
        this raises ``ValueError`` here, rather than letting every prediction's
        search fail. ``max_cost_usd`` stops a search once the token
        cost at ``prices`` reaches it, keeping the best program found so far.
        The cap resets for each prediction, not each client. One response may
        cross it; that response's tools are not executed. Nori cost is separate.
        """
        require_agentic_modules("anthropic")
        import anthropic

        from nori_rel.feature.agentic.agent_claude import ClaudeProgramAgent

        client = anthropic.Anthropic(
            api_key=_get_required_api_key(
                api_key, "ANTHROPIC_API_KEY", "AgenticSearch.claude"
            )
        )
        agent = ClaudeProgramAgent(
            model=model,
            effort=effort,
            max_turns=MAX_MODEL_RESPONSES,
            cache_ttl="1h",
            prices=prices,
            max_cost_usd=max_cost_usd,
            min_trials=MIN_SCORED_TRIALS,
            client=client,
        )
        return cls(agent=agent, task_hint=task_hint)

    @classmethod
    def openai(
        cls,
        *,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        task_hint: str = "",
    ) -> AgenticSearch:
        """Build an agent on an OpenAI-compatible endpoint; needs ``nori-rel[agentic]``.

        ``model`` has no default because the search was developed with Claude only.
        ``base_url`` points the OpenAI SDK at any compatible server, such as a
        hosted open-weight model; ``api_key`` defaults to ``OPENAI_API_KEY``,
        and without either this raises ``ValueError``. A server that needs no
        key still takes any non-blank ``api_key``, as the OpenAI SDK requires
        one. Without ``base_url`` the agent caps responses
        with ``max_completion_tokens``, which OpenAI's reasoning models require;
        with one it sends ``max_tokens``, which compatible servers accept more
        widely. The agent keeps the floor of 12 scored trials and the cap
        of 80 model responses. It records no token cost, so there is no dollar
        cap and ``limits.max_spend_usd`` is refused.
        """
        require_agentic_modules("openai")
        from nori_rel.feature.agentic.agent_openai import OpenAIProgramAgent

        agent = OpenAIProgramAgent(
            model=model,
            max_turns=MAX_MODEL_RESPONSES,
            min_trials=MIN_SCORED_TRIALS,
            base_url=base_url,
            api_key=_get_required_api_key(
                api_key, "OPENAI_API_KEY", "AgenticSearch.openai"
            ),
            token_limit_field=(
                "max_completion_tokens" if base_url is None else "max_tokens"
            ),
        )
        return cls(agent=agent, task_hint=task_hint)

    def backend(
        self,
        seed: NoriDFS,
        *,
        labels: pd.Series,
        task: PredictionTask,
        predictor: Predictor,
        run_context: RunContext | None = None,
    ) -> AgenticFeatures:
        """Return the search for one prediction; ``labels`` follow context order.

        ``run_context`` receives the search's progress events.
        """
        kind: TaskKind = (
            "binary" if task is PredictionTask.CLASSIFICATION else "regression"
        )
        return AgenticFeatures(
            agent=self.agent,
            scorer=ClientProgramScorer(predictor=predictor, task=kind),
            train_target=labels.reset_index(drop=True),
            fallback=seed,
            config=self.config,
            task_hint=self.task_hint,
            run_context=run_context,
            fall_back_on_search_failure=True,
        )


@dataclass(frozen=True)
class ClientProgramScorer:
    """Score a program with one call to the client's Nori predictor.

    Regression scores negative MAE on the median, binary scores ROC-AUC on the
    probability: the outputs the client returns. One call has no seed spread,
    so the reported noise is zero. The client's acceptance checks also require
    the configured minimum relative gain; zero noise does not remove that floor.
    """

    predictor: Predictor
    task: TaskKind = "regression"

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
        """Fit Nori on the context rows and score the holdout rows."""
        if text_columns:
            raise ValueError("the client sends Nori no text columns")
        binary = self.task == "binary"
        predictions, _ = self.predictor.predict_arrays(
            context_features.reset_index(drop=True),
            context_target.reset_index(drop=True),
            holdout_features.reset_index(drop=True),
            task=PredictionTask.CLASSIFICATION if binary else PredictionTask.REGRESSION,
            output_type=None if binary else RelationalOutputType.MEDIAN,
            categorical_columns=list(categorical_columns),
        )
        values = np.asarray(predictions, dtype=float)
        if values.shape != (len(holdout_features),) or not np.isfinite(values).all():
            raise ValueError("Nori returned invalid predictions or the wrong row count")
        target = holdout_target.to_numpy(dtype=float)
        return ScoreReport(
            metric=metric_name(self.task),
            score=task_score(self.task, target, values),
            noise=0.0,
            predictions=values,
        )


def is_agent_reporting_spend_in_dollars(agent: ProgramAgent) -> bool:
    """Return whether ``agent`` reports a dollar cost with each model response.

    Only an agent with token ``prices`` does (the Claude agent given prices);
    without prices a spend limit could never end its search.
    """
    return getattr(agent, "prices", None) is not None


def _get_required_api_key(
    api_key: str | None, environment_variable: str, factory_name: str
) -> str:
    """Return ``api_key``, or the key in ``environment_variable``.

    Raises ``ValueError`` naming both when neither is set: a search without a
    key would fail on its first model call, in every prediction.
    """
    key = api_key if api_key is not None else os.environ.get(environment_variable)
    if key is None or not key.strip():
        raise ValueError(
            f"{factory_name} needs an API key: pass api_key= or set "
            f"{environment_variable}"
        )
    return key


def agentic_summary(
    backend: Any, *, query: str, entity_created_at: str, schema: SchemaGraph
) -> dict[str, Any]:
    """Return what a prediction's metadata records about its agent features.

    ``agentic_program`` is the winning program as a ``program_record``, or
    ``None`` when NoriDFS alone was kept, and ``agentic_fallback_reason`` then
    says why, with or without an ``on_event`` listener. A saved program reports
    zero trials.
    """
    if not isinstance(backend, AgenticFeatures):
        return {}
    session = backend.last_session
    program = backend.last_program
    run_result = backend.last_run_result
    replay = session is None
    return {
        "agentic_model": None if run_result is None else run_result.model,
        "agentic_cost_usd": 0.0
        if replay
        else (None if run_result is None else run_result.cost_usd),
        "agentic_usage": TokenUsage().as_dict()
        if replay
        else (
            None
            if run_result is None or run_result.usage is None
            else run_result.usage.as_dict()
        ),
        "agentic_stop_reason": "replay"
        if replay
        else (None if run_result is None else run_result.stop_reason),
        "agentic_fallback_reason": None
        if backend.last_fallback_reason is None
        else backend.last_fallback_reason.value,
        "agentic_trials": 0 if session is None else len(session.trials),
        "agentic_queries": None
        if program is None
        else [query.name for query in program.queries],
        "agentic_program": None
        if program is None
        else program_record(
            program,
            query=query,
            entity_created_at=entity_created_at,
            schema=schema,
            baseline_columns=backend.last_baseline_columns,
        ),
    }


__all__ = [
    "AGENTIC_INSTALL_HINT",
    "DEFAULT_SEARCH_CONFIG",
    "FEATURE_BACKENDS",
    "NORIDFS_CONFIG",
    "PROGRAM_RECORD_FORMAT",
    "PROGRAM_RECORD_VERSION",
    "AgenticSearch",
    "ClientProgramScorer",
    "FeatureBackendName",
    "agentic_summary",
    "noridfs_backend",
    "program_from_record",
    "program_record",
    "require_agentic_modules",
]
