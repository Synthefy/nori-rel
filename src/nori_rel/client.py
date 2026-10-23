"""Local relational feature processing with hosted or local Nori inference."""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal, Required, TypedDict, Unpack, cast

import pandas as pd

from nori_rel._pql_static import (
    get_pql_target_kind,
    require_static_target_arguments,
)
from nori_rel.contracts import (
    AgentSearchMetric,
    ConnectionStatus,
    CredentialReference,
    Database,
    DatabaseCreateRequest,
    DatabaseEngine,
    DatabaseUrl,
    ForeignKeySchema,
    RelationalOutputType,
    SchemaGraph,
    WarehouseConnection,
)
from nori_rel.contracts import (
    PredictionTask as WirePredictionTask,
)
from nori_rel.pql.context_sql import DEFAULT_MAX_CONTEXT_ROWS
from nori_rel.run import LocalExecutor, PredictionRun, RunContext, RunEvent

if TYPE_CHECKING:
    from nori_rel._feature_search import AgenticSearch
    from nori_rel._runtime import LocalRelationalRuntime
    from nori_rel.context_policy import ContextPolicy
    from nori_rel.feature import FeatureBackend, FeatureBackendName
    from nori_rel.limits import RelationalLimits
    from nori_rel.pipeline import NoriRelPipeline, PredictionDetails

PredictionOperation = Literal["average", "total", "minimum", "maximum", "count"]


class PredictionRequestArguments(TypedDict, total=False):
    """The keyword arguments of ``predict``, except ``return_details``.

    ``start_predict`` takes them as ``**request``; see ``predict`` for what
    each one means.
    """

    source: Required[Database]
    entity: str | None
    target: str | None
    task: WirePredictionTask | PredictionTask | str | None
    target_time: str | None
    operation: PredictionOperation | None
    lookahead: str | None
    as_of: datetime | None
    relationship_path: list[str] | None
    entity_ids: list[Any] | None
    output_type: RelationalOutputType | str | None
    quantiles: list[float] | None
    positive_class: Any
    decision_threshold: float | None
    query: str | None
    query_filter: str | None
    anchor_times: Sequence[datetime] | None
    anchor_time: datetime | None
    observed_through: datetime | None
    entity_created_at: str | None
    max_context_rows: int | None
    context: ContextPolicy | None
    agent_instructions: str | None
    feature_program: Mapping[str, Any] | None
    metric: AgentSearchMetric | None
    feature_strategy: FeatureBackendName | FeatureBackend | AgenticSearch | None
    log_summary: bool
    pipeline: NoriRelPipeline | None
    expected_context_sha256: str | None
    entity_time: str | None


PredictionTask = Literal["regression", "classification"]
ExecutionMode = Literal["local", "hosted"]
API_KEY_ENV = "SYNTHEFY_NORI_API_KEY"


class NoriRelClient:
    """Predict from a relational database without a separate Nori-Rel service.

    Database reads and relational feature search run in this Python process.
    By default, Nori also runs in this process (mode="local"), without an API
    key; local mode supports regression targets only. mode="hosted" sends only
    the resulting context and query feature matrices to hosted Nori.

    ``predict(feature_strategy=...)`` picks the relational feature search for
    each prediction: ``"fastdfs"`` (the default), ``"noridfs"`` (the NoriDFS
    relationship search), or an ``AgenticSearch``, which adds agent-written SQL
    features to NoriDFS and sends schema, column profiles with raw categorical
    values, sampled query results, and query error messages to the agent's
    model provider. A configured ``FeatureBackend`` instance is also accepted;
    its caller is responsible for concurrent use. See ``AgenticSearch``.

    ``features`` on the constructor is deprecated: it still sets the search
    for predictions that pass no ``feature_strategy``, with a
    ``DeprecationWarning``, and will be removed in 0.5.0.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        mode: ExecutionMode = "local",
        model: str = "nori-1-5",
        classification_model: str | None = None,
        timeout: float = 300.0,
        max_retries: int = 2,
        device: str | None = None,
        limits: RelationalLimits | None = None,
        features: FeatureBackendName | FeatureBackend | AgenticSearch | None = None,
        _runtime: LocalRelationalRuntime | None = None,
    ) -> None:
        """Configure Nori inference and bounded local database execution.

        device applies to mode="local" only. None (default) picks CUDA, then
        Apple MPS, then CPU; pass a PyTorch device such as "cuda:1" to override.
        Hosted mode reads api_key or SYNTHEFY_NORI_API_KEY.
        features is deprecated; pass feature_strategy to predict instead.
        """
        if features is not None:
            warnings.warn(
                "NoriRelClient(features=...) is deprecated and will be removed in "
                "0.5.0; pass feature_strategy= to predict() or start_predict()",
                DeprecationWarning,
                stacklevel=2,
            )
        if mode == "hosted":
            api_key = api_key or os.getenv(API_KEY_ENV)
            if not api_key:
                raise ValueError(f"mode='hosted' requires api_key or {API_KEY_ENV}")
            if device is not None:
                raise ValueError("device requires mode='local'")
        elif mode == "local":
            if api_key is not None:
                raise ValueError(
                    "mode='local' does not use an API key; "
                    "pass mode='hosted' for hosted Nori"
                )
            if classification_model is not None:
                raise ValueError(
                    "mode='local' supports regression targets only; "
                    "classification_model requires mode='hosted'"
                )
        else:
            raise ValueError("mode must be 'local' or 'hosted'")
        if not model or not model.strip():
            raise ValueError("model must not be blank")
        if classification_model is not None and not classification_model.strip():
            raise ValueError("classification_model must not be blank")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")

        self.mode = mode
        self.model = model
        self.classification_model = classification_model or model
        self.timeout = timeout
        self.max_retries = max_retries
        if _runtime is None:
            try:
                from nori_rel._runtime import LocalRelationalRuntime
            except ImportError as exc:
                raise ImportError(
                    'Install a connector extra: "nori-rel[database]", '
                    '"nori-rel[snowflake]", or "nori-rel[databricks]"'
                ) from exc
            predictor = None
            if mode == "local":
                from nori_rel._runtime import LocalNoriPredictor

                predictor = LocalNoriPredictor(model, device=device)
            _runtime = LocalRelationalRuntime(
                api_key=api_key,
                model=model,
                classification_model=classification_model,
                timeout=timeout,
                max_retries=max_retries,
                limits=limits,
                predictor=predictor,
                features="fastdfs" if features is None else features,
            )
        self._runtime = _runtime
        self._executor = LocalExecutor()

    def __repr__(self) -> str:
        """Return a credential-free client description."""
        return (
            f"{type(self).__name__}(mode={self.mode!r}, model={self.model!r}, "
            f"classification_model={self.classification_model!r})"
        )

    def __enter__(self) -> NoriRelClient:
        """Return this client for context-manager use."""
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """Release resources when leaving a context manager.

        Leaving with an exception cancels the runs still going, so the exception
        is not held up for the length of a search.
        """
        if exc_type is not None:
            self._executor.cancel_runs(self._executor.close())
        self.close()

    def close(self, *, timeout_seconds: float | None = None) -> None:
        """Close registered database pools after active operations finish.

        Safe to repeat. This client cannot perform database operations or start
        runs afterward; create a new client to reconnect. Also called on
        context-manager exit. Runs started with start_predict are waited for
        first, and a paused run is cancelled, since only a command could end
        it. With ``timeout_seconds``, runs still going after that many seconds
        are cancelled and then waited for; a cancel applies at a run's next
        boundary.
        """
        self._executor.close()
        unfinished_runs = self._executor.wait_for_unfinished_runs(timeout_seconds)
        if unfinished_runs:
            self._executor.cancel_runs(unfinished_runs)
            self._executor.wait_for_unfinished_runs()
        self._runtime.close()

    def connect(
        self,
        *,
        engine: DatabaseEngine | str,
        database_url: str | None = None,
        credential: CredentialReference | None = None,
        schema_name: str | None = None,
        tables: list[str] | None = None,
        time_columns: dict | None = None,
        connection: WarehouseConnection | None = None,
        primary_keys: dict[str, list[str]] | None = None,
        foreign_keys: dict[str, list[ForeignKeySchema]] | None = None,
    ) -> Database:
        """Connect one database using an explicit engine and credential source."""
        if (database_url is None) == (credential is None):
            raise ValueError("provide exactly one of database_url or credential")
        if database_url is not None:
            if not database_url.strip():
                raise ValueError("database_url must not be blank")
            credential = DatabaseUrl(value=database_url)
        credential = cast(CredentialReference, credential)
        engine = DatabaseEngine(engine)
        if schema_name is None:
            if engine is not DatabaseEngine.POSTGRESQL:
                raise ValueError(
                    "warehouse connections require an explicit schema_name"
                )
            schema_name = "public"
        config = DatabaseCreateRequest(
            engine=engine,
            credential=credential,
            schema_name=schema_name,
            tables=tables,
            time_columns=time_columns or {},
            connection=connection,
            primary_keys=primary_keys or {},
            foreign_keys=foreign_keys or {},
        )
        return self._runtime.connect(config)

    def test_connection(self, source: Database) -> ConnectionStatus:
        """Verify connectivity; read-only enforcement depends on the connector."""
        return self._runtime.test_connection(source)

    def discover(self, source: Database) -> SchemaGraph:
        """Read the selected source schema and foreign-key graph."""
        return self._runtime.discover(source)

    def predict(
        self,
        *,
        source: Database,
        entity: str | None = None,
        target: str | None = None,
        task: WirePredictionTask | PredictionTask | str | None = None,
        target_time: str | None = None,
        operation: PredictionOperation | None = None,
        lookahead: str | None = None,
        as_of: datetime | None = None,
        relationship_path: list[str] | None = None,
        entity_ids: list[Any] | None = None,
        output_type: RelationalOutputType | str | None = None,
        quantiles: list[float] | None = None,
        positive_class: Any = None,
        decision_threshold: float | None = None,
        query: str | None = None,
        query_filter: str | None = None,
        anchor_times: Sequence[datetime] | None = None,
        anchor_time: datetime | None = None,
        observed_through: datetime | None = None,
        entity_created_at: str | None = None,
        max_context_rows: int | None = None,
        context: ContextPolicy | None = None,
        agent_instructions: str | None = None,
        feature_program: Mapping[str, Any] | None = None,
        metric: AgentSearchMetric | None = None,
        feature_strategy: FeatureBackendName
        | FeatureBackend
        | AgenticSearch
        | None = None,
        log_summary: bool = True,
        pipeline: NoriRelPipeline | None = None,
        expected_context_sha256: str | None = None,
        entity_time: str | None = None,
        return_details: bool = False,
        on_event: Callable[[RunEvent], None] | None = None,
    ) -> pd.DataFrame | PredictionDetails:
        """Predict using either PQL text or legacy entity/target arguments.

        PQL example:
            client.predict(source=db, query="PREDICT COUNT(orders.*,0,30,days)=0 "
                "FOR users.user_id=42", anchor_times=[historical_datetime],
                observed_through=complete_through_datetime,
                entity_created_at="created_at")

        Temporal PQL requires historical anchor_times, a verified completeness
        watermark and an entity creation column. anchor_time defaults to the
        latest registered event timestamp. Register every event table's timestamp
        at connect(); unmapped tables/entity attributes must be immutable and
        events immediately available. max_context_rows defaults to 5,000.

        A static target predicts a stored column from the other rows whose
        value is stored, with no dates needed:
            client.predict(source=db,
                query="PREDICT accounts.balance FOR EACH accounts.account_id")
        FOR EACH scores the rows whose value is NULL; FOR IDs scores those
        rows, which then never join the context. It takes no anchor_times,
        anchor_time or observed_through. entity_created_at is optional: with
        it each row's features are built as of its own time, without it as of
        the read. The target column never reaches the features. A Boolean
        column or a comparison is binary; a 0/1 numeric column needs task=.

        context is a ContextPolicy that chooses the training rows in place of
        max_context_rows (passing both raises). It selects among candidates
        whose label is known, so an unknown label never takes a row, and can
        balance a 0/1 label, spread rows over cutoffs, cap rows per entity and
        map balanced probabilities back to the population rate. Without it,
        the context is the newest max_context_rows candidates, with unknown
        labels dropped afterwards. Either way attrs["context_policy"] reports
        the rule used, who chose it, the class counts before and after
        selection and the rows kept at each cutoff.

        entity_time names a date/timestamp column of the entity table to predict
        per event, e.g. "view_time" for "will this page view lead to a signup?".
        Each entity row is then one example anchored at its own time: its label
        window starts there, its features read only rows from before that time
        plus its own columns, and rows with the same time do not see each other.
        Pass it instead of anchor_times and entity_created_at. Context rows are
        those whose label window ends by min(anchor_time, observed_through);
        scored rows are those that exist by anchor_time, after query_filter, each
        scored at its own time. The agent search, saved programs and pipelines
        do not support it yet, and no pipeline is recorded.

        agent_instructions is plain text sent verbatim in the agent's request,
        for example "Signups on 1-3 Jun had broken tracking; do not build
        features from them". It needs features=AgenticSearch(...), replaces
        AgenticSearch(task_hint=...), and cannot loosen the leak checks or the
        scoring rules. metadata["context_policy"] records it with the agent
        stages that received it; today the feature search is the only one.

        query_filter is an optional PQL entity predicate, e.g. "users.country='CA'".
        It selects scoring rows independently of historical WHERE, before features
        and inference. It intersects FOR IDs (legacy entity_ids), after validating
        that requested IDs exist at anchor_time. NULL predicates exclude rows;
        no matches returns an empty result. Entity attributes must be immutable.
        PQL plus entity_ids remains an error: use FOR in the query instead.

        feature_program replays a saved agent program: pass the agentic_program
        record from an earlier prediction's metadata to rerun its SQL features on
        top of NoriDFS, with no agent search and no model-provider calls. It
        cannot be combined with feature_strategy, and needs the agentic extra
        for DuckDB.
        Replay requires the same PQL text, entity_created_at, schema and NoriDFS
        columns. Version-1 records lack these checks and must be regenerated.

        feature_strategy picks this prediction's feature search: "fastdfs"
        (the default), "noridfs", a FeatureBackend, or an AgenticSearch. It is
        checked before the run starts and cannot be combined with pipeline or
        feature_program.

        metric chooses the error an agent search (feature_strategy=AgenticSearch)
        scores each candidate SQL program with on held-out history, and so which
        program it keeps: "mae" (mean absolute error, the default), "mse",
        "rmse", "nmae" and "nrmse" (MAE and RMSE divided by the range of the
        actuals), "medae" (median absolute error), "wape" (total absolute error
        divided by total actual), "mape" (mean absolute percentage error over
        rows whose actual is not zero) or "smape" (symmetric MAPE). On one
        holdout NMAE and WAPE rank programs as MAE does; MAPE and SMAPE weigh
        every row equally in relative terms, so small series count as much as
        large ones. It does not change the forecast itself (the median, or
        output_type). It applies only to regression targets with an agent
        search; any other use, or an unknown metric, raises ValueError.

        pipeline predicts with a saved NoriRelPipeline and no feature search.
        Pass source, as_of (the prediction time) and observed_through, and
        optionally query_filter or a query with another FOR clause; the
        pipeline fixes the rest, and passing anything else raises ValueError.
        Each run reads new context rows with the pipeline's saved rules, at
        anchors placed before as_of as in the first run, so the context moves
        forward with as_of; a ContextPolicy the first run used is applied
        again. A PipelineError is raised before Nori runs if the
        schema, the model or the feature columns do not fit the pipeline.
        expected_context_sha256 asks to reproduce an earlier run, with or
        without a pipeline: pass its attrs["context_rows_sha256"], and a run on
        other context rows raises PipelineContextError. Every prediction
        reports the hash of its context rows and of its feature matrix in attrs
        and in the run summary.

        PQL returns entity_id, anchor_time, prediction; binary adds probability,
        quantile output adds quantiles. attrs holds task/model and context counts.
        return_details=True returns PredictionDetails with predictions
        (DataFrame), metadata (dict), pipeline (NoriRelPipeline or None) and
        features, the scored rows' feature matrix (one row per entity_id, one
        column per feature, under dotted names such as results.grid.mean);
        details.get_feature_table() lists each feature column with its
        internal name, source, SQL and value summary. details["predictions"]
        works too. Save the pipeline to predict again in another process
        without a feature search.
        log_summary emits INFO records through Python logging. Task is resolved
        explicitly from the target; conflicting task overrides error.
        Larger-context retrieval is not implemented yet.

        Legacy compatibility supports only regression count/total/average/minimum/
        maximum over whole days and requires the same temporal guarantees.
        as_of aliases anchor_time; with neither, legacy calls retain wall-clock
        now. Results use the PQL format and scoring IDs never restrict context.
        Direct/NEXT/named-class/path requests are retired and raise errors.

        on_event receives each of the prediction's events as it happens, on
        this thread: its stages, the agent search's trials and spend, and its
        final status; attrs["run"] then holds the run's summary. start_predict
        runs the same prediction in the background.
        """
        # Every argument except these three describes the prediction itself.
        request = dict(locals())
        for name in ("self", "return_details", "on_event"):
            del request[name]
        run = self._executor.run_on_this_thread(
            self._prepare_pql_prediction(
                **request, keep_scoring_features=return_details
            ),
            on_event=on_event,
            summary_in_result_attrs=on_event is not None,
            **self._get_run_options_for_features(
                feature_program is not None, pipeline, feature_strategy
            ),
        )
        frame = run.result()
        if not return_details:
            return frame
        from nori_rel.pipeline import (
            CONTEXT_FEATURES_ATTR,
            CONTEXT_LABELS_ATTR,
            PIPELINE_ATTR,
            SCORING_FEATURE_SQL_ATTR,
            SCORING_FEATURES_ATTR,
            NoriRelPipeline,
            PredictionDetails,
        )

        features = frame.attrs.pop(SCORING_FEATURES_ATTR, None)
        feature_sql = frame.attrs.pop(SCORING_FEATURE_SQL_ATTR, None) or {}
        context = frame.attrs.pop(CONTEXT_LABELS_ATTR, None)
        context_features = frame.attrs.pop(CONTEXT_FEATURES_ATTR, None)
        return PredictionDetails(
            predictions=frame,
            metadata={
                key: value for key, value in frame.attrs.items() if key != PIPELINE_ATTR
            },
            pipeline=NoriRelPipeline.from_prediction(frame),
            features=features,
            feature_sql=feature_sql,
            context=context,
            context_features=context_features,
        )

    def start_predict(
        self,
        *,
        on_event: Callable[[RunEvent], None] | None = None,
        **request: Unpack[PredictionRequestArguments],
    ) -> PredictionRun:
        """Start a prediction in the background and return its run.

        Takes the keyword arguments of predict except return_details. Their
        combinations are checked before it returns (legacy arguments also read
        the schema); problems found while running, such as invalid PQL or a
        task the target contradicts, fail the run. The run reports progress
        through run.events() and on_event, which is called on the run's worker
        thread. run.result() returns the frame, whose attrs hold the metadata
        and attrs["run"] the run's summary. Runs do not wait for one another,
        even when they share an agent. Find the run again with
        get_run(run.run_id).
        """
        if "return_details" in request:
            raise TypeError(
                "start_predict does not take return_details; "
                "run.result() returns the frame, and frame.attrs holds the metadata"
            )
        unknown = sorted(set(request) - set(PredictionRequestArguments.__annotations__))
        if unknown:
            raise TypeError(
                f"start_predict() got an unexpected keyword argument {unknown[0]!r}"
            )
        return self._executor.start_run(
            self._prepare_pql_prediction(**request),
            on_event=on_event,
            **self._get_run_options_for_features(
                request.get("feature_program") is not None,
                request.get("pipeline"),
                request.get("feature_strategy"),
            ),
        )

    def get_run(self, run_id: str) -> PredictionRun:
        """Return a run started by start_predict; KeyError if unknown or forgotten."""
        return self._executor.get_run(run_id)

    def _get_run_options_for_features(
        self,
        replaying_program: bool,
        pipeline: NoriRelPipeline | None = None,
        feature_strategy: FeatureBackendName
        | FeatureBackend
        | AgenticSearch
        | None = None,
    ) -> dict[str, Any]:
        """Return a run's strategy and, for an agent search, trial cap and limits.

        Strategy names match the ``features`` entry of prediction metadata.
        ``feature_strategy`` is the prediction's own feature search; ``None``
        uses the client's. A saved pipeline fixes its own strategy and runs no
        search.
        """
        from nori_rel._pql_runtime import (
            get_feature_strategy_name,
            get_pipeline_strategy_name,
        )

        if pipeline is not None:
            return {"strategy": get_pipeline_strategy_name(pipeline.manifest.features)}
        features = (
            getattr(self._runtime, "features", "fastdfs")
            if feature_strategy is None
            else feature_strategy
        )
        strategy = get_feature_strategy_name(
            features, replaying_program=replaying_program
        )
        if strategy != "agentic":
            return {"strategy": strategy}
        return {
            "strategy": strategy,
            "max_trials": int(features.config.max_trials),
            "limits": features.limits,
        }

    def _prepare_pql_prediction(
        self,
        *,
        source: Database,
        entity: str | None = None,
        target: str | None = None,
        task: WirePredictionTask | PredictionTask | str | None = None,
        target_time: str | None = None,
        operation: PredictionOperation | None = None,
        lookahead: str | None = None,
        as_of: datetime | None = None,
        relationship_path: list[str] | None = None,
        entity_ids: list[Any] | None = None,
        output_type: RelationalOutputType | str | None = None,
        quantiles: list[float] | None = None,
        positive_class: Any = None,
        decision_threshold: float | None = None,
        query: str | None = None,
        query_filter: str | None = None,
        anchor_times: Sequence[datetime] | None = None,
        anchor_time: datetime | None = None,
        observed_through: datetime | None = None,
        entity_created_at: str | None = None,
        max_context_rows: int | None = None,
        context: ContextPolicy | None = None,
        agent_instructions: str | None = None,
        feature_program: Mapping[str, Any] | None = None,
        metric: AgentSearchMetric | None = None,
        feature_strategy: FeatureBackendName
        | FeatureBackend
        | AgenticSearch
        | None = None,
        log_summary: bool = True,
        pipeline: NoriRelPipeline | None = None,
        expected_context_sha256: str | None = None,
        keep_scoring_features: bool = False,
        entity_time: str | None = None,
    ) -> Callable[[RunContext], pd.DataFrame]:
        """Check a predict request now; return the prediction to run later.

        ``keep_scoring_features`` makes the run keep the scored rows' feature
        matrix for ``PredictionDetails.features``.
        """
        if pipeline is not None:
            return self._get_pipeline_prediction_run(
                pipeline,
                source=source,
                request={
                    "as_of": as_of,
                    "observed_through": observed_through,
                    "query": query,
                    "query_filter": query_filter,
                    "expected_context_sha256": expected_context_sha256,
                },
                fixed={
                    "entity": entity,
                    "target": target,
                    "task": task,
                    "target_time": target_time,
                    "operation": operation,
                    "lookahead": lookahead,
                    "relationship_path": relationship_path,
                    "entity_ids": entity_ids,
                    "output_type": output_type,
                    "quantiles": quantiles,
                    "positive_class": positive_class,
                    "decision_threshold": decision_threshold,
                    "anchor_times": anchor_times,
                    "anchor_time": anchor_time,
                    "entity_created_at": entity_created_at,
                    "max_context_rows": max_context_rows,
                    "context": context,
                    "agent_instructions": agent_instructions,
                    "feature_program": feature_program,
                    "metric": metric,
                    "feature_strategy": feature_strategy,
                    "entity_time": entity_time,
                },
                log_summary=log_summary,
                keep_scoring_features=keep_scoring_features,
            )
        if feature_strategy is not None:
            from nori_rel._runtime import get_validated_features

            get_validated_features(feature_strategy)
            if feature_program is not None:
                raise ValueError(
                    "feature_strategy cannot be passed with feature_program: the "
                    "saved program always replays on NoriDFS"
                )
        if context is not None:
            from nori_rel.context_policy import ContextPolicy

            if not isinstance(context, ContextPolicy):
                raise TypeError("context must be a ContextPolicy")
            if max_context_rows is not None:
                raise ValueError(
                    "pass max_rows on the ContextPolicy, not max_context_rows"
                )
        if agent_instructions is not None:
            self._check_agent_instructions(
                agent_instructions,
                feature_strategy=feature_strategy,
                replaying_program=feature_program is not None,
            )
        program = None
        if feature_program is not None:
            from nori_rel._feature_search import (
                program_from_record,
                require_agentic_modules,
            )

            require_agentic_modules("duckdb")
            program = program_from_record(feature_program)
        legacy = query is None
        if legacy:
            if entity is None or target is None:
                raise ValueError("provide entity and target, or a PQL query")
            if operation not in {"count", "total", "average", "minimum", "maximum"}:
                raise ValueError(
                    "legacy direct/NEXT targets are retired; use aggregate PQL"
                )
            if target_time is None or lookahead is None:
                raise ValueError(
                    "target_time, operation, and lookahead must be provided together"
                )
            if (
                task not in (None, "regression")
                or relationship_path is not None
                or positive_class is not None
                or decision_threshold is not None
            ):
                raise ValueError(
                    "legacy adapter supports aggregate regression only, "
                    "without paths or class options; use PQL"
                )
            if as_of is not None and anchor_time is not None:
                raise ValueError("provide as_of or anchor_time, not both")
        elif any(
            v is not None
            for v in (
                entity,
                target,
                target_time,
                operation,
                lookahead,
                as_of,
                relationship_path,
                entity_ids,
                positive_class,
            )
        ):
            raise ValueError("provide PQL query or legacy target arguments, not both")
        # Invalid PQL skips the time-argument checks, so the run reports the
        # syntax error and its location rather than missing time arguments.
        target_kind = "temporal" if legacy else get_pql_target_kind(query)
        if target_kind == "static":
            from nori_rel._pql_runtime import get_feature_strategy_name

            require_static_target_arguments(
                anchor_times=anchor_times,
                anchor_time=anchor_time,
                observed_through=observed_through,
                feature_strategy=get_feature_strategy_name(
                    feature_strategy
                    if feature_strategy is not None
                    else getattr(self._runtime, "features", "fastdfs"),
                    replaying_program=feature_program is not None,
                ),
                has_pipeline=False,
                entity_time=entity_time,
            )
        elif entity_time is not None:
            if legacy:
                raise ValueError("entity_time needs a PQL query")
            if not isinstance(entity_time, str) or not entity_time.strip():
                raise ValueError("entity_time must name a date/timestamp column")
            if anchor_times is not None:
                raise ValueError(
                    "entity_time anchors each row at its own time; omit anchor_times"
                )
            if entity_created_at not in (None, entity_time):
                raise ValueError(
                    "with entity_time, a row exists from its own time; "
                    "omit entity_created_at or pass the same column"
                )
            if observed_through is None:
                raise ValueError("PQL with entity_time requires observed_through")
            # A row exists from its event time, so it is also the creation column.
            entity_created_at = entity_time
        elif target_kind == "temporal" and (
            anchor_times is None or observed_through is None or not entity_created_at
        ):
            raise ValueError(
                "temporal PQL requires anchor_times, observed_through and "
                "entity_created_at; a static target such as PREDICT users.income "
                "takes none of them"
            )
        if legacy:
            from nori_rel._legacy import aggregate_query

            query = aggregate_query(
                self.discover(source),
                entity=entity,
                target=target,
                target_time=target_time,
                operation=operation,
                lookahead=lookahead,
                entity_ids=entity_ids,
            )
            anchor_time = as_of or anchor_time or datetime.now(UTC)
            warnings.warn(
                "Legacy aggregate arguments are deprecated; use query=PQL. "
                "PQL semantics apply: scoring IDs do not restrict historical context, "
                "all-null SUM/AVG/MIN/MAX remain unknown; results use anchor_time "
                "and PQL metadata.",
                FutureWarning,
                stacklevel=3,
            )

        def predict_frame(run_context: RunContext) -> pd.DataFrame:
            return self._runtime.predict_pql(
                source,
                query,
                anchor_times=anchor_times,
                anchor_time=anchor_time,
                observed_through=observed_through,
                entity_created_at=entity_created_at,
                max_context_rows=(
                    DEFAULT_MAX_CONTEXT_ROWS
                    if max_context_rows is None
                    else max_context_rows
                ),
                context_policy=context,
                agent_instructions=agent_instructions,
                task=task,
                output_type=output_type,
                quantiles=quantiles,
                decision_threshold=decision_threshold,
                log_summary=log_summary,
                query_filter=query_filter,
                feature_program=program,
                run_context=run_context,
                metric=metric,
                keep_scoring_features=keep_scoring_features,
                feature_strategy=feature_strategy,
                expected_context_sha256=expected_context_sha256,
                **({"entity_time": entity_time} if entity_time is not None else {}),
            )

        return predict_frame

    def _check_agent_instructions(
        self,
        agent_instructions: str,
        *,
        feature_strategy: FeatureBackendName | FeatureBackend | AgenticSearch | None,
        replaying_program: bool,
    ) -> None:
        """Refuse agent_instructions that no agent would read.

        The prediction's feature search is ``feature_strategy`` when given, and
        otherwise the client's.

        Raises:
            ValueError: Blank text, a prediction without an agent search, a
                replayed program, or a search that already has a task_hint.
        """
        from nori_rel._pql_runtime import get_feature_strategy_name

        if not isinstance(agent_instructions, str) or not agent_instructions.strip():
            raise ValueError("agent_instructions must be non-blank text")
        features = (
            feature_strategy
            if feature_strategy is not None
            else getattr(self._runtime, "features", "fastdfs")
        )
        if (
            get_feature_strategy_name(features, replaying_program=replaying_program)
            != "agentic"
        ):
            raise ValueError(
                "agent_instructions needs an agent feature search: pass "
                "feature_strategy=AgenticSearch(...). A FastDFS or NoriDFS "
                "run, or a replayed feature_program, sends nothing to an agent"
            )
        if features.task_hint:
            raise ValueError(
                "pass agent_instructions to predict or AgenticSearch(task_hint=...), "
                "not both"
            )

    def _get_pipeline_prediction_run(
        self,
        pipeline: NoriRelPipeline,
        *,
        source: Database,
        request: Mapping[str, Any],
        fixed: Mapping[str, Any],
        log_summary: bool,
        keep_scoring_features: bool = False,
    ) -> Callable[[RunContext], pd.DataFrame]:
        """Check a request to run a saved pipeline; return the run to execute.

        ``fixed`` holds the arguments the pipeline decides, which must be unset.
        ``keep_scoring_features`` keeps the scored rows' feature matrix.
        """
        from nori_rel.pipeline import NoriRelPipeline, PipelinePredictRequest
        from nori_rel.pipeline.checks import get_pipeline_query

        if not isinstance(pipeline, NoriRelPipeline):
            raise TypeError(
                "pipeline must be a NoriRelPipeline; see NoriRelPipeline.load"
            )
        given = sorted(name for name, value in fixed.items() if value is not None)
        if given:
            raise ValueError(
                f"{', '.join(given)} cannot be passed with a pipeline: the "
                "pipeline fixes them. Pass source, as_of, observed_through and "
                "optionally query_filter or a query with another FOR clause"
            )
        if request["as_of"] is None or request["observed_through"] is None:
            raise ValueError(
                "a pipeline prediction requires as_of and observed_through"
            )
        run_request = PipelinePredictRequest(**request)
        manifest = pipeline.manifest
        policy = manifest.context_policy
        prediction = manifest.prediction
        query = get_pipeline_query(manifest, run_request.query)

        def predict_frame(run_context: RunContext) -> pd.DataFrame:
            """Predict at the request's ``as_of`` from the saved pipeline."""
            return self._runtime.predict_pql(
                source,
                query,
                anchor_times=policy.get_anchor_times(run_request.as_of),
                anchor_time=run_request.as_of,
                observed_through=run_request.observed_through,
                entity_created_at=manifest.task.entity_created_at,
                max_context_rows=(
                    DEFAULT_MAX_CONTEXT_ROWS
                    if policy.max_context_rows is None
                    else policy.max_context_rows
                ),
                context_policy=policy.row_policy,
                task=manifest.task.task,
                output_type=prediction.output_type,
                quantiles=(
                    None if prediction.quantiles is None else list(prediction.quantiles)
                ),
                decision_threshold=prediction.decision_threshold,
                log_summary=log_summary,
                query_filter=run_request.query_filter,
                run_context=run_context,
                pipeline=manifest,
                expected_context_sha256=run_request.expected_context_sha256,
                keep_scoring_features=keep_scoring_features,
            )

        return predict_frame
