"""Temporal PQL preparation and explicit-task inference on aligned matrices."""

import logging
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

from nori_rel._context_selection import (
    get_probabilities_with_population_odds,
    get_selected_context,
)
from nori_rel._pql_features import build_pql_features, build_prediction_rows
from nori_rel._pql_static import (
    READ_TIME_STAND_IN,
    get_known_target_values,
    get_static_target_task,
    require_static_target_arguments,
)
from nori_rel.connectors.base import get_dropped_foreign_keys
from nori_rel.context_policy import ContextPolicy, ContextPolicyChooser
from nori_rel.contracts import AgentSearchMetric, PredictionTask, RelationalOutputType
from nori_rel.feature import FeatureBackend
from nori_rel.pipeline import (
    CONTEXT_FEATURES_ATTR,
    CONTEXT_LABELS_ATTR,
    PIPELINE_ATTR,
    SCORING_FEATURE_SQL_ATTR,
    SCORING_FEATURES_ATTR,
)
from nori_rel.pipeline.audit import get_context_rows_sha256, get_feature_matrix_sha256
from nori_rel.pipeline.build import get_pipeline_manifest
from nori_rel.pipeline.checks import (
    require_expected_context,
    require_same_columns,
    require_same_schema,
    warn_if_other_nori_rel,
)
from nori_rel.pipeline.features import (
    get_saved_feature_backend,
    get_tables_with_saved_dtypes,
)
from nori_rel.pql import parse_pql
from nori_rel.pql.context_sql import (
    DEFAULT_MAX_CONTEXT_ROWS,
    get_eligible_anchor_times,
)
from nori_rel.pql.tree_access import query_parts
from nori_rel.pql.window import target_window
from nori_rel.run import RunContext, Stage

logger = logging.getLogger(__name__)


def _require_agent_search_metric(
    metric: AgentSearchMetric, task: PredictionTask, *, searching: bool
) -> None:
    """Raise ``ValueError`` unless ``metric`` can score this run's agent search.

    A metric applies only when an agent search runs, on a regression target:
    binary searches score ROC-AUC, and FastDFS, NoriDFS, a saved program or a
    pipeline score nothing.
    """
    from nori_rel.feature.agentic.scoring import require_agent_search_metric

    require_agent_search_metric(metric)
    if task is PredictionTask.CLASSIFICATION:
        raise ValueError(
            "metric applies to regression targets; binary agent searches score ROC-AUC"
        )
    if not searching:
        raise ValueError(
            "metric chooses how an agent search scores its feature programs; "
            "pass feature_strategy=AgenticSearch.claude(...) or "
            "AgenticSearch.openai(...) to predict, and do not pass feature_program "
            "or pipeline"
        )


def predict_pql(
    runtime,
    source,
    query: str,
    *,
    anchor_times,
    anchor_time,
    observed_through,
    entity_created_at: str | None,
    max_context_rows=DEFAULT_MAX_CONTEXT_ROWS,
    context_policy: ContextPolicy | None = None,
    agent_instructions: str | None = None,
    task=None,
    output_type=None,
    quantiles=None,
    decision_threshold=None,
    log_summary=True,
    query_filter=None,
    feature_program=None,
    run_context: RunContext | None = None,
    pipeline=None,
    expected_context_sha256=None,
    metric: AgentSearchMetric | None = None,
    keep_scoring_features: bool = False,
    feature_strategy=None,
    entity_time: str | None = None,
) -> pd.DataFrame:
    """Run temporal PQL through preparation, features and Nori inference.

    runtime supplies connector_factory, feature_backend, predictor and limits.
    Temporal options match connector.prepare_inputs. Predicates resolve to
    binary classification; numeric aggregates resolve to regression. An explicit
    task must agree. Returns entity_id, anchor_time, prediction; binary adds
    probability and quantile output adds quantiles. attrs contains task, model,
    feature_count and context counts, plus the feature backend and, for an
    agent search, its usage, cost, stop reason, trial count and winning program.
    attrs["pipeline"] is the prediction's pipeline record and
    attrs["pipeline_id"] its identifier; both are None without a pipeline.
    attrs also holds context_rows_sha256 and feature_matrix_sha256, the audit
    hashes of this run's inputs, and dropped_foreign_keys, the keys from a
    selected table to one the source did not select, which no feature follows.
    A saved feature_program replays on NoriDFS instead of
    the runtime's feature search. A ``PipelineManifest`` in ``pipeline``
    replaces the feature search and the model with its saved ones: the caller
    passes the context anchors and limit the manifest's policy gives for this
    run, and the run checks the schema and the feature columns, raising a
    ``PipelineError`` before Nori runs. Context rows are not compared with the
    pipeline's first run; with ``expected_context_sha256`` the run raises
    unless its context rows hash to that value. No silent context truncation
    or refill. run_context receives stage and search events and applies the
    run's commands between stages; without one, nothing is reported.
    metric chooses how an agent search scores its candidate programs on a
    regression target ("mae" when None); it is refused for any other run.
    keep_scoring_features puts the scored rows' feature matrix, indexed by
    entity_id, in attrs[SCORING_FEATURES_ATTR] and the SQL of each feature,
    for FastDFS, in attrs[SCORING_FEATURE_SQL_ATTR], the labeled context rows
    (entity_id, anchor_time, label) in attrs[CONTEXT_LABELS_ATTR], and their
    feature matrix, the one Nori was fitted on, in attrs[CONTEXT_FEATURES_ATTR]
    in the same row order; all four stay out of the pipeline.
    feature_strategy is this prediction's feature search; None uses the
    runtime's.

    context_policy chooses the context among every candidate with a known
    label, in place of the newest max_context_rows candidates; balanced
    probabilities and 0/1 means are mapped back to the population rate.
    attrs["context_policy"] is the ContextPolicyReport of either rule: the
    policy and who chose it, class counts before and after selection, rows per
    cutoff and, for an agent search, how the agent split the rows.
    agent_instructions is sent verbatim in an agent search's request, and the
    report records it with the stages that received it.

    entity_time names a date/timestamp column of the entity table for
    event-time rows: each entity row is one example anchored at its own time,
    for context and for scoring, and its features read only rows from before
    that time plus its own columns. anchor_times must then be None, and the
    report groups its cutoffs by UTC day. Agent search, saved programs and
    pipelines do not support event-time rows yet; no pipeline is recorded.

    A static target (PREDICT users.income FOR EACH users.user_id) learns a
    stored column from the other rows whose value is stored. It takes no
    anchor_times, anchor_time or observed_through, and entity_created_at is
    optional: without it every row's features use the read time, with it each
    row's own time. FOR EACH scores the rows whose value is NULL; scored rows
    never join the context. Without entity_created_at, the read time is left
    out of the random context keys and of context_rows_sha256, so a seeded
    ContextPolicy and expected_context_sha256 give the same rows on every
    run of an unchanged source. The target column is removed from the tables
    before features run. Its task follows the column (Boolean or a comparison
    is binary, numeric is regression; a 0/1 numeric column needs task=). It
    uses FastDFS or a custom backend, and makes no saved pipeline.
    """
    run_context = run_context or RunContext.unobserved()
    chosen_features = runtime.features if feature_strategy is None else feature_strategy
    if entity_time is not None:
        if feature_program is not None or pipeline is not None:
            raise ValueError(
                "entity_time does not support saved feature programs or pipelines yet"
            )
        if get_feature_strategy_name(chosen_features, replaying_program=False) == (
            "agentic"
        ):
            raise ValueError("entity_time does not support the agent search yet")
    # An agent search holds steering from the run's first boundary until its
    # agent delivers it; a DFS run or a saved program rejects steering.
    run_context.steering_enabled = (
        feature_program is None
        and pipeline is None
        and get_feature_strategy_name(chosen_features, replaying_program=False)
        == "agentic"
    )
    tree = parse_pql(query)
    parts = query_parts(tree)
    is_static = parts["aggregate"] is None
    # Without entity_created_at, every row of a static target is cut off at
    # the read time, which differs on every run.
    is_read_time_cutoff = is_static and entity_created_at is None
    if is_static:
        require_static_target_arguments(
            anchor_times=anchor_times,
            anchor_time=anchor_time,
            observed_through=observed_through,
            feature_strategy=get_feature_strategy_name(
                chosen_features, replaying_program=feature_program is not None
            ),
            has_pipeline=pipeline is not None,
            entity_time=entity_time,
        )
    elif feature_program is not None:
        feature_program.validate_query(query, entity_created_at)
    threshold = 0.5 if decision_threshold is None else decision_threshold
    if not 0 < threshold < 1:
        raise ValueError("decision_threshold must lie strictly between 0 and 1")
    predictor = (
        runtime.predictor
        if pipeline is None
        else runtime.get_pipeline_predictor(pipeline.model)
    )
    # A static target's task depends on its column type and values, so it is
    # resolved once the source has been read; the options that are wrong
    # whatever the column holds fail here, before the read.
    resolved = output = model = None
    if is_static:
        require_valid_static_target_options(
            has_comparison=parts["comparison"] is not None,
            task=task,
            output_type=output_type,
            quantiles=quantiles,
            decision_threshold=decision_threshold,
            context_policy=context_policy,
        )
    else:
        resolved = (
            PredictionTask.CLASSIFICATION
            if parts["comparison"] is not None
            else PredictionTask.REGRESSION
        )
        if task is not None and PredictionTask(task) != resolved:
            raise ValueError("task contradicts the PQL target")
        output = get_checked_output_type(
            resolved,
            output_type=output_type,
            quantiles=quantiles,
            decision_threshold=decision_threshold,
            context_policy=context_policy,
        )
        model = predictor.model_for(resolved)
    if metric is not None:
        _require_agent_search_metric(
            metric,
            resolved,
            searching=feature_program is None
            and pipeline is None
            and get_feature_strategy_name(chosen_features, replaying_program=False)
            == "agentic",
        )
    if pipeline is not None:
        warn_if_other_nori_rel(pipeline)
    if log_summary and resolved is not None:
        logger.info("PQL task=%s; target=%s", resolved.value, parts["target"].getText())
    connector = runtime._connector(source)
    # Only a policy's own SQL is passed, so custom connectors need no new option.
    candidate_options = (
        {"candidate_sql": context_policy.candidate_sql}
        if context_policy is not None and context_policy.candidate_sql is not None
        else {}
    )
    if entity_time is not None:
        candidate_options["entity_time"] = entity_time
    with run_context.report_stage(Stage.MATERIALIZE):
        snapshot, context, counts = connector.prepare_inputs(
            source.id,
            query,
            anchor_times=anchor_times,
            anchor_time=anchor_time,
            observed_through=observed_through,
            entity_created_at=entity_created_at,
            # None reads every candidate that can have a known label, so the
            # policy below chooses among labelled rows only.
            max_context_rows=None if context_policy is not None else max_context_rows,
            query_filter=query_filter,
            **candidate_options,
        )
        context, context_report = get_selected_context(
            context,
            context_policy,
            chosen_by=get_context_policy_chooser(context_policy, pipeline),
            max_context_rows=max_context_rows,
            # Event-time rows have one cutoff each; the report counts them by day.
            # A static target's rows share the read time, or each has its own
            # creation time.
            eligible_anchor_times=(
                []
                if entity_time is not None
                else ([snapshot.as_of] if entity_created_at is None else [])
                if is_static
                else get_eligible_anchor_times(
                    tree,
                    anchor_times=anchor_times,
                    anchor_time=snapshot.as_of,
                    observed_through=observed_through,
                )
            ),
            max_kept_rows=runtime.limits.max_context_rows,
            cutoff_frequency="D" if entity_time is not None else None,
            random_key_anchor_time=(
                READ_TIME_STAND_IN if is_read_time_cutoff else None
            ),
        )
        if feature_program is not None:
            feature_program.validate_schema(snapshot.schema)
        if pipeline is not None:
            require_same_schema(pipeline.source, source.engine, snapshot.schema)
            snapshot = replace(
                snapshot,
                tables=get_tables_with_saved_dtypes(
                    snapshot.tables, pipeline.source.column_dtypes
                ),
            )
        scoring = build_prediction_rows(
            snapshot, query, entity_created_at, entity_time=entity_time
        )
    if is_static:
        resolved = get_static_target_task(
            tree, snapshot.schema, task, get_known_target_values(snapshot, tree)
        )
        output = get_checked_output_type(
            resolved,
            output_type=output_type,
            quantiles=quantiles,
            decision_threshold=decision_threshold,
            context_policy=context_policy,
        )
        model = predictor.model_for(resolved)
        if log_summary:
            logger.info(
                "PQL task=%s; static target=%s",
                resolved.value,
                parts["target"].getText(),
            )
    metadata = {
        "task": resolved.value,
        "model": model,
        "features": (
            get_pipeline_strategy_name(pipeline.features)
            if pipeline is not None
            else get_feature_strategy_name(
                chosen_features, replaying_program=feature_program is not None
            )
        ),
        **counts,
        "feature_count": 0,
        "context_policy": context_report.model_dump(mode="json"),
        "dropped_foreign_keys": get_dropped_foreign_keys(
            snapshot.schema, snapshot.tables
        ),
    }
    if snapshot.provenance:
        metadata["snapshot"] = snapshot.provenance.model_dump(mode="json")
    if entity_time is not None:
        metadata["entity_time"] = entity_time
    if log_summary:
        logger.info(
            "PQL context=%d usable/%d candidates; scoring=%d",
            counts["usable_rows"],
            counts["candidate_rows"],
            len(scoring),
        )
        if context_policy is not None:
            logger.info(
                "PQL context policy kept %d of %d labelled rows over %d cutoffs",
                len(context),
                counts["usable_rows"],
                sum(cutoff.kept_rows > 0 for cutoff in context_report.cutoffs),
            )
    if scoring.empty:
        columns = ["entity_id", "anchor_time", "prediction"]
        if resolved is PredictionTask.CLASSIFICATION:
            columns.append("probability")
        elif output is RelationalOutputType.QUANTILES:
            columns.append("quantiles")
        result = pd.DataFrame(columns=columns)
        result.attrs.update(
            {
                **metadata,
                "pipeline_id": None if pipeline is None else pipeline.pipeline_id,
                PIPELINE_ATTR: None
                if pipeline is None
                else pipeline.model_dump(mode="json"),
            }
        )
        return result
    if len(context) < runtime.limits.min_training_rows:
        raise ValueError(
            f"only {len(context)} usable context rows; "
            f"at least {runtime.limits.min_training_rows} required"
        )
    labels = context.set_index("context_row_id")["label"]
    if resolved is PredictionTask.CLASSIFICATION and set(labels) != {False, True}:
        raise ValueError("binary PQL requires both false and true historical labels")
    labels = pd.to_numeric(labels, errors="raise").astype(float)
    if not np.isfinite(labels).all():
        raise ValueError("context labels must be finite numeric values")
    context_labels = labels
    # The hash leaves out a shared read time, so an unchanged source hashes
    # the same on every run and expected_context_sha256 can reproduce it.
    context_hash = get_context_rows_sha256(
        context.assign(anchor_time=READ_TIME_STAND_IN)
        if is_read_time_cutoff
        else context,
        context_labels,
    )
    require_expected_context(expected_context_sha256, context_hash)
    if pipeline is None:
        backend = runtime.feature_backend(
            None if is_static else target_window(parts["aggregate"]),
            resolved,
            context,
            labels,
            feature_program,
            run_context=run_context,
            metric=metric,
            feature_strategy=chosen_features,
            agent_instructions=agent_instructions,
        )
    else:
        backend = get_saved_feature_backend(pipeline.features)
    with run_context.report_stage(Stage.PLAN_SEARCH):
        matrices, scoring = build_pql_features(
            snapshot, context, query, scoring, backend
        )
    if pipeline is not None:
        require_same_columns(pipeline.columns, matrices.feature_names)
    features = len(matrices.feature_names)
    if features == 0 or len(context) * features > runtime.limits.max_feature_elements:
        raise ValueError(
            "context exceeds the model feature-element budget; reduce max_context_rows"
        )
    labels = labels.loc[matrices.train.index]
    declared = (
        backend.categorical_columns()
        if pipeline is None
        else pipeline.columns.categorical
    )
    with run_context.report_stage(Stage.PREDICT_BATCH):
        predictions, bands = predictor.predict_arrays(
            matrices.train.reset_index(drop=True),
            labels.reset_index(drop=True),
            matrices.query.reset_index(drop=True),
            task=resolved,
            output_type=output,
            quantiles=quantiles,
            categorical_columns=None if declared is None else list(declared),
        )
    values = np.asarray(predictions, dtype=float)
    if values.shape != (len(scoring),) or not np.isfinite(values).all():
        raise ValueError("Nori returned invalid predictions or the wrong row count")
    result = scoring[["entity_id", "anchor_time"]].copy()
    correction_factor = context_report.probability_correction_factor
    if resolved is PredictionTask.CLASSIFICATION:
        if ((values < 0) | (values > 1)).any():
            raise ValueError("Nori returned probabilities outside [0, 1]")
        if correction_factor is not None:
            values = get_probabilities_with_population_odds(values, correction_factor)
        result["prediction"] = values >= threshold
        result["probability"] = values
    else:
        if correction_factor is not None:
            values = get_probabilities_with_population_odds(values, correction_factor)
        result["prediction"] = values
        if output is RelationalOutputType.QUANTILES:
            array = np.asarray(bands, dtype=float)
            if (
                array.shape != (len(scoring), len(quantiles))
                or not np.isfinite(array).all()
            ):
                raise ValueError("Nori returned invalid quantile rows")
            result["quantiles"] = list(bands)
    result.attrs.update(
        {
            **metadata,
            "feature_count": features,
            "context_rows_sha256": context_hash,
            "feature_matrix_sha256": get_feature_matrix_sha256(matrices),
        }
    )
    if metadata["features"] in ("agentic", "saved_program"):
        from nori_rel._feature_search import (
            agentic_summary,
            get_agent_search_split_report,
        )

        result.attrs.update(
            agentic_summary(
                backend,
                query=query,
                entity_created_at=entity_created_at,
                schema=snapshot.schema,
            )
        )
        split_report = get_agent_search_split_report(backend)
        report_updates = {}
        if split_report is not None:
            report_updates["agent_search_split"] = split_report
        if agent_instructions is not None:
            # A search session exists only once its agent has been asked.
            report_updates["agent_instructions"] = agent_instructions
            report_updates["agent_instruction_stages"] = (
                ("feature_search",) if split_report is not None else ()
            )
        if report_updates:
            context_report = context_report.model_copy(update=report_updates)
            result.attrs["context_policy"] = context_report.model_dump(mode="json")
    if pipeline is None and entity_time is None and not is_static:
        pipeline = get_pipeline_manifest(
            engine=source.engine,
            query=query,
            entity_created_at=entity_created_at,
            task=resolved,
            as_of=scoring["anchor_time"].iloc[0],
            anchor_times=anchor_times,
            max_context_rows=max_context_rows,
            context_policy=context_policy,
            schema=snapshot.schema,
            tables=snapshot.tables,
            backend=backend,
            matrices=matrices,
            categorical_columns=declared,
            predictor=predictor,
            output_type=output,
            quantiles=quantiles,
            decision_threshold=(
                threshold if resolved is PredictionTask.CLASSIFICATION else None
            ),
            metadata=dict(result.attrs),
        )
    result.attrs["pipeline_id"] = None if pipeline is None else pipeline.pipeline_id
    result.attrs[PIPELINE_ATTR] = (
        None if pipeline is None else pipeline.model_dump(mode="json")
    )
    if pipeline is not None:
        run_context.report_pipeline(
            pipeline.pipeline_id,
            context_sha256=context_hash,
            features_sha256=result.attrs["feature_matrix_sha256"],
        )
    if keep_scoring_features:
        result.attrs[SCORING_FEATURES_ATTR] = get_scoring_feature_frame(
            matrices.query, scoring
        )
        result.attrs[SCORING_FEATURE_SQL_ATTR] = get_feature_sql_by_name(backend)
        result.attrs[CONTEXT_LABELS_ATTR] = get_context_label_frame(context, labels)
        result.attrs[CONTEXT_FEATURES_ATTR] = matrices.train.reset_index(drop=True)
    return result


def get_feature_sql_by_name(backend: FeatureBackend) -> dict[str, str]:
    """Return the SQL that computed each feature column, by column name.

    Only FastDFS reports it, since it runs one query per feature; the result is
    empty for any other backend. NoriDFS computes in pandas, and an agent's
    SQL is already in the prediction's ``agentic_program`` metadata.
    """
    fastdfs = sys.modules.get("nori_rel.feature.dfs.fastdfs")
    if fastdfs is not None and isinstance(backend, fastdfs.FastDFS):
        return dict(backend.last_feature_sql)
    return {}


def get_scoring_feature_frame(
    query_matrix: pd.DataFrame, scoring: pd.DataFrame
) -> pd.DataFrame:
    """Return the scored rows' feature matrix with one row per entity_id.

    ``query_matrix`` is indexed by query_row_id, as build_pql_features returns
    it; ``scoring`` maps each query_row_id to its entity_id.
    """
    entity_ids = scoring.set_index("query_row_id")["entity_id"]
    features = query_matrix.copy()
    features.index = pd.Index(
        entity_ids.reindex(query_matrix.index).to_numpy(), name="entity_id"
    )
    return features


def get_context_label_frame(context: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Return entity_id, anchor_time and the label Nori read for each context row.

    ``labels`` is indexed by context_row_id, in the row order of the matrix
    Nori was fitted on, and holds the numeric values the model was given, so a
    database's decimal or boolean labels arrive as floats. The frame follows
    that order, so its rows line up with the fitted matrix.
    """
    rows = context.set_index("context_row_id").loc[labels.index]
    frame = rows[["entity_id", "anchor_time"]].infer_objects().reset_index(drop=True)
    frame["anchor_time"] = pd.to_datetime(frame["anchor_time"], utc=True)
    frame["label"] = labels.to_numpy(dtype=float)
    return frame


def get_checked_output_type(
    resolved: PredictionTask,
    *,
    output_type,
    quantiles,
    decision_threshold,
    context_policy: ContextPolicy | None,
) -> RelationalOutputType | None:
    """Return the regression output type, or None for binary, after checks.

    Binary PQL takes no output_type or quantiles; regression takes no
    decision_threshold, defaults to the median, and needs sorted, unique
    quantiles in (0, 1) for output_type='quantiles'. A balanced ContextPolicy
    that corrects a 0/1 regression label needs output_type='mean'.
    """
    if resolved is PredictionTask.CLASSIFICATION:
        if output_type is not None or quantiles is not None:
            raise ValueError("binary PQL does not accept regression output options")
        return None
    if decision_threshold is not None:
        raise ValueError("decision_threshold requires binary PQL")
    output = RelationalOutputType(output_type or "median")
    require_valid_quantiles(output, quantiles)
    if (
        context_policy is not None
        and context_policy.balance is not None
        and context_policy.correct_probabilities
        and output is not RelationalOutputType.MEAN
    ):
        raise ValueError(
            "a balanced ContextPolicy maps a 0/1 regression label back to "
            "its population rate only for output_type='mean'; pass "
            "output_type='mean' or correct_probabilities=False"
        )
    return output


def require_valid_quantiles(output: RelationalOutputType, quantiles) -> None:
    """Raise ValueError for quantiles that do not fit the regression output.

    output_type='quantiles' needs a sorted, unique list in (0, 1); any other
    output type takes no quantiles.
    """
    if output is RelationalOutputType.QUANTILES:
        if (
            not quantiles
            or any(not 0 < q < 1 for q in quantiles)
            or list(quantiles) != sorted(set(quantiles))
        ):
            raise ValueError("quantiles must be sorted, unique and between 0 and 1")
    elif quantiles is not None:
        raise ValueError("quantiles requires output_type='quantiles'")


def require_valid_static_target_options(
    *,
    has_comparison: bool,
    task,
    output_type,
    quantiles,
    decision_threshold,
    context_policy: ContextPolicy | None,
) -> None:
    """Raise ValueError, before any read, for static-target options that are wrong.

    A comparison is always binary, and so is task='classification': their
    options are checked in full, as get_checked_output_type does. Otherwise
    the task string, the output type and the quantile list are checked, and
    only the rules that depend on the column (a Boolean or 0/1 column is
    binary) wait for the read.
    """
    requested = None if task is None else PredictionTask(task)
    if has_comparison and requested is PredictionTask.REGRESSION:
        raise ValueError("task contradicts the PQL target")
    known = PredictionTask.CLASSIFICATION if has_comparison else requested
    if known is not None:
        get_checked_output_type(
            known,
            output_type=output_type,
            quantiles=quantiles,
            decision_threshold=decision_threshold,
            context_policy=context_policy,
        )
        return
    require_valid_quantiles(RelationalOutputType(output_type or "median"), quantiles)


def get_context_policy_chooser(
    context_policy: ContextPolicy | None, pipeline
) -> ContextPolicyChooser:
    """Return who chose a prediction's context rule, as its report names it.

    ``pipeline`` for a saved pipeline's rule, ``caller`` for a policy passed
    to ``predict``, and ``default`` for the newest-candidates rule.
    """
    if pipeline is not None:
        return "pipeline"
    return "default" if context_policy is None else "caller"


def get_feature_strategy_name(features, *, replaying_program: bool) -> str:
    """Return the feature strategy a prediction uses, as metadata and runs name it.

    ``saved_program`` for a replayed program; otherwise the backend name
    (``fastdfs``, ``noridfs``), a custom backend's class name, or ``agentic``
    for an ``AgenticSearch``. It imports nothing, so the client can name a
    run's strategy without loading the agent search.
    """
    if replaying_program:
        return "saved_program"
    if isinstance(features, str):
        return features
    return (
        type(features).__name__ if isinstance(features, FeatureBackend) else "agentic"
    )


def get_pipeline_strategy_name(features) -> str:
    """Return the run strategy of a saved pipeline's features.

    The names are the ones run control and prediction metadata use:
    ``noridfs`` and ``fastdfs`` for those backends, and ``saved_program`` for
    an agent's SQL program replayed on its NoriDFS plan.
    """
    return "saved_program" if features.kind == "agentic" else features.kind
