"""Temporal PQL preparation and explicit-task inference on aligned matrices."""

import logging
import sys
from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import pandas as pd

from nori_rel._pql_features import build_pql_features, build_prediction_rows
from nori_rel.contracts import AgentSearchMetric, PredictionTask, RelationalOutputType
from nori_rel.feature import FeatureBackend
from nori_rel.feature.dfs.noridfs import NoriDFS
from nori_rel.pipeline import (
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
from nori_rel.pql.context_sql import DEFAULT_MAX_CONTEXT_ROWS
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
    entity_created_at: str,
    max_context_rows=DEFAULT_MAX_CONTEXT_ROWS,
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
    hashes of this run's inputs.
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
    entity_id, in attrs[SCORING_FEATURES_ATTR] and the SQL of each FastDFS or
    NoriDFS feature in attrs[SCORING_FEATURE_SQL_ATTR]; both stay out of the
    pipeline.
    feature_strategy is this prediction's feature search; None uses the
    runtime's.
    """
    run_context = run_context or RunContext.unobserved()
    chosen_features = runtime.features if feature_strategy is None else feature_strategy
    # An agent search holds steering from the run's first boundary until its
    # agent delivers it; a DFS run or a saved program rejects steering.
    run_context.steering_enabled = (
        feature_program is None
        and pipeline is None
        and get_feature_strategy_name(chosen_features, replaying_program=False)
        == "agentic"
    )
    if feature_program is not None:
        feature_program.validate_query(query, entity_created_at)
    parts = query_parts(parse_pql(query))
    if parts["aggregate"] is None:
        raise ValueError("PQL inference currently requires a temporal aggregate target")
    resolved = (
        PredictionTask.CLASSIFICATION
        if parts["comparison"] is not None
        else PredictionTask.REGRESSION
    )
    if task is not None and PredictionTask(task) != resolved:
        raise ValueError("task contradicts the PQL target")
    threshold = 0.5 if decision_threshold is None else decision_threshold
    if not 0 < threshold < 1:
        raise ValueError("decision_threshold must lie strictly between 0 and 1")
    if resolved is PredictionTask.CLASSIFICATION:
        if output_type is not None or quantiles is not None:
            raise ValueError("binary PQL does not accept regression output options")
        output = None
    else:
        if decision_threshold is not None:
            raise ValueError("decision_threshold requires binary PQL")
        output = RelationalOutputType(output_type or "median")
        if output is RelationalOutputType.QUANTILES:
            if (
                not quantiles
                or any(not 0 < q < 1 for q in quantiles)
                or list(quantiles) != sorted(set(quantiles))
            ):
                raise ValueError("quantiles must be sorted, unique and between 0 and 1")
        elif quantiles is not None:
            raise ValueError("quantiles requires output_type='quantiles'")
    if metric is not None:
        _require_agent_search_metric(
            metric,
            resolved,
            searching=feature_program is None
            and pipeline is None
            and get_feature_strategy_name(chosen_features, replaying_program=False)
            == "agentic",
        )
    predictor = (
        runtime.predictor
        if pipeline is None
        else runtime.get_pipeline_predictor(pipeline.model)
    )
    model = predictor.model_for(resolved)
    if pipeline is not None:
        warn_if_other_nori_rel(pipeline)
    if log_summary:
        logger.info("PQL task=%s; target=%s", resolved.value, parts["target"].getText())
    connector = runtime._connector(source)
    with run_context.report_stage(Stage.MATERIALIZE):
        snapshot, context, counts = connector.prepare_inputs(
            source.id,
            query,
            anchor_times=anchor_times,
            anchor_time=anchor_time,
            observed_through=observed_through,
            entity_created_at=entity_created_at,
            max_context_rows=max_context_rows,
            query_filter=query_filter,
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
        scoring = build_prediction_rows(snapshot, query, entity_created_at)
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
    }
    if snapshot.provenance:
        metadata["snapshot"] = snapshot.provenance.model_dump(mode="json")
    if log_summary:
        logger.info(
            "PQL context=%d usable/%d candidates; scoring=%d",
            counts["usable_rows"],
            counts["candidate_rows"],
            len(scoring),
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
    context_hash = get_context_rows_sha256(context, context_labels)
    require_expected_context(expected_context_sha256, context_hash)
    if pipeline is None:
        backend = runtime.feature_backend(
            target_window(parts["aggregate"]),
            resolved,
            context,
            labels,
            feature_program,
            run_context=run_context,
            metric=metric,
            feature_strategy=chosen_features,
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
    if resolved is PredictionTask.CLASSIFICATION:
        if ((values < 0) | (values > 1)).any():
            raise ValueError("Nori returned probabilities outside [0, 1]")
        result["prediction"] = values >= threshold
        result["probability"] = values
    else:
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
        from nori_rel._feature_search import agentic_summary

        result.attrs.update(
            agentic_summary(
                backend,
                query=query,
                entity_created_at=entity_created_at,
                schema=snapshot.schema,
            )
        )
    if pipeline is None:
        pipeline = get_pipeline_manifest(
            engine=source.engine,
            query=query,
            entity_created_at=entity_created_at,
            task=resolved,
            as_of=scoring["anchor_time"].iloc[0],
            anchor_times=anchor_times,
            max_context_rows=max_context_rows,
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
        result.attrs[SCORING_FEATURE_SQL_ATTR] = get_feature_sql_by_name(
            backend, matrices.feature_names
        )
    return result


def get_feature_sql_by_name(
    backend: FeatureBackend, feature_names: Sequence[str]
) -> dict[str, str]:
    """Return the SQL behind each of ``feature_names`` that has one, by name.

    FastDFS reports the statements it ran for each feature. NoriDFS reports a
    DuckDB query per feature that gives the values its pandas executor
    computed, and an agent search reports those of its NoriDFS baseline. An
    agent's own SQL is already in the prediction's ``agentic_program``
    metadata. Any other backend reports none.
    """
    fastdfs = sys.modules.get("nori_rel.feature.dfs.fastdfs")
    if fastdfs is not None and isinstance(backend, fastdfs.FastDFS):
        sql_by_name = backend.last_feature_sql
    else:
        agentic = sys.modules.get("nori_rel.feature.agentic.backend")
        if agentic is not None and isinstance(backend, agentic.AgenticFeatures):
            backend = backend.fallback
        sql_by_name = backend.last_feature_sql if isinstance(backend, NoriDFS) else {}
    return {name: sql_by_name[name] for name in feature_names if name in sql_by_name}


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
