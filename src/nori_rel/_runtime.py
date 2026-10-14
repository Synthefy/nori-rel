"""In-process relational execution with hosted or local Nori inference."""

from __future__ import annotations

import importlib.metadata
import json
import pickle
import threading
import uuid
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import pandas as pd
from synthefy.nori_client import SynthefyNoriClient

from nori_rel._checkpoint import NoriCheckpoint, get_pinned_checkpoint
from nori_rel.connectors import create_connector
from nori_rel.connectors.base import RelationalConnector
from nori_rel.contracts import (
    ConnectionStatus,
    Database,
    DatabaseCreateRequest,
    DatabaseStatus,
    PredictionTask,
    RelationalOutputType,
    SchemaGraph,
)
from nori_rel.feature import FEATURE_BACKENDS, FeatureBackend, FeatureBackendName
from nori_rel.feature.dfs.fastdfs import FastDFS
from nori_rel.limits import RelationalLimits
from nori_rel.pipeline.errors import PipelineModelError
from nori_rel.pql.window import TargetWindow
from nori_rel.run import RunContext

if TYPE_CHECKING:
    from nori_rel._feature_program import SavedFeatureProgram
    from nori_rel._feature_search import AgenticSearch
    from nori_rel.pipeline.manifest import HostedModelRecord, LocalModelRecord


class Predictor(Protocol):
    def model_for(self, task: PredictionTask) -> str: ...

    def predict_arrays(
        self,
        train: pd.DataFrame,
        target: pd.Series,
        query: pd.DataFrame,
        *,
        task: PredictionTask,
        output_type: RelationalOutputType | None = None,
        quantiles: list[float] | None = None,
        categorical_columns: list[str] | None = None,
    ) -> tuple[list[float | None], list[list[float | None]] | None]: ...


class HostedNoriPredictor:
    """Send only flat feature matrices to the hosted Nori endpoint."""

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        classification_model: str | None = None,
        timeout: float = 300.0,
        max_retries: int = 2,
    ) -> None:
        """Keep the key, the model names and the request limits."""
        self.api_key = api_key
        self.regression_model = model
        self.classification_model = classification_model or model
        self.timeout = timeout
        self.max_retries = max_retries

    def model_for(self, task: PredictionTask) -> str:
        if task is PredictionTask.CLASSIFICATION:
            return self.classification_model
        return self.regression_model

    def get_pipeline_model_record(self) -> HostedModelRecord:
        """Record the hosted model names; the server's weights are not pinned."""
        from nori_rel.pipeline.manifest import HostedModelRecord

        return HostedModelRecord(
            model=self.regression_model,
            classification_model=self.classification_model,
            synthefy_version=importlib.metadata.version("synthefy"),
        )

    def predict_arrays(
        self,
        train: pd.DataFrame,
        target: pd.Series,
        query: pd.DataFrame,
        *,
        task: PredictionTask,
        output_type: RelationalOutputType | None = None,
        quantiles: list[float] | None = None,
        categorical_columns: list[str] | None = None,
    ) -> tuple[list[float | None], list[list[float | None]] | None]:
        """Send aligned feature rows/labels with explicit task and output options.

        Input matrices have identical feature columns; target aligns with train.
        Returns one point/probability per query row and optional quantile rows.
        No query AST or source tables reach Nori. ``categorical_columns``, when
        given, is declared to the SDK, which encodes those columns locally;
        ``None`` keeps the SDK's automatic detection.
        """
        declared = (
            {}
            if categorical_columns is None
            else {"categorical_columns": categorical_columns}
        )
        with SynthefyNoriClient(
            api_key=self.api_key,
            model=self.model_for(task),
            timeout=self.timeout,
            max_retries=self.max_retries,
        ) as client:
            if task is PredictionTask.CLASSIFICATION:
                result = client.predict(
                    train,
                    target,
                    query,
                    task="classification",
                    as_pandas=True,
                    **declared,
                )
                if not isinstance(result, pd.Series):
                    raise TypeError("Nori returned an invalid classification response")
                probabilities = result.astype(float).to_numpy()
                if (
                    not np.isfinite(probabilities).all()
                    or ((probabilities < 0) | (probabilities > 1)).any()
                ):
                    raise ValueError("Nori returned invalid class probabilities")
                return probabilities.tolist(), None

            if output_type is None:
                raise RuntimeError("regression output type is missing")
            result = client.predict(
                train,
                target,
                query,
                output_type=output_type.value,
                quantiles=quantiles,
                as_pandas=True,
                **declared,
            )

        if output_type is RelationalOutputType.QUANTILES:
            if not isinstance(result, pd.DataFrame):
                raise TypeError("Nori returned an invalid quantile response")
            rows = result.astype(float).replace({np.nan: None}).values.tolist()
            levels = quantiles or []
            median_index = min(
                range(len(levels)), key=lambda index: abs(levels[index] - 0.5)
            )
            return [row[median_index] for row in rows], rows
        if not isinstance(result, pd.Series):
            raise TypeError("Nori returned an invalid point response")
        return result.astype(float).replace({np.nan: None}).tolist(), None


class LocalNoriPredictor:
    """Run Nori in this process with the public NoriRegressor.

    Feature matrices, labels and predictions never leave the process. Only
    regression targets are supported. The checkpoint downloads on first use
    and is then read from the Hugging Face cache, including with
    HF_HUB_OFFLINE=1. ``revision`` and ``sha256`` pin the file, and
    ``regressor_params`` the recorded ``NoriRegressor`` settings, as a saved
    pipeline does. Without them the latest revision loads, and ``checkpoint``
    records which file it was.
    """

    def __init__(
        self,
        model: str,
        *,
        device: str | None = None,
        revision: str | None = None,
        sha256: str | None = None,
        regressor_params: Mapping[str, Any] | None = None,
    ) -> None:
        try:
            from synthefy_nori import NoriRegressor
        except ImportError as exc:
            raise ImportError(
                'mode="local" requires `pip install synthefy-nori==0.20.1`'
            ) from exc
        self.model = model
        self.device = device
        self.revision = revision
        self.expected_sha256 = sha256
        self.expected_regressor_params = regressor_params
        #: The file the regressor loads, known after the first prediction.
        self.checkpoint: NoriCheckpoint | None = None
        self._regressor_class = NoriRegressor
        self._regressor: Any = None
        self._loaded = False
        self._lock = threading.Lock()

    def model_for(self, task: PredictionTask) -> str:
        if task is PredictionTask.CLASSIFICATION:
            raise ValueError(
                "mode='local' supports regression PQL targets only; binary "
                "predicate targets require mode='hosted'"
            )
        return self.model

    def predict_arrays(
        self,
        train: pd.DataFrame,
        target: pd.Series,
        query: pd.DataFrame,
        *,
        task: PredictionTask,
        output_type: RelationalOutputType | None = None,
        quantiles: list[float] | None = None,
        categorical_columns: list[str] | None = None,
    ) -> tuple[list[float | None], list[list[float | None]] | None]:
        """Fit on context rows and predict query rows in this process.

        ``categorical_columns``, when given, is declared to the regressor for
        this call only; ``None`` leaves its automatic detection untouched.
        """
        self.model_for(task)
        if output_type is None:
            raise RuntimeError("regression output type is missing")
        with self._lock:
            regressor = self._get_pinned_regressor()
            with _Declared(regressor, categorical_columns):
                try:
                    values = regressor.fit(train, target).predict(
                        query,
                        output_type=output_type.value,
                        **(
                            {"quantiles": quantiles}
                            if output_type is RelationalOutputType.QUANTILES
                            else {}
                        ),
                    )
                except (OSError, EOFError, pickle.UnpicklingError, RuntimeError) as exc:
                    # Weights load on the first predict; later failures are not
                    # checkpoint problems.
                    if self._loaded or (
                        isinstance(exc, RuntimeError)
                        and "PytorchStreamReader" not in str(exc)
                    ):
                        raise
                    raise RuntimeError(self._get_weights_error_message()) from exc
            self._loaded = True
        if output_type is RelationalOutputType.QUANTILES:
            levels = quantiles or []
            rows = np.asarray(values, dtype=float).T.tolist()
            median_index = min(
                range(len(levels)), key=lambda index: abs(levels[index] - 0.5)
            )
            return [row[median_index] for row in rows], rows
        return np.asarray(values, dtype=float).tolist(), None

    def _get_pinned_regressor(self) -> Any:
        """Find the checkpoint file once, check its hash, and load from it."""
        if self._regressor is not None:
            return self._regressor
        try:
            checkpoint = get_pinned_checkpoint(self.model, self.revision)
        except OSError as exc:
            raise RuntimeError(self._get_weights_error_message()) from exc
        if (
            self.expected_sha256 is not None
            and checkpoint.sha256 != self.expected_sha256
        ):
            raise PipelineModelError(
                f"the checkpoint at {checkpoint.path} has SHA-256 "
                f"{checkpoint.sha256}; the pipeline saved {self.expected_sha256}"
            )
        regressor = self._regressor_class(
            model_path=checkpoint.path, device=self.device
        )
        expected = self.expected_regressor_params
        if expected is not None:
            actual = get_recorded_regressor_params(regressor)
            changed = sorted(
                name
                for name in set(actual) | set(expected)
                if actual.get(name) != expected.get(name)
            )
            if changed:
                raise PipelineModelError(
                    f"NoriRegressor settings {changed} differ from the pipeline"
                )
        self._regressor = regressor
        self.checkpoint = checkpoint
        return regressor

    def get_pipeline_model_record(self) -> LocalModelRecord | None:
        """Record the checkpoint used so far, or None if it cannot be pinned.

        A checkpoint outside the Hugging Face cache has no revision to pin.
        """
        checkpoint = self.checkpoint
        if checkpoint is None or checkpoint.revision is None:
            return None
        import synthefy_nori

        from nori_rel.pipeline.manifest import LocalModelRecord

        return LocalModelRecord(
            model=self.model,
            repo_id=checkpoint.repo_id,
            revision=checkpoint.revision,
            filename=checkpoint.filename,
            sha256=checkpoint.sha256,
            synthefy_nori_version=synthefy_nori.__version__,
            regressor_params=get_recorded_regressor_params(self._regressor),
        )

    def _get_weights_error_message(self) -> str:
        """Return the error text for weights that cannot load, with the revision."""
        pinned = "" if self.revision is None else f" at revision {self.revision}"
        return (
            f"Could not load local Nori weights for model={self.model!r}{pinned}. "
            "Run once with network access to download them. With "
            "HF_HUB_OFFLINE=1 they must already be in the Hugging Face "
            "cache; delete a corrupted cached file and download it again."
        )


#: ``NoriRegressor`` parameters a pipeline does not record: the per-call
#: category declaration, the device, machine-local paths and the token.
_UNRECORDED_REGRESSOR_PARAMS = frozenset(
    {
        "categorical_columns",
        "device",
        "inference_config",
        "model",
        "model_path",
        "token",
    }
)


def get_recorded_regressor_params(regressor: Any) -> dict[str, Any]:
    """Return the regressor settings that change predictions, as JSON values."""
    params = regressor.get_params()
    return json.loads(
        json.dumps(
            {
                name: value
                for name, value in sorted(params.items())
                if name not in _UNRECORDED_REGRESSOR_PARAMS
            }
        )
    )


class _Declared:
    """Set a regressor's ``categorical_columns`` for one call, then restore it."""

    def __init__(self, regressor: Any, columns: list[str] | None) -> None:
        self.regressor = regressor
        self.columns = columns
        self.previous: Any = None

    def __enter__(self) -> None:
        if self.columns is not None:
            self.previous = self.regressor.get_params()["categorical_columns"]
            self.regressor.set_params(categorical_columns=self.columns)

    def __exit__(self, *args: object) -> None:
        if self.columns is not None:
            self.regressor.set_params(categorical_columns=self.previous)


class LocalRelationalRuntime:
    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        classification_model: str | None,
        timeout: float,
        max_retries: int,
        limits: RelationalLimits | None = None,
        connector_factory: Callable[..., RelationalConnector] = create_connector,
        predictor: Predictor | None = None,
        features: FeatureBackendName | FeatureBackend | AgenticSearch = "fastdfs",
    ) -> None:
        if isinstance(features, str):
            if features not in FEATURE_BACKENDS:
                raise ValueError(f"features must be one of {FEATURE_BACKENDS}")
        elif not isinstance(features, FeatureBackend):
            from nori_rel._feature_search import AgenticSearch

            if not isinstance(features, AgenticSearch):
                raise ValueError("features must be a FeatureBackend or AgenticSearch")
        self.limits = limits or RelationalLimits()
        self.connector_factory = connector_factory
        self.features = features
        self.predictor = predictor or HostedNoriPredictor(
            api_key,
            model,
            classification_model=classification_model,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._connectors: dict[str, RelationalConnector] = {}
        self._closed = False
        self._pipeline_predictors: dict[str, LocalNoriPredictor] = {}
        self._pipeline_lock = threading.Lock()

    def connect(self, config: DatabaseCreateRequest) -> Database:
        if self._closed:
            raise RuntimeError("NoriRelClient is closed")
        database_id = str(uuid.uuid4())
        connector = self.connector_factory(config, self.limits)
        try:
            connector.test()
        except Exception:
            connector.close()
            raise
        now = datetime.now(UTC)
        self._connectors[database_id] = connector
        return Database(
            id=database_id,
            engine=config.engine,
            schema_name=config.schema_name,
            status=DatabaseStatus.READY,
            created_at=now,
            updated_at=now,
        )

    def close(self) -> None:
        """Close every registered source pool, even if one cleanup raises.

        Call after active predictions finish. Idempotent; subsequent source
        registration or database operations fail instead of reopening pools.
        """
        self._closed = True
        connectors = list(self._connectors.values())
        self._connectors.clear()
        with ExitStack() as cleanup:
            for connector in connectors:
                cleanup.callback(connector.close)

    def _connector(self, source: Database) -> RelationalConnector:
        """Return this client's registered source connector and its reusable pool."""
        if self._closed:
            raise RuntimeError("NoriRelClient is closed")
        try:
            return self._connectors[source.id]
        except KeyError:
            raise ValueError("source was not created by this client") from None

    def test_connection(self, source: Database) -> ConnectionStatus:
        latency = self._connector(source).test()
        return ConnectionStatus(
            database_id=source.id,
            status=DatabaseStatus.READY,
            latency_ms=latency,
        )

    def discover(self, source: Database) -> SchemaGraph:
        return self._connector(source).discover(source.id)

    def feature_backend(
        self,
        window: TargetWindow,
        task: PredictionTask,
        context: pd.DataFrame,
        labels: pd.Series,
        program: SavedFeatureProgram | None = None,
        run_context: RunContext | None = None,
        metric: str | None = None,
    ) -> FeatureBackend:
        """Return the feature backend for one prediction.

        A saved ``program`` runs on top of NoriDFS with no search, whatever
        ``features`` is. NoriDFS windows scale with the target's label
        interval. An agent search also needs the context labels, in context
        order, to score its programs, and scores them with ``metric`` (MAE
        when ``None``). The planning session validates the context and
        validation slices according to the search configuration.
        """
        if program is not None:
            from nori_rel._feature_search import (
                DEFAULT_SEARCH_CONFIG,
                AgenticSearch,
                noridfs_backend,
            )
            from nori_rel.feature.agentic import AgenticFeatures

            config = (
                self.features.config
                if isinstance(self.features, AgenticSearch)
                else DEFAULT_SEARCH_CONFIG
            )
            return AgenticFeatures(
                program=program.to_program(),
                fallback=noridfs_backend(window.end - window.start),
                config=config,
                expected_baseline_columns=program.baseline_columns,
                run_context=run_context,
            )
        if isinstance(self.features, FeatureBackend):
            return self.features
        if self.features == "fastdfs":
            return FastDFS(max_depth=2, max_features=self.limits.max_features)
        from nori_rel._feature_search import AgenticSearch, noridfs_backend

        seed = noridfs_backend(window.end - window.start)
        if self.features == "noridfs":
            return seed
        assert isinstance(self.features, AgenticSearch)
        from nori_rel.feature.agentic.scoring import DEFAULT_REGRESSION_METRIC

        return self.features.backend(
            seed,
            labels=labels,
            task=task,
            predictor=self.predictor,
            run_context=run_context,
            metric=DEFAULT_REGRESSION_METRIC if metric is None else metric,
        )

    def get_pipeline_predictor(self, record: Any) -> Predictor:
        """Return a predictor for a saved model record, or raise if it differs.

        A hosted record needs this client's hosted models. A local record
        needs local mode, the same model name and ``synthefy-nori`` version,
        and loads the saved revision, checking its SHA-256 and settings.
        """
        if record.mode == "hosted":
            predictor = self.predictor
            if not isinstance(predictor, HostedNoriPredictor):
                raise PipelineModelError(
                    "this pipeline uses hosted Nori; create the client with "
                    "mode='hosted'"
                )
            names = (predictor.regression_model, predictor.classification_model)
            if names != (record.model, record.classification_model):
                raise PipelineModelError(
                    f"this pipeline uses hosted models {record.model!r} and "
                    f"{record.classification_model!r}; the client uses {names}"
                )
            return predictor
        local = self.predictor
        if not isinstance(local, LocalNoriPredictor):
            raise PipelineModelError(
                "this pipeline uses local Nori; create the client with mode='local'"
            )
        if local.model != record.model:
            raise PipelineModelError(
                f"this pipeline uses model {record.model!r}; "
                f"the client uses {local.model!r}"
            )
        import synthefy_nori

        if synthefy_nori.__version__ != record.synthefy_nori_version:
            raise PipelineModelError(
                f"this pipeline needs synthefy-nori=={record.synthefy_nori_version}; "
                f"{synthefy_nori.__version__} is installed"
            )
        model_record_key = json.dumps(
            record.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )
        with self._pipeline_lock:
            predictor = self._pipeline_predictors.get(model_record_key)
            if predictor is None:
                predictor = LocalNoriPredictor(
                    record.model,
                    device=local.device,
                    revision=record.revision,
                    sha256=record.sha256,
                    regressor_params=record.regressor_params,
                )
                self._pipeline_predictors[model_record_key] = predictor
        return predictor

    def predict_pql(self, source: Database, query: str, **options) -> pd.DataFrame:
        """Prepare and score temporal PQL; return entity/anchor predictions."""
        from nori_rel._pql_runtime import predict_pql

        return predict_pql(self, source, query, **options)
