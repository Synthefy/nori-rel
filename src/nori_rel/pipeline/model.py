"""The Nori that made a prediction, fitted on its context rows."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nori_rel.pipeline.manifest import LocalModelRecord

#: Format of a saved model file, recorded in its Parquet metadata.
MODEL_FORMAT = "nori-rel.model"
MODEL_FORMAT_VERSION = 1
#: Parquet metadata key that holds the model's settings.
_MODEL_METADATA_KEY = b"nori_rel_model"
#: Column of a saved model file that holds the context labels.
_LABEL_COLUMN = "__nori_rel_label__"


class NoriRelModel:
    """Predict as the prediction that produced this model did.

    ``regressor`` is a NoriRegressor fitted on the prediction's context rows
    (``context_features`` and ``labels``, in the order Nori read them) under
    the model's internal feature names. ``predict`` takes rows under the
    dotted names of ``PredictionDetails.features``, as a DataFrame or as an
    array in that column order, which is what SHAP explainers pass, and
    returns the prediction's own output type unless asked for another.

    ``save`` writes the context rows and the model settings to one Parquet
    file and ``load`` fits the same pinned checkpoint on them again, so a
    loaded model predicts exactly as the original. The file holds the derived
    feature rows and labels Nori learned from: treat it like that data.
    """

    def __init__(
        self,
        regressor: Any,
        *,
        context_features: pd.DataFrame,
        labels: pd.Series,
        internal_feature_names: Sequence[str],
        model_record: LocalModelRecord,
        categorical_columns: Sequence[str] | None,
        output_type: str,
        quantiles: Sequence[float] | None = None,
    ) -> None:
        """Wrap a regressor fitted on ``context_features`` and ``labels``."""
        if context_features.shape[1] != len(internal_feature_names):
            raise ValueError(
                "internal_feature_names needs one name per column of context_features"
            )
        if len(context_features) != len(labels):
            raise ValueError("context_features and labels need the same rows")
        self.regressor = regressor
        self.context_features = context_features.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True).rename("label")
        self.internal_feature_names = tuple(internal_feature_names)
        self.model_record = model_record
        self.categorical_columns = (
            None if categorical_columns is None else tuple(categorical_columns)
        )
        self.output_type = output_type
        self.quantiles = None if quantiles is None else tuple(quantiles)

    @classmethod
    def from_context(
        cls,
        context_features: pd.DataFrame,
        labels: pd.Series,
        *,
        internal_feature_names: Sequence[str],
        model_record: LocalModelRecord,
        categorical_columns: Sequence[str] | None,
        output_type: str,
        quantiles: Sequence[float] | None = None,
    ) -> NoriRelModel:
        """Return the model fitted on context rows with ``model_record``'s checkpoint.

        The checkpoint loads through the same pinned path and settings check
        as a prediction, so fitting on a prediction's context rows in their
        stored order reproduces its predictions.
        """
        from nori_rel._runtime import LocalNoriPredictor

        regressor = LocalNoriPredictor(
            model_record.model,
            revision=model_record.revision,
            sha256=model_record.sha256,
            regressor_params=model_record.regressor_params,
        ).get_fitted_regressor(
            context_features.set_axis(list(internal_feature_names), axis=1).reset_index(
                drop=True
            ),
            # A prediction fits on labels named "label", whatever their source.
            labels.reset_index(drop=True).rename("label"),
            categorical_columns=(
                None if categorical_columns is None else list(categorical_columns)
            ),
        )
        return cls(
            regressor,
            context_features=context_features,
            labels=labels,
            internal_feature_names=internal_feature_names,
            model_record=model_record,
            categorical_columns=categorical_columns,
            output_type=output_type,
            quantiles=quantiles,
        )

    @classmethod
    def load(cls, path: str | Path) -> NoriRelModel:
        """Return the model saved at ``path``, fitted again on its context rows.

        Raises:
            ValueError: the file is not a saved nori-rel model of this format.
        """
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        raw = (table.schema.metadata or {}).get(_MODEL_METADATA_KEY)
        if raw is None:
            raise ValueError(f"{path} is not a saved nori-rel model")
        settings = json.loads(raw)
        if (
            settings.get("format") != MODEL_FORMAT
            or settings.get("format_version") != MODEL_FORMAT_VERSION
        ):
            raise ValueError(
                f"{path} has model format {settings.get('format')!r} version "
                f"{settings.get('format_version')!r}; this nori-rel reads "
                f"{MODEL_FORMAT!r} version {MODEL_FORMAT_VERSION}"
            )
        frame = table.to_pandas()
        labels = frame.pop(_LABEL_COLUMN)
        return cls.from_context(
            frame,
            labels,
            internal_feature_names=settings["internal_feature_names"],
            model_record=LocalModelRecord.model_validate(settings["model"]),
            categorical_columns=settings["categorical_columns"],
            output_type=settings["output_type"],
            quantiles=settings["quantiles"],
        )

    def save(self, path: str | Path) -> Path:
        """Write the context rows and the model settings to one Parquet file.

        Returns the path written. The file holds no model weights: ``load``
        reads the pinned checkpoint from the Hugging Face cache.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        if _LABEL_COLUMN in self.context_features.columns:
            raise ValueError(f"a feature column is named {_LABEL_COLUMN!r}")
        frame = self.context_features.assign(**{_LABEL_COLUMN: self.labels})
        table = pa.Table.from_pandas(frame, preserve_index=False)
        settings = {
            "format": MODEL_FORMAT,
            "format_version": MODEL_FORMAT_VERSION,
            "internal_feature_names": list(self.internal_feature_names),
            "model": self.model_record.model_dump(mode="json"),
            "categorical_columns": (
                None
                if self.categorical_columns is None
                else list(self.categorical_columns)
            ),
            "output_type": self.output_type,
            "quantiles": None if self.quantiles is None else list(self.quantiles),
        }
        table = table.replace_schema_metadata(
            {**(table.schema.metadata or {}), _MODEL_METADATA_KEY: json.dumps(settings)}
        )
        written = Path(path)
        pq.write_table(table, written)
        return written

    @property
    def feature_names(self) -> tuple[str, ...]:
        """The dotted names of the model's features, in column order."""
        return tuple(str(name) for name in self.context_features.columns)

    def __repr__(self) -> str:
        """Return the output type, feature count and context rows of this model."""
        return (
            f"{type(self).__name__}(output_type={self.output_type!r}, "
            f"features={len(self.feature_names)}, context_rows={len(self.labels)})"
        )

    def predict(
        self, X: pd.DataFrame | np.ndarray, *, output_type: str | None = None
    ) -> np.ndarray:
        """Return Nori's prediction for each row of ``X``.

        ``output_type`` defaults to the prediction's own, such as "median", so
        ``predict(details.features)`` returns ``details.predictions``. Another
        output type, such as "mean", returns that statistic of the same
        predicted distribution.
        """
        chosen = output_type or self.output_type
        options = (
            {"quantiles": list(self.quantiles)}
            if chosen == "quantiles" and self.quantiles is not None
            else {}
        )
        return np.asarray(
            self.regressor.predict(
                self._get_internal_frame(X), output_type=chosen, **options
            )
        )

    def _get_internal_frame(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """Return ``X`` under the internal names, with the context rows' dtypes.

        A DataFrame needs exactly the model's feature columns, in any order;
        a missing or an extra column is refused. An array arrives in the order
        of ``feature_names``, often as floats, so each column takes back the
        dtype the model was fitted on.
        """
        names = list(self.feature_names)
        if isinstance(X, pd.DataFrame):
            missing = sorted(set(names) - set(X.columns))
            if missing:
                raise ValueError(f"X is missing feature columns {missing}")
            extra = sorted(str(name) for name in set(X.columns) - set(names))
            if extra:
                raise ValueError(
                    f"X has columns {extra} that this model was not fitted on"
                )
            frame = X[names]
        else:
            array = np.asarray(X)
            if array.ndim == 1:
                array = array.reshape(1, -1)
            if array.ndim != 2 or array.shape[1] != len(names):
                raise ValueError(
                    f"X needs {len(names)} columns, in the order of feature_names"
                )
            frame = pd.DataFrame(array, columns=names).astype(
                self.context_features.dtypes.to_dict()
            )
        return frame.set_axis(list(self.internal_feature_names), axis=1).reset_index(
            drop=True
        )
