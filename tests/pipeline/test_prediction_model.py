"""NoriRelModel predicts like the prediction that produced it, and saves exactly."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import nori_rel._runtime
from nori_rel.pipeline import PredictionDetails
from nori_rel.pipeline.manifest import LocalModelRecord
from nori_rel.pipeline.model import NoriRelModel

_INTERNAL_NAMES = (
    "nr|anchor|entity_key",
    "nr|path|drivers%3C-results.driverId|grid|mean|all",
)
_RECORD = LocalModelRecord(
    model="nori-30m",
    repo_id="Synthefy/Nori-30M",
    revision="0123456789abcdef0123456789abcdef01234567",
    filename="nori.pt",
    sha256="0" * 64,
    synthefy_nori_version="0.20.1",
    regressor_params={"augmentations": ["yj"]},
)


class RecordingRegressor:
    """Return the sum of each row, and record what fit and predict were given."""

    def fit(self, train, target):
        self.train, self.target = train, target
        return self

    def predict(self, X, *, output_type="mean", quantiles=None):
        self.frame, self.output_type, self.quantiles = X, output_type, quantiles
        return X.sum(axis=1).to_numpy(dtype=float)


@pytest.fixture
def fitted_regressors(monkeypatch) -> list[dict]:
    """Make from_context fit a RecordingRegressor; return each fit's arguments."""
    calls: list[dict] = []

    def get_fitted_regressor(self, train, target, *, categorical_columns):
        calls.append(
            {
                "model": self.model,
                "revision": self.revision,
                "train": train,
                "target": target,
                "categorical_columns": categorical_columns,
            }
        )
        return RecordingRegressor().fit(train, target)

    monkeypatch.setattr(
        nori_rel._runtime.LocalNoriPredictor,
        "get_fitted_regressor",
        get_fitted_regressor,
    )
    monkeypatch.setattr(
        nori_rel._runtime.LocalNoriPredictor, "__init__", _get_predictor_without_nori
    )
    return calls


def _get_predictor_without_nori(self, model, **options) -> None:
    """Initialise a LocalNoriPredictor's names without importing synthefy-nori."""
    self.model = model
    self.revision = options.get("revision")


def _get_model(**options) -> NoriRelModel:
    """Return a model fitted on three context rows of two features."""
    return NoriRelModel.from_context(
        pd.DataFrame(
            {
                "anchor.entity_key": pd.Series([829, 1, 829], dtype="int64"),
                "results.grid.mean": [5.5, 12.0, 5.4],
            }
        ),
        pd.Series([1.0, 9.0, 1.3]),
        internal_feature_names=_INTERNAL_NAMES,
        model_record=_RECORD,
        categorical_columns=(_INTERNAL_NAMES[0],),
        output_type=options.pop("output_type", "median"),
        **options,
    )


def test_the_model_is_fitted_under_internal_names_with_the_pinned_checkpoint(
    fitted_regressors,
) -> None:
    model = _get_model()
    (call,) = fitted_regressors
    assert call["model"] == "nori-30m" and call["revision"] == _RECORD.revision
    assert call["train"].columns.tolist() == list(_INTERNAL_NAMES)
    assert call["target"].tolist() == [1.0, 9.0, 1.3]
    assert call["categorical_columns"] == [_INTERNAL_NAMES[0]]
    assert model.feature_names == ("anchor.entity_key", "results.grid.mean")


def test_predict_passes_the_internal_names_and_the_prediction_output_type(
    fitted_regressors,
) -> None:
    model = _get_model()
    rows = pd.DataFrame(
        {"results.grid.mean": [5.5, 2.0], "anchor.entity_key": [829, 1]},
        index=pd.Index([829, 1], name="entity_id"),
    )
    assert model.predict(rows).tolist() == [834.5, 3.0]
    frame = model.regressor.frame
    assert frame.columns.tolist() == list(_INTERNAL_NAMES)
    assert frame.index.tolist() == [0, 1]
    assert model.regressor.output_type == "median"
    model.predict(rows, output_type="mean")
    assert model.regressor.output_type == "mean"


def test_predict_takes_an_array_and_restores_the_fitted_dtypes(
    fitted_regressors,
) -> None:
    model = _get_model()
    assert model.predict(np.array([[829.0, 5.5]])).tolist() == [834.5]
    assert model.predict(np.array([829.0, 5.5])).tolist() == [834.5]
    assert model.regressor.frame.dtypes.tolist() == [
        np.dtype("int64"),
        np.dtype("float64"),
    ]


def test_predict_passes_the_saved_quantiles_for_a_quantile_forecast(
    fitted_regressors,
) -> None:
    model = _get_model(output_type="quantiles", quantiles=(0.1, 0.9))
    model.predict(np.array([[1.0, 2.0]]))
    assert model.regressor.quantiles == [0.1, 0.9]


def test_predict_refuses_rows_without_exactly_the_model_features(
    fitted_regressors,
) -> None:
    model = _get_model()
    with pytest.raises(ValueError, match="missing feature columns"):
        model.predict(pd.DataFrame({"anchor.entity_key": [1]}))
    with pytest.raises(
        ValueError, match=r"columns \['results.count'\] that this model"
    ):
        model.predict(
            pd.DataFrame(
                {
                    "anchor.entity_key": [1],
                    "results.grid.mean": [2.0],
                    "results.count": [3],
                }
            )
        )
    with pytest.raises(ValueError, match="needs 2 columns"):
        model.predict(np.zeros((1, 3)))


def test_a_saved_model_loads_as_the_same_fit(fitted_regressors, tmp_path) -> None:
    model = _get_model(output_type="quantiles", quantiles=(0.1, 0.9))
    path = model.save(tmp_path / "model.parquet")
    loaded = NoriRelModel.load(path)
    first, second = fitted_regressors
    pd.testing.assert_frame_equal(second["train"], first["train"])
    pd.testing.assert_series_equal(second["target"], first["target"], check_names=False)
    assert second["categorical_columns"] == first["categorical_columns"]
    pd.testing.assert_frame_equal(loaded.context_features, model.context_features)
    assert loaded.model_record == _RECORD
    assert (loaded.output_type, loaded.quantiles) == ("quantiles", (0.1, 0.9))
    assert loaded.predict(np.array([[829.0, 5.5]])).tolist() == [834.5]


def test_loading_refuses_a_file_that_is_not_a_saved_model(tmp_path) -> None:
    path = tmp_path / "rows.parquet"
    pd.DataFrame({"x": [1.0]}).to_parquet(path)
    with pytest.raises(ValueError, match="not a saved nori-rel model"):
        NoriRelModel.load(path)


def test_details_have_no_model_without_a_pipeline_or_context_rows() -> None:
    details = PredictionDetails(
        predictions=pd.DataFrame({"entity_id": [1], "prediction": [1.0]}),
        metadata={},
        pipeline=None,
    )
    assert details.model is None
