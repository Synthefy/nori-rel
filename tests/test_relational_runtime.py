"""Hosted transport receives only flat feature matrices."""

import pandas as pd

from nori_rel._runtime import HostedNoriPredictor
from nori_rel.contracts import PredictionTask, RelationalOutputType


def test_hosted_boundary_receives_only_flat_feature_matrices(monkeypatch) -> None:
    captured = {}

    class NoriClient:
        def __init__(self, **kwargs) -> None:
            captured["client"] = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def predict(self, train, target, query, **kwargs):
            captured["prediction"] = (train, target, query, kwargs)
            return pd.Series([2.5])

    monkeypatch.setattr("nori_rel._runtime.SynthefyNoriClient", NoriClient)
    predictor = HostedNoriPredictor("synthefy-key", "nori-30m")
    train = pd.DataFrame({"orders.total": [1.0, 2.0]})
    target = pd.Series([1.5, 3.0])
    query = pd.DataFrame({"orders.total": [4.0]})

    predictions, quantiles = predictor.predict_arrays(
        train,
        target,
        query,
        task=PredictionTask.REGRESSION,
        output_type=RelationalOutputType.MEDIAN,
    )

    assert captured["client"]["api_key"] == "synthefy-key"
    assert captured["client"]["model"] == "nori-30m"
    assert captured["prediction"][0] is train
    assert captured["prediction"][1] is target
    assert captured["prediction"][2] is query
    assert predictions == [2.5]
    assert quantiles is None
