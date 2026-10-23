"""Local Nori inference keeps feature matrices and predictions in process."""

import hashlib
import socket

import numpy as np
import pandas as pd
import pytest
from huggingface_hub.errors import LocalEntryNotFoundError

from nori_rel import NoriRelClient
from nori_rel._runtime import HostedNoriPredictor, LocalNoriPredictor
from nori_rel.contracts import PredictionTask, RelationalOutputType
from nori_rel.pipeline import PipelineModelError

FAKE_CHECKPOINT_REVISION = "0123456789abcdef0123456789abcdef01234567"


class FakeRegressor:
    error = None

    def __init__(self, **options) -> None:
        self.options = options
        self.fits = []

    def get_params(self):
        return {"augmentations": ("yj",)}

    def fit(self, train, target):
        self.fits.append((train, target))
        return self

    def predict(self, query, *, output_type="mean", quantiles=None):
        if self.error is not None:
            raise self.error
        if output_type == "quantiles":
            return np.array([[level * 10] * len(query) for level in quantiles])
        return np.full(len(query), 2.5)


@pytest.fixture
def no_network(monkeypatch, fake_checkpoint):
    def refuse(*args, **kwargs):
        raise AssertionError("local prediction made a hosted or network request")

    monkeypatch.setattr("nori_rel._runtime.SynthefyNoriClient", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr("synthefy_nori.NoriRegressor", FakeRegressor)
    return fake_checkpoint


def _arrays():
    return (
        pd.DataFrame({"orders.total": [1.0, 2.0]}),
        pd.Series([1.5, 3.0]),
        pd.DataFrame({"orders.total": [4.0, 5.0]}),
    )


def test_local_is_the_default_and_needs_no_api_key(monkeypatch, no_network) -> None:
    monkeypatch.delenv("SYNTHEFY_NORI_API_KEY", raising=False)

    client = NoriRelClient(device="cpu")

    predictor = client._runtime.predictor
    assert client.mode == "local"
    assert isinstance(predictor, LocalNoriPredictor)
    assert predictor._regressor is None
    _median(predictor)
    assert predictor._regressor.options == {
        "model_path": str(no_network.path),
        "device": "cpu",
    }
    assert predictor.checkpoint.revision == FAKE_CHECKPOINT_REVISION
    assert no_network.requests == [("nori-1-5", None)]


def test_hosted_mode_is_explicit() -> None:
    client = NoriRelClient(mode="hosted", api_key="hosted-key")

    assert isinstance(client._runtime.predictor, HostedNoriPredictor)


@pytest.mark.parametrize(
    "options,match",
    [
        ({"api_key": "key"}, "pass mode='hosted'"),
        ({"classification_model": "nori-30m"}, "regression targets only"),
        ({"mode": "hosted", "api_key": "key", "device": "cpu"}, "mode='local'"),
        ({"mode": "remote"}, "'local' or 'hosted'"),
    ],
)
def test_invalid_mode_options_fail_clearly(options, match) -> None:
    with pytest.raises(ValueError, match=match):
        NoriRelClient(_runtime=object(), **options)


def test_local_point_and_quantile_predictions(no_network) -> None:
    predictor = LocalNoriPredictor("nori-30m")
    train, target, query = _arrays()

    median = predictor.predict_arrays(
        train,
        target,
        query,
        task=PredictionTask.REGRESSION,
        output_type=RelationalOutputType.MEDIAN,
    )
    bands = predictor.predict_arrays(
        train,
        target,
        query,
        task=PredictionTask.REGRESSION,
        output_type=RelationalOutputType.QUANTILES,
        quantiles=[0.1, 0.5, 0.9],
    )

    assert predictor._regressor.fits[0] == (train, target)
    assert median == ([2.5, 2.5], None)
    assert bands == ([5.0, 5.0], [[1.0, 5.0, 9.0], [1.0, 5.0, 9.0]])


def test_local_binary_targets_fail_clearly(no_network) -> None:
    with pytest.raises(ValueError, match="regression PQL targets only"):
        LocalNoriPredictor("nori-30m").model_for(PredictionTask.CLASSIFICATION)


@pytest.mark.parametrize(
    "error",
    [
        LocalEntryNotFoundError("not in the local cache"),
        RuntimeError("PytorchStreamReader failed reading zip archive"),
    ],
)
def test_missing_or_corrupted_weights_fail_clearly(
    monkeypatch, no_network, error
) -> None:
    monkeypatch.setattr(FakeRegressor, "error", error)

    with pytest.raises(RuntimeError, match="HF_HUB_OFFLINE=1"):
        _median(LocalNoriPredictor("nori-30m"))


def test_an_uncached_pinned_revision_fails_clearly(no_network) -> None:
    predictor = LocalNoriPredictor("nori-30m", revision="f" * 40)

    with pytest.raises(RuntimeError, match=f"at revision {'f' * 40}"):
        _median(predictor)


@pytest.mark.parametrize(
    "model,match",
    [("nori-9", "has no model 'nori-9'"), ("Synthefy/Private", "access denied")],
)
def test_a_failed_download_names_a_model_the_runtime_lacks(
    monkeypatch, no_network, model, match
) -> None:
    def deny(*args):
        raise RuntimeError("access denied")

    monkeypatch.setattr("nori_rel._checkpoint.get_checkpoint_path", deny)

    with pytest.raises(RuntimeError, match=match):
        _median(LocalNoriPredictor(model))


def test_a_pinned_checkpoint_loads_only_with_its_hash(no_network) -> None:
    digest = hashlib.sha256(no_network.path.read_bytes()).hexdigest()
    pinned = LocalNoriPredictor(
        "nori-30m", revision=FAKE_CHECKPOINT_REVISION, sha256=digest
    )
    wrong = LocalNoriPredictor(
        "nori-30m", revision=FAKE_CHECKPOINT_REVISION, sha256="0" * 64
    )

    assert _median(pinned) == ([2.5, 2.5], None)
    assert pinned.checkpoint.sha256 == digest
    with pytest.raises(PipelineModelError, match="the pipeline saved 0000"):
        _median(wrong)
    assert no_network.requests[-1] == ("nori-30m", FAKE_CHECKPOINT_REVISION)


def _median(predictor):
    return predictor.predict_arrays(
        *_arrays(),
        task=PredictionTask.REGRESSION,
        output_type=RelationalOutputType.MEDIAN,
    )
