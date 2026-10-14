"""Saved model settings participate in the local predictor cache identity."""

import sys
import threading
from types import ModuleType

from nori_rel._runtime import LocalRelationalRuntime
from nori_rel.pipeline.manifest import LocalModelRecord


def test_same_weights_with_different_settings_do_not_share_predictors(monkeypatch):
    """Each saved recipe must validate its own settings before local inference."""

    class FakeLocalPredictor:
        def __init__(self, model, *, device=None, **settings):
            self.model = model
            self.device = device
            self.settings = settings

    runtime_module = ModuleType("synthefy_nori")
    runtime_module.__version__ = "0.20.1"
    monkeypatch.setitem(sys.modules, "synthefy_nori", runtime_module)
    monkeypatch.setattr("nori_rel._runtime.LocalNoriPredictor", FakeLocalPredictor)
    runtime = LocalRelationalRuntime.__new__(LocalRelationalRuntime)
    runtime.predictor = FakeLocalPredictor("nori-30m")
    runtime._pipeline_lock = threading.Lock()
    runtime._pipeline_predictors = {}
    first_record = LocalModelRecord(
        model="nori-30m",
        repo_id="Synthefy/Nori-30M",
        revision="a" * 40,
        filename="model.pt",
        sha256="b" * 64,
        synthefy_nori_version="0.20.1",
        regressor_params={"augmentations": ["yj"]},
    )
    changed_record = first_record.model_copy(
        update={"regressor_params": {"augmentations": []}}
    )

    first = runtime.get_pipeline_predictor(first_record)
    changed = runtime.get_pipeline_predictor(changed_record)

    assert changed is not first
    assert changed.settings["regressor_params"] == {"augmentations": []}
    assert first.settings["regressor_params"] == {"augmentations": ["yj"]}
    assert runtime.get_pipeline_predictor(first_record) is first
