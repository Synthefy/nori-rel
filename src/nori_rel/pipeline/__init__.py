"""Save a NoriRel prediction pipeline and predict with it in another process.

The errors and ``PIPELINE_ATTR`` import eagerly. Everything else loads on first
use, so importing ``nori_rel`` does not import the feature backends a pipeline
record names.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

from nori_rel.pipeline.errors import (
    PipelineContextError,
    PipelineError,
    PipelineFeatureError,
    PipelineFormatError,
    PipelineModelError,
    PipelineSchemaError,
    PipelineTaskError,
    PipelineVersionWarning,
)

#: ``DataFrame.attrs`` key that carries a prediction's pipeline record.
PIPELINE_ATTR = "pipeline"
#: ``DataFrame.attrs`` key that carries the scored rows' feature matrix from
#: ``predict_pql`` to ``predict``, which moves it to ``PredictionDetails``.
SCORING_FEATURES_ATTR = "scoring_features"

if TYPE_CHECKING:
    from nori_rel.pipeline.artifact import NoriRelPipeline
    from nori_rel.pipeline.details import PredictionDetails
    from nori_rel.pipeline.manifest import (
        PIPELINE_FORMAT,
        PIPELINE_FORMAT_VERSION,
        PipelineManifest,
        get_pql_task_signature,
    )
    from nori_rel.pipeline.request import PipelinePredictRequest

_LAZY_MODULES = {
    "NoriRelPipeline": "nori_rel.pipeline.artifact",
    "PipelinePredictRequest": "nori_rel.pipeline.request",
    "PredictionDetails": "nori_rel.pipeline.details",
    "PIPELINE_FORMAT": "nori_rel.pipeline.manifest",
    "PIPELINE_FORMAT_VERSION": "nori_rel.pipeline.manifest",
    "PipelineManifest": "nori_rel.pipeline.manifest",
    "get_pql_task_signature": "nori_rel.pipeline.manifest",
}


def __getattr__(name: str) -> Any:
    """Load the pipeline records only when one of their names is used."""
    if name in _LAZY_MODULES:
        return getattr(import_module(_LAZY_MODULES[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PIPELINE_ATTR",
    "PIPELINE_FORMAT",
    "PIPELINE_FORMAT_VERSION",
    "SCORING_FEATURES_ATTR",
    "NoriRelPipeline",
    "PipelineContextError",
    "PipelineError",
    "PipelineFeatureError",
    "PipelineFormatError",
    "PipelineManifest",
    "PipelineModelError",
    "PipelinePredictRequest",
    "PipelineSchemaError",
    "PipelineTaskError",
    "PipelineVersionWarning",
    "PredictionDetails",
    "get_pql_task_signature",
]
