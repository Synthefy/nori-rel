"""What ``predict(return_details=True)`` returns."""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, SkipValidation

from nori_rel.pipeline.artifact import NoriRelPipeline


class PredictionDetails(BaseModel):
    """A prediction with its metadata and the pipeline that produced it.

    ``details["predictions"]``, ``details["metadata"]`` and
    ``details["pipeline"]`` work as they did when ``return_details`` returned
    a dict.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    predictions: pd.DataFrame
    metadata: dict[str, Any]
    #: ``None`` when the prediction cannot be exported (a custom feature
    #: backend, or a model that cannot be pinned).
    pipeline: SkipValidation[NoriRelPipeline | None]

    def __getitem__(self, key: str) -> Any:
        """Return a field by name, for code written against the old dict."""
        if key not in type(self).model_fields:
            raise KeyError(key)
        return getattr(self, key)


__all__ = ["PredictionDetails"]
