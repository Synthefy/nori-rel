"""Run record fields that a predict request or a saved pipeline decides.

The client reports them when a run starts, so a run that is refused, fails
before it reads anything or whose process dies still names its pipeline,
source and times. The prediction then fills in what each step learns.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

import pandas as pd

from nori_rel.contracts import Database, PredictionTask


def get_record_time(value: Any) -> datetime | None:
    """Return a timezone-aware time in UTC for the run record, else None.

    A naive time has no instant to record and a value that is not a time has
    nothing to record, so the record leaves both out.
    """
    if value is None:
        return None
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(stamp) or stamp.tzinfo is None:
        return None
    return stamp.tz_convert("UTC").to_pydatetime()


def get_pipeline_record_fields(manifest: Any) -> dict[str, Any]:
    """Return the run record's fields a saved pipeline fixes before Nori runs.

    The pipeline id and the model it pins: the transport, the model name for
    its task, and the checkpoint's SHA-256 when it is a local file. A replay
    that fails, before or after its model loads, records these.
    """
    model = manifest.model
    fields: dict[str, Any] = {"pipeline_id": manifest.pipeline_id}
    if model.mode == "local":
        return fields | {
            "transport": "local",
            "model": model.model,
            "checkpoint_sha256": model.sha256,
        }
    is_binary = manifest.task.task is PredictionTask.CLASSIFICATION
    return fields | {
        "transport": model.mode,
        "model": model.classification_model if is_binary else model.model,
    }


def get_loaded_checkpoint_sha256(predictor: Any) -> str | None:
    """Return the SHA-256 of the checkpoint local Nori loaded, if it loaded one."""
    return getattr(getattr(predictor, "checkpoint", None), "sha256", None)


def get_request_record_fields(request: Mapping[str, Any]) -> dict[str, Any]:
    """Return the record fields a predict request decides before its run starts.

    The source's engine and schema, the prediction time and completeness
    watermark, and a pipeline's id and pinned model. A request can be refused
    for holding a wrong value, so each value is read only when it has the
    expected type, and this never raises for the request's contents.
    """
    from nori_rel.pipeline import NoriRelPipeline

    fields: dict[str, Any] = {
        "as_of": get_record_time(request.get("as_of") or request.get("anchor_time")),
        "observed_through": get_record_time(request.get("observed_through")),
    }
    source = request.get("source")
    if isinstance(source, Database):
        fields |= {"engine": str(source.engine), "schema_name": source.schema_name}
    pipeline = request.get("pipeline")
    if isinstance(pipeline, NoriRelPipeline):
        fields |= get_pipeline_record_fields(pipeline.manifest)
    return fields


__all__ = [
    "get_loaded_checkpoint_sha256",
    "get_pipeline_record_fields",
    "get_record_time",
    "get_request_record_fields",
]
