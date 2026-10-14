"""Save, load and inspect a NoriRel pipeline file."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import ValidationError

from nori_rel.feature.agentic.guard import blocked_findings
from nori_rel.pipeline import PIPELINE_ATTR
from nori_rel.pipeline.errors import PipelineFeatureError, PipelineFormatError
from nori_rel.pipeline.manifest import (
    PIPELINE_FORMAT,
    PIPELINE_FORMAT_VERSION,
    AgenticFeaturesRecord,
    PipelineManifest,
    get_pipeline_id,
)


class NoriRelPipeline:
    """A complete, versioned NoriRel prediction pipeline.

    ``client.predict(..., return_details=True)`` returns one as
    ``details["pipeline"]``. Save it with ``save(path)``, load it in another
    process with ``NoriRelPipeline.load(path)``, and predict with
    ``client.predict(source=..., pipeline=pipeline, as_of=..., ...)``: no
    feature search runs again. The pipeline is a recipe: each run builds new
    context rows from the database with the saved rules, relative to its own
    ``as_of`` time. The file is JSON. It holds no credentials and no database
    rows.
    """

    def __init__(self, manifest: PipelineManifest) -> None:
        """Wrap a validated manifest; use ``load`` or ``from_dict`` for files."""
        self.manifest = manifest

    def __repr__(self) -> str:
        """Return the identifier, feature kind and model of this pipeline."""
        return (
            f"{type(self).__name__}(pipeline_id={self.pipeline_id!r}, "
            f"features={self.manifest.features.kind!r}, "
            f"model={self.manifest.model.model!r})"
        )

    def __eq__(self, other: object) -> bool:
        """Compare pipelines by their full manifest."""
        return isinstance(other, NoriRelPipeline) and self.manifest == other.manifest

    __hash__ = None  # type: ignore[assignment]

    @property
    def pipeline_id(self) -> str:
        """Return the content-derived identifier, e.g. ``pipe_0123abcd4567ef89``."""
        return self.manifest.pipeline_id

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Return the model input columns, in the order Nori reads them."""
        return self.manifest.columns.names

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-ready record that ``save`` writes."""
        return self.manifest.model_dump(mode="json")

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> NoriRelPipeline:
        """Validate a pipeline record and reject other formats or edited files."""
        return cls(_get_validated_manifest(record))

    @classmethod
    def from_prediction(cls, frame: pd.DataFrame) -> NoriRelPipeline | None:
        """Return the pipeline that produced ``frame``, or None if it has none."""
        record = frame.attrs.get(PIPELINE_ATTR)
        return None if record is None else cls.from_dict(record)

    def save(self, path: str | os.PathLike[str]) -> Path:
        """Write the pipeline to ``path`` as JSON and return the path.

        The file is replaced atomically, so a reader never sees half of it.
        """
        target = Path(path)
        text = json.dumps(self.to_dict(), indent=2) + "\n"
        handle, temporary = tempfile.mkstemp(
            dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                stream.write(text)
            os.replace(temporary, target)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
        return target

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> NoriRelPipeline:
        """Read a pipeline file written by ``save``.

        Raises ``PipelineFormatError`` for a file that is not a pipeline, a
        format version this nori-rel cannot read, or content that changed
        after it was saved. Loading opens no database connection.
        """
        try:
            record = json.loads(Path(path).read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise PipelineFormatError(f"{path} is not a JSON pipeline file") from exc
        return cls.from_dict(record)


def _get_validated_manifest(record: Mapping[str, Any]) -> PipelineManifest:
    """Check the format and version first, then the fields, then the digest."""
    if not isinstance(record, Mapping):
        raise PipelineFormatError("a pipeline record must be a JSON object")
    if record.get("format") != PIPELINE_FORMAT:
        raise PipelineFormatError(
            f"not a NoriRel pipeline: format is {record.get('format')!r}, "
            f"expected {PIPELINE_FORMAT!r}"
        )
    version = record.get("format_version")
    if not isinstance(version, int) or isinstance(version, bool):
        raise PipelineFormatError("format_version must be an integer")
    if version > PIPELINE_FORMAT_VERSION:
        raise PipelineFormatError(
            f"this pipeline has format_version {version}, but this nori-rel "
            f"reads up to {PIPELINE_FORMAT_VERSION}; upgrade nori-rel"
        )
    if version < PIPELINE_FORMAT_VERSION:
        raise PipelineFormatError(
            f"format_version {version} is no longer supported; "
            "export the pipeline again"
        )
    try:
        manifest = PipelineManifest.model_validate(record)
    except ValidationError as exc:
        raise PipelineFormatError(f"invalid pipeline record: {exc}") from exc
    if get_pipeline_id(manifest) != manifest.pipeline_id:
        raise PipelineFormatError(
            f"pipeline {manifest.pipeline_id} does not match its content; "
            "the file changed after it was saved"
        )
    if isinstance(manifest.features, AgenticFeaturesRecord):
        findings = [
            finding
            for query in manifest.features.program.queries
            for finding in blocked_findings(query.name, query.sql)
        ]
        if findings:
            reasons = "; ".join(f"{f.query}: {f.reason}" for f in findings)
            raise PipelineFeatureError(f"saved SQL is not allowed: {reasons}")
    return manifest


__all__ = ["NoriRelPipeline"]
