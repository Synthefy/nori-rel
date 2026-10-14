"""The versioned record of one NoriRel prediction pipeline.

A manifest is a recipe, not a snapshot. It holds what a new process needs to
predict again without a feature search: the PQL task, the schema it read, the
fitted feature definitions, the model input columns, the pinned Nori model,
the output settings, and the rules that build the context rows. It holds no
credentials and no database rows. Each run builds new context rows from the
database with the same rules, relative to its own ``as_of`` time.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import datetime, timedelta
from itertools import pairwise
from typing import Annotated, Any, Literal, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    model_validator,
)

from nori_rel.contracts import (
    DatabaseEngine,
    PredictionTask,
    RelationalOutputType,
    TableSchema,
)
from nori_rel.pipeline.task import get_pql_task_signature

PIPELINE_FORMAT = "nori-rel.pipeline"
PIPELINE_FORMAT_VERSION = 1
PIPELINE_ID_PREFIX = "pipe_"
#: Hex characters of the SHA-256 content digest kept in ``pipeline_id``.
PIPELINE_ID_HEX_CHARS = 16
#: Sections that decide predictions. ``pipeline_id`` hashes exactly these, so
#: an edit to any of them is detected on load; ``created_at``, the writer's
#: version and the search provenance do not change the pipeline's identity.
PIPELINE_ID_FIELDS = frozenset(
    {
        "task",
        "source",
        "features",
        "columns",
        "model",
        "prediction",
        "context_policy",
    }
)

_SHA256_HEX = r"^[0-9a-f]{64}$"
_GIT_COMMIT_HEX = r"^[0-9a-f]{40}$"


class _Record(BaseModel):
    """Reject unknown fields and keep loaded records immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class TaskRecord(_Record):
    """The PQL task: what is predicted, for which entities, and from when."""

    pql_query: str = Field(min_length=1)
    #: ``get_pql_task_signature(pql_query)``: target, entity and WHERE, not FOR.
    signature: str = Field(min_length=1)
    entity_created_at: str = Field(min_length=1)
    task: PredictionTask


class ContextPolicyRecord(_Record):
    """The rules that build the labeled context rows for each run.

    A run takes its ``as_of`` time, puts one context anchor at ``as_of``
    minus each offset, and keeps at most ``max_context_rows`` labeled rows.
    The context therefore moves forward with ``as_of`` and includes the
    labels that became known since the pipeline was saved. Offsets are listed
    from the oldest anchor to the newest, so their count and spacing are
    the policy.
    """

    anchor_offsets: tuple[timedelta, ...] = Field(min_length=1)
    max_context_rows: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_offsets(self) -> Self:
        """Require positive offsets, longest first and all different."""
        offsets = self.anchor_offsets
        if any(offset <= timedelta(0) for offset in offsets):
            raise ValueError("anchor offsets must be positive")
        if any(later >= earlier for earlier, later in pairwise(offsets)):
            raise ValueError("anchor offsets must be listed from longest to shortest")
        return self

    def get_anchor_times(self, as_of: datetime) -> list[datetime]:
        """Return the context anchors for ``as_of``, oldest first."""
        if as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        return [as_of - offset for offset in self.anchor_offsets]

    @classmethod
    def from_anchor_times(
        cls, anchor_times: Sequence[datetime], *, as_of: datetime, max_context_rows: int
    ) -> ContextPolicyRecord:
        """Return the policy that gives ``anchor_times`` when run at ``as_of``."""
        offsets = sorted({as_of - anchor for anchor in anchor_times}, reverse=True)
        return cls(anchor_offsets=tuple(offsets), max_context_rows=max_context_rows)


class SourceRecord(_Record):
    """The schema the pipeline read, and the pandas dtypes of its tables.

    ``column_dtypes`` maps each table to its columns' dtypes in the snapshot
    the features were fitted on. A later snapshot can read an all-NULL column
    or an empty table with another dtype; prediction restores the saved one.
    """

    engine: DatabaseEngine
    schema_name: str = Field(min_length=1)
    tables: tuple[TableSchema, ...] = Field(min_length=1)
    column_dtypes: dict[str, dict[str, str]]

    @model_validator(mode="after")
    def validate_dtype_tables(self) -> Self:
        """Require one dtype entry for each saved table column."""
        expected = {
            table.name: {column.name for column in table.columns}
            for table in self.tables
        }
        actual = {table: set(columns) for table, columns in self.column_dtypes.items()}
        if expected != actual:
            raise ValueError("column_dtypes must name every saved table column")
        return self


FastDFSColumnType = Literal["float", "category", "datetime", "text", "timestamp"]


class FastDFSFeaturesRecord(_Record):
    """FastDFS settings, the column types it inferred, and its version.

    FastDFS plans its features again on every snapshot. A replay passes the
    saved ``column_types`` back, so a text column that gains values does not
    change type and feature set, and refuses another FastDFS version, since
    a new version can compute a feature of the same name differently.
    """

    kind: Literal["fastdfs"] = "fastdfs"
    max_depth: int = Field(gt=0)
    max_features: int
    fastdfs_version: str = Field(min_length=1)
    #: Type of each non-key column, by database table and column name.
    column_types: dict[str, dict[str, FastDFSColumnType]]


FeaturesRecord = FastDFSFeaturesRecord


class ColumnsRecord(_Record):
    """The model input columns in order, and the categories declared to Nori."""

    names: tuple[str, ...] = Field(min_length=1)
    dtypes: tuple[str, ...] = Field(min_length=1)
    #: ``None`` keeps Nori's automatic category detection.
    categorical: tuple[str, ...] | None

    @model_validator(mode="after")
    def validate_columns(self) -> Self:
        """Require one dtype per unique name and known categorical columns."""
        if len(self.names) != len(self.dtypes):
            raise ValueError("columns need one dtype for each name")
        if len(set(self.names)) != len(self.names):
            raise ValueError("column names must be unique")
        if self.categorical is not None and set(self.categorical).difference(
            self.names
        ):
            raise ValueError("categorical columns must be model input columns")
        return self


class LocalModelRecord(_Record):
    """A local Nori checkpoint pinned by revision and file hash."""

    mode: Literal["local"] = "local"
    model: str = Field(min_length=1)
    repo_id: str = Field(min_length=1)
    revision: str = Field(pattern=_GIT_COMMIT_HEX)
    filename: str = Field(min_length=1)
    sha256: str = Field(pattern=_SHA256_HEX)
    synthefy_nori_version: str = Field(min_length=1)
    #: ``NoriRegressor`` settings that change predictions.
    regressor_params: dict[str, JsonValue]


class HostedModelRecord(_Record):
    """Hosted Nori model names. The server's weights are not pinned."""

    mode: Literal["hosted"] = "hosted"
    model: str = Field(min_length=1)
    classification_model: str = Field(min_length=1)
    synthefy_version: str = Field(min_length=1)


ModelRecord = Annotated[
    LocalModelRecord | HostedModelRecord, Field(discriminator="mode")
]


class PredictionRecord(_Record):
    """How Nori's output becomes the returned prediction."""

    output_type: RelationalOutputType | None
    quantiles: tuple[float, ...] | None = None
    decision_threshold: float | None = None


class PipelineManifest(_Record):
    """Everything needed to predict again without a feature search."""

    format: Literal["nori-rel.pipeline"] = PIPELINE_FORMAT
    format_version: Literal[1] = PIPELINE_FORMAT_VERSION
    pipeline_id: str = Field(pattern=rf"^{PIPELINE_ID_PREFIX}[0-9a-f]{{16}}$")
    created_at: AwareDatetime
    nori_rel_version: str = Field(min_length=1)
    task: TaskRecord
    source: SourceRecord
    features: FeaturesRecord
    columns: ColumnsRecord
    model: ModelRecord
    prediction: PredictionRecord
    context_policy: ContextPolicyRecord
    #: Search statistics and the first run's audit data, for people reading the
    #: file; not used to predict.
    provenance: dict[str, JsonValue] = Field(default_factory=dict)

    @classmethod
    def create(cls, **fields: Any) -> PipelineManifest:
        """Build a manifest and compute its ``pipeline_id`` from its content."""
        placeholder = PIPELINE_ID_PREFIX + "0" * PIPELINE_ID_HEX_CHARS
        draft = cls.model_validate({**fields, "pipeline_id": placeholder})
        return draft.model_copy(update={"pipeline_id": get_pipeline_id(draft)})


def get_pipeline_id(manifest: PipelineManifest) -> str:
    """Hash the sections that decide predictions into a ``pipe_`` identifier."""
    payload = manifest.model_dump(mode="json", include=set(PIPELINE_ID_FIELDS))
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    return PIPELINE_ID_PREFIX + digest[:PIPELINE_ID_HEX_CHARS]


__all__ = [
    "PIPELINE_FORMAT",
    "PIPELINE_FORMAT_VERSION",
    "PIPELINE_ID_FIELDS",
    "ColumnsRecord",
    "ContextPolicyRecord",
    "FastDFSColumnType",
    "FastDFSFeaturesRecord",
    "FeaturesRecord",
    "HostedModelRecord",
    "LocalModelRecord",
    "ModelRecord",
    "PipelineManifest",
    "PredictionRecord",
    "SourceRecord",
    "TaskRecord",
    "get_pipeline_id",
    "get_pql_task_signature",
]
