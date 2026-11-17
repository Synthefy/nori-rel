"""A run's record: what produced a prediction, kept after the process exits.

Every run builds a :class:`RunRecord` when it ends, whether it completed,
failed or was cancelled. For any number in a predictions table it names the
pipeline, the data snapshot, the package and model versions, the identity the
connector read as, and whether the run finished. A client created with
``run_records=`` writes each record to a :class:`RunRecordStore`, first with
status ``running`` when the run starts and again when it ends; a
:class:`JsonRunRecordStore` keeps one JSON file per run. See
``docs/pipeline-export.md``.

A record holds names, versions, hashes, counts and times. It holds no
credentials and no row values: its error message is the run's redacted one
(:func:`~nori_rel.run.redact.redact`), and its principal is the database's
own name for the user.
"""

from __future__ import annotations

import functools
import importlib.metadata
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Final, Literal, Protocol, runtime_checkable

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

RUN_RECORD_FORMAT: Final = "nori-rel.run-record"
RUN_RECORD_FORMAT_VERSION: Final = 1
#: ``DataFrame.attrs`` keys that carry a completed run's id and its record.
RUN_ID_ATTR: Final = "run_id"
RUN_RECORD_ATTR: Final = "run_record"
#: Run ids a store accepts as file names, such as ``run_3f2b9c0d1e4a5b6c``.
_STORE_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RunSnapshot(_Frozen):
    """The pinned warehouse state a run read, as ``metadata["snapshot"]`` has it.

    ``table_references`` holds each table's pinned SQL reference: a Snowflake
    Time Travel ``AT (TIMESTAMP => …)`` clause or a Delta ``VERSION AS OF``.
    The types are the connector's ``SnapshotProvenance`` types, so the record
    holds whatever a connector reported.
    """

    engine: str = Field(min_length=1)
    extraction_timestamp: datetime
    table_references: dict[str, str]


class RunRecord(_Frozen):
    """What one run read, ran and produced, for an audit of its predictions.

    Fields a run never reached stay ``None``: a run that failed before its
    read has no snapshot or row counts. A ``running`` record is the one a run
    stores when it starts; it stays ``running``, with no ``finished_at``,
    only if the process died before the run ended. ``output_table`` and
    ``rows_written`` are set when write-back stores the run's predictions.
    """

    format: Literal["nori-rel.run-record"] = RUN_RECORD_FORMAT
    format_version: Literal[1] = RUN_RECORD_FORMAT_VERSION
    run_id: str = Field(min_length=1)
    status: Literal["running", "completed", "failed", "cancelled"]
    #: When the run started running and, once it has ended, when it ended.
    started_at: AwareDatetime
    finished_at: AwareDatetime | None = None
    #: The failure's exception class, such as ``PipelineSchemaError``, and its
    #: redacted message. Set only for a failed run.
    error_class: str | None = None
    error_message: str | None = None
    #: The pipeline the run replayed or built; None for a prediction that has
    #: no pipeline, such as a static target.
    pipeline_id: str | None = None
    #: The feature strategy used (``fastdfs``, ``noridfs``, ``agentic`` or
    #: ``saved_program``), or the one planned when the run failed first.
    strategy: str | None = None
    as_of: AwareDatetime | None = None
    observed_through: AwareDatetime | None = None
    engine: str | None = None
    schema_name: str | None = None
    #: The database user the connector read as, as the database names it.
    principal: str | None = None
    #: The pinned warehouse state; None for PostgreSQL, which cannot pin one.
    snapshot: RunSnapshot | None = None
    #: Installed versions: synthefy-nori runs local Nori, and synthefy calls
    #: hosted Nori.
    nori_rel_version: str | None = None
    synthefy_nori_version: str | None = None
    synthefy_version: str | None = None
    #: Where Nori ran: ``local`` or ``hosted``.
    transport: str | None = None
    model: str | None = None
    #: SHA-256 of the checkpoint file local Nori loaded or the pipeline pins.
    checkpoint_sha256: str | None = None
    #: The digest a deployment reported for its model; reserved, since the
    #: hosted client does not report one yet.
    model_version: str | None = None
    #: Candidate context rows read from the database, labelled or not; the
    #: rows Nori read as context after selection; prediction rows returned.
    context_rows_read: int | None = Field(default=None, ge=0)
    context_rows_used: int | None = Field(default=None, ge=0)
    entities_scored: int | None = Field(default=None, ge=0)
    #: The audit hashes of prediction metadata. Passing the first as
    #: ``expected_context_sha256`` reproduces the run on unchanged data.
    context_rows_sha256: str | None = None
    feature_matrix_sha256: str | None = None
    output_table: str | None = None
    rows_written: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _check_outcome_fields(self) -> RunRecord:
        if (self.status == "failed") != (self.error_class is not None):
            raise ValueError("error_class is required for failed runs only")
        if (self.status == "running") != (self.finished_at is None):
            raise ValueError("finished_at is required once a run has ended")
        return self

    @classmethod
    def from_prediction(cls, frame: Any) -> RunRecord | None:
        """Return the record of the run that produced ``frame``, or None."""
        record = getattr(frame, "attrs", {}).get(RUN_RECORD_ATTR)
        return None if record is None else cls.model_validate(record)


#: The fields prediction code reports through ``RunContext``; the run itself
#: sets the others when it ends.
REPORTED_RUN_RECORD_FIELDS: Final = frozenset(RunRecord.model_fields) - {
    "format",
    "format_version",
    "run_id",
    "status",
    "started_at",
    "finished_at",
    "error_class",
    "error_message",
    "strategy",
    "nori_rel_version",
    "synthefy_nori_version",
    "synthefy_version",
    "output_table",
    "rows_written",
}


def get_installed_versions() -> dict[str, str | None]:
    """Return the installed nori-rel, synthefy-nori and synthefy versions."""
    return dict(_read_installed_versions())


@functools.cache
def _read_installed_versions() -> tuple[tuple[str, str | None], ...]:
    """Read the versions once per process: an upgrade on disk changes nothing loaded."""
    versions: list[tuple[str, str | None]] = []
    for field, package in (
        ("nori_rel_version", "nori-rel"),
        ("synthefy_nori_version", "synthefy-nori"),
        ("synthefy_version", "synthefy"),
    ):
        try:
            versions.append((field, importlib.metadata.version(package)))
        except importlib.metadata.PackageNotFoundError:
            versions.append((field, None))
    return tuple(versions)


@runtime_checkable
class RunRecordStore(Protocol):
    """Where a client keeps the record of each run it executes."""

    def write(self, record: RunRecord) -> None:
        """Keep ``record``, replacing any earlier record of the same run."""
        ...

    def read(self, run_id: str) -> RunRecord:
        """Return the record of ``run_id``; ``KeyError`` if there is none."""
        ...


class JsonRunRecordStore:
    """Keep each run's record as ``<directory>/<run_id>.json``.

    The directory is created if missing. Each file is replaced atomically, so
    a reader never sees half of one, and runs in parallel write separate
    files.
    """

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        """Use ``directory``, creating it and its parents if needed."""
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        """Return the store's directory."""
        return f"{type(self).__name__}({str(self.directory)!r})"

    def get_path(self, run_id: str) -> Path:
        """Return the file that holds the record of ``run_id``."""
        if not _STORE_RUN_ID.fullmatch(run_id):
            raise ValueError(f"not a run id: {run_id!r}")
        return self.directory / f"{run_id}.json"

    def write(self, record: RunRecord) -> None:
        """Write ``record`` to its run's file, replacing it atomically."""
        target = self.get_path(record.run_id)
        handle, temporary = tempfile.mkstemp(
            dir=self.directory, prefix=f".{target.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                stream.write(record.model_dump_json(indent=2) + "\n")
            os.replace(temporary, target)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise

    def read(self, run_id: str) -> RunRecord:
        """Return the record of ``run_id``; ``KeyError`` if it has no file."""
        path = self.get_path(run_id)
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise KeyError(f"no record of run {run_id!r} in {self.directory}") from None
        return RunRecord.model_validate_json(text)


def get_run_record_store(
    run_records: str | os.PathLike[str] | RunRecordStore | None,
) -> RunRecordStore | None:
    """Return the store a client's ``run_records`` argument names.

    A path is a directory for a :class:`JsonRunRecordStore`; a store is used
    as given; None keeps no records.
    """
    if run_records is None or isinstance(run_records, RunRecordStore):
        return run_records
    if isinstance(run_records, str | os.PathLike):
        return JsonRunRecordStore(run_records)
    raise TypeError("run_records must be a directory path or a RunRecordStore")


__all__ = [
    "REPORTED_RUN_RECORD_FIELDS",
    "RUN_ID_ATTR",
    "RUN_RECORD_ATTR",
    "RUN_RECORD_FORMAT",
    "RUN_RECORD_FORMAT_VERSION",
    "JsonRunRecordStore",
    "RunRecord",
    "RunRecordStore",
    "RunSnapshot",
    "get_installed_versions",
    "get_run_record_store",
]
