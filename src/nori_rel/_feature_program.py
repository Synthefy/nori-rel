"""Validated saved feature programs and their replay constraints."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from nori_rel.contracts import SchemaGraph, StrictModel
from nori_rel.feature.agentic.contracts import FeatureProgram, FeatureQuery

PROGRAM_RECORD_FORMAT = "nori-rel.feature-program"
PROGRAM_RECORD_VERSION = 2


class SavedFeatureProgram(StrictModel):
    """Bind SQL features to a PQL task, schema and ordered NoriDFS columns."""

    format: Literal["nori-rel.feature-program"] = PROGRAM_RECORD_FORMAT
    version: Literal[2] = PROGRAM_RECORD_VERSION
    baseline: Literal["noridfs"] = "noridfs"
    pql_query: str = Field(min_length=1)
    entity_created_at: str = Field(min_length=1)
    schema_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline_columns: tuple[str, ...] = Field(min_length=1)
    queries: tuple[FeatureQuery, ...] = Field(min_length=1)
    drop_baseline: tuple[str, ...] = ()
    target_transform: Literal["auto"] = "auto"

    @model_validator(mode="after")
    def validate_program_columns(self) -> Self:
        """Reject invalid programs and inconsistent baseline column names."""
        self.to_program()
        if len(self.baseline_columns) != len(set(self.baseline_columns)):
            raise ValueError("baseline_columns must not repeat a column")
        if set(self.drop_baseline).difference(self.baseline_columns):
            raise ValueError("drop_baseline must name saved baseline columns")
        return self

    def to_program(self) -> FeatureProgram:
        """Return the executable program without discarding its target transform."""
        return FeatureProgram(
            self.queries, self.drop_baseline, target_transform=self.target_transform
        )

    def validate_query(self, query: str, entity_created_at: str) -> None:
        """Reject another PQL query before the database is read."""
        if query != self.pql_query:
            raise ValueError(
                "saved feature program does not match this PQL query; run a new search"
            )
        if entity_created_at != self.entity_created_at:
            raise ValueError(
                "saved feature program does not match entity_created_at; "
                "run a new search"
            )

    def validate_schema(self, schema: SchemaGraph) -> None:
        """Reject changed tables, columns, keys or timestamp mappings."""
        if compute_schema_digest(schema) != self.schema_digest:
            raise ValueError(
                "saved feature program does not match this database schema; "
                "run a new search"
            )


def compute_schema_digest(schema: SchemaGraph) -> str:
    """Hash schema structure, not source handles, row data or discovery time."""
    payload = schema.model_dump(mode="json", include={"schema_name", "tables"})
    payload["tables"].sort(key=lambda table: table["name"])
    for table in payload["tables"]:
        table["columns"].sort(key=lambda column: column["name"])
        table["foreign_keys"].sort(key=lambda key: json.dumps(key, sort_keys=True))
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def program_record(
    program: FeatureProgram,
    *,
    query: str,
    entity_created_at: str,
    schema: SchemaGraph,
    baseline_columns: Sequence[str],
) -> dict[str, Any]:
    """Return a JSON-ready program with the task and schema used by its search."""
    if program.target_transform != "auto":
        raise ValueError("saved client programs require target_transform='auto'")
    return SavedFeatureProgram(
        pql_query=query,
        entity_created_at=entity_created_at,
        schema_digest=compute_schema_digest(schema),
        baseline_columns=tuple(baseline_columns),
        queries=program.queries,
        drop_baseline=program.drop_baseline,
    ).model_dump(mode="json")


def program_from_record(record: Mapping[str, Any]) -> SavedFeatureProgram:
    """Validate a saved program; refuse old records without replay constraints."""
    if not isinstance(record, Mapping):
        raise ValueError(
            "feature_program must be the agentic_program record of a prediction"
        )
    if (
        record.get("format") != PROGRAM_RECORD_FORMAT
        or record.get("version") != PROGRAM_RECORD_VERSION
    ):
        raise ValueError(
            f"feature_program must be a {PROGRAM_RECORD_FORMAT!r} record, "
            f"version {PROGRAM_RECORD_VERSION}; run a new search"
        )
    return SavedFeatureProgram.model_validate(record)
