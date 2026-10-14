"""Connector contract shared by PostgreSQL and future relational backends."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import pandas as pd
from pydantic import BaseModel, ConfigDict

from nori_rel.contracts import DatabaseEngine, SchemaGraph
from nori_rel.pql.context_sql import DEFAULT_MAX_CONTEXT_ROWS


class SnapshotProvenance(BaseModel):
    """Typed extraction identity, separate from the event-time scoring cutoff."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    engine: DatabaseEngine
    extraction_timestamp: datetime
    table_references: dict[str, str]


@dataclass(frozen=True)
class Snapshot:
    """In-memory, point-in-time relational database snapshot."""

    tables: dict[str, pd.DataFrame]
    primary_keys: dict[str, list[str]]
    foreign_keys: list[tuple[str, str, str, str]]
    time_columns: dict[str, str]
    schema: SchemaGraph
    as_of: datetime
    approximate_bytes: int
    # Entity IDs matched by query_filter, read in the same transaction.
    # E.g. users.country='CA' -> (900, 901), intersected with FOR IDs.
    # These select who to score; target_filter selects events used for labels.
    # None means no scoring filter; () means no eligible entity matched.
    scoring_filter_ids: tuple[object, ...] | None = None
    provenance: SnapshotProvenance | None = None


class RelationalConnector(Protocol):
    """Backend contract for bounded relational database reads."""

    def close(self) -> None:
        """Release this source's pool after operations finish; safe to repeat."""
        ...

    def test(self) -> float:
        """Return connection latency in milliseconds."""
        ...

    def discover(self, database_id: str) -> SchemaGraph:
        """Return tables, columns, keys, and capabilities."""
        ...

    def snapshot(
        self,
        database_id: str,
        as_of: datetime,
        *,
        time_columns: dict[str, str] | None = None,
    ) -> Snapshot:
        """Return bounded tables with optional per-call timestamp overrides."""
        ...

    def prepare_context(
        self,
        database_id: str,
        query: str,
        *,
        anchor_times: Sequence[datetime],
        anchor_time: datetime | None,
        observed_through: datetime,
        entity_created_at: str,
        max_context_rows: int = DEFAULT_MAX_CONTEXT_ROWS,
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        """Return bounded labeled context and candidate/usable/dropped counts.

        Inputs are PQL, aware cutoffs/anchors, and a registered creation column.
        DataFrame columns: context_row_id, entity_id, anchor_time, label.
        Counts: candidate_rows, usable_rows, dropped_unknown_labels (integers).
        """
        ...

    def prepare_inputs(
        self, database_id: str, query: str, **options
    ) -> tuple[Snapshot, pd.DataFrame, dict[str, int]]:
        """Return same-snapshot feature tables, labeled context rows and counts.

        Keyword options match prepare_context; query_filter is an optional
        entity predicate. Its matches are in snapshot.scoring_filter_ids.
        Context columns are
        context_row_id, entity_id, anchor_time, label.
        """
        ...
