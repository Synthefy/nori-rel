"""Connector contract shared by PostgreSQL and future relational backends."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import pandas as pd
from pydantic import BaseModel, ConfigDict

from nori_rel.connectors.errors import SnapshotLimitError
from nori_rel.contracts import DatabaseEngine, SchemaGraph
from nori_rel.limits import RelationalLimits
from nori_rel.pql.context_sql import DEFAULT_MAX_CONTEXT_ROWS


def get_context_read_row_cap(
    limits: RelationalLimits, max_context_rows: int | None
) -> int | None:
    """Return the row cap of a context read, or None for an uncapped read.

    An integer ``max_context_rows`` is the newest-candidates cap and must fit
    the context, per-table and total row limits. None reads every candidate
    that can have a known label; that read has no row cap of its own and is
    bounded by ``max_snapshot_bytes`` and the statement timeout.

    Raises:
        SnapshotLimitError: An integer cap that is not in range.
    """
    if max_context_rows is None:
        return None
    ceiling = min(
        limits.max_context_rows, limits.max_rows_per_table, limits.max_total_rows
    )
    if type(max_context_rows) is not int or not 0 < max_context_rows <= ceiling:
        raise SnapshotLimitError(f"max_context_rows must be an integer in 1..{ceiling}")
    return max_context_rows


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
        max_context_rows: int | None = DEFAULT_MAX_CONTEXT_ROWS,
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        """Return bounded labeled context and candidate/usable/dropped counts.

        Inputs are PQL, aware cutoffs/anchors, and a registered creation column.
        max_context_rows=None returns every candidate that can have a known
        label, for the caller to choose from.
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
        candidate_sql optionally narrows an uncapped (max_context_rows=None)
        read with the caller's SELECT over candidates(entity_id, anchor_time).
        Context columns are
        context_row_id, entity_id, anchor_time, label.
        """
        ...
