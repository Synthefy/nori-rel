"""Two layers of point-in-time protection for agent-written SQL."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from ..dfs.noridfs.contracts import DatabaseView
from .contracts import CUTOFF_ALIAS, FeatureProgram, FeatureQuery
from .executor import ProgramExecutor, frames_match, subset_view

_BLOCKED = (
    "alter",
    "attach",
    "copy",
    "create",
    "delete",
    "drop",
    "export",
    "insert",
    "install",
    "load",
    "pragma",
    "update",
)
#: `x >= cutoff`, `cutoff <= x`, and `cutoff + INTERVAL` all read the future.
_FORWARD = re.compile(
    rf"(?:>=|>)\s*(?:\w+\.)?{CUTOFF_ALIAS}\b"
    rf"|{CUTOFF_ALIAS}\s*(?:<=|<)\s*\w"
    rf"|{CUTOFF_ALIAS}\s*\+\s*INTERVAL",
    re.IGNORECASE,
)
_BACKWARD = re.compile(rf"<\s*(?:\w+\.)?{CUTOFF_ALIAS}\b", re.IGNORECASE)


@dataclass(frozen=True)
class LeakFinding:
    """Why a query or program was rejected by a guard layer."""

    query: str
    reason: str


def blocked_findings(name: str, sql: str) -> tuple[LeakFinding, ...]:
    """Reject statements that write, load files, or change session state."""
    lowered = f" {_strip_comments(sql).lower()} "
    return tuple(
        LeakFinding(name, f"blocked statement {keyword!r}")
        for keyword in _BLOCKED
        if re.search(rf"\b{keyword}\b", lowered)
    )


def static_findings(
    query: FeatureQuery, database: DatabaseView
) -> tuple[LeakFinding, ...]:
    """Reject blocked statements and syntactically forward-looking filters.

    Flags ``>=`` or ``>`` against the cutoff, ``cutoff + INTERVAL``, and any
    temporal table read without a ``< cutoff`` filter. Cheap and advisory;
    the session trusts ``point_in_time_findings``.
    """
    text = _strip_comments(query.sql)
    findings = list(blocked_findings(query.name, query.sql))
    forward = _FORWARD.search(text)
    if forward:
        findings.append(
            LeakFinding(
                query.name,
                f"forward-looking cutoff comparison {forward.group(0).strip()!r}; "
                f"filter events with `time_col < e.{CUTOFF_ALIAS}`",
            )
        )
    temporal = [
        table
        for table in database.time_columns
        if re.search(rf"\b{re.escape(table)}\b", text, re.IGNORECASE)
    ]
    if temporal and not _BACKWARD.search(text):
        findings.append(
            LeakFinding(
                query.name,
                f"reads temporal tables {sorted(temporal)} without a "
                f"`< {CUTOFF_ALIAS}` filter",
            )
        )
    return tuple(findings)


def point_in_time_findings(
    program: FeatureProgram,
    database: DatabaseView,
    anchors: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    entity_column: str,
    sample: Sequence[int],
    timeout_seconds: float,
    fanout_limit: int = 4,
    max_categorical_cardinality: int = 100,
    allow_text: bool = True,
) -> tuple[LeakFinding, ...]:
    """Recompute sampled anchors on truncated snapshots and diff the features.

    For each sampled anchor the snapshot loses every temporal row at or after
    that anchor's cutoff. A feature that changes was reading the future, no
    matter how the SQL was written.
    """
    if not len(sample):
        return ()
    cutoff_column = anchors.columns[1]
    findings: list[LeakFinding] = []
    seen: set[str] = set()
    for position in sample:
        anchor = anchors.iloc[[position]]
        truncated = subset_view(database, pd.Timestamp(anchor[cutoff_column].iloc[0]))
        executor = ProgramExecutor(
            truncated,
            entity_column=entity_column,
            timeout_seconds=timeout_seconds,
            fanout_limit=fanout_limit,
            max_categorical_cardinality=max_categorical_cardinality,
            allow_text=allow_text,
        )
        try:
            recomputed = executor.run(program, anchor).frame
        finally:
            executor.close()
        matches = frames_match(reference.iloc[[position]], recomputed)
        for column, same in matches.items():
            if same or column in seen:
                continue
            seen.add(column)
            findings.append(
                LeakFinding(
                    column.split("__", 1)[0],
                    f"feature {column!r} changes when rows at or after the anchor "
                    f"cutoff are removed (anchor {position})",
                )
            )
    return tuple(findings)


def _strip_comments(sql: str) -> str:
    text = re.sub(r"--[^\n]*", "", sql)
    return re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)


__all__ = [
    "LeakFinding",
    "blocked_findings",
    "point_in_time_findings",
    "static_findings",
]
