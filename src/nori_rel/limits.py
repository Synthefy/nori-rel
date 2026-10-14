"""Bounded defaults for local relational execution."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RelationalLimits:
    """Resource ceilings for local database snapshots and Nori contexts."""

    max_tables: int = 32
    max_rows_per_table: int = 1_000_000
    max_total_rows: int = 3_000_000
    max_snapshot_bytes: int = 4_000_000_000
    statement_timeout_ms: int = 300_000
    min_training_rows: int = 32
    max_context_rows: int = 60_000
    max_feature_elements: int = 3_000_000
    max_features: int = 512

    def __post_init__(self) -> None:
        """Require every resource ceiling to be positive."""
        for name, value in vars(self).items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
