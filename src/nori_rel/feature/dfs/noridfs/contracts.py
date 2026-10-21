"""Backend-neutral contracts for native relational feature generation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import quote

import pandas as pd

ColumnKind = Literal["numeric", "categorical", "datetime"]
PathDirection = Literal["bridge", "descendant", "parent", "sibling"]
FeatureOperation = Literal[
    "anchor",
    "direct",
    "count",
    "distinct",
    "entropy",
    "first",
    "last",
    "last_share",
    "lookup",
    "match_count",
    "match_previous",
    "match_recency",
    "max",
    "mean",
    "mean_age",
    "missing_fraction",
    "min",
    "mode",
    "mode_share",
    "oldest_age",
    "recency",
    "std",
    "sum",
]

_HOUR_NS = 3_600_000_000_000


@dataclass(frozen=True, order=True)
class ForeignKey:
    """One single-column child-to-parent relationship."""

    child_table: str
    child_column: str
    parent_table: str
    parent_column: str

    @property
    def identity(self) -> str:
        """Return a stable, role-aware relationship identity."""
        return (
            f"{self.child_table}.{self.child_column}->"
            f"{self.parent_table}.{self.parent_column}"
        )


@dataclass(frozen=True)
class DatabaseView:
    """In-memory relational frames and their structural metadata.

    The shape mirrors the public fields on ``Snapshot`` without importing the
    connector package, so another source can construct the same view.
    """

    tables: Mapping[str, pd.DataFrame]
    primary_keys: Mapping[str, str]
    foreign_keys: tuple[ForeignKey, ...] = ()
    time_columns: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_parts(
        cls,
        *,
        tables: Mapping[str, pd.DataFrame],
        primary_keys: Mapping[str, str | Sequence[str]],
        foreign_keys: Sequence[ForeignKey | tuple[str, str, str, str]] = (),
        time_columns: Mapping[str, str] | None = None,
    ) -> DatabaseView:
        """Build a view from Snapshot-like mappings and fail on composite keys."""
        keys: dict[str, str] = {}
        for table, value in primary_keys.items():
            if isinstance(value, str):
                keys[table] = value
                continue
            columns = tuple(value)
            if len(columns) != 1:
                raise ValueError(
                    f"native features require one primary-key column for {table!r}"
                )
            keys[table] = columns[0]
        relations = tuple(
            relation if isinstance(relation, ForeignKey) else ForeignKey(*relation)
            for relation in foreign_keys
        )
        view = cls(
            tables=dict(tables),
            primary_keys=keys,
            foreign_keys=tuple(sorted(set(relations))),
            time_columns=dict(time_columns or {}),
        )
        view.validate()
        return view

    def validate(self) -> None:
        """Validate only structural facts needed by the native executor."""
        for name, frame in self.tables.items():
            if not isinstance(name, str) or not name:
                raise ValueError("table names must be non-empty strings")
            if not all(isinstance(column, str) for column in frame.columns):
                raise TypeError(f"columns in {name!r} must be strings")
            if not frame.columns.is_unique:
                raise ValueError(f"columns in {name!r} must be unique")
        for table, column in self.primary_keys.items():
            self._require_column(table, column, "primary key")
        for table, column in self.time_columns.items():
            self._require_column(table, column, "time column")
        for relation in self.foreign_keys:
            self._require_column(
                relation.child_table, relation.child_column, "foreign key"
            )
            self._require_column(
                relation.parent_table, relation.parent_column, "referred key"
            )
            if self.primary_keys.get(relation.parent_table) != relation.parent_column:
                raise ValueError(
                    "native features require foreign keys to reference the "
                    f"parent primary key: {relation.identity}"
                )

    def _require_column(self, table: str, column: str, role: str) -> None:
        if table not in self.tables:
            raise ValueError(f"{role} references missing table {table!r}")
        if column not in self.tables[table]:
            raise ValueError(f"{role} references missing column {table}.{column}")

    @property
    def schema_digest(self) -> str:
        """Hash the order-independent schema used to validate transforms."""
        schema = {
            "tables": {
                name: sorted(
                    (str(column), str(frame[column].dtype)) for column in frame.columns
                )
                for name, frame in sorted(self.tables.items())
            },
            "primary_keys": sorted(self.primary_keys.items()),
            "foreign_keys": sorted(relation.identity for relation in self.foreign_keys),
            "time_columns": sorted(self.time_columns.items()),
        }
        encoded = json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.blake2s(encoded, digest_size=16).hexdigest()


@dataclass(frozen=True)
class AnchorDefinition:
    """How prediction rows identify an entity and point-in-time cutoff."""

    entity_table: str
    entity_column: str
    cutoff_column: str | None
    horizon_ns: int | None = None

    @classmethod
    def create(
        cls,
        *,
        entity_table: str,
        entity_column: str,
        cutoff_column: str | None,
        horizon: str | pd.Timedelta | None = None,
    ) -> AnchorDefinition:
        """Normalize an optional prediction horizon to nanoseconds."""
        horizon_ns = None
        if horizon is not None:
            delta = pd.Timedelta(horizon)
            if delta <= pd.Timedelta(0):
                raise ValueError("horizon must be positive")
            horizon_ns = int(delta.value)
        if cutoff_column is None and horizon_ns is not None:
            raise ValueError("horizon requires a cutoff column")
        return cls(entity_table, entity_column, cutoff_column, horizon_ns)


@dataclass(frozen=True)
class NativeFeatureConfig:
    """Small, deterministic native feature-search budget."""

    max_depth: int = 2
    max_paths: int = 24
    max_features: int = 256
    max_direct_features: int = 64
    max_direct_categorical_cardinality: int | None = None
    max_categorical_cardinality: int = 32
    window_multipliers: tuple[int, ...] = (1, 4, 16)
    include_missing_in_modes: bool = False
    include_missing_fractions: bool = False

    def __post_init__(self) -> None:
        """Validate the bounded search configuration."""
        if self.max_depth < 1:
            raise ValueError("max_depth must be positive")
        if self.max_paths < 1:
            raise ValueError("max_paths must be positive")
        if self.max_features < 1:
            raise ValueError("max_features must be positive")
        if self.max_direct_features < 0:
            raise ValueError("max_direct_features must be non-negative")
        if (
            self.max_direct_categorical_cardinality is not None
            and self.max_direct_categorical_cardinality < 1
        ):
            raise ValueError(
                "max_direct_categorical_cardinality must be positive when set"
            )
        if self.max_categorical_cardinality < 1:
            raise ValueError("max_categorical_cardinality must be positive")
        if any(value < 1 for value in self.window_multipliers):
            raise ValueError("window multipliers must be positive")
        if len(set(self.window_multipliers)) != len(self.window_multipliers):
            raise ValueError("window multipliers must be unique")


@dataclass(frozen=True)
class SiblingPathConfig:
    """Opt-in sibling paths: up to a parent, then down to that parent's other rows.

    A sibling path goes from the entity to a parent it references (a page view to
    its device) and back down to the time-stamped rows that reference the same
    parent (that device's views, or its signups). Its features describe those
    rows strictly before each row's cutoff:

    - ``count`` over every earlier row and over each window in ``window_ns``;
    - ``recency`` and ``oldest_age``: days since the latest and the earliest one;
    - ``distinct``: distinct values of each of their foreign keys, and distinct
      timestamps;
    - with ``include_match_features``, for each foreign key the entity and those
      rows share (both a view and its device's earlier views name a drug):
      ``match_count`` (earlier rows with the entity's own value),
      ``match_recency`` (days since the latest one) and ``match_previous``
      (1 when the latest earlier timestamp had it).
    - with ``include_numeric_aggregates``, for each numeric column of those rows
      (not a key or the time column): ``sum`` and ``mean`` over every earlier
      row and over each window in ``window_ns``, and ``min`` and ``max`` over
      every earlier row. A drug's earlier weekly claims give its claims volume
      over the last 12 weeks with ``window_ns`` set to 84 days.
      ``numeric_columns`` names the ``table.column`` pairs to aggregate when not
      every numeric column should be: each one adds two features per window.

    With event-time rows (``entity_time``) each row's cutoff is its own time, so
    these are the device's views before this view, including earlier the same
    day. With a shared anchor they also include the entity's own row when it is
    before the anchor. Off unless passed to ``NoriDFS(sibling_paths=...)``.
    ``parent_tables`` limits the parents to go through, e.g. ``("devices",)``.

    ``parent_tables`` may also name a table one step above a parent, to group by
    that parent's attribute: with ``("drug_class",)``, a signup reaches the
    earlier signups and paid subscriptions of every drug in its drug's class,
    through ``drug.drug_class_id``. Such grouped paths are planned only for
    tables named here, and not for a table the entity already references.

    ``excluded_paths`` leaves out (parent, child table) pairs, the parent being
    the grouping table for a grouped path. Use it when a child table holds only
    some of a parent's rows in a way that depends on the future: a views table
    restricted to devices that later sign up describes each device's history
    completely, but a drug class's views only partly.
    """

    max_paths: int = 8
    window_ns: tuple[int, ...] = (_HOUR_NS, 6 * _HOUR_NS, 24 * _HOUR_NS, 168 * _HOUR_NS)
    include_match_features: bool = True
    parent_tables: tuple[str, ...] | None = None
    include_numeric_aggregates: bool = False
    numeric_columns: tuple[str, ...] | None = None
    excluded_paths: tuple[tuple[str, str], ...] = ()
    #: The most features the sibling family takes from the budget; ``None`` lets
    #: it take as many as the budget allows, before any other path.
    max_features: int | None = None

    def __post_init__(self) -> None:
        """Require a positive path cap, positive unique windows and named parents."""
        if self.parent_tables is not None and not self.parent_tables:
            raise ValueError("parent_tables must name at least one table, or be None")
        if self.max_paths < 1:
            raise ValueError("max_paths must be positive")
        if self.max_features is not None and self.max_features < 1:
            raise ValueError("max_features must be positive, or None")
        if any(window <= 0 for window in self.window_ns):
            raise ValueError("sibling windows must be positive")
        if len(set(self.window_ns)) != len(self.window_ns):
            raise ValueError("sibling windows must be unique")


def get_sibling_edges(
    path: RelationPath,
) -> tuple[ForeignKey, ForeignKey | None, ForeignKey]:
    """Return a sibling path's (up edge, grouping edge or None, down edge).

    ``views.device_id -> devices`` then ``views.device_id -> devices`` is a plain
    sibling path; ``signups.drug_id -> drug``, ``drug.drug_class_id ->
    drug_class`` then ``paid_subscriptions.drug_id -> drug`` groups the parent's
    rows by its drug class.
    """
    if len(path.edges) == 3:
        up, group, down = path.edges
        return up, group, down
    up, down = path.edges
    return up, None, down


@dataclass(frozen=True)
class RelationPath:
    """One bounded descendant, parent, association-to-parent or sibling path."""

    entity_table: str
    edges: tuple[ForeignKey, ...]
    direction: PathDirection = "descendant"

    def __post_init__(self) -> None:
        """Require a non-empty path whose edges match its direction."""
        if self.direction not in {"bridge", "descendant", "parent", "sibling"}:
            raise ValueError(f"unknown path direction {self.direction!r}")
        if not self.edges:
            raise ValueError("relation paths require at least one edge")
        if self.is_sibling:
            if len(self.edges) not in (2, 3):
                raise ValueError(
                    "sibling paths require an up edge, an optional grouping edge "
                    "and a down edge"
                )
            up, group, down = get_sibling_edges(self)
            if up.child_table != self.entity_table:
                raise ValueError("sibling path must start at the entity table")
            if (down.parent_table, down.parent_column) != (
                up.parent_table,
                up.parent_column,
            ):
                raise ValueError("sibling path edges must share their parent")
            if group is not None and group.child_table != up.parent_table:
                raise ValueError(
                    "a sibling path's grouping edge must start at the shared parent"
                )
            return
        if self.is_bridge:
            if len(self.edges) < 2:
                raise ValueError("bridge paths require a descendant and a parent edge")
            first, *parents = self.edges
            if first.parent_table != self.entity_table:
                raise ValueError("bridge path must start at the entity table")
            current = first.child_table
            visited = {self.entity_table, current}
            for edge in parents:
                if edge.child_table != current:
                    raise ValueError("bridge path parent edges are not contiguous")
                current = edge.parent_table
                if current in visited:
                    raise ValueError("relation paths cannot revisit a table")
                visited.add(current)
            return
        current = self.entity_table
        visited = {current}
        for edge in self.edges:
            expected = (
                edge.parent_table
                if self.direction == "descendant"
                else edge.child_table
            )
            if expected != current:
                raise ValueError("relation path edges are not contiguous")
            current = (
                edge.child_table
                if self.direction == "descendant"
                else edge.parent_table
            )
            if current in visited:
                raise ValueError("relation paths cannot revisit a table")
            visited.add(current)

    @property
    def terminal_table(self) -> str:
        """Return the final table reached by the path."""
        edge = self.edges[-1]
        if self.direction in {"descendant", "sibling"}:
            return edge.child_table
        return edge.parent_table

    @property
    def is_parent(self) -> bool:
        """Return whether every edge is traversed child-to-parent."""
        return self.direction == "parent"

    @property
    def is_bridge(self) -> bool:
        """Return whether one descendant edge is followed by parent edges."""
        return self.direction == "bridge"

    @property
    def is_sibling(self) -> bool:
        """Return whether the path goes up to a parent and down to its rows."""
        return self.direction == "sibling"

    @property
    def identity(self) -> str:
        """Return a canonical path identity."""
        if self.is_bridge:
            first, *parents = self.edges
            upward = "/".join(f"U:{edge.identity}" for edge in parents)
            return f"{self.entity_table}|D:{first.identity}|{upward}"
        if self.is_sibling:
            up, group, down = get_sibling_edges(self)
            if group is not None:
                return (
                    f"{self.entity_table}|S:{up.identity}/G:{group.identity}"
                    f"|{down.child_table}.{down.child_column}"
                )
            return (
                f"{self.entity_table}|S:{up.identity}|"
                f"{down.child_table}.{down.child_column}"
            )
        if self.is_parent:
            steps = "/".join(edge.identity for edge in self.edges)
            return f"{self.entity_table}=>{steps}"
        steps = "<-".join(
            f"{edge.child_table}.{edge.child_column}" for edge in self.edges
        )
        return f"{self.entity_table}<-{steps}"


@dataclass(frozen=True)
class FeatureSpec:
    """One output column in a frozen native feature plan."""

    name: str
    operation: FeatureOperation
    kind: ColumnKind
    table: str
    column: str | None = None
    path: RelationPath | None = None
    window_ns: int | None = None
    #: For a pair feature, the second bridge path and its column. The pair's
    #: rows are those both paths reach from the same association row.
    paired_path: RelationPath | None = None
    paired_column: str | None = None

    @classmethod
    def anchor(
        cls,
        table: str,
        component: str,
        kind: ColumnKind,
    ) -> FeatureSpec:
        """Construct a feature known directly from each prediction anchor."""
        name = f"nr|anchor|{_escape(component)}"
        return cls(name, "anchor", kind, table, component)

    @classmethod
    def direct(cls, table: str, column: str, kind: ColumnKind) -> FeatureSpec:
        """Construct a direct entity-column feature."""
        name = f"nr|direct|{_escape(table)}.{_escape(column)}"
        return cls(name, "direct", kind, table, column)

    @classmethod
    def lookup(
        cls,
        path: RelationPath,
        column: str,
        kind: ColumnKind,
    ) -> FeatureSpec:
        """Construct one many-to-one parent lookup feature."""
        if not path.is_parent:
            raise ValueError("lookup features require a parent path")
        name = f"nr|path|{_escape(path.identity)}|{_escape(column)}|lookup|all"
        return cls(name, "lookup", kind, path.terminal_table, column, path)

    @classmethod
    def aggregate(
        cls,
        path: RelationPath,
        operation: FeatureOperation,
        *,
        column: str | None = None,
        kind: ColumnKind = "numeric",
        window_ns: int | None = None,
    ) -> FeatureSpec:
        """Construct a path aggregate with a canonical readable name."""
        source = "rows" if column is None else _escape(column)
        window = "all" if window_ns is None else f"{window_ns}ns"
        name = f"nr|path|{_escape(path.identity)}|{source}|{operation}|{window}"
        return cls(
            name,
            operation,
            kind,
            path.terminal_table,
            column,
            path,
            window_ns,
        )

    @classmethod
    def pair_aggregate(
        cls,
        path: RelationPath,
        paired_path: RelationPath,
        operation: FeatureOperation,
        *,
        column: str,
        paired_column: str,
        kind: ColumnKind = "numeric",
    ) -> FeatureSpec:
        """Construct a composition feature of value pairs from two bridge paths.

        Both paths start with the same edge down to an association row and go up
        from it, so each association row gives one pair: ``column`` from the end
        of ``path`` and ``paired_column`` from the end of ``paired_path``.
        """
        if not (path.is_bridge and paired_path.is_bridge):
            raise ValueError("pair features need two bridge paths")
        if path.edges[0] != paired_path.edges[0]:
            raise ValueError("pair features need paths from the same association rows")
        name = (
            f"nr|pair|{_escape(path.identity)}|{_escape(column)}"
            f"|{_escape(paired_path.identity)}|{_escape(paired_column)}|{operation}|all"
        )
        return cls(
            name,
            operation,
            kind,
            path.terminal_table,
            column,
            path,
            None,
            paired_path,
            paired_column,
        )


@dataclass(frozen=True)
class FeaturePlan:
    """Fit-frozen paths and output columns used for every transform."""

    schema_digest: str
    anchors: AnchorDefinition
    config: NativeFeatureConfig
    paths: tuple[RelationPath, ...]
    features: tuple[FeatureSpec, ...]

    @property
    def columns(self) -> tuple[str, ...]:
        """Return the exact fitted output schema."""
        return tuple(feature.name for feature in self.features)

    @property
    def categorical_columns(self) -> tuple[str, ...]:
        """Return the fitted categorical subset."""
        return tuple(
            feature.name for feature in self.features if feature.kind == "categorical"
        )

    @property
    def digest(self) -> str:
        """Return a stable digest suitable for persistent cache keys."""
        payload = {
            "schema": self.schema_digest,
            "anchors": self.anchors.__dict__,
            "config": self.config.__dict__,
            "paths": [path.identity for path in self.paths],
            "features": [_get_digest_entry(feature) for feature in self.features],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.blake2s(encoded, digest_size=16).hexdigest()


@dataclass(frozen=True)
class NativeFeatureMatrix:
    """One aligned native feature matrix and its categorical schema."""

    frame: pd.DataFrame
    categorical_columns: tuple[str, ...]
    plan_digest: str


def _get_digest_entry(feature: FeatureSpec) -> dict[str, object]:
    """Return the fields of one feature that a plan digest hashes.

    The pair fields are left out when unset, so plans without pair features keep
    their digests.
    """
    entry: dict[str, object] = {
        "name": feature.name,
        "operation": feature.operation,
        "kind": feature.kind,
        "table": feature.table,
        "column": feature.column,
        "path": feature.path.identity if feature.path is not None else None,
        "window_ns": feature.window_ns,
    }
    if feature.paired_path is not None:
        entry["paired_path"] = feature.paired_path.identity
        entry["paired_column"] = feature.paired_column
    return entry


def _escape(value: str) -> str:
    return quote(value, safe="")


__all__ = [
    "AnchorDefinition",
    "ColumnKind",
    "DatabaseView",
    "FeaturePlan",
    "FeatureSpec",
    "ForeignKey",
    "NativeFeatureConfig",
    "NativeFeatureMatrix",
    "PathDirection",
    "RelationPath",
    "SiblingPathConfig",
]
