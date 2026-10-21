"""Native point-in-time relational feature execution."""

from __future__ import annotations

import dataclasses
import heapq
import math
from collections import Counter, defaultdict
from collections.abc import Mapping

import numpy as np
import pandas as pd

from .contracts import (
    AnchorDefinition,
    DatabaseView,
    FeaturePlan,
    FeatureSpec,
    NativeFeatureConfig,
    NativeFeatureMatrix,
    RelationPath,
    SiblingPathConfig,
    get_sibling_edges,
)
from .planner import (
    build_feature_plan,
    get_sibling_match_columns,
)

_ROW = "__nr_row__"
_ENTITY = "__nr_entity__"
_NODE = "__nr_node__"
_JOIN = "__nr_join__"
_NEXT = "__nr_next__"
_TIME = "__nr_time__"
_STEP_TIME = "__nr_step_time__"
_AT = "__nr_at__"
_TIE = "__nr_tie__"
#: A path row's own time: the terminal (descendant) or association (bridge) row's
#: time. A path's time is the latest along it, so a signup's journey views all
#: carry the signup's time; their own times keep their order for first and last.
_OWN_TIME = "__nr_own_time__"
_DAY_NS = float(pd.Timedelta(days=1).value)
_MISSING_CATEGORY = "__nori_rel_missing__"
_CATEGORY_ESCAPE = "__nori_rel_escape__:"


class NativeRelationalFeaturizer:
    """Execute one immutable, fit-frozen relational feature plan."""

    def __init__(self, plan: FeaturePlan) -> None:
        """Store an immutable fitted plan."""
        self.plan = plan

    @classmethod
    def fit(
        cls,
        database: DatabaseView,
        anchors: AnchorDefinition,
        config: NativeFeatureConfig | None = None,
        *,
        excluded_columns: Mapping[str, set[str]] | None = None,
        sibling_paths: SiblingPathConfig | None = None,
        include_composition_features: bool = False,
        composition_pairs: tuple[tuple[str, str], ...] = (),
    ) -> NativeRelationalFeaturizer:
        """Freeze paths, operations, windows, and output order from fit schema.

        ``sibling_paths`` opts into the sibling family; see ``SiblingPathConfig``.
        ``include_composition_features`` opts into first values, and entropy,
        the most common and last value's shares and distinct counts of
        categorical columns, over the rows a path reaches; ``composition_pairs``
        adds the same over pairs of columns. See ``build_feature_plan``.
        """
        excluded = {
            table: set(columns) for table, columns in (excluded_columns or {}).items()
        }
        plan = build_feature_plan(
            database,
            anchors,
            config or NativeFeatureConfig(),
            excluded_columns=excluded,
            sibling_paths=sibling_paths,
            include_composition_features=include_composition_features,
            composition_pairs=composition_pairs,
        )
        return cls(plan)

    def transform(
        self, database: DatabaseView, anchor_frame: pd.DataFrame
    ) -> NativeFeatureMatrix:
        """Build one row per anchor using only rows strictly before its cutoff."""
        database.validate()
        if database.schema_digest != self.plan.schema_digest:
            raise ValueError("database schema does not match the fitted feature plan")
        anchors = _prepare_anchors(anchor_frame, self.plan.anchors)
        values: dict[str, pd.Series | np.ndarray] = {}
        anchor_specs = [
            feature for feature in self.plan.features if feature.operation == "anchor"
        ]
        values.update(_anchor_features(anchors, anchor_specs))
        direct = [
            feature for feature in self.plan.features if feature.operation == "direct"
        ]
        values.update(_direct_features(database, anchors, self.plan.anchors, direct))

        by_path: dict[RelationPath, list[FeatureSpec]] = defaultdict(list)
        by_pair: dict[
            tuple[RelationPath, RelationPath, str | None, str | None],
            list[FeatureSpec],
        ] = defaultdict(list)
        for feature in self.plan.features:
            if feature.paired_path is not None and feature.path is not None:
                pair = (
                    feature.path,
                    feature.paired_path,
                    feature.column,
                    feature.paired_column,
                )
                by_pair[pair].append(feature)
            elif feature.path is not None:
                by_path[feature.path].append(feature)
        for path in self.plan.paths:
            specs = by_path.get(path)
            if specs and path.is_sibling:
                values.update(
                    _get_sibling_path_features(database, anchors, path, specs)
                )
            elif specs:
                values.update(
                    _path_features(
                        database,
                        anchors,
                        path,
                        specs,
                        self.plan.config.include_missing_in_modes,
                    )
                )

        for (path, paired_path, _, _), specs in by_pair.items():
            values.update(
                _get_pair_path_features(
                    database,
                    anchors,
                    path,
                    paired_path,
                    specs,
                    self.plan.config.include_missing_in_modes,
                )
            )

        missing = [column for column in self.plan.columns if column not in values]
        if missing:
            raise RuntimeError(
                f"native executor omitted planned features: {missing[:3]}"
            )
        frame = pd.DataFrame(values).reindex(columns=list(self.plan.columns))
        if len(frame) != len(anchor_frame):
            raise RuntimeError("native feature row count changed during execution")
        numeric_columns = frame.select_dtypes(include=[np.number]).columns
        frame[numeric_columns] = frame[numeric_columns].replace(
            [np.inf, -np.inf], np.nan
        )
        frame.reset_index(drop=True, inplace=True)
        return NativeFeatureMatrix(
            frame=frame,
            categorical_columns=self.plan.categorical_columns,
            plan_digest=self.plan.digest,
        )


def _prepare_anchors(frame: pd.DataFrame, definition: AnchorDefinition) -> pd.DataFrame:
    required = [definition.entity_column]
    if definition.cutoff_column is not None:
        required.append(definition.cutoff_column)
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"anchor columns are missing: {missing}")
    anchors = pd.DataFrame(
        {
            _ROW: np.arange(len(frame), dtype=np.int64),
            _ENTITY: frame[definition.entity_column].reset_index(drop=True),
        }
    )
    if anchors[_ENTITY].isna().any():
        raise ValueError("anchor entity keys must be present")
    if definition.cutoff_column is not None:
        anchors[_AT] = _time_ns(frame[definition.cutoff_column])
        if anchors[_AT].isna().any():
            raise ValueError("anchor cutoffs must be valid timestamps")
    return anchors


def _anchor_features(
    anchors: pd.DataFrame,
    specs: list[FeatureSpec],
) -> dict[str, pd.Series | np.ndarray]:
    """Materialize fitted entity-key and compact cutoff features."""
    if not specs:
        return {}
    timestamps = (
        pd.to_datetime(anchors[_AT], unit="ns", utc=True) if _AT in anchors else None
    )
    output: dict[str, pd.Series | np.ndarray] = {}
    for spec in specs:
        if spec.column == "entity_key":
            output[spec.name] = anchors[_ENTITY].astype(object).reset_index(drop=True)
        elif timestamps is None:
            raise RuntimeError(f"anchor feature {spec.column!r} requires a cutoff")
        elif spec.column == "cutoff_epoch_days":
            output[spec.name] = anchors[_AT].to_numpy(dtype=float) / _DAY_NS
        elif spec.column == "cutoff_year":
            output[spec.name] = timestamps.dt.year.to_numpy(dtype=float)
        elif spec.column == "cutoff_month":
            output[spec.name] = timestamps.dt.month.to_numpy(dtype=float)
        elif spec.column == "cutoff_day":
            output[spec.name] = timestamps.dt.day.to_numpy(dtype=float)
        elif spec.column == "cutoff_day_of_week":
            output[spec.name] = timestamps.dt.dayofweek.to_numpy(dtype=float)
        else:
            raise RuntimeError(f"unknown anchor feature {spec.column!r}")
    return output


def _direct_features(
    database: DatabaseView,
    anchors: pd.DataFrame,
    definition: AnchorDefinition,
    specs: list[FeatureSpec],
) -> dict[str, pd.Series]:
    if not specs:
        return {}
    root = database.tables[definition.entity_table]
    primary_key = database.primary_keys[definition.entity_table]
    if root[primary_key].isna().any():
        raise ValueError(
            f"primary key {definition.entity_table}.{primary_key} is missing"
        )
    anchor_keys, root_keys = _aligned_keys(anchors[_ENTITY], root[primary_key])
    if root_keys.dropna().duplicated().any():
        raise ValueError(
            f"primary key {definition.entity_table}.{primary_key} is not unique"
        )

    left = anchors.assign(**{_ENTITY: anchor_keys})
    right = pd.DataFrame({_NODE: root_keys})
    aliases: dict[str, str] = {}
    for index, column in enumerate(sorted({spec.column for spec in specs})):
        if column is None:
            continue
        alias = f"__nr_direct_{index}__"
        aliases[column] = alias
        right[alias] = root[column].reset_index(drop=True)
    root_time = database.time_columns.get(definition.entity_table)
    if root_time is not None:
        right[_TIME] = _get_root_time_ns(root[root_time])
    merged = left.merge(
        right,
        how="left",
        left_on=_ENTITY,
        right_on=_NODE,
        sort=False,
        validate="many_to_one",
    ).sort_values(_ROW, kind="mergesort")
    visible = np.ones(len(merged), dtype=bool)
    if definition.cutoff_column is not None and root_time is not None:
        visible = merged[_TIME].to_numpy(dtype=float) < merged[_AT].to_numpy(
            dtype=float
        )

    output: dict[str, pd.Series] = {}
    for spec in specs:
        if spec.column is None:
            raise RuntimeError("direct feature has no source column")
        source = merged[aliases[spec.column]]
        if spec.kind == "datetime":
            timestamp = _time_ns(source)
            if definition.cutoff_column is None:
                values = timestamp / _DAY_NS
            else:
                values = (merged[_AT].to_numpy(dtype=float) - timestamp) / _DAY_NS
            values[~visible] = np.nan
            output[spec.name] = pd.Series(values, dtype=float)
        elif spec.kind == "numeric":
            values = pd.to_numeric(source, errors="coerce").astype(float)
            output[spec.name] = values.where(visible).reset_index(drop=True)
        else:
            values = source.astype(object).where(visible, None)
            output[spec.name] = values.reset_index(drop=True)
    return output


def _path_features(
    database: DatabaseView,
    anchors: pd.DataFrame,
    path: RelationPath,
    specs: list[FeatureSpec],
    include_missing_in_modes: bool,
) -> dict[str, pd.Series | np.ndarray]:
    columns = sorted({spec.column for spec in specs if spec.column is not None})
    if path.is_parent:
        return _parent_path_features(database, anchors, path, columns, specs)
    events, sources = (
        _materialize_bridge_path(database, path, columns)
        if path.is_bridge
        else _materialize_path(database, path, columns)
    )
    if _AT not in anchors:
        return _static_path_features(
            anchors,
            events,
            sources,
            specs,
            include_missing_in_modes,
        )
    return _temporal_path_features(
        anchors,
        events,
        sources,
        specs,
        include_missing_in_modes,
    )


_PAIR_VALUE = "__nr_pair_value__"
_PAIR_SECOND = "__nr_pair_second__"


def _get_pair_path_features(
    database: DatabaseView,
    anchors: pd.DataFrame,
    path: RelationPath,
    paired_path: RelationPath,
    specs: list[FeatureSpec],
    include_missing_in_modes: bool,
) -> dict[str, pd.Series | np.ndarray]:
    """Return composition features of value pairs from two bridge paths.

    Both paths reach up from the same association rows, which carry the same
    entity, time and tie on each path. Each association row with both values
    gives one pair value: the two joined by a unit separator, which category
    values do not hold. A row missing either value is left out.
    The pairs then go through the same sweep as one categorical column.
    """
    column, paired_column = specs[0].column, specs[0].paired_column
    if column is None or paired_column is None:
        raise RuntimeError("pair features need both columns")
    events, sources = _materialize_bridge_path(database, path, [column])
    paired_events, paired_sources = _materialize_bridge_path(
        database, paired_path, [paired_column]
    )
    second = (
        paired_events[[_ENTITY, _TIE, paired_sources[paired_column]]]
        .rename(columns={paired_sources[paired_column]: _PAIR_SECOND})
        .drop_duplicates([_ENTITY, _TIE])
    )
    joined = events.merge(second, on=[_ENTITY, _TIE], how="inner", validate="m:1")
    first_values = joined[sources[column]]
    known = first_values.notna() & joined[_PAIR_SECOND].notna()
    joined = joined.loc[known].reset_index(drop=True)
    joined[_PAIR_VALUE] = (
        joined[sources[column]].astype(str) + "\x1f" + joined[_PAIR_SECOND].astype(str)
    )
    pair_events = joined[[_ENTITY, _TIME, _OWN_TIME, _TIE, _PAIR_VALUE]]
    pair_specs = [
        dataclasses.replace(
            spec, column=_PAIR_VALUE, paired_path=None, paired_column=None
        )
        for spec in specs
    ]
    pair_sources = {_PAIR_VALUE: _PAIR_VALUE}
    if _AT not in anchors:
        return _static_path_features(
            anchors, pair_events, pair_sources, pair_specs, include_missing_in_modes
        )
    return _temporal_path_features(
        anchors, pair_events, pair_sources, pair_specs, include_missing_in_modes
    )


def _parent_path_features(
    database: DatabaseView,
    anchors: pd.DataFrame,
    path: RelationPath,
    columns: list[str],
    specs: list[FeatureSpec],
) -> dict[str, pd.Series]:
    """Look up one ancestor row per entity with point-in-time visibility."""
    rows, sources, timed = _materialize_parent_path(database, path, columns)
    anchor_keys, row_keys = _aligned_keys(anchors[_ENTITY], rows[_ENTITY])
    left = anchors.assign(**{_ENTITY: anchor_keys})
    right = rows.assign(**{_ENTITY: row_keys})
    if right[_ENTITY].dropna().duplicated().any():
        raise RuntimeError("parent path produced more than one row per entity")
    merged = left.merge(
        right,
        how="left",
        on=_ENTITY,
        sort=False,
        validate="many_to_one",
    ).sort_values(_ROW, kind="mergesort")
    visible = np.ones(len(merged), dtype=bool)
    if timed and _AT in anchors:
        event_time = merged[_TIME].to_numpy(dtype=float)
        visible = np.isfinite(event_time) & (
            event_time < merged[_AT].to_numpy(dtype=float)
        )

    output: dict[str, pd.Series] = {}
    for spec in specs:
        if spec.operation != "lookup" or spec.column is None:
            raise RuntimeError("parent paths only support lookup features")
        source = merged[sources[spec.column]]
        if spec.kind == "datetime":
            timestamp = _time_ns(source)
            if _AT in anchors:
                values = (merged[_AT].to_numpy(dtype=float) - timestamp) / _DAY_NS
            else:
                values = timestamp / _DAY_NS
            values[~visible] = np.nan
            output[spec.name] = pd.Series(values, dtype=float)
        elif spec.kind == "numeric":
            values = pd.to_numeric(source, errors="coerce").astype(float)
            output[spec.name] = values.where(visible).reset_index(drop=True)
        else:
            values = source.astype(object).where(visible, None)
            output[spec.name] = values.reset_index(drop=True)
    return output


def _materialize_parent_path(
    database: DatabaseView,
    path: RelationPath,
    source_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, str], bool]:
    """Follow a many-to-one parent chain without changing root cardinality."""
    root_keys = _primary_key(database, path.entity_table)
    mapping = pd.DataFrame(
        {
            _ENTITY: root_keys,
            _NODE: root_keys,
            _TIME: np.full(len(root_keys), np.nan),
        }
    )
    timed = False
    current = path.entity_table
    for relation in path.edges:
        if relation.child_table != current:
            raise RuntimeError("parent relation path is not contiguous")
        frame = database.tables[current]
        current_keys = _primary_key(database, current)
        mapping_keys, current_keys = _aligned_keys(mapping[_NODE], current_keys)
        mapping = mapping.assign(**{_NODE: mapping_keys})
        payload = pd.DataFrame(
            {
                _JOIN: current_keys,
                _NEXT: frame[relation.child_column].reset_index(drop=True),
            }
        )
        time_column = database.time_columns.get(current)
        get_step_time_ns = (
            _get_root_time_ns if current == path.entity_table else _time_ns
        )
        payload[_STEP_TIME] = (
            get_step_time_ns(frame[time_column])
            if time_column is not None
            else np.full(len(frame), np.nan)
        )
        step_timed = time_column is not None
        mapping = mapping.merge(
            payload,
            how="left",
            left_on=_NODE,
            right_on=_JOIN,
            sort=False,
            validate="many_to_one",
        )
        mapping[_TIME] = _compose_time(
            mapping[_TIME].to_numpy(dtype=float),
            mapping[_STEP_TIME].to_numpy(dtype=float),
            accumulated_timed=timed,
            step_timed=step_timed,
        )
        timed = timed or step_timed
        mapping = mapping[[_ENTITY, _NEXT, _TIME]].rename(columns={_NEXT: _NODE})
        current = relation.parent_table

    terminal = database.tables[current]
    terminal_keys = _primary_key(database, current)
    mapping_keys, terminal_keys = _aligned_keys(mapping[_NODE], terminal_keys)
    mapping = mapping.assign(**{_NODE: mapping_keys})
    payload = pd.DataFrame({_JOIN: terminal_keys})
    time_column = database.time_columns.get(current)
    payload[_STEP_TIME] = (
        _time_ns(terminal[time_column])
        if time_column is not None
        else np.full(len(terminal), np.nan)
    )
    step_timed = time_column is not None
    sources: dict[str, str] = {}
    for index, column in enumerate(source_columns):
        alias = f"__nr_source_{index}__"
        sources[column] = alias
        payload[alias] = terminal[column].reset_index(drop=True)
    mapping = mapping.merge(
        payload,
        how="left",
        left_on=_NODE,
        right_on=_JOIN,
        sort=False,
        validate="many_to_one",
    )
    mapping[_TIME] = _compose_time(
        mapping[_TIME].to_numpy(dtype=float),
        mapping[_STEP_TIME].to_numpy(dtype=float),
        accumulated_timed=timed,
        step_timed=step_timed,
    )
    timed = timed or step_timed
    columns = [_ENTITY, _TIME, *sources.values()]
    result = mapping[columns].reset_index(drop=True)
    if len(result) != len(root_keys):
        raise RuntimeError("parent path changed root cardinality")
    return result, sources, timed


def _primary_key(database: DatabaseView, table: str) -> pd.Series:
    """Return one validated primary-key vector."""
    column = database.primary_keys[table]
    values = database.tables[table][column].reset_index(drop=True)
    if values.isna().any():
        raise ValueError(f"primary key {table}.{column} is missing")
    if values.duplicated().any():
        raise ValueError(f"primary key {table}.{column} is not unique")
    return values


def _materialize_bridge_path(
    database: DatabaseView,
    path: RelationPath,
    source_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Preserve association occurrences while following a many-to-one tail."""
    if not path.is_bridge:
        raise RuntimeError("bridge materialization requires a bridge path")
    association_edge, *parent_edges = path.edges
    root = database.tables[path.entity_table]
    root_key = database.primary_keys[path.entity_table]
    root_keys = _primary_key(database, path.entity_table)
    root_time = database.time_columns.get(path.entity_table)
    mapping = pd.DataFrame(
        {
            _NODE: root_keys,
            _ENTITY: root_keys,
            _TIME: (
                _get_root_time_ns(root[root_time])
                if root_time is not None
                else np.full(len(root), np.nan)
            ),
        }
    )

    association = database.tables[association_edge.child_table]
    association_keys, aligned_roots = _aligned_keys(
        association[association_edge.child_column], mapping[_NODE]
    )
    mapping = mapping.assign(**{_NODE: aligned_roots})
    association_time = database.time_columns.get(association_edge.child_table)
    first_parent = parent_edges[0]
    payload = pd.DataFrame(
        {
            _JOIN: association_keys,
            _NEXT: association[first_parent.child_column].reset_index(drop=True),
            _STEP_TIME: (
                _time_ns(association[association_time])
                if association_time is not None
                else np.full(len(association), np.nan)
            ),
        }
    )
    association_key = database.primary_keys.get(association_edge.child_table)
    if association_key is not None:
        tie_values: pd.Series | pd.DataFrame = _primary_key(
            database, association_edge.child_table
        )
    else:
        tie_columns = list(
            dict.fromkeys(
                [
                    association_edge.child_column,
                    first_parent.child_column,
                    *([association_time] if association_time is not None else []),
                ]
            )
        )
        tie_values = association.reindex(columns=tie_columns)
    payload[_TIE] = pd.util.hash_pandas_object(
        tie_values,
        index=False,
        categorize=True,
    ).to_numpy(dtype=np.uint64)
    joined = payload[payload[_JOIN].notna()].merge(
        mapping[mapping[_NODE].notna()],
        how="inner",
        left_on=_JOIN,
        right_on=_NODE,
        sort=False,
        validate="many_to_one",
    )
    joined[_OWN_TIME] = joined[_STEP_TIME].to_numpy(dtype=float)
    joined[_TIME] = _compose_time(
        joined[_TIME].to_numpy(dtype=float),
        joined[_STEP_TIME].to_numpy(dtype=float),
        accumulated_timed=root_time is not None,
        step_timed=association_time is not None,
    )
    mapping = joined[[_ENTITY, _NEXT, _TIME, _OWN_TIME, _TIE]].rename(
        columns={_NEXT: _NODE}
    )
    accumulated_timed = root_time is not None or association_time is not None
    current = association_edge.child_table
    sources: dict[str, str] = {}

    for index, relation in enumerate(parent_edges):
        if relation.child_table != current:
            raise RuntimeError("bridge parent tail is not contiguous")
        parent = database.tables[relation.parent_table]
        parent_keys = _primary_key(database, relation.parent_table)
        mapping_keys, parent_keys = _aligned_keys(mapping[_NODE], parent_keys)
        mapping = mapping.assign(**{_NODE: mapping_keys})
        parent_time = database.time_columns.get(relation.parent_table)
        parent_payload = pd.DataFrame(
            {
                _JOIN: parent_keys,
                _STEP_TIME: (
                    _time_ns(parent[parent_time])
                    if parent_time is not None
                    else np.full(len(parent), np.nan)
                ),
            }
        )
        terminal = index == len(parent_edges) - 1
        if terminal:
            for position, column in enumerate(source_columns):
                alias = f"__nr_source_{position}__"
                sources[column] = alias
                parent_payload[alias] = parent[column].reset_index(drop=True)
        else:
            next_relation = parent_edges[index + 1]
            parent_payload[_NEXT] = parent[next_relation.child_column].reset_index(
                drop=True
            )

        before = len(mapping)
        joined = mapping[mapping[_NODE].notna()].merge(
            parent_payload,
            how="inner",
            left_on=_NODE,
            right_on=_JOIN,
            sort=False,
            validate="many_to_one",
        )
        if len(joined) > before:
            raise RuntimeError("bridge parent join multiplied association rows")
        joined[_TIME] = _compose_time(
            joined[_TIME].to_numpy(dtype=float),
            joined[_STEP_TIME].to_numpy(dtype=float),
            accumulated_timed=accumulated_timed,
            step_timed=parent_time is not None,
        )
        accumulated_timed = accumulated_timed or parent_time is not None
        current = relation.parent_table
        if terminal:
            columns = [_ENTITY, _TIME, _OWN_TIME, _TIE, *sources.values()]
            return joined[columns].reset_index(drop=True), sources
        mapping = joined[[_ENTITY, _NEXT, _TIME, _OWN_TIME, _TIE]].rename(
            columns={_NEXT: _NODE}
        )
    raise RuntimeError(f"bridge from {path.entity_table}.{root_key} has no parent tail")


def _materialize_path(
    database: DatabaseView,
    path: RelationPath,
    source_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, str]]:
    if path.direction != "descendant":
        raise RuntimeError("descendant materialization requires a descendant path")
    root = database.tables[path.entity_table]
    root_key = database.primary_keys[path.entity_table]
    keys = root[root_key].reset_index(drop=True)
    if keys.isna().any():
        raise ValueError(f"primary key {path.entity_table}.{root_key} is missing")
    if keys.dropna().duplicated().any():
        raise ValueError(f"primary key {path.entity_table}.{root_key} is not unique")
    mapping = pd.DataFrame({_NODE: keys, _ENTITY: keys})
    root_time = database.time_columns.get(path.entity_table)
    mapping[_TIME] = (
        _get_root_time_ns(root[root_time])
        if root_time is not None
        else np.full(len(root), np.nan)
    )
    accumulated_timed = root_time is not None
    current = path.entity_table
    sources: dict[str, str] = {}
    for depth, relation in enumerate(path.edges):
        if relation.parent_table != current:
            raise RuntimeError("relation path is not contiguous")
        child = database.tables[relation.child_table]
        child_keys, parent_keys = _aligned_keys(
            child[relation.child_column], mapping[_NODE]
        )
        mapping = mapping.assign(**{_NODE: parent_keys})
        payload = pd.DataFrame(
            {
                _JOIN: child_keys,
            }
        )
        child_time = database.time_columns.get(relation.child_table)
        payload[_TIME] = (
            _time_ns(child[child_time])
            if child_time is not None
            else np.full(len(child), np.nan)
        )
        terminal = depth == len(path.edges) - 1
        if terminal:
            terminal_key = database.primary_keys.get(relation.child_table)
            if terminal_key is not None:
                tie_values = child[terminal_key]
            else:
                # Excluded and unsupported fields must not affect a planned
                # feature. Use only structural, temporal, and planned values.
                tie_columns = list(
                    dict.fromkeys(
                        [
                            relation.child_column,
                            *([child_time] if child_time is not None else []),
                            *source_columns,
                        ]
                    )
                )
                tie_values = child.reindex(columns=tie_columns)
            payload[_TIE] = pd.util.hash_pandas_object(
                tie_values,
                index=False,
                categorize=True,
            ).to_numpy(dtype=np.uint64)
            for index, column in enumerate(source_columns):
                alias = f"__nr_source_{index}__"
                sources[column] = alias
                payload[alias] = child[column].reset_index(drop=True)
        else:
            try:
                child_key = database.primary_keys[relation.child_table]
            except KeyError as exc:
                raise ValueError(
                    f"intermediate table {relation.child_table!r} has no primary key"
                ) from exc
            payload[_NODE] = child[child_key].reset_index(drop=True)
            if payload[_NODE].isna().any():
                raise ValueError(
                    f"primary key {relation.child_table}.{child_key} is missing"
                )
            if payload[_NODE].dropna().duplicated().any():
                raise ValueError(
                    f"primary key {relation.child_table}.{child_key} is not unique"
                )

        joined = payload[payload[_JOIN].notna()].merge(
            mapping[mapping[_NODE].notna()],
            how="inner",
            left_on=_JOIN,
            right_on=_NODE,
            suffixes=("_child", "_parent"),
            sort=False,
            validate="many_to_one",
        )
        child_values = joined[f"{_TIME}_child"].to_numpy(dtype=float)
        parent_values = joined[f"{_TIME}_parent"].to_numpy(dtype=float)
        child_timed = child_time is not None
        joined[_TIME] = _compose_time(
            parent_values,
            child_values,
            accumulated_timed=accumulated_timed,
            step_timed=child_timed,
        )
        accumulated_timed = accumulated_timed or child_timed
        if terminal:
            joined[_OWN_TIME] = child_values
            columns = [_ENTITY, _TIME, _OWN_TIME, _TIE, *sources.values()]
            return joined[columns].reset_index(drop=True), sources
        mapping = joined[[_NODE + "_child", _ENTITY, _TIME]].rename(
            columns={_NODE + "_child": _NODE}
        )
        current = relation.child_table
    raise RuntimeError("relation path must contain at least one edge")


def _temporal_path_features(
    anchors: pd.DataFrame,
    events: pd.DataFrame,
    sources: dict[str, str],
    specs: list[FeatureSpec],
    include_missing_in_modes: bool,
) -> dict[str, pd.Series | np.ndarray]:
    if _OWN_TIME not in events:
        events = events.assign(**{_OWN_TIME: np.nan})
    valid_time = np.isfinite(events[_TIME])
    if not valid_time.all():
        events = events.loc[valid_time].reset_index(drop=True)
    anchor_keys, event_keys = _aligned_keys(anchors[_ENTITY], events[_ENTITY])
    anchors = anchors.assign(**{_ENTITY: anchor_keys})
    events[_ENTITY] = event_keys
    events = events.loc[events[_ENTITY].isin(anchor_keys.unique())].reset_index(
        drop=True
    )
    if events.empty:
        return _empty_path_values(len(anchors), specs)
    mode_values = _temporal_mode_features(
        anchors,
        events,
        sources,
        [spec for spec in specs if spec.operation in _CATEGORY_OPERATIONS],
        include_missing_in_modes,
    )
    needed = {
        spec.column
        for spec in specs
        if spec.column is not None and spec.operation not in _CATEGORY_MIX_OPERATIONS
    }
    states, state_names = _prefix_states(events, sources, needed, specs)
    del events
    upper = _lookup_states(anchors, states, _AT)
    lower: dict[int, pd.DataFrame] = {}
    for window in sorted(
        {spec.window_ns for spec in specs if spec.window_ns is not None}
    ):
        shifted = anchors.assign(**{_AT: anchors[_AT] - float(window)})
        lower[int(window)] = _lookup_states(shifted, states, _AT)

    output: dict[str, pd.Series | np.ndarray] = {}
    for spec in specs:
        if spec.operation == "count":
            values = _state(upper, "rows")
            if spec.window_ns is not None:
                values = values - _state(lower[spec.window_ns], "rows")
            output[spec.name] = values.clip(min=0)
        elif spec.operation == "recency":
            event_time = pd.to_numeric(
                upper[state_names[(None, "latest_time")]], errors="coerce"
            ).to_numpy(dtype=float)
            values = (anchors[_AT].to_numpy(dtype=float) - event_time) / _DAY_NS
            output[spec.name] = values
        elif spec.operation in {"mean_age", "oldest_age"}:
            output[spec.name] = _age_stat(spec.operation, anchors, upper, state_names)
        elif spec.operation in {"first", "last"}:
            output[spec.name] = upper[
                state_names[(spec.column, spec.operation)]
            ].reset_index(drop=True)
        elif spec.operation in _CATEGORY_OPERATIONS:
            output[spec.name] = mode_values[spec.name]
        elif spec.operation == "missing_fraction":
            missing = _state(upper, state_names[(spec.column, "missing")])
            rows = _state(upper, "rows")
            output[spec.name] = np.divide(
                missing,
                rows,
                out=np.full(len(rows), np.nan),
                where=rows > 0,
            )
        else:
            output[spec.name] = _numeric_stat(
                spec, upper, lower.get(spec.window_ns), state_names
            )
    return output


def _temporal_mode_features(
    anchors: pd.DataFrame,
    events: pd.DataFrame,
    sources: dict[str, str],
    specs: list[FeatureSpec],
    include_missing: bool,
) -> dict[str, pd.Series]:
    """Sweep strict-past events for each spec's category counts.

    ``mode`` takes the most common value, resolving ties by canonical value;
    ``distinct`` the number of values seen; ``mode_share`` the most common
    value's share of the counted rows and ``last_share`` the latest counted
    row's value's share; ``entropy`` the Shannon entropy (natural log) of the
    values, kept incrementally as log(n) - sum(c log c) / n.
    """
    if not specs:
        return {}
    ordered_events = events.sort_values(
        [_ENTITY, _TIME, _OWN_TIME, _TIE], kind="mergesort"
    ).reset_index(drop=True)
    ordered_anchors = anchors.sort_values(
        [_ENTITY, _AT, _ROW], kind="mergesort"
    ).reset_index(drop=True)
    event_groups = ordered_events.groupby(_ENTITY, sort=False).indices
    anchor_groups = ordered_anchors.groupby(_ENTITY, sort=False).indices
    mode_sources: list[np.ndarray] = []
    output_arrays: list[np.ndarray] = []
    for spec in specs:
        if spec.column is None:
            raise RuntimeError("mode requires a source column")
        mode_sources.append(ordered_events[sources[spec.column]].to_numpy(dtype=object))
        if spec.operation == "mode":
            output_arrays.append(np.full(len(anchors), None, dtype=object))
        else:
            empty = 0.0 if spec.operation == "distinct" else np.nan
            output_arrays.append(np.full(len(anchors), empty, dtype=float))

    event_times = ordered_events[_TIME].to_numpy(dtype=float)
    anchor_rows = ordered_anchors[_ROW].to_numpy(dtype=np.int64)
    anchor_cutoffs = ordered_anchors[_AT].to_numpy(dtype=float)
    for entity, anchor_positions in anchor_groups.items():
        event_positions = event_groups.get(entity)
        if event_positions is None:
            continue
        counts = [{} for _ in specs]
        values = [{} for _ in specs]
        heaps: list[list[tuple[int, tuple[str, str]]]] = [[] for _ in specs]
        totals = [0] * len(specs)
        count_log_sums = [0.0] * len(specs)
        latest_identities: list[tuple[str, str] | None] = [None] * len(specs)
        position = 0
        for anchor_position in anchor_positions:
            cutoff = anchor_cutoffs[anchor_position]
            while (
                position < len(event_positions)
                and event_times[event_positions[position]] < cutoff
            ):
                event_position = event_positions[position]
                position += 1
                for index, source in enumerate(mode_sources):
                    category = _category(
                        source[event_position],
                        include_missing=include_missing,
                    )
                    if category is None:
                        continue
                    identity, value = category
                    count = counts[index].get(identity, 0) + 1
                    counts[index][identity] = count
                    values[index][identity] = value
                    heapq.heappush(heaps[index], (-count, identity))
                    totals[index] += 1
                    latest_identities[index] = identity
                    count_log_sums[index] += count * math.log(count)
                    if count > 1:
                        count_log_sums[index] -= (count - 1) * math.log(count - 1)
            row = anchor_rows[anchor_position]
            for index, heap in enumerate(heaps):
                while heap and -heap[0][0] != counts[index][heap[0][1]]:
                    heapq.heappop(heap)
                operation = specs[index].operation
                total = totals[index]
                if operation == "distinct":
                    output_arrays[index][row] = float(len(counts[index]))
                elif not heap:
                    continue
                elif operation == "mode":
                    output_arrays[index][row] = values[index][heap[0][1]]
                elif operation == "mode_share":
                    output_arrays[index][row] = -heap[0][0] / total
                elif operation == "last_share":
                    latest = latest_identities[index]
                    output_arrays[index][row] = counts[index][latest] / total
                else:
                    output_arrays[index][row] = (
                        math.log(total) - count_log_sums[index] / total
                    )
    return {
        spec.name: pd.Series(
            result, dtype=object if spec.operation == "mode" else float
        )
        for spec, result in zip(specs, output_arrays, strict=True)
    }


#: How mixed a categorical column's values are over a path's rows.
_CATEGORY_MIX_OPERATIONS = frozenset(
    {"distinct", "entropy", "last_share", "mode_share"}
)
#: Operations computed from a path's per-category counts.
_CATEGORY_OPERATIONS = _CATEGORY_MIX_OPERATIONS | {"mode"}


def get_category_mix(
    values: pd.Series, operation: str, *, include_missing: bool = False
) -> float:
    """Return how mixed ``values`` are: their distinct count, entropy or a share.

    ``distinct`` counts distinct values (0 without any); ``entropy`` is the
    Shannon entropy in nats; ``mode_share`` is the most common value's share and
    ``last_share`` the share of the last value, with ``values`` in row order.
    Missing values are left out unless ``include_missing``; entropy and the
    shares are NaN when nothing is left.
    """
    identities = [
        category[0]
        for value in values
        if (category := _category(value, include_missing=include_missing)) is not None
    ]
    if operation == "distinct":
        return float(len(set(identities)))
    if not identities:
        return float("nan")
    counted = Counter(identities)
    if operation == "last_share":
        return counted[identities[-1]] / len(identities)
    counts = np.array(list(counted.values()), dtype=float)
    shares = counts / counts.sum()
    if operation == "mode_share":
        return float(shares.max())
    return float(-(shares * np.log(shares)).sum())


def _category(
    value: object,
    *,
    include_missing: bool = False,
) -> tuple[tuple[str, str], object] | None:
    """Return a deterministic identity and normalized scalar category."""
    if not pd.api.types.is_scalar(value):
        raise ValueError("native categorical values must be scalar")
    missing = pd.isna(value)
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        if not include_missing:
            return None
        normalized = _MISSING_CATEGORY
    else:
        normalized = value.item() if isinstance(value, np.generic) else value
        if (
            include_missing
            and isinstance(normalized, str)
            and (
                normalized == _MISSING_CATEGORY
                or normalized.startswith(_CATEGORY_ESCAPE)
            )
        ):
            normalized = f"{_CATEGORY_ESCAPE}{normalized}"
    kind = type(normalized)
    identity = (f"{kind.__module__}.{kind.__qualname__}", repr(normalized))
    return identity, normalized


def _deterministic_mode(
    values: pd.Series,
    *,
    include_missing: bool = False,
) -> object:
    """Return the most frequent scalar with a stable equal-count tie break."""
    counts: dict[tuple[str, str], int] = {}
    originals: dict[tuple[str, str], object] = {}
    for value in values:
        category = _category(value, include_missing=include_missing)
        if category is None:
            continue
        identity, normalized = category
        counts[identity] = counts.get(identity, 0) + 1
        originals[identity] = normalized
    if not counts:
        return None
    winner = min(counts, key=lambda identity: (-counts[identity], identity))
    return originals[winner]


def _prefix_states(
    events: pd.DataFrame,
    sources: dict[str, str],
    needed: set[str | None],
    specs: list[FeatureSpec],
) -> tuple[pd.DataFrame, dict[tuple[str | None, str], str]]:
    events.sort_values(
        [_ENTITY, _TIME, _OWN_TIME, _TIE], kind="mergesort", inplace=True
    )
    events.reset_index(drop=True, inplace=True)
    ordered = events
    states = ordered[[_ENTITY, _TIME, _OWN_TIME, _TIE]].copy()
    groups = ordered[_ENTITY]
    names: dict[tuple[str | None, str], str] = {
        (None, "rows"): "__nr_state_rows__",
        (None, "time"): _TIME,
    }
    states[names[(None, "rows")]] = (
        ordered.groupby(_ENTITY, sort=False).cumcount() + 1
    ).astype(float)
    path_operations = {spec.operation for spec in specs}
    # Ages count from each row's own time. A path's time is the latest along it, so
    # every journey view of an event-time signup carries the signup's time, and ages
    # from it would all be zero. Rows without a time of their own keep the path's.
    own_time = ordered[_OWN_TIME].astype(float)
    age_time = own_time.where(np.isfinite(own_time), ordered[_TIME].astype(float))
    if "recency" in path_operations:
        name = "__nr_state_latest_time__"
        names[(None, "latest_time")] = name
        states[name] = age_time.groupby(groups, sort=False).cummax()
    if "mean_age" in path_operations:
        name = "__nr_state_time_days_sum__"
        names[(None, "time_days_sum")] = name
        states[name] = (age_time / _DAY_NS).groupby(groups, sort=False).cumsum()
    if "oldest_age" in path_operations:
        name = "__nr_state_oldest_time__"
        names[(None, "oldest_time")] = name
        states[name] = age_time.groupby(groups, sort=False).cummin()
    requested: dict[str, set[str]] = defaultdict(set)
    for spec in specs:
        if spec.column is not None:
            requested[spec.column].add(spec.operation)
    for position, column in enumerate(sorted(value for value in needed if value)):
        source = ordered[sources[column]]
        base = f"__nr_state_{position}"
        last_name = f"{base}_last__"
        operations = requested[column]
        if "missing_fraction" in operations:
            name = f"{base}_missing__"
            names[(column, "missing")] = name
            states[name] = (
                source.isna().astype(float).groupby(groups, sort=False).cumsum()
            )
        categorical = any(
            spec.column == column and spec.kind == "categorical" for spec in specs
        )
        if "first" in operations:
            values = (
                source.astype(object)
                if categorical
                else pd.to_numeric(source, errors="coerce").astype(float)
            )
            present = values.notna()
            is_first = present & (
                present.astype(np.int64).groupby(groups, sort=False).cumsum() == 1
            )
            first_name = f"{base}_first__"
            names[(column, "first")] = first_name
            states[first_name] = (
                values.where(is_first).groupby(groups, sort=False).ffill()
            )
        if "last" in operations:
            names[(column, "last")] = last_name
            states[last_name] = (
                source.astype(object).groupby(groups, sort=False).ffill()
                if categorical
                else pd.to_numeric(source, errors="coerce")
                .astype(float)
                .groupby(groups, sort=False)
                .ffill()
            )
        if categorical:
            continue

        numeric = pd.to_numeric(source, errors="coerce").astype(float)
        valid = numeric.notna().astype(float)
        numeric_zero = numeric.fillna(0.0)
        additive: list[tuple[str, pd.Series]] = []
        if operations & {"mean", "std", "sum"}:
            additive.extend([("n", valid), ("sum", numeric_zero)])
        if "std" in operations:
            additive.append(("sumsq", numeric_zero * numeric_zero))
        for statistic, values in additive:
            name = f"{base}_{statistic}__"
            names[(column, statistic)] = name
            states[name] = values.groupby(groups, sort=False).cumsum()
        extrema: list[tuple[str, pd.Series]] = []
        if "min" in operations:
            extrema.append(("min", numeric.groupby(groups, sort=False).cummin()))
        if "max" in operations:
            extrema.append(("max", numeric.groupby(groups, sort=False).cummax()))
        for statistic, values in extrema:
            name = f"{base}_{statistic}__"
            names[(column, statistic)] = name
            states[name] = values.groupby(groups, sort=False).ffill()
    del ordered, groups
    states.sort_values(
        [_TIME, _ENTITY, _OWN_TIME, _TIE], kind="mergesort", inplace=True
    )
    states.reset_index(drop=True, inplace=True)
    return states, names


def _age_stat(
    operation: str,
    anchors: pd.DataFrame,
    upper: pd.DataFrame,
    names: dict[tuple[str | None, str], str],
) -> np.ndarray:
    """Compute event ages from strict-cutoff temporal prefix states."""
    cutoff_days = anchors[_AT].to_numpy(dtype=float) / _DAY_NS
    if operation == "mean_age":
        count = _state(upper, "rows")
        total_days = _state(upper, names[(None, "time_days_sum")])
        mean_days = np.divide(
            total_days,
            count,
            out=np.full(len(total_days), np.nan),
            where=count > 0,
        )
        return cutoff_days - mean_days
    if operation == "oldest_age":
        oldest = pd.to_numeric(
            upper[names[(None, "oldest_time")]], errors="coerce"
        ).to_numpy(dtype=float)
        return (anchors[_AT].to_numpy(dtype=float) - oldest) / _DAY_NS
    raise RuntimeError(f"unsupported age operation {operation!r}")


def _lookup_states(
    anchors: pd.DataFrame, states: pd.DataFrame, at_column: str
) -> pd.DataFrame:
    left = anchors[[_ROW, _ENTITY, at_column]].sort_values(
        [at_column, _ENTITY, _ROW], kind="mergesort"
    )
    merged = pd.merge_asof(
        left,
        states,
        left_on=at_column,
        right_on=_TIME,
        by=_ENTITY,
        allow_exact_matches=False,
        direction="backward",
    )
    return merged.sort_values(_ROW, kind="mergesort").reset_index(drop=True)


def _numeric_stat(
    spec: FeatureSpec,
    upper: pd.DataFrame,
    lower: pd.DataFrame | None,
    names: dict[tuple[str | None, str], str],
) -> np.ndarray:
    if spec.column is None:
        raise RuntimeError(f"{spec.operation} requires a source column")
    if spec.operation in {"min", "max"}:
        return pd.to_numeric(
            upper[names[(spec.column, spec.operation)]], errors="coerce"
        ).to_numpy(dtype=float)
    if spec.operation == "last":
        return pd.to_numeric(
            upper[names[(spec.column, "last")]], errors="coerce"
        ).to_numpy(dtype=float)

    count = _state(upper, names[(spec.column, "n")])
    total = _state(upper, names[(spec.column, "sum")])
    squares = (
        _state(upper, names[(spec.column, "sumsq")])
        if spec.operation == "std"
        else None
    )
    if lower is not None:
        count -= _state(lower, names[(spec.column, "n")])
        total -= _state(lower, names[(spec.column, "sum")])
        if squares is not None:
            squares -= _state(lower, names[(spec.column, "sumsq")])
    if spec.operation == "sum":
        total[count <= 0] = np.nan
        return total
    if spec.operation == "mean":
        return np.divide(
            total,
            count,
            out=np.full(len(total), np.nan),
            where=count > 0,
        )
    if spec.operation == "std":
        if squares is None:
            raise RuntimeError("standard deviation state is missing")
        numerator = squares - np.divide(
            total * total,
            count,
            out=np.zeros(len(total)),
            where=count > 0,
        )
        variance = np.divide(
            np.maximum(numerator, 0.0),
            count - 1,
            out=np.full(len(total), np.nan),
            where=count > 1,
        )
        return np.sqrt(variance)
    raise RuntimeError(f"unsupported native operation {spec.operation!r}")


def _state(frame: pd.DataFrame, name: str) -> np.ndarray:
    if name == "rows":
        name = "__nr_state_rows__"
    elif name == "time":
        name = _TIME
    if name not in frame:
        raise RuntimeError(f"native prefix state is missing: {name}")
    return (
        pd.to_numeric(frame[name], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
        .copy()
    )


def _static_path_features(
    anchors: pd.DataFrame,
    events: pd.DataFrame,
    sources: dict[str, str],
    specs: list[FeatureSpec],
    include_missing_in_modes: bool,
) -> dict[str, pd.Series | np.ndarray]:
    if _OWN_TIME not in events:
        events = events.assign(**{_OWN_TIME: np.nan})
    anchor_keys, event_keys = _aligned_keys(anchors[_ENTITY], events[_ENTITY])
    anchors = anchors.assign(**{_ENTITY: anchor_keys})
    events = events.assign(**{_ENTITY: event_keys})
    events = events.loc[events[_ENTITY].isin(anchor_keys.unique())].reset_index(
        drop=True
    )
    if events.empty:
        return _empty_path_values(len(anchors), specs)
    ordered = events.sort_values([_ENTITY, _OWN_TIME, _TIE], kind="mergesort")
    result = pd.DataFrame({_ENTITY: anchors[_ENTITY]})
    groups = ordered.groupby(_ENTITY, sort=False)
    output: dict[str, pd.Series | np.ndarray] = {}
    for spec in specs:
        if spec.window_ns is not None or spec.operation in {
            "mean_age",
            "oldest_age",
            "recency",
        }:
            raise RuntimeError("static paths cannot execute temporal operations")
        if spec.operation == "count":
            aggregate = groups.size().rename(spec.name)
        else:
            if spec.column is None:
                raise RuntimeError(f"{spec.operation} requires a source column")
            source = sources[spec.column]
            if spec.operation == "last":
                aggregate = groups[source].last().rename(spec.name)
            elif spec.operation == "first":
                aggregate = groups[source].first().rename(spec.name)
            elif spec.operation == "mode":
                aggregate = (
                    groups[source]
                    .agg(
                        lambda values: _deterministic_mode(
                            values,
                            include_missing=include_missing_in_modes,
                        )
                    )
                    .rename(spec.name)
                )
            elif spec.operation == "missing_fraction":
                aggregate = (
                    groups[source]
                    .agg(lambda values: values.isna().mean())
                    .rename(spec.name)
                )
            elif spec.operation in _CATEGORY_MIX_OPERATIONS:
                aggregate = (
                    groups[source]
                    .agg(
                        lambda values, operation=spec.operation: get_category_mix(
                            values, operation, include_missing=include_missing_in_modes
                        )
                    )
                    .rename(spec.name)
                )
            else:
                numeric = ordered.assign(
                    __nr_numeric__=pd.to_numeric(ordered[source], errors="coerce")
                ).groupby(_ENTITY, sort=False)["__nr_numeric__"]
                aggregate = (
                    numeric.sum(min_count=1)
                    if spec.operation == "sum"
                    else getattr(numeric, spec.operation)()
                ).rename(spec.name)
        joined = result.merge(
            aggregate,
            how="left",
            left_on=_ENTITY,
            right_index=True,
            sort=False,
            validate="many_to_one",
        )
        values = joined[spec.name]
        if spec.operation == "count":
            values = values.fillna(0.0).astype(float)
        output[spec.name] = values.reset_index(drop=True)
    return output


def _empty_path_values(
    rows: int, specs: list[FeatureSpec]
) -> dict[str, pd.Series | np.ndarray]:
    output: dict[str, pd.Series | np.ndarray] = {}
    for spec in specs:
        if spec.operation in {"count", "distinct"}:
            output[spec.name] = np.zeros(rows, dtype=float)
        elif spec.kind == "categorical":
            output[spec.name] = pd.Series([None] * rows, dtype=object)
        else:
            output[spec.name] = np.full(rows, np.nan)
    return output


_SIBLING_PARENT = "__nr_sibling_parent__"
_SIBLING_MATCH = "__nr_sibling_match__"
_SIBLING_ROWS = "__nr_sibling_rows__"


def _get_shared_codes(
    left: pd.Series, right: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    """Return integer codes for two key series in one code space; -1 is missing."""
    left, right = _aligned_keys(left, right)
    codes, _ = pd.factorize(pd.concat([left, right], ignore_index=True))
    return codes[: len(left)].astype(np.int64), codes[len(left) :].astype(np.int64)


def _get_parent_attribute(
    keys: pd.Series, parent_keys: pd.Series, attribute: pd.Series
) -> pd.Series:
    """Return the parent's ``attribute`` for each of ``keys``; missing when absent.

    ``keys`` are a child table's references into the parent, ``parent_keys`` the
    parent's primary key and ``attribute`` its column, row-aligned with it.
    """
    keys, parent_keys = _aligned_keys(keys, parent_keys)
    lookup = pd.Series(attribute.to_numpy(), index=parent_keys.to_numpy())
    lookup = lookup[~lookup.index.duplicated()]
    return pd.Series(lookup.reindex(keys.to_numpy()).to_numpy())


def _get_rows_before(
    anchors: pd.DataFrame,
    rows: pd.DataFrame,
    group_columns: list[str],
    cutoff_ns: np.ndarray,
) -> pd.DataFrame:
    """Return each anchor's latest group row strictly before its ``cutoff_ns``.

    ``rows`` is sorted by ``_TIME``; ``anchors`` has ``_ROW`` and the
    ``group_columns``. The result is in anchor row order; anchors with no
    earlier row in their group get NaN.
    """
    left = (
        anchors[[_ROW, *group_columns]]
        .assign(**{_AT: cutoff_ns})
        .sort_values(_AT, kind="mergesort")
    )
    matched = pd.merge_asof(
        left,
        rows,
        left_on=_AT,
        right_on=_TIME,
        by=group_columns,
        direction="backward",
        allow_exact_matches=False,
    )
    return matched.sort_values(_ROW, kind="mergesort").reset_index(drop=True)


#: Numeric aggregates a sibling path computes from running totals.
_SIBLING_NUMERIC_OPERATIONS = frozenset({"sum", "mean", "max", "min"})


def _get_sibling_path_features(
    database: DatabaseView,
    anchors: pd.DataFrame,
    path: RelationPath,
    specs: list[FeatureSpec],
) -> dict[str, np.ndarray]:
    """Compute sibling features: the parent's time-stamped rows before each cutoff.

    One sort of the sibling rows by (parent, time) gives running counts, running
    sums, counts, maxima and minima of numeric columns, first occurrences of each
    foreign key and timestamp, and per-value counts for the match features; one
    as-of join per window or match column then reads each anchor's totals
    strictly before its cutoff. A window's sum is the running sum before the
    cutoff minus the running sum before the window's start. Rows at the cutoff
    itself are not earlier, so an event-time row never counts itself or rows
    logged at the same instant.
    """
    up, group, down = get_sibling_edges(path)
    entity = database.tables[path.entity_table]
    entity_key = entity[database.primary_keys[path.entity_table]].reset_index(drop=True)
    siblings = database.tables[down.child_table].reset_index(drop=True)
    anchor_keys, entity_keys = _aligned_keys(anchors[_ENTITY], entity_key)
    position = pd.Series(np.arange(len(entity_keys)), index=entity_keys.to_numpy())
    entity_rows = position.reindex(anchor_keys.to_numpy()).to_numpy()
    if np.isnan(entity_rows.astype(float)).any():
        raise ValueError("anchor entity keys are missing from the entity table")
    entity_rows = entity_rows.astype(np.int64)

    anchor_parent_keys = (
        entity[up.child_column].reset_index(drop=True).iloc[entity_rows]
    )
    sibling_parent_keys = siblings[down.child_column]
    if group is not None:
        # Group the parent's rows by its attribute: a drug's class, not the drug.
        parent = database.tables[up.parent_table]
        anchor_parent_keys = _get_parent_attribute(
            anchor_parent_keys, parent[up.parent_column], parent[group.child_column]
        )
        sibling_parent_keys = _get_parent_attribute(
            sibling_parent_keys, parent[up.parent_column], parent[group.child_column]
        )
    anchor_parent, sibling_parent = _get_shared_codes(
        anchor_parent_keys, sibling_parent_keys
    )
    at = anchors[_AT].to_numpy(dtype=float)
    keyed = anchors[[_ROW]].assign(**{_SIBLING_PARENT: anchor_parent})
    rows = pd.DataFrame(
        {
            _SIBLING_PARENT: sibling_parent,
            _TIME: _time_ns(siblings[database.time_columns[down.child_table]]),
        }
    )
    distinct_columns = sorted(
        {spec.column for spec in specs if spec.operation == "distinct"}
    )
    time_column = database.time_columns[down.child_table]
    for column in distinct_columns:
        if column != time_column:
            rows[column] = pd.factorize(siblings[column])[0]
    numeric_columns = sorted(
        {spec.column for spec in specs if spec.operation in _SIBLING_NUMERIC_OPERATIONS}
    )
    for column in numeric_columns:
        rows[f"value|{column}"] = pd.to_numeric(siblings[column], errors="coerce")
    usable = (rows[_SIBLING_PARENT] >= 0) & np.isfinite(rows[_TIME])
    rows = rows.loc[usable].sort_values(
        [_SIBLING_PARENT, _TIME], kind="mergesort", ignore_index=True
    )
    by_parent = rows.groupby(_SIBLING_PARENT, sort=False)
    totals = rows[[_SIBLING_PARENT, _TIME]].copy()
    totals[_SIBLING_ROWS] = by_parent.cumcount().to_numpy() + 1
    for column in distinct_columns:
        key = [_SIBLING_PARENT, _TIME if column == time_column else column]
        first = ~rows.duplicated(key)
        if column != time_column:
            first &= rows[column] >= 0
        totals[f"distinct|{column}"] = (
            first.astype(np.int64).groupby(rows[_SIBLING_PARENT], sort=False).cumsum()
        )
    for column in numeric_columns:
        values = rows[f"value|{column}"]
        groups = rows[_SIBLING_PARENT]
        totals[f"sum|{column}"] = (
            values.fillna(0.0).groupby(groups, sort=False).cumsum()
        )
        totals[f"rows|{column}"] = (
            values.notna().astype(np.int64).groupby(groups, sort=False).cumsum()
        )
        totals[f"max|{column}"] = (
            values.fillna(-np.inf)
            .groupby(groups, sort=False)
            .cummax()
            .replace(-np.inf, np.nan)
        )
        totals[f"min|{column}"] = (
            values.fillna(np.inf)
            .groupby(groups, sort=False)
            .cummin()
            .replace(np.inf, np.nan)
        )
    first_time = by_parent[_TIME].min()
    totals = totals.sort_values(_TIME, kind="mergesort", ignore_index=True)
    latest = _get_rows_before(keyed, totals, [_SIBLING_PARENT], at)
    earlier_rows = latest[_SIBLING_ROWS].fillna(0).to_numpy(dtype=float)
    latest_time = latest[_TIME].to_numpy(dtype=float)
    oldest_time = pd.Series(anchor_parent).map(first_time).to_numpy(dtype=float)
    oldest_time[earlier_rows == 0] = np.nan

    output: dict[str, np.ndarray] = {}
    before_windows: dict[int, pd.DataFrame] = {}

    def get_totals_before_window(window_ns: int) -> pd.DataFrame:
        """Return each anchor's running totals strictly before its window starts."""
        if window_ns not in before_windows:
            before_windows[window_ns] = _get_rows_before(
                keyed, totals, [_SIBLING_PARENT], at - float(window_ns)
            )
        return before_windows[window_ns]

    def get_total(frame: pd.DataFrame, column: str) -> np.ndarray:
        """Return one running total per anchor, 0 where no earlier row exists."""
        return frame[column].fillna(0).to_numpy(dtype=float)

    for spec in specs:
        if spec.operation in _SIBLING_NUMERIC_OPERATIONS:
            column = spec.column
            if spec.operation in ("max", "min"):
                output[spec.name] = latest[f"{spec.operation}|{column}"].to_numpy(
                    dtype=float
                )
                continue
            value_sum = get_total(latest, f"sum|{column}")
            value_rows = get_total(latest, f"rows|{column}")
            if spec.window_ns is not None:
                before = get_totals_before_window(spec.window_ns)
                value_sum = value_sum - get_total(before, f"sum|{column}")
                value_rows = value_rows - get_total(before, f"rows|{column}")
            if spec.operation == "sum":
                output[spec.name] = value_sum
            else:
                with np.errstate(divide="ignore", invalid="ignore"):
                    output[spec.name] = np.where(
                        value_rows > 0, value_sum / value_rows, np.nan
                    )
        elif spec.operation == "count" and spec.window_ns is not None:
            before = get_totals_before_window(spec.window_ns)
            output[spec.name] = earlier_rows - get_total(before, _SIBLING_ROWS)
        elif spec.operation == "count":
            output[spec.name] = earlier_rows
        elif spec.operation == "recency":
            output[spec.name] = (at - latest_time) / _DAY_NS
        elif spec.operation == "oldest_age":
            output[spec.name] = (at - oldest_time) / _DAY_NS
        elif spec.operation == "distinct":
            output[spec.name] = (
                latest[f"distinct|{spec.column}"].fillna(0).to_numpy(dtype=float)
            )

    match_specs = [spec for spec in specs if spec.operation.startswith("match_")]
    pairs = dict(
        (sibling_column, entity_column)
        for entity_column, sibling_column in get_sibling_match_columns(database, path)
    )
    for column in sorted({spec.column for spec in match_specs}):
        anchor_values, sibling_values = _get_shared_codes(
            entity[pairs[column]].reset_index(drop=True).iloc[entity_rows],
            siblings[column],
        )
        matched = pd.DataFrame(
            {
                _SIBLING_PARENT: sibling_parent,
                _SIBLING_MATCH: sibling_values,
                _TIME: _time_ns(siblings[time_column]),
            }
        )
        matched = matched.loc[
            (matched[_SIBLING_PARENT] >= 0)
            & (matched[_SIBLING_MATCH] >= 0)
            & np.isfinite(matched[_TIME])
        ].sort_values([_SIBLING_PARENT, _SIBLING_MATCH, _TIME], kind="mergesort")
        matched[_SIBLING_ROWS] = (
            matched.groupby([_SIBLING_PARENT, _SIBLING_MATCH], sort=False).cumcount()
            + 1
        )
        matched = matched.sort_values(_TIME, kind="mergesort", ignore_index=True)
        latest_match = _get_rows_before(
            keyed.assign(**{_SIBLING_MATCH: anchor_values}),
            matched,
            [_SIBLING_PARENT, _SIBLING_MATCH],
            at,
        )
        match_rows = latest_match[_SIBLING_ROWS].fillna(0).to_numpy(dtype=float)
        match_time = latest_match[_TIME].to_numpy(dtype=float)
        for spec in match_specs:
            if spec.column != column:
                continue
            if spec.operation == "match_count":
                output[spec.name] = match_rows
            elif spec.operation == "match_recency":
                output[spec.name] = (at - match_time) / _DAY_NS
            else:
                output[spec.name] = (
                    (match_rows > 0) & (match_time == latest_time)
                ).astype(float)

    missing = [spec.name for spec in specs if spec.name not in output]
    if missing:
        raise RuntimeError(f"sibling executor omitted planned features: {missing[:3]}")
    return output


def _aligned_keys(left: pd.Series, right: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Preserve join-key dtypes unless the two sides are incompatible."""
    left = left.reset_index(drop=True)
    right = right.reset_index(drop=True)
    if left.dtype == right.dtype and not isinstance(left.dtype, pd.CategoricalDtype):
        return left, right
    if pd.api.types.is_integer_dtype(left) and pd.api.types.is_integer_dtype(right):
        if left.isna().any() or right.isna().any():
            return left.astype("Int64"), right.astype("Int64")
        return left.astype(np.int64), right.astype(np.int64)
    if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
        return left.astype(float), right.astype(float)
    if pd.api.types.is_string_dtype(left) and pd.api.types.is_string_dtype(right):
        return left.astype(object), right.astype(object)
    return left.astype("string"), right.astype("string")


# Databases store microseconds, and float nanoseconds near 2026 resolve about 256 ns,
# so one microsecond is the smallest step that survives both.
_ROOT_TIME_LEAD_NS = 1_000.0


def _get_root_time_ns(series: pd.Series) -> np.ndarray:
    """Return the entity's own times in float ns, one microsecond early.

    The entity exists at its own time (entity_created_at <= cutoff), so its row
    and the rows reached through it are visible at that time; a related row at
    the same instant stays invisible, because path times take the later of the
    two. An event-time row, anchored at its own timestamp, therefore reads its
    own columns and the rows strictly before it.
    """
    return _time_ns(series) - _ROOT_TIME_LEAD_NS


def _time_ns(series: pd.Series) -> np.ndarray:
    timestamps = pd.to_datetime(
        series.reset_index(drop=True), errors="coerce", utc=True
    ).astype("datetime64[ns, UTC]")
    values = timestamps.astype("int64").to_numpy(dtype=float)
    values[timestamps.isna().to_numpy()] = np.nan
    return values


def _compose_time(
    accumulated: np.ndarray,
    step: np.ndarray,
    *,
    accumulated_timed: bool,
    step_timed: bool,
) -> np.ndarray:
    """Compose declared timestamps while keeping invalid values fail-closed."""
    if accumulated_timed and step_timed:
        # Unlike fmax, maximum propagates NaN. A missing value in any declared
        # time column therefore makes the path row invisible.
        return np.maximum(accumulated, step)
    if accumulated_timed:
        return accumulated.copy()
    if step_timed:
        return step.copy()
    return np.full(len(accumulated), np.nan)


__all__ = ["NativeRelationalFeaturizer"]
