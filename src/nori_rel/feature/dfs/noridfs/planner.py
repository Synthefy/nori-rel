"""Deterministic, bounded planning for native relational features."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable

import numpy as np
import pandas as pd

from .contracts import (
    AnchorDefinition,
    ColumnKind,
    DatabaseView,
    FeatureOperation,
    FeaturePlan,
    FeatureSpec,
    ForeignKey,
    NativeFeatureConfig,
    PathDirection,
    RelationPath,
    SiblingPathConfig,
    get_sibling_edges,
)


def build_feature_plan(
    database: DatabaseView,
    anchors: AnchorDefinition,
    config: NativeFeatureConfig,
    *,
    excluded_columns: dict[str, set[str]] | None = None,
    sibling_paths: SiblingPathConfig | None = None,
    include_composition_features: bool = False,
    composition_pairs: tuple[tuple[str, str], ...] = (),
) -> FeaturePlan:
    """Create a task-name-free plan from fitted schema information.

    ``include_composition_features`` describes the rows that descendant and
    bridge paths reach, in their own time order: for each categorical column,
    how mixed its values are (entropy, the most common and the last value's
    shares, distinct values), and for every column its first value. Over a
    signup's journey views and their drugs and pharmacies, that is the journey's
    drug-class mix and its first-touch pharmacy. The mode and the last value are
    planned either way.

    ``composition_pairs`` names pairs of categorical columns as
    ``("table.column", "table.column")``, each at the end of a bridge path from
    the same association rows. Each pair gets the same composition features over
    its value pairs: over a journey, the most common drug class and pharmacy
    company together.

    ``sibling_paths`` adds the opt-in sibling family (see ``SiblingPathConfig``)
    when the anchors have a cutoff. Its features take the budget first, after the
    anchor and direct features, because a caller asked for them.
    """
    database.validate()
    if anchors.entity_table not in database.tables:
        raise ValueError(f"missing entity table {anchors.entity_table!r}")
    if anchors.entity_table not in database.primary_keys:
        raise ValueError("entity table requires a single-column primary key")
    excluded = excluded_columns or {}
    root = database.tables[anchors.entity_table]
    root_key = database.primary_keys[anchors.entity_table]
    root_foreign_keys = {
        relation.child_column
        for relation in database.foreign_keys
        if relation.child_table == anchors.entity_table
    }
    anchor_features = [
        FeatureSpec.anchor(anchors.entity_table, "entity_key", "categorical")
    ]
    if anchors.cutoff_column is not None:
        anchor_features.extend(
            FeatureSpec.anchor(anchors.entity_table, component, "numeric")
            for component in (
                "cutoff_epoch_days",
                "cutoff_year",
                "cutoff_month",
                "cutoff_day",
                "cutoff_day_of_week",
            )
        )
    direct: list[FeatureSpec] = []
    direct_cardinality: dict[str, int] = {}
    for column in sorted(root.columns):
        if (
            column == root_key
            or column in root_foreign_keys
            or column in excluded.get(anchors.entity_table, set())
        ):
            continue
        kind = _column_kind(root[column])
        if kind is not None:
            if (
                kind == "categorical"
                and config.max_direct_categorical_cardinality is not None
            ):
                cardinality = _bounded_cardinality(
                    root[column], config.max_direct_categorical_cardinality
                )
                if cardinality is None:
                    continue
                direct_cardinality[column] = cardinality
            direct.append(FeatureSpec.direct(anchors.entity_table, column, kind))
    direct.sort(
        key=lambda feature: (
            feature.kind == "categorical",
            direct_cardinality.get(feature.column or "", 0),
            feature.name,
        )
    )

    paths = _enumerate_paths(database, anchors, config)
    siblings = (
        get_sibling_paths(database, anchors, sibling_paths)
        if sibling_paths is not None and anchors.cutoff_column is not None
        else ()
    )
    direct_share = (
        config.max_features
        if not paths and not siblings
        else max(1, config.max_features // 3)
    )
    selected = anchor_features[: config.max_features]
    direct_limit = min(
        config.max_direct_features,
        direct_share,
        config.max_features - len(selected),
    )
    selected.extend(direct[:direct_limit])
    tiers = _candidate_tiers(
        database,
        anchors,
        config,
        paths,
        excluded,
        include_composition_features,
        composition_pairs,
    )
    if siblings:
        sibling_tiers = get_sibling_feature_tiers(database, siblings, sibling_paths)
        if sibling_paths.max_features is not None:
            # Cap the family as a whole, in its tier order, so the paths below it
            # (a signup's own journey views) still get part of the budget.
            capped = [feature for tier in sibling_tiers for feature in tier]
            sibling_tiers = [capped[: sibling_paths.max_features]]
        tiers = sibling_tiers + tiers
    for tier in tiers:
        for feature in tier:
            if len(selected) >= config.max_features:
                break
            selected.append(feature)
        if len(selected) >= config.max_features:
            break
    if not selected:
        raise ValueError("native relational planning produced no features")
    names = [feature.name for feature in selected]
    if len(names) != len(set(names)):
        raise RuntimeError("native feature names are not unique")
    return FeaturePlan(
        schema_digest=database.schema_digest,
        anchors=anchors,
        config=config,
        paths=paths + siblings,
        features=tuple(selected),
    )


def get_sibling_paths(
    database: DatabaseView, anchors: AnchorDefinition, config: SiblingPathConfig
) -> tuple[RelationPath, ...]:
    """Return the entity's sibling paths, at most ``config.max_paths``.

    A sibling path goes up one foreign key of the entity to its parent and down
    one foreign key into that parent from a time-stamped table: a view's device,
    then the device's views. A table in ``config.parent_tables`` that the entity
    does not reference, but one of its parents does, groups that parent's rows
    by the reference: a signup's drug, grouped by ``drug.drug_class_id``, then
    the paid subscriptions of every drug in the class. Siblings in the entity's
    own table come first, then other tables, each in identity order.
    """
    entity = anchors.entity_table
    entity_edges = [up for up in database.foreign_keys if up.child_table == entity]
    direct_parents = {up.parent_table for up in entity_edges}

    def get_down_edges(up: ForeignKey) -> list[ForeignKey]:
        """Return the time-stamped tables' foreign keys into ``up``'s parent."""
        return [
            down
            for down in database.foreign_keys
            if (down.parent_table, down.parent_column)
            == (up.parent_table, up.parent_column)
            and down.child_table in database.time_columns
        ]

    excluded = set(config.excluded_paths)
    paths = [
        RelationPath(entity, (up, down), "sibling")
        for up in entity_edges
        if config.parent_tables is None or up.parent_table in config.parent_tables
        for down in get_down_edges(up)
        if (up.parent_table, down.child_table) not in excluded
    ]
    if config.parent_tables is not None:
        paths.extend(
            RelationPath(entity, (up, group, down), "sibling")
            for up in entity_edges
            for group in database.foreign_keys
            if group.child_table == up.parent_table
            and group.parent_table in config.parent_tables
            and group.parent_table not in direct_parents
            for down in get_down_edges(up)
            if (group.parent_table, down.child_table) not in excluded
        )
    paths.sort(key=lambda path: (path.edges[-1].child_table != entity, path.identity))
    return tuple(paths[: config.max_paths])


def get_sibling_match_columns(
    database: DatabaseView, path: RelationPath
) -> list[tuple[str, str]]:
    """Return (entity column, sibling column) pairs that name the same parent.

    For a view's device's views, (drug_id, drug_id) pairs the view's drug with
    each earlier view's drug. The path's own foreign keys are left out, and so is
    a parent the entity references through more than one column.
    """
    up, _, down = get_sibling_edges(path)
    entity_columns: dict[tuple[str, str], list[str]] = defaultdict(list)
    for relation in database.foreign_keys:
        if relation.child_table == path.entity_table and relation != up:
            entity_columns[(relation.parent_table, relation.parent_column)].append(
                relation.child_column
            )
    pairs = []
    for relation in database.foreign_keys:
        if (
            relation.child_table == down.child_table
            and relation.child_column != down.child_column
        ):
            matches = entity_columns.get(
                (relation.parent_table, relation.parent_column), []
            )
            if len(matches) == 1:
                pairs.append((matches[0], relation.child_column))
    return sorted(pairs)


def get_sibling_distinct_columns(
    database: DatabaseView, path: RelationPath
) -> list[str]:
    """Return the sibling table's foreign keys, other than the path's, then its time."""
    _, _, down = get_sibling_edges(path)
    columns = sorted(
        relation.child_column
        for relation in database.foreign_keys
        if relation.child_table == down.child_table
        and relation.child_column != down.child_column
    )
    return [*columns, database.time_columns[down.child_table]]


def get_sibling_numeric_columns(
    database: DatabaseView, path: RelationPath
) -> list[str]:
    """Return the sibling table's numeric columns other than its keys and time.

    A drug name's earlier weekly claims give their claim counts and amounts; their
    ``drug_id`` and ``week_start`` are not summed.
    """
    _, _, down = get_sibling_edges(path)
    table = database.tables[down.child_table]
    excluded = {
        database.primary_keys.get(down.child_table),
        database.time_columns[down.child_table],
        *(
            relation.child_column
            for relation in database.foreign_keys
            if relation.child_table == down.child_table
        ),
    }
    return sorted(
        str(column)
        for column in table.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(table[column])
        and not pd.api.types.is_bool_dtype(table[column])
    )


def get_sibling_feature_tiers(
    database: DatabaseView,
    paths: tuple[RelationPath, ...],
    config: SiblingPathConfig,
) -> list[list[FeatureSpec]]:
    """Return the sibling features in budget order, each tier interleaving paths.

    Tiers: counts (all, then each window); recency and oldest age; with
    ``include_numeric_aggregates``, sums and means of numeric columns (all, then
    each window) and their all-time min and max; match features; distinct
    counts. Numeric aggregates come before matches because a caller who asks for
    them names the columns they want.
    """
    counts = [FeatureSpec.aggregate(path, "count") for path in paths]
    for window in config.window_ns:
        counts.extend(
            FeatureSpec.aggregate(path, "count", window_ns=window) for path in paths
        )
    ages = [
        FeatureSpec.aggregate(path, operation)
        for operation in ("recency", "oldest_age")
        for path in paths
    ]
    matches = []
    if config.include_match_features:
        for path in paths:
            for _, column in get_sibling_match_columns(database, path):
                matches.extend(
                    FeatureSpec.aggregate(path, operation, column=column)
                    for operation in ("match_count", "match_recency", "match_previous")
                )
    numeric = []
    if config.include_numeric_aggregates:
        columns = {
            path: [
                column
                for column in get_sibling_numeric_columns(database, path)
                if config.numeric_columns is None
                or f"{path.edges[-1].child_table}.{column}" in config.numeric_columns
            ]
            for path in paths
        }
        for window in (None, *config.window_ns):
            for operation in ("sum", "mean"):
                numeric.extend(
                    FeatureSpec.aggregate(
                        path, operation, column=column, window_ns=window
                    )
                    for path in paths
                    for column in columns[path]
                )
        for operation in ("max", "min"):
            numeric.extend(
                FeatureSpec.aggregate(path, operation, column=column)
                for path in paths
                for column in columns[path]
            )
    distinct = [
        FeatureSpec.aggregate(path, "distinct", column=column)
        for path in paths
        for column in get_sibling_distinct_columns(database, path)
    ]
    return [counts, ages, numeric, matches, distinct]


def _enumerate_paths(
    database: DatabaseView,
    anchors: AnchorDefinition,
    config: NativeFeatureConfig,
) -> tuple[RelationPath, ...]:
    incoming: dict[str, list[ForeignKey]] = defaultdict(list)
    outgoing: dict[str, list[ForeignKey]] = defaultdict(list)
    for relation in database.foreign_keys:
        incoming[relation.parent_table].append(relation)
        outgoing[relation.child_table].append(relation)
    for relations in (*incoming.values(), *outgoing.values()):
        relations.sort()

    candidates: dict[PathDirection, list[RelationPath]] = defaultdict(list)
    directions: tuple[tuple[PathDirection, dict[str, list[ForeignKey]]], ...] = (
        ("parent", outgoing),
        ("descendant", incoming),
    )
    for direction, adjacency in directions:
        queue: deque[tuple[str, tuple[ForeignKey, ...], frozenset[str]]] = deque(
            [(anchors.entity_table, (), frozenset({anchors.entity_table}))]
        )
        while queue and len(candidates[direction]) < config.max_paths:
            current, prefix, visited = queue.popleft()
            if len(prefix) >= config.max_depth:
                continue
            for relation in adjacency.get(current, []):
                next_table = (
                    relation.parent_table
                    if direction == "parent"
                    else relation.child_table
                )
                if next_table in visited:
                    continue
                edges = (*prefix, relation)
                candidate = RelationPath(anchors.entity_table, edges, direction)
                if candidate.is_parent or _path_has_usable_time(
                    database, candidate, anchors
                ):
                    candidates[direction].append(candidate)
                    if len(candidates[direction]) >= config.max_paths:
                        break
                if next_table in database.primary_keys:
                    queue.append((next_table, edges, visited | {next_table}))
    candidates["bridge"].extend(
        _enumerate_bridge_paths(database, anchors, config, incoming, outgoing)
    )
    ordered_families = [
        sorted(
            candidates[direction], key=lambda path: (len(path.edges), path.identity)
        )[: config.max_paths]
        for direction in ("parent", "descendant", "bridge")
    ]
    # Give every available family a deterministic share before taking a second
    # path from any one family. A wide schema cannot starve all bridge paths.
    ordered = [
        family[index]
        for index in range(max(map(len, ordered_families), default=0))
        for family in ordered_families
        if index < len(family)
    ]
    return tuple(ordered[: config.max_paths])


def _enumerate_bridge_paths(
    database: DatabaseView,
    anchors: AnchorDefinition,
    config: NativeFeatureConfig,
    incoming: dict[str, list[ForeignKey]],
    outgoing: dict[str, list[ForeignKey]],
) -> list[RelationPath]:
    """Enumerate one descendant association edge followed by parent edges."""
    if config.max_depth < 2:
        return []
    candidates: list[RelationPath] = []
    queue: deque[tuple[str, tuple[ForeignKey, ...], frozenset[str]]] = deque()
    for association_edge in incoming.get(anchors.entity_table, []):
        association = association_edge.child_table
        if (
            anchors.cutoff_column is not None
            and association not in database.time_columns
        ):
            continue
        queue.append(
            (
                association,
                (association_edge,),
                frozenset({anchors.entity_table, association}),
            )
        )
    while queue and len(candidates) < config.max_paths:
        current, prefix, visited = queue.popleft()
        if len(prefix) >= config.max_depth:
            continue
        for relation in outgoing.get(current, []):
            next_table = relation.parent_table
            if next_table in visited:
                continue
            edges = (*prefix, relation)
            candidate = RelationPath(anchors.entity_table, edges, "bridge")
            candidates.append(candidate)
            if len(candidates) >= config.max_paths:
                break
            if len(edges) < config.max_depth:
                queue.append((next_table, edges, visited | {next_table}))
    return candidates


def _path_has_usable_time(
    database: DatabaseView,
    path: RelationPath,
    anchors: AnchorDefinition,
) -> bool:
    if anchors.cutoff_column is None:
        return True
    if path.is_bridge:
        # A parent timestamp cannot establish when a timeless association
        # became visible. The first child is the bridge's occurrence clock.
        return path.edges[0].child_table in database.time_columns
    # A root creation time cannot establish when timeless child facts occurred.
    return any(edge.child_table in database.time_columns for edge in path.edges)


#: Composition features of a pair of values, and the kind of each.
_PAIR_OPERATIONS: tuple[tuple[FeatureOperation, ColumnKind], ...] = (
    ("mode", "categorical"),
    ("first", "categorical"),
    ("entropy", "numeric"),
    ("mode_share", "numeric"),
    ("last_share", "numeric"),
    ("distinct", "numeric"),
)


def get_composition_pair_features(
    aggregate_paths: tuple[RelationPath, ...],
    aggregate_columns: dict[RelationPath, tuple[list[str], list[str]]],
    composition_pairs: tuple[tuple[str, str], ...],
) -> list[FeatureSpec]:
    """Return the composition features of each named pair of categorical columns.

    A pair ``("drug_class.drug_class_name", "pharmacy.pharmacy_company_hash")``
    matches every two bridge paths that start with the same edge down to an
    association row, one ending at each column's table, where both columns are
    planned categories. A name that matches no paths plans nothing.
    """
    bridges = [path for path in aggregate_paths if path.is_bridge]
    features: list[FeatureSpec] = []
    for first_name, second_name in composition_pairs:
        first_table, first_column = first_name.split(".", 1)
        second_table, second_column = second_name.split(".", 1)
        for path in bridges:
            if (
                path.terminal_table != first_table
                or first_column not in aggregate_columns[path][1]
            ):
                continue
            for paired_path in bridges:
                if (
                    paired_path == path
                    or paired_path.terminal_table != second_table
                    or second_column not in aggregate_columns[paired_path][1]
                    or paired_path.edges[0] != path.edges[0]
                ):
                    continue
                features.extend(
                    FeatureSpec.pair_aggregate(
                        path,
                        paired_path,
                        operation,
                        column=first_column,
                        paired_column=second_column,
                        kind=kind,
                    )
                    for operation, kind in _PAIR_OPERATIONS
                )
    return features


def _candidate_tiers(
    database: DatabaseView,
    anchors: AnchorDefinition,
    config: NativeFeatureConfig,
    paths: tuple[RelationPath, ...],
    excluded: dict[str, set[str]],
    include_composition_features: bool = False,
    composition_pairs: tuple[tuple[str, str], ...] = (),
) -> list[list[FeatureSpec]]:
    parent_paths = tuple(path for path in paths if path.is_parent)
    aggregate_paths = tuple(
        path for path in paths if path.direction in {"bridge", "descendant"}
    )
    parent_columns: dict[RelationPath, tuple[list[str], list[str]]] = {}
    parent_kinds: dict[tuple[RelationPath, str], ColumnKind] = {}
    for path in parent_paths:
        table = path.terminal_table
        frame = database.tables[table]
        keys = {database.primary_keys.get(table)} | {
            relation.child_column
            for relation in database.foreign_keys
            if relation.child_table == table
        }
        keys.discard(None)
        ignored = keys | excluded.get(table, set())
        primary: list[str] = []
        categorical: list[str] = []
        for column in sorted(frame.columns):
            if column in ignored:
                continue
            kind = _column_kind(frame[column])
            if kind is None:
                continue
            parent_kinds[(path, column)] = kind
            destination = categorical if kind == "categorical" else primary
            destination.append(column)
        parent_columns[path] = (primary, categorical)

    aggregate_columns: dict[RelationPath, tuple[list[str], list[str]]] = {}
    for path in aggregate_paths:
        table = path.terminal_table
        frame = database.tables[table]
        keys = {database.primary_keys.get(table)} | {
            relation.child_column
            for relation in database.foreign_keys
            if relation.child_table == table
        }
        keys.discard(None)
        ignored = keys | {database.time_columns.get(table)} | excluded.get(table, set())
        ignored.discard(None)
        numeric: list[str] = []
        categorical: list[str] = []
        for column in sorted(frame.columns):
            if column in ignored:
                continue
            kind = _column_kind(frame[column])
            if kind == "numeric":
                numeric.append(column)
            elif (
                kind == "categorical"
                and _bounded_cardinality(
                    frame[column], config.max_categorical_cardinality
                )
                is not None
            ):
                categorical.append(column)
        aggregate_columns[path] = (numeric, categorical)

    tiers: list[list[FeatureSpec]] = [
        [FeatureSpec.aggregate(path, "count") for path in aggregate_paths]
    ]
    for group in (0, 1):
        tiers.append(
            _round_robin_columns(
                parent_paths,
                parent_columns,
                group,
                lambda path, column: FeatureSpec.lookup(
                    path,
                    column,
                    parent_kinds[(path, column)],
                ),
            )
        )
    if anchors.cutoff_column is not None:
        tiers.append(
            [FeatureSpec.aggregate(path, "recency") for path in aggregate_paths]
        )
        tiers.append(
            [FeatureSpec.aggregate(path, "mean_age") for path in aggregate_paths]
        )
        tiers.append(
            [FeatureSpec.aggregate(path, "oldest_age") for path in aggregate_paths]
        )

    # Spend the compact budget on stable distribution summaries first. Latest
    # values and sums remain available to the richer arms below.
    for operation in ("mean", "std", "min", "max"):
        tiers.append(
            _round_robin_columns(
                aggregate_paths,
                aggregate_columns,
                0,
                lambda path, column, op=operation: FeatureSpec.aggregate(
                    path, op, column=column
                ),
            )
        )

    tiers.append(
        _round_robin_columns(
            aggregate_paths,
            aggregate_columns,
            1,
            lambda path, column: FeatureSpec.aggregate(
                path, "mode", column=column, kind="categorical"
            ),
        )
    )
    if include_composition_features:
        tiers.append(
            _round_robin_columns(
                aggregate_paths,
                aggregate_columns,
                1,
                lambda path, column: FeatureSpec.aggregate(
                    path, "first", column=column, kind="categorical"
                ),
            )
        )
        tiers.append(
            _round_robin_columns(
                aggregate_paths,
                aggregate_columns,
                0,
                lambda path, column: FeatureSpec.aggregate(
                    path, "first", column=column
                ),
            )
        )
        for operation in ("entropy", "mode_share", "last_share", "distinct"):
            tiers.append(
                _round_robin_columns(
                    aggregate_paths,
                    aggregate_columns,
                    1,
                    lambda path, column, operation=operation: FeatureSpec.aggregate(
                        path, operation, column=column
                    ),
                )
            )
    if composition_pairs:
        tiers.append(
            get_composition_pair_features(
                aggregate_paths, aggregate_columns, composition_pairs
            )
        )
    if config.include_missing_fractions:
        tiers.append(
            _round_robin_columns(
                aggregate_paths,
                aggregate_columns,
                1,
                lambda path, column: FeatureSpec.aggregate(
                    path,
                    "missing_fraction",
                    column=column,
                ),
            )
        )

    tiers.append(
        _round_robin_columns(
            aggregate_paths,
            aggregate_columns,
            0,
            lambda path, column: FeatureSpec.aggregate(path, "last", column=column),
        )
    )
    tiers.append(
        _round_robin_columns(
            aggregate_paths,
            aggregate_columns,
            1,
            lambda path, column: FeatureSpec.aggregate(
                path, "last", column=column, kind="categorical"
            ),
        )
    )

    tiers.append(
        _round_robin_columns(
            aggregate_paths,
            aggregate_columns,
            0,
            lambda path, column: FeatureSpec.aggregate(path, "sum", column=column),
        )
    )
    if anchors.horizon_ns is not None:
        for multiplier in config.window_multipliers:
            window = anchors.horizon_ns * multiplier
            tiers.append(
                [
                    FeatureSpec.aggregate(path, "count", window_ns=window)
                    for path in aggregate_paths
                ]
            )
            for operation in ("mean", "std", "sum"):
                tiers.append(
                    _round_robin_columns(
                        aggregate_paths,
                        aggregate_columns,
                        0,
                        lambda path, column, op=operation, win=window: (
                            FeatureSpec.aggregate(
                                path,
                                op,
                                column=column,
                                window_ns=win,
                            )
                        ),
                    )
                )
    return tiers


def _round_robin_columns(
    paths: tuple[RelationPath, ...],
    columns: dict[RelationPath, tuple[list[str], list[str]]],
    group: int,
    build: Callable[[RelationPath, str], FeatureSpec],
) -> list[FeatureSpec]:
    """Interleave columns across paths so one wide table cannot take the budget."""
    width = max((len(columns[path][group]) for path in paths), default=0)
    return [
        build(path, columns[path][group][position])
        for position in range(width)
        for path in paths
        if position < len(columns[path][group])
    ]


def _column_kind(series: pd.Series) -> ColumnKind | None:
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_bool_dtype(series):
        return "categorical"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    # Object columns can mix scalar and nested values while retaining the same
    # dtype. Inspect every present value so row order cannot change the plan.
    for value in series:
        if value is None or (np.ndim(value) == 0 and pd.isna(value)):
            continue
        if not pd.api.types.is_scalar(value):
            return None
    return "categorical"


def _bounded_cardinality(series: pd.Series, limit: int) -> int | None:
    """Return bounded cardinality without materializing every distinct value."""
    values: set[object] = set()
    for value in series:
        if value is None or bool(pd.isna(value)):
            continue
        normalized = value.item() if isinstance(value, np.generic) else value
        values.add(normalized)
        if len(values) > limit:
            return None
    return len(values)


__all__ = [
    "build_feature_plan",
    "get_sibling_distinct_columns",
    "get_sibling_match_columns",
    "get_sibling_paths",
]
