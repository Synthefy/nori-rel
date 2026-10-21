"""Rules for static PQL targets, which have no label window.

A static target is a stored column of the entity row, such as
``PREDICT accounts.balance FOR EACH accounts.account_id``: the model learns it
from the other rows whose value is stored. These helpers decide which request
arguments apply, which task the target is, and remove the target column from
the tables features are built from.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime
from typing import TYPE_CHECKING, Literal

import pandas as pd

from nori_rel.contracts import PredictionTask, SchemaGraph
from nori_rel.pql import parse_pql
from nori_rel.pql._generated.PQLGrammarParser import PQLGrammarParser as Grammar
from nori_rel.pql.binding import bind_columns
from nori_rel.pql.sql_expressions import column_kind
from nori_rel.pql.tree_access import node_ids, query_parts

if TYPE_CHECKING:
    # The client imports this module; connectors stay out of light imports.
    from nori_rel.connectors.base import Snapshot

#: Stands in for a static target's shared read time wherever a run's result
#: must not depend on when it read the source: random context keys and the
#: context-rows hash. The read time itself still sets every feature cutoff.
READ_TIME_STAND_IN = pd.Timestamp(0, tz="UTC")

# Feature strategies whose windows or search scale with a label horizon.
_HORIZON_FEATURE_STRATEGIES = {"noridfs", "agentic", "saved_program"}


def get_pql_target_kind(query: str) -> Literal["static", "temporal"] | None:
    """Return the kind of target a PQL query predicts, or None if it is invalid.

    ``PREDICT users.income FOR EACH users.user_id`` is "static" (a column, no
    time window); ``PREDICT COUNT(orders.*,0,30,days) FOR EACH users.user_id``
    is "temporal". A query that does not parse or is outside v0 returns None,
    so the prediction reports that error, with its location, instead of a
    complaint about time arguments.
    """
    try:
        tree = parse_pql(query)
        return "static" if query_parts(tree)["aggregate"] is None else "temporal"
    except ValueError:
        return None


def require_static_target_arguments(
    *,
    anchor_times: Sequence[datetime] | None,
    anchor_time: datetime | None,
    observed_through: datetime | None,
    feature_strategy: str,
    has_pipeline: bool,
    entity_time: str | None = None,
) -> None:
    """Raise ValueError for a static-target request argument that cannot apply.

    A static target is read as stored at the time of the read, so it takes no
    anchor_times, observed_through or anchor_time. Its features come from
    FastDFS or a custom FeatureBackend: NoriDFS, the agent search and saved
    programs scale with a label horizon it does not have. Saved pipelines do
    not replay static targets yet. entity_time anchors a label window at each
    row's time, which a static target does not have; entity_created_at gives
    each row its own feature cutoff instead.
    """
    given = [
        name
        for name, value in (
            ("anchor_times", anchor_times),
            ("anchor_time", anchor_time),
            ("observed_through", observed_through),
        )
        if value is not None
    ]
    if given:
        raise ValueError(
            f"a static target such as PREDICT users.income takes no "
            f"{', '.join(given)}: its labels are read as stored now, with no "
            "label window. Omit them; pass entity_created_at to compute each "
            "row's features as of its own time"
        )
    if entity_time is not None:
        raise ValueError(
            "a static target takes no entity_time: it has no label window to "
            "anchor. Pass entity_created_at to compute each row's features as "
            "of its own time"
        )
    if feature_strategy in _HORIZON_FEATURE_STRATEGIES:
        raise ValueError(
            "a static target uses FastDFS or a custom FeatureBackend; "
            f"{feature_strategy} scales with a label horizon that a static "
            "target does not have"
        )
    if has_pipeline:
        raise ValueError("saved pipelines do not replay static targets yet")


def get_static_target_task(
    tree: Grammar.ProgContext,
    schema: SchemaGraph,
    task: PredictionTask | str | None,
    known_values: pd.Series,
) -> PredictionTask:
    """Return the task a static target resolves to, after checking task=.

    A comparison (``users.income > 50000``) or a Boolean column is binary
    classification, and a task='regression' on it raises. A numeric column is
    regression, except that one whose known values are only 0 and 1 is
    ambiguous and needs task=; task='classification' on a numeric column
    needs values in {0, 1}. known_values is every non-NULL value of the
    target column, from get_known_target_values, not only the context rows a
    cap keeps. Text targets without a comparison are rejected earlier, by the
    label compiler.
    """
    parts = query_parts(tree)
    column = bind_columns(tree, schema)[node_ids(tree)[parts["target_column"]]][1]
    name = parts["target_column"].getText()
    requested = None if task is None else PredictionTask(task)
    if parts["comparison"] is not None or column_kind(column) == "bool":
        if requested is PredictionTask.REGRESSION:
            raise ValueError("task contradicts the PQL target")
        return PredictionTask.CLASSIFICATION
    is_zero_one = bool(len(known_values)) and set(known_values).issubset({0, 1})
    if requested is PredictionTask.CLASSIFICATION:
        if not is_zero_one:
            raise ValueError(
                f"task='classification' needs a 0/1 column, but {name} holds "
                f"other values; compare it in the query, e.g. PREDICT {name} > 0, "
                "or use task='regression'"
            )
        return PredictionTask.CLASSIFICATION
    if requested is None and is_zero_one:
        raise ValueError(
            f"{name} holds only 0 and 1: pass task='classification' for a "
            "probability or task='regression' for a number"
        )
    return PredictionTask.REGRESSION


def get_known_target_values(snapshot: Snapshot, tree: Grammar.ProgContext) -> pd.Series:
    """Return every non-NULL value of a static target's column in the snapshot.

    The snapshot holds the whole entity table from the same read as the
    context, so these are all the stored values, not only the context rows
    that max_context_rows or a ContextPolicy keeps.
    """
    parts = query_parts(tree)
    table, column = bind_columns(tree, snapshot.schema)[
        node_ids(tree)[parts["target_column"]]
    ]
    return snapshot.tables[table.name][column.name].dropna()


def get_snapshot_without_target_column(
    snapshot: Snapshot, tree: Grammar.ProgContext
) -> Snapshot:
    """Return the snapshot with a static target's column removed everywhere.

    The column leaves its table's frame and its schema, with any foreign key
    built on it, so no feature can read any row's label: not the row's own
    column, and not a path through related tables such as accounts ->
    transfers -> accounts. Other columns that copy or encode the target are
    the caller's responsibility. Raises ValueError when the target is the
    entity key, which every row knows.
    """
    parts = query_parts(tree)
    table, column = bind_columns(tree, snapshot.schema)[
        node_ids(tree)[parts["target_column"]]
    ]
    if column.name in table.primary_key:
        raise ValueError("a static target cannot be the entity's primary key")
    tables = dict(snapshot.tables)
    tables[table.name] = tables[table.name].drop(columns=[column.name])
    schema_tables = [
        other
        if other.name != table.name
        else other.model_copy(
            update={
                "columns": [c for c in other.columns if c.name != column.name],
                "foreign_keys": [
                    key for key in other.foreign_keys if column.name not in key.columns
                ],
            }
        )
        for other in snapshot.schema.tables
    ]
    return replace(
        snapshot,
        tables=tables,
        schema=snapshot.schema.model_copy(update={"tables": schema_tables}),
        foreign_keys=[
            key
            for key in snapshot.foreign_keys
            if (key[0], key[1]) != (table.name, column.name)
        ],
    )
