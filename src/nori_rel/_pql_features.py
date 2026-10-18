"""Align temporal PQL context and scoring rows with cutoff-aware DFS inputs."""

import pandas as pd

from nori_rel.connectors.base import Snapshot
from nori_rel.feature import FeatureBackend, FeatureMatrices
from nori_rel.feature.dfs import CUTOFF_COLUMN as _CUTOFF
from nori_rel.pql import parse_pql
from nori_rel.pql.binding import bind_columns, get_entity_created_at_column
from nori_rel.pql.sql_expressions import _typed_literal, column_kind
from nori_rel.pql.tree_access import node_ids, query_parts

_ROW_ID = "__nori_pql_row_id"


def build_prediction_rows(
    snapshot: Snapshot, query: str, entity_created_at: str
) -> pd.DataFrame:
    """Resolve FOR into (query_row_id, entity_id, anchor_time) rows.

    Inputs: same-snapshot tables/schema, PQL text and entity creation column.
    FOR EACH scores eligible entities; = / IN uses typed IDs and rejects missing
    or not-yet-created entities. Top-level WHERE only filters historical context.
    Optional snapshot.scoring_filter_ids intersects IDs after missing-ID checks.
    Output IDs are unique within this scoring batch; original PK types survive.
    """
    tree = parse_pql(query)
    parts = query_parts(tree)
    table, key = bind_columns(tree, snapshot.schema)[
        node_ids(tree)[parts["entity_column"]]
    ]
    frame = snapshot.tables[table.name]
    created_column = get_entity_created_at_column(
        snapshot.schema, table, entity_created_at
    )
    created = pd.to_datetime(frame[created_column.name], utc=True)
    ids = frame.loc[created.notna() & (created <= snapshot.as_of), key.name]
    selection = parts["selection"]
    if selection is not None:
        constant = selection.constant()
        nodes = (
            constant.array().constant() if constant.array() is not None else [constant]
        )
        requested = [_typed_literal(n, column_kind(key)) for n in nodes]
        if any(value is None for value in requested):
            raise RuntimeError("scoring IDs cannot be NULL")
        if set(requested).difference(ids):
            raise RuntimeError(
                "scoring IDs are missing or not yet created at anchor_time"
            )
        ids = ids[ids.isin(requested)]
    if snapshot.scoring_filter_ids is not None:
        ids = ids[ids.isin(snapshot.scoring_filter_ids)]
    ids = ids.sort_values().reset_index(drop=True)
    return pd.DataFrame(
        {
            "query_row_id": range(1, len(ids) + 1),
            "entity_id": ids,
            "anchor_time": snapshot.as_of,
        }
    )


def build_pql_features(
    snapshot: Snapshot,
    context: pd.DataFrame,
    query: str,
    prediction_rows: pd.DataFrame,
    engine: FeatureBackend,
) -> tuple[FeatureMatrices, pd.DataFrame]:
    """Return aligned feature matrices and scoring identities from one snapshot.

    prediction_rows is build_prediction_rows output for this query/snapshot.
    context columns: context_row_id, entity_id, anchor_time, label. Labels never
    enter DFS. Train matrix index is context_row_id; query matrix index is
    query_row_id. Scoring rows contain query_row_id, entity_id, anchor_time.
    Each feature uses its row's cutoff; unmapped source tables must be immutable
    and mapped events immediately available. Rejects duplicate context IDs.
    """
    if context["context_row_id"].duplicated().any():
        raise RuntimeError("context_row_id must be unique")
    tree = parse_pql(query)
    parts = query_parts(tree)
    table, key = bind_columns(tree, snapshot.schema)[
        node_ids(tree)[parts["entity_column"]]
    ]
    if key.name in {_ROW_ID, _CUTOFF}:
        raise RuntimeError("entity key conflicts with a reserved feature column")
    scoring = prediction_rows
    train = context[["entity_id", "anchor_time"]].rename(
        columns={"entity_id": key.name, "anchor_time": _CUTOFF}
    )
    train[_ROW_ID] = ["context:" + str(i) for i in context["context_row_id"]]
    query_targets = scoring[["entity_id", "anchor_time"]].rename(
        columns={"entity_id": key.name, "anchor_time": _CUTOFF}
    )
    query_targets[_ROW_ID] = ["query:" + str(i) for i in scoring["query_row_id"]]
    matrices = engine.compute(
        snapshot, train, query_targets, table.name, row_id_column=_ROW_ID
    )
    matrices.train.index = pd.Index(context["context_row_id"], name="context_row_id")
    matrices.query.index = pd.Index(scoring["query_row_id"], name="query_row_id")
    return matrices, scoring
