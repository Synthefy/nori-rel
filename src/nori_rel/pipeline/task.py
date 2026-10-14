"""The identity of a PQL task, apart from the entities it scores."""

from __future__ import annotations

import json

from nori_rel.pql import parse_pql
from nori_rel.pql.tree_access import query_parts


def get_pql_task_signature(query: str) -> str:
    """Return the target, entity and WHERE of a PQL query, without its FOR IDs.

    Two queries with the same signature predict the same task for possibly
    different entities, so a pipeline can score either one.
    """
    parts = query_parts(parse_pql(query))
    context_filter = parts["context_filter"]
    return json.dumps(
        {
            "target": parts["target"].getText(),
            "entity": parts["entity_column"].getText(),
            "where": None if context_filter is None else context_filter.getText(),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


__all__ = ["get_pql_task_signature"]
