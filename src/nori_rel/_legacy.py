"""Limited aggregate-argument compatibility; all execution goes through PQL."""

import json
import re

from nori_rel.contracts import SchemaGraph

_OPERATIONS = {
    "count": "COUNT",
    "total": "SUM",
    "average": "AVG",
    "minimum": "MIN",
    "maximum": "MAX",
}
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_$]*")


def aggregate_query(
    schema: SchemaGraph,
    *,
    entity: str,
    target: str,
    target_time: str,
    operation: str,
    lookahead: str,
    entity_ids: list | None,
) -> str:
    """Translate simple legacy aggregates into PQL; never read rows or build labels.

    Input example: entity='users', target='orders.amount',
    target_time='orders.event_time', operation='total', lookahead='30 days',
    entity_ids=[42], plus discovered schema with users' single PK user_id and
    orders' registered timestamp event_time. Output:
    PREDICT SUM(`orders`.`amount`,0,30,days) FOR `users`.`user_id`=42.

    COUNT counts non-null target values, preserving the old count operation.
    IDs are integers or strings; identifiers must be simple names. Rejects
    absent columns, mismatched timestamps and unsupported shapes. PQL binding
    subsequently validates types and the unambiguous direct relationship.
    """
    if operation not in _OPERATIONS:
        raise ValueError(
            "legacy adapter supports count/total/average/minimum/maximum only"
        )
    horizon = re.fullmatch(r"([1-9][0-9]*)\s+days?", lookahead.strip(), re.IGNORECASE)
    if horizon is None:
        raise ValueError("legacy lookahead must be positive whole days; use PQL")
    names = [entity, *target.split("."), *target_time.split(".")]
    if len(names) != 5 or any(not _NAME.fullmatch(name) for name in names):
        raise ValueError(
            "legacy adapter requires simple entity and table.column names; use PQL"
        )
    _, table, column, time_table, time_column = names
    tables = {t.name: t for t in schema.tables}
    if entity not in tables or table not in tables:
        raise ValueError("legacy entity/target table is not registered")
    key = tables[entity].primary_key
    if len(key) != 1 or not _NAME.fullmatch(key[0]):
        raise ValueError(
            "legacy adapter requires one simple entity primary key; use PQL"
        )
    if column not in {c.name for c in tables[table].columns}:
        raise ValueError("legacy target column is not registered")
    if time_table != table or tables[table].time_column != time_column:
        raise ValueError("target_time must match the timestamp registered at connect()")
    selection = f"EACH `{entity}`.`{key[0]}`"
    if entity_ids is not None:
        if not entity_ids or any(type(v) not in (int, str) for v in entity_ids):
            raise ValueError(
                "entity_ids must be a nonempty list of integers or strings"
            )
        if any(isinstance(v, str) and "\x00" in v for v in entity_ids):
            raise ValueError("entity_ids cannot contain NUL")
        ids = [json.dumps(v) for v in entity_ids]
        suffix = "=" + ids[0] if len(ids) == 1 else " IN (" + ",".join(ids) + ")"
        selection = f"`{entity}`.`{key[0]}`{suffix}"
    return (
        f"PREDICT {_OPERATIONS[operation]}(`{table}`.`{column}`,"
        f"0,{int(horizon[1])},days) FOR {selection}"
    )
