"""Build bounded historical entity/anchor candidates for temporal PQL labels."""

import warnings
from collections.abc import Sequence
from datetime import UTC, datetime

from nori_rel.contracts import SchemaGraph
from nori_rel.pql._generated.PQLGrammarParser import PQLGrammarParser as Grammar
from nori_rel.pql.binding import bind_columns, get_entity_created_at_column
from nori_rel.pql.dialects import SQLDialect
from nori_rel.pql.label_events import (
    get_label_event_aggregates,
    get_label_event_candidates_sql,
)
from nori_rel.pql.label_sql import compile_label_sql
from nori_rel.pql.sql_expressions import Scalar, filter_sql
from nori_rel.pql.tree_access import (
    node_ids,
    query_parts,
    target_aggregates,
)
from nori_rel.pql.window import target_window

DEFAULT_MAX_CONTEXT_ROWS = 5_000


def _utc(value: datetime) -> datetime:
    """Normalize an aware datetime to UTC; reject naive dates and strings."""
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ValueError("Times must be timezone-aware datetime objects")
    return value.astimezone(UTC)


def get_eligible_anchor_times(
    tree: Grammar.ProgContext,
    *,
    anchor_times: Sequence[datetime],
    anchor_time: datetime,
    observed_through: datetime,
) -> list[datetime]:
    """Return the distinct UTC anchors, oldest first, whose labels are observed.

    An anchor is eligible when every aggregate's whole label interval ends by
    min(anchor_time, observed_through); the furthest interval end governs, and
    a delayed start does not bring an anchor forward. Calendar months vary in
    length, so each anchor's own interval end is checked, not one threshold.
    """
    cutoff = min(_utc(anchor_time), _utc(observed_through))
    windows = [target_window(a) for a in target_aggregates(tree)]
    requested = sorted(set(map(_utc, anchor_times)))
    return [a for a in requested if all(w.is_observed(a, cutoff) for w in windows)]


def compile_context_sql(
    tree: Grammar.ProgContext,
    schema: SchemaGraph,
    *,
    anchor_times: Sequence[datetime],
    anchor_time: datetime,
    observed_through: datetime,
    entity_created_at: str,
    max_context_rows: int | None = DEFAULT_MAX_CONTEXT_ROWS,
    dialect: SQLDialect | None = None,
    candidate_sql: str | None = None,
) -> tuple[str, dict[str, Scalar]]:
    """Compile a read-only SELECT of eligible historical candidates.

    Args:
        tree: parse_pql("PREDICT COUNT(orders.*,0,30,days) FOR users.user_id=42 "
            "WHERE users.country='US'"). Supports v0 temporal aggregate targets;
            direct labels need a different exclusion policy and are rejected.
        schema: Registered SchemaGraph: schema_name plus tables with columns
            [{name, data_type, nullable}], primary_key, foreign_keys and event
            time_column. For example orders.buyer_id -> users.user_id, with
            orders.event_time mapped to TIMESTAMPTZ.
        anchor_times: Explicit candidate historical timestamps, e.g.
            [datetime(2026, 1, 1, tzinfo=UTC)]. Duplicates are removed.
            The caller chooses the schedule; there is no per-entity 24-row cap.
        anchor_time: Scoring cutoff, as an aware datetime. Historical outcomes
            must be completely observed by this instant.
        observed_through: Verified target-table completeness watermark, as an
            aware datetime. This is NOT inferred from MAX(event_time). The
            effective cutoff is min(anchor_time, observed_through).
        entity_created_at: Exact registered entity column name, e.g. "created_at".
            Caller guarantees it means when that entity first existed/was known.
            Null creation times exclude entities. Other entity attributes must
            be immutable or valid at all supplied anchors; no versioned joins.
        max_context_rows: Positive integer cap on candidate output rows, or
            None for every candidate that can have a known label. Defaults to
            DEFAULT_MAX_CONTEXT_ROWS (5,000); callers may override it. An
            integer cap applies after time eligibility and the PQL context
            filter, newest anchor first. It is NOT a guarantee of that many
            usable labels: later preparation drops unknown labels without
            refilling the cap, and then applies model/element budgets. None
            removes the LIMIT, and when the label is unknown unless an
            aggregate has an event in its window (a ratio's denominator; AVG,
            MIN, MAX, FIRST, LAST), candidates start from that aggregate's
            events in each anchor's window instead of every entity at every
            anchor. The caller then chooses among the labelled rows.
        dialect: Engine SQL syntax and optional snapshot-pinned source relations.
            Defaults to PostgreSQL for existing callers.
        candidate_sql: Optional caller SELECT over ``candidates(entity_id,
            anchor_time)`` that returns the pairs to keep; needs
            ``max_context_rows=None``. It is embedded as a CTE, can only
            narrow the candidates, and duplicates in it do not repeat rows.

    Returns:
        (sql, params), for cursor.execute(sql, params).fetchall(). Rows are
        (context_row_id: int, entity_id: native primary-key type,
        anchor_time: datetime),
        e.g. [(1, 7, datetime(2026, 1, 1, tzinfo=UTC)), (2, 42, same_time)].
        All anchors satisfy anchor + end <= the effective cutoff, where end is
        the furthest label-interval end across the target's aggregates; entities
        existed by their anchor. FOR IDs do not restrict this historical cohort.

        Selection is deterministic: newest anchor first, then primary key.
        IDs are unique within this result, not durable across data changes,
        and follow the same order without a cap.

    Raises:
        ValueError: Direct target, invalid limits, naive times or invalid creation
            column. PQL scope/binding/type errors propagate from the compilers.

    No data is read here. The executor must use a consistent UTC source snapshot,
    enforce statement/scan limits, and persist these rows as the context relation
    for compile_label_sql. LIMIT bounds returned rows, not database work. Inline
    target filters belong to label SQL, so zero-event context rows remain eligible.
    """
    dialect = dialect or SQLDialect()
    identifier = dialect.identifier
    parts = query_parts(tree)
    aggregate = parts["aggregate"]
    if aggregate is None:
        raise ValueError("Historical context SQL requires a temporal aggregate target")
    # Validate the entire target, relationship, filters and horizon before execution.
    compile_label_sql(tree, schema, dialect=dialect)
    if max_context_rows is not None and (
        type(max_context_rows) is not int or max_context_rows <= 0
    ):
        raise ValueError("max_context_rows must be a positive integer or None")
    if candidate_sql is not None and max_context_rows is not None:
        raise ValueError(
            "candidate_sql filters the uncapped read: max_context_rows=None"
        )
    anchors = get_eligible_anchor_times(
        tree,
        anchor_times=anchor_times,
        anchor_time=anchor_time,
        observed_through=observed_through,
    )
    windows = [target_window(a) for a in target_aggregates(tree)]
    _warn_about_unobserved_anchors(
        sorted(set(map(_utc, anchor_times))),
        anchors,
        min(_utc(anchor_time), _utc(observed_through)),
        in_days=any(w.unit == "days" for w in windows),
    )

    ids, bound = node_ids(tree), bind_columns(tree, schema)
    entity, key = bound[ids[parts["entity_column"]]]
    entity_created_at = get_entity_created_at_column(
        schema, entity, entity_created_at
    ).name
    params: dict[str, Scalar] = {}
    if max_context_rows is not None:
        params["max_context_rows"] = max_context_rows

    def column_ref(node):
        """Return ('u."country"', ColumnSchema) for a bound context column."""
        return f"u.{identifier(bound[ids[node]][1].name)}", bound[ids[node]][1]

    predicate = filter_sql(parts["context_filter"], column_ref, params)
    values = []
    for i, value in enumerate(anchors):
        name = f"anchor_{i}"
        params[name] = value
        values.append(f"%({name})s")
    anchor_sql = dialect.anchor_relation(values)
    # TODO: https://github.com/Synthefy/nori-rel/issues/36
    # Keep the explicit candidate row cap for now. Large-context preparation,
    # retrieval and inference need separate budgets; LIMIT does not bound scans.
    users = dialect.table_reference(schema.schema_name, entity.name)
    label_event_aggregates = (
        get_label_event_aggregates(tree) if max_context_rows is None else []
    )
    if label_event_aggregates:
        # Any one required aggregate gives a superset of the labelled rows;
        # label SQL computes the labels and unknown ones are still dropped.
        event_candidates = get_label_event_candidates_sql(
            tree,
            schema,
            label_event_aggregates[0],
            anchors_relation="anchors",
            params=params,
            dialect=dialect,
        )
        sql = f"""WITH anchors(anchor_time) AS ({anchor_sql}),
label_event_candidates AS (
    {event_candidates}
),
selected AS (
    SELECT u.{identifier(key.name)} AS entity_id, c.anchor_time
    FROM label_event_candidates AS c
    JOIN {users} AS u ON u.{identifier(key.name)} = c.entity_id
    WHERE u.{identifier(entity_created_at)} <= c.anchor_time"""
    else:
        sql = f"""WITH anchors(anchor_time) AS ({anchor_sql}),
selected AS (
    SELECT u.{identifier(key.name)} AS entity_id, a.anchor_time
    FROM {users} AS u CROSS JOIN anchors AS a
    WHERE u.{identifier(entity_created_at)} <= a.anchor_time"""
    if predicate:
        sql += f"\n      AND {predicate}"
    if max_context_rows is not None:
        sql += """
    ORDER BY a.anchor_time DESC, entity_id
    LIMIT %(max_context_rows)s"""
    sql += "\n)"
    candidate_filter = ""
    if candidate_sql is not None:
        # Literal percent signs must survive pyformat parameter binding.
        sql += f""",
candidates AS (SELECT entity_id, anchor_time FROM selected),
chosen AS (
{candidate_sql.replace("%", "%%")}
)"""
        candidate_filter = """ AS s
WHERE EXISTS (
    SELECT 1 FROM chosen AS c
    WHERE c.entity_id = s.entity_id AND c.anchor_time = s.anchor_time
)"""
    sql += f"""
SELECT ROW_NUMBER() OVER (ORDER BY anchor_time DESC, entity_id) AS context_row_id,
       entity_id, anchor_time
FROM selected{candidate_filter}
ORDER BY context_row_id"""
    return sql, params


#: How many dropped anchors a warning lists by date.
_MAX_LISTED_ANCHORS = 5


def _warn_about_unobserved_anchors(
    requested: Sequence[datetime],
    kept: Sequence[datetime],
    cutoff: datetime,
    *,
    in_days: bool,
) -> None:
    """Warn when requested anchor_times give no training rows.

    An anchor is dropped when its label interval ends after the cutoff, the
    earlier of anchor_time and observed_through. Without a warning, a 30-day
    window on monthly data silently loses every February anchor. ``in_days``
    adds the suggestion of a calendar-month window.
    """
    dropped = [anchor for anchor in requested if anchor not in set(kept)]
    if not dropped:
        return
    listed = ", ".join(a.isoformat() for a in dropped[:_MAX_LISTED_ANCHORS])
    if len(dropped) > _MAX_LISTED_ANCHORS:
        listed += f" and {len(dropped) - _MAX_LISTED_ANCHORS} more"
    hint = (
        " On monthly data, a window in months, such as (0, 1, months), holds "
        "exactly one calendar month."
        if in_days
        else ""
    )
    warnings.warn(
        f"{len(dropped)} of {len(requested)} anchor_times give no training rows "
        f"because their label window ends after {cutoff.isoformat()}, the earlier "
        f"of anchor_time and observed_through: {listed}.{hint}",
        UserWarning,
        stacklevel=3,
    )
