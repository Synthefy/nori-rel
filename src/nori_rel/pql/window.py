"""One owner for a target's label interval: read it, validate it, render it.

An aggregate's two offsets, as in SUM(orders.amount,30,60,days), are the
window the label is computed over: what happens between 30 and 60 days after
a context row's anchor_time. Every rule about that window lives here, because
the same two numbers drive three different things and used to be re-read, and
re-interpreted, at each of them:

    label SQL    the event-time bounds in the LEFT JOIN that aggregates rows
    context SQL  which historical anchors are eligible at all
    scope        whether the offsets are a form v0 accepts

Scope still judges the form (two integer offsets of at most nine digits, the
first unsigned, the unit days) and rejects a backward-looking interval, whose
events would already sit in the row's own features. This module judges the
values and renders them, so an interval that could not label anything cannot
be constructed, whether it came from a parse tree or from a cache.

Two rules here are easy to get wrong in ways nothing would report:

Eligibility keys on the end alone. An anchor is usable when anchor + end is
observed, never when anchor + (end - start) is. The second reading looks
sensible for a delayed window, "it is only 30 days long", and silently admits
anchors whose interval lies entirely in unobserved time, labelling them from
data that has not arrived.

Features are not this module's business, and must stay that way. A delayed
window leaves a gap between the anchor and the interval; the snapshot holds
that gap in full and it predicts the label well, but at scoring time the
anchor is now, so those days have not happened. Feature building cuts at the
anchor for every window, and a test pins the two feature matrices equal.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

from nori_rel.pql._generated.PQLGrammarParser import PQLGrammarParser as Grammar
from nori_rel.pql.dialects import SQLDialect
from nori_rel.pql.sql_expressions import Scalar
from nori_rel.pql.tree_access import window_text


@dataclass(frozen=True)
class TargetWindow:
    """The label interval of one aggregate, (anchor + start, anchor + end] days.

    Both bounds are whole days measured from the context row's anchor_time in
    24-hour steps. The interval is left-exclusive and right-inclusive, so
    (0, 30] is today's "next 30 days" and (30, 60] follows it with no overlap
    and no gap. start is 0 for an immediate target and positive for a delayed
    one such as SUM(orders.amount,30,60,days). A negative start, a non-positive
    end or an interval that does not end after it starts cannot be built:
    scope rejects the first as a form, and the constructor rejects all three
    as values, so a window that came from a cache is held to the same rule.

    Example:
        >>> from nori_rel.pql import parse_pql
        >>> from nori_rel.pql.tree_access import query_parts
        >>> text = "PREDICT SUM(orders.amount,30,60,days) FOR EACH users.user_id"
        >>> tree = parse_pql(text)
        >>> window = target_window(query_parts(tree)["aggregate"])
        >>> window
        TargetWindow(start=30, end=60)
        >>> from datetime import UTC, datetime
        >>> window.latest_anchor(datetime(2026, 3, 3, tzinfo=UTC)).date().isoformat()
        '2026-01-02'
    """

    start: int
    end: int

    def __post_init__(self) -> None:
        """Refuse an interval that could not label anything correctly.

        Raises:
            ValueError: start is negative, end is not positive, or end <= start.
        """
        if self.start < 0:
            raise ValueError("Label interval cannot start before the anchor")
        if self.end <= 0:
            raise ValueError("Label horizon must be positive")
        if self.end <= self.start:
            raise ValueError(
                "Label interval must end after it starts; "
                f"got ({self.start}, {self.end}] days"
            )

    def latest_anchor(self, cutoff: datetime) -> datetime:
        """Return the newest anchor whose whole interval is observed by cutoff.

        Only end matters: the interval closes at anchor + end, so an anchor is
        eligible when anchor + end <= cutoff. The interval's length end - start
        plays no part. Using it would admit anchors whose delayed interval lies
        entirely after the cutoff and label them from data that has not arrived.

        Raises:
            ValueError: end is too large to subtract from cutoff.
        """
        try:
            return cutoff - timedelta(days=self.end)
        except OverflowError:
            raise ValueError("Horizon exceeds the supported datetime range") from None

    def predicate(
        self,
        event_time: str,
        anchor: str,
        params: dict[str, Scalar],
        *,
        dialect: SQLDialect | None = None,
    ) -> str:
        """Render the SQL bounds on event_time and bind the offsets as parameters.

        Args:
            event_time: SQL expression for the event timestamp, e.g. 'o."event_time"'.
            anchor: SQL expression for the row's anchor, e.g. "e.anchor_time".
            params: Statement parameters; horizon_days is always added, and
                start_days only for a delayed window, so an immediate window
                renders exactly as it did before delayed windows existed.
            dialect: SQL syntax for exact 24-hour additions; defaults to PostgreSQL.

        Returns:
            Two comparisons, the second on its own line indented one space
            and led by AND, so the text drops straight into the LEFT JOIN
            block that label SQL builds.
        """
        dialect = dialect or SQLDialect()
        params["horizon_days"] = self.end
        lower = anchor
        if self.start:
            params["start_days"] = self.start
            lower = dialect.add_days(anchor, "%(start_days)s")
        upper = dialect.add_days(anchor, "%(horizon_days)s")
        return f"{event_time} > {lower}\n AND {event_time} <= {upper}"


def target_window(node: Grammar.AggregationContext) -> TargetWindow:
    """Read an aggregate's window as whole days.

    Scope has already established the form: two integer offsets of at most
    nine digits, the first unsigned, the unit days. The constructor checks
    the values.

    Raises:
        ValueError: end is not positive, or the interval is empty or reversed.
    """
    start_text, end_text, _ = window_text(node)
    return TargetWindow(int(start_text), int(end_text))
