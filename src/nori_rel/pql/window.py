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
first unsigned, the unit days or months) and rejects a backward-looking interval, whose
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

import calendar
from dataclasses import dataclass
from datetime import datetime, timedelta

from nori_rel.pql._generated.PQLGrammarParser import PQLGrammarParser as Grammar
from nori_rel.pql.dialects import SQLDialect
from nori_rel.pql.sql_expressions import Scalar
from nori_rel.pql.tree_access import window_text

#: The units a label window can be counted in.
WINDOW_UNITS = ("days", "months")
#: The mean length of a calendar month in days, for a month window's length.
_DAYS_PER_MONTH = 365.25 / 12


@dataclass(frozen=True)
class TargetWindow:
    """The label interval of one aggregate, (anchor + start, anchor + end] units.

    With unit "days", both bounds are whole days measured from the context
    row's anchor_time in 24-hour steps. With unit "months", they are calendar
    months in UTC: (0, 1] from an anchor on the 1st is that whole month, and
    a month added to the 31st lands on the last day of a shorter month, as the
    databases compute it. The interval is left-exclusive and right-inclusive, so
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
    unit: str = "days"

    def __post_init__(self) -> None:
        """Refuse an interval that could not label anything correctly.

        Raises:
            ValueError: the unit is not days or months, start is negative, end
                is not positive, or end <= start.
        """
        if self.unit not in WINDOW_UNITS:
            raise ValueError(
                f"Label window unit must be one of {', '.join(WINDOW_UNITS)}"
            )
        if self.start < 0:
            raise ValueError("Label interval cannot start before the anchor")
        if self.end <= 0:
            raise ValueError("Label horizon must be positive")
        if self.end <= self.start:
            raise ValueError(
                "Label interval must end after it starts; "
                f"got ({self.start}, {self.end}] {self.unit}"
            )

    def latest_anchor(self, cutoff: datetime) -> datetime:
        """Return the newest anchor whose whole interval is observed by cutoff.

        Only end matters: the interval closes at anchor + end, so an anchor is
        eligible when anchor + end <= cutoff. The interval's length end - start
        plays no part. Using it would admit anchors whose delayed interval lies
        entirely after the cutoff and label them from data that has not arrived.
        For a month window this is cutoff minus end calendar months, which can
        exclude an anchor late in a long month that ``is_observed`` admits;
        ``is_observed`` is the exact rule.

        Raises:
            ValueError: end is too large to subtract from cutoff.
        """
        try:
            if self.unit == "months":
                return _get_moment_plus_months(cutoff, -self.end)
            return cutoff - timedelta(days=self.end)
        except (OverflowError, ValueError):
            raise ValueError("Horizon exceeds the supported datetime range") from None

    def get_label_end(self, anchor: datetime) -> datetime:
        """Return when an anchor's label interval closes: anchor + end units.

        Raises:
            ValueError: anchor + end is outside the supported datetime range.
        """
        try:
            if self.unit == "months":
                return _get_moment_plus_months(anchor, self.end)
            return anchor + timedelta(days=self.end)
        except (OverflowError, ValueError):
            raise ValueError("Horizon exceeds the supported datetime range") from None

    def is_observed(self, anchor: datetime, cutoff: datetime) -> bool:
        """Return whether an anchor's whole label interval closes by cutoff.

        This is the eligibility rule for a historical anchor: its interval ends
        at anchor + end, and every event in it must have arrived by cutoff.

        Raises:
            ValueError: anchor + end is outside the supported datetime range.
        """
        return self.get_label_end(anchor) <= cutoff

    def get_length_in_days(self) -> int:
        """Return the interval's length end - start in whole days.

        A month counts as its mean length, 30.44 days, rounded; at least 1.
        """
        if self.unit == "months":
            return max(1, round((self.end - self.start) * _DAYS_PER_MONTH))
        return self.end - self.start

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
            params: Statement parameters; horizon_days (horizon_months for a
                month window) is always added, and start_days (start_months)
                only for a delayed window, so an immediate window renders
                exactly as it did before delayed windows existed.
            dialect: SQL syntax for exact 24-hour and calendar-month additions;
                defaults to PostgreSQL.

        Returns:
            Two comparisons, the second on its own line indented one space
            and led by AND, so the text drops straight into the LEFT JOIN
            block that label SQL builds.
        """
        dialect = dialect or SQLDialect()
        add = dialect.add_months if self.unit == "months" else dialect.add_days
        params[f"horizon_{self.unit}"] = self.end
        lower = anchor
        if self.start:
            params[f"start_{self.unit}"] = self.start
            lower = add(anchor, f"%(start_{self.unit})s")
        upper = add(anchor, f"%(horizon_{self.unit})s")
        return f"{event_time} > {lower}\n AND {event_time} <= {upper}"


def _get_moment_plus_months(moment: datetime, months: int) -> datetime:
    """Return moment moved by a whole number of calendar months.

    The time of day and time zone are kept. A day past the end of the target
    month becomes its last day (January 31 plus one month is February 28 or
    29), which is how PostgreSQL, Snowflake and Databricks add months.

    Raises:
        OverflowError: the result falls outside years 1 to 9999.
    """
    month_index = moment.year * 12 + moment.month - 1 + months
    year, month = divmod(month_index, 12)
    month += 1
    if not 1 <= year <= 9999:
        raise OverflowError("month arithmetic left the supported year range")
    day = min(moment.day, calendar.monthrange(year, month)[1])
    return moment.replace(year=year, month=month, day=day)


def target_window(node: Grammar.AggregationContext) -> TargetWindow:
    """Read an aggregate's window as whole days or calendar months.

    Scope has already established the form: two integer offsets of at most
    nine digits, the first unsigned, the unit days or months. The constructor
    checks the values.

    Raises:
        ValueError: end is not positive, or the interval is empty or reversed.
    """
    start_text, end_text, unit_text = window_text(node)
    return TargetWindow(int(start_text), int(end_text), unit_text.lower())
