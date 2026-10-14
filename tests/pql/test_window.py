"""The label interval's owner refuses what it cannot label and renders one shape."""

from datetime import UTC, datetime

import pytest

from nori_rel.pql.window import TargetWindow


@pytest.mark.parametrize(
    "start,end,message",
    [
        (-30, 60, "cannot start before the anchor"),
        (0, 0, "must be positive"),
        (30, 30, "end after it starts"),
        (60, 30, "end after it starts"),
    ],
)
def test_invalid_intervals_cannot_be_built(start, end, message):
    """The check lives on the class, so a cached window meets the same rule."""
    with pytest.raises(ValueError, match=message):
        TargetWindow(start, end)


def test_latest_anchor_uses_the_end_and_reports_overflow():
    cutoff = datetime(2026, 3, 3, tzinfo=UTC)
    assert TargetWindow(30, 60).latest_anchor(cutoff) == datetime(
        2026, 1, 2, tzinfo=UTC
    )
    with pytest.raises(ValueError, match="supported datetime range"):
        TargetWindow(0, 999_999_999).latest_anchor(cutoff)


def test_predicate_binds_start_only_when_delayed():
    params = {}
    text = TargetWindow(0, 30).predicate("t", "a", params)
    assert text == "t > a\n AND t <= a + %(horizon_days)s * INTERVAL '24 hours'"
    assert params == {"horizon_days": 30}
    params = {}
    text = TargetWindow(30, 60).predicate("t", "a", params)
    assert text == (
        "t > a + %(start_days)s * INTERVAL '24 hours'"
        "\n AND t <= a + %(horizon_days)s * INTERVAL '24 hours'"
    )
    assert params == {"horizon_days": 60, "start_days": 30}
