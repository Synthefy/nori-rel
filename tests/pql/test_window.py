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


def test_an_unsupported_unit_cannot_be_built():
    with pytest.raises(ValueError, match="unit must be one of days, months"):
        TargetWindow(0, 1, "weeks")


def test_month_windows_end_on_the_calendar_and_clamp_short_months():
    window = TargetWindow(0, 1, "months")
    assert window.get_label_end(datetime(2026, 1, 31, 12, tzinfo=UTC)) == datetime(
        2026, 2, 28, 12, tzinfo=UTC
    )
    assert window.get_label_end(datetime(2026, 12, 15, tzinfo=UTC)) == datetime(
        2027, 1, 15, tzinfo=UTC
    )
    assert window.latest_anchor(datetime(2026, 3, 1, tzinfo=UTC)) == datetime(
        2026, 2, 1, tzinfo=UTC
    )


def test_a_month_window_keeps_the_february_anchor_that_30_days_drops():
    february, march = datetime(2026, 2, 1, tzinfo=UTC), datetime(2026, 3, 1, tzinfo=UTC)
    assert TargetWindow(0, 1, "months").is_observed(february, march)
    assert not TargetWindow(0, 30).is_observed(february, march)
    assert TargetWindow(0, 28).is_observed(february, march)


def test_month_window_overflow_is_a_value_error():
    window = TargetWindow(0, 999_999_999, "months")
    with pytest.raises(ValueError, match="supported datetime range"):
        window.get_label_end(datetime(2026, 1, 1, tzinfo=UTC))
    with pytest.raises(ValueError, match="supported datetime range"):
        window.latest_anchor(datetime(2026, 1, 1, tzinfo=UTC))


def test_window_lengths_in_days_size_noridfs():
    assert TargetWindow(30, 60).get_length_in_days() == 30
    assert TargetWindow(0, 1, "months").get_length_in_days() == 30
    assert TargetWindow(0, 3, "months").get_length_in_days() == 91


def test_month_predicate_binds_month_offsets():
    params = {}
    text = TargetWindow(0, 1, "months").predicate("t", "a", params)
    assert text == "t > a\n AND t <= a + %(horizon_months)s * INTERVAL '1 month'"
    assert params == {"horizon_months": 1}
