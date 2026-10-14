from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from nori_rel.run import (
    EVENT_TYPE_BY_KIND,
    FINAL_STATUSES,
    CommandAcceptedResponse,
    CommandRejectedResponse,
    EventKind,
    RunEvent,
)

NOW = datetime(2026, 10, 6, 12, 0, tzinfo=UTC)


def _event(**overrides):
    fields = dict(
        run_id="run_1",
        event_id=1,
        timestamp=NOW,
        type="progress",
        message="Started planning features.",
        data={"kind": "stage_started", "stage": "plan_search"},
    )
    return RunEvent(**(fields | overrides))


def test_event_round_trips_through_json() -> None:
    event = _event()

    assert RunEvent.model_validate_json(event.model_dump_json()) == event


def test_nested_data_round_trips_through_json() -> None:
    event = _event(
        data={"kind": "stage_started", "usage": {"calls": 3, "tokens": [1, 2.5, None]}}
    )

    assert RunEvent.model_validate_json(event.model_dump_json()) == event


def test_events_can_be_collected_in_a_set() -> None:
    first, again, second = _event(), _event(), _event(event_id=2)

    assert {first, again, second} == {first, second}


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"data": {}}, "data.kind"),
        ({"data": {"kind": 3}}, "data.kind"),
        ({"data": {"kind": ["error"]}}, "data.kind"),
        ({"data": {"kind": "unknown"}}, "data.kind"),
        ({"type": "status"}, "stage_started events have type 'progress'"),
        ({"event_id": 0}, "greater than or equal to 1"),
        ({"timestamp": datetime(2026, 10, 6)}, "timezone"),
        ({"message": ""}, "at least 1 character"),
        ({"extra": 1}, "Extra inputs"),
    ],
)
def test_invalid_events_are_rejected(overrides, match) -> None:
    with pytest.raises(ValidationError, match=match):
        _event(**overrides)


def test_events_are_immutable() -> None:
    with pytest.raises(ValidationError, match="frozen"):
        _event().message = "changed"


def test_every_kind_maps_to_an_event_type() -> None:
    assert set(EVENT_TYPE_BY_KIND) == set(EventKind)
    assert set(EVENT_TYPE_BY_KIND.values()) == {
        "status",
        "progress",
        "action",
        "warning",
        "error",
    }
    assert {"completed", "failed", "cancelled"} == FINAL_STATUSES


def test_command_receipt_defaults_to_accepted() -> None:
    receipt = CommandAcceptedResponse(run_id="run_1", command_id="cmd_1")

    assert receipt.status == "accepted"


def test_a_rejected_command_receipt_names_why_and_is_immutable() -> None:
    receipt = CommandRejectedResponse(run_id="run_1", command_id="cmd_1")

    assert (receipt.status, receipt.reason) == ("rejected", "run_finished")
    assert receipt.model_dump() == {
        "run_id": "run_1",
        "command_id": "cmd_1",
        "status": "rejected",
        "reason": "run_finished",
    }
    with pytest.raises(ValidationError, match="frozen"):
        receipt.reason = "no_op"
    with pytest.raises(ValidationError):
        CommandRejectedResponse(run_id="run_1", command_id="cmd_1", reason="no_op")
