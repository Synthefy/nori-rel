import pytest

from nori_rel.run import EventKind, Stage
from nori_rel.run.messages import render

EXAMPLES = {
    "status_changed": (
        {"from": "queued", "to": "running", "strategy": "fastdfs"},
        "Run started: fastdfs search.",
    ),
    "stage_started": ({"stage": "plan_search"}, "Started planning features."),
    "stage_finished": (
        {"stage": "materialize", "seconds": 1.25},
        "Finished reading the database snapshot in 1.2 s.",
    ),
    "command": ({"command": "pause", "status": "applied"}, "Pause applied."),
    "limit_reached": (
        {"limit": "max_runtime_seconds"},
        "Reached max_runtime_seconds; cancelling the run.",
    ),
    "error": ({"message": "Database unavailable."}, "Database unavailable."),
}


def test_a_resumed_run_says_so() -> None:
    assert render("status_changed", {"from": "paused", "to": "running"}) == (
        "Run resumed."
    )


def test_error_text_is_redacted_in_the_message() -> None:
    assert render("error", {"message": "login failed: password=hunter2"}) == (
        "login failed: password=***"
    )


@pytest.mark.parametrize("kind", sorted(EXAMPLES))
def test_templates_render(kind) -> None:
    data, expected = EXAMPLES[kind]

    assert render(kind, data) == expected


def test_enum_members_and_plain_strings_render_alike() -> None:
    data = {"stage": Stage.PREDICT_BATCH}

    assert render(EventKind.STAGE_STARTED, data) == render("stage_started", data)
    assert render("stage_started", {"stage": "predict_batch"}) == (
        "Started predicting."
    )


def test_unknown_kind_or_missing_field_fails() -> None:
    with pytest.raises(ValueError):
        render("unknown", {})
    with pytest.raises(KeyError):
        render("stage_started", {})
