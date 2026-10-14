import pytest

from nori_rel.run import EventKind, FallbackReason, Stage, TrialOutcome
from nori_rel.run.messages import (
    FALLBACK_MESSAGES,
    STAGE_LABELS,
    TRIAL_OUTCOME_MESSAGES,
    render,
)

EXAMPLES = {
    "status_changed": (
        {"from": "queued", "to": "running", "strategy": "agent", "max_trials": 24},
        "Run started: agent search, up to 24 trials.",
    ),
    "stage_started": ({"stage": "plan_search"}, "Started planning features."),
    "stage_finished": (
        {"stage": "materialize", "seconds": 1.25},
        "Finished reading the database snapshot in 1.2 s.",
    ),
    "baseline_scored": (
        {"metric": "mae", "score": 3.8, "feature_count": 48},
        "Baseline: MAE 3.8 (48 features).",
    ),
    "trial_scored": (
        {
            "trial": 3,
            "max_trials": 24,
            "metric": "mae",
            "score": 3.91,
            "best_score": 3.74,
            "baseline_score": 3.8,
        },
        "Trial 3/24: MAE 3.91 (best 3.74, baseline 3.8).",
    ),
    "spend_updated": (
        {"spend_usd": 0.42, "llm_calls": 8},
        "Agent spend: $0.42 (8 LLM calls).",
    ),
    "command": (
        {"command": "steer", "status": "applied", "text": "Use order history."},
        'Steer applied: "Use order history."',
    ),
    "fallback_used": (
        {"reason": "no_program_confirmed", "to_strategy": "dfs"},
        "No program beat the baseline; using dfs.",
    ),
    "search_stopped": (
        {"trials": 3},
        "Search stopped after 3 trials; finishing with the best result so far.",
    ),
    "limit_reached": (
        {"limit": "max_spend_usd"},
        "Reached max_spend_usd; finishing with the best result so far.",
    ),
    "error": (
        {"message": "Permission denied when reading sales.orders."},
        "Permission denied when reading sales.orders.",
    ),
}


def test_large_scores_are_shown_in_full() -> None:
    data = {"trial": 1, "metric": "rmse", "score": 12345.67, "best_score": 98765.4}

    assert render("trial_scored", data) == "Trial 1: RMSE 12,345.7 (best 98,765.4)."


def test_a_score_that_is_not_a_number_fails() -> None:
    with pytest.raises(ValueError, match="score must be a number"):
        render("trial_scored", {"trial": 1, "metric": "mae", "score": None})


def test_a_resumed_run_says_so() -> None:
    assert render("status_changed", {"from": "paused", "to": "running"}) == (
        "Run resumed."
    )


def test_error_text_is_redacted_in_the_message() -> None:
    assert render("error", {"message": "login failed: password=hunter2"}) == (
        "login failed: password=***"
    )


def test_every_name_has_a_template_or_label() -> None:
    assert set(EXAMPLES) == set(EventKind)
    assert set(STAGE_LABELS) == set(Stage)
    assert set(FALLBACK_MESSAGES) == set(FallbackReason)
    assert set(TRIAL_OUTCOME_MESSAGES) == set(TrialOutcome)


@pytest.mark.parametrize("kind", sorted(EXAMPLES))
def test_templates_render(kind) -> None:
    data, expected = EXAMPLES[kind]

    assert render(kind, data) == expected


def test_optional_fields_and_variants() -> None:
    assert render("status_changed", {"from": "running", "to": "paused"}) == (
        "Run paused."
    )
    assert render("trial_scored", {"trial": 1, "metric": "mae", "score": 4.0}) == (
        "Trial 1: MAE 4."
    )
    assert render("command", {"command": "pause", "status": "applied"}) == (
        "Pause applied."
    )
    assert render(
        "command", {"command": "steer", "status": "rejected", "reason": "no_op"}
    ) == ("Steer rejected (no_op).")


def test_long_steering_text_is_shortened() -> None:
    message = render(
        "command", {"command": "steer", "status": "applied", "text": "x" * 600}
    )

    assert message == f'Steer applied: "{"x" * 499}…"'
    assert (
        render("command", {"command": "steer", "status": "applied", "text": "x" * 500})
        == f'Steer applied: "{"x" * 500}"'
    )


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
        render("baseline_scored", {"metric": "mae"})


TRIAL = {"trial": 1, "max_trials": 6, "metric": "mae", "score": 365.6}


@pytest.mark.parametrize(
    "outcome,expected",
    [
        ("new_best", "Trial 1/6: MAE 365.6 (baseline 386.2); new best."),
        (
            "best_below_baseline",
            "Trial 1/6: MAE 365.6 (baseline 386.2); "
            "best so far, but below the baseline.",
        ),
        (
            "not_significant_gain",
            "Trial 1/6: MAE 365.6 (baseline 386.2); not a significant gain.",
        ),
        (
            "failed_point_in_time_check",
            "Trial 1/6: MAE 365.6 (baseline 386.2); "
            "rejected, failed the point-in-time check.",
        ),
    ],
)
def test_a_trial_message_says_what_the_search_decided(outcome, expected) -> None:
    data = TRIAL | {"baseline_score": 386.2, "outcome": outcome}

    assert render("trial_scored", data) == expected


def test_a_trial_without_a_known_outcome_renders_without_one() -> None:
    plain = "Trial 1/6: MAE 365.6."

    assert render("trial_scored", TRIAL) == plain
    assert render("trial_scored", TRIAL | {"outcome": None}) == plain
    assert render("trial_scored", TRIAL | {"outcome": "from_a_newer_server"}) == plain
    assert render("trial_scored", TRIAL | {"outcome": ["new_best"]}) == plain
