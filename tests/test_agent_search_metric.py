"""How an agent search scores candidate programs: MAE, WAPE or MAPE."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nori_rel import AgenticSearch
from nori_rel._feature_search import ClientProgramScorer
from nori_rel._pql_runtime import predict_pql
from nori_rel._runtime import LocalRelationalRuntime
from nori_rel.contracts import PredictionTask
from nori_rel.feature.agentic.run_reporting import displayed_metric_score
from nori_rel.feature.agentic.scoring import (
    get_regression_error,
    metric_name,
    task_score,
)
from nori_rel.pql.window import TargetWindow
from nori_rel.run.messages import render

# Absolute errors 2, 3 and 6; the second row's actual is zero.
TARGET = np.array([10.0, 0.0, 30.0])
PREDICTION = np.array([12.0, 3.0, 24.0])
REGRESSION_QUERY = "PREDICT SUM(orders.amount,0,30,days) FOR EACH users.user_id"
BINARY_QUERY = "PREDICT COUNT(orders.*,0,30,days)=0 FOR EACH users.user_id"
ANCHOR = datetime(2026, 1, 1, tzinfo=UTC)


class _Agent:
    def run(self, session, task_hint):
        raise AssertionError("not searched in these tests")


class _Predictor:
    """Answers every Nori call with fixed values."""

    def __init__(self, values):
        self.values = values

    def model_for(self, task):
        return "nori-30m"

    def predict_arrays(self, train, target, query, **options):
        return list(self.values[: len(query)]), None


def _predict_pql(features, query=REGRESSION_QUERY, **options):
    """Run ``predict_pql`` far enough to validate its request; it reads no data."""
    return predict_pql(
        SimpleNamespace(features=features),
        source=None,
        query=query,
        anchor_times=[ANCHOR],
        anchor_time=datetime(2026, 2, 1, tzinfo=UTC),
        observed_through=datetime(2026, 2, 1, tzinfo=UTC),
        entity_created_at="created_at",
        **options,
    )


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        ("mae", 11 / 3),
        ("wape", 11 / 40),
        ("mape", (2 / 10 + 6 / 30) / 2),
    ],
)
def test_regression_errors_follow_their_definitions(metric, expected) -> None:
    assert get_regression_error(metric, TARGET, PREDICTION) == pytest.approx(expected)


def test_wape_ranks_programs_as_mae_does_on_one_holdout() -> None:
    better, worse = np.array([11.0, 1.0, 29.0]), np.array([5.0, 0.0, 40.0])
    for metric in ("mae", "wape"):
        assert get_regression_error(metric, TARGET, better) < get_regression_error(
            metric, TARGET, worse
        )


@pytest.mark.parametrize("metric", ["wape", "mape"])
def test_relative_errors_need_a_nonzero_target(metric) -> None:
    with pytest.raises(ValueError, match="not"):
        get_regression_error(metric, np.zeros(3), PREDICTION)
    assert get_regression_error("mae", np.zeros(3), PREDICTION) == pytest.approx(13)


def test_an_unknown_metric_is_refused() -> None:
    with pytest.raises(ValueError, match="mae, wape, mape"):
        get_regression_error("rmse", TARGET, PREDICTION)


def test_scores_are_higher_is_better_and_binary_keeps_roc_auc() -> None:
    assert metric_name("regression", "wape") == "neg_wape"
    assert metric_name("binary", "wape") == "roc_auc"
    assert task_score("regression", TARGET, PREDICTION, "wape") == pytest.approx(
        -11 / 40
    )


def test_the_client_scorer_reports_the_chosen_metric() -> None:
    scorer = ClientProgramScorer(_Predictor([12.0, 3.0, 24.0]), metric="wape")
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    report = scorer(
        frame,
        pd.Series([1.0, 2.0, 3.0]),
        frame,
        pd.Series(TARGET),
        categorical_columns=[],
    )
    assert report.metric == "neg_wape"
    assert report.score == pytest.approx(-11 / 40)


def test_the_runtime_passes_the_metric_to_the_agent_search() -> None:
    runtime = LocalRelationalRuntime(
        api_key="test-only",
        model="nori-30m",
        classification_model=None,
        timeout=10,
        max_retries=0,
        features=AgenticSearch(agent=_Agent()),
    )
    context = pd.DataFrame({"anchor_time": [ANCHOR] * 2, "label": 1.0})
    labels = pd.Series([1.0, 2.0])
    window = TargetWindow(0, 30)
    chosen = runtime.feature_backend(
        window, PredictionTask.REGRESSION, context, labels, metric="mape"
    )
    default = runtime.feature_backend(
        window, PredictionTask.REGRESSION, context, labels
    )
    assert chosen.scorer.metric == "mape"
    assert default.scorer.metric == "mae"


def test_wape_and_mape_events_read_as_percentages() -> None:
    assert displayed_metric_score("neg_wape", -0.112) == ("wape", 0.112)
    assert (
        render(
            "baseline_scored", {"metric": "wape", "score": 0.1123, "feature_count": 48}
        )
        == "Baseline: WAPE 11.23% (48 features)."
    )
    assert (
        render(
            "trial_scored",
            {"trial": 1, "metric": "mape", "score": 0.2, "best_score": 0.25},
        )
        == "Trial 1: MAPE 20.00% (best 25.00%)."
    )


@pytest.mark.parametrize(
    ("features", "options", "match"),
    [
        ("fastdfs", {"metric": "wape"}, "agent search scores"),
        ("noridfs", {"metric": "wape"}, "agent search scores"),
        (None, {"metric": "rmse"}, "mae, wape, mape"),
        (None, {"metric": "wape", "query": BINARY_QUERY}, "regression targets"),
        (
            None,
            {
                "metric": "wape",
                "feature_program": SimpleNamespace(validate_query=lambda *_: None),
            },
            "agent search scores",
        ),
        (None, {"metric": "wape", "pipeline": object()}, "agent search scores"),
    ],
)
def test_predict_refuses_a_metric_nothing_would_use(features, options, match) -> None:
    features = AgenticSearch(agent=_Agent()) if features is None else features
    with pytest.raises(ValueError, match=match):
        _predict_pql(features, **options)
