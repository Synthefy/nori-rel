"""Metrics and a deterministic stand-in scorer for candidate feature programs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from nori_rel.contracts import AGENT_SEARCH_METRICS, AgentSearchMetric

from .contracts import ScoreReport, TaskKind

_EPSILON = 1e-9
#: The regression metric used when none is chosen.
DEFAULT_REGRESSION_METRIC: AgentSearchMetric = "mae"


def metric_name(
    task: TaskKind, metric: AgentSearchMetric = DEFAULT_REGRESSION_METRIC
) -> str:
    """Return the higher-is-better metric each task is scored with.

    Binary tasks score ``"roc_auc"``. Regression scores the negative of
    ``metric``, e.g. ``"neg_mae"`` or ``"neg_wape"``.
    """
    if task == "binary":
        return "roc_auc"
    require_agent_search_metric(metric)
    return f"neg_{metric}"


def task_score(
    task: TaskKind,
    target: np.ndarray,
    prediction: np.ndarray,
    metric: AgentSearchMetric = DEFAULT_REGRESSION_METRIC,
) -> float:
    """Return a higher-is-better score of ``prediction`` against ``target``.

    Binary tasks score rank-based ROC-AUC; regression scores the negative of
    ``metric`` (see ``get_regression_error``).
    """
    if task == "binary":
        return roc_auc(target, prediction)
    return -get_regression_error(metric, target, prediction)


def get_regression_error(
    metric: AgentSearchMetric, target: np.ndarray, prediction: np.ndarray
) -> float:
    """Return the error of ``prediction`` against ``target`` under ``metric``.

    Lower is better. Relative and normalized errors are fractions, not
    percentages. Raises ``ValueError`` for an unknown metric, or when a
    relative or normalized error has nothing to divide by (see each metric).
    """
    require_agent_search_metric(metric)
    actual = np.asarray(target, dtype=float)
    return _REGRESSION_ERRORS[metric](actual, np.asarray(prediction, dtype=float))


def require_agent_search_metric(metric: str) -> None:
    """Raise ``ValueError`` unless ``metric`` is one of ``AGENT_SEARCH_METRICS``."""
    if metric not in AGENT_SEARCH_METRICS:
        raise ValueError(
            f"metric must be one of {', '.join(AGENT_SEARCH_METRICS)}, not {metric!r}"
        )


def _get_mean_absolute_error(actual: np.ndarray, prediction: np.ndarray) -> float:
    """Return the mean absolute error."""
    return float(np.mean(np.abs(prediction - actual)))


def _get_mean_squared_error(actual: np.ndarray, prediction: np.ndarray) -> float:
    """Return the mean squared error."""
    return float(np.mean((prediction - actual) ** 2))


def _get_root_mean_squared_error(actual: np.ndarray, prediction: np.ndarray) -> float:
    """Return the root mean squared error."""
    return float(np.sqrt(_get_mean_squared_error(actual, prediction)))


def _get_actual_range(actual: np.ndarray, metric: str) -> float:
    """Return max minus min of ``actual``; raise ``ValueError`` when it is zero."""
    span = float(np.max(actual) - np.min(actual))
    if span == 0.0:
        raise ValueError(f"{metric.upper()} needs targets that are not all equal")
    return span


def _get_range_normalized_mae(actual: np.ndarray, prediction: np.ndarray) -> float:
    """Return the mean absolute error divided by the range of the targets."""
    return _get_mean_absolute_error(actual, prediction) / _get_actual_range(
        actual, "nmae"
    )


def _get_range_normalized_rmse(actual: np.ndarray, prediction: np.ndarray) -> float:
    """Return the root mean squared error divided by the range of the targets."""
    return _get_root_mean_squared_error(actual, prediction) / _get_actual_range(
        actual, "nrmse"
    )


def _get_median_absolute_error(actual: np.ndarray, prediction: np.ndarray) -> float:
    """Return the median absolute error."""
    return float(np.median(np.abs(prediction - actual)))


def _get_weighted_absolute_percentage_error(
    actual: np.ndarray, prediction: np.ndarray
) -> float:
    """Return the total absolute error divided by the total absolute target."""
    total = float(np.abs(actual).sum())
    if total == 0.0:
        raise ValueError("WAPE needs a holdout whose targets are not all zero")
    return float(np.abs(prediction - actual).sum() / total)


def _get_mean_absolute_percentage_error(
    actual: np.ndarray, prediction: np.ndarray
) -> float:
    """Return the mean of absolute error over absolute target.

    Rows whose target is zero are skipped: their percentage error is undefined.
    """
    nonzero = actual != 0
    if not nonzero.any():
        raise ValueError("MAPE needs a holdout row whose target is not zero")
    return float(
        np.mean(np.abs(prediction[nonzero] - actual[nonzero]) / np.abs(actual[nonzero]))
    )


def _get_symmetric_mean_absolute_percentage_error(
    actual: np.ndarray, prediction: np.ndarray
) -> float:
    """Return the mean of 2|error| / (|target| + |prediction|), between 0 and 2.

    Rows where both the target and the prediction are zero have no error and
    count as 0.
    """
    scale = np.abs(actual) + np.abs(prediction)
    ratios = np.divide(
        2 * np.abs(prediction - actual),
        scale,
        out=np.zeros_like(scale),
        where=scale > 0,
    )
    return float(np.mean(ratios))


#: The function computing each metric from (actual, prediction).
_REGRESSION_ERRORS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "mae": _get_mean_absolute_error,
    "mse": _get_mean_squared_error,
    "rmse": _get_root_mean_squared_error,
    "nmae": _get_range_normalized_mae,
    "nrmse": _get_range_normalized_rmse,
    "medae": _get_median_absolute_error,
    "wape": _get_weighted_absolute_percentage_error,
    "mape": _get_mean_absolute_percentage_error,
    "smape": _get_symmetric_mean_absolute_percentage_error,
}


def roc_auc(target: np.ndarray, score: np.ndarray) -> float:
    """Return the rank-based area under the ROC curve, 0.5 when degenerate."""
    positive = np.asarray(target) > 0.5
    n_positive = int(positive.sum())
    n_negative = len(positive) - n_positive
    if n_positive == 0 or n_negative == 0:
        return 0.5
    ranks = pd.Series(score).rank(method="average").to_numpy()
    expected = n_positive * (n_positive + 1) / 2.0
    return float((ranks[positive].sum() - expected) / (n_positive * n_negative))


@dataclass(frozen=True)
class RidgeScorer:
    """Standardized ridge on numeric columns; a stand-in for Nori in tests.

    Carries the same contract, including a per-seed context subsample so the
    report has a comparable noise estimate.
    """

    task: TaskKind = "regression"
    max_context_rows: int = 2_000
    seeds: tuple[int, ...] = (0, 1)
    ridge: float = 1.0

    def __call__(
        self,
        context_features: pd.DataFrame,
        context_target: pd.Series,
        holdout_features: pd.DataFrame,
        holdout_target: pd.Series,
        *,
        categorical_columns: Sequence[str],
        text_columns: Sequence[str] = (),
    ) -> ScoreReport:
        """Fit per seed, predict the holdout, and summarize the task metric.

        Prose is accepted and ignored. Ridge on numeric columns has nothing to
        do with an embedding, and a stand-in that refused the argument would
        make the two scorers answer different contracts.
        """
        del text_columns
        columns = context_features.select_dtypes(include=[np.number]).columns
        holdout = np.nan_to_num(holdout_features[columns].to_numpy(dtype=float))
        target = holdout_target.to_numpy(dtype=float)
        scores: list[float] = []
        first: np.ndarray | None = None
        for seed in self.seeds:
            index = _sample(len(context_features), self.max_context_rows, seed)
            train = np.nan_to_num(
                context_features[columns].iloc[index].to_numpy(dtype=float)
            )
            labels = context_target.iloc[index].to_numpy(dtype=float)
            prediction = _ridge(train, labels, holdout, self.ridge)
            if self.task == "binary":
                prediction = np.clip(prediction, 0.0, 1.0)
            scores.append(task_score(self.task, target, prediction))
            if first is None:
                first = prediction
        if first is None:
            raise ValueError("a scorer needs at least one seed")
        return ScoreReport(
            metric=metric_name(self.task),
            score=float(np.mean(scores)),
            noise=float(np.std(scores)),
            predictions=first,
        )


def _sample(count: int, rows: int, seed: int) -> np.ndarray:
    if count <= rows:
        return np.arange(count)
    return np.sort(np.random.default_rng(seed).choice(count, rows, replace=False))


def _ridge(
    train: np.ndarray, labels: np.ndarray, holdout: np.ndarray, ridge: float
) -> np.ndarray:
    intercept = float(labels.mean()) if len(labels) else 0.0
    if train.shape[1] == 0 or len(train) == 0:
        return np.full(len(holdout), intercept)
    center = train.mean(axis=0)
    scale = train.std(axis=0) + _EPSILON
    train = (train - center) / scale
    holdout = (holdout - center) / scale
    penalty = ridge * len(train) * np.eye(train.shape[1])
    weights = np.linalg.solve(train.T @ train + penalty, train.T @ (labels - intercept))
    return holdout @ weights + intercept


__all__ = [
    "DEFAULT_REGRESSION_METRIC",
    "RidgeScorer",
    "get_regression_error",
    "metric_name",
    "require_agent_search_metric",
    "roc_auc",
    "task_score",
]
