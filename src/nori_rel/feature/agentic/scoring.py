"""Metrics and a deterministic stand-in scorer for candidate feature programs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .contracts import ScoreReport, TaskKind

_EPSILON = 1e-9


def metric_name(task: TaskKind) -> str:
    """Return the higher-is-better metric each task is scored with."""
    return "roc_auc" if task == "binary" else "neg_mae"


def task_score(task: TaskKind, target: np.ndarray, prediction: np.ndarray) -> float:
    """Score predictions with negative MAE or rank-based ROC-AUC."""
    if task == "binary":
        return roc_auc(target, prediction)
    return -float(np.mean(np.abs(prediction - target)))


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


__all__ = ["RidgeScorer", "metric_name", "roc_auc", "task_score"]
