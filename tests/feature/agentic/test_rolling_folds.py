"""Rolling-cutoff folds score each window with only the anchors before it."""

from __future__ import annotations

import pandas as pd
import pytest

from nori_rel.feature import CUTOFF_COLUMN
from nori_rel.feature.agentic import (
    AgenticFeatureConfig,
    PlanningSession,
    RidgeScorer,
    ScoreReport,
)
from nori_rel.feature.dfs.noridfs import AnchorDefinition, DatabaseView


class _RecordingScorer:
    """Scores a window by its context size and records which rows it saw."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[int], list[int]]] = []

    def __call__(
        self,
        context_features,
        context_target,
        holdout_features,
        holdout_target,
        *,
        categorical_columns,
        text_columns=(),
    ) -> ScoreReport:
        context = list(context_features.index)
        holdout = list(holdout_features.index)
        self.calls.append((context, holdout))
        return ScoreReport(
            metric="neg_mae", score=float(len(context)), noise=0.5, predictions=None
        )


def _session(snapshot, train, config, scorer, **kwargs) -> PlanningSession:
    view = DatabaseView.from_parts(
        tables=snapshot.tables,
        primary_keys=snapshot.primary_keys,
        foreign_keys=snapshot.foreign_keys,
        time_columns=snapshot.time_columns,
    )
    return PlanningSession(
        database=view,
        anchors=AnchorDefinition.create(
            entity_table="accounts",
            entity_column="account_id",
            cutoff_column=CUTOFF_COLUMN,
        ),
        config=config,
        train_anchors=train,
        train_target=pd.Series([float(i) for i in range(len(train))]),
        scorer=scorer,
        **kwargs,
    )


def test_each_window_is_scored_on_its_strict_past(snapshot, anchors) -> None:
    config = AgenticFeatureConfig(
        holdout_fraction=0.2,
        confirmation_fraction=0.2,
        leak_check_anchors=0,
        n_folds=2,
        extend_baseline=True,
    )
    train = anchors(10, start="2020-01-03")
    scorer = _RecordingScorer()
    session = _session(
        snapshot,
        train,
        config,
        scorer,
        baseline_features=pd.DataFrame({"base": range(10)}),
    )
    try:
        # Ten anchors: 2 for confirmation, two windows of 2, contexts of 4 and 6.
        older, newer = scorer.calls
        assert [len(older[0]), len(newer[0])] == [4, 6]
        assert len(older[1]) == len(newer[1]) == 2
        for context, holdout in scorer.calls:
            newest_context = train[CUTOFF_COLUMN].iloc[context].max()
            assert (train[CUTOFF_COLUMN].iloc[holdout] >= newest_context).all()
        # The older window is the tail of the newer window's context.
        assert older[1] == newer[0][-2:]
        assert session.baseline.score == pytest.approx((4 + 6) / 2)
        assert session.baseline.noise == pytest.approx(0.5)
    finally:
        session.close()


def test_one_fold_is_the_single_holdout(snapshot, anchors) -> None:
    config = AgenticFeatureConfig(
        holdout_fraction=0.3, confirmation_fraction=0.2, leak_check_anchors=0
    )
    session = _session(
        snapshot,
        anchors(10, start="2020-01-03"),
        config,
        RidgeScorer(max_context_rows=4),
    )
    try:
        assert len(session._folds) == 1
        context, holdout = session._folds[0]
        assert list(context) == list(session._context)
        assert list(holdout) == list(session._holdout)
    finally:
        session.close()


def _grouped_anchors(groups: list[int]) -> pd.DataFrame:
    """Anchors where cutoff group ``g`` holds ``groups[g]`` rows."""
    days = [day for day, size in enumerate(groups) for _ in range(size)]
    return pd.DataFrame(
        {
            "account_id": [1 + index % 3 for index in range(len(days))],
            CUTOFF_COLUMN: pd.to_datetime("2020-01-03", utc=True)
            + pd.to_timedelta(days, unit="D"),
        }
    )


def test_aligned_windows_never_split_a_cutoff(snapshot) -> None:
    # Six cutoff groups of 4 anchors: 24 anchors, 2 windows of >= 4 and >= 4
    # for confirmation, all at group boundaries.
    train = _grouped_anchors([4, 4, 4, 4, 4, 4])
    config = AgenticFeatureConfig(
        holdout_fraction=0.15,
        confirmation_fraction=0.15,
        leak_check_anchors=0,
        n_folds=2,
        align_to_cutoffs=True,
        extend_baseline=True,
    )
    scorer = _RecordingScorer()
    session = _session(
        snapshot,
        train,
        config,
        scorer,
        baseline_features=pd.DataFrame({"base": range(len(train))}),
    )
    try:
        cutoffs = train[CUTOFF_COLUMN]
        for context, holdout in session._folds:
            assert cutoffs.iloc[context].max() < cutoffs.iloc[holdout].min()
            assert cutoffs.iloc[holdout].nunique() == 1
        _, newest_window = session._folds[-1]
        confirmation = session._confirmation
        assert cutoffs.iloc[newest_window].max() < cutoffs.iloc[confirmation].min()
        assert [len(window) for _, window in session._folds] == [4, 4]
        assert len(confirmation) == 4
    finally:
        session.close()


def test_few_cutoffs_get_fewer_windows(snapshot) -> None:
    # Three cutoffs: one context group, one window, one confirmation group.
    train = _grouped_anchors([5, 5, 5])
    config = AgenticFeatureConfig(
        holdout_fraction=0.15,
        confirmation_fraction=0.15,
        leak_check_anchors=0,
        n_folds=3,
        align_to_cutoffs=True,
    )
    session = _session(snapshot, train, config, RidgeScorer(max_context_rows=4))
    try:
        assert len(session._folds) == 1
        context, window = session._folds[0]
        assert len(context) == len(window) == len(session._confirmation) == 5
    finally:
        session.close()


def test_two_cutoffs_cannot_be_aligned(snapshot) -> None:
    config = AgenticFeatureConfig(leak_check_anchors=0, align_to_cutoffs=True)
    with pytest.raises(ValueError, match="cutoff"):
        _session(
            snapshot,
            _grouped_anchors([5, 5]),
            config,
            RidgeScorer(max_context_rows=4),
        )


def test_three_context_cutoffs_require_five_total_cutoffs(snapshot) -> None:
    config = AgenticFeatureConfig(align_to_cutoffs=True, min_context_cutoffs=3)
    with pytest.raises(ValueError, match="3 context cutoffs"):
        _session(snapshot, _grouped_anchors([2] * 4), config, _RecordingScorer())
    train = _grouped_anchors([2] * 5)
    session = _session(snapshot, train, config, _RecordingScorer())
    try:
        cutoffs = train[CUTOFF_COLUMN]
        assert cutoffs.iloc[session._context].nunique() == 3
        assert cutoffs.iloc[session._holdout].nunique() == 1
        assert cutoffs.iloc[session._confirmation].nunique() == 1
    finally:
        session.close()


@pytest.mark.parametrize("groups", [[40, 1, 1, 1, 1], [1, 1, 1, 40, 1]])
def test_uneven_cutoff_groups_preserve_the_context_floor(snapshot, groups) -> None:
    train = _grouped_anchors(groups)
    config = AgenticFeatureConfig(align_to_cutoffs=True, min_context_cutoffs=3)
    session = _session(snapshot, train, config, _RecordingScorer())
    try:
        cutoffs = train[CUTOFF_COLUMN]
        context, holdout = session._folds[0]
        assert cutoffs.iloc[context].nunique() == 3
        assert cutoffs.iloc[context].max() < cutoffs.iloc[holdout].min()
        assert cutoffs.iloc[holdout].max() < cutoffs.iloc[session._confirmation].min()
        assert len(context) + len(holdout) + len(session._confirmation) == len(train)
    finally:
        session.close()


@pytest.mark.parametrize("cutoff_count, expected_folds", [(5, 1), (7, 3)])
def test_earliest_rolling_fold_preserves_the_context_floor(
    snapshot, cutoff_count, expected_folds
) -> None:
    train = _grouped_anchors([2] * cutoff_count)
    config = AgenticFeatureConfig(
        align_to_cutoffs=True,
        min_context_cutoffs=3,
        n_folds=3,
        holdout_fraction=0.1,
        confirmation_fraction=0.1,
    )
    session = _session(snapshot, train, config, _RecordingScorer())
    try:
        cutoffs = train[CUTOFF_COLUMN]
        assert len(session._folds) == expected_folds
        assert cutoffs.iloc[session._folds[0][0]].nunique() == 3
        for context, holdout in session._folds:
            assert cutoffs.iloc[context].nunique() >= 3
            assert cutoffs.iloc[context].max() < cutoffs.iloc[holdout].min()
    finally:
        session.close()


def test_unaligned_three_row_split_does_not_require_distinct_cutoffs(snapshot) -> None:
    config = AgenticFeatureConfig(align_to_cutoffs=False, min_context_cutoffs=3)
    session = _session(snapshot, _grouped_anchors([3]), config, _RecordingScorer())
    try:
        assert len(session._context) == len(session._holdout) == 1
        assert len(session._confirmation) == 1
    finally:
        session.close()


@pytest.mark.parametrize("min_context_cutoffs", [0, -1, 1.5, True])
def test_context_cutoff_floor_must_be_a_positive_integer(min_context_cutoffs) -> None:
    with pytest.raises(ValueError, match="min_context_cutoffs"):
        AgenticFeatureConfig(min_context_cutoffs=min_context_cutoffs)


def test_folds_must_leave_context() -> None:
    with pytest.raises(ValueError, match="n_folds"):
        AgenticFeatureConfig(n_folds=0)
    with pytest.raises(ValueError, match="leave context"):
        AgenticFeatureConfig(n_folds=4, holdout_fraction=0.25)


def test_the_split_report_lists_every_fold_and_the_confirmation(
    snapshot, anchors
) -> None:
    config = AgenticFeatureConfig(
        holdout_fraction=0.2,
        confirmation_fraction=0.2,
        leak_check_anchors=0,
        n_folds=2,
    )
    # Ten daily anchors from 3 Jan with labels 0..9 in cutoff order.
    session = _session(
        snapshot, anchors(10, start="2020-01-03"), config, _RecordingScorer()
    )
    try:
        report = session.get_split_report()
    finally:
        session.close()

    def day(number: int) -> pd.Timestamp:
        """Return midnight UTC of January ``number``, 2020."""
        return pd.Timestamp(2020, 1, number, tz="UTC")

    assert report.split_boundary == "row_count"
    assert (report.holdout_fraction, report.requested_folds) == (0.2, 2)
    # Rows 0-3 are context for rows 4-5; rows 0-5 for rows 6-7; 8-9 confirm.
    older, newer = report.folds
    assert (older.context_rows, older.validation_rows) == (4, 2)
    assert older.context_last_cutoff == day(6)
    assert (older.validation_first_cutoff, older.validation_last_cutoff) == (
        day(7),
        day(8),
    )
    assert older.validation_label_mean == pytest.approx(4.5)
    assert (newer.context_rows, newer.validation_label_mean) == (6, 6.5)
    assert report.confirmation_rows == 2
    assert (report.confirmation_first_cutoff, report.confirmation_last_cutoff) == (
        day(11),
        day(12),
    )
