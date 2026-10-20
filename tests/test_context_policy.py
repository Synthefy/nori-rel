"""ContextPolicy selection, report and correction, with hand-computed counts."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from nori_rel import ClassBalance, ContextPolicy, ContextPolicyReport
from nori_rel._context_selection import (
    get_negative_quotas_by_cutoff,
    get_probabilities_with_population_odds,
    get_probability_correction_factor,
    get_row_random_keys,
    get_rows_within_class_quotas,
    get_selected_context,
)

DAY_1, DAY_2, DAY_3, DAY_4 = (datetime(2026, 7, d, tzinfo=UTC) for d in (1, 2, 3, 4))


def make_labelled(rows):
    """Return a labelled frame numbered newest cutoff first, then entity key.

    rows are (entity_id, anchor_time, label); context_row_id follows the
    connector's order so tests can state newest_first expectations by ID.
    """
    ordered = sorted(rows, key=lambda row: (-row[1].timestamp(), row[0]))
    return pd.DataFrame(
        [
            (row_id, entity_id, anchor_time, label)
            for row_id, (entity_id, anchor_time, label) in enumerate(ordered, 1)
        ],
        columns=["context_row_id", "entity_id", "anchor_time", "label"],
    )


def select(labelled, policy, *, eligible=(), max_kept_rows=60_000):
    """Return (context, report) for a policy chosen by the caller."""
    return get_selected_context(
        labelled,
        policy,
        chosen_by="default" if policy is None else "caller",
        max_context_rows=5000 if policy is None else None,
        eligible_anchor_times=list(eligible),
        max_kept_rows=max_kept_rows,
    )


def get_report_totals(report):
    """Return (labelled rows, labelled positives, kept rows, kept positives).

    The report holds these per cutoff only; positives are None for a label
    that is not 0/1.
    """
    cutoffs = report.cutoffs
    has_positives = report.positive_label is not None
    return (
        sum(cutoff.labelled_rows for cutoff in cutoffs),
        sum(cutoff.labelled_positives for cutoff in cutoffs) if has_positives else None,
        sum(cutoff.kept_rows for cutoff in cutoffs),
        sum(cutoff.kept_positives for cutoff in cutoffs) if has_positives else None,
    )


def rare_label_rows():
    """10 positives (entities 0-9) and 90 negatives (10-99) on DAY_1 and DAY_2.

    Positives and negatives alternate between the two days by entity parity,
    so each day holds 5 positives and 45 negatives.
    """
    return make_labelled(
        [
            (entity, DAY_1 if entity % 2 else DAY_2, 1 if entity < 10 else 0)
            for entity in range(100)
        ]
    )


@pytest.mark.parametrize(
    "fields",
    [
        {"max_rows": 0},
        {"max_rows": 1.5},
        {"max_rows_per_entity": True},
        {"min_positives": -1},
        {"seed": "7"},
        {"time_sampling": "recent_weighted"},
        {"labelled_only": True},
    ],
)
def test_policy_rejects_invalid_fields(fields):
    with pytest.raises(ValidationError):
        ContextPolicy(**fields)


@pytest.mark.parametrize(
    "candidate_sql",
    [
        "",
        "   ",
        "DELETE FROM candidates",
        "SELECT 1; SELECT 2",
        "UPDATE users SET a = 1",
    ],
)
def test_candidate_sql_must_be_one_select(candidate_sql):
    with pytest.raises(ValidationError, match="candidate_sql"):
        ContextPolicy(candidate_sql=candidate_sql)


def test_candidate_sql_is_kept_stripped_and_defaults_to_none():
    assert ContextPolicy().candidate_sql is None
    sql = "  WITH kept AS (SELECT * FROM candidates) SELECT * FROM kept\n"
    assert ContextPolicy(candidate_sql=sql).candidate_sql == sql.strip()


@pytest.mark.parametrize(
    "fields",
    [
        {"positives": 0},
        {"positives": "some"},
        {"negatives_per_positive": 0},
        {"negatives_per_positive": float("inf")},
        {"positive_label": 2},
        {"positive_share": 0.5},
    ],
)
def test_class_balance_rejects_invalid_fields(fields):
    with pytest.raises(ValidationError):
        ClassBalance(**fields)


def test_no_policy_keeps_every_labelled_row_unchanged():
    labelled = rare_label_rows()
    context, report = select(labelled, None)
    pd.testing.assert_frame_equal(context, labelled)
    assert report.chosen_by == "default"
    assert report.policy is None
    assert report.max_context_rows == 5000
    assert get_report_totals(report) == (100, 10, 100, 10)
    assert report.probability_correction_factor is None


def test_newest_first_keeps_the_lowest_row_ids():
    labelled = rare_label_rows()
    context, report = select(labelled, ContextPolicy(max_rows=7, min_positives=None))
    # IDs 1..50 are DAY_2 (entities 0, 2, ..., 98 in key order); the first 7
    # are entities 0, 2, 4, 6, 8, 10, 12.
    assert context["context_row_id"].tolist() == [1, 2, 3, 4, 5, 6, 7]
    assert context["entity_id"].tolist() == [0, 2, 4, 6, 8, 10, 12]
    assert get_report_totals(report)[3] == 5


@pytest.mark.parametrize(
    ("balance", "max_rows", "expected_positives", "expected_negatives"),
    [
        # Every positive (10), 2 negatives per positive (20).
        (ClassBalance(positives="all", negatives_per_positive=2), None, 10, 20),
        # At most 4 positives, 1 negative each.
        (ClassBalance(positives=4), None, 4, 4),
        # More positives asked for than exist: all 10 are kept.
        (ClassBalance(positives=50), None, 10, 10),
        # 9 rows at 2:1 leave room for floor(9 / 3) = 3 positives, 6 negatives.
        (ClassBalance(negatives_per_positive=2), 9, 3, 6),
        # 10 * 0.29 = 2.8999999999999995 in floats; rounding down gives 2.
        (ClassBalance(negatives_per_positive=0.29), None, 10, 2),
        # 20 negatives per positive need 200; only 90 exist.
        (ClassBalance(negatives_per_positive=20), None, 10, 90),
    ],
)
def test_balanced_counts_are_exact(
    balance, max_rows, expected_positives, expected_negatives
):
    policy = ContextPolicy(
        balance=balance, max_rows=max_rows, time_sampling="random", min_positives=1
    )
    context, report = select(rare_label_rows(), policy)
    labels = context["label"].astype(float)
    assert int((labels == 1).sum()) == expected_positives
    assert int((labels == 0).sum()) == expected_negatives
    _, _, kept_rows, kept_positives = get_report_totals(report)
    assert kept_positives == expected_positives
    assert kept_rows == expected_positives + expected_negatives
    assert report.kept_label_counts == {
        "0": expected_negatives,
        "1": expected_positives,
    }


def test_balance_with_positive_label_zero_keeps_every_zero():
    labelled = make_labelled(
        [(entity, DAY_1, 0 if entity < 3 else 1) for entity in range(20)]
    )
    policy = ContextPolicy(
        balance=ClassBalance(positive_label=0, negatives_per_positive=2),
        min_positives=3,
    )
    context, report = select(labelled, policy)
    assert sorted(context["label"].tolist()) == [0, 0, 0, 1, 1, 1, 1, 1, 1]
    assert (report.positive_label, get_report_totals(report)[3]) == (0, 3)


def test_a_seed_reproduces_the_rows_and_another_seed_changes_them():
    labelled = rare_label_rows()

    def get_kept_ids(seed):
        """Return the kept row IDs of a random 30-row sample."""
        policy = ContextPolicy(
            time_sampling="random", max_rows=30, seed=seed, min_positives=None
        )
        return select(labelled, policy)[0]["context_row_id"].tolist()

    assert get_kept_ids(3) == get_kept_ids(3)
    assert get_kept_ids(3) != get_kept_ids(4)


def test_random_keys_depend_only_on_seed_entity_and_cutoff():
    anchors = [pd.Timestamp(DAY_1), pd.Timestamp(DAY_2)]
    alone = get_row_random_keys([7, 8], anchors, seed=1)
    with_more_rows = get_row_random_keys([5, 7, 8], [anchors[0], *anchors], seed=1)
    assert alone.tolist() == with_more_rows[1:].tolist()
    assert get_row_random_keys([7], anchors[:1], seed=2)[0] != alone[0]


def test_random_keys_are_pinned_so_a_seed_keeps_its_rows_across_releases():
    # A changed key changes which rows a seed keeps, including on pipeline
    # replay; this fails if that happens by accident.
    microsecond_anchors = pd.Series([DAY_1, DAY_2]).astype("datetime64[us, UTC]")
    keys = get_row_random_keys(pd.Series(["a", 7]), microsecond_anchors, seed=0)
    assert keys.tolist() == get_row_random_keys(["a", 7], [DAY_1, DAY_2], 0).tolist()
    assert keys.tolist() == [10801818594554991721, 12149029779827613182]


def test_random_keys_reject_mismatched_lengths():
    with pytest.raises(ValueError, match="2 entity IDs but 1 anchor times"):
        get_row_random_keys([7, 8], [DAY_1], seed=0)


def test_equal_per_cutoff_fills_cutoffs_equally_until_one_runs_out():
    # DAY_3 (newest) has 10 rows, DAY_2 has 5 and DAY_1 has 1.
    labelled = make_labelled(
        [(entity, DAY_3, 0) for entity in range(10)]
        + [(entity, DAY_2, 0) for entity in range(5)]
        + [(0, DAY_1, 0)]
    )

    def get_rows_per_cutoff(max_rows):
        """Return kept rows per cutoff, newest first."""
        policy = ContextPolicy(
            time_sampling="equal_per_cutoff",
            max_rows=max_rows,
            seed=11,
            min_positives=None,
        )
        _, report = select(labelled, policy)
        return [cutoff.kept_rows for cutoff in report.cutoffs]

    # Rank 0 takes one row from each cutoff (3), DAY_1 is then empty, and
    # ranks 1-3 take one row each from DAY_3 and DAY_2: 4, 4, 1.
    assert get_rows_per_cutoff(9) == [4, 4, 1]
    # A tenth row is rank 4 of the newest cutoff first.
    assert get_rows_per_cutoff(10) == [5, 4, 1]
    assert get_rows_per_cutoff(None) == [10, 5, 1]


def test_entity_cap_keeps_the_newest_rows_or_positives_first():
    # Entity 1 at four cutoffs, positive only at the oldest; entity 2 once.
    labelled = make_labelled(
        [
            (1, DAY_4, 0),
            (1, DAY_3, 0),
            (1, DAY_2, 0),
            (1, DAY_1, 1),
            (2, DAY_4, 1),
        ]
    )
    newest, _ = select(
        labelled, ContextPolicy(max_rows_per_entity=2, min_positives=None)
    )
    assert sorted(zip(newest["entity_id"], newest["anchor_time"], strict=True)) == [
        (1, DAY_3),
        (1, DAY_4),
        (2, DAY_4),
    ]
    balanced, _ = select(
        labelled,
        ContextPolicy(
            max_rows_per_entity=1,
            balance=ClassBalance(negatives_per_positive=1),
            min_positives=1,
            correct_probabilities=False,
        ),
    )
    # The cap keeps entity 1's positive (DAY_1) over its newer negatives, so
    # no negative is left to balance the two positives.
    assert sorted(zip(balanced["entity_id"], balanced["anchor_time"], strict=True)) == [
        (1, DAY_1),
        (2, DAY_4),
    ]


def test_report_counts_every_eligible_cutoff_newest_first():
    labelled = rare_label_rows()
    policy = ContextPolicy(
        balance=ClassBalance(negatives_per_positive=1),
        time_sampling="equal_per_cutoff",
        min_positives=1,
    )
    _, report = select(labelled, policy, eligible=[DAY_1, DAY_2, DAY_3])
    assert [cutoff.anchor_time for cutoff in report.cutoffs] == [DAY_3, DAY_2, DAY_1]
    day_3, day_2, day_1 = report.cutoffs
    assert (day_3.labelled_rows, day_3.kept_rows, day_3.kept_positives) == (0, 0, 0)
    # Each day has 5 positives; the 10 negatives split 5 and 5.
    for day in (day_2, day_1):
        assert (day.labelled_rows, day.labelled_positives) == (50, 5)
        assert (day.kept_rows, day.kept_positives) == (10, 5)
    assert report.population_positive_rate == pytest.approx(10 / 100)
    assert report.sample_positive_rate == pytest.approx(10 / 20)
    # Round-trips through JSON, as prediction metadata stores it.
    assert ContextPolicyReport.model_validate_json(report.model_dump_json()) == report


def test_report_omits_positives_for_a_non_binary_label():
    labelled = make_labelled([(entity, DAY_1, entity / 4) for entity in range(30)])
    _, report = select(labelled, ContextPolicy(max_rows=10))
    assert report.positive_label is None
    assert get_report_totals(report)[3] is None
    assert report.population_positive_rate is None
    assert report.labelled_label_counts is None  # 30 distinct values


def test_correction_matches_the_closed_form():
    population_rate, sample_rate = 0.1, 0.5
    factor = get_probability_correction_factor(population_rate, sample_rate, 1)
    assert factor == pytest.approx((0.1 / 0.9) / (0.5 / 0.5))
    probabilities = np.array([0.0, 0.2, 0.5, 0.9, 1.0, 1.3, -0.1])
    expected = [0.0, 0.2 * factor / (0.2 * factor + 0.8)]
    expected += [0.5 * factor / (0.5 * factor + 0.5)]
    expected += [0.9 * factor / (0.9 * factor + 0.1), 1.0, 1.0, 0.0]
    corrected = get_probabilities_with_population_odds(probabilities, factor)
    assert corrected.tolist() == pytest.approx(expected)
    # A model that outputs the sample rate returns the population rate.
    assert corrected[2] == pytest.approx(population_rate)
    # Rates of label 0 give the same factor for label-1 probabilities.
    assert get_probability_correction_factor(0.9, 0.5, 0) == pytest.approx(factor)


def test_balanced_selection_reports_its_correction_factor():
    policy = ContextPolicy(
        balance=ClassBalance(negatives_per_positive=3), min_positives=1
    )
    _, report = select(rare_label_rows(), policy)
    # pi = 10/100, s = 10/40.
    assert report.probability_correction_factor == pytest.approx(
        (0.1 / 0.9) / (0.25 / 0.75)
    )
    unbalanced = ContextPolicy(max_rows=40, min_positives=1)
    assert select(rare_label_rows(), unbalanced)[1].probability_correction_factor is (
        None
    )
    uncorrected = policy.model_copy(update={"correct_probabilities": False})
    assert select(rare_label_rows(), uncorrected)[1].probability_correction_factor is (
        None
    )


def test_too_few_positives_fail_the_selection():
    policy = ContextPolicy(max_rows=20, min_positives=10)
    with pytest.raises(ValueError, match="5 positives across 20 rows"):
        select(rare_label_rows(), policy)


def test_a_policy_above_the_row_limit_fails_instead_of_truncating():
    with pytest.raises(ValueError, match="keeps 100 rows, above the limit of 60"):
        select(rare_label_rows(), ContextPolicy(), max_kept_rows=60)


def test_balance_needs_a_zero_one_label():
    labelled = make_labelled([(entity, DAY_1, entity) for entity in range(5)])
    with pytest.raises(ValueError, match="0/1 label"):
        select(labelled, ContextPolicy(balance=ClassBalance()))


def test_boolean_labels_count_true_as_positive():
    labelled = make_labelled(
        [
            (entity, DAY_1 - timedelta(days=entity % 3), entity < 4)
            for entity in range(12)
        ]
    )
    policy = ContextPolicy(balance=ClassBalance(), min_positives=4)
    context, report = select(labelled, policy)
    assert sorted(context["label"].tolist()) == [False] * 4 + [True] * 4
    assert report.kept_label_counts == {"0": 4, "1": 4}


def test_label_counts_keep_close_values_apart():
    labelled = make_labelled(
        [(0, DAY_1, 2 / 3), (1, DAY_1, 2 / 3 + 1e-9), (2, DAY_1, 1.0)]
    )
    _, report = select(labelled, ContextPolicy())
    assert report.labelled_label_counts == {
        repr(2 / 3): 1,
        repr(2 / 3 + 1e-9): 1,
        "1": 1,
    }


def test_balanced_negatives_follow_the_positives_through_time():
    # DAY_3 (newest) has no positive; DAY_2 has 2 and DAY_1 has 6. Every day
    # has 30 negatives. Filling negatives newest first would take all 8 from
    # DAY_3, where nothing is positive, so the date would predict the class.
    labelled = make_labelled(
        [(entity, DAY_3, 0) for entity in range(30)]
        + [(entity, DAY_2, 1 if entity < 2 else 0) for entity in range(32)]
        + [(entity, DAY_1, 1 if entity < 6 else 0) for entity in range(36)]
    )
    policy = ContextPolicy(balance=ClassBalance(), min_positives=1)
    _, report = select(labelled, policy, eligible=[DAY_1, DAY_2, DAY_3])
    assert [
        (cutoff.anchor_time, cutoff.kept_rows, cutoff.kept_positives)
        for cutoff in report.cutoffs
    ] == [(DAY_3, 0, 0), (DAY_2, 4, 2), (DAY_1, 12, 6)]


def test_balanced_rows_within_an_anchor_time_are_drawn_at_random():
    # One positive (entity 0) and 29 negatives (1-29) on one day; 5 negatives
    # are kept. Entity-key order would keep 1-5.
    labelled = make_labelled(
        [(entity, DAY_1, int(entity == 0)) for entity in range(30)]
    )

    def get_kept_negatives(seed):
        """Return the sorted entity IDs of the negatives kept with this seed."""
        policy = ContextPolicy(
            balance=ClassBalance(negatives_per_positive=5), min_positives=1, seed=seed
        )
        context, _ = select(labelled, policy)
        return sorted(context.loc[context["label"] == 0, "entity_id"])

    assert len(get_kept_negatives(1)) == 5
    assert get_kept_negatives(1) != [1, 2, 3, 4, 5]
    assert get_kept_negatives(1) == get_kept_negatives(1)
    assert get_kept_negatives(1) != get_kept_negatives(2)


def test_a_positive_quota_takes_newest_days_first_and_random_rows_within_one():
    # 10 positives on DAY_2 (entities 0-9) and 10 on DAY_1; keep 3. Newest
    # first takes them from DAY_2, at random rather than entities 0-2.
    labelled = make_labelled(
        [(entity, DAY_2, 1 if entity < 10 else 0) for entity in range(20)]
        + [(entity, DAY_1, 1 if entity < 10 else 0) for entity in range(20)]
    )
    policy = ContextPolicy(balance=ClassBalance(positives=3), min_positives=1, seed=4)
    context, _ = select(labelled, policy)
    positives = context.loc[context["label"] == 1]
    assert set(positives["anchor_time"]) == {DAY_2}
    assert sorted(positives["entity_id"]) != [0, 1, 2]


def test_negative_shares_add_up_by_largest_remainder_and_keep_shortfalls():
    days = pd.to_datetime([DAY_1, DAY_2, DAY_3], utc=True)
    # One kept positive per day and a quota of floor(3 * 1.5) = 4: shares of
    # 1.5 each round to 1, and the leftover row goes to the newest day.
    quotas = get_negative_quotas_by_cutoff(
        pd.Series([1, 1, 1], index=days), pd.Series([10, 10, 10], index=days), 4
    )
    assert quotas.loc[days].tolist() == [1, 1, 2]
    # Shares of 4 and 4, but DAY_2 has 1 negative: it keeps that one, and the
    # 3 it lacks are not taken from DAY_1.
    quotas = get_negative_quotas_by_cutoff(
        pd.Series([4, 4], index=days[:2]), pd.Series([10, 1], index=days[:2]), 8
    )
    assert quotas.loc[days[:2]].tolist() == [4, 1]


def test_event_time_rows_are_balanced_per_cutoff_group_not_per_instant():
    # Event-time rows sit at their own minute, so no positive shares an exact
    # anchor time with a negative. Grouped by day, DAY_2 has 6 positives and
    # DAY_1 has 2, each day with 20 negatives: 1:1 keeps 6 + 6 and 2 + 2.
    rows = []
    for day, positives in ((DAY_2, 6), (DAY_1, 2)):
        for index in range(positives + 20):
            rows.append(
                {
                    "context_row_id": len(rows) + 1,
                    "entity_id": index,
                    "anchor_time": pd.Timestamp(day) + pd.Timedelta(minutes=index),
                    "cutoff_group": pd.Timestamp(day),
                    "is_positive": index < positives,
                }
            )
    pool = pd.DataFrame(rows)
    pool["random_key"] = get_row_random_keys(
        pool["entity_id"], pool["anchor_time"], seed=0
    )
    policy = ContextPolicy(balance=ClassBalance(), time_sampling="random")
    kept = get_rows_within_class_quotas(pool, policy)
    counts = kept.groupby("cutoff_group")["is_positive"].agg(["size", "sum"])
    assert counts.loc[pd.Timestamp(DAY_2)].tolist() == [12, 6]
    assert counts.loc[pd.Timestamp(DAY_1)].tolist() == [4, 2]
