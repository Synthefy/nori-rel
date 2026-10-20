"""Apply a ContextPolicy to labelled context rows and report the result.

The connector returns every (entity, cutoff) candidate whose label is known,
numbered newest cutoff first, then entity key. This module chooses among them
in Python, so the choice is the same on every SQL engine: the per-entity cap,
the class balance, then the overall row cap, each filling its quota in the
policy's ``time_sampling`` order. It also builds the ``ContextPolicyReport``
and the probability correction that undoes a balanced class mix.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from datetime import datetime

import numpy as np
import pandas as pd

from nori_rel.context_policy import (
    MAX_REPORTED_LABEL_VALUES,
    ContextPolicy,
    ContextPolicyChooser,
    ContextPolicyReport,
    CutoffContextCounts,
    TimeSampling,
)

CONTEXT_COLUMNS = ["context_row_id", "entity_id", "anchor_time", "label"]
#: Slack for products such as 100 * 0.29 that land just below a whole number.
_ROUNDING_TOLERANCE = 1e-9


_SPLITMIX_INCREMENT = np.uint64(0x9E3779B97F4A7C15)
_SPLITMIX_FIRST_MULTIPLIER = np.uint64(0xBF58476D1CE4E5B9)
_SPLITMIX_SECOND_MULTIPLIER = np.uint64(0x94D049BB133111EB)


def get_splitmix64_scrambled(values: np.ndarray) -> np.ndarray:
    """Return each unsigned 64-bit value passed through the splitmix64 finalizer.

    The finalizer is a one-to-one map on 64-bit integers in which every output
    bit depends on every input bit, so close inputs, such as anchor times one
    day apart, get unrelated outputs. Arithmetic wraps modulo 2**64.
    """
    scrambled = values + _SPLITMIX_INCREMENT
    scrambled = (scrambled ^ (scrambled >> np.uint64(30))) * _SPLITMIX_FIRST_MULTIPLIER
    scrambled = (scrambled ^ (scrambled >> np.uint64(27))) * _SPLITMIX_SECOND_MULTIPLIER
    return scrambled ^ (scrambled >> np.uint64(31))


def get_row_random_keys(
    entity_ids: Sequence[object] | pd.Series,
    anchor_times: Sequence[datetime] | pd.Series,
    seed: int,
) -> np.ndarray:
    """Return one pseudo-random unsigned 64-bit key per (entity, cutoff) row.

    Each distinct entity is hashed once: the first eight bytes of BLAKE2b over
    the seed and the entity ID's string form. The cutoff's UTC nanoseconds go
    through splitmix64, are combined with the entity hash by XOR, and the
    result goes through splitmix64 again. The key depends on nothing else, so
    a row keeps its key when other rows are added or removed, and sorting by
    it gives the same random order on every run with the same seed. Hashing
    each entity once and the cutoffs in numpy keeps this to a fraction of a
    second per million rows.
    """
    entity_codes, distinct_entity_ids = pd.factorize(
        pd.Series(entity_ids), sort=False, use_na_sentinel=False
    )
    anchor_nanoseconds = (
        pd.DatetimeIndex(pd.to_datetime(anchor_times, utc=True))
        .as_unit("ns")
        .asi8.view(np.uint64)
    )
    if len(entity_codes) != len(anchor_nanoseconds):
        raise ValueError(
            f"{len(entity_codes)} entity IDs but {len(anchor_nanoseconds)} anchor times"
        )
    entity_hashes = np.fromiter(
        (
            int.from_bytes(
                hashlib.blake2b(
                    f"{seed}\x1f{entity_id}".encode(), digest_size=8
                ).digest(),
                "big",
            )
            for entity_id in distinct_entity_ids
        ),
        dtype=np.uint64,
        count=len(distinct_entity_ids),
    )
    return get_splitmix64_scrambled(
        entity_hashes[entity_codes] ^ get_splitmix64_scrambled(anchor_nanoseconds)
    )


def get_rows_in_selection_order(
    pool: pd.DataFrame, time_sampling: TimeSampling
) -> pd.DataFrame:
    """Return ``pool`` sorted so that a quota of n rows keeps its first n rows.

    ``pool`` holds ``context_row_id`` (newest cutoff first, then entity key),
    ``anchor_time`` and, unless the order is ``"newest_first"``,
    ``random_key``. ``"equal_per_cutoff"`` ranks rows by random key within
    their cutoff and then takes rank 0 of every cutoff, newest cutoff first,
    then rank 1, and so on, so cutoffs fill equally until one runs out.
    """
    if time_sampling == "newest_first":
        return pool.sort_values("context_row_id", kind="stable")
    by_random_key = pool.sort_values(["random_key", "context_row_id"], kind="stable")
    if time_sampling == "random":
        return by_random_key
    rank_within_cutoff = by_random_key.groupby("anchor_time", sort=False).cumcount()
    return (
        by_random_key.assign(rank_within_cutoff=rank_within_cutoff)
        .sort_values(
            ["rank_within_cutoff", "anchor_time"],
            ascending=[True, False],
            kind="stable",
        )
        .drop(columns="rank_within_cutoff")
    )


def get_rows_within_entity_cap(
    pool: pd.DataFrame, policy: ContextPolicy
) -> pd.DataFrame:
    """Return the rows that survive ``policy.max_rows_per_entity``.

    Each entity keeps its first rows in ``time_sampling`` order. With a class
    balance it keeps its positive rows first, so the cap does not discard the
    rare class first, and then rows at random.
    """
    if policy.max_rows_per_entity is None:
        return pool
    if policy.balance is not None:
        # Positives first, then at random: a time order here would keep each
        # entity's newest negatives and tie class to time.
        ordered = pool.sort_values(
            ["is_positive", "random_key", "context_row_id"],
            ascending=[False, True, True],
            kind="stable",
        )
    else:
        ordered = get_rows_in_selection_order(pool, policy.time_sampling)
    rank_within_entity = ordered.groupby("entity_id", sort=False).cumcount()
    return ordered.loc[rank_within_entity < policy.max_rows_per_entity]


def get_cutoff_column_name(pool: pd.DataFrame) -> str:
    """Return the name of the column whose values are the pool's cutoffs.

    ``cutoff_group`` when the pool has one: event-time rows, each anchored at
    its own timestamp, are grouped into cutoffs by a period such as a day.
    Otherwise ``anchor_time``, so each anchor time is one cutoff.
    """
    return "cutoff_group" if "cutoff_group" in pool.columns else "anchor_time"


def get_positives_in_balance_order(
    positives: pd.DataFrame, time_sampling: TimeSampling
) -> pd.DataFrame:
    """Return positives sorted so that a quota of n keeps the first n.

    ``time_sampling`` decides how the kept positives spread over cutoffs, and
    rows within one cutoff are taken at random: ``"newest_first"`` sorts by
    cutoff, newest first, then by random key, instead of by entity key.
    """
    if time_sampling == "newest_first":
        return positives.sort_values(
            [get_cutoff_column_name(positives), "random_key", "context_row_id"],
            ascending=[False, True, True],
            kind="stable",
        )
    return get_rows_in_selection_order(positives, time_sampling)


def get_negative_quotas_by_cutoff(
    kept_positives_by_cutoff: pd.Series,
    available_negatives_by_cutoff: pd.Series,
    negative_quota: int,
) -> pd.Series:
    """Return how many negatives to keep at each cutoff.

    A cutoff is an anchor time, or a day of event-time rows (see
    ``get_cutoff_column_name``). Each cutoff gets a share of
    ``negative_quota`` proportional to its kept positives, so the class ratio
    is the same at every cutoff and class carries no information about time.
    Shares are rounded down and the rows left over go to the largest
    remainders, newest cutoff first on ties, so the shares add up to the
    quota. A cutoff with fewer negatives than its share keeps every one it
    has; the shortfall is not moved to other cutoffs, because that would tie
    class to time again. The result is indexed by cutoff, with 0 for cutoffs
    without a kept positive.
    """
    cutoffs = available_negatives_by_cutoff.index.union(kept_positives_by_cutoff.index)
    positives = kept_positives_by_cutoff.reindex(cutoffs, fill_value=0)
    available = available_negatives_by_cutoff.reindex(cutoffs, fill_value=0)
    if negative_quota <= 0 or positives.sum() == 0:
        return pd.Series(0, index=cutoffs, dtype=int)
    shares = positives / positives.sum() * negative_quota
    quotas = np.floor(shares + _ROUNDING_TOLERANCE).astype(int)
    leftover = negative_quota - int(quotas.sum())
    if leftover > 0:
        remainders = pd.DataFrame(
            {"remainder": shares - quotas, "cutoff": cutoffs}, index=cutoffs
        ).sort_values(["remainder", "cutoff"], ascending=[False, False])
        quotas.loc[remainders.index[:leftover]] += 1
    return np.minimum(quotas, available).astype(int)


def get_rows_within_class_quotas(
    pool: pd.DataFrame, policy: ContextPolicy
) -> pd.DataFrame:
    """Return the rows kept by the class balance and ``max_rows``.

    Without a balance, the first ``max_rows`` rows in ``time_sampling`` order
    (every row when ``max_rows`` is None). With one, positives fill their
    quota in ``time_sampling`` order, at random within a cutoff. The negative
    quota, ``floor(positives kept * ratio)``, is then split over cutoffs
    (anchor times, or days for event-time rows) in proportion to the
    positives kept at each, and each cutoff's negatives are drawn at random.
    Negatives therefore follow the positives through time instead of coming
    from the newest days.
    """
    balance = policy.balance
    if balance is None:
        ordered = get_rows_in_selection_order(pool, policy.time_sampling)
        return ordered if policy.max_rows is None else ordered.head(policy.max_rows)
    positive_pool = pool.loc[pool["is_positive"]]
    negative_pool = pool.loc[~pool["is_positive"]]
    ratio = balance.negatives_per_positive
    positive_quota = (
        len(positive_pool)
        if balance.positives == "all"
        else min(balance.positives, len(positive_pool))
    )
    if policy.max_rows is not None:
        room_for_positives = math.floor(
            policy.max_rows / (1 + ratio) + _ROUNDING_TOLERANCE
        )
        positive_quota = min(positive_quota, room_for_positives)
    negative_quota = min(
        len(negative_pool), math.floor(positive_quota * ratio + _ROUNDING_TOLERANCE)
    )
    if policy.max_rows is not None:
        negative_quota = min(negative_quota, policy.max_rows - positive_quota)
    kept_positives = get_positives_in_balance_order(
        positive_pool, policy.time_sampling
    ).head(positive_quota)
    cutoff_column = get_cutoff_column_name(pool)
    quotas = get_negative_quotas_by_cutoff(
        kept_positives.groupby(cutoff_column).size(),
        negative_pool.groupby(cutoff_column).size(),
        negative_quota,
    )
    negatives_at_random = negative_pool.sort_values(
        ["random_key", "context_row_id"], kind="stable"
    )
    rank_within_cutoff = negatives_at_random.groupby(
        cutoff_column, sort=False
    ).cumcount()
    kept_negatives = negatives_at_random.loc[
        rank_within_cutoff < negatives_at_random[cutoff_column].map(quotas).fillna(0)
    ]
    return pd.concat([kept_positives, kept_negatives])


def get_positive_label(
    policy: ContextPolicy | None, *, is_binary_label: bool
) -> int | None:
    """Return the label value counted as positive, or None for a non-0/1 label.

    The class balance's ``positive_label`` when there is one, otherwise 1.
    """
    if not is_binary_label:
        return None
    if policy is None or policy.balance is None:
        return 1
    return policy.balance.positive_label


def get_label_value_text(value: float) -> str:
    """Return a label value as report text: ``"1"`` for 1.0, ``"0.25"`` for 0.25.

    The short form is used only when it reads back as the same number, so two
    different values never share a key; otherwise the exact repr is used.
    """
    short = f"{value:g}"
    return short if float(short) == value else repr(float(value))


def get_label_value_counts(label_values: pd.Series) -> dict[str, int] | None:
    """Return rows per label value, or None above the reported value count.

    Keys are the values as ``get_label_value_text`` writes them.
    """
    counts = label_values.value_counts().sort_index()
    if len(counts) > MAX_REPORTED_LABEL_VALUES:
        return None
    return {get_label_value_text(value): int(count) for value, count in counts.items()}


def get_probability_correction_factor(
    population_positive_rate: float, sample_positive_rate: float, positive_label: int
) -> float:
    """Return the odds multiplier that maps sample probabilities of label 1 back.

    The rates are shares of ``positive_label``; the factor applies to the
    probability of label 1 (True), which is what Nori outputs, so a positive
    label of 0 uses the complementary rates.

    Raises:
        ValueError: The kept rows hold a single class, so no odds exist.
    """
    if not 0 < sample_positive_rate < 1:
        raise ValueError(
            "probabilities cannot be corrected: the balanced context holds one "
            "class only"
        )
    if positive_label == 0:
        population_positive_rate = 1 - population_positive_rate
        sample_positive_rate = 1 - sample_positive_rate
    population_odds = population_positive_rate / (1 - population_positive_rate)
    sample_odds = sample_positive_rate / (1 - sample_positive_rate)
    return population_odds / sample_odds


def get_probabilities_with_population_odds(
    probabilities: np.ndarray, correction_factor: float
) -> np.ndarray:
    """Return probabilities with their odds multiplied by ``correction_factor``.

    Values are clipped to [0, 1] first, so a regression mean of a 0/1 label
    that strays slightly outside the range stays a probability. 0 and 1 map
    to themselves.
    """
    clipped = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    scaled = clipped * correction_factor
    return scaled / (scaled + 1.0 - clipped)


def get_selected_context(
    labelled: pd.DataFrame,
    policy: ContextPolicy | None,
    *,
    chosen_by: ContextPolicyChooser,
    max_context_rows: int | None,
    eligible_anchor_times: Sequence[datetime],
    max_kept_rows: int,
    random_key_anchor_time: datetime | None = None,
) -> tuple[pd.DataFrame, ContextPolicyReport]:
    """Return the context rows Nori gets and the report of how they were chosen.

    Args:
        labelled: Every candidate whose label is known, with columns
            context_row_id, entity_id, anchor_time and label.
        policy: The policy to apply, or None to keep every row, as the older
            newest-candidates rule does after its SQL cap.
        chosen_by: Who chose the policy, recorded in the report.
        max_context_rows: The older rule's candidate cap, recorded when
            ``policy`` is None.
        eligible_anchor_times: The training cutoffs whose label window is
            observed; each gets a report entry, even with no labelled rows.
        random_key_anchor_time: When given, every row's random key uses this
            time in place of its own cutoff. A static target's rows all share
            the read time, which moves between runs, so keying on it would
            draw a new sample from the same seed every time.
        max_kept_rows: The most rows a policy may keep
            (``RelationalLimits.max_context_rows``).

    Returns:
        (context, report). ``context`` holds the kept rows of ``labelled``,
        unchanged and in their original order.

    Raises:
        ValueError: A balance on a label that is not 0/1, a policy that keeps
            more than ``max_kept_rows`` rows, fewer positives than
            ``min_positives``, or a correction on a one-class context.
    """
    pool = labelled.loc[:, CONTEXT_COLUMNS].copy()
    pool["anchor_time"] = pd.to_datetime(pool["anchor_time"], utc=True)
    pool["label_value"] = pool["label"].astype(float)
    distinct_values = set(np.unique(pool["label_value"].to_numpy()))
    is_binary_label = bool(distinct_values) and distinct_values <= {0.0, 1.0}
    if policy is not None and policy.balance is not None and not is_binary_label:
        raise ValueError(
            "ClassBalance needs a 0/1 label; this target's labels take other values"
        )
    positive_label = get_positive_label(policy, is_binary_label=is_binary_label)
    pool["is_positive"] = (
        pool["label_value"] == float(positive_label)
        if positive_label is not None
        else False
    )
    if policy is None:
        kept = pool
    else:
        if policy.time_sampling != "newest_first" or policy.balance is not None:
            key_times = (
                pool["anchor_time"]
                if random_key_anchor_time is None
                else pd.Series(pd.Timestamp(random_key_anchor_time), index=pool.index)
            )
            pool["random_key"] = get_row_random_keys(
                pool["entity_id"], key_times, policy.seed
            )
        kept = get_rows_within_class_quotas(
            get_rows_within_entity_cap(pool, policy), policy
        )
        if len(kept) > max_kept_rows:
            raise ValueError(
                f"the context policy keeps {len(kept)} rows, above the limit of "
                f"{max_kept_rows} (RelationalLimits.max_context_rows); set "
                "ContextPolicy.max_rows or ClassBalance.positives"
            )
    report = _get_report(
        pool,
        kept,
        policy,
        chosen_by=chosen_by,
        max_context_rows=max_context_rows,
        positive_label=positive_label,
        eligible_anchor_times=eligible_anchor_times,
    )
    kept_positives = int(kept["is_positive"].sum())
    if (
        policy is not None
        and policy.min_positives is not None
        and positive_label is not None
        and kept_positives < policy.min_positives
    ):
        raise ValueError(
            f"the context holds {kept_positives} positives across "
            f"{len(kept)} rows, fewer than ContextPolicy.min_positives="
            f"{policy.min_positives}; keep every positive with "
            "ClassBalance(positives='all') or pass more anchor_times"
        )
    is_kept = labelled["context_row_id"].isin(kept["context_row_id"])
    return labelled.loc[is_kept].reset_index(drop=True), report


def _get_report(
    pool: pd.DataFrame,
    kept: pd.DataFrame,
    policy: ContextPolicy | None,
    *,
    chosen_by: ContextPolicyChooser,
    max_context_rows: int | None,
    positive_label: int | None,
    eligible_anchor_times: Sequence[datetime],
) -> ContextPolicyReport:
    """Return the report of a selection from ``pool`` (labelled) to ``kept``."""
    has_positives = positive_label is not None
    anchor_times = sorted(
        set(pd.to_datetime(list(eligible_anchor_times), utc=True))
        | set(pool["anchor_time"]),
        reverse=True,
    )
    labelled_by_cutoff = pool.groupby("anchor_time")["is_positive"].agg(["size", "sum"])
    kept_by_cutoff = kept.groupby("anchor_time")["is_positive"].agg(["size", "sum"])

    def get_count(counts: pd.DataFrame, anchor_time, column: str) -> int:
        """Return one cutoff's count from a size/sum table, 0 when absent."""
        return int(counts.at[anchor_time, column]) if anchor_time in counts.index else 0

    cutoffs = tuple(
        CutoffContextCounts(
            anchor_time=anchor_time.to_pydatetime(),
            labelled_rows=get_count(labelled_by_cutoff, anchor_time, "size"),
            labelled_positives=get_count(labelled_by_cutoff, anchor_time, "sum")
            if has_positives
            else None,
            kept_rows=get_count(kept_by_cutoff, anchor_time, "size"),
            kept_positives=get_count(kept_by_cutoff, anchor_time, "sum")
            if has_positives
            else None,
        )
        for anchor_time in anchor_times
    )
    labelled_positives = int(pool["is_positive"].sum()) if has_positives else None
    kept_positives = int(kept["is_positive"].sum()) if has_positives else None
    population_rate = (
        labelled_positives / len(pool) if has_positives and len(pool) else None
    )
    sample_rate = kept_positives / len(kept) if has_positives and len(kept) else None
    correction_factor = None
    if (
        policy is not None
        and policy.balance is not None
        and policy.correct_probabilities
        and positive_label is not None
    ):
        correction_factor = get_probability_correction_factor(
            population_rate, sample_rate or 0.0, positive_label
        )
    return ContextPolicyReport(
        chosen_by=chosen_by,
        policy=policy,
        max_context_rows=max_context_rows if policy is None else None,
        positive_label=positive_label,
        population_positive_rate=population_rate,
        sample_positive_rate=sample_rate,
        probability_correction_factor=correction_factor,
        labelled_label_counts=get_label_value_counts(pool["label_value"]),
        kept_label_counts=get_label_value_counts(kept["label_value"]),
        cutoffs=cutoffs,
    )


__all__ = [
    "CONTEXT_COLUMNS",
    "get_cutoff_column_name",
    "get_label_value_counts",
    "get_label_value_text",
    "get_negative_quotas_by_cutoff",
    "get_positive_label",
    "get_positives_in_balance_order",
    "get_probabilities_with_population_odds",
    "get_probability_correction_factor",
    "get_row_random_keys",
    "get_rows_in_selection_order",
    "get_rows_within_class_quotas",
    "get_rows_within_entity_cap",
    "get_selected_context",
]
