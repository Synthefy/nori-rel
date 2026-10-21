"""Composition features: first and last values, entropy, the most common and the
last value's shares and distinct values over the rows a path reaches, in the
rows' own time order.

A signup's journey is its views; each view has a channel and a drug, and each drug
a class. Every value is checked against a direct computation over the journey
views visible at the signup's own time, and over all of them for static anchors.
"""

import math
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from nori_rel.feature.dfs.noridfs import (
    AnchorDefinition,
    DatabaseView,
    NativeFeatureConfig,
    NativeRelationalFeaturizer,
)


def get_journey_database(seed: int = 0) -> DatabaseView:
    """Return signups, their journey views (some after the signup) and drugs."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-07-01", tz="UTC")
    signups = pd.DataFrame(
        {
            "signup_id": np.arange(1, 41),
            "signup_time": start
            + pd.to_timedelta(rng.integers(60, 2000, 40), unit="min"),
        }
    )
    owner = rng.integers(1, 41, 300)
    views = pd.DataFrame(
        {
            "view_id": np.arange(1, 301),
            "signup_id": owner,
            "drug_id": rng.integers(1, 9, 300),
            "channel": rng.choice(["search", "email", "direct", None], 300),
            # Up to an hour either side of the signup: later views are not history.
            "view_time": signups.set_index("signup_id")
            .loc[owner, "signup_time"]
            .reset_index(drop=True)
            + pd.to_timedelta(rng.integers(-60, 30, 300), unit="min")
            # Distinct seconds, so the journey order is unambiguous.
            + pd.to_timedelta(np.arange(300), unit="s"),
        }
    )
    drugs = pd.DataFrame(
        {
            "drug_id": np.arange(1, 9),
            "drug_class": ["a", "a", "b", "b", "c", "c", "d", "a"],
        }
    )
    return DatabaseView.from_parts(
        tables={"signups": signups, "views": views, "drugs": drugs},
        primary_keys={"signups": "signup_id", "views": "view_id", "drugs": "drug_id"},
        foreign_keys=[
            ("views", "signup_id", "signups", "signup_id"),
            ("views", "drug_id", "drugs", "drug_id"),
        ],
        time_columns={"signups": "signup_time", "views": "view_time"},
    )


def get_expected_mix(values: list) -> dict[str, float]:
    """Return the entropy, most common and last value's shares and distinct count.

    ``values`` are in row order, so the last one is the latest row's.
    """
    values = [value for value in values if value is not None]
    if not values:
        return {
            "entropy": np.nan,
            "mode_share": np.nan,
            "last_share": np.nan,
            "distinct": 0.0,
        }
    counts = Counter(values)
    total = sum(counts.values())
    return {
        "entropy": -sum(c / total * math.log(c / total) for c in counts.values()),
        "mode_share": max(counts.values()) / total,
        "last_share": counts[values[-1]] / total,
        "distinct": float(len(counts)),
    }


def get_mix_columns(frame: pd.DataFrame, source: str) -> dict[str, str]:
    """Return {operation: column} for the category-mix features of ``source``."""
    columns = {}
    for column in frame.columns:
        parts = column.split("|")
        if (
            len(parts) >= 3
            and parts[-3] == source
            and parts[-1] == "all"
            and parts[-2] in {"entropy", "mode_share", "last_share", "distinct"}
        ):
            columns[parts[-2]] = column
    return columns


@pytest.mark.parametrize("seed", [0, 1])
def test_category_mix_at_each_signup_matches_a_direct_computation(seed):
    database = get_journey_database(seed)
    signups, views, drugs = (
        database.tables[name] for name in ("signups", "views", "drugs")
    )
    anchors = AnchorDefinition.create(
        entity_table="signups", entity_column="signup_id", cutoff_column="cutoff"
    )
    featurizer = NativeRelationalFeaturizer.fit(
        database,
        anchors,
        NativeFeatureConfig(max_features=256),
        include_composition_features=True,
    )
    frame = featurizer.transform(
        database, signups[["signup_id"]].assign(cutoff=signups["signup_time"])
    ).frame
    channel = get_mix_columns(frame, "channel")
    drug_class = get_mix_columns(frame, "drug_class")
    assert (
        set(channel)
        == set(drug_class)
        == {"entropy", "mode_share", "last_share", "distinct"}
    )
    classes = drugs.set_index("drug_id")["drug_class"]
    for position, row in enumerate(signups.itertuples(index=False)):
        journey = views.loc[
            (views["signup_id"] == row.signup_id)
            & (views["view_time"] < row.signup_time)
        ].sort_values("view_time")
        for columns, values in (
            (channel, journey["channel"].tolist()),
            (drug_class, journey["drug_id"].map(classes).tolist()),
        ):
            for operation, value in get_expected_mix(values).items():
                assert frame[columns[operation]].iloc[position] == pytest.approx(
                    value, nan_ok=True, abs=1e-9
                ), (row.signup_id, columns[operation])


def test_category_mix_on_static_anchors_counts_every_row():
    database = get_journey_database(2)
    signups, views = database.tables["signups"], database.tables["views"]
    anchors = AnchorDefinition.create(
        entity_table="signups", entity_column="signup_id", cutoff_column=None
    )
    frame = (
        NativeRelationalFeaturizer.fit(
            database,
            anchors,
            NativeFeatureConfig(max_features=256),
            include_composition_features=True,
        )
        .transform(database, signups[["signup_id"]])
        .frame
    )
    channel = get_mix_columns(frame, "channel")
    for position, signup_id in enumerate(signups["signup_id"]):
        values = (
            views.loc[views["signup_id"] == signup_id]
            .sort_values("view_time")["channel"]
            .tolist()
        )
        for operation, value in get_expected_mix(values).items():
            assert frame[channel[operation]].iloc[position] == pytest.approx(
                value, nan_ok=True, abs=1e-9
            )


def test_category_mix_is_opt_in():
    database = get_journey_database()
    anchors = AnchorDefinition.create(
        entity_table="signups", entity_column="signup_id", cutoff_column="cutoff"
    )
    plan = NativeRelationalFeaturizer.fit(
        database, anchors, NativeFeatureConfig(max_features=256)
    ).plan
    assert not any(
        spec.operation in {"entropy", "mode_share", "last_share", "distinct"}
        for spec in plan.features
    )


@pytest.mark.parametrize("seed", [0, 1])
def test_first_and_last_values_follow_the_journey_order(seed):
    # Every journey view carries the signup's time on the path, because a
    # path's time is the latest along it; first and last follow the views' own
    # times instead.
    database = get_journey_database(seed)
    signups, views, drugs = (
        database.tables[name] for name in ("signups", "views", "drugs")
    )
    anchors = AnchorDefinition.create(
        entity_table="signups", entity_column="signup_id", cutoff_column="cutoff"
    )
    frame = (
        NativeRelationalFeaturizer.fit(
            database,
            anchors,
            NativeFeatureConfig(max_features=256),
            include_composition_features=True,
        )
        .transform(
            database, signups[["signup_id"]].assign(cutoff=signups["signup_time"])
        )
        .frame
    )

    def get_column(source: str, operation: str) -> str:
        """Return the one feature column for ``source`` and ``operation``."""
        (column,) = [
            name
            for name in frame.columns
            if name.split("|")[-3:] == [source, operation, "all"]
        ]
        return column

    classes = drugs.set_index("drug_id")["drug_class"]
    for position, row in enumerate(signups.itertuples(index=False)):
        journey = views.loc[
            (views["signup_id"] == row.signup_id)
            & (views["view_time"] < row.signup_time)
        ].sort_values("view_time")
        channels = journey["channel"].dropna()
        drug_classes = journey["drug_id"].map(classes).dropna()
        expected = {
            ("channel", "first"): channels.iloc[0] if len(channels) else None,
            ("channel", "last"): channels.iloc[-1] if len(channels) else None,
            ("drug_class", "first"): drug_classes.iloc[0]
            if len(drug_classes)
            else None,
        }
        for (source, operation), value in expected.items():
            actual = frame[get_column(source, operation)].iloc[position]
            if value is None:
                assert pd.isna(actual), (row.signup_id, source, operation)
            else:
                assert actual == value, (row.signup_id, source, operation)
