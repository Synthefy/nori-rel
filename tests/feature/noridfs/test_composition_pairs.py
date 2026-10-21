"""Composition features of value pairs: a journey's drug class and pharmacy company
together.

A signup's journey is its views; each view has a drug and a pharmacy, each drug a
class and each pharmacy a company. The pair features describe the (class, company)
pairs of the journey views visible at the signup's own time, in the views' order,
and are checked against a direct computation.
"""

import math
import urllib.parse
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
from nori_rel.pipeline.features import _get_feature_record, _get_feature_spec

PAIR = ("drugs.drug_class", "pharmacies.company")


def get_journey_database(seed: int) -> DatabaseView:
    """Return signups, journey views (some after the signup), drugs and pharmacies."""
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
    pharmacy = rng.integers(1, 7, 300).astype(float)
    pharmacy[rng.random(300) < 0.1] = np.nan  # some views have no pharmacy
    views = pd.DataFrame(
        {
            "view_id": np.arange(1, 301),
            "signup_id": owner,
            "drug_id": rng.integers(1, 9, 300),
            "pharmacy_id": pd.array(pharmacy, dtype="Int64"),
            "view_time": signups.set_index("signup_id")
            .loc[owner, "signup_time"]
            .reset_index(drop=True)
            + pd.to_timedelta(rng.integers(-60, 30, 300), unit="min")
            # Distinct seconds, so the journey order is unambiguous.
            + pd.to_timedelta(np.arange(300), unit="s"),
        }
    )
    drugs = pd.DataFrame({"drug_id": np.arange(1, 9), "drug_class": list("aabbccda")})
    pharmacies = pd.DataFrame(
        {"pharmacy_id": np.arange(1, 7), "company": ["x", "y", "x", "z", "y", "w"]}
    )
    return DatabaseView.from_parts(
        tables={
            "signups": signups,
            "views": views,
            "drugs": drugs,
            "pharmacies": pharmacies,
        },
        primary_keys={
            "signups": "signup_id",
            "views": "view_id",
            "drugs": "drug_id",
            "pharmacies": "pharmacy_id",
        },
        foreign_keys=[
            ("views", "signup_id", "signups", "signup_id"),
            ("views", "drug_id", "drugs", "drug_id"),
            ("views", "pharmacy_id", "pharmacies", "pharmacy_id"),
        ],
        time_columns={"signups": "signup_time", "views": "view_time"},
    )


def get_pair_columns(frame: pd.DataFrame) -> dict[str, str]:
    """Return {operation: column} for the pair features of the named pair."""
    columns = {}
    for column in frame.columns:
        parts = urllib.parse.unquote(column).split("|")
        if parts[1] == "pair" and "drug_class" in parts and "company" in parts:
            columns[parts[-2]] = column
    return columns


def get_expected_pair_features(pairs: list[str]) -> dict[str, object]:
    """Return the expected pair features of ``pairs``, given in journey order."""
    if not pairs:
        return {
            "first": None,
            "entropy": np.nan,
            "mode_share": np.nan,
            "last_share": np.nan,
            "distinct": 0.0,
        }
    counts = Counter(pairs)
    total = len(pairs)
    expected: dict[str, object] = {
        "first": pairs[0],
        "entropy": -sum(c / total * math.log(c / total) for c in counts.values()),
        "mode_share": max(counts.values()) / total,
        "last_share": counts[pairs[-1]] / total,
        "distinct": float(len(counts)),
    }
    top = counts.most_common(2)
    if len(top) == 1 or top[0][1] > top[1][1]:  # the mode is checked when unique
        expected["mode"] = top[0][0]
    return expected


def get_featurizer(database: DatabaseView, cutoff_column: str | None, **options):
    """Return a featurizer fitted with a wide budget and ``options``."""
    anchors = AnchorDefinition.create(
        entity_table="signups", entity_column="signup_id", cutoff_column=cutoff_column
    )
    return NativeRelationalFeaturizer.fit(
        database, anchors, NativeFeatureConfig(max_features=512), **options
    )


@pytest.mark.parametrize("seed", [0, 1])
def test_pair_features_at_each_signup_match_a_direct_computation(seed):
    database = get_journey_database(seed)
    signups, views = database.tables["signups"], database.tables["views"]
    drugs = database.tables["drugs"].set_index("drug_id")["drug_class"]
    companies = database.tables["pharmacies"].set_index("pharmacy_id")["company"]
    featurizer = get_featurizer(database, "cutoff", composition_pairs=(PAIR,))
    frame = featurizer.transform(
        database, signups[["signup_id"]].assign(cutoff=signups["signup_time"])
    ).frame
    columns = get_pair_columns(frame)
    assert set(columns) == {
        "mode",
        "first",
        "entropy",
        "mode_share",
        "last_share",
        "distinct",
    }
    for position, row in enumerate(signups.itertuples(index=False)):
        journey = views.loc[
            (views["signup_id"] == row.signup_id)
            & (views["view_time"] < row.signup_time)
            & views["pharmacy_id"].notna()
        ].sort_values("view_time")
        pairs = [
            f"{drugs[drug]}\x1f{companies[pharmacy]}"
            for drug, pharmacy in zip(
                journey["drug_id"], journey["pharmacy_id"], strict=True
            )
        ]
        for operation, value in get_expected_pair_features(pairs).items():
            actual = frame[columns[operation]].iloc[position]
            if operation in {"mode", "first"}:
                assert (None if pd.isna(actual) else actual) == value, (
                    row.signup_id,
                    operation,
                )
            else:
                assert actual == pytest.approx(value, nan_ok=True, abs=1e-9), (
                    row.signup_id,
                    operation,
                )


def test_pair_features_on_static_anchors_count_every_view():
    database = get_journey_database(2)
    signups, views = database.tables["signups"], database.tables["views"]
    drugs = database.tables["drugs"].set_index("drug_id")["drug_class"]
    companies = database.tables["pharmacies"].set_index("pharmacy_id")["company"]
    frame = (
        get_featurizer(database, None, composition_pairs=(PAIR,))
        .transform(database, signups[["signup_id"]])
        .frame
    )
    columns = get_pair_columns(frame)
    for position, signup_id in enumerate(signups["signup_id"]):
        journey = views.loc[
            (views["signup_id"] == signup_id) & views["pharmacy_id"].notna()
        ].sort_values("view_time")
        pairs = [
            f"{drugs[drug]}\x1f{companies[pharmacy]}"
            for drug, pharmacy in zip(
                journey["drug_id"], journey["pharmacy_id"], strict=True
            )
        ]
        expected = get_expected_pair_features(pairs)
        for operation in ("entropy", "mode_share", "last_share", "distinct"):
            assert frame[columns[operation]].iloc[position] == pytest.approx(
                expected[operation], nan_ok=True, abs=1e-9
            )


def test_pair_features_are_planned_only_when_named():
    database = get_journey_database(0)
    plain = get_featurizer(database, "cutoff", include_composition_features=True)
    assert not any(spec.paired_path is not None for spec in plain.plan.features)
    named = get_featurizer(database, "cutoff", composition_pairs=(PAIR,))
    assert sum(spec.paired_path is not None for spec in named.plan.features) == 6
    unknown = get_featurizer(
        database, "cutoff", composition_pairs=(("drugs.drug_class", "nowhere.x"),)
    )
    assert not any(spec.paired_path is not None for spec in unknown.plan.features)


def test_plans_without_pairs_keep_their_digest():
    database = get_journey_database(0)
    plan = get_featurizer(database, "cutoff").plan
    named = get_featurizer(database, "cutoff", composition_pairs=(PAIR,)).plan
    unpaired = [spec for spec in named.features if spec.paired_path is None]
    assert [spec.name for spec in plan.features] == [
        spec.name for spec in unpaired[: len(plan.features)]
    ]
    rebuilt = type(plan)(
        plan.schema_digest, plan.anchors, plan.config, plan.paths, plan.features
    )
    assert rebuilt.digest == plan.digest
    assert "paired" not in str(
        [
            _get_feature_record(spec).model_dump(exclude_none=True)
            for spec in plan.features
        ]
    )


def test_a_pair_feature_round_trips_through_its_record():
    database = get_journey_database(0)
    plan = get_featurizer(database, "cutoff", composition_pairs=(PAIR,)).plan
    for spec in plan.features:
        if spec.paired_path is not None:
            assert _get_feature_spec(_get_feature_record(spec)) == spec
