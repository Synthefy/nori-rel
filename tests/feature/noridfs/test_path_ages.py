"""Ages of the rows a path reaches count from each row's own time.

A signup's journey is the views that led to it, all before the signup. A path's
time is the latest along it, so each journey view carries the signup's time on
the path; its age must still be measured from when the view happened. Every age is
checked against a direct computation over the journey views visible at the
signup's own time.
"""

import urllib.parse

import numpy as np
import pandas as pd
import pytest

from nori_rel.feature.dfs.noridfs import (
    AnchorDefinition,
    DatabaseView,
    NativeFeatureConfig,
    NativeRelationalFeaturizer,
)

JOURNEY = "signups<-views.signup_id|rows|"


def get_journey_database(seed: int) -> DatabaseView:
    """Return signups and their journey views, some of them after the signup."""
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
            "price": rng.normal(20, 5, 300),
            # Up to a day before the signup and an hour after: later views are not
            # history.
            "view_time": signups.set_index("signup_id")
            .loc[owner, "signup_time"]
            .reset_index(drop=True)
            + pd.to_timedelta(rng.integers(-1440, 60, 300), unit="min"),
        }
    )
    return DatabaseView.from_parts(
        tables={"signups": signups, "views": views},
        primary_keys={"signups": "signup_id", "views": "view_id"},
        foreign_keys=[("views", "signup_id", "signups", "signup_id")],
        time_columns={"signups": "signup_time", "views": "view_time"},
    )


@pytest.mark.parametrize("seed", [0, 1])
def test_journey_ages_count_from_each_view_own_time(seed):
    database = get_journey_database(seed)
    signups, views = database.tables["signups"], database.tables["views"]
    anchors = AnchorDefinition.create(
        entity_table="signups", entity_column="signup_id", cutoff_column="cutoff"
    )
    frame = (
        NativeRelationalFeaturizer.fit(
            database, anchors, NativeFeatureConfig(max_features=256)
        )
        .transform(
            database, signups[["signup_id"]].assign(cutoff=signups["signup_time"])
        )
        .frame
    )
    columns = {
        operation: next(
            column
            for column in frame.columns
            if urllib.parse.unquote(column).endswith(JOURNEY + f"{operation}|all")
        )
        for operation in ("recency", "mean_age", "oldest_age")
    }
    day = pd.Timedelta(days=1)
    for position, row in enumerate(signups.itertuples(index=False)):
        journey = views.loc[
            (views["signup_id"] == row.signup_id)
            & (views["view_time"] < row.signup_time),
            "view_time",
        ]
        expected = (
            {"recency": np.nan, "mean_age": np.nan, "oldest_age": np.nan}
            if journey.empty
            else {
                "recency": (row.signup_time - journey.max()) / day,
                "mean_age": ((row.signup_time - journey) / day).mean(),
                "oldest_age": (row.signup_time - journey.min()) / day,
            }
        )
        for operation, value in expected.items():
            # The cutoff sits a microsecond before the signup, so rows at its own time
            # stay out; 1e-9 days is about 0.1 ms.
            assert frame[columns[operation]].iloc[position] == pytest.approx(
                value, nan_ok=True, abs=1e-9
            ), (row.signup_id, operation)
