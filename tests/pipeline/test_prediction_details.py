"""PredictionDetails names its feature columns in dots and maps them back."""

from __future__ import annotations

import pandas as pd
import pytest

from nori_rel.feature.dfs.noridfs.contracts import FeatureSpec, ForeignKey, RelationPath
from nori_rel.pipeline import PredictionDetails

DAY_NS = 86_400 * 10**9
DRIVER_RESULTS = ForeignKey("results", "driverId", "drivers", "driverId")
RESULT_RACES = ForeignKey("results", "raceId", "races", "raceId")
DRIVER_QUALIFYING = ForeignKey("qualifying", "driverId", "drivers", "driverId")
QUALIFYING_CONSTRUCTORS = ForeignKey(
    "qualifying", "constructorId", "constructors", "constructorId"
)
RESULT_PIT_STOPS = ForeignKey("pit_stops", "resultId", "results", "resultId")
DRIVER_TEAMS = ForeignKey("drivers", "teamId", "teams", "teamId")
TEAM_COUNTRIES = ForeignKey("teams", "countryId", "countries", "countryId")


def _get_details(
    internal_names: list[str], metadata: dict | None = None, **fields
) -> PredictionDetails:
    """Return details for a one-row matrix with these internal column names."""
    return PredictionDetails(
        predictions=pd.DataFrame(),
        metadata=metadata or {"features": "noridfs"},
        pipeline=None,
        features=pd.DataFrame([[1.0] * len(internal_names)], columns=internal_names),
        **fields,
    )


def _get_dotted_names(
    internal_names: list[str], metadata: dict | None = None
) -> list[str]:
    """Return the column names the details give these internal names."""
    return _get_details(internal_names, metadata).features.columns.tolist()


def _get_bridge(*parents: ForeignKey, first: ForeignKey) -> RelationPath:
    """Return a drivers path down ``first`` and up through ``parents``."""
    return RelationPath("drivers", (first, *parents), "bridge")


@pytest.mark.parametrize(
    ("feature", "expected"),
    [
        (FeatureSpec.anchor("drivers", "cutoff_year", "numeric"), "anchor.cutoff_year"),
        (
            FeatureSpec.anchor("drivers", "entity_key", "categorical"),
            "anchor.entity_key",
        ),
        (FeatureSpec.direct("drivers", "dob", "numeric"), "drivers.dob"),
        (
            FeatureSpec.aggregate(RelationPath("drivers", (DRIVER_RESULTS,)), "count"),
            "results.count",
        ),
        (
            FeatureSpec.aggregate(
                RelationPath("drivers", (DRIVER_RESULTS,)), "recency"
            ),
            "results.recency",
        ),
        (
            FeatureSpec.aggregate(
                RelationPath("drivers", (DRIVER_RESULTS,)), "mean", column="grid"
            ),
            "results.grid.mean",
        ),
        (
            FeatureSpec.aggregate(
                RelationPath("drivers", (DRIVER_RESULTS,)),
                "mean",
                column="grid",
                window_ns=60 * DAY_NS,
            ),
            "results.grid.mean_60d",
        ),
        (
            FeatureSpec.aggregate(
                RelationPath("drivers", (DRIVER_RESULTS,)),
                "count",
                window_ns=12 * 3_600 * 10**9,
            ),
            "results.count_12h",
        ),
        (
            FeatureSpec.aggregate(
                RelationPath("drivers", (DRIVER_RESULTS,)),
                "mode",
                column="status",
                kind="categorical",
            ),
            "results.status.mode",
        ),
        (
            FeatureSpec.aggregate(
                RelationPath("drivers", (DRIVER_RESULTS, RESULT_PIT_STOPS)),
                "max",
                column="duration",
            ),
            "results.pit_stops.duration.max",
        ),
        (
            FeatureSpec.aggregate(
                _get_bridge(QUALIFYING_CONSTRUCTORS, first=DRIVER_QUALIFYING), "count"
            ),
            "qualifying.constructors.count",
        ),
        (
            FeatureSpec.aggregate(
                _get_bridge(RESULT_RACES, first=DRIVER_RESULTS),
                "mean",
                column="round",
                window_ns=30 * DAY_NS,
            ),
            "results.races.round.mean_30d",
        ),
        (
            FeatureSpec.lookup(
                RelationPath("drivers", (DRIVER_TEAMS,), "parent"),
                "name",
                "categorical",
            ),
            "teams.name",
        ),
        (
            FeatureSpec.lookup(
                RelationPath("drivers", (DRIVER_TEAMS, TEAM_COUNTRIES), "parent"),
                "name",
                "categorical",
            ),
            "teams.countries.name",
        ),
    ],
)
def test_each_noridfs_name_form_reads_as_dots(
    feature: FeatureSpec, expected: str
) -> None:
    details = _get_details([feature.name])
    assert details.features.columns.tolist() == [expected]
    assert details.internal_feature_names == (feature.name,)


def test_noridfs_names_escape_nothing_the_database_did_not_name() -> None:
    orders = ForeignKey("order items", "user id", "users", "id")
    feature = FeatureSpec.aggregate(
        RelationPath("users", (orders,)), "sum", column="unit|price"
    )
    assert "%" in feature.name
    assert _get_dotted_names([feature.name]) == ["order items.unit|price.sum"]


def test_colliding_names_show_the_foreign_key_where_the_paths_differ() -> None:
    home = ForeignKey("matches", "home_team_id", "teams", "team_id")
    away = ForeignKey("matches", "away_team_id", "teams", "team_id")
    venue = ForeignKey("matches", "venue_id", "venues", "venue_id")
    city = ForeignKey("matches", "city_venue_id", "venues", "venue_id")
    features = [
        FeatureSpec.aggregate(RelationPath("teams", (home,)), "count"),
        FeatureSpec.aggregate(RelationPath("teams", (away,)), "count"),
        FeatureSpec.aggregate(RelationPath("teams", (home, venue), "bridge"), "count"),
        FeatureSpec.aggregate(RelationPath("teams", (home, city), "bridge"), "count"),
        FeatureSpec.aggregate(RelationPath("teams", (home,)), "mean", column="goals"),
    ]
    assert _get_dotted_names([feature.name for feature in features]) == [
        "matches[home_team_id].count",
        "matches[away_team_id].count",
        "matches.venues[venue_id].count",
        "matches.venues[city_venue_id].count",
        "matches.goals.mean",
    ]


def test_an_agent_column_named_like_a_dfs_column_leaves_both_readable() -> None:
    noridfs = FeatureSpec.aggregate(RelationPath("drivers", (DRIVER_RESULTS,)), "count")
    metadata = {
        "features": "agentic",
        "agentic_program": {"queries": [{"name": "results", "sql": "SELECT 1"}]},
    }
    assert _get_dotted_names([noridfs.name, "results__count"], metadata) == [
        "results[driverId].count",
        "results.count",
    ]


def test_a_name_no_foreign_key_can_tell_apart_stays_internal() -> None:
    noridfs = FeatureSpec.direct("drivers", "dob", "numeric")
    metadata = {
        "features": "agentic",
        "agentic_program": {"queries": [{"name": "drivers", "sql": "SELECT 1"}]},
    }
    assert _get_dotted_names(
        [noridfs.name, "drivers__dob", "drivers__age"], metadata
    ) == [
        noridfs.name,
        "drivers__dob",
        "drivers.age",
    ]


def test_names_are_unique_and_the_same_on_every_build() -> None:
    features = [
        FeatureSpec.aggregate(RelationPath("drivers", (DRIVER_RESULTS,)), operation)
        for operation in ("count", "recency", "mean_age", "oldest_age")
    ] + [
        FeatureSpec.aggregate(_get_bridge(RESULT_RACES, first=DRIVER_RESULTS), "count"),
        FeatureSpec.anchor("drivers", "cutoff_month", "numeric"),
    ]
    internal_names = [feature.name for feature in features]
    first = _get_dotted_names(internal_names)
    assert len(set(first)) == len(first)
    assert _get_dotted_names(internal_names) == first
    assert _get_dotted_names(internal_names[::-1]) == first[::-1]


@pytest.mark.parametrize(
    ("internal_name", "expected"),
    [
        ("d2__drivers.COUNT(results)", "results.count"),
        ("d2__drivers.MAX(results.grid)", "results.grid.max"),
        ("d2__drivers.NUM_UNIQUE(results.statusId)", "results.statusId.num_unique"),
        ("d1__drivers.nationality", "drivers.nationality"),
        ("d2__drivers.races.year", "races.year"),
        ("d2__drivers.MEAN(results.races.round)", "results.races.round.mean"),
        (
            "d2__drivers.MEAN(results.MAX(pit_stops.duration))",
            "results.pit_stops.duration.max.mean",
        ),
        ("d2__teams.COUNT(matches[home_team_id])", "matches[home_team_id].count"),
        ("d2__users.MEAN(`order.items`.amount)", "`order.items`.amount.mean"),
        ("d2__drivers.TREND(results.grid, date)", "drivers.TREND(results.grid, date)"),
        (
            "d2__drivers.COUNT(results WHERE grid = 1)",
            "drivers.COUNT(results WHERE grid = 1)",
        ),
    ],
)
def test_fastdfs_names_read_as_dots(internal_name: str, expected: str) -> None:
    assert _get_dotted_names([internal_name], {"features": "fastdfs"}) == [expected]


def test_names_of_other_backends_stay_as_they_are() -> None:
    assert _get_dotted_names(["d2__x.MAX(y.z)"], {"features": "MyBackend"}) == [
        "d2__x.MAX(y.z)"
    ]


def test_the_feature_table_carries_fastdfs_sql_by_internal_name() -> None:
    sql = 'SELECT COUNT("results"."resultId") FROM "results";'
    details = _get_details(
        ["d2__drivers.COUNT(results)", "d2__drivers.MAX(results.grid)"],
        {"features": "fastdfs"},
        feature_sql={"d2__drivers.COUNT(results)": sql},
    )
    table = details.get_feature_table()
    assert table["feature"].tolist() == ["results.count", "results.grid.max"]
    assert table["source"].tolist() == ["fastdfs", "fastdfs"]
    assert table["sql"].tolist() == [sql, None]


def test_details_built_from_dotted_names_keep_them() -> None:
    details = _get_details(["d2__drivers.COUNT(results)"], {"features": "fastdfs"})
    rebuilt = PredictionDetails(
        predictions=details.predictions,
        metadata=details.metadata,
        pipeline=None,
        features=details.features,
        internal_feature_names=details.internal_feature_names,
    )
    assert rebuilt.features.columns.tolist() == ["results.count"]
    assert rebuilt.get_feature_table()["internal_name"].tolist() == [
        "d2__drivers.COUNT(results)"
    ]
    with pytest.raises(ValueError, match="one name per column"):
        PredictionDetails(
            predictions=details.predictions,
            metadata=details.metadata,
            pipeline=None,
            features=details.features,
            internal_feature_names=("a", "b"),
        )
