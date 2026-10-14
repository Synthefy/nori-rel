"""The DuckDB executor returns one aligned row per anchor or says why not."""

from __future__ import annotations

import numpy as np
import pytest

from nori_rel.feature.agentic import FeatureProgram, FeatureQuery, ProgramExecutor
from nori_rel.feature.dfs.noridfs import DatabaseView


def _view(snapshot) -> DatabaseView:
    return DatabaseView.from_parts(
        tables=snapshot.tables,
        primary_keys=snapshot.primary_keys,
        foreign_keys=snapshot.foreign_keys,
        time_columns=snapshot.time_columns,
    )


def _executor(snapshot, **kwargs) -> ProgramExecutor:
    return ProgramExecutor(
        _view(snapshot),
        entity_column="account_id",
        timeout_seconds=kwargs.pop("timeout_seconds", 30.0),
        **kwargs,
    )


def test_features_follow_anchor_order_and_prefix_names(snapshot, anchors, sql) -> None:
    executor = _executor(snapshot)
    try:
        rows = anchors(4, start="2020-01-05")
        result = executor.run(
            FeatureProgram((FeatureQuery("hist", sql["safe"]),)), rows
        )
    finally:
        executor.close()
    assert list(result.frame.columns) == ["hist__n_events", "hist__mean_value"]
    assert result.frame["hist__n_events"].tolist() == [4.0, 1.0, 0.0, 4.0]
    assert result.diagnostics["hist"].coverage == 1.0
    assert result.categorical_columns == ()
    assert result.text_columns == ()


def test_fanout_and_missing_row_id_are_rejected(snapshot, anchors) -> None:
    executor = _executor(snapshot, fanout_limit=2)
    rows = anchors(3, start="2020-01-05")
    fanout = FeatureQuery(
        "bad",
        "SELECT e.row_id, ev.value FROM eval_table e JOIN events ev "
        "ON ev.account_id = e.account_id AND ev.timestamp < e.cutoff",
    )
    try:
        with pytest.raises(ValueError, match=r"aggregate to one row|several rows"):
            executor.run(FeatureProgram((fanout,)), rows)
        no_row = FeatureQuery("worse", "SELECT COUNT(*) AS n FROM events")
        with pytest.raises(ValueError, match="row_id"):
            executor.run(FeatureProgram((no_row,)), rows)
    finally:
        executor.close()


def test_types_are_coerced_and_missing_values_survive(snapshot, anchors) -> None:
    executor = _executor(snapshot)
    query = FeatureQuery(
        "mixed",
        "SELECT e.row_id, a.region, a.region = 'west' AS is_west, "
        "       NULLIF(e.row_id, 0) AS maybe_null, "
        "       CAST(e.cutoff AS TIMESTAMP) AS anchor_day "
        "FROM eval_table e JOIN accounts a ON a.account_id = e.account_id",
    )
    try:
        result = executor.run(FeatureProgram((query,)), anchors(3, start="2020-01-05"))
    finally:
        executor.close()
    assert set(result.categorical_columns) == {"mixed__region", "mixed__is_west"}
    assert result.frame["mixed__is_west"].tolist() == ["true", "false", "true"]
    assert np.isnan(result.frame["mixed__maybe_null"].iloc[0])
    # 2020-01-05 is 18,266 days after the epoch; the unit must survive.
    assert result.frame["mixed__anchor_day"].iloc[0] == 18_266.0


def test_unbounded_values_become_missing(snapshot, anchors) -> None:
    executor = _executor(snapshot)
    query = FeatureQuery("huge", "SELECT e.row_id, 1e35 AS boom FROM eval_table e")
    try:
        result = executor.run(FeatureProgram((query,)), anchors(2, start="2020-01-05"))
    finally:
        executor.close()
    assert result.frame["huge__boom"].isna().all()


def test_write_statements_and_high_cardinality_text_are_refused(
    snapshot, anchors
) -> None:
    executor = _executor(snapshot, max_categorical_cardinality=1)
    query = FeatureQuery(
        "wide",
        "SELECT e.row_id, a.region FROM eval_table e "
        "JOIN accounts a ON a.account_id = e.account_id",
    )
    try:
        with pytest.raises(ValueError, match="neither numeric, prose, nor a category"):
            executor.run(FeatureProgram((query,)), anchors(3, start="2020-01-05"))
    finally:
        executor.close()


def test_prose_above_the_cardinality_cap_becomes_text_not_an_error(
    snapshot, anchors
) -> None:
    """A description is a feature, and it used to be a rejected program.

    Above the cardinality cap every string was refused, so an agent that
    selected a description got an error rather than the text path the DFS
    entry has always had. Short labels over the cap are still refused, which
    is the case the rule has to keep separating.
    """
    # Two levels admits region and leaves the per-account summary over the cap.
    executor = _executor(snapshot, max_categorical_cardinality=2)
    query = FeatureQuery(
        "prose",
        "SELECT e.row_id, a.region, "
        "       'account ' || a.account_id || ' operates in the ' || a.region "
        "       || ' region and was reviewed in the most recent cycle' AS summary "
        "FROM eval_table e JOIN accounts a ON a.account_id = e.account_id",
    )
    try:
        result = executor.run(FeatureProgram((query,)), anchors(3, start="2020-01-05"))
    finally:
        executor.close()
    assert result.text_columns == ("prose__summary",)
    assert "prose__summary" not in result.categorical_columns
    # Short and over the cap stays a rejection, so region is a category here.
    assert result.categorical_columns == ("prose__region",)
    assert result.frame["prose__summary"].iloc[0].startswith("account ")


def test_allow_text_off_restores_the_pre_text_refusal(snapshot, anchors) -> None:
    """The ablation has to actually ablate.

    With the switch off a prose column is refused exactly as it was before
    the text path existed, so an "agent without text" arm is the same
    implementation minus one capability rather than a different codebase.
    """
    query = FeatureQuery(
        "prose",
        "SELECT e.row_id, "
        "       'account ' || a.account_id || ' operates in the ' || a.region "
        "       || ' region and was reviewed in the most recent cycle' AS summary "
        "FROM eval_table e JOIN accounts a ON a.account_id = e.account_id",
    )
    rows = anchors(3, start="2020-01-05")

    on = _executor(snapshot, max_categorical_cardinality=2)
    try:
        assert on.run(FeatureProgram((query,)), rows).text_columns == (
            "prose__summary",
        )
    finally:
        on.close()

    off = _executor(snapshot, max_categorical_cardinality=2, allow_text=False)
    try:
        with pytest.raises(ValueError, match="neither numeric, prose, nor a category"):
            off.run(FeatureProgram((query,)), rows)
    finally:
        off.close()


def test_exploration_sql_is_row_capped(snapshot, anchors) -> None:
    executor = _executor(snapshot)
    try:
        executor.bind_anchors(anchors(6, start="2020-01-05"))
        assert len(executor.sql("SELECT * FROM events", limit=3)) == 3
        assert len(executor.sql("SELECT * FROM eval_table", limit=50)) == 6
    finally:
        executor.close()


@pytest.mark.parametrize(
    "statement",
    [
        "SELECT * FROM read_text('{path}')",
        "SELECT * FROM read_csv('{path}')",
        "SELECT * FROM glob('{directory}/*')",
    ],
)
def test_agent_sql_cannot_read_local_files(snapshot, tmp_path, statement) -> None:
    # DuckDB's disabled external access is the only barrier between
    # agent-written SQL and the files on the caller's machine.
    secret = tmp_path / "secret.csv"
    secret.write_text("token\nnot-for-the-model\n")
    executor = _executor(snapshot)
    try:
        with pytest.raises(ValueError, match="disabled by configuration"):
            executor.sql(statement.format(path=secret, directory=tmp_path), limit=50)
    finally:
        executor.close()
