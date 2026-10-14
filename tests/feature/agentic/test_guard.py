"""Static findings catch syntax; the point-in-time check catches semantics."""

from __future__ import annotations

from nori_rel.feature.agentic import (
    FeatureProgram,
    FeatureQuery,
    ProgramExecutor,
    point_in_time_findings,
    static_findings,
)
from nori_rel.feature.dfs.noridfs import DatabaseView


def _view(snapshot) -> DatabaseView:
    return DatabaseView.from_parts(
        tables=snapshot.tables,
        primary_keys=snapshot.primary_keys,
        foreign_keys=snapshot.foreign_keys,
        time_columns=snapshot.time_columns,
    )


def test_static_layer_flags_forward_windows_and_writes(snapshot, sql) -> None:
    view = _view(snapshot)
    assert static_findings(FeatureQuery("ok", sql["safe"]), view) == ()
    leaky = static_findings(FeatureQuery("leak", sql["leaky"]), view)
    assert any("forward-looking" in finding.reason for finding in leaky)
    write = static_findings(FeatureQuery("w", "DROP TABLE events"), view)
    assert any("blocked" in finding.reason for finding in write)
    unfiltered = static_findings(
        FeatureQuery(
            "u", "SELECT e.row_id, (SELECT COUNT(*) FROM events) AS n FROM eval_table e"
        ),
        view,
    )
    assert any("without a" in finding.reason for finding in unfiltered)


def test_a_comment_cannot_hide_a_forward_filter(snapshot, sql) -> None:
    view = _view(snapshot)
    disguised = FeatureQuery(
        "sneaky",
        "SELECT e.row_id, COUNT(*) AS n FROM eval_table e JOIN events ev\n"
        "  ON ev.account_id = e.account_id /* ev.timestamp < e.cutoff */\n"
        " AND ev.timestamp >= e.cutoff GROUP BY e.row_id",
    )
    assert any("forward-looking" in f.reason for f in static_findings(disguised, view))


def test_point_in_time_layer_catches_a_leak_the_regex_misses(
    snapshot, anchors, sql
) -> None:
    view = _view(snapshot)
    rows = anchors(3, start="2020-01-05")
    hidden = FeatureQuery("hidden", sql["hidden"])
    assert static_findings(hidden, view) == ()

    program = FeatureProgram((FeatureQuery("safe", sql["safe"]), hidden))
    executor = ProgramExecutor(view, entity_column="account_id", timeout_seconds=30.0)
    try:
        reference = executor.run(program, rows).frame
    finally:
        executor.close()

    findings = point_in_time_findings(
        program,
        view,
        rows,
        reference,
        entity_column="account_id",
        sample=[0, 1, 2],
        timeout_seconds=30.0,
    )
    assert findings
    assert {finding.query for finding in findings} == {"hidden"}


def test_point_in_time_layer_clears_a_safe_program(snapshot, anchors, sql) -> None:
    view = _view(snapshot)
    rows = anchors(3, start="2020-01-05")
    program = FeatureProgram((FeatureQuery("safe", sql["safe"]),))
    executor = ProgramExecutor(view, entity_column="account_id", timeout_seconds=30.0)
    try:
        reference = executor.run(program, rows).frame
    finally:
        executor.close()
    assert (
        point_in_time_findings(
            program,
            view,
            rows,
            reference,
            entity_column="account_id",
            sample=[0, 1, 2],
            timeout_seconds=30.0,
        )
        == ()
    )
