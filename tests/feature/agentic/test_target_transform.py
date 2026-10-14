"""A program may choose how Nori reads the target, when the search allows it."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from nori_rel.feature import CUTOFF_COLUMN
from nori_rel.feature.agentic import (
    AgenticFeatureConfig,
    FeatureProgram,
    FeatureQuery,
    PlanningSession,
    ProgramAgentBase,
    ProgramRejected,
    ScoreReport,
)
from nori_rel.feature.agentic._agent_base import parse_program
from nori_rel.feature.dfs.noridfs import AnchorDefinition, DatabaseView


class _TransformScorer:
    """Records the target_transform keyword, if any, of every call."""

    def __init__(self) -> None:
        self.transforms: list[str | None] = []

    def __call__(
        self,
        context_features,
        context_target,
        holdout_features,
        holdout_target,
        *,
        categorical_columns,
        text_columns=(),
        **kwargs,
    ) -> ScoreReport:
        self.transforms.append(kwargs.get("target_transform"))
        return ScoreReport(metric="neg_mae", score=0.0, noise=0.0, predictions=None)


def _session(snapshot, anchors, target, *, allow: bool, scorer) -> PlanningSession:
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
        config=AgenticFeatureConfig(
            max_trials=4, leak_check_anchors=0, allow_target_transform=allow
        ),
        train_anchors=anchors(10, start="2020-01-03"),
        train_target=target,
        scorer=scorer,
    )


def _program(sql: str, transform: str) -> FeatureProgram:
    return FeatureProgram(
        (FeatureQuery(name="hist", sql=sql),), target_transform=transform
    )


def test_a_transform_is_refused_unless_the_search_allows_it(
    snapshot, anchors, sql
) -> None:
    scorer = _TransformScorer()
    session = _session(
        snapshot, anchors, pd.Series(range(10), dtype=float), allow=False, scorer=scorer
    )
    try:
        with pytest.raises(ProgramRejected, match="leave it at auto"):
            session.validate_program(_program(sql["safe"], "log1p"))
        session.validate_program(_program(sql["safe"], "auto"))
        # An auto program reaches the scorer without the keyword at all.
        assert scorer.transforms == [None]
    finally:
        session.close()


def test_an_allowed_transform_reaches_the_scorer(snapshot, anchors, sql) -> None:
    scorer = _TransformScorer()
    session = _session(
        snapshot, anchors, pd.Series(range(10), dtype=float), allow=True, scorer=scorer
    )
    try:
        session.validate_program(_program(sql["safe"], "log1p"))
        session.validate_program(_program(sql["safe"], "identity"))
        assert scorer.transforms == ["log1p", "identity"]
    finally:
        session.close()


def test_log1p_needs_a_non_negative_target(snapshot, anchors, sql) -> None:
    target = pd.Series([float(i) - 3 for i in range(10)])
    session = _session(snapshot, anchors, target, allow=True, scorer=_TransformScorer())
    try:
        with pytest.raises(ProgramRejected, match="non-negative"):
            session.validate_program(_program(sql["safe"], "log1p"))
    finally:
        session.close()


def test_the_tool_offers_the_transform_only_when_allowed(snapshot, anchors) -> None:
    for allow in (False, True):
        session = _session(
            snapshot,
            anchors,
            pd.Series(range(10), dtype=float),
            allow=allow,
            scorer=_TransformScorer(),
        )
        try:
            tools = {spec.name: spec for spec in ProgramAgentBase.tools(None, session)}
            properties = tools["validate_program"].parameters["properties"]
            assert ("target_transform" in properties) is allow
        finally:
            session.close()


def test_programs_parse_and_check_their_transform() -> None:
    queries = json.dumps([{"name": "a", "sql": "SELECT 1"}])
    assert parse_program(queries, (), "log1p").target_transform == "log1p"
    assert parse_program(queries).target_transform == "auto"
    with pytest.raises(ValueError, match="target_transform"):
        parse_program(queries, (), "sqrt")
    plain = parse_program(queries)
    assert plain.digest != parse_program(queries, (), "identity").digest
