"""Limited compatibility rejects unsupported shapes explicitly."""

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from nori_rel import NoriRelClient
from nori_rel._legacy import aggregate_query
from nori_rel.contracts import SchemaGraph
from nori_rel.pql import parse_pql
from nori_rel.pql.binding import bind_columns
from nori_rel.pql.sql_expressions import _literal
from nori_rel.pql.tree_access import query_parts


@pytest.fixture
def schema():
    return SchemaGraph(
        database_id="test",
        schema_name="public",
        discovered_at=datetime.now(UTC),
        tables=[
            dict(
                name="users",
                primary_key=["id"],
                columns=[dict(name="id", data_type="TEXT", nullable=False)],
            ),
            dict(
                name="orders",
                time_column="event_time",
                columns=[
                    dict(name="amount", data_type="NUMERIC", nullable=True),
                    dict(name="event_time", data_type="TIMESTAMPTZ", nullable=False),
                ],
            ),
        ],
    )


def arguments(**changes):
    return (
        dict(
            entity="users",
            target="orders.amount",
            target_time="orders.event_time",
            operation="total",
            lookahead="30 days",
            entity_ids=[42],
        )
        | changes
    )


@pytest.mark.parametrize(
    "ids,suffix",
    [
        (None, "FOR EACH `users`.`id`"),
        ([42], "FOR `users`.`id`=42"),
        ([42, 43], "FOR `users`.`id` IN (42,43)"),
    ],
)
def test_exact_translation(schema, ids, suffix):
    text = aggregate_query(schema, **arguments(entity_ids=ids))
    assert text == "PREDICT SUM(`orders`.`amount`,0,30,days) " + suffix
    bind_columns(parse_pql(text), schema)


def test_string_ids_are_literals_not_query_syntax(schema):
    value = "user\"\\x' OR 1=1"
    tree = parse_pql(aggregate_query(schema, **arguments(entity_ids=[value])))
    assert _literal(query_parts(tree)["selection"].constant()) == value
    bind_columns(tree, schema)


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"operation": "next"}, "supports count"),
        ({"lookahead": "1 hour"}, "whole days"),
        ({"lookahead": "0 days"}, "whole days"),
        ({"lookahead": "1.5 days"}, "whole days"),
        ({"target": "orders.amount); DROP TABLE users"}, "simple"),
        ({"target": "orders.missing"}, "column is not registered"),
        ({"entity": "missing"}, "table is not registered"),
        ({"target_time": "orders.created_at"}, "timestamp registered"),
        ({"entity_ids": []}, "nonempty"),
        ({"entity_ids": [True]}, "integers or strings"),
        ({"entity_ids": [1.5]}, "integers or strings"),
        ({"entity_ids": ["\x00"]}, "NUL"),
    ],
)
def test_rejected_adapter_shapes(schema, changes, match):
    with pytest.raises(ValueError, match=match):
        aggregate_query(schema, **arguments(**changes))


def test_adapter_time_defaults_and_output_contract(schema, monkeypatch):
    now = datetime(2026, 3, 3, tzinfo=UTC)
    clock = Mock()
    clock.now.return_value = now
    monkeypatch.setattr("nori_rel.client.datetime", clock)
    runtime = Mock()
    runtime.features = "fastdfs"
    runtime.discover.return_value = schema
    frame = pd.DataFrame({"entity_id": [42], "anchor_time": [now], "prediction": [2.5]})
    runtime.predict_pql.return_value = frame
    client = NoriRelClient(mode="hosted", api_key="test-only", _runtime=runtime)
    options = dict(
        source=Mock(),
        anchor_times=[datetime(2026, 1, 1, tzinfo=UTC)],
        observed_through=now,
        entity_created_at="created_at",
        **arguments(),
    )
    for time_options, expected in [
        ({}, now),
        ({"as_of": now}, now),
        ({"anchor_time": now}, now),
    ]:
        with pytest.warns(FutureWarning) as warnings_raised:
            assert client.predict(**options, **time_options) is frame
        assert Path(warnings_raised[0].filename).resolve() == Path(__file__).resolve()
        assert runtime.predict_pql.call_args.kwargs["anchor_time"] == expected
    with pytest.raises(ValueError, match="not both"):
        client.predict(**options, as_of=now, anchor_time=now)


@pytest.mark.parametrize(
    "changes",
    [
        {"task": "classification"},
        {"positive_class": "yes"},
        {"relationship_path": ["users", "orders"]},
        {"decision_threshold": 0.5},
    ],
)
def test_unsupported_legacy_options_fail_before_discovery(changes):
    runtime = Mock()
    client = NoriRelClient(mode="hosted", api_key="test-only", _runtime=runtime)
    with pytest.raises(ValueError, match="aggregate regression only"):
        client.predict(source=Mock(), **arguments(), **changes)
    runtime.discover.assert_not_called()
    runtime.predict_pql.assert_not_called()
