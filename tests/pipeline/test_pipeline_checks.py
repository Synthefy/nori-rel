import importlib.metadata
import warnings
from datetime import UTC, datetime

import pytest
from pipeline_records import QUERY

from nori_rel.contracts import SchemaGraph
from nori_rel.pipeline import (
    PipelineContextError,
    PipelineFeatureError,
    PipelineSchemaError,
    PipelineTaskError,
    PipelineVersionWarning,
)
from nori_rel.pipeline.checks import (
    get_pipeline_query,
    get_schema_differences,
    require_expected_context,
    require_same_columns,
    require_same_schema,
    warn_if_other_nori_rel,
)


def _schema(manifest, **changes) -> SchemaGraph:
    tables = [table.model_copy(deep=True) for table in manifest.source.tables]
    for table in tables:
        for name, value in changes.get(table.name, {}).items():
            setattr(table, name, value)
    return SchemaGraph(
        database_id="db",
        schema_name=manifest.source.schema_name,
        tables=[table for table in tables if table.name not in changes.get("drop", ())],
        discovered_at=datetime(2026, 10, 8, tzinfo=UTC),
    )


def test_the_same_schema_passes(manifest) -> None:
    require_same_schema(manifest.source, "postgresql", _schema(manifest))


def test_schema_differences_name_each_change(manifest) -> None:
    orders = next(t for t in manifest.source.tables if t.name == "orders")
    columns = [c.model_copy(deep=True) for c in orders.columns]
    columns[2].data_type = "DATE"
    changed = _schema(
        manifest,
        orders={"columns": columns, "time_column": None, "primary_key": ["buyer_id"]},
    )

    assert get_schema_differences(manifest.source, "postgresql", changed) == [
        "column orders.event_time is DATE NULL, saved TIMESTAMP NULL",
        "primary key of orders is ['buyer_id'], saved ['id']",
        "time column of orders is None, saved 'event_time'",
    ]
    assert get_schema_differences(
        manifest.source, "postgresql", _schema(manifest, drop=("orders",))
    ) == ["table 'orders' is missing"]
    with pytest.raises(PipelineSchemaError, match="engine is snowflake"):
        require_same_schema(manifest.source, "snowflake", _schema(manifest))


def test_only_the_for_clause_may_change(manifest) -> None:
    other = "PREDICT COUNT(orders.*,0,30,days) FOR users.user_id IN (1, 2)"

    assert get_pipeline_query(manifest, None) == QUERY
    assert get_pipeline_query(manifest, other) == other
    with pytest.raises(PipelineTaskError, match="only the FOR clause"):
        get_pipeline_query(manifest, QUERY.replace("COUNT(orders.*", "COUNT(users.*"))


def test_feature_columns_must_keep_their_order(manifest) -> None:
    require_same_columns(manifest.columns, list(manifest.columns.names))
    with pytest.raises(PipelineFeatureError, match=r"missing \['nr\|path"):
        require_same_columns(manifest.columns, ["other"])


def test_no_context_hash_is_expected_unless_asked() -> None:
    require_expected_context(None, "a" * 64)
    require_expected_context("a" * 64, "a" * 64)
    with pytest.raises(PipelineContextError, match="history changed"):
        require_expected_context("b" * 64, "a" * 64)


def test_another_nori_rel_version_is_reported_not_refused(manifest) -> None:
    other = manifest.model_copy(update={"nori_rel_version": "0.0.1"})
    same = manifest.model_copy(
        update={"nori_rel_version": importlib.metadata.version("nori-rel")}
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_if_other_nori_rel(same)
    with pytest.warns(PipelineVersionWarning, match="saved with nori-rel 0.0.1"):
        warn_if_other_nori_rel(other)
