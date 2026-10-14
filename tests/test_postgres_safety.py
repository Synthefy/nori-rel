"""PostgreSQL error and streaming boundaries without a database dependency."""

import traceback
from datetime import UTC, datetime
from unittest.mock import Mock

import pandas as pd
import pytest
from sqlalchemy import column, select
from sqlalchemy.exc import OperationalError

from nori_rel.connectors.errors import ConnectorError, SnapshotLimitError
from nori_rel.connectors.postgres import PostgreSQLConnector
from nori_rel.contracts import DatabaseCreateRequest, EnvironmentCredential
from nori_rel.limits import RelationalLimits


def _postgresql_connector():
    return PostgreSQLConnector(
        DatabaseCreateRequest(
            engine="postgresql", credential=EnvironmentCredential(variable="TEST_ONLY")
        ),
        RelationalLimits(),
    )


@pytest.mark.parametrize(
    "operation", ["test", "discover", "snapshot", "prepare_context"]
)
def test_vendor_errors_do_not_expose_parameters_in_tracebacks(monkeypatch, operation):
    reader = _postgresql_connector()
    engine = Mock()
    secret = "synthetic-sensitive-parameter-123"
    engine.connect.side_effect = OperationalError(
        "SELECT protected_data", {"private_value": secret}, Exception(secret)
    )
    monkeypatch.setattr(reader, "_engine", lambda: engine)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    with pytest.raises(ConnectorError) as caught:
        if operation == "test":
            reader.test()
        elif operation == "discover":
            reader.discover("db")
        elif operation == "snapshot":
            reader.snapshot("db", now)
        else:
            reader.prepare_context(
                "db",
                "PREDICT COUNT(orders.*,0,30,days) FOR EACH users.id",
                anchor_times=[now],
                anchor_time=now,
                observed_through=now,
                entity_created_at="created_at",
            )
    assert secret not in "".join(traceback.format_exception(caught.value))
    assert "protected_data" not in "".join(traceback.format_exception(caught.value))
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_source_reads_stream_and_enforce_remaining_budget(monkeypatch):
    reader = _postgresql_connector()
    batches = [pd.DataFrame({"id": [1, 2]}), pd.DataFrame({"id": [3, 4]})]
    consumed = []

    def read_sql(query, connection, **kwargs):
        assert query.get_execution_options()["stream_results"] is True
        assert query.get_execution_options()["max_row_buffer"] == 1000
        for batch in batches:
            consumed.append(batch)
            yield batch

    monkeypatch.setattr(pd, "read_sql", read_sql)
    with pytest.raises(SnapshotLimitError, match="remaining row limit"):
        reader._read_bounded(
            select(column("id")), object(), max_rows=1, max_bytes=10000
        )
    assert len(consumed) == 1
    with pytest.raises(SnapshotLimitError, match="memory limit"):
        reader._read_bounded(select(column("id")), object(), max_rows=100, max_bytes=1)
