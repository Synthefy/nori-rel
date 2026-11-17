"""Write predictions and run records back to PostgreSQL with a writer credential."""

import json
import os
from datetime import UTC, datetime
from uuid import uuid4

import psycopg
import pytest
from sqlalchemy.exc import InternalError

from nori_rel import (
    EnvironmentCredential,
    NoriRelClient,
    PredictionDestination,
    PredictionWriteError,
    TableRunRecordStore,
)
from nori_rel.run import JsonRunRecordStore

QUERY = (
    "PREDICT COUNT(orders.*,0,30,days) FOR users.user_id IN (7, 42, 99) "
    "WHERE users.country='US'"
)
OPTIONS = dict(
    anchor_times=[datetime(2026, 1, 1, tzinfo=UTC)],
    anchor_time=datetime(2026, 2, 1, tzinfo=UTC),
    observed_through=datetime(2026, 2, 1, tzinfo=UTC),
    entity_created_at="created_at",
)
WRITER_ENV = "NORI_REL_TEST_DATABASE_URL"


@pytest.fixture
def output_schema(source):
    """An empty schema the writer credential owns, dropped afterwards."""
    _, db = source
    name = "predictions_" + uuid4().hex
    db.execute(psycopg.sql.SQL("CREATE SCHEMA {}").format(psycopg.sql.Identifier(name)))
    try:
        yield name, db
    finally:
        db.execute(
            psycopg.sql.SQL("DROP SCHEMA {} CASCADE").format(
                psycopg.sql.Identifier(name)
            )
        )


@pytest.fixture
def client(runtime_source, tmp_path):
    runtime, database, _ = runtime_source
    client = NoriRelClient(
        mode="hosted",
        api_key="test-only",
        run_records=tmp_path / "runs",
        _runtime=runtime,
    )
    return client, database


def _destination(schema: str, variable: str = WRITER_ENV) -> PredictionDestination:
    return PredictionDestination(
        engine="postgresql",
        credential=EnvironmentCredential(variable=variable),
        schema_name=schema,
    )


def _rows(db, schema, table, order="run_id, entity_id"):
    return db.execute(
        psycopg.sql.SQL("SELECT * FROM {}.{} ORDER BY " + order).format(
            psycopg.sql.Identifier(schema), psycopg.sql.Identifier(table)
        )
    ).fetchall()


def test_predictions_land_once_per_run_with_their_record(client, output_schema):
    client, database = client
    schema, db = output_schema
    frame = client.predict(source=database, query=QUERY, **OPTIONS)
    destination = _destination(schema)

    record = client.write_predictions(
        frame, destination=destination, table="spend_forecast", create_table=True
    )
    again = client.write_predictions(
        frame, destination=destination, table="spend_forecast"
    )

    rows = _rows(db, schema, "spend_forecast")
    ordered = frame.sort_values("entity_id")
    assert [row[0] for row in rows] == ordered["entity_id"].tolist()
    assert {row[1] for row in rows} == {OPTIONS["anchor_time"]}
    assert [row[2] for row in rows] == ordered["prediction"].tolist()
    assert {(row[3], row[4]) for row in rows} == {
        (frame.attrs["run_id"], frame.attrs["pipeline_id"])
    }
    assert (record.output_table, record.rows_written) == (
        f"{schema}.spend_forecast",
        len(frame),
    )
    assert again.rows_written == record.rows_written
    runs = _rows(db, schema, "nori_rel_runs", order="run_id")
    assert len(runs) == 1
    stored = runs[0][-1]
    assert stored["output_table"] == f"{schema}.spend_forecast"
    assert stored["snapshot"] == frame.attrs.get("snapshot")
    assert client.run_records.read(record.run_id).rows_written == len(frame)
    assert frame.attrs["run_record"]["output_table"] == record.output_table


def test_each_run_keeps_its_own_rows(client, output_schema):
    client, database = client
    schema, db = output_schema
    destination = _destination(schema)
    first = client.predict(source=database, query=QUERY, **OPTIONS)
    second = client.predict(source=database, query=QUERY, **OPTIONS)

    for frame in (first, second, first):
        client.write_predictions(
            frame, destination=destination, table="spend_forecast", create_table=True
        )

    rows = _rows(db, schema, "spend_forecast")
    assert sorted({row[3] for row in rows}) == sorted(
        {first.attrs["run_id"], second.attrs["run_id"]}
    )
    assert len(rows) == len(first) + len(second)
    assert len(_rows(db, schema, "nori_rel_runs", order="run_id")) == 2


def test_binary_and_quantile_predictions_keep_their_output(client, output_schema):
    client, database = client
    schema, db = output_schema
    destination = _destination(schema)
    binary = client.predict(
        source=database, query=QUERY.replace(") FOR", ")>0 FOR"), **OPTIONS
    )
    quantiles = client.predict(
        source=database,
        query=QUERY,
        **OPTIONS,
        output_type="quantiles",
        quantiles=[0.1, 0.5, 0.9],
    )

    client.write_predictions(
        binary, destination=destination, table="churn", create_table=True
    )
    client.write_predictions(
        quantiles, destination=destination, table="spend_bands", create_table=True
    )

    ordered = binary.sort_values("entity_id")
    assert [row[2:4] for row in _rows(db, schema, "churn")] == list(
        zip(ordered["prediction"], ordered["probability"], strict=True)
    )
    assert [row[3] for row in _rows(db, schema, "spend_bands")] == [
        [1.0, 2.0, 3.0]
    ] * len(quantiles)


def test_a_missing_table_or_other_columns_are_refused(client, output_schema):
    client, database = client
    schema, db = output_schema
    frame = client.predict(source=database, query=QUERY, **OPTIONS)
    destination = _destination(schema)

    with pytest.raises(ValueError, match="does not exist; pass create_table=True"):
        client.write_predictions(frame, destination=destination, table="forecast")
    db.execute(
        psycopg.sql.SQL(
            "CREATE TABLE {}.forecast (entity_id bigint, score float)"
        ).format(psycopg.sql.Identifier(schema))
    )
    with pytest.raises(
        ValueError, match=r"missing \['as_of', 'prediction'.*unexpected \['score'\]"
    ):
        client.write_predictions(frame, destination=destination, table="forecast")
    with pytest.raises(ValueError, match="holds run records"):
        client.write_predictions(frame, destination=destination, table="nori_rel_runs")


def test_the_source_connection_still_refuses_writes(source, output_schema):
    connector, _ = source
    schema, _ = output_schema

    with (
        pytest.raises(InternalError, match="read-only transaction"),
        connector._engine().begin() as connection,
    ):
        connection.exec_driver_sql(f'CREATE TABLE "{schema}".probe (id int)')


def test_a_writer_error_never_shows_the_credential(client, output_schema, monkeypatch):
    client, database = client
    schema, _ = output_schema
    frame = client.predict(source=database, query=QUERY, **OPTIONS)
    url = os.environ[WRITER_ENV]
    wrong = url.replace("://postgres:integration@", "://postgres:not-the-password-41@")
    if wrong == url:
        pytest.skip("the test URL does not have the expected credentials")
    monkeypatch.setenv("NORI_REL_TEST_WRONG_WRITER", wrong)

    with pytest.raises(PredictionWriteError) as raised:
        client.write_predictions(
            frame,
            destination=_destination(schema, "NORI_REL_TEST_WRONG_WRITER"),
            table="spend_forecast",
            create_table=True,
        )
    assert "not-the-password-41" not in str(raised.value)
    assert raised.value.__cause__ is None


def test_a_table_store_keeps_every_run_record(runtime_source, output_schema):
    runtime, database, _ = runtime_source
    schema, db = output_schema
    store = TableRunRecordStore(_destination(schema))
    client = NoriRelClient(
        mode="hosted", api_key="test-only", run_records=store, _runtime=runtime
    )

    frame = client.predict(source=database, query=QUERY, **OPTIONS)
    with pytest.raises(ValueError, match="requires anchor_times") as refused:
        client.predict(source=database, query=QUERY)

    completed = store.read(frame.attrs["run_id"])
    assert completed.status == "completed"
    assert completed.entities_scored == len(frame)
    refused_id = refused.value.__notes__[-1].removeprefix("Nori-Rel run ")
    failed = store.read(refused_id)
    assert (failed.status, failed.error_class) == ("failed", "ValueError")
    stored = {
        row[0]: (row[2], json.dumps(row[-1]))
        for row in _rows(db, schema, "nori_rel_runs", order="run_id")
    }
    assert {status for status, _ in stored.values()} == {"completed", "failed"}
    with pytest.raises(KeyError, match="run_ffffffffffffffff"):
        store.read("run_ffffffffffffffff")
    assert "integration" not in repr(store)
    assert isinstance(client.run_records, TableRunRecordStore)
    assert not isinstance(client.run_records, JsonRunRecordStore)
