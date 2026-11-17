"""Snowflake and Databricks writers: the statements they send and the values in them."""

import json
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal

import pandas as pd
import pytest

from nori_rel import (
    DatabricksConnection,
    EnvironmentCredential,
    PredictionDestination,
    PredictionWriteError,
    SnowflakeConnection,
)
from nori_rel.run import RunRecord
from nori_rel.writeback import write_predictions, writers
from nori_rel.writeback.rows import (
    Column,
    TableRows,
    get_prediction_rows,
    get_run_record_rows,
)

RUN_ID = "run_0123456789abcdef"
PIPELINE_ID = "pipe_0123456789abcdef"
AS_OF = datetime(2026, 2, 1, tzinfo=UTC)
WRITTEN_AT = datetime(2026, 2, 2, 3, 4, 5, 678901, tzinfo=UTC)


class FakeCursor:
    def __init__(self, connection):
        self.connection = connection
        self.rows = []

    def execute(self, sql, params=None):
        self.connection.calls.append((sql, params))
        self.rows = self.connection.respond(sql, params)

    def fetchall(self):
        return self.rows

    def close(self):
        pass


class FakeConnection:
    def __init__(self, respond=None):
        self.calls = []
        self.respond = respond or (lambda sql, params: [])

    def cursor(self):
        return FakeCursor(self)


class FakeConnector:
    def __init__(self, connection):
        self.connection = connection
        self.closed = False

    def _get_connection(self):
        return self.connection

    def close(self):
        self.closed = True


def _destination(engine):
    connection = (
        SnowflakeConnection(account="acme", warehouse="compute", database="ANALYTICS")
        if engine == "snowflake"
        else DatabricksConnection(
            server_hostname="example.cloud.databricks.com",
            http_path="/sql/1.0/warehouses/abc",
            catalog="main",
        )
    )
    return PredictionDestination(
        engine=engine,
        credential=EnvironmentCredential(variable="NORI_REL_WRITER"),
        schema_name="FORECASTS" if engine == "snowflake" else "forecasts",
        connection=connection,
    )


@pytest.fixture
def connection(monkeypatch):
    fake = FakeConnection()
    monkeypatch.setattr(
        "nori_rel.connectors.create_connector",
        lambda config, limits: FakeConnector(fake),
    )
    return fake


def _record(**fields):
    return RunRecord(
        run_id=RUN_ID,
        status="completed",
        started_at=AS_OF,
        finished_at=AS_OF,
        pipeline_id=PIPELINE_ID,
        **fields,
    )


def _frame(**columns):
    frame = pd.DataFrame(
        {
            "entity_id": [7, 42],
            "anchor_time": [AS_OF, AS_OF],
            "prediction": [12.5, 0.1],
            **columns,
        }
    )
    frame.attrs["run_record"] = _record().model_dump(mode="json")
    return frame


def _prediction_rows(frame=None):
    return get_prediction_rows(
        _frame() if frame is None else frame,
        table="spend",
        record=_record(),
        written_at=WRITTEN_AT,
    )


def test_snowflake_replaces_a_runs_rows_in_one_transaction(connection) -> None:
    writer = writers.open_table_writer(_destination("snowflake"))
    run_rows = get_run_record_rows(_record(), written_at=WRITTEN_AT)

    writer.replace_run_rows([_prediction_rows(), run_rows])

    statements = [sql for sql, _ in connection.calls]
    assert statements[0] == "BEGIN"
    assert statements[1] == (
        'DELETE FROM "ANALYTICS"."FORECASTS"."spend" WHERE "RUN_ID" = ?'
    )
    assert statements[2] == (
        'INSERT INTO "ANALYTICS"."FORECASTS"."spend" ("ENTITY_ID", "AS_OF", '
        '"PREDICTION", "RUN_ID", "PIPELINE_ID", "WRITTEN_AT") SELECT '
        "value[0]::VARCHAR::NUMBER(38,0), TO_TIMESTAMP_TZ(value[1]::VARCHAR), "
        "value[2]::VARCHAR::FLOAT, value[3]::VARCHAR, value[4]::VARCHAR, "
        "TO_TIMESTAMP_TZ(value[5]::VARCHAR) "
        "FROM TABLE(FLATTEN(INPUT => PARSE_JSON(?)))"
    )
    assert statements[3].startswith(
        'DELETE FROM "ANALYTICS"."FORECASTS"."NORI_REL_RUNS"'
    )
    assert "PARSE_JSON(value[9]::VARCHAR)" in statements[4]
    assert statements[-1] == "COMMIT"
    assert connection.calls[1][1] == [RUN_ID]
    assert json.loads(connection.calls[2][1][0]) == [
        [
            "7",
            "2026-02-01 00:00:00.000000+00:00",
            "12.5",
            RUN_ID,
            PIPELINE_ID,
            "2026-02-02 03:04:05.678901+00:00",
        ],
        [
            "42",
            "2026-02-01 00:00:00.000000+00:00",
            "0.1",
            RUN_ID,
            PIPELINE_ID,
            "2026-02-02 03:04:05.678901+00:00",
        ],
    ]


def test_snowflake_rolls_back_and_hides_the_driver_message(monkeypatch) -> None:
    class DriverError(Exception):
        errno = 2003
        sqlstate = "02000"

    def respond(sql, params):
        if sql.startswith("INSERT"):
            raise DriverError("token=secret-token; INSERT INTO ... rows 7, 42")
        return []

    failing = FakeConnection(respond)
    monkeypatch.setattr(
        "nori_rel.connectors.create_connector",
        lambda config, limits: FakeConnector(failing),
    )
    writer = writers.open_table_writer(_destination("snowflake"))

    with pytest.raises(PredictionWriteError) as raised:
        writer.replace_run_rows([_prediction_rows()])

    assert str(raised.value) == ("Snowflake write failed (code=2003, sqlstate=02000)")
    assert [sql for sql, _ in failing.calls][-1] == "ROLLBACK"
    assert "COMMIT" not in [sql for sql, _ in failing.calls]


def test_snowflake_creates_and_reads_tables_by_their_given_names(connection) -> None:
    connection.respond = lambda sql, params: (
        [("ENTITY_ID",), ("AS_OF",)] if "INFORMATION_SCHEMA" in sql else []
    )
    writer = writers.open_table_writer(_destination("snowflake"))

    assert writer.get_column_names("spend") == ["entity_id", "as_of"]
    writer.create_table("spend", [Column("entity_id", "integer")])
    assert writer.get_qualified_name("nori_rel_runs") == (
        "ANALYTICS.FORECASTS.NORI_REL_RUNS"
    )
    assert connection.calls[0][1] == ["FORECASTS", "spend"]
    assert connection.calls[1][0] == (
        'CREATE TABLE IF NOT EXISTS "ANALYTICS"."FORECASTS"."spend" '
        '("ENTITY_ID" NUMBER(38,0))'
    )


def test_databricks_replaces_a_runs_rows_with_one_statement(connection) -> None:
    writer = writers.open_table_writer(_destination("databricks"))

    writer.replace_run_rows([_prediction_rows()])

    [(sql, params)] = connection.calls
    assert sql == (
        "INSERT INTO `main`.`forecasts`.`spend` REPLACE WHERE "
        f"`run_id` = '{RUN_ID}' SELECT CAST(r[0] AS BIGINT), "
        "CAST(r[1] AS TIMESTAMP), CAST(r[2] AS DOUBLE), CAST(r[3] AS STRING), "
        "CAST(r[4] AS STRING), CAST(r[5] AS TIMESTAMP) FROM "
        "(SELECT explode(from_json(?, 'ARRAY<ARRAY<STRING>>')) AS r)"
    )
    assert [row[0] for row in json.loads(params[0])] == ["7", "42"]


def test_databricks_stages_a_write_bigger_than_one_parameter(
    connection, monkeypatch
) -> None:
    monkeypatch.setattr(writers, "MAX_PAYLOAD_BYTES", 150)
    writer = writers.open_table_writer(_destination("databricks"))

    writer.replace_run_rows([_prediction_rows()])

    statements = [sql for sql, _ in connection.calls]
    staging = statements[0].split()[2]
    assert statements[0].startswith(f"CREATE TABLE {staging} (`entity_id` BIGINT")
    assert staging.startswith("`main`.`forecasts`.`nori_rel_write_")
    assert [sql.split(" SELECT")[0] for sql in statements[1:3]] == [
        f"INSERT INTO {staging}"
    ] * 2
    assert statements[3] == (
        "INSERT INTO `main`.`forecasts`.`spend` REPLACE WHERE "
        f"`run_id` = '{RUN_ID}' SELECT `entity_id`, `as_of`, `prediction`, "
        f"`run_id`, `pipeline_id`, `written_at` FROM {staging}"
    )
    assert statements[4] == f"DROP TABLE IF EXISTS {staging}"


def test_databricks_drops_its_staging_table_when_a_write_fails(monkeypatch) -> None:
    def respond(sql, params):
        if "REPLACE WHERE" in sql:
            raise RuntimeError("DELTA_CONCURRENT_APPEND")
        return []

    failing = FakeConnection(respond)
    monkeypatch.setattr(
        "nori_rel.connectors.create_connector",
        lambda config, limits: FakeConnector(failing),
    )
    monkeypatch.setattr(writers, "MAX_PAYLOAD_BYTES", 150)
    writer = writers.open_table_writer(_destination("databricks"))

    with pytest.raises(PredictionWriteError, match=r"^Databricks write failed$"):
        writer.replace_run_rows([_prediction_rows()])
    assert failing.calls[-1][0].startswith("DROP TABLE IF EXISTS `main`")


def test_databricks_refuses_a_run_id_it_would_quote(connection) -> None:
    writer = writers.open_table_writer(_destination("databricks"))
    rows = TableRows("spend", "run_1' OR '1'='1", (Column("run_id", "text"),), ())

    with pytest.raises(ValueError, match="not a run id"):
        writer.replace_run_rows([rows])
    assert connection.calls == []


def test_payloads_keep_exact_values_and_stay_under_the_limit(monkeypatch) -> None:
    eastern = timezone(timedelta(hours=-5))
    rows = TableRows(
        "spend",
        RUN_ID,
        (
            Column("entity_id", "decimal"),
            Column("as_of", "timestamp"),
            Column("prediction", "boolean"),
            Column("quantiles", "json"),
            Column("pipeline_id", "text"),
        ),
        (
            (
                Decimal(2**70),
                datetime(2026, 1, 31, 19, tzinfo=eastern),
                True,
                [1.0, 2.5],
                None,
            ),
        )
        * 3,
    )

    [payload] = writers.get_json_payloads(rows)
    assert json.loads(payload)[0] == [
        str(2**70),
        "2026-02-01 00:00:00.000000+00:00",
        "true",
        "[1.0, 2.5]",
        None,
    ]
    limit = len(payload) - 1
    monkeypatch.setattr(writers, "MAX_PAYLOAD_BYTES", limit)
    chunks = list(writers.get_json_payloads(rows))
    assert [len(json.loads(chunk)) for chunk in chunks] == [2, 1]
    assert all(len(chunk) <= limit for chunk in chunks)


def test_prediction_rows_follow_the_frame_and_refuse_bad_values() -> None:
    wide = _frame()
    wide["entity_id"] = [2**63, 7]
    assert _prediction_rows(wide).columns[0] == Column("entity_id", "decimal")
    named = _frame()
    named["entity_id"] = ["C-7", "C-42"]
    assert _prediction_rows(named).rows[0][0] == "C-7"
    binary = _frame(probability=[0.9, 0.2])
    binary["prediction"] = [True, False]
    assert [column.name for column in _prediction_rows(binary).columns][:4] == [
        "entity_id",
        "as_of",
        "prediction",
        "probability",
    ]

    naive = _frame()
    naive["anchor_time"] = [datetime(2026, 2, 1), datetime(2026, 2, 1)]
    with pytest.raises(ValueError, match="timezone-aware"):
        _prediction_rows(naive)
    missing = _frame()
    missing.loc[0, "prediction"] = float("nan")
    with pytest.raises(ValueError, match="missing or infinite"):
        _prediction_rows(missing)


class FakeWriter:
    """A destination whose table has the written columns in another order."""

    def __init__(self, columns):
        self.columns = columns
        self.writes = []
        self.created = []
        self.closed = False

    def get_qualified_name(self, table):
        return f"forecasts.{table}"

    def get_column_names(self, table):
        return self.columns.get(table)

    def create_table(self, table, columns):
        self.created.append(table)
        self.columns[table] = [column.name for column in columns]

    def replace_run_rows(self, writes):
        self.writes.extend(writes)

    def close(self):
        self.closed = True


def test_rows_follow_an_existing_tables_column_order(monkeypatch) -> None:
    order = ["run_id", "prediction", "entity_id", "as_of", "pipeline_id", "written_at"]
    fake = FakeWriter({"spend": order})
    monkeypatch.setattr(
        "nori_rel.writeback.write.open_table_writer", lambda destination, limits: fake
    )

    record = write_predictions(
        _frame(), destination=_destination("databricks"), table="spend"
    )

    predictions, runs = fake.writes
    assert [column.name for column in predictions.columns] == order
    assert predictions.rows[0][:3] == (RUN_ID, 12.5, 7)
    assert fake.created == ["nori_rel_runs"]
    assert runs.rows[0][-1]["output_table"] == "forecasts.spend"
    assert (record.output_table, record.rows_written) == ("forecasts.spend", 2)
    assert fake.closed


def test_a_frame_without_a_completed_run_record_is_refused() -> None:
    frame = _frame()
    del frame.attrs["run_record"]
    with pytest.raises(ValueError, match="no completed run record"):
        write_predictions(frame, destination=_destination("databricks"), table="spend")
