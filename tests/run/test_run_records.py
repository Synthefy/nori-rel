"""Run records: one per run whatever its outcome, kept in a store, without secrets."""

import json
import subprocess
import sys
import textwrap
import threading
from datetime import UTC, datetime

import pandas as pd
import pytest
from pydantic import ValidationError

from nori_rel.run import (
    JsonRunRecordStore,
    LocalExecutor,
    LocalRun,
    PredictionCancelled,
    PredictionFailed,
    RunContext,
    RunRecord,
    Stage,
    get_run_record_store,
    new_run_id,
)

WAIT_SECONDS = 5.0
PIPELINE_ID = "pipe_0123456789abcdef"
STARTED_AT = datetime(2026, 11, 17, 9, 0, tzinfo=UTC)


def _reporting_prediction(**fields):
    """Return a prediction that reports record fields, then returns one row."""

    def prediction(run_context: RunContext) -> pd.DataFrame:
        run_context.report_record_fields(**fields)
        with run_context.report_stage(Stage.PREDICT_BATCH):
            return pd.DataFrame({"entity_id": [7], "prediction": [3.5]})

    return prediction


def _execute(run: LocalRun, prediction, **overrides) -> LocalRun:
    """Execute ``prediction`` as ``run`` on this thread, as an executor does."""
    options = {
        "strategy": "fastdfs",
        "max_trials": None,
        "summary_in_result_attrs": False,
        "reraise_failure": False,
    }
    run._execute_prediction(prediction, **(options | overrides))
    return run


class FailingStore:
    """A store whose writes of some statuses fail, as a read-only directory would."""

    def __init__(self, *failing_statuses: str) -> None:
        self.failing_statuses = failing_statuses or (
            "running",
            "completed",
            "failed",
            "cancelled",
        )
        self.attempted_statuses: list[str] = []
        self.records: dict[str, RunRecord] = {}

    def write(self, record: RunRecord) -> None:
        self.attempted_statuses.append(record.status)
        if record.status in self.failing_statuses:
            raise PermissionError("the records directory is read-only")
        self.records[record.run_id] = record

    def read(self, run_id: str) -> RunRecord:
        return self.records[run_id]


def test_a_completed_run_writes_its_record_and_attaches_it(tmp_path) -> None:
    store = JsonRunRecordStore(tmp_path / "runs")
    run = LocalRun(new_run_id(), record_store=store)
    _execute(
        run,
        _reporting_prediction(
            pipeline_id=PIPELINE_ID,
            engine="postgresql",
            principal="nori_reader",
            context_rows_read=12,
            entities_scored=1,
        ),
    )

    frame = run.result()
    record = store.read(run.run_id)
    assert record == run.record == RunRecord.from_prediction(frame)
    assert frame.attrs["run_id"] == run.run_id
    assert (record.status, record.strategy, record.pipeline_id) == (
        "completed",
        "fastdfs",
        PIPELINE_ID,
    )
    assert (record.principal, record.context_rows_read, record.entities_scored) == (
        "nori_reader",
        12,
        1,
    )
    assert record.error_class is record.error_message is None
    assert record.nori_rel_version
    # The record is made before the final status, which stays the last event.
    final_event = list(run.events())[-1]
    assert record.started_at <= record.finished_at <= final_event.timestamp


def test_a_failed_run_records_its_error_class_and_what_it_read(tmp_path) -> None:
    class SchemaChanged(ValueError):
        pass

    def prediction(run_context: RunContext) -> pd.DataFrame:
        run_context.report_record_fields(
            pipeline_id=PIPELINE_ID, principal="nori_reader", context_rows_read=3
        )
        raise SchemaChanged("column users.tier is new; password=hunter2")

    store = JsonRunRecordStore(tmp_path)
    run = LocalRun(new_run_id(), record_store=store)
    with pytest.raises(SchemaChanged) as raised:
        _execute(run, prediction, reraise_failure=True)

    record = store.read(run.run_id)
    assert f"Nori-Rel run {run.run_id}" in raised.value.__notes__
    assert (record.status, record.error_class) == ("failed", "SchemaChanged")
    assert record.error_message == "column users.tier is new; password=***"
    assert (record.pipeline_id, record.context_rows_read) == (PIPELINE_ID, 3)
    assert record.entities_scored is None


def test_a_cancelled_run_writes_its_record(tmp_path) -> None:
    reached, release = threading.Event(), threading.Event()

    def prediction(run_context: RunContext) -> pd.DataFrame:
        run_context.report_record_fields(pipeline_id=PIPELINE_ID, as_of=STARTED_AT)
        reached.set()
        assert release.wait(WAIT_SECONDS)
        with run_context.report_stage(Stage.PREDICT_BATCH):
            return pd.DataFrame({"entity_id": [7], "prediction": [3.5]})

    store = JsonRunRecordStore(tmp_path)
    run = LocalRun(new_run_id(), record_store=store)
    threading.Thread(target=_execute, args=(run, prediction), daemon=True).start()
    assert reached.wait(WAIT_SECONDS)
    run.cancel()
    release.set()

    with pytest.raises(PredictionCancelled):
        run.result(timeout=WAIT_SECONDS)
    record = store.read(run.run_id)
    assert (record.status, record.error_class) == ("cancelled", None)
    assert (record.pipeline_id, record.as_of) == (PIPELINE_ID, STARTED_AT)


def test_a_run_writes_a_running_record_before_any_work(tmp_path) -> None:
    store = JsonRunRecordStore(tmp_path)
    stored_while_running: list[RunRecord] = []

    def prediction(run_context: RunContext) -> pd.DataFrame:
        stored_while_running.append(store.read(run_context.run_id))
        return pd.DataFrame({"entity_id": [7], "prediction": [3.5]})

    run = LocalRun(
        new_run_id(),
        record_store=store,
        record_fields={"pipeline_id": PIPELINE_ID, "as_of": STARTED_AT},
    )
    _execute(run, prediction)

    [started] = stored_while_running
    assert (started.status, started.finished_at) == ("running", None)
    assert (started.pipeline_id, started.as_of) == (PIPELINE_ID, STARTED_AT)
    assert started.strategy == "fastdfs"
    assert store.read(run.run_id).status == "completed"


def test_a_run_whose_process_is_killed_leaves_a_running_record(tmp_path) -> None:
    script = textwrap.dedent(
        """
        import sys
        import threading

        from nori_rel.run import JsonRunRecordStore, LocalExecutor

        def prediction(run_context):
            print("predicting", flush=True)
            threading.Event().wait()

        LocalExecutor(JsonRunRecordStore(sys.argv[1])).run_on_this_thread(
            prediction,
            strategy="noridfs",
            record_fields={"pipeline_id": sys.argv[2]},
        )
        """
    )
    child = subprocess.Popen(
        [sys.executable, "-c", script, str(tmp_path), PIPELINE_ID],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout.readline().strip() == "predicting"
    finally:
        child.kill()
        child.wait(WAIT_SECONDS)
        child.stdout.close()

    [path] = tmp_path.glob("run_*.json")
    record = RunRecord.model_validate_json(path.read_text())
    assert (record.status, record.finished_at) == ("running", None)
    assert (record.pipeline_id, record.strategy) == (PIPELINE_ID, "noridfs")


def test_a_run_that_cannot_write_its_start_record_does_no_work() -> None:
    store = FailingStore()
    calls: list[str] = []

    def prediction(run_context: RunContext) -> pd.DataFrame:
        calls.append(run_context.run_id)
        return pd.DataFrame({"entity_id": [7], "prediction": [3.5]})

    run = _execute(LocalRun(new_run_id(), record_store=store), prediction)

    with pytest.raises(PredictionFailed) as raised:
        run.result()
    assert raised.value.error_kind == "permission_error"
    assert calls == []
    assert store.attempted_statuses == ["running", "failed"]


def test_a_run_that_cannot_write_its_final_record_fails(caplog) -> None:
    store = FailingStore("completed")
    records_seen_at_the_error: list[RunRecord | None] = []

    def watch(event) -> None:
        if event.data["kind"] == "error":
            records_seen_at_the_error.append(run.record)

    run = LocalRun(new_run_id(), record_store=store, on_event=watch)
    _execute(run, _reporting_prediction())

    with pytest.raises(PredictionFailed) as raised:
        run.result()
    assert raised.value.error_kind == "permission_error"
    # The completed record was never written, so the run never showed it.
    assert records_seen_at_the_error == [None]
    assert store.attempted_statuses == ["running", "completed", "failed"]
    assert store.read(run.run_id) == run.record
    assert (run.record.status, run.record.error_class) == ("failed", "PermissionError")


def test_a_failed_run_keeps_its_own_error_when_its_record_cannot_be_written(
    caplog,
) -> None:
    def prediction(run_context: RunContext) -> pd.DataFrame:
        raise LookupError("no table named orders")

    store = FailingStore("failed")
    run = LocalRun(new_run_id(), record_store=store)
    with pytest.raises(LookupError):
        _execute(run, prediction, reraise_failure=True)

    assert store.attempted_statuses == ["running", "failed"]
    assert run.progress.error_kind == "lookup_error"
    assert run.record.error_class == "LookupError"
    assert "could not keep its record" in caplog.text


def test_a_run_without_a_store_still_builds_its_record() -> None:
    run = _execute(LocalRun(new_run_id()), _reporting_prediction(model="nori-30m"))

    frame = run.result()
    assert RunRecord.from_prediction(frame) == run.record
    assert run.record.model == "nori-30m"


def test_a_long_service_principal_is_kept_as_the_database_named_it() -> None:
    principal = "svc-nori-rel-scheduled-predictions-prod-01"
    run = _execute(LocalRun(new_run_id()), _reporting_prediction(principal=principal))

    assert run.record.principal == principal


def test_values_the_record_cannot_hold_are_left_out_not_fatal(caplog) -> None:
    run = _execute(
        LocalRun(new_run_id()),
        _reporting_prediction(
            principal=42, as_of=datetime(2026, 1, 1), model="nori-30m"
        ),
    )

    assert run.result()["prediction"].tolist() == [3.5]
    assert (run.record.status, run.record.model) == ("completed", "nori-30m")
    assert run.record.principal is run.record.as_of is None
    assert "left ['as_of', 'principal'] out of its record" in caplog.text


def test_runs_started_together_write_separate_records(tmp_path) -> None:
    executor = LocalExecutor(JsonRunRecordStore(tmp_path))
    runs = [
        executor.start_run(_reporting_prediction(), strategy="noridfs")
        for _ in range(4)
    ]
    for run in runs:
        run.result(timeout=WAIT_SECONDS)

    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(
        f"{run.run_id}.json" for run in runs
    )


def test_reported_fields_are_checked_and_never_cleared() -> None:
    context = RunContext.unobserved()
    for owned_by_the_run in ("status", "strategy", "nori_rel_version", "run_id"):
        with pytest.raises(ValueError, match=owned_by_the_run):
            context.report_record_fields(**{owned_by_the_run: "x"})
    with pytest.raises(ValueError, match="checkpoint_hash"):
        context.report_record_fields(checkpoint_hash="a" * 64)

    context.report_record_fields(model="nori-30m", checkpoint_sha256="a" * 64)
    context.report_record_fields(model="ft-churn", checkpoint_sha256=None)

    assert context.record_fields == {"model": "ft-churn", "checkpoint_sha256": "a" * 64}


def test_the_json_store_round_trips_a_record(tmp_path) -> None:
    store = JsonRunRecordStore(tmp_path / "nested" / "runs")
    record = RunRecord(
        run_id="run_0123456789abcdef",
        status="completed",
        started_at=STARTED_AT,
        finished_at=STARTED_AT,
        snapshot={
            "engine": "snowflake",
            "extraction_timestamp": STARTED_AT,
            "table_references": {"orders": '"DB"."S"."ORDERS" AT (TIMESTAMP => …)'},
        },
    )
    store.write(record)

    path = store.get_path(record.run_id)
    assert [child.name for child in store.directory.iterdir()] == [path.name]
    saved = json.loads(path.read_text())
    assert (saved["format"], saved["format_version"]) == ("nori-rel.run-record", 1)
    assert store.read(record.run_id) == record
    with pytest.raises(KeyError, match="run_ffffffffffffffff"):
        store.read("run_ffffffffffffffff")


@pytest.mark.parametrize("run_id", ["../escape", "runs/run_1", ".hidden", ""])
def test_the_json_store_takes_only_run_ids_as_file_names(tmp_path, run_id) -> None:
    with pytest.raises(ValueError, match="not a run id"):
        JsonRunRecordStore(tmp_path).get_path(run_id)


def test_a_record_has_its_outcome_fields_exactly_when_it_needs_them() -> None:
    times = {"started_at": STARTED_AT, "finished_at": STARTED_AT}
    with pytest.raises(ValidationError, match="error_class"):
        RunRecord(run_id="run_1", status="failed", **times)
    with pytest.raises(ValidationError, match="error_class"):
        RunRecord(run_id="run_1", status="cancelled", error_class="Exception", **times)
    with pytest.raises(ValidationError, match="finished_at"):
        RunRecord(run_id="run_1", status="completed", started_at=STARTED_AT)
    with pytest.raises(ValidationError, match="finished_at"):
        RunRecord(run_id="run_1", status="running", **times)
    with pytest.raises(ValidationError, match="credential"):
        RunRecord(run_id="run_1", status="completed", credential="x", **times)


def test_run_records_takes_a_directory_or_a_store(tmp_path) -> None:
    store = FailingStore()
    assert get_run_record_store(None) is None
    assert get_run_record_store(store) is store
    for directory in (tmp_path / "a", str(tmp_path / "b")):
        assert isinstance(get_run_record_store(directory), JsonRunRecordStore)
    with pytest.raises(TypeError, match="run_records"):
        get_run_record_store(42)
