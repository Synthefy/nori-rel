import threading

import pandas as pd
import pytest

from nori_rel.run import LocalExecutor, RunContext, RunEvent, RunLimits, Stage
from nori_rel.run import executor as executor_module

#: Long enough for a worker thread to reach a point a test waits for.
WAIT_SECONDS = 5.0


class GatedPrediction:
    """A prediction that reports a stage and waits for a test inside it."""

    def __init__(self) -> None:
        self.reached_gate = threading.Event()
        self.open_gate = threading.Event()

    def __call__(self, run_context: RunContext) -> pd.DataFrame:
        with run_context.report_stage(Stage.MATERIALIZE):
            self.reached_gate.set()
            assert self.open_gate.wait(WAIT_SECONDS)
        return pd.DataFrame({"entity_id": [7], "prediction": [3.5]})


def test_a_started_run_returns_at_once_and_reports_on_its_thread() -> None:
    prediction = GatedPrediction()
    callback_thread_ids: set[int] = set()

    def record_thread(event: RunEvent) -> None:
        callback_thread_ids.add(threading.get_ident())

    run = LocalExecutor().start_run(
        prediction, strategy="fastdfs", on_event=record_thread
    )

    assert prediction.reached_gate.wait(WAIT_SECONDS)
    assert run.progress.status == "running"
    prediction.open_gate.set()
    frame = run.result(timeout=WAIT_SECONDS)
    assert frame.attrs["run"]["status"] == "completed"
    assert callback_thread_ids and threading.get_ident() not in callback_thread_ids


def test_a_run_on_this_thread_reports_here_and_raises_the_original() -> None:
    callback_thread_ids: set[int] = set()

    def record_thread(event: RunEvent) -> None:
        callback_thread_ids.add(threading.get_ident())

    def failing_prediction(run_context: RunContext) -> None:
        raise ValueError("bad request")

    executor = LocalExecutor()
    with pytest.raises(ValueError, match="bad request"):
        executor.run_on_this_thread(
            failing_prediction, strategy="fastdfs", on_event=record_thread
        )

    assert callback_thread_ids == {threading.get_ident()}


def test_runs_on_this_thread_are_not_kept() -> None:
    executor = LocalExecutor()

    run = executor.run_on_this_thread(lambda run_context: "done", strategy="fastdfs")

    assert run.result() == "done"
    with pytest.raises(KeyError, match="unknown run"):
        executor.get_run(run.run_id)


def test_runs_are_found_by_id_and_old_finished_runs_are_forgotten(
    monkeypatch,
) -> None:
    monkeypatch.setattr(executor_module, "_MAX_FINISHED_RUNS_KEPT", 1)
    executor = LocalExecutor()
    first = executor.start_run(lambda run_context: "done", strategy="fastdfs")
    first.result(timeout=WAIT_SECONDS)
    second = executor.start_run(lambda run_context: "done", strategy="fastdfs")
    second.result(timeout=WAIT_SECONDS)

    third = executor.start_run(lambda run_context: "done", strategy="fastdfs")

    assert executor.get_run(third.run_id) is third
    assert executor.get_run(second.run_id) is second
    with pytest.raises(KeyError, match="unknown run"):
        executor.get_run(first.run_id)


def test_running_runs_are_never_forgotten(monkeypatch) -> None:
    monkeypatch.setattr(executor_module, "_MAX_FINISHED_RUNS_KEPT", 0)
    executor = LocalExecutor()
    prediction = GatedPrediction()
    running = executor.start_run(prediction, strategy="fastdfs")
    assert prediction.reached_gate.wait(WAIT_SECONDS)

    executor.start_run(lambda run_context: "done", strategy="fastdfs")

    assert executor.get_run(running.run_id) is running
    prediction.open_gate.set()
    running.result(timeout=WAIT_SECONDS)


def test_waiting_for_unfinished_runs_returns_once_they_end() -> None:
    executor = LocalExecutor()
    prediction = GatedPrediction()
    run = executor.start_run(prediction, strategy="fastdfs")
    assert prediction.reached_gate.wait(WAIT_SECONDS)
    waiter = threading.Thread(target=executor.wait_for_unfinished_runs, daemon=True)
    waiter.start()

    waiter.join(0.05)
    assert waiter.is_alive()
    prediction.open_gate.set()
    waiter.join(WAIT_SECONDS)

    assert not waiter.is_alive()
    assert run.is_finished


def test_a_closed_executor_refuses_new_runs() -> None:
    executor = LocalExecutor()

    assert executor.close() == []
    with pytest.raises(RuntimeError, match="closed"):
        executor.start_run(lambda run_context: "done", strategy="fastdfs")


def test_a_run_whose_thread_cannot_start_is_not_kept(monkeypatch) -> None:
    def refuse_to_start(thread) -> None:
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(threading.Thread, "start", refuse_to_start)
    executor = LocalExecutor()

    with pytest.raises(RuntimeError, match="can't start new thread"):
        executor.start_run(lambda run_context: "done", strategy="fastdfs")
    assert executor.wait_for_unfinished_runs(timeout=0) == []


def test_waiting_with_a_timeout_returns_the_runs_still_going() -> None:
    executor = LocalExecutor()
    prediction = GatedPrediction()
    run = executor.start_run(prediction, strategy="fastdfs")
    assert prediction.reached_gate.wait(WAIT_SECONDS)

    assert executor.wait_for_unfinished_runs(timeout=0.05) == [run]
    prediction.open_gate.set()
    assert executor.wait_for_unfinished_runs(timeout=WAIT_SECONDS) == []


def test_waiting_cancels_a_paused_run() -> None:
    reached_boundary = threading.Event()

    def pause_at_the_next_boundary(run_context: RunContext) -> str:
        run_context.commands.submit_command("pause")
        reached_boundary.set()
        run_context.apply_pending_commands()
        return "done"

    executor = LocalExecutor()
    run = executor.start_run(pause_at_the_next_boundary, strategy="fastdfs")
    assert reached_boundary.wait(WAIT_SECONDS)

    assert executor.wait_for_unfinished_runs(timeout=WAIT_SECONDS) == []
    assert run.progress.status == "cancelled"


def test_a_run_on_this_thread_gets_its_limits() -> None:
    seen_limits: list[RunLimits] = []

    def record_limits(run_context: RunContext) -> str:
        seen_limits.append(run_context.limits)
        return "done"

    limits = RunLimits(max_runtime_seconds=30)
    LocalExecutor().run_on_this_thread(record_limits, strategy="fastdfs", limits=limits)

    assert seen_limits == [limits]


def test_an_error_leaving_the_worker_is_logged_with_the_run_id(
    monkeypatch, caplog
) -> None:
    def explode(*args, **kwargs) -> None:
        raise SystemExit("worker stopped")

    executor = LocalExecutor()
    monkeypatch.setattr(executor_module.LocalRun, "_execute_prediction", explode)
    run = executor.start_run(lambda run_context: "done", strategy="fastdfs")

    for _ in range(50):
        if any(run.run_id in record.getMessage() for record in caplog.records):
            break
        threading.Event().wait(0.02)
    assert any(run.run_id in record.getMessage() for record in caplog.records)
