from __future__ import annotations

import threading
from datetime import UTC, datetime

import pandas as pd
import pytest

from nori_rel import (
    AwsSecret,
    Database,
    DatabaseStatus,
    DatabaseUrl,
    NoriRelClient,
)


class FakeRuntime:
    def __init__(self) -> None:
        self.connected = None
        self.prediction = None
        self.source = Database(
            id="db-1",
            engine="postgresql",
            schema_name="public",
            status=DatabaseStatus.READY,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            updated_at=datetime(2026, 1, 1, tzinfo=UTC),
        )

    def close(self):
        self.closed = True

    def connect(self, config):
        self.connected = config
        return self.source

    def test_connection(self, source):
        return {"source": source}

    def discover(self, source):
        return {"source": source}

    def predict_pql(self, source, request, **options):
        self.prediction = (source, request)
        return pd.DataFrame({"entity_id": [7], "prediction": [3.5]})


def _client(runtime: FakeRuntime | None = None) -> NoriRelClient:
    return NoriRelClient(
        mode="hosted",
        api_key="hosted-model-key",
        _runtime=runtime or FakeRuntime(),
    )


def test_hosted_client_uses_the_synthefy_nori_api_key(monkeypatch) -> None:
    monkeypatch.delenv("SYNTHEFY_NORI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="SYNTHEFY_NORI_API_KEY"):
        NoriRelClient(mode="hosted", _runtime=FakeRuntime())

    monkeypatch.setenv("SYNTHEFY_NORI_API_KEY", "environment-key")
    client = NoriRelClient(mode="hosted", _runtime=FakeRuntime())

    assert "environment-key" not in repr(client)
    assert not hasattr(client, "base_url")


def test_connect_keeps_the_database_endpoint_on_the_source() -> None:
    runtime = FakeRuntime()
    client = _client(runtime)

    source = client.connect(
        engine="postgresql",
        database_url="postgresql://readonly:secret@db.internal/customer",
        tables=["customers"],
    )

    assert source is runtime.source
    assert isinstance(runtime.connected.credential, DatabaseUrl)
    assert "secret" not in repr(runtime.connected)


def test_connect_accepts_an_indirect_secret_reference() -> None:
    runtime = FakeRuntime()
    secret = AwsSecret(secret_id="production/postgres", region_name="us-east-1")

    _client(runtime).connect(engine="postgresql", credential=secret)

    assert runtime.connected.credential == secret


@pytest.mark.parametrize(
    ("database_url", "credential"),
    [(None, None), ("postgresql://db/name", AwsSecret(secret_id="db"))],
)
def test_connect_requires_one_database_credential(database_url, credential) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        _client().connect(
            engine="postgresql",
            database_url=database_url,
            credential=credential,
        )


@pytest.mark.parametrize(
    "extra,match",
    [
        ({}, "direct/NEXT"),
        (
            {
                "operation": "next",
                "target_time": "results.race_date",
                "lookahead": "30 days",
            },
            "direct/NEXT",
        ),
        ({"operation": "total"}, "provided together"),
        (
            {
                "operation": "total",
                "target_time": "results.race_date",
                "lookahead": "30 days",
            },
            "requires anchor_times",
        ),
    ],
)
def test_retired_and_incomplete_requests_fail_before_database_access(extra, match):
    runtime = FakeRuntime()
    with pytest.raises(ValueError, match=match):
        _client(runtime).predict(
            source=runtime.source, entity="drivers", target="results.position", **extra
        )
    assert runtime.prediction is None


def test_version_one_is_synchronous() -> None:
    assert not hasattr(_client(), "submit")


def test_context_manager_closes_runtime_on_error():
    runtime = FakeRuntime()
    with pytest.raises(ValueError, match="caller failed"), _client(runtime):
        raise ValueError("caller failed")
    assert runtime.closed


PQL_OPTIONS = dict(
    query="PREDICT COUNT(orders.*,0,30,days) FOR users.user_id=7",
    anchor_times=[datetime(2026, 1, 1, tzinfo=UTC)],
    observed_through=datetime(2026, 2, 1, tzinfo=UTC),
    entity_created_at="created_at",
)


def test_predict_reports_its_run_to_on_event_and_in_attrs() -> None:
    runtime = FakeRuntime()
    events = []
    callback_thread_ids = set()

    def record_event(event) -> None:
        events.append(event)
        callback_thread_ids.add(threading.get_ident())

    frame = _client(runtime).predict(
        source=runtime.source, on_event=record_event, **PQL_OPTIONS
    )

    assert [event.data.get("to") for event in events] == ["running", "completed"]
    assert events[0].message == "Run started: fastdfs search."
    assert threading.get_ident() in callback_thread_ids
    assert len(callback_thread_ids) == 1
    assert frame.attrs["run"]["status"] == "completed"
    assert frame.attrs["run"]["run_id"] == events[0].run_id


def test_predict_without_on_event_leaves_attrs_unchanged() -> None:
    runtime = FakeRuntime()

    frame = _client(runtime).predict(source=runtime.source, **PQL_OPTIONS)

    assert "run" not in frame.attrs


def test_start_predict_returns_a_run_the_client_can_find_again() -> None:
    runtime = FakeRuntime()
    client = _client(runtime)

    events = []
    run = client.start_predict(
        source=runtime.source, on_event=events.append, **PQL_OPTIONS
    )

    assert client.get_run(run.run_id) is run
    frame = run.result(timeout=5)
    assert frame["prediction"].tolist() == [3.5]
    assert frame.attrs["run"]["status"] == "completed"
    assert [event.data.get("to") for event in events] == ["running", "completed"]
    with pytest.raises(KeyError):
        client.get_run("run_unknown")


def test_start_predict_checks_its_request_before_returning() -> None:
    runtime = FakeRuntime()
    with pytest.raises(ValueError, match="requires anchor_times"):
        _client(runtime).start_predict(source=runtime.source, query="PREDICT x")
    with pytest.raises(TypeError, match="return_details"):
        _client(runtime).start_predict(
            source=runtime.source, return_details=True, **PQL_OPTIONS
        )
    assert runtime.prediction is None


def test_predict_raises_the_original_failure() -> None:
    class FailingRuntime(FakeRuntime):
        def predict_pql(self, source, request, **options):
            raise PermissionError("permission denied for table orders")

    runtime = FailingRuntime()
    with pytest.raises(PermissionError, match="orders"):
        _client(runtime).predict(source=runtime.source, **PQL_OPTIONS)


class BlockingRuntime(FakeRuntime):
    """A runtime whose predictions wait until a test releases them."""

    def __init__(self, features="fastdfs") -> None:
        super().__init__()
        self.features = features
        self.release_predictions = threading.Event()

    def predict_pql(self, source, request, **options):
        assert self.release_predictions.wait(5)
        return super().predict_pql(source, request, **options)


class PollingRuntime(BlockingRuntime):
    """A runtime that applies run commands while it waits, as a pipeline does."""

    def predict_pql(self, source, request, **options):
        run_context = options["run_context"]
        while not self.release_predictions.wait(0.01):
            run_context.apply_pending_commands()
        return FakeRuntime.predict_pql(self, source, request, **options)


def test_two_runs_with_one_client_run_at_the_same_time() -> None:
    runtime = BlockingRuntime()
    client = _client(runtime)
    started = threading.Barrier(2, timeout=5)

    def predict_pql_after_both_start(source, request, **options):
        started.wait()
        return FakeRuntime.predict_pql(runtime, source, request, **options)

    runtime.predict_pql = predict_pql_after_both_start
    first = client.start_predict(source=runtime.source, **PQL_OPTIONS)
    second = client.start_predict(source=runtime.source, **PQL_OPTIONS)

    for run in (first, second):
        assert run.result(timeout=5)["prediction"].tolist() == [3.5]
        assert run.progress.strategy == "fastdfs"


def test_two_threads_can_predict_with_one_client() -> None:
    runtime = FakeRuntime()
    client = _client(runtime)
    frames = []

    threads = [
        threading.Thread(
            target=lambda: frames.append(
                client.predict(source=runtime.source, **PQL_OPTIONS)
            )
        )
        for _ in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)

    assert [frame["prediction"].tolist() for frame in frames] == [[3.5], [3.5]]


def test_leaving_with_an_exception_cancels_runs_still_going() -> None:
    runtime = PollingRuntime()
    with pytest.raises(ValueError, match="caller failed"), _client(runtime) as client:
        run = client.start_predict(source=runtime.source, **PQL_OPTIONS)
        raise ValueError("caller failed")

    assert run.progress.status == "cancelled"
    assert runtime.closed


def test_close_cancels_runs_still_going_after_its_timeout() -> None:
    runtime = PollingRuntime()
    client = _client(runtime)
    run = client.start_predict(source=runtime.source, **PQL_OPTIONS)

    client.close(timeout_seconds=0.05)

    assert run.progress.status == "cancelled"
    with pytest.raises(RuntimeError, match="closed"):
        client.start_predict(source=runtime.source, **PQL_OPTIONS)


def test_start_predict_names_itself_for_an_unknown_argument() -> None:
    runtime = FakeRuntime()

    with pytest.raises(TypeError, match=r"start_predict\(\) got an unexpected"):
        _client(runtime).start_predict(
            source=runtime.source, anchor_tiems=[], **PQL_OPTIONS
        )


def test_a_failing_on_event_does_not_stop_predict() -> None:
    runtime = FakeRuntime()

    def failing_callback(event) -> None:
        raise RuntimeError("dashboard offline")

    frame = _client(runtime).predict(
        source=runtime.source, on_event=failing_callback, **PQL_OPTIONS
    )

    assert frame["prediction"].tolist() == [3.5]


def test_run_types_are_exported_from_the_package() -> None:
    import nori_rel

    for name in (
        "PredictionRun",
        "RunEvent",
        "PredictionFailed",
        "PredictionCancelled",
        "RunTimeout",
    ):
        assert name in nori_rel.__all__ and hasattr(nori_rel, name)


def test_close_waits_for_background_runs() -> None:
    runtime = BlockingRuntime()
    client = _client(runtime)
    run = client.start_predict(source=runtime.source, **PQL_OPTIONS)
    closer = threading.Thread(target=client.close)
    closer.start()

    closer.join(timeout=0.1)
    assert closer.is_alive()
    runtime.release_predictions.set()
    closer.join(timeout=5)

    assert run.progress.status == "completed"
    assert runtime.closed
