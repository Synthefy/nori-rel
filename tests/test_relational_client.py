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
from nori_rel._feature_search import AgenticSearch
from nori_rel._pql_runtime import get_context_label_frame, get_scoring_feature_frame
from nori_rel.pipeline import (
    CONTEXT_LABELS_ATTR,
    SCORING_FEATURE_SQL_ATTR,
    SCORING_FEATURES_ATTR,
    PredictionDetails,
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
        self.options = options
        frame = pd.DataFrame({"entity_id": [7], "prediction": [3.5]})
        if options.get("keep_scoring_features"):
            frame.attrs[SCORING_FEATURES_ATTR] = pd.DataFrame(
                {"nr|direct|users.age": [41.0]}, index=pd.Index([7], name="entity_id")
            )
            frame.attrs[SCORING_FEATURE_SQL_ATTR] = {}
            frame.attrs[CONTEXT_LABELS_ATTR] = pd.DataFrame(
                {"entity_id": [7], "anchor_time": [None], "label": [2.0]}
            )
        return frame


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


def test_two_runs_with_one_agent_run_at_the_same_time() -> None:
    search = AgenticSearch(agent=object())
    runtime = BlockingRuntime(features=search)
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
        assert run.progress.strategy == "agentic"
        assert run.progress.max_trials == search.config.max_trials


def test_two_threads_can_predict_with_one_agent() -> None:
    search = AgenticSearch(agent=object())
    runtime = FakeRuntime()
    runtime.features = search
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


def test_a_replayed_program_reports_its_strategy() -> None:
    client = _client(BlockingRuntime(features=AgenticSearch(agent=object())))

    assert client._get_run_options_for_features(True) == {"strategy": "saved_program"}


def test_an_agent_search_brings_its_limits_to_the_run() -> None:
    from nori_rel.run import RunLimits

    limits = RunLimits(max_runtime_seconds=120, stop_grace_s=10)
    search = AgenticSearch(agent=object(), limits=limits)
    runtime = BlockingRuntime(features=search)
    client = _client(runtime)

    run = client.start_predict(source=runtime.source, **PQL_OPTIONS)
    runtime.release_predictions.set()
    run.result(timeout=5)

    assert run._context.limits is limits


def test_an_agent_search_reports_its_spend_limit_on_its_run() -> None:
    pytest.importorskip("duckdb")
    from nori_rel.feature.agentic.agent_claude import ClaudeProgramAgent
    from nori_rel.feature.agentic.usage import TokenPrices
    from nori_rel.run import RunLimits

    prices = TokenPrices(
        input=3.0, output=15.0, cache_read=0.3, cache_write_5m=3.75, cache_write_1h=6.0
    )
    search = AgenticSearch(
        agent=ClaudeProgramAgent(prices=prices),
        limits=RunLimits(max_spend_usd=2.5),
    )
    runtime = BlockingRuntime(features=search)
    client = _client(runtime)

    run = client.start_predict(source=runtime.source, **PQL_OPTIONS)
    runtime.release_predictions.set()
    run.result(timeout=5)

    assert run.progress.max_spend_usd == 2.5


def _predict(client, **options):
    return client.predict(
        source=client._runtime.source,
        query="PREDICT SUM(orders.amount,0,30,days) FOR EACH users.user_id",
        anchor_times=[datetime(2026, 1, 1, tzinfo=UTC)],
        observed_through=datetime(2026, 2, 1, tzinfo=UTC),
        entity_created_at="created_at",
        **options,
    )


def test_details_carry_the_scored_rows_features_and_frames_do_not() -> None:
    runtime = FakeRuntime()
    with _client(runtime) as client:
        frame = _predict(client)
        assert runtime.options["keep_scoring_features"] is False
        assert SCORING_FEATURES_ATTR not in frame.attrs
        details = _predict(client, return_details=True)
    assert runtime.options["keep_scoring_features"] is True
    assert details.features.loc[7, "users.age"] == 41.0
    assert details.internal_feature_names == ("nr|direct|users.age",)
    assert details.context["label"].tolist() == [2.0]
    for key in (SCORING_FEATURES_ATTR, SCORING_FEATURE_SQL_ATTR, CONTEXT_LABELS_ATTR):
        assert key not in details.predictions.attrs
        assert key not in details.metadata


def test_the_feature_table_maps_each_column_to_its_source_and_sql() -> None:
    windowed_mean = "nr|path|users%3C-orders.user_id|amount|mean|2592000000000000ns"
    features = pd.DataFrame(
        {
            "recent__qty_ly": [10.0, None],
            windowed_mean: [2.0, 4.0],
            "nr|anchor|cutoff_month": [9.0, 9.0],
        },
        index=pd.Index([1, 2], name="entity_id"),
    )
    details = PredictionDetails(
        predictions=pd.DataFrame(),
        metadata={
            "features": "agentic",
            "agentic_program": {"queries": [{"name": "recent", "sql": "SELECT 1"}]},
        },
        pipeline=None,
        features=features,
    )
    table = details.get_feature_table()
    assert table.columns.tolist() == [
        "feature",
        "internal_name",
        "source",
        "sql",
        "dtype",
        "missing_share",
        "mean",
        "min",
        "max",
    ]
    assert table["feature"].tolist() == details.features.columns.tolist()
    assert table["feature"].tolist() == [
        "recent.qty_ly",
        "orders.amount.mean_30d",
        "anchor.cutoff_month",
    ]
    assert table["internal_name"].tolist() == features.columns.tolist()
    assert table["source"].tolist() == ["agent", "noridfs", "noridfs"]
    assert table["sql"].tolist() == ["SELECT 1", None, None]
    assert table.loc[0, "missing_share"] == 0.5
    assert table.loc[1, ["mean", "min", "max"]].tolist() == [3.0, 2.0, 4.0]


def test_a_feature_table_needs_features() -> None:
    details = PredictionDetails(predictions=pd.DataFrame(), metadata={}, pipeline=None)
    with pytest.raises(ValueError, match="return_details=True"):
        details.get_feature_table()


def test_scoring_features_are_indexed_by_entity() -> None:
    query = pd.DataFrame({"x": [1.0, 2.0]}, index=pd.Index([1, 2], name="query_row_id"))
    scoring = pd.DataFrame({"query_row_id": [2, 1], "entity_id": ["b", "a"]})
    frame = get_scoring_feature_frame(query, scoring)
    assert frame.index.tolist() == ["a", "b"] and frame.index.name == "entity_id"
    assert frame["x"].tolist() == [1.0, 2.0]


def test_context_labels_are_the_values_nori_read() -> None:
    from decimal import Decimal

    context = pd.DataFrame(
        {
            "context_row_id": [11, 10],
            "entity_id": pd.Series([1, 2], dtype=object),
            "anchor_time": [datetime(2023, 4, 1), datetime(2023, 2, 1)],
            "label": [Decimal("4.5"), Decimal("9")],
        }
    )
    labels = pd.Series([9.0, 4.5], index=pd.Index([10, 11], name="context_row_id"))
    frame = get_context_label_frame(context, labels)
    assert frame.columns.tolist() == ["entity_id", "anchor_time", "label"]
    assert frame["entity_id"].tolist() == [1, 2]
    assert str(frame["anchor_time"].dt.tz) == "UTC"
    assert frame["label"].dtype == float and frame["label"].tolist() == [4.5, 9.0]


def test_predict_passes_its_own_feature_search_to_the_runtime() -> None:
    runtime = FakeRuntime()
    with _client(runtime) as client:
        _predict(client, feature_strategy="noridfs")
    assert runtime.options["feature_strategy"] == "noridfs"


def test_predict_without_a_feature_strategy_uses_the_default() -> None:
    runtime = FakeRuntime()
    with _client(runtime) as client:
        _predict(client)
    assert runtime.options["feature_strategy"] is None


def test_an_unknown_feature_search_fails_before_the_run_starts() -> None:
    runtime = FakeRuntime()
    with (
        _client(runtime) as client,
        pytest.raises(ValueError, match="features must be one of"),
    ):
        _predict(client, feature_strategy="unknown")
    assert runtime.prediction is None


def test_start_predict_takes_an_agent_search_for_one_run() -> None:
    from nori_rel.run import RunLimits

    limits = RunLimits(max_runtime_seconds=120, stop_grace_s=10)
    search = AgenticSearch(agent=object(), limits=limits)
    runtime = BlockingRuntime()
    client = _client(runtime)

    run = client.start_predict(
        source=runtime.source, feature_strategy=search, **PQL_OPTIONS
    )
    runtime.release_predictions.set()
    run.result(timeout=5)

    assert runtime.features == "fastdfs"
    assert runtime.options["feature_strategy"] is search
    assert run._context.limits is limits


def test_a_feature_strategy_cannot_be_passed_with_a_feature_program() -> None:
    runtime = FakeRuntime()
    with (
        _client(runtime) as client,
        pytest.raises(ValueError, match="cannot be passed with feature_program"),
    ):
        _predict(client, feature_strategy="fastdfs", feature_program={"queries": []})
    assert runtime.prediction is None
