"""Feature-backend selection for the database client, without a database."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nori_rel import AgenticSearch, NoriRelClient
from nori_rel._feature_program import compute_schema_digest
from nori_rel._feature_search import (
    DEFAULT_SEARCH_CONFIG,
    NORIDFS_CONFIG,
    ClientProgramScorer,
    agentic_summary,
    program_from_record,
    program_record,
)
from nori_rel._runtime import (
    HostedNoriPredictor,
    LocalNoriPredictor,
    LocalRelationalRuntime,
)
from nori_rel.contracts import PredictionTask, RelationalOutputType, SchemaGraph
from nori_rel.feature.agentic import (
    AgenticFeatureConfig,
    AgenticFeatures,
    FeatureProgram,
    FeatureQuery,
)
from nori_rel.feature.agentic.usage import AgentRunResult, TokenUsage
from nori_rel.feature.dfs.fastdfs import FastDFS
from nori_rel.feature.dfs.noridfs import NativeFeatureConfig, NoriDFS
from nori_rel.pql.window import TargetWindow
from nori_rel.run import RunLimits


class _Agent:
    def run(self, session, task_hint):
        raise AssertionError("not searched in these tests")


class _Predictor:
    """Records each Nori call and answers with fixed values."""

    def __init__(self, values):
        self.values = values
        self.calls = []

    def model_for(self, task):
        return "nori-30m"

    def predict_arrays(self, train, target, query, **options):
        self.calls.append((train, target, query, options))
        return list(self.values[: len(query)]), None


def _runtime(**options) -> LocalRelationalRuntime:
    return LocalRelationalRuntime(
        api_key="test-only",
        model="nori-30m",
        classification_model=None,
        timeout=10,
        max_retries=0,
        **options,
    )


def _context(anchor_count: int) -> pd.DataFrame:
    times = pd.date_range("2026-01-01", periods=anchor_count, freq="MS", tz="UTC")
    return pd.DataFrame({"anchor_time": np.repeat(times, 2), "label": 1.0})


def test_fastdfs_stays_the_default_backend() -> None:
    runtime = _runtime()
    backend = runtime.feature_backend(
        TargetWindow(0, 30), PredictionTask.REGRESSION, _context(1), pd.Series()
    )
    assert runtime.features == "fastdfs"
    assert isinstance(backend, FastDFS)


def test_a_prediction_can_choose_its_own_feature_search() -> None:
    runtime = _runtime()
    backend = runtime.feature_backend(
        TargetWindow(0, 30),
        PredictionTask.REGRESSION,
        _context(1),
        pd.Series(),
        feature_strategy="noridfs",
    )
    assert isinstance(backend, NoriDFS)
    assert runtime.features == "fastdfs"


def test_noridfs_windows_scale_with_the_label_interval() -> None:
    backend = _runtime(features="noridfs").feature_backend(
        TargetWindow(30, 60), PredictionTask.REGRESSION, _context(1), pd.Series()
    )
    assert isinstance(backend, NoriDFS)
    assert backend.config == NORIDFS_CONFIG
    assert backend.horizon == pd.Timedelta(days=30)


def test_unknown_backends_are_refused_and_custom_backends_are_supported() -> None:
    with pytest.raises(ValueError, match="features must be one of"):
        _runtime(features="kumo")
    custom = FastDFS(max_depth=1)
    runtime = _runtime(features=custom)
    assert (
        runtime.feature_backend(
            TargetWindow(0, 30), PredictionTask.REGRESSION, _context(1), pd.Series()
        )
        is custom
    )


def test_an_agent_search_extends_noridfs_with_context_labels() -> None:
    labels = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=list("abcdef"))
    runtime = _runtime(features=AgenticSearch(agent=_Agent(), task_hint="orders"))
    backend = runtime.feature_backend(
        TargetWindow(0, 7), PredictionTask.CLASSIFICATION, _context(3), labels
    )
    assert isinstance(backend, AgenticFeatures)
    assert isinstance(backend.fallback, NoriDFS)
    assert backend.fallback.horizon == pd.Timedelta(days=7)
    assert backend.config == DEFAULT_SEARCH_CONFIG
    assert backend.task_hint == "orders"
    assert backend.train_target.tolist() == labels.tolist()
    assert backend.train_target.index.tolist() == list(range(6))
    assert backend.scorer == ClientProgramScorer(runtime.predictor, task="binary")


def test_the_session_owns_anchor_validation() -> None:
    search = AgenticSearch(
        agent=_Agent(), config=replace(DEFAULT_SEARCH_CONFIG, align_to_cutoffs=False)
    )
    backend = _runtime(features=search).feature_backend(
        TargetWindow(0, 30),
        PredictionTask.REGRESSION,
        _context(1),
        pd.Series([1.0, 1.0]),
    )
    assert isinstance(backend, AgenticFeatures)
    assert not backend.config.align_to_cutoffs


def test_default_search_has_explicit_budgets() -> None:
    assert DEFAULT_SEARCH_CONFIG.extend_baseline
    assert DEFAULT_SEARCH_CONFIG.align_to_cutoffs
    assert DEFAULT_SEARCH_CONFIG.max_trials == 24
    assert DEFAULT_SEARCH_CONFIG.n_folds == 1
    assert DEFAULT_SEARCH_CONFIG.min_relative_gain == 0.0
    assert DEFAULT_SEARCH_CONFIG.noise_margin == 0.0
    assert DEFAULT_SEARCH_CONFIG.min_context_cutoffs == 3
    assert not DEFAULT_SEARCH_CONFIG.allow_target_transform
    assert not DEFAULT_SEARCH_CONFIG.allow_text


@pytest.mark.parametrize(
    "config,match",
    [
        (AgenticFeatureConfig(allow_text=False), "extends NoriDFS"),
        (AgenticFeatureConfig(extend_baseline=True), "no text columns"),
        (
            AgenticFeatureConfig(
                extend_baseline=True, allow_text=False, allow_target_transform=True
            ),
            "target transform",
        ),
    ],
)
def test_searches_the_client_path_cannot_carry_are_refused(config, match) -> None:
    with pytest.raises(ValueError, match=match):
        AgenticSearch(agent=_Agent(), config=config)


def test_the_claude_factory_uses_default_agent_settings(monkeypatch) -> None:
    pytest.importorskip("anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    search = AgenticSearch.claude()
    agent = search.agent
    assert agent.model == "claude-sonnet-5-5"
    assert agent.effort == "high"
    assert agent.min_trials == 12
    assert agent.max_turns == 80
    assert agent.cache_ttl == "1h"
    assert search.config == DEFAULT_SEARCH_CONFIG


@pytest.mark.parametrize(
    "factory,variable,options",
    [
        ("claude", "ANTHROPIC_API_KEY", {}),
        ("openai", "OPENAI_API_KEY", {"model": "gpt-test"}),
    ],
)
def test_a_factory_without_an_api_key_fails_when_configured(
    monkeypatch, factory, variable, options
) -> None:
    pytest.importorskip("anthropic" if factory == "claude" else "openai")
    monkeypatch.delenv(variable, raising=False)

    with pytest.raises(ValueError, match=f"pass api_key= or set {variable}"):
        getattr(AgenticSearch, factory)(**options)
    with pytest.raises(ValueError, match=variable):
        getattr(AgenticSearch, factory)(api_key="  ", **options)


def test_the_claude_factory_takes_any_anthropic_model(monkeypatch) -> None:
    pytest.importorskip("anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    search = AgenticSearch.claude(model="claude-opus-5-5", effort="medium")
    assert search.agent.model == "claude-opus-5-5"
    assert search.agent.effort == "medium"


def test_the_openai_factory_uses_the_named_model_and_default_budgets(
    monkeypatch,
) -> None:
    pytest.importorskip("openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    search = AgenticSearch.openai(
        model="qwen3-coder", base_url="https://models.example/v1", api_key="test"
    )
    agent = search.agent
    assert agent.model == "qwen3-coder"
    assert agent.base_url == "https://models.example/v1"
    assert agent.token_limit_field == "max_tokens"
    assert agent.min_trials == 12
    assert agent.max_turns == 80
    assert search.config == DEFAULT_SEARCH_CONFIG


def test_the_openai_factory_caps_openai_models_with_max_completion_tokens(
    monkeypatch,
) -> None:
    pytest.importorskip("openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    search = AgenticSearch.openai(model="gpt-5", api_key="test")
    assert search.agent.token_limit_field == "max_completion_tokens"


def _hide_modules(monkeypatch, *hidden: str) -> None:
    """Make ``find_spec`` report ``hidden`` modules as not installed."""
    find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *args: None if name in hidden else find_spec(name, *args),
    )


def test_a_search_without_duckdb_fails_when_configured(monkeypatch) -> None:
    _hide_modules(monkeypatch, "duckdb")
    with pytest.raises(ImportError, match=r"duckdb.*nori-rel\[database,agentic\]"):
        AgenticSearch(agent=_Agent())


@pytest.mark.parametrize(
    "factory,module",
    [
        (lambda: AgenticSearch.claude(), "anthropic"),
        (lambda: AgenticSearch.openai(model="qwen3-coder"), "openai"),
    ],
)
def test_a_factory_without_its_provider_sdk_fails_when_called(
    monkeypatch, factory, module
) -> None:
    _hide_modules(monkeypatch, module)
    with pytest.raises(ImportError, match=rf"{module}.*nori-rel\[database,agentic\]"):
        factory()


def test_regression_programs_score_negative_mae_on_the_median() -> None:
    predictor = _Predictor([1.0, 4.0])
    scorer = ClientProgramScorer(predictor)
    context = pd.DataFrame({"x": [1.0, 2.0], "mode": ["a", "b"]}, index=[5, 9])
    holdout = pd.DataFrame({"x": [3.0, 4.0], "mode": ["a", "a"]}, index=[2, 7])
    report = scorer(
        context,
        pd.Series([1.0, 2.0], index=[5, 9]),
        holdout,
        pd.Series([2.0, 2.0], index=[2, 7]),
        categorical_columns=["mode"],
    )
    assert report.metric == "neg_mae"
    assert report.score == pytest.approx(-1.5)
    assert report.noise == 0.0
    train, target, query, options = predictor.calls[0]
    assert train.index.tolist() == [0, 1] and query.index.tolist() == [0, 1]
    assert target.index.tolist() == [0, 1]
    assert options == {
        "task": PredictionTask.REGRESSION,
        "output_type": RelationalOutputType.MEDIAN,
        "categorical_columns": ["mode"],
    }


def test_binary_programs_score_roc_auc_on_the_probability() -> None:
    predictor = _Predictor([0.9, 0.1, 0.8, 0.3])
    scorer = ClientProgramScorer(predictor, task="binary")
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    report = scorer(
        frame,
        pd.Series([1.0, 0.0, 1.0, 0.0]),
        frame,
        pd.Series([1.0, 0.0, 1.0, 0.0]),
        categorical_columns=[],
    )
    assert report.metric == "roc_auc"
    assert report.score == 1.0
    assert predictor.calls[0][3]["task"] is PredictionTask.CLASSIFICATION
    assert predictor.calls[0][3]["output_type"] is None


def test_the_scorer_refuses_text_and_invalid_predictions() -> None:
    frame = pd.DataFrame({"x": [1.0, 2.0]})
    target = pd.Series([1.0, 2.0])
    with pytest.raises(ValueError, match="no text columns"):
        ClientProgramScorer(_Predictor([1.0, 1.0]))(
            frame, target, frame, target, categorical_columns=[], text_columns=["d"]
        )
    with pytest.raises(ValueError, match="invalid predictions"):
        ClientProgramScorer(_Predictor([1.0, None]))(
            frame, target, frame, target, categorical_columns=[]
        )


def test_noridfs_categoricals_come_from_its_plan() -> None:
    backend = NoriDFS()
    backend.last_plan = SimpleNamespace(categorical_columns=("numeric_category",))
    assert backend.categorical_columns() == ("numeric_category",)


def test_agentic_categoricals_come_from_the_backend() -> None:
    program = FeatureProgram((FeatureQuery("q", "SELECT 1"),))
    backend = AgenticFeatures(program=program)
    backend.last_categorical = ("mode",)
    assert backend.categorical_columns() == ("mode",)


def test_the_hosted_predictor_declares_categoricals_only_when_given(
    monkeypatch,
) -> None:
    calls = []

    class Transport:
        def __init__(self, **options):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def predict(self, train, target, query, **options):
            calls.append(options)
            return pd.Series([1.0] * len(query))

    monkeypatch.setattr("nori_rel._runtime.SynthefyNoriClient", Transport)
    predictor = HostedNoriPredictor("test-only", "nori-30m")
    frame = pd.DataFrame({"x": [1.0]})
    for declared in (None, ["key"]):
        predictor.predict_arrays(
            frame,
            pd.Series([1.0]),
            frame,
            task=PredictionTask.REGRESSION,
            output_type=RelationalOutputType.MEDIAN,
            categorical_columns=declared,
        )
    assert "categorical_columns" not in calls[0]
    assert calls[1]["categorical_columns"] == ["key"]


def test_the_noridfs_search_settings_are_pinned() -> None:
    """A change here changes every saved NoriDFS plan, so it must be deliberate."""
    assert (
        NativeFeatureConfig(
            max_depth=2,
            max_paths=24,
            max_features=48,
            max_direct_features=1,
            max_categorical_cardinality=32,
            window_multipliers=(1, 4, 16),
        )
        == NORIDFS_CONFIG
    )


class _Regressor:
    """A scikit-learn-style regressor that records its categorical setting."""

    def __init__(self, **options):
        self.categorical_columns = "auto"
        self.seen = []

    def get_params(self):
        return {"categorical_columns": self.categorical_columns}

    def set_params(self, **params):
        self.categorical_columns = params["categorical_columns"]

    def fit(self, train, target):
        self.seen.append(self.categorical_columns)
        return self

    def predict(self, query, *, output_type="mean", quantiles=None):
        return np.full(len(query), 1.0)


def test_local_nori_declares_categoricals_for_one_call(
    monkeypatch, fake_checkpoint
) -> None:
    monkeypatch.setattr("synthefy_nori.NoriRegressor", _Regressor)
    predictor = LocalNoriPredictor("nori-30m", device="cpu")
    frame = pd.DataFrame({"x": [1.0]})
    for declared in (["key"], None):
        predictor.predict_arrays(
            frame,
            pd.Series([1.0]),
            frame,
            task=PredictionTask.REGRESSION,
            output_type=RelationalOutputType.MEDIAN,
            categorical_columns=declared,
        )
    assert predictor._regressor.seen == [["key"], "auto"]
    assert predictor._regressor.categorical_columns == "auto"


def test_the_client_feature_search_still_works_with_a_deprecation_warning(
    monkeypatch,
) -> None:
    monkeypatch.setattr("synthefy_nori.NoriRegressor", _Regressor)
    with pytest.warns(DeprecationWarning, match="feature_strategy= to predict"):
        client = NoriRelClient(features="noridfs", device="cpu")
    assert client.mode == "local"
    assert client._runtime.features == "noridfs"
    assert isinstance(client._runtime.predictor, LocalNoriPredictor)


_SAVED = FeatureProgram(
    (
        FeatureQuery(
            "history",
            "SELECT e.row_id, COUNT(*) AS n FROM eval_table e GROUP BY e.row_id",
        ),
    ),
    drop_baseline=("nr|anchor|cutoff_year",),
)

_SAVED_QUERY = "PREDICT COUNT(orders.*,0,30,days) FOR EACH users.user_id"
_SAVED_SCHEMA = SchemaGraph(
    database_id="source",
    schema_name="public",
    tables=[
        {
            "name": "orders",
            "columns": [
                {"name": "id", "data_type": "bigint", "nullable": False},
                {"name": "user_id", "data_type": "bigint", "nullable": False},
                {"name": "created_at", "data_type": "timestamp", "nullable": False},
            ],
            "primary_key": ["id"],
            "time_column": "created_at",
            "foreign_keys": [
                {
                    "columns": ["user_id"],
                    "referred_table": "users",
                    "referred_columns": ["id"],
                }
            ],
        }
    ],
    discovered_at=pd.Timestamp("2026-01-01", tz="UTC").to_pydatetime(),
)
_RECORD_OPTIONS = {
    "query": _SAVED_QUERY,
    "entity_created_at": "created_at",
    "schema": _SAVED_SCHEMA,
    "baseline_columns": ("nr|anchor|cutoff_year",),
}


def _saved_record():
    return program_record(_SAVED, **_RECORD_OPTIONS)


def test_a_program_record_survives_a_json_round_trip() -> None:
    record = json.loads(json.dumps(_saved_record()))
    assert record["format"] == "nori-rel.feature-program"
    assert record["version"] == 2
    assert record["queries"] == [{"name": "history", "sql": _SAVED.queries[0].sql}]
    saved = program_from_record(record)
    assert saved.to_program() == _SAVED
    assert saved.pql_query == _SAVED_QUERY
    assert saved.baseline_columns == _RECORD_OPTIONS["baseline_columns"]


@pytest.mark.parametrize(
    "change,match",
    [
        ({"format": "something-else"}, r"nori-rel\.feature-program"),
        ({"version": 1}, "version 2"),
        ({"baseline": "fastdfs"}, "noridfs"),
        ({"queries": "SELECT 1"}, "queries"),
        ({"queries": [{"name": "history"}]}, "queries"),
        ({"drop_baseline": "nr|anchor|cutoff_year"}, "drop_baseline"),
        ({"target_transform": "log1p"}, "target_transform"),
        ({"baseline_columns": []}, "baseline_columns"),
        ({"schema_digest": ""}, "schema_digest"),
        ({"pql_query": ""}, "pql_query"),
    ],
)
def test_malformed_program_records_are_refused(change, match) -> None:
    with pytest.raises(ValueError, match=match):
        program_from_record({**_saved_record(), **change})


def test_a_program_record_must_be_a_mapping() -> None:
    with pytest.raises(ValueError, match="agentic_program record"):
        program_from_record([_saved_record()])


@pytest.mark.parametrize("transform", ["identity", "log1p"])
def test_record_rejects_unsupported_target_transforms(transform):
    with pytest.raises(ValueError, match="target_transform='auto'"):
        program_record(replace(_SAVED, target_transform=transform), **_RECORD_OPTIONS)


def test_schema_digest_ignores_source_handle_and_discovery_order():
    schema = _SAVED_SCHEMA.model_copy(deep=True)
    schema.database_id = "reconnected"
    schema.discovered_at = pd.Timestamp("2026-02-01", tz="UTC").to_pydatetime()
    schema.tables.reverse()
    schema.tables[0].columns.reverse()
    schema.tables[0].foreign_keys.reverse()
    assert compute_schema_digest(schema) == compute_schema_digest(_SAVED_SCHEMA)


@pytest.mark.parametrize(
    "change", ["type", "time_column", "primary_key", "foreign_key"]
)
def test_schema_digest_detects_structural_changes(change):
    schema = _SAVED_SCHEMA.model_copy(deep=True)
    table = schema.tables[0]
    if change == "type":
        table.columns[0].data_type = "text"
    elif change == "time_column":
        table.time_column = None
    elif change == "primary_key":
        table.primary_key = ["user_id"]
    else:
        table.foreign_keys[0].referred_columns = ["other_id"]
    with pytest.raises(ValueError, match="database schema"):
        program_from_record(_saved_record()).validate_schema(schema)


def test_metadata_records_the_program_that_was_used() -> None:
    backend = AgenticFeatures(program=_SAVED)
    backend.last_baseline_columns = _RECORD_OPTIONS["baseline_columns"]
    summary = agentic_summary(
        backend,
        query=_SAVED_QUERY,
        entity_created_at="created_at",
        schema=_SAVED_SCHEMA,
    )
    assert summary == {
        "agentic_model": None,
        "agentic_cost_usd": 0.0,
        "agentic_usage": TokenUsage().as_dict(),
        "agentic_stop_reason": "replay",
        "agentic_fallback_reason": None,
        "agentic_trials": 0,
        "agentic_queries": ["history"],
        "agentic_program": _saved_record(),
    }


def test_prediction_metadata_keeps_its_own_agent_run():
    backend = AgenticFeatures(program=_SAVED)
    backend.last_baseline_columns = _RECORD_OPTIONS["baseline_columns"]
    backend.last_session = SimpleNamespace(trials=[object()])
    backend.last_run_result = AgentRunResult(
        model="test-model",
        usage=TokenUsage(calls=2, input_tokens=10),
        cost_usd=0.5,
        stop_reason="max_cost",
    )
    summary = agentic_summary(
        backend,
        query=_SAVED_QUERY,
        entity_created_at="created_at",
        schema=_SAVED_SCHEMA,
    )
    backend.last_run_result = AgentRunResult(
        model="other-model",
        usage=TokenUsage(calls=1),
        cost_usd=0.1,
        stop_reason="model_finished",
    )
    assert summary["agentic_model"] == "test-model"
    assert summary["agentic_cost_usd"] == 0.5
    assert summary["agentic_usage"]["calls"] == 2
    assert summary["agentic_usage"]["input_tokens"] == 10
    assert summary["agentic_stop_reason"] == "max_cost"


@pytest.mark.parametrize("create_client", [False, True])
def test_root_and_fastdfs_client_do_not_import_agent_or_noridfs(create_client):
    script = """
import sys
import nori_rel
"""
    if create_client:
        script += 'client = nori_rel.NoriRelClient(mode="hosted", api_key="test")\n'
    script += """
optional = ("nori_rel.feature.agentic", "nori_rel.feature.dfs.noridfs")
assert not any(name.startswith(optional) for name in sys.modules)
from nori_rel import AgenticSearch
assert AgenticSearch.__name__ == "AgenticSearch"
"""
    subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True, text=True
    )


def test_metadata_records_no_program_when_noridfs_was_kept() -> None:
    search = AgenticFeatures(
        agent=_Agent(),
        scorer=ClientProgramScorer(predictor=_Predictor([0.0])),
        train_target=pd.Series([1.0]),
        fallback=NoriDFS(NORIDFS_CONFIG),
        config=DEFAULT_SEARCH_CONFIG,
    )
    assert (
        agentic_summary(
            search,
            query=_SAVED_QUERY,
            entity_created_at="created_at",
            schema=_SAVED_SCHEMA,
        )["agentic_program"]
        is None
    )


@pytest.mark.parametrize("features", ["fastdfs", "noridfs"])
def test_a_saved_program_replays_on_noridfs_without_a_search(features) -> None:
    runtime = _runtime(features=features)
    context = _context(1)
    backend = runtime.feature_backend(
        TargetWindow(0, 30),
        PredictionTask.REGRESSION,
        context,
        context["label"],
        program_from_record(_saved_record()),
    )
    assert isinstance(backend, AgenticFeatures)
    assert backend.program == _SAVED
    assert backend.agent is None
    assert isinstance(backend.fallback, NoriDFS)
    assert backend.config.extend_baseline


def test_a_replay_keeps_the_configured_search_settings() -> None:
    search = AgenticSearch(agent=_Agent())
    backend = _runtime(features=search).feature_backend(
        TargetWindow(0, 30),
        PredictionTask.REGRESSION,
        _context(1),
        pd.Series([1.0, 1.0]),
        program_from_record(_saved_record()),
    )
    assert backend.config is search.config


class _UnusedRuntime:
    """Fails the test if the client reaches the database."""

    def predict_pql(self, *args, **options):
        raise AssertionError("the client read the database")


def test_the_client_refuses_a_malformed_program_before_reading() -> None:
    client = NoriRelClient(mode="hosted", api_key="test", _runtime=_UnusedRuntime())
    with pytest.raises(ValueError, match=r"nori-rel\.feature-program"):
        client.predict(
            source=SimpleNamespace(id="db"),
            query="PREDICT COUNT(orders.*,0,30,days) FOR EACH users.user_id",
            anchor_times=[pd.Timestamp("2026-01-01", tz="UTC").to_pydatetime()],
            observed_through=pd.Timestamp("2026-02-01", tz="UTC").to_pydatetime(),
            entity_created_at="created_at",
            feature_program={"format": "something-else"},
        )


def _claude_agent(*, priced: bool):
    from nori_rel.feature.agentic.agent_claude import ClaudeProgramAgent
    from nori_rel.feature.agentic.usage import TokenPrices

    prices = TokenPrices(
        input=3.0, output=15.0, cache_read=0.3, cache_write_5m=3.75, cache_write_1h=6.0
    )
    return ClaudeProgramAgent(prices=prices if priced else None)


SPEND_LIMIT = RunLimits(max_spend_usd=5.0)


@pytest.mark.parametrize(
    "make_agent",
    [lambda: _claude_agent(priced=False), _Agent],
    ids=["claude_without_prices", "agent_reporting_no_spend"],
)
def test_a_spend_limit_needs_an_agent_that_reports_dollars(make_agent) -> None:
    pytest.importorskip("duckdb")
    with pytest.raises(ValueError, match=r"max_spend_usd needs .*prices=TokenPrices"):
        AgenticSearch(agent=make_agent(), limits=SPEND_LIMIT)


def test_a_spend_limit_is_accepted_for_a_priced_claude_agent() -> None:
    pytest.importorskip("duckdb")
    search = AgenticSearch(agent=_claude_agent(priced=True), limits=SPEND_LIMIT)
    assert search.limits.max_spend_usd == 5.0


def test_an_agent_that_reports_no_dollars_still_takes_a_runtime_limit() -> None:
    pytest.importorskip("duckdb")
    limits = RunLimits(max_runtime_seconds=600)
    assert AgenticSearch(agent=_claude_agent(priced=False), limits=limits).limits
    assert AgenticSearch(agent=_Agent(), limits=limits).limits


def test_new_limits_on_a_claude_search_are_checked_against_its_prices(
    monkeypatch,
) -> None:
    import dataclasses

    pytest.importorskip("anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    unpriced = AgenticSearch.claude()
    with pytest.raises(ValueError, match="max_spend_usd needs"):
        dataclasses.replace(unpriced, limits=SPEND_LIMIT)
    priced = AgenticSearch.claude(prices=_claude_agent(priced=True).prices)
    assert dataclasses.replace(priced, limits=SPEND_LIMIT).limits == SPEND_LIMIT


def test_a_spend_limit_is_refused_for_the_openai_agent() -> None:
    import dataclasses

    pytest.importorskip("openai")
    search = AgenticSearch.openai(model="gpt-5", api_key="test")
    with pytest.raises(ValueError, match="OpenAI agent reports no spend"):
        dataclasses.replace(search, limits=SPEND_LIMIT)
