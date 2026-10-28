"""Every skill script runs end to end on PostgreSQL: fake Nori in CI, real on request.

The scripts under skills/*/scripts/ are run in-process through their ``main()``
on the synthetic users/orders fixture, in the order the quickstart skill
follows: check the source, predict, backtest, replay the saved pipeline. The
default parameter swaps ``synthefy_nori.NoriRegressor`` for a stand-in, so the
run needs no weights; ``-m live_nori --require-local-model`` runs the same flow
on cached Nori 1.5 weights.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import psycopg
import pytest
from psycopg import sql

REPOSITORY = Path(__file__).resolve().parents[1]
SKILLS = REPOSITORY / "skills"
DATABASE_VARIABLE = "NORI_REL_TEST_DATABASE_URL"
QUANTILES = [0.1, 0.5, 0.9]
USERS = 48
PAST_DATES = 4
USERS_WITHOUT_INCOME = 8
USERS_IN_CANADA = 16
TEMPORAL_TASK = {
    "query": "PREDICT SUM(orders.amount,0,30,days) FOR EACH users.user_id",
    "window": {"end": 30, "unit": "days"},
    "anchor_time": "2026-03-02",
    "observed_through": "2026-03-02",
    "past_dates": PAST_DATES,
    "entity_created_at": "created_at",
    "max_context_rows": 5000,
    "feature_strategy": "fastdfs",
    "output_type": "quantiles",
    "quantiles": QUANTILES,
    "query_filter": None,
    "mode": "local",
    "model": None,
    "device": None,
}
STATIC_TASK = {
    "query": "PREDICT users.income FOR EACH users.user_id",
    "anchor_time": None,
    "max_context_rows": 5000,
    "feature_strategy": "fastdfs",
    "mode": "local",
    "model": None,
    "device": None,
}

pytestmark = pytest.mark.skipif(
    DATABASE_VARIABLE not in os.environ, reason=f"{DATABASE_VARIABLE} is not configured"
)


class FakeRegressor:
    """A stand-in for ``synthefy_nori.NoriRegressor`` that needs no weights."""

    def __init__(self, **options) -> None:
        self.options = {"categorical_columns": None, **options}

    def get_params(self):
        return {"augmentations": ("yj",), **self.options}

    def set_params(self, **options):
        self.options.update(options)
        return self

    def fit(self, train, target):
        return self

    def predict(self, query, *, output_type="mean", quantiles=None):
        if output_type == "quantiles":
            return np.array([[level * 10.0] * len(query) for level in quantiles])
        return np.full(len(query), 2.5)


def load_module(path: Path):
    """Import a script by path under a unique module name."""
    spec = importlib.util.spec_from_file_location(
        f"skill_script_{path.stem}_{uuid4().hex}", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def script(skill: str, name: str):
    """Import ``skills/<skill>/scripts/<name>``."""
    return load_module(SKILLS / skill / "scripts" / name)


def write_json(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run(monkeypatch, capsys, module, *arguments) -> tuple[int, str]:
    """Run a script's ``main()`` as the command line would; return exit and output."""
    monkeypatch.setattr(sys, "argv", [module.__file__, *map(str, arguments)])
    code = module.main()
    return code, capsys.readouterr().out


@pytest.fixture(scope="module")
def schema_name():
    """Load the synthetic fixture into a schema of its own and drop it after."""
    name = "skills_" + uuid4().hex
    loader = load_module(REPOSITORY / "examples" / "load_fixture.py")
    with psycopg.connect(os.environ[DATABASE_VARIABLE], autocommit=True) as database:
        users, orders, _ = loader.load_fixture(database, name)
        assert (users, orders) == (USERS, 1248)
        try:
            yield name
        finally:
            database.execute(
                sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(name))
            )


@pytest.fixture
def workspace(tmp_path, schema_name, monkeypatch) -> Path:
    """A working directory holding the handoff files the skills exchange."""
    monkeypatch.chdir(tmp_path)
    write_json(
        tmp_path / "nori_rel_source.json",
        {
            "engine": "postgresql",
            "credential": {"environment": DATABASE_VARIABLE},
            "schema_name": schema_name,
            "tables": ["users", "orders"],
            "time_columns": {"orders": "event_time"},
            "primary_keys": {},
            "foreign_keys": {},
            "connection": None,
        },
    )
    write_json(tmp_path / "nori_rel_task.json", TEMPORAL_TASK)
    write_json(tmp_path / "nori_rel_task_static.json", STATIC_TASK)
    return tmp_path


@pytest.fixture(params=["fake", pytest.param("live", marks=pytest.mark.live_nori)])
def nori(request, monkeypatch) -> str:
    """Make local Nori available: a fake regressor, or cached real weights."""
    monkeypatch.delenv("SYNTHEFY_NORI_API_KEY", raising=False)
    if request.param == "fake":
        request.getfixturevalue("fake_checkpoint")
        monkeypatch.setattr("synthefy_nori.NoriRegressor", FakeRegressor)
        return request.param
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_OFFLINE", True)
    try:
        from synthefy_nori.hf import download_checkpoint

        download_checkpoint(model="nori-1-5")
    except (ImportError, OSError):
        if request.config.getoption("--require-local-model"):
            pytest.fail(
                "local runtime and cached nori-1-5 weights are required", pytrace=False
            )
        pytest.skip("local runtime or nori-1-5 weights are not in the cache")
    return request.param


def test_check_source_reports_the_schema(workspace, monkeypatch, capsys):
    check = script("connect", "check_source.py")
    code, out = run(
        monkeypatch, capsys, check, "nori_rel_source.json", "--entity", "users"
    )
    assert code == 0
    assert "connection: ready" in out
    assert "PASS: orders is an event table on event_time" in out
    assert "PASS: event tables linked straight to users: orders" in out
    assert "0 failures" in out

    broken = read_json(workspace / "nori_rel_source.json")
    broken["time_columns"] = {"orders": "shipped_at"}
    write_json(workspace / "broken.json", broken)
    code, out = run(monkeypatch, capsys, check, "broken.json", "--entity", "users")
    assert code == 1
    assert "FAIL: time column orders.shipped_at does not exist" in out


@pytest.mark.parametrize(
    "skill,name",
    [
        ("connect", "check_source.py"),
        ("predict", "predict.py"),
        ("backtest", "backtest.py"),
        ("schedule", "refresh.py"),
    ],
)
def test_scripts_read_an_rds_iam_credential(skill, name):
    from nori_rel import RdsIamToken

    settings = {
        "url": "mysql://nori_reader@127.0.0.1:3307/shop?ssl_mode=VERIFY_CA",
        "endpoint": "shop.abc123.us-east-1.rds.amazonaws.com:3306",
    }
    credential = script(skill, name).make_credential(
        {"credential": {"rds_iam": settings}}
    )
    assert credential == RdsIamToken(**settings)


@pytest.mark.parametrize("strategy", ["fastdfs", "noridfs"])
def test_predict_writes_every_output(workspace, nori, strategy, monkeypatch, capsys):
    task = read_json(workspace / "nori_rel_task.json")
    task["feature_strategy"] = strategy
    write_json(workspace / "nori_rel_task.json", task)
    predict = script("predict", "predict.py")
    code, out = run(
        monkeypatch,
        capsys,
        predict,
        "nori_rel_source.json",
        "nori_rel_task.json",
        "--out-dir",
        "out",
    )
    assert code == 0
    out_dir = workspace / "out"
    assert sorted(path.name for path in out_dir.iterdir()) == [
        "context.csv",
        "feature_table.csv",
        "features.csv",
        "metadata.json",
        "pipeline.json",
        "predictions.csv",
    ]
    predictions = pd.read_csv(out_dir / "predictions.csv")
    assert len(predictions) == USERS
    assert predictions["entity_id"].is_unique
    assert np.isfinite(predictions["prediction"]).all()
    assert "quantiles" in predictions
    context = pd.read_csv(out_dir / "context.csv")
    assert len(context) == USERS * PAST_DATES
    assert context["anchor_time"].nunique() == PAST_DATES
    metadata = read_json(out_dir / "metadata.json")
    assert metadata["features"] == strategy
    assert metadata["usable_rows"] == USERS * PAST_DATES
    assert metadata["pipeline_id"]
    assert len(pd.read_csv(out_dir / "feature_table.csv")) == metadata["feature_count"]
    assert f"scored rows: {USERS}" in out
    assert f"pipeline {metadata['pipeline_id']} saved" in out


def test_predict_static_target_has_no_pipeline(workspace, nori, monkeypatch, capsys):
    predict = script("predict", "predict.py")
    code, out = run(
        monkeypatch,
        capsys,
        predict,
        "nori_rel_source.json",
        "nori_rel_task_static.json",
        "--out-dir",
        "static",
    )
    assert code == 0
    out_dir = workspace / "static"
    assert not (out_dir / "pipeline.json").exists()
    assert len(pd.read_csv(out_dir / "predictions.csv")) == USERS_WITHOUT_INCOME
    assert len(pd.read_csv(out_dir / "context.csv")) == USERS - USERS_WITHOUT_INCOME
    assert "no pipeline" in out


def test_backtest_compares_with_known_outcomes(workspace, nori, monkeypatch, capsys):
    backtest = script("backtest", "backtest.py")
    code, out = run(
        monkeypatch,
        capsys,
        backtest,
        "nori_rel_source.json",
        "nori_rel_task.json",
        "--folds",
        "2",
        "--out-dir",
        "out",
    )
    assert code == 0
    table = pd.read_csv(workspace / "out" / "backtest.csv")
    assert table["fold"].tolist() == [1, 2]
    assert table["cutoff"].str[:10].tolist() == ["2026-01-31", "2026-01-01"]
    assert (table["entities"] == USERS).all()
    assert {
        "mae_nori",
        "rmse_nori",
        "mae_repeat_previous",
        "mae_context_median",
        "coverage",
    } <= set(table.columns)
    assert table["coverage"].between(0, 1).all()
    rows = pd.read_csv(workspace / "out" / "backtest_rows.csv")
    assert len(rows) == 2 * USERS
    assert rows["actual"].notna().all()
    assert "fold 2 at 2026-01-01" in out
    assert "repeat previous window" in out


def test_backtest_static_holds_out_stored_values(workspace, nori, monkeypatch, capsys):
    backtest = script("backtest", "backtest.py")
    code, out = run(
        monkeypatch,
        capsys,
        backtest,
        "nori_rel_source.json",
        "nori_rel_task_static.json",
        "--holdout",
        "5",
        "--out-dir",
        "static",
    )
    assert code == 0
    table = pd.read_csv(workspace / "static" / "backtest.csv")
    assert table["entities"].tolist() == [5]
    assert table["cutoff"].isna().all()
    assert len(pd.read_csv(workspace / "static" / "backtest_rows.csv")) == 5
    assert "held out 5 rows" in out


def test_refresh_replays_the_saved_pipeline(workspace, nori, monkeypatch, capsys):
    predict = script("predict", "predict.py")
    refresh = script("schedule", "refresh.py")
    assert (
        run(
            monkeypatch,
            capsys,
            predict,
            "nori_rel_source.json",
            "nori_rel_task.json",
            "--out-dir",
            "out",
        )[0]
        == 0
    )
    metadata = read_json(workspace / "out" / "metadata.json")
    common = (
        "nori_rel_source.json",
        "out/pipeline.json",
        "--task",
        "nori_rel_task.json",
    )

    code, out = run(
        monkeypatch,
        capsys,
        refresh,
        *common,
        "--as-of",
        "2026-03-02",
        "--observed-through",
        "2026-03-02",
        "--out",
        "out/replay.csv",
    )
    assert code == 0
    assert metadata["context_rows_sha256"] in out
    original = pd.read_csv(workspace / "out" / "predictions.csv")
    replay = pd.read_csv(workspace / "out" / "replay.csv")
    assert replay["entity_id"].tolist() == original["entity_id"].tolist()
    assert np.allclose(replay["prediction"], original["prediction"])

    code, out = run(
        monkeypatch,
        capsys,
        refresh,
        *common,
        "--as-of",
        "2026-02-23",
        "--observed-through",
        "2026-02-23",
        "--query-filter",
        "users.country='CA'",
        "--out",
        "out/canada.csv",
    )
    assert code == 0
    assert len(pd.read_csv(workspace / "out" / "canada.csv")) == USERS_IN_CANADA

    other = {**TEMPORAL_TASK, "model": "nori-30m"}
    write_json(workspace / "other_model.json", other)
    code, out = run(
        monkeypatch,
        capsys,
        refresh,
        "nori_rel_source.json",
        "out/pipeline.json",
        "--task",
        "other_model.json",
        "--as-of",
        "2026-03-02",
        "--observed-through",
        "2026-03-02",
        "--out",
        "out/other.csv",
    )
    assert code == 2
    assert "PipelineModelError" in out
    assert not (workspace / "out" / "other.csv").exists()
