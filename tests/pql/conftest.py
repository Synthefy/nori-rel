"""Isolated PostgreSQL source shared by PQL pipeline integration tests."""

import os
from uuid import uuid4

import pandas as pd
import psycopg
import pytest

from nori_rel._runtime import LocalRelationalRuntime
from nori_rel.connectors.postgres import PostgreSQLConnector
from nori_rel.contracts import DatabaseCreateRequest, EnvironmentCredential
from nori_rel.limits import RelationalLimits


@pytest.fixture
def source(request):
    if not os.environ.get("NORI_REL_TEST_DATABASE_URL"):
        if request.config.getoption(
            "--require-hosted-model"
        ) or request.config.getoption("--require-local-model"):
            pytest.fail(
                "NORI_REL_TEST_DATABASE_URL is required for real inference checks",
                pytrace=False,
            )
        pytest.skip("PostgreSQL integration database is supplied by CI")
    name = "pql_execution_" + uuid4().hex
    with psycopg.connect(
        os.environ["NORI_REL_TEST_DATABASE_URL"], autocommit=True
    ) as db:
        db.execute(
            psycopg.sql.SQL("CREATE SCHEMA {}").format(psycopg.sql.Identifier(name))
        )
        try:
            db.execute("SET TIME ZONE 'UTC'")
            db.execute(
                psycopg.sql.SQL("SET search_path TO {}").format(
                    psycopg.sql.Identifier(name)
                )
            )
            db.execute(
                "CREATE TABLE users (user_id bigint PRIMARY KEY, "
                "created_at timestamptz, country text)"
            )
            db.execute(
                "CREATE TABLE orders (id integer PRIMARY KEY, buyer_id bigint "
                "REFERENCES users(user_id), event_time timestamptz, amount numeric, "
                "channel text)"
            )
            db.execute(
                "INSERT INTO users VALUES (7,'2025-01-01','US'), "
                "(42,'2025-01-01','US'), (99,'2025-01-01','CA'), "
                "(500,'2026-02-01','US')"
            )
            db.execute(
                "INSERT INTO orders VALUES (1,7,'2026-01-02',1.25,'web'), "
                "(2,7,'2026-01-03',2.50,'store'), "
                "(3,99,'2026-01-02',99,'web')"
            )
            config = DatabaseCreateRequest(
                engine="postgresql",
                credential=EnvironmentCredential(variable="NORI_REL_TEST_DATABASE_URL"),
                schema_name=name,
                tables=["users", "orders"],
                time_columns={"orders": "event_time"},
            )
            connector = PostgreSQLConnector(config, RelationalLimits())
            try:
                yield connector, db
            finally:
                connector.close()
        finally:
            db.execute(
                psycopg.sql.SQL("DROP SCHEMA {} CASCADE").format(
                    psycopg.sql.Identifier(name)
                )
            )


@pytest.fixture
def runtime_source(source, monkeypatch):
    calls = []

    class Transport:
        def __init__(self, **options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def predict(self, train, target, query, **options):
            calls.append((train.copy(), target.copy(), query.copy(), options))
            if options.get("task") == "classification":
                return pd.Series([0.8] * len(query))
            if options.get("output_type") == "quantiles":
                return pd.DataFrame([[1.0, 2.0, 3.0]] * len(query))
            return pd.Series([12.5] * len(query))

    monkeypatch.setattr("nori_rel._runtime.SynthefyNoriClient", Transport)
    runtime = LocalRelationalRuntime(
        api_key="test-only",
        model="nori-30m",
        classification_model=None,
        timeout=10,
        max_retries=0,
        limits=RelationalLimits(min_training_rows=2),
    )
    database = runtime.connect(source[0].config)
    try:
        yield runtime, database, calls
    finally:
        runtime.close()
