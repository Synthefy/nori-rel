"""Real warehouses, independent label expectations, and simulated inference.

No live service is used unless its test configuration is supplied. Explicit
certification uses --require-warehouse to turn absent settings into failures.
"""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from nori_rel import NoriRelClient, RelationalLimits
from nori_rel.connectors.errors import ConnectorError, SnapshotLimitError
from nori_rel.contracts import DatabaseCreateRequest, ForeignKeySchema

from .conftest import JANUARY_ANCHOR, LARGE_ENTITY_ID, SCORING_CUTOFF, _grant_reader


def _query(expression: str) -> str:
    return (
        f"PREDICT {expression} FOR users.user_id={LARGE_ENTITY_ID} "
        "WHERE users.country='US'"
    )


def _prepare(source, expression: str, **options):
    return source.connector.prepare_inputs(
        "test-warehouse",
        _query(expression),
        anchor_times=[JANUARY_ANCHOR, JANUARY_ANCHOR],
        anchor_time=SCORING_CUTOFF,
        observed_through=SCORING_CUTOFF,
        entity_created_at="created_at",
        **options,
    )


def _expected_labels(operation: str) -> dict[int, float | bool]:
    """Derive expectations from fixture arithmetic, without querying a driver.

    Normal entities have five web rows: NULL, 10+i, 10+i, 20+i, NULL.
    Entity 5 has no events; entity 6 has one NULL event. Ordered events
    30+i and 40+i share a timestamp and have increasing primary keys.
    """
    normal = {
        "count_rows": lambda entity: 5,
        "count_values": lambda entity: 3,
        "count_distinct": lambda entity: 2,
        "sum": lambda entity: 40 + 3 * entity,
        "average": lambda entity: (40 + 3 * entity) / 3,
        "minimum": lambda entity: 10 + entity,
        "maximum": lambda entity: 20 + entity,
        "first_ordered": lambda entity: 30 + entity,
        "last_ordered": lambda entity: 40 + entity,
        "arithmetic": lambda entity: 50 + 3 * entity,
        "binary": lambda entity: False,
        "delayed": lambda entity: 10 + entity,
        "first_null": lambda entity: None,
        "last_null": lambda entity: None,
    }
    labels = {
        entity: normal[operation](entity)
        for entity in range(1, 5)
        if normal[operation](entity) is not None
    }
    if operation in {"count_rows", "count_values", "count_distinct"}:
        labels[5] = 0
        labels[6] = 1 if operation == "count_rows" else 0
    elif operation in {"sum", "arithmetic", "delayed"}:
        labels[5] = 0
    elif operation == "binary":
        labels.update({5: True, 6: False})
    return labels


CASES = [
    ("count_rows", "COUNT(orders.* WHERE orders.channel='web',0,30,days)"),
    ("count_values", "COUNT(orders.amount WHERE orders.channel='web',0,30,days)"),
    (
        "count_distinct",
        "COUNT_DISTINCT(orders.amount WHERE orders.channel='web',0,30,days)",
    ),
    ("sum", "SUM(orders.amount WHERE orders.channel='web',0,30,days)"),
    ("average", "AVG(orders.amount WHERE orders.channel='web',0,30,days)"),
    ("minimum", "MIN(orders.amount WHERE orders.channel='web',0,30,days)"),
    ("maximum", "MAX(orders.amount WHERE orders.channel='web',0,30,days)"),
    ("first_ordered", "FIRST(orders.amount WHERE orders.channel='ordered',0,30,days)"),
    ("last_ordered", "LAST(orders.amount WHERE orders.channel='ordered',0,30,days)"),
    ("first_null", "FIRST(orders.amount WHERE orders.channel='web',0,30,days)"),
    ("last_null", "LAST(orders.amount WHERE orders.channel='web',0,30,days)"),
    (
        "arithmetic",
        "SUM(orders.amount WHERE orders.channel='web',0,30,days) "
        "+ COUNT(orders.* WHERE orders.channel='web',0,30,days)*2",
    ),
    ("binary", "COUNT(orders.* WHERE orders.channel='web',0,30,days)=0"),
    ("delayed", "SUM(orders.amount WHERE orders.channel='web',1,29,days)"),
]


@pytest.mark.parametrize("operation,expression", CASES)
def test_live_label_semantics(warehouse_source, operation, expression):
    snapshot, context, counts = _prepare(warehouse_source, expression)
    expected = _expected_labels(operation)
    assert context["entity_id"].tolist() == sorted(expected)
    assert context["context_row_id"].tolist() == sorted(expected)
    assert context["anchor_time"].tolist() == [JANUARY_ANCHOR] * len(expected)
    assert context["label"].tolist() == pytest.approx(list(expected.values()))
    assert counts == {
        "candidate_rows": 6,
        "usable_rows": len(expected),
        "dropped_unknown_labels": 6 - len(expected),
    }
    assert snapshot.tables["orders"]["event_time"].max() <= SCORING_CUTOFF
    # BIGINT identities must never pass through a floating-point intermediate.
    assert LARGE_ENTITY_ID in snapshot.tables["users"]["user_id"].tolist()


def test_live_discovery_connectivity_and_mapped_relationship(warehouse_source):
    source = warehouse_source
    assert source.connector.test() >= 0
    schema = source.connector.discover("test-warehouse")
    tables = {table.name: table for table in schema.tables}
    assert set(tables) == {"users", "orders"}
    assert tables["users"].primary_key == ["user_id"]
    assert tables["orders"].primary_key == ["order_id"]
    assert tables["orders"].time_column == "event_time"
    assert tables["orders"].foreign_keys[0].referred_table == "users"
    assert schema.capabilities.consistent_snapshot


def test_live_pinned_session_survives_concurrent_update(warehouse_source):
    source = warehouse_source
    amount_column = source.identifier("amount")
    order_column = source.identifier("order_id")
    with source.connector._session("test-warehouse") as session:
        statement = (
            f"SELECT {amount_column} FROM {session.table_references['orders']} "
            f"WHERE {order_column}=1"
        )
        before = source.connector._get_query_frame(
            session.connection, statement, max_rows=2
        )
        source.execute_write(
            f"UPDATE {source.reference('orders')} SET {amount_column}=424242 "
            f"WHERE {order_column}=1"
        )
        after = source.connector._get_query_frame(
            session.connection, statement, max_rows=2
        )
    pd.testing.assert_frame_equal(before, after)
    # A new extraction observes the write, proving the concurrent write committed.
    new_snapshot = source.connector.snapshot("test-warehouse", SCORING_CUTOFF)
    order = new_snapshot.tables["orders"].set_index("order_id")
    assert order.loc[1, "amount"] == 424242


def test_live_context_and_features_use_one_pin(warehouse_source, monkeypatch):
    source = warehouse_source
    pin_tables = type(source.connector)._pin_tables
    mutated = False

    def pin_then_update(connector, connection, schema):
        nonlocal mutated
        pinned = pin_tables(connector, connection, schema)
        if not mutated:
            source.execute_write(
                f"UPDATE {source.reference('orders')} "
                f"SET {source.identifier('amount')}=12345 "
                f"WHERE {source.identifier('order_id')}=4"
            )
            mutated = True
        return pinned

    monkeypatch.setattr(type(source.connector), "_pin_tables", pin_then_update)
    snapshot, context, _ = _prepare(source, CASES[3][1])
    assert context["label"].tolist() == list(_expected_labels("sum").values())
    order = snapshot.tables["orders"].set_index("order_id")
    assert order.loc[4, "amount"] == 11
    assert snapshot.provenance


def test_live_reader_privileges_deny_writes(warehouse_source):
    source = warehouse_source
    with (
        source.connector._session("test-warehouse") as session,
        session.connection.cursor() as cursor,
        pytest.raises(Exception) as failure,
    ):
        cursor.execute(
            f"INSERT INTO {source.reference('users')} VALUES (8080, '2025-01-01', 'US')"
        )
    message = str(failure.value).lower()
    assert any(
        term in message
        for term in (
            "privilege",
            "permission",
            "authorized",
            "access denied",
            "read-only",
        )
    ), "Write denial must be an authorization failure, not invalid SQL"


@pytest.mark.parametrize(
    "overrides", [{"max_rows_per_table": 10}, {"max_snapshot_bytes": 1}]
)
def test_live_snapshot_limits_fail_explicitly(warehouse_source, overrides):
    connector = type(warehouse_source.connector)(
        warehouse_source.config, replace(RelationalLimits(), **overrides)
    )
    try:
        with pytest.raises(SnapshotLimitError):
            connector.snapshot("test-warehouse", SCORING_CUTOFF)
    finally:
        connector.close()


def test_live_scoring_filters_leave_context_unchanged(warehouse_source):
    source = warehouse_source
    matched, context, counts = _prepare(
        source, CASES[3][1], query_filter="users.country='CA'"
    )
    excluded, other_context, other_counts = _prepare(
        source, CASES[3][1], query_filter="users.country='US'"
    )
    assert matched.scoring_filter_ids == (LARGE_ENTITY_ID,)
    assert set(excluded.scoring_filter_ids) == set(range(1, 7))
    pd.testing.assert_frame_equal(context, other_context)
    assert counts == other_counts


def test_live_invalid_explicit_primary_key_fails_before_prediction(warehouse_source):
    source = warehouse_source
    source.insert_rows("users", [(1, JANUARY_ANCHOR, "US")])
    with pytest.raises(ConnectorError):
        _prepare(source, CASES[3][1])


def test_live_null_explicit_primary_key_fails(warehouse_source):
    source = warehouse_source
    source.execute_write(
        f"CREATE TABLE {source.reference('bad_users')} AS SELECT "
        f"CAST(NULL AS BIGINT) AS {source.identifier('user_id')}, "
        f"{source.identifier('created_at')}, {source.identifier('country')} "
        f"FROM {source.reference('users')} WHERE {source.identifier('user_id')}=1"
    )
    _grant_reader(source)
    config = DatabaseCreateRequest.model_validate(
        {
            **source.config.model_dump(),
            "tables": ["bad_users"],
            "time_columns": {},
            "primary_keys": {"bad_users": ["user_id"]},
            "foreign_keys": {},
        }
    )
    connector = type(source.connector)(config, RelationalLimits())
    try:
        with pytest.raises(ConnectorError):
            connector.snapshot("test-bad-key", SCORING_CUTOFF)
    finally:
        connector.close()


def test_live_orphaned_foreign_key_fails(warehouse_source):
    source = warehouse_source
    source.insert_rows("orders", [(90001, 123456, JANUARY_ANCHOR, 42, "web")])
    with pytest.raises(ConnectorError):
        _prepare(source, CASES[3][1])


def test_live_null_foreign_key_preserves_labels_and_large_ids(warehouse_source):
    source = warehouse_source
    source.insert_rows("orders", [(90001, None, JANUARY_ANCHOR, 42, "web")])
    snapshot, context, _ = _prepare(source, CASES[3][1])
    assert context["label"].tolist() == list(_expected_labels("sum").values())
    identifiers = snapshot.tables["orders"]["user_id"]
    assert identifiers.isna().sum() == 1
    assert LARGE_ENTITY_ID in identifiers.dropna().tolist()


def test_live_quoted_table_and_column_names(warehouse_source):
    source = warehouse_source
    users_table = "user-accounts" if source.engine == "databricks" else "user.accounts"
    orders_table = "order-items" if source.engine == "databricks" else "order.items"
    table_properties = (
        " USING DELTA TBLPROPERTIES ('delta.columnMapping.mode'='name')"
        if source.engine == "databricks"
        else ""
    )
    source.execute_write(
        f"CREATE TABLE {source.reference(users_table)}{table_properties} AS SELECT "
        f"{source.identifier('user_id')} AS {source.identifier('user.id')}, "
        f"{source.identifier('created_at')} AS {source.identifier('created.at')}, "
        f"{source.identifier('country')} AS {source.identifier('home country')} "
        f"FROM {source.reference('users')}"
    )
    source.execute_write(
        f"CREATE TABLE {source.reference(orders_table)}{table_properties} AS SELECT "
        f"{source.identifier('order_id')} AS {source.identifier('order.id')}, "
        f"{source.identifier('user_id')} AS {source.identifier('user.id')}, "
        f"{source.identifier('event_time')} AS {source.identifier('event.time')}, "
        f"{source.identifier('amount')} AS {source.identifier('amount.value')}, "
        f"{source.identifier('channel')} AS {source.identifier('sales channel')} "
        f"FROM {source.reference('orders')}"
    )
    _grant_reader(source)
    config = DatabaseCreateRequest.model_validate(
        {
            **source.config.model_dump(),
            "tables": [users_table, orders_table],
            "time_columns": {orders_table: "event.time"},
            "primary_keys": {
                users_table: ["user.id"],
                orders_table: ["order.id"],
            },
            "foreign_keys": {
                orders_table: [
                    ForeignKeySchema(
                        columns=["user.id"],
                        referred_table=users_table,
                        referred_columns=["user.id"],
                    )
                ]
            },
        }
    )
    connector = type(source.connector)(config, RelationalLimits())
    try:
        snapshot, context, _ = connector.prepare_inputs(
            "test-warehouse-quoted",
            f"PREDICT SUM(`{orders_table}`.`amount.value` "
            f"WHERE `{orders_table}`.`sales channel`='web',0,30,days) "
            f"FOR `{users_table}`.`user.id`={LARGE_ENTITY_ID} "
            f"WHERE `{users_table}`.`home country`='US'",
            anchor_times=[JANUARY_ANCHOR],
            anchor_time=SCORING_CUTOFF,
            observed_through=SCORING_CUTOFF,
            entity_created_at="created.at",
        )
        assert context["label"].tolist() == list(_expected_labels("sum").values())
        assert set(snapshot.tables) == {users_table, orders_table}
    finally:
        connector.close()


@pytest.fixture
def captured_transport(monkeypatch):
    calls = []

    class DeterministicTransport:
        def __init__(self, **options):
            assert options["api_key"] == "warehouse-test-only"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def predict(self, train, target, query, **options):
            calls.append((train.copy(), target.copy(), query.copy(), options))
            value = 0.8 if options.get("task") == "classification" else 12.5
            return pd.Series([value] * len(query))

    monkeypatch.setattr("nori_rel._runtime.SynthefyNoriClient", DeterministicTransport)
    return calls


def _predict(client, database, expression):
    return client.predict(
        source=database,
        query=_query(expression),
        anchor_times=[JANUARY_ANCHOR],
        anchor_time=SCORING_CUTOFF,
        observed_through=SCORING_CUTOFF,
        entity_created_at="created_at",
        return_details=True,
    )


def test_live_public_hosted_pipeline_and_future_leakage(
    warehouse_source, captured_transport
):
    source = warehouse_source
    with NoriRelClient(
        mode="hosted",
        api_key="warehouse-test-only",
        limits=RelationalLimits(min_training_rows=2),
    ) as client:
        database = source.connect(client)
        assert client.test_connection(database).latency_ms >= 0
        assert len(client.discover(database).tables) == 2
        before = _predict(client, database, CASES[3][1])
        source.execute_write(
            f"UPDATE {source.reference('orders')} "
            f"SET {source.identifier('amount')}=-999999 "
            f"WHERE {source.identifier('event_time')} > '2026-02-01'"
        )
        after = _predict(client, database, CASES[3][1])
        binary = _predict(client, database, CASES[12][1])
    assert before["predictions"]["entity_id"].tolist() == [LARGE_ENTITY_ID]
    assert before["predictions"]["prediction"].tolist() == [12.5]
    pd.testing.assert_frame_equal(before["predictions"], after["predictions"])
    assert binary["predictions"]["probability"].tolist() == [0.8]
    assert captured_transport[0][1].tolist() == list(_expected_labels("sum").values())
    for left, right in zip(
        captured_transport[0][:3], captured_transport[1][:3], strict=True
    ):
        if isinstance(left, pd.Series):
            pd.testing.assert_series_equal(left, right)
        else:
            pd.testing.assert_frame_equal(left, right)
    train, target, query, _ = captured_transport[0]
    assert list(train.columns) == list(query.columns)
    assert len(train) == len(target) == 5
    assert len(query) == 1


def test_live_public_local_pipeline_does_not_create_hosted_client(
    warehouse_source, monkeypatch
):
    fitted = []

    class DeterministicRegressor:
        def __init__(self, **options):
            self.options = options

        def get_params(self):
            """Expose constructor settings for the complete pipeline record."""
            return dict(self.options)

        def fit(self, train, target):
            fitted.append((train.copy(), target.copy()))
            return self

        def predict(self, query, **options):
            return np.full(len(query), 2.5)

    def reject_hosted_client(*args, **kwargs):
        pytest.fail("Local prediction created a hosted inference client")

    monkeypatch.setattr("synthefy_nori.NoriRegressor", DeterministicRegressor)
    monkeypatch.setattr("nori_rel._runtime.SynthefyNoriClient", reject_hosted_client)
    with NoriRelClient(limits=RelationalLimits(min_training_rows=2)) as client:
        database = warehouse_source.connect(client)
        result = _predict(client, database, CASES[3][1])
        with pytest.raises(ValueError, match="regression PQL targets only"):
            _predict(client, database, CASES[12][1])
    assert result["predictions"]["prediction"].tolist() == [2.5]
    assert fitted[0][1].tolist() == list(_expected_labels("sum").values())


@pytest.mark.parity
@pytest.mark.warehouse
def test_live_warehouse_parity(warehouse_pair, captured_transport):
    snapshots = []
    outputs = []
    for source in warehouse_pair:
        snapshot, context, counts = _prepare(source, CASES[3][1])
        snapshots.append((snapshot, context, counts))
        with NoriRelClient(
            mode="hosted",
            api_key="warehouse-test-only",
            limits=RelationalLimits(min_training_rows=2),
        ) as client:
            outputs.append(_predict(client, source.connect(client), CASES[3][1]))
    pd.testing.assert_frame_equal(snapshots[0][1], snapshots[1][1], check_dtype=False)
    assert snapshots[0][2] == snapshots[1][2]
    pd.testing.assert_frame_equal(
        outputs[0]["predictions"], outputs[1]["predictions"], check_dtype=False
    )
    for left, right in zip(
        captured_transport[0][:3], captured_transport[1][:3], strict=True
    ):
        if isinstance(left, pd.Series):
            pd.testing.assert_series_equal(left, right, check_dtype=False)
        else:
            pd.testing.assert_frame_equal(left, right, check_dtype=False)


@pytest.mark.parity
@pytest.mark.parametrize("operation,expression", CASES)
def test_postgres_shared_fixture_labels(postgres_parity_source, operation, expression):
    """Warehouse and PostgreSQL labels share one independently expected fixture."""
    _, context, counts = _prepare(postgres_parity_source, expression)
    expected = _expected_labels(operation)
    assert context["entity_id"].tolist() == sorted(expected)
    assert context["label"].tolist() == pytest.approx(list(expected.values()))
    assert counts["usable_rows"] == len(expected)


def test_live_unavailable_snapshot_fails_without_current_fallback(
    warehouse_source, monkeypatch
):
    source = warehouse_source
    pin_tables = type(source.connector)._pin_tables

    def pin_unavailable_versions(connector, connection, schema):
        pinned_at, _ = pin_tables(connector, connection, schema)
        references = {
            table: (
                source.reference(table)
                + (
                    " AT (TIMESTAMP => '1900-01-01 00:00:00 +00:00'::TIMESTAMP_TZ)"
                    if source.engine == "snowflake"
                    else " VERSION AS OF 999999999"
                )
            )
            for table in source.config.tables
        }
        return pinned_at, references

    monkeypatch.setattr(type(source.connector), "_pin_tables", pin_unavailable_versions)
    with pytest.raises(ConnectorError):
        _prepare(source, CASES[3][1])


def test_live_schema_change_preserves_pin_or_fails_explicitly(
    warehouse_source, monkeypatch
):
    source = warehouse_source
    pin_tables = type(source.connector)._pin_tables
    altered = False

    def pin_then_alter(connector, connection, schema):
        nonlocal altered
        pinned = pin_tables(connector, connection, schema)
        if not altered:
            schema_change = (
                f"DROP COLUMN {source.identifier('country')}"
                if source.engine == "snowflake"
                else f"ADD COLUMNS ({source.identifier('schema_drift_probe')} BIGINT)"
            )
            source.execute_write(
                f"ALTER TABLE {source.reference('users')} {schema_change}"
            )
            altered = True
        return pinned

    monkeypatch.setattr(type(source.connector), "_pin_tables", pin_then_alter)
    try:
        snapshot, context, _ = _prepare(source, CASES[3][1])
    except ConnectorError:
        return
    # Add a column on Delta so this does not require its optional column-mapping
    # protocol. Snowflake may reject a read after the pinned column is dropped.
    assert "country" in snapshot.tables["users"].columns
    assert "schema_drift_probe" not in snapshot.tables["users"].columns
    users_schema = next(
        table for table in snapshot.schema.tables if table.name == "users"
    )
    assert "schema_drift_probe" not in {column.name for column in users_schema.columns}
    assert context["label"].tolist() == list(_expected_labels("sum").values())


@pytest.mark.live_nori
def test_live_warehouse_to_cached_real_local_nori(
    warehouse_source, monkeypatch, request
):
    """Opt-in smoke test with real cached weights; never download a checkpoint."""
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
    with NoriRelClient(
        device="cpu", limits=RelationalLimits(min_training_rows=2)
    ) as client:
        source = warehouse_source.connect(client)
        result = _predict(client, source, CASES[3][1])
    assert result["predictions"]["entity_id"].tolist() == [LARGE_ENTITY_ID]
    assert np.isfinite(result["predictions"]["prediction"].astype(float)).all()
