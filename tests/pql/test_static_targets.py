"""Real PostgreSQL -> static PQL target -> FastDFS -> real SDK; HTTP is simulated.

The bank has no dates at all: branches, accounts with a stored balance and
premium flag, and transfers between accounts. Static targets learn a column
from the other accounts whose value is stored.
"""

import json
from datetime import UTC, datetime

import httpx
import pandas as pd
import pytest

from nori_rel import ContextPolicy, NoriRelClient
from nori_rel._runtime import HostedNoriPredictor
from nori_rel.connectors.postgres import PostgreSQLConnector
from nori_rel.feature.dfs.fastdfs import FastDFS
from nori_rel.limits import RelationalLimits
from nori_rel.pipeline import PipelineContextError
from nori_rel.pql.parser import PQLSyntaxError

ACCOUNTS = range(1, 13)


def get_branch(account_id):
    """Return the branch of account_id: 1 for accounts 1-6, 2 for 7-12."""
    return 1 if account_id <= 6 else 2


def get_balance(account_id):
    """Return account_id's stored balance: 100 x id, unknown for 11 and 12."""
    return None if account_id > 10 else 100 * account_id


def get_is_premium(account_id):
    """Return the stored premium flag: even accounts, unknown for 12."""
    return None if account_id == 12 else account_id % 2 == 0


def get_transfers():
    """Return (transfer_id, sender_id, receiver_id, amount) rows.

    Account i sends 10 x i to the next account and 1 x i to the one after,
    wrapping around, so every account sends and receives two transfers.
    """
    rows = []
    for i in ACCOUNTS:
        for step, amount in ((1, 10 * i), (2, i)):
            receiver = (i + step - 1) % 12 + 1
            rows.append((len(rows) + 1, i, receiver, amount))
    return rows


@pytest.fixture
def bank(source, monkeypatch, request):
    """Yield (client, database, db, captured) over the bank source.

    Indirect parametrization picks the client's feature backend; the default
    is FastDFS at depth 2.
    """
    connector, db = source
    # The source fixture owns this isolated schema and drops it afterward.
    db.execute("CREATE TABLE branches (branch_id integer PRIMARY KEY, city text)")
    db.execute(
        "CREATE TABLE accounts (account_id bigint PRIMARY KEY, "
        "branch_id integer REFERENCES branches(branch_id), "
        "tenure_years integer, balance numeric, is_premium boolean, "
        "has_loan smallint)"
    )
    db.execute(
        "CREATE TABLE transfers (transfer_id integer PRIMARY KEY, "
        "sender_id bigint REFERENCES accounts(account_id), "
        "receiver_id bigint REFERENCES accounts(account_id), amount numeric)"
    )
    with db.cursor() as cursor:
        cursor.execute("INSERT INTO branches VALUES (1, 'Oslo'), (2, 'Lima')")
        cursor.executemany(
            "INSERT INTO accounts VALUES (%s,%s,%s,%s,%s,%s)",
            [
                (
                    i,
                    get_branch(i),
                    i,
                    get_balance(i),
                    get_is_premium(i),
                    int(i % 3 == 0),
                )
                for i in ACCOUNTS
            ],
        )
        cursor.executemany(
            "INSERT INTO transfers VALUES (%s,%s,%s,%s)", get_transfers()
        )
    captured = {"prepared": [], "arrays": [], "http": []}
    prepare = PostgreSQLConnector.prepare_inputs
    predict = HostedNoriPredictor.predict_arrays

    def observe_inputs(self, *args, **kwargs):
        snapshot, context, counts = prepare(self, *args, **kwargs)
        captured["prepared"].append((snapshot, context.copy(), counts.copy()))
        return snapshot, context, counts

    def observe_arrays(self, train, target, query, **options):
        captured["arrays"].append((train.copy(), target.copy(), query.copy()))
        return predict(self, train, target, query, **options)

    def reply(self, request, **kwargs):
        payload = json.loads(request.content)
        captured["http"].append(payload)
        rows = len(payload["X_test"])
        if payload.get("task") == "classification":
            result = {"task": "classification", "predictions": [0.75] * rows}
        else:
            result = {
                "task": "regression",
                "output_type": payload.get("output_type", "median"),
                "predictions": [1150.0 + i for i in range(rows)],
            }
        return httpx.Response(200, json=result, request=request)

    monkeypatch.setattr(PostgreSQLConnector, "prepare_inputs", observe_inputs)
    monkeypatch.setattr(HostedNoriPredictor, "predict_arrays", observe_arrays)
    monkeypatch.setattr(httpx.Client, "send", reply)
    with NoriRelClient(
        mode="hosted",
        api_key="fixture-only",
        model="synthefy/nori-30m-dev",
        max_retries=0,
        limits=RelationalLimits(min_training_rows=2),
        features=getattr(request, "param", "fastdfs"),
    ) as client:
        database = client.connect(
            engine="postgresql",
            credential=connector.config.credential,
            schema_name=connector.config.schema_name,
            tables=["branches", "accounts", "transfers"],
        )
        yield client, database, db, captured


BALANCE = "PREDICT accounts.balance FOR EACH accounts.account_id"


def get_expected_features(account_ids, *, target):
    """Return the depth-2 FastDFS matrix for account_ids, calculated by hand.

    Every column of accounts except the key and the target, the branch city,
    and statistics of each account's sent and received transfers. MODE ties
    keep the first value in transfer order; FastDFS returns an ID's MODE as
    its category text. No column reads the target.
    """
    transfers = pd.DataFrame(
        get_transfers(), columns=["transfer_id", "sender_id", "receiver_id", "amount"]
    )
    rows = []
    for i in account_ids:
        row = {
            "d1__accounts.tenure_years": i,
            "d1__accounts.balance": get_balance(i),
            "d1__accounts.is_premium": get_is_premium(i),
            "d1__accounts.has_loan": int(i % 3 == 0),
        }
        for key in ("sender_id", "receiver_id"):
            own = transfers.loc[transfers[key] == i]
            prefix = f"d2__accounts.{{}}(transfers[{key}]"
            row |= {
                prefix.format("COUNT") + ")": len(own),
                prefix.format("MAX") + ".amount)": own["amount"].max(),
                prefix.format("MEAN") + ".amount)": own["amount"].mean(),
                prefix.format("MIN") + ".amount)": own["amount"].min(),
                prefix.format("MODE") + ".receiver_id)": str(
                    own["receiver_id"].iloc[0]
                ),
                prefix.format("MODE") + ".sender_id)": str(own["sender_id"].iloc[0]),
                prefix.format("STD") + ".amount)": own["amount"].std(),
            }
        row["d2__accounts.branches.city"] = "Oslo" if get_branch(i) == 1 else "Lima"
        del row[f"d1__accounts.{target}"]
        rows.append(row)
    return pd.DataFrame(rows)


def assert_features_equal(actual, expected):
    """Compare feature matrices by column name and value, ignoring dtypes."""
    assert sorted(actual.columns) == sorted(expected.columns)
    actual = actual[expected.columns].astype(object).where(actual.notna(), None)
    expected = expected.astype(object).where(expected.notna(), None)
    for column in expected.columns:
        assert [
            v if isinstance(v, str) or v is None else pytest.approx(float(v))
            for v in expected[column]
        ] == actual[column].tolist(), column


def test_numeric_target_learns_from_known_rows_and_scores_unknown_ones(bank):
    client, database, _, captured = bank
    details = client.predict(source=database, query=BALANCE, return_details=True)
    frame, metadata = details["predictions"], details["metadata"]
    snapshot, context, counts = captured["prepared"][0]
    # Accounts 11 and 12 have no stored balance: they are scored, never context.
    assert frame["entity_id"].tolist() == [11, 12]
    assert frame["prediction"].tolist() == [1150.0, 1151.0]
    assert (frame["anchor_time"] == snapshot.as_of).all()
    assert context["entity_id"].tolist() == list(range(1, 11))
    assert context["label"].astype(float).tolist() == [100.0 * i for i in range(1, 11)]
    assert counts == {
        "candidate_rows": 10,
        "usable_rows": 10,
        "dropped_unknown_labels": 0,
    }
    assert metadata["task"] == "regression"
    assert details["pipeline"] is None
    train, target, query = captured["arrays"][0]
    assert_features_equal(train, get_expected_features(range(1, 11), target="balance"))
    assert_features_equal(query, get_expected_features([11, 12], target="balance"))
    assert target.tolist() == [100.0 * i for i in range(1, 11)]
    (payload,) = captured["http"]
    assert payload["y_train"] == [100.0 * i for i in range(1, 11)]
    assert len(payload["X_train"]) == 10 and len(payload["X_test"]) == 2


def test_boolean_target_is_binary_and_returns_probabilities(bank):
    client, database, _, captured = bank
    frame = client.predict(
        source=database,
        query="PREDICT accounts.is_premium FOR EACH accounts.account_id",
    )
    assert frame["entity_id"].tolist() == [12]
    assert frame["probability"].tolist() == [0.75]
    assert frame["prediction"].tolist() == [True]
    train, target, query = captured["arrays"][0]
    assert target.tolist() == [float(i % 2 == 0) for i in range(1, 12)]
    assert_features_equal(
        train, get_expected_features(range(1, 12), target="is_premium")
    )
    assert_features_equal(query, get_expected_features([12], target="is_premium"))
    assert captured["http"][0]["task"] == "classification"


def test_comparison_target_scores_named_rows_and_leaves_them_out_of_context(bank):
    client, database, _, captured = bank
    frame = client.predict(
        source=database,
        query="PREDICT accounts.balance > 450 FOR accounts.account_id IN (8, 3)",
    )
    assert frame["entity_id"].tolist() == [3, 8]
    assert frame["probability"].tolist() == [0.75, 0.75]
    _, context, _ = captured["prepared"][0]
    assert context["entity_id"].tolist() == [1, 2, 4, 5, 6, 7, 9, 10]
    assert context["label"].tolist() == [
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        True,
    ]
    train, _, query = captured["arrays"][0]
    assert not any("balance" in column for column in [*train.columns, *query.columns])


@pytest.mark.parametrize(
    "bank", ["fastdfs", FastDFS(max_depth=3)], indirect=True, ids=["depth2", "depth3"]
)
def test_changing_a_held_out_label_changes_no_model_input(bank):
    """Depth 3 adds branch statistics over every account of the branch.

    Without the target's removal, branches.MEAN(accounts.balance) would carry
    accounts 3 and 8's held-out balances into every context row's features.
    """
    client, database, db, captured = bank
    query = "PREDICT accounts.balance FOR accounts.account_id IN (3, 8)"
    first = client.predict(source=database, query=query, return_details=True)
    db.execute("UPDATE accounts SET balance = 1e9 WHERE account_id = 3")
    db.execute("UPDATE accounts SET balance = NULL WHERE account_id = 8")
    second = client.predict(source=database, query=query, return_details=True)
    (train_1, target_1, query_1), (train_2, target_2, query_2) = captured["arrays"]
    # An UPDATE moves rows in PostgreSQL, and the snapshot reads them in that
    # order, so a depth-3 STD can round one bit differently even when no value
    # changed. Floats are compared to 1e-12; a leak moves them far more.
    pd.testing.assert_frame_equal(train_1, train_2, rtol=1e-12)
    pd.testing.assert_series_equal(target_1, target_2, check_exact=True)
    pd.testing.assert_frame_equal(query_1, query_2, rtol=1e-12)
    assert first["predictions"]["entity_id"].tolist() == [3, 8]
    # The read time moved between the runs, but the context hash leaves it out.
    assert (
        first["metadata"]["context_rows_sha256"]
        == second["metadata"]["context_rows_sha256"]
    )
    assert second["predictions"]["prediction"].tolist() == [1150.0, 1151.0]
    assert not any("balance" in column for column in train_1.columns)
    if any(column.startswith("d3__") for column in train_1.columns):
        assert "d3__accounts.branches.MEAN(accounts.tenure_years)" in train_1.columns


@pytest.mark.parametrize("bank", [FastDFS(max_depth=3)], indirect=True)
def test_without_removal_the_held_out_label_would_reach_the_features(bank, monkeypatch):
    """Guard the test above: it fails if the target column stays in the tables."""
    client, database, db, captured = bank
    monkeypatch.setattr(
        "nori_rel._pql_features.get_snapshot_without_target_column",
        lambda snapshot, tree: snapshot,
    )
    query = "PREDICT accounts.balance FOR accounts.account_id IN (3, 8)"
    client.predict(source=database, query=query)
    db.execute("UPDATE accounts SET balance = 1e9 WHERE account_id = 3")
    client.predict(source=database, query=query)
    (train_1, _, _), (train_2, _, _) = captured["arrays"]
    column = "d3__accounts.branches.MEAN(accounts.balance)"
    assert train_1[column].tolist() != train_2[column].tolist()


def test_zero_one_numeric_column_needs_an_explicit_task(bank):
    client, database, _, captured = bank
    query = "PREDICT accounts.has_loan FOR accounts.account_id IN (1, 2)"
    with pytest.raises(ValueError, match="holds only 0 and 1: pass task="):
        client.predict(source=database, query=query)
    assert captured["http"] == []
    binary = client.predict(source=database, query=query, task="classification")
    assert binary["probability"].tolist() == [0.75, 0.75]
    number = client.predict(source=database, query=query, task="regression")
    assert number["prediction"].tolist() == [1150.0, 1151.0]
    assert [payload.get("task") for payload in captured["http"]] == [
        "classification",
        "regression",
    ]


@pytest.mark.parametrize(
    ("query", "task", "message"),
    [
        (BALANCE, "classification", "needs a 0/1 column"),
        (
            "PREDICT accounts.is_premium FOR EACH accounts.account_id",
            "regression",
            "task contradicts",
        ),
        (
            "PREDICT accounts.account_id FOR EACH accounts.account_id",
            None,
            "primary key",
        ),
        (
            BALANCE + " WHERE accounts.tenure_years > 9",
            None,
            "only 1 usable context rows",
        ),
    ],
)
def test_bad_tasks_and_too_little_context_fail_before_the_model(
    bank, query, task, message
):
    client, database, _, captured = bank
    with pytest.raises(ValueError, match=message):
        client.predict(source=database, query=query, task=task)
    assert captured["http"] == []


@pytest.mark.parametrize(
    "options",
    [
        {"anchor_times": [datetime(2026, 1, 1, tzinfo=UTC)]},
        {"observed_through": datetime(2026, 1, 1, tzinfo=UTC)},
        {"anchor_time": datetime(2026, 1, 1, tzinfo=UTC)},
    ],
)
def test_time_arguments_are_rejected_before_reading(bank, options):
    client, database, _, captured = bank
    with pytest.raises(ValueError, match="takes no"):
        client.predict(source=database, query=BALANCE, **options)
    assert captured["prepared"] == []


@pytest.mark.parametrize("bank", ["noridfs"], indirect=True)
def test_horizon_feature_strategies_are_rejected_before_reading(bank):
    client, database, _, captured = bank
    with pytest.raises(ValueError, match="uses FastDFS or a custom FeatureBackend"):
        client.predict(source=database, query=BALANCE)
    assert captured["prepared"] == []


def test_row_time_cutoffs_hide_later_events_from_each_row(runtime_source, source):
    """Each user's features stop at its own creation time.

    Users 7, 42 and 99 were created on 2025-01-01, before any order, so with
    entity_created_at they see none; without it, 99 sees its order of
    2026-01-02 at the read time.
    """
    runtime, database, calls = runtime_source
    db = source[1]
    db.execute("ALTER TABLE users ADD COLUMN income numeric")
    db.execute(
        "UPDATE users SET income = CASE user_id WHEN 7 THEN 10 WHEN 42 THEN 20 "
        "WHEN 500 THEN 50 END"
    )
    client = NoriRelClient(mode="hosted", api_key="test-only", _runtime=runtime)
    query = "PREDICT users.income FOR EACH users.user_id"
    by_row = client.predict(
        source=database, query=query, entity_created_at="created_at"
    )
    assert by_row["entity_id"].tolist() == [99]
    assert by_row["anchor_time"].tolist() == [datetime(2025, 1, 1, tzinfo=UTC)]
    at_read = client.predict(source=database, query=query)
    assert at_read["anchor_time"].iloc[0] > datetime(2026, 2, 1, tzinfo=UTC)
    (train_1, target_1, query_1, _), (train_2, _, query_2, _) = calls
    count = "d2__users.COUNT(orders)"
    # Context rows are newest first with row times: 500 (2026-02-01) leads.
    assert target_1.tolist() == [50.0, 10.0, 20.0]
    assert train_1[count].tolist() == [0, 0, 0]
    assert query_1[count].tolist() == [0]
    assert query_2[count].tolist() == [1]
    assert train_2[count].tolist() == [2, 0, 0]


def test_a_typo_reports_the_syntax_error_not_time_arguments(bank):
    client, database, _, captured = bank
    with pytest.raises(PQLSyntaxError, match="Invalid PQL at line 1, column"):
        client.predict(
            source=database,
            query="PREDICT accounts.balance FOR EAC accounts.account_id",
        )
    assert captured["prepared"] == []


def test_the_zero_one_check_reads_every_stored_value_not_the_capped_context(bank):
    """Account 12 holds a 5; the three context rows the cap keeps hold 0 or 1."""
    client, database, db, captured = bank
    db.execute("UPDATE accounts SET has_loan = 5 WHERE account_id = 12")
    query = "PREDICT accounts.has_loan FOR accounts.account_id IN (1, 2)"
    frame = client.predict(source=database, query=query, max_context_rows=3)
    _, context, _ = captured["prepared"][0]
    assert context["entity_id"].tolist() == [3, 4, 5]
    assert set(context["label"]) <= {0, 1}
    assert frame["prediction"].tolist() == [1150.0, 1151.0]
    with pytest.raises(ValueError, match="needs a 0/1 column"):
        client.predict(
            source=database, query=query, max_context_rows=3, task="classification"
        )


@pytest.mark.parametrize(
    ("query", "options", "message"),
    [
        (BALANCE, {"output_type": "quantiles"}, "quantiles must be sorted"),
        (BALANCE, {"quantiles": [0.5]}, "quantiles requires output_type"),
        (BALANCE, {"task": "regresion"}, "is not a valid PredictionTask"),
        (
            "PREDICT accounts.balance > 450 FOR EACH accounts.account_id",
            {"output_type": "median"},
            "binary PQL does not accept regression output options",
        ),
        (
            "PREDICT accounts.balance > 450 FOR EACH accounts.account_id",
            {"task": "regression"},
            "task contradicts",
        ),
        (BALANCE, {"task": "regression", "decision_threshold": 0.3}, "requires binary"),
    ],
)
def test_option_errors_come_before_the_source_is_read(bank, query, options, message):
    client, database, _, captured = bank
    with pytest.raises(ValueError, match=message):
        client.predict(source=database, query=query, **options)
    assert captured["prepared"] == []


def test_a_random_context_policy_samples_instead_of_the_lowest_keys(bank):
    """Above the cap the default keeps the lowest keys; a random policy does not.

    Each balance is 100 x its account ID, so the labels Nori receives name the
    accounts kept as context.
    """
    client, database, _, captured = bank
    client.predict(source=database, query=BALANCE, max_context_rows=3)
    for seed in (0, 1):
        client.predict(
            source=database,
            query=BALANCE,
            context=ContextPolicy(max_rows=3, time_sampling="random", seed=seed),
        )
    client.predict(
        source=database,
        query=BALANCE,
        context=ContextPolicy(max_rows=3, time_sampling="random", seed=0),
    )
    kept = [
        sorted(int(label) // 100 for label in arrays[1])
        for arrays in captured["arrays"]
    ]
    # The same seed keeps the same rows on a later read.
    assert kept[3] == kept[1]
    assert kept[0] == [1, 2, 3]
    assert all(len(ids) == 3 and set(ids) <= set(range(1, 11)) for ids in kept)
    assert kept[1] != kept[2]
    assert [kept[1], kept[2]] != [[1, 2, 3], [1, 2, 3]]


def test_a_nan_value_is_scored_as_missing_and_never_context(bank):
    client, database, db, captured = bank
    db.execute("UPDATE accounts SET balance = 'NaN' WHERE account_id = 5")
    frame = client.predict(source=database, query=BALANCE)
    _, context, _ = captured["prepared"][0]
    assert 5 not in context["entity_id"].tolist()
    assert frame["entity_id"].tolist() == [5, 11, 12]


def test_expected_context_hash_reproduces_a_static_run(bank):
    client, database, db, _ = bank
    first = client.predict(source=database, query=BALANCE, return_details=True)
    expected = first["metadata"]["context_rows_sha256"]
    client.predict(source=database, query=BALANCE, expected_context_sha256=expected)
    db.execute("UPDATE accounts SET balance = 1 WHERE account_id = 4")
    with pytest.raises(PipelineContextError):
        client.predict(source=database, query=BALANCE, expected_context_sha256=expected)
