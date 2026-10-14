"""Exercise PostgreSQL + FastDFS + hosted-client transport without network calls."""

from datetime import UTC, datetime

import pytest

from nori_rel.limits import RelationalLimits

QUERY = (
    "PREDICT COUNT(orders.*,0,30,days) FOR users.user_id=99 WHERE users.country='US'"
)
OPTIONS = dict(
    anchor_times=[datetime(2026, 1, 1, tzinfo=UTC)],
    anchor_time=datetime(2026, 2, 1, tzinfo=UTC),
    observed_through=datetime(2026, 2, 1, tzinfo=UTC),
    entity_created_at="created_at",
)


def test_regression_pipeline_maps_predictions_and_reports_counts(runtime_source):
    runtime, database, calls = runtime_source
    result = runtime.predict_pql(database, QUERY, **OPTIONS)
    assert result[["entity_id", "prediction"]].values.tolist() == [[99, 12.5]]
    assert result.attrs["usable_rows"] == 2
    assert result.attrs["task"] == "regression"
    train, target, query, options = calls[0]
    assert target.tolist() == [2.0, 0.0]
    assert train.columns.tolist() == query.columns.tolist()
    assert not {"label", "context_row_id", "query_row_id"} & set(train.columns)
    assert options["output_type"] == "median"


def test_predicate_pipeline_sends_binary_labels(runtime_source):
    runtime, database, calls = runtime_source
    query = QUERY.replace(") FOR", ")>0 FOR")
    result = runtime.predict_pql(database, query, **OPTIONS, decision_threshold=0.9)
    assert result["prediction"].tolist() == [False]
    assert result["probability"].tolist() == [0.8]
    assert calls[0][1].tolist() == [1.0, 0.0]
    assert calls[0][3]["task"] == "classification"


def test_quantile_pipeline(runtime_source):
    runtime, database, _ = runtime_source
    result = runtime.predict_pql(
        database, QUERY, **OPTIONS, output_type="quantiles", quantiles=[0.1, 0.5, 0.9]
    )
    assert result["prediction"].tolist() == [2.0]
    assert result["quantiles"].tolist() == [[1.0, 2.0, 3.0]]


@pytest.mark.parametrize(
    "limits,match",
    [
        (RelationalLimits(min_training_rows=3), "usable context"),
        (RelationalLimits(min_training_rows=2, max_feature_elements=1), "budget"),
    ],
)
def test_inference_limits_error_without_model_call(runtime_source, limits, match):
    runtime, database, calls = runtime_source
    runtime.limits = limits
    with pytest.raises(ValueError, match=match):
        runtime.predict_pql(database, QUERY, **OPTIONS)
    assert not calls


def test_single_class_errors_instead_of_inventing_probabilities(runtime_source):
    runtime, database, calls = runtime_source
    with pytest.raises(ValueError, match="both false and true"):
        runtime.predict_pql(database, QUERY.replace(") FOR", ")>=0 FOR"), **OPTIONS)
    assert not calls


@pytest.mark.parametrize(
    "options",
    [
        dict(task="classification"),
        dict(quantiles=[0.5]),
        dict(output_type="quantiles", quantiles=[0.9, 0.1]),
        dict(decision_threshold=0.5),
    ],
)
def test_invalid_task_output_options_fail_before_preparation(runtime_source, options):
    runtime, database, calls = runtime_source
    with pytest.raises(ValueError):
        runtime.predict_pql(database, QUERY, **OPTIONS, **options)
    assert not calls
