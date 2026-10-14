import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest
from pipeline_records import manifest_fields

from nori_rel import NoriRelPipeline, PipelineError
from nori_rel.pipeline import (
    PIPELINE_ATTR,
    PipelineFormatError,
    get_pql_task_signature,
)
from nori_rel.pipeline.manifest import ContextPolicyRecord, PipelineManifest


def test_pipeline_id_is_derived_from_the_prediction_sections(manifest) -> None:
    again = PipelineManifest.create(
        **{**manifest_fields(), "provenance": {"feature_count": 9}}
    )
    changed = manifest_fields()
    changed["context_policy"] = {
        "anchor_offsets": ["P60D", "P30D"],
        "max_context_rows": 5000,
    }

    assert manifest.pipeline_id.startswith("pipe_")
    assert again.pipeline_id == manifest.pipeline_id
    assert PipelineManifest.create(**changed).pipeline_id != manifest.pipeline_id


def test_save_and_load_round_trip(manifest, tmp_path) -> None:
    pipeline = NoriRelPipeline(manifest)
    path = pipeline.save(tmp_path / "orders.json")

    loaded = NoriRelPipeline.load(path)

    assert loaded == pipeline
    assert loaded.pipeline_id == manifest.pipeline_id
    assert loaded.feature_names == ("COUNT(orders)",)
    assert json.loads(path.read_text())["format"] == "nori-rel.pipeline"
    assert list(tmp_path.iterdir()) == [path]


def test_load_in_a_new_process_reads_the_same_pipeline(manifest, tmp_path) -> None:
    path = NoriRelPipeline(manifest).save(tmp_path / "orders.json")
    script = (
        "import sys\n"
        "from nori_rel import NoriRelPipeline\n"
        "print(NoriRelPipeline.load(sys.argv[1]).pipeline_id)\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == manifest.pipeline_id


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        ({"format": "nori-rel.feature-program"}, "not a NoriRel pipeline"),
        ({"format_version": 2}, "upgrade nori-rel"),
        ({"format_version": 0}, "export the pipeline again"),
        ({"format_version": "1"}, "must be an integer"),
        ({"unexpected": 1}, "invalid pipeline record"),
    ],
)
def test_load_rejects_other_formats_and_versions(manifest, edit, message) -> None:
    record = {**NoriRelPipeline(manifest).to_dict(), **edit}

    with pytest.raises(PipelineFormatError, match=message):
        NoriRelPipeline.from_dict(record)


def test_load_rejects_a_record_edited_after_save(manifest, tmp_path) -> None:
    record = NoriRelPipeline(manifest).to_dict()
    record["prediction"]["output_type"] = "mean"
    path = tmp_path / "edited.json"
    path.write_text(json.dumps(record))

    with pytest.raises(PipelineFormatError, match="changed after it was saved"):
        NoriRelPipeline.load(path)


def test_load_ignores_edits_outside_the_prediction_sections(manifest) -> None:
    record = NoriRelPipeline(manifest).to_dict()
    record["provenance"] = {"note": "reviewed"}

    assert NoriRelPipeline.from_dict(record).pipeline_id == manifest.pipeline_id


def test_load_rejects_non_json_files(tmp_path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{")

    with pytest.raises(PipelineFormatError, match="not a JSON pipeline file"):
        NoriRelPipeline.load(path)


def test_errors_share_one_value_error_base() -> None:
    assert issubclass(PipelineFormatError, PipelineError)
    assert issubclass(PipelineError, ValueError)


def test_from_prediction_reads_the_frame_attrs(manifest) -> None:
    frame = pd.DataFrame({"prediction": [1.0]})
    assert NoriRelPipeline.from_prediction(frame) is None

    frame.attrs[PIPELINE_ATTR] = NoriRelPipeline(manifest).to_dict()

    assert NoriRelPipeline.from_prediction(frame).pipeline_id == manifest.pipeline_id


def test_task_signature_ignores_the_scored_entities_only() -> None:
    base = "PREDICT SUM(orders.amount,0,30,days) FOR EACH users.user_id"
    some = "PREDICT SUM(orders.amount, 0, 30, days) FOR users.user_id IN (1, 2)"
    where = base + " WHERE users.country='US'"
    target = base.replace("SUM", "AVG")

    assert get_pql_task_signature(base) == get_pql_task_signature(some)
    assert get_pql_task_signature(where) != get_pql_task_signature(base)
    assert get_pql_task_signature(target) != get_pql_task_signature(base)


def test_manifest_rejects_inconsistent_columns() -> None:
    fields = manifest_fields()
    fields["columns"] = {"names": ["a", "b"], "dtypes": ["float64"], "categorical": []}

    with pytest.raises(ValueError, match="one dtype for each name"):
        PipelineManifest.create(**fields)


def test_the_context_policy_places_anchors_before_each_as_of(manifest) -> None:
    policy = manifest.context_policy
    first = datetime(2026, 4, 1, tzinfo=UTC)
    later = datetime(2026, 4, 8, tzinfo=UTC)

    assert [first - anchor for anchor in policy.get_anchor_times(first)] == [
        timedelta(days=90),
        timedelta(days=60),
        timedelta(days=30),
    ]
    assert policy.get_anchor_times(later)[-1] == datetime(2026, 3, 9, tzinfo=UTC)
    with pytest.raises(ValueError, match="timezone-aware"):
        policy.get_anchor_times(datetime(2026, 4, 1))


def test_the_policy_comes_from_the_anchors_of_the_first_run() -> None:
    as_of = datetime(2026, 1, 1, tzinfo=UTC)
    anchors = [datetime(2025, month, 1, tzinfo=UTC) for month in (11, 9, 10)]

    policy = ContextPolicyRecord.from_anchor_times(
        anchors, as_of=as_of, max_context_rows=50
    )

    assert policy.anchor_offsets == (
        timedelta(days=122),
        timedelta(days=92),
        timedelta(days=61),
    )
    assert sorted(policy.get_anchor_times(as_of)) == sorted(anchors)


@pytest.mark.parametrize(
    ("offsets", "message"),
    [
        ([timedelta(0)], "must be positive"),
        ([timedelta(days=30), timedelta(days=60)], "longest to shortest"),
        ([timedelta(days=30), timedelta(days=30)], "longest to shortest"),
    ],
)
def test_the_policy_rejects_bad_offsets(offsets, message) -> None:
    with pytest.raises(ValueError, match=message):
        ContextPolicyRecord(anchor_offsets=offsets, max_context_rows=10)


def test_a_misspelled_plan_setting_fails_on_load(manifest) -> None:
    record = NoriRelPipeline(manifest).to_dict()
    record["features"]["max_dpeth"] = 3

    with pytest.raises(PipelineFormatError, match="max_dpeth"):
        NoriRelPipeline.from_dict(record)


def test_the_first_run_does_not_gate_later_runs(manifest) -> None:
    record = NoriRelPipeline(manifest).to_dict()

    assert "context" not in record
    assert "anchor_times" not in record["task"]
