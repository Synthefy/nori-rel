from datetime import UTC, datetime
from types import SimpleNamespace

from nori_rel.feature import FeatureBackend
from nori_rel.pipeline.build import get_pipeline_manifest

JAN = datetime(2026, 1, 1, tzinfo=UTC)


class _CustomBackend(FeatureBackend):
    def compute(self, *args, **kwargs):
        raise AssertionError("not called")


def test_custom_backends_and_unpinned_models_have_no_pipeline() -> None:
    options = dict(
        engine="postgresql",
        query="PREDICT COUNT(orders.*,0,30,days) FOR EACH users.user_id",
        entity_created_at="created_at",
        task="regression",
        as_of=JAN,
        anchor_times=[JAN],
        max_context_rows=10,
        schema=None,
        tables={},
        matrices=None,
        categorical_columns=None,
        output_type=None,
        quantiles=None,
        decision_threshold=None,
        metadata={},
    )

    assert (
        get_pipeline_manifest(
            backend=_CustomBackend(),
            predictor=SimpleNamespace(get_pipeline_model_record=lambda: object()),
            **options,
        )
        is None
    )
