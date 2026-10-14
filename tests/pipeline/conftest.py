import pytest
from pipeline_records import manifest_fields

from nori_rel.pipeline.manifest import PipelineManifest


@pytest.fixture
def manifest() -> PipelineManifest:
    return PipelineManifest.create(**manifest_fields())
