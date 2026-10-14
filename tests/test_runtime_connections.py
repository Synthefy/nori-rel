"""Client ownership and cleanup of registered source connectors."""

from unittest.mock import Mock

import pytest

from nori_rel._runtime import LocalRelationalRuntime
from nori_rel.contracts import DatabaseCreateRequest, EnvironmentCredential
from nori_rel.feature import FeatureBackend


def runtime_with(factory):
    return LocalRelationalRuntime(
        api_key="test-only",
        model="nori-30m",
        classification_model=None,
        timeout=10,
        max_retries=0,
        connector_factory=factory,
        features=Mock(spec=FeatureBackend),
        predictor=Mock(),
    )


def config():
    return DatabaseCreateRequest(
        engine="postgresql", credential=EnvironmentCredential(variable="TEST_DB")
    )


def test_sources_reuse_their_connector_and_close_all_even_on_cleanup_error():
    first, second = Mock(), Mock()
    first.test.return_value = second.test.return_value = 1.0
    factory = Mock(side_effect=[first, second])
    runtime = runtime_with(factory)
    a, b = runtime.connect(config()), runtime.connect(config())
    runtime.test_connection(a)
    runtime.discover(a)
    assert runtime._connector(a) is first
    assert runtime._connector(b) is second
    with pytest.raises(ValueError, match="not created by this client"):
        runtime.discover(a.model_copy(update={"id": "unregistered-source"}))
    assert factory.call_count == 2
    assert first.test.call_count == 2
    first.discover.assert_called_once_with(a.id)
    first.close.side_effect = RuntimeError("cleanup failed")
    with pytest.raises(RuntimeError, match="cleanup failed"):
        runtime.close()
    first.close.assert_called_once()
    second.close.assert_called_once()
    runtime.close()
    with pytest.raises(RuntimeError, match="closed"):
        runtime.connect(config())
    with pytest.raises(RuntimeError, match="closed"):
        runtime.discover(a)


def test_failed_registration_closes_connector_without_retaining_source():
    connector = Mock()
    connector.test.side_effect = ValueError("invalid database")
    runtime = runtime_with(Mock(return_value=connector))
    with pytest.raises(ValueError, match="invalid database"):
        runtime.connect(config())
    connector.close.assert_called_once()
    assert not runtime._connectors
    runtime.close()
    connector.close.assert_called_once()
