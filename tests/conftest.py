"""Fixtures shared by the whole test tree."""

import os
from types import SimpleNamespace

import pytest

FAKE_CHECKPOINT_REVISION = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture
def fake_checkpoint(tmp_path, monkeypatch):
    """Serve a small file in the hub cache layout as the Nori checkpoint.

    Returns ``path`` and ``requests``, each (model, revision) asked for.
    """
    path = (
        tmp_path
        / "hub"
        / "models--Synthefy--Nori-30M"
        / "snapshots"
        / FAKE_CHECKPOINT_REVISION
        / "model.pt"
    )
    path.parent.mkdir(parents=True)
    path.write_bytes(b"fake nori checkpoint")
    requests = []

    def download(model, revision):
        requests.append((model, revision))
        if revision not in (None, FAKE_CHECKPOINT_REVISION):
            raise FileNotFoundError(f"revision {revision} is not cached")
        return str(path)

    monkeypatch.setattr("nori_rel._checkpoint.get_checkpoint_path", download)
    return SimpleNamespace(path=path, requests=requests)


def pytest_addoption(parser):
    parser.addoption(
        "--require-local-model",
        action="store_true",
        help="Require the runtime and cached weights for real local inference checks.",
    )
    parser.addoption(
        "--require-hosted-model",
        action="store_true",
        help="Require credentials for real hosted inference checks.",
    )


@pytest.fixture(scope="session")
def hosted_api_key(request):
    """Require an explicit test credential before any hosted smoke fixture runs."""
    api_key = os.environ.get("NORI_REL_TEST_HOSTED_API_KEY", "").strip()
    if not api_key:
        message = "NORI_REL_TEST_HOSTED_API_KEY is required for hosted inference checks"
        if request.config.getoption("--require-hosted-model"):
            pytest.fail(message, pytrace=False)
        pytest.skip("opt-in real hosted Nori smoke test")
    return api_key
