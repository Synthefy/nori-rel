"""PostgreSQL to local Nori: no hosted request and no off-host network access."""

import hashlib
import socket
from datetime import UTC, datetime

import numpy as np
import pytest

from nori_rel import EnvironmentCredential, NoriRelClient, RelationalLimits

QUERY = "PREDICT COUNT(orders.*,0,30,days) FOR users.user_id=99"
OPTIONS = dict(
    anchor_times=[datetime(2026, 1, 1, tzinfo=UTC)],
    anchor_time=datetime(2026, 2, 1, tzinfo=UTC),
    observed_through=datetime(2026, 2, 1, tzinfo=UTC),
    entity_created_at="created_at",
)
LOOPBACK = {"127.0.0.1", "::1", "localhost"}


class FakeRegressor:
    def __init__(self, **options) -> None:
        self.fits = []

    def get_params(self):
        return {"augmentations": ("yj",)}

    def fit(self, train, target):
        self.fits.append((train, target))
        return self

    def predict(self, query, *, output_type="mean", quantiles=None):
        return np.full(len(query), 2.5)


@pytest.fixture
def loopback_only(monkeypatch):
    """Allow the local PostgreSQL connection; fail any hosted or remote request."""
    connect = socket.socket.connect

    def guarded(sock, address):
        if isinstance(address, tuple) and address[0] not in LOOPBACK:
            raise AssertionError(f"prediction opened a network connection: {address}")
        return connect(sock, address)

    def refuse(*args, **kwargs):
        raise AssertionError("local prediction created a hosted Nori client")

    monkeypatch.setattr(socket.socket, "connect", guarded)
    monkeypatch.setattr("nori_rel._runtime.SynthefyNoriClient", refuse)


def _connect(client, source):
    config = source[0].config
    return client.connect(
        engine="postgresql",
        credential=EnvironmentCredential(variable="NORI_REL_TEST_DATABASE_URL"),
        schema_name=config.schema_name,
        tables=config.tables,
        time_columns=config.time_columns,
    )


def test_postgres_to_local_nori_makes_no_hosted_request(
    source, monkeypatch, loopback_only, fake_checkpoint
):
    monkeypatch.delenv("SYNTHEFY_NORI_API_KEY", raising=False)
    monkeypatch.setattr("synthefy_nori.NoriRegressor", FakeRegressor)
    with NoriRelClient(limits=RelationalLimits(min_training_rows=2)) as client:
        database = _connect(client, source)
        result = client.predict(
            source=database, query=QUERY, **OPTIONS, return_details=True
        )
        regressor = client._runtime.predictor._regressor
        with pytest.raises(ValueError, match="regression PQL targets only"):
            client.predict(
                source=database, query=QUERY.replace(") FOR", ")>0 FOR"), **OPTIONS
            )

    assert result["predictions"]["prediction"].tolist() == [2.5]
    assert result["metadata"]["model"] == "nori-30m"
    model = result["pipeline"].manifest.model
    assert model.mode == "local"
    assert model.revision == fake_checkpoint.path.parent.name
    assert model.sha256 == hashlib.sha256(fake_checkpoint.path.read_bytes()).hexdigest()
    assert result["metadata"]["pipeline_id"] == result["pipeline"].pipeline_id
    assert len(regressor.fits) == 1
    assert regressor.fits[0][1].tolist() == [2.0, 0.0, 1.0]


@pytest.mark.live_nori
def test_offline_postgres_to_cached_local_nori(
    source, monkeypatch, loopback_only, request
):
    """Run real Nori weights from the Hugging Face cache, offline."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_OFFLINE", True)
    try:
        from synthefy_nori.hf import download_checkpoint

        download_checkpoint(model="nori-30m")
    except (ImportError, OSError):
        if request.config.getoption("--require-local-model"):
            pytest.fail(
                "local runtime and cached nori-30m weights are required", pytrace=False
            )
        pytest.skip("local runtime or nori-30m weights are not in the cache")
    with NoriRelClient(
        device="cpu", limits=RelationalLimits(min_training_rows=2)
    ) as client:
        frame = client.predict(source=_connect(client, source), query=QUERY, **OPTIONS)

    assert frame["entity_id"].tolist() == [99]
    assert np.isfinite(frame["prediction"].astype(float)).all()
