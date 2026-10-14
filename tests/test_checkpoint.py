import hashlib

import pytest

from nori_rel._checkpoint import get_pinned_checkpoint

REVISION = "0123456789abcdef0123456789abcdef01234567"


def test_revision_and_hash_come_from_the_cached_file(fake_checkpoint) -> None:
    checkpoint = get_pinned_checkpoint("nori-30m")

    assert checkpoint.revision == REVISION
    assert checkpoint.repo_id == "Synthefy/Nori-30M"
    assert checkpoint.filename == "model.pt"
    assert checkpoint.sha256 == hashlib.sha256(b"fake nori checkpoint").hexdigest()


def test_a_file_outside_the_hub_cache_has_no_revision(tmp_path, monkeypatch) -> None:
    path = tmp_path / "fine-tuned.pt"
    path.write_bytes(b"weights")
    monkeypatch.setattr(
        "nori_rel._checkpoint.get_checkpoint_path", lambda *_: str(path)
    )

    assert get_pinned_checkpoint("nori-30m").revision is None
    with pytest.raises(ValueError, match="asked for revision"):
        get_pinned_checkpoint("nori-30m", REVISION)
