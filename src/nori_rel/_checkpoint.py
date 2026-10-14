"""Find, pin and hash the local Nori checkpoint file.

The database client imports this module without loading the local Nori runtime.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

#: A Hugging Face revision is the 40-character commit hash of the repo.
_COMMIT_HASH = re.compile(r"^[0-9a-f]{40}$")
_SNAPSHOTS_DIR = "snapshots"


@dataclass(frozen=True)
class NoriCheckpoint:
    """One local checkpoint file and where it came from.

    ``revision`` is the Hugging Face commit of the file, read from the hub
    cache layout ``.../snapshots/<commit>/<filename>``. It is ``None`` for a
    file outside that layout, which a pipeline cannot pin.
    """

    model: str
    repo_id: str
    filename: str
    path: str
    revision: str | None

    @property
    def sha256(self) -> str:
        """Return the SHA-256 of the file, computed once per file version."""
        path = Path(self.path)
        stat = path.stat()
        return get_checkpoint_sha256(path, stat.st_size, stat.st_mtime_ns)


def get_checkpoint_path(model: str, revision: str | None) -> str:
    """Return the cached checkpoint path, downloading it on first use."""
    from synthefy_nori.hf import download_checkpoint

    return download_checkpoint(model=model, revision=revision)


def get_pinned_checkpoint(model: str, revision: str | None = None) -> NoriCheckpoint:
    """Return the checkpoint of ``model`` at ``revision``, or at the latest one.

    With ``HF_HUB_OFFLINE=1`` the file must already be in the hub cache.
    """
    from synthefy_nori.hf import resolve_model_repo

    path = Path(get_checkpoint_path(model, revision))
    commit = path.parent.name
    found = (
        commit
        if path.parent.parent.name == _SNAPSHOTS_DIR and _COMMIT_HASH.match(commit)
        else None
    )
    if revision is not None and found != revision:
        raise ValueError(
            f"asked for revision {revision} of {model!r}, got the file at {path}"
        )
    return NoriCheckpoint(
        model=model,
        repo_id=resolve_model_repo(model),
        filename=path.name,
        path=str(path),
        revision=found,
    )


@lru_cache(maxsize=4)
def get_checkpoint_sha256(path: Path, size: int, modified_ns: int) -> str:
    """Hash a checkpoint once per file version (path, size and mtime)."""
    del size, modified_ns
    with path.open("rb") as checkpoint_file:
        return hashlib.file_digest(checkpoint_file, "sha256").hexdigest()


__all__ = [
    "NoriCheckpoint",
    "get_checkpoint_path",
    "get_checkpoint_sha256",
    "get_pinned_checkpoint",
]
