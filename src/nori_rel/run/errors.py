"""Errors a run's result raises, each carrying the run's progress so far."""

from __future__ import annotations

from nori_rel.run.progress import RunProgress


class RunError(Exception):
    """A run ended without a result, or the caller stopped waiting for one.

    ``progress`` holds what the run had done: its status, stage, trials, spend
    and any error, so the caller can report or retry without another lookup.
    """

    def __init__(self, message: str, *, progress: RunProgress) -> None:
        """Keep ``message`` and the run's ``progress`` when the error was raised."""
        super().__init__(message)
        self.progress = progress

    def __reduce__(self) -> tuple:
        """Rebuild with ``progress``, which ``__init__`` takes by keyword only."""
        return (_rebuild_run_error, (type(self), str(self), self.progress))

    @property
    def run_id(self) -> str:
        """The run this error is about."""
        return self.progress.run_id


def _rebuild_run_error(
    error_type: type[RunError], message: str, progress: RunProgress
) -> RunError:
    """Return a ``RunError`` of ``error_type``, for ``pickle`` and ``copy``."""
    return error_type(message, progress=progress)


class PredictionCancelled(RunError):
    """The run was cancelled; it has no predictions."""


class PredictionFailed(RunError):
    """The run failed; ``error_kind`` names the failure."""

    @property
    def error_kind(self) -> str | None:
        """The failure's kind, such as ``permission_denied``."""
        return self.progress.error_kind


class RunTimeout(RunError):
    """The caller's wait ended; the run itself is still going."""


__all__ = ["PredictionCancelled", "PredictionFailed", "RunError", "RunTimeout"]
