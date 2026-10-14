"""Fakes shared by the run-control tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta


class FakeClock:
    """A UTC clock that moves only when a test advances it."""

    def __init__(self, start: datetime | None = None) -> None:
        """Start at ``start``, or at a fixed moment in October 2026."""
        self.now = start or datetime(2026, 10, 7, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        """Return the current fake time."""
        return self.now

    def advance(self, seconds: float) -> None:
        """Move the clock forward by ``seconds``."""
        self.now += timedelta(seconds=seconds)
