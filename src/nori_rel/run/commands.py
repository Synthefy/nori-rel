"""Commands sent to a run, queued until the run reaches its next boundary."""

from __future__ import annotations

import threading
import uuid

from pydantic import BaseModel, ConfigDict, Field

from nori_rel.run.models import (
    Command,
    CommandAcceptedResponse,
    CommandReceipt,
    CommandRejectedResponse,
)

#: Steering text longer than this is refused when it is sent: it is echoed in
#: the run's events, replayed to every reader, and sent to the model.
MAX_STEERING_TEXT_CHARS = 4_000


class QueuedCommand(BaseModel):
    """A recorded command waiting for the run's next boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    command_id: str = Field(min_length=1)
    command: Command
    steering_text: str | None = None


class CommandInbox:
    """Thread-safe queue of one run's commands.

    This is the only run object a caller, transport or signal handler touches;
    the pipeline drains it at boundaries. Once the run ends the inbox is closed:
    a command is then answered with a rejected receipt and never recorded.
    """

    def __init__(self, run_id: str) -> None:
        """Start an empty queue for ``run_id``."""
        self.run_id = run_id
        self._queued: list[QueuedCommand] = []
        self._closed = False
        self._condition = threading.Condition()

    def submit_command(
        self, command: Command, steering_text: str | None = None
    ) -> CommandReceipt:
        """Record ``command``; it is applied, or rejected, at the next boundary.

        ``steer`` needs non-blank ``steering_text`` of at most
        ``MAX_STEERING_TEXT_CHARS`` characters; other commands take none.
        A closed inbox records nothing and returns a rejected receipt.
        """
        if (command == "steer") != (steering_text is not None):
            raise ValueError("steer needs steering_text; other commands take none")
        if steering_text is not None and not steering_text.strip():
            raise ValueError("steering_text must not be blank")
        if steering_text is not None and len(steering_text) > MAX_STEERING_TEXT_CHARS:
            raise ValueError(
                f"steering_text must be at most {MAX_STEERING_TEXT_CHARS} characters"
            )
        queued = QueuedCommand(
            command_id=f"cmd_{uuid.uuid4().hex[:16]}",
            command=command,
            steering_text=steering_text,
        )
        with self._condition:
            if self._closed:
                return CommandRejectedResponse(
                    run_id=self.run_id, command_id=queued.command_id
                )
            self._queued.append(queued)
            self._condition.notify_all()
        return CommandAcceptedResponse(run_id=self.run_id, command_id=queued.command_id)

    def take_all_commands(self) -> list[QueuedCommand]:
        """Remove and return every queued command, oldest first."""
        with self._condition:
            queued, self._queued = self._queued, []
        return queued

    def close_and_take_all_commands(self) -> list[QueuedCommand]:
        """Close the inbox and return every queued command, oldest first.

        Closing and taking are one step, so a command is either returned here
        or answered with a rejected receipt; none is accepted and then missed.
        """
        with self._condition:
            self._closed = True
            queued, self._queued = self._queued, []
        return queued

    def wait_for_command(self, timeout_seconds: float | None = None) -> bool:
        """Block until a command is queued; return whether one is.

        Only a command wakes the waiter, so a paused run is ended by sending it
        a cancel or a stop.
        """
        with self._condition:
            return self._condition.wait_for(
                lambda: bool(self._queued), timeout=timeout_seconds
            )


__all__ = ["MAX_STEERING_TEXT_CHARS", "CommandInbox", "QueuedCommand"]
