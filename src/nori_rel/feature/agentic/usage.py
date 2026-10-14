"""Token counts and dollar cost for one agent search.

A search makes many model calls, and each call re-sends the conversation so
far. What it costs depends on how much of that prefix the provider served from
its prompt cache, so the counts keep cache reads and cache writes apart from
plain input. Prices come from the run's configuration, never from this module,
so a price change is a config edit rather than a code change.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

_PRICE_PER_MILLION_TO_PRICE_PER_TOKEN = 1e-6


class AgentStopReason(StrEnum):
    """Local stop conditions; provider stop reasons are retained as strings."""

    MODEL_FINISHED = "model_finished"
    MAX_TURNS = "max_turns"
    MAX_COST = "max_cost"
    INVALID_RESPONSE = "invalid_response"
    #: The run asked the search to end: a stop, or a runtime or spend limit.
    STOPPED = "stopped"


@dataclass(frozen=True)
class TokenPrices:
    """US dollars per million tokens for one model.

    ``cache_write_5m`` and ``cache_write_1h`` are the prices for writing a
    prompt-cache entry that lives five minutes or one hour.
    """

    input: float
    output: float
    cache_read: float
    cache_write_5m: float
    cache_write_1h: float

    def __post_init__(self) -> None:
        """Require finite positive prices before applying a dollar budget."""
        if any(
            not math.isfinite(value) or value <= 0 for value in asdict(self).values()
        ):
            raise ValueError("token prices must be finite and positive")

    @classmethod
    def from_mapping(cls, prices: dict[str, Any]) -> TokenPrices:
        """Build prices from a config mapping with exactly these five keys."""
        expected = {
            "input",
            "output",
            "cache_read",
            "cache_write_5m",
            "cache_write_1h",
        }
        if set(prices) != expected:
            raise ValueError(
                f"prices need exactly {sorted(expected)}, got {sorted(prices)}"
            )
        return cls(**{key: float(value) for key, value in prices.items()})


@dataclass
class TokenUsage:
    """Running token totals over every model call in one search."""

    calls: int = 0
    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_5m_tokens: int = 0
    cache_write_1h_tokens: int = 0
    output_tokens: int = 0

    def add_anthropic(self, usage: Any) -> None:
        """Add one Anthropic response's ``usage`` block.

        ``input_tokens`` counts only the uncached part of the prompt. Cache
        writes are split by lifetime when the response says so; otherwise the
        whole write is counted at the five-minute price.
        """
        self.calls += 1
        self.input_tokens += usage.input_tokens or 0
        self.cache_read_tokens += usage.cache_read_input_tokens or 0
        self.output_tokens += usage.output_tokens or 0
        written = usage.cache_creation_input_tokens or 0
        split = getattr(usage, "cache_creation", None)
        if split is not None:
            one_hour = split.ephemeral_1h_input_tokens or 0
            five_minute = split.ephemeral_5m_input_tokens or 0
            self.cache_write_1h_tokens += one_hour
            self.cache_write_5m_tokens += five_minute
            # Any write the split does not account for is billed at 5 minutes.
            self.cache_write_5m_tokens += max(written - one_hour - five_minute, 0)
        else:
            self.cache_write_5m_tokens += written

    def cost(self, prices: TokenPrices) -> float:
        """Return the dollar cost of these totals at ``prices``."""
        return _PRICE_PER_MILLION_TO_PRICE_PER_TOKEN * (
            self.input_tokens * prices.input
            + self.output_tokens * prices.output
            + self.cache_read_tokens * prices.cache_read
            + self.cache_write_5m_tokens * prices.cache_write_5m
            + self.cache_write_1h_tokens * prices.cache_write_1h
        )

    def cache_hit_fraction(self) -> float:
        """Return the share of prompt tokens served from the cache."""
        prompt = (
            self.input_tokens
            + self.cache_read_tokens
            + self.cache_write_5m_tokens
            + self.cache_write_1h_tokens
        )
        return self.cache_read_tokens / prompt if prompt else 0.0

    def as_dict(self) -> dict[str, int]:
        """Return the counts as a plain dictionary for a result row."""
        return asdict(self)


@dataclass(frozen=True)
class AgentRunResult:
    """Per-run accounting, never stored on a reusable provider agent."""

    model: str
    usage: TokenUsage | None
    cost_usd: float | None
    stop_reason: str
    continuation_requests: int = 0
    response_count: int = 0


__all__ = ["AgentRunResult", "AgentStopReason", "TokenPrices", "TokenUsage"]
