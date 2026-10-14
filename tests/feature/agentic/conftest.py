"""Shared fixtures: a small temporal snapshot and cutoff-anchored rows."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from nori_rel.feature import CUTOFF_COLUMN

pytest.importorskip("duckdb")

_SAFE_SQL = """
SELECT e.row_id,
       COUNT(ev.event_id) AS n_events,
       AVG(ev.value) AS mean_value
FROM eval_table e
LEFT JOIN events ev
  ON ev.account_id = e.account_id AND ev.timestamp < e.cutoff
GROUP BY e.row_id
"""

_LEAKY_SQL = """
SELECT e.row_id, MAX(ev.value) AS max_soon
FROM eval_table e
LEFT JOIN events ev
  ON ev.account_id = e.account_id
 AND ev.timestamp < e.cutoff + INTERVAL '30 days'
GROUP BY e.row_id
"""

_HIDDEN_LEAK_SQL = """
SELECT e.row_id, s.total AS total_value
FROM eval_table e
LEFT JOIN (
  SELECT account_id, SUM(value) AS total FROM events GROUP BY account_id
) s ON s.account_id = e.account_id
WHERE EXISTS (SELECT 1 FROM events x WHERE x.timestamp < e.cutoff)
"""

_FLAT_SQL = "SELECT e.row_id, 1.0 AS one FROM eval_table e"


@pytest.fixture
def snapshot() -> SimpleNamespace:
    return SimpleNamespace(
        tables={
            "accounts": pd.DataFrame(
                {"account_id": [1, 2, 3], "region": ["west", "east", "west"]}
            ),
            "events": pd.DataFrame(
                {
                    "event_id": list(range(12)),
                    "account_id": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
                    "timestamp": pd.date_range("2020-01-01", periods=12, tz="UTC"),
                    "value": [float(i + 1) for i in range(12)],
                }
            ),
        },
        primary_keys={"accounts": ["account_id"], "events": ["event_id"]},
        foreign_keys=[("events", "account_id", "accounts", "account_id")],
        time_columns={"events": "timestamp"},
    )


def _anchors(count: int, *, start: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "account_id": [1 + index % 3 for index in range(count)],
            CUTOFF_COLUMN: pd.date_range(start, periods=count, tz="UTC"),
        }
    )


@pytest.fixture
def anchors():
    return _anchors


@pytest.fixture
def sql() -> dict[str, str]:
    return {
        "safe": _SAFE_SQL,
        "leaky": _LEAKY_SQL,
        "hidden": _HIDDEN_LEAK_SQL,
        "flat": _FLAT_SQL,
    }
