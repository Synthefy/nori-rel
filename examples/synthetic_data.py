"""Generate a deterministic two-table fixture without downloading any data."""

from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

SCORING_TIME = datetime(2026, 3, 2, tzinfo=UTC)
HISTORICAL_ANCHOR_TIMES = [
    SCORING_TIME - timedelta(days=37 + 7 * week) for week in range(8)
]
CUSTOMER_SPEND_QUERY = "PREDICT SUM(orders.amount,0,30,days) FOR EACH users.user_id"
USER_COLUMNS = ("user_id", "created_at", "country")
ORDER_COLUMNS = ("order_id", "user_id", "event_time", "amount")


def make_customer_order_fixture() -> tuple[list[tuple], list[tuple]]:
    """Return 48 users and their weekly orders, all before the scoring cutoff."""
    created_at = datetime(2025, 8, 1, tzinfo=UTC)
    countries = ("US", "CA", "DE")
    users = [
        (user_id, created_at, countries[(user_id - 1) % len(countries)])
        for user_id in range(1, 49)
    ]
    orders = []
    first_week = datetime(2025, 9, 1, tzinfo=UTC)
    for week in range(26):
        for user_id, _, _ in users:
            event_time = first_week + timedelta(weeks=week, days=user_id % 4, hours=12)
            amount = Decimal(1000 + 37 * user_id + 19 * (week % 5)) / 100
            orders.append((len(orders) + 1, user_id, event_time, amount))
    return users, orders


def write_fixture_csvs(directory: Path) -> None:
    """Write portable UTF-8 CSVs with timezone-aware ISO-8601 timestamps."""
    users, orders = make_customer_order_fixture()
    directory.mkdir(parents=True, exist_ok=True)
    for name, columns, rows in (
        ("users", USER_COLUMNS, users),
        ("orders", ORDER_COLUMNS, orders),
    ):
        with (directory / f"{name}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.writer(file)
            writer.writerow(columns)
            writer.writerows(
                tuple(
                    value.isoformat() if isinstance(value, datetime) else value
                    for value in row
                )
                for row in rows
            )
        print(f"{name}: {len(rows)} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    write_fixture_csvs(parser.parse_args().directory)
