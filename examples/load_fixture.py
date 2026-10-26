"""Load the synthetic users and orders into a PostgreSQL schema.

Usage:
    python examples/load_fixture.py [--schema nori_rel_example]

Set DATABASE_URL privately to a disposable PostgreSQL database where your
identity can create schemas. The schema is dropped first if it already exists.
The users table gains a nullable ``income`` column that one user in six lacks,
so a static target (``PREDICT users.income FOR EACH users.user_id``) has rows
to score. The script prints the row counts and the newest order time, which is
the completeness watermark (``observed_through``) for this data. The agent
skills in skills/ and tests/test_skill_scripts.py use this loader.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthetic_data import make_customer_order_fixture

USERS_DDL = (
    "CREATE TABLE users (user_id bigint PRIMARY KEY, "
    "created_at timestamptz NOT NULL, country text NOT NULL, income numeric(12, 2))"
)
ORDERS_DDL = (
    "CREATE TABLE orders (order_id bigint PRIMARY KEY, "
    "user_id bigint NOT NULL REFERENCES users(user_id), "
    "event_time timestamptz NOT NULL, amount numeric(12, 2) NOT NULL)"
)


def make_income(user_id: int, country: str) -> Decimal | None:
    """Return a stored income for five users in six, and None for the sixth."""
    if user_id % 6 == 0:
        return None
    return Decimal(30000 + 900 * user_id + (250 if country == "US" else 0))


def load_fixture(
    database: psycopg.Connection, schema_name: str
) -> tuple[int, int, datetime]:
    """Create the schema and both tables; return the counts and newest order."""
    users, orders = make_customer_order_fixture()
    schema = sql.Identifier(schema_name)
    database.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(schema))
    database.execute(sql.SQL("CREATE SCHEMA {}").format(schema))
    database.execute(sql.SQL("SET search_path TO {}").format(schema))
    database.execute(USERS_DDL)
    database.execute(ORDERS_DDL)
    with database.cursor() as cursor:
        cursor.executemany(
            "INSERT INTO users VALUES (%s, %s, %s, %s)",
            [(*user, make_income(user[0], user[2])) for user in users],
        )
        cursor.executemany("INSERT INTO orders VALUES (%s, %s, %s, %s)", orders)
    newest = max(order[2] for order in orders)
    return len(users), len(orders), newest


def main() -> int:
    """Load the fixture into the schema named on the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--schema", default="nori_rel_example")
    arguments = parser.parse_args()
    options = conninfo_to_dict(os.environ["DATABASE_URL"])
    options.setdefault("sslmode", "prefer")
    with psycopg.connect(**options, autocommit=True) as database:
        users, orders, newest = load_fixture(database, arguments.schema)
    print(f"schema {arguments.schema}: {users} users, {orders} orders")
    print(f"newest order {newest:%Y-%m-%d}: use it as observed_through")
    return 0


if __name__ == "__main__":
    sys.exit(main())
