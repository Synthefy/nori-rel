import time

import pytest

from nori_rel.run import redact
from nori_rel.run.redact import MAX_SCANNED_CHARS


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "could not connect to postgresql://reader:hunter2@db.internal/app",
            "could not connect to postgresql://***@db.internal/app",
        ),
        ("401: Bearer abc.def.ghi rejected", "401: Bearer *** rejected"),
        (
            "bad api_key=sk-live-123 and password: hunter2",
            "bad api_key=*** and password: ***",
        ),
        ("key sk-ant-REDACTMEREDACTME1234 is invalid", "key *** is invalid"),
        ("token " + "a" * 48 + " expired", "token *** expired"),
        (
            'invalid input syntax for type integer: "4111-1111"',
            'invalid input syntax for type integer: "?"',
        ),
        (
            'could not convert string to float: "O\'Brien"',
            'could not convert string to float: "?"',
        ),
        (
            'Unable to parse string "x1" at position 0',
            'Unable to parse string "?" at position 0',
        ),
        (
            'Binder Error: Referenced column "amout" not found\n'
            "LINE 1: SELECT AVG(amout) AS gap FROM orders\n               ^",
            'Binder Error: Referenced column "amout" not found (SQL omitted)',
        ),
        (
            "Key (email)=(jane@example.com) already exists.",
            "Key (email)=(?) already exists.",
        ),
        (
            'column "amount" of relation "sales.orders" where note = \'secret value\'',
            'column "amount" of relation "sales.orders" where note = \'?\'',
        ),
        (
            "DETAIL: Failing row contains (3, jane@example.com, 4111111111111111).",
            "DETAIL: Failing row contains (?).",
        ),
        (
            'value "9999999999" is out of range for type integer',
            'value "?" is out of range for type integer',
        ),
        (
            'date/time field value out of range: "2026-13-45"',
            'date/time field value out of range: "?"',
        ),
        (
            'could not convert string to float: "O\'Brien"',
            'could not convert string to float: "?"',
        ),
        ("where note = E'secret value'", "where note = '?'"),
        ("PGPASSWORD=hunter2 was refused", "PGPASSWORD=*** was refused"),
        ("DATABASE_PASSWORD=x", "DATABASE_PASSWORD=***"),
        ("OPENAI_API_KEY=abc123 is invalid", "OPENAI_API_KEY=*** is invalid"),
        ("access_token=abc expired", "access_token=*** expired"),
        ('{"password": "hunter2"}', '{"password": "?"}'),
        (
            "secret wJalrXUtnFEMIK7MDENGbPxRfiCYEXAMPLEKEY1234 leaked",
            "secret *** leaked",
        ),
        (
            "key hf_" + "a1B2" * 9 + " and sk_live_" + "Z9" * 10 + " leaked",
            "key *** and *** leaked",
        ),
        (
            "LINE 1: SELECT * FROM users WHERE ssn = '123-45-6789'\n ^",
            "(SQL omitted)",
        ),
    ],
)
def test_credentials_and_values_are_removed(text, expected) -> None:
    assert redact(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Permission denied when reading sales.orders; can't continue.",
        'duplicate key value violates unique constraint "pk_sales_orders_id"',
        "column orders__amount__sum__last_30_days__by_customer_segment is missing",
        "no such file /data/warehouse/exports/orders_2026_part_000001.parquet",
        "prompt_tokens: 5 and tokens: 7",
    ],
)
def test_names_paths_and_prose_are_kept(text) -> None:
    assert redact(text) == text


@pytest.mark.parametrize(
    "text", ["a1" * 500_000, "'x" * 500_000, "x://" + "a" * 1_000_000]
)
def test_long_text_is_redacted_quickly(text) -> None:
    started = time.perf_counter()
    redact(text)

    assert time.perf_counter() - started < 0.5


def test_only_the_start_of_long_text_is_scanned() -> None:
    words = "word " * (MAX_SCANNED_CHARS // 5)

    assert redact(words + "tail", max_length=10**6) == words.strip()
