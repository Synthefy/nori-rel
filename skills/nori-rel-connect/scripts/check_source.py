"""Connect to a database described by nori_rel_source.json and check its schema.

Usage:
    python check_source.py nori_rel_source.json [--entity TABLE]

Prints every selected table with its key, time column, links and columns, then
checks the schema against what a Nori-Rel prediction needs. Exit code 1 means a
check failed. Credentials are read from the environment or AWS Secrets Manager,
as the source file says; nothing is written anywhere.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nori_rel import (
    AwsSecret,
    DatabricksConnection,
    ForeignKeySchema,
    NoriRelClient,
    SnowflakeConnection,
)

TIME_TYPES = ("timestamp", "date", "datetime")


def load_json(path: Path) -> dict:
    """Return the JSON object stored at ``path``."""
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def make_credential(source: dict):
    """Return the credential reference described by the source file."""
    credential = source["credential"]
    if "environment" in credential:
        from nori_rel import EnvironmentCredential

        return EnvironmentCredential(variable=credential["environment"])
    if "aws_secret" in credential:
        return AwsSecret(**credential["aws_secret"])
    raise ValueError('credential needs "environment" or "aws_secret"')


def connect_source(client: NoriRelClient, source: dict):
    """Register the database described by a nori_rel_source.json file."""
    engine = source["engine"]
    connection = None
    if engine == "snowflake":
        connection = SnowflakeConnection(**source["connection"])
    elif engine == "databricks":
        connection = DatabricksConnection(**source["connection"])
    foreign_keys = {
        table: [ForeignKeySchema(**key) for key in keys]
        for table, keys in (source.get("foreign_keys") or {}).items()
    }
    return client.connect(
        engine=engine,
        credential=make_credential(source),
        schema_name=source.get("schema_name"),
        tables=source.get("tables"),
        time_columns=source.get("time_columns") or {},
        primary_keys=source.get("primary_keys") or {},
        foreign_keys=foreign_keys,
        connection=connection,
    )


def is_time_type(data_type: str) -> bool:
    """Return whether a column type holds a date or timestamp."""
    return any(kind in data_type.lower() for kind in TIME_TYPES)


def print_schema(schema) -> None:
    """Print each table's key, time column, links and columns."""
    print(f"schema {schema.schema_name}: {len(schema.tables)} tables")
    capabilities = schema.capabilities
    print(
        f"snapshot: {capabilities.snapshot_mode}; "
        f"read-only: {capabilities.read_only_mode}"
    )
    for table in schema.tables:
        links = ", ".join(
            f"{'+'.join(key.columns)} -> {key.referred_table}"
            for key in table.foreign_keys
        )
        print(f"\n{table.name}")
        print(f"  key: {'+'.join(table.primary_key) or '(none)'}")
        print(f"  time column: {table.time_column or '(none: entity table)'}")
        print(f"  links: {links or '(none)'}")
        columns = ", ".join(f"{c.name} {c.data_type}" for c in table.columns)
        print(f"  columns: {columns}")


def check_schema(schema, source: dict, entity: str | None) -> list[str]:
    """Return the failures found; print every check as PASS, WARN or FAIL."""
    failures: list[str] = []
    tables = {table.name: table for table in schema.tables}
    referenced = {key.referred_table for t in schema.tables for key in t.foreign_keys}

    def report(level: str, text: str) -> None:
        print(f"{level}: {text}")
        if level == "FAIL":
            failures.append(text)

    for table in schema.tables:
        if len(table.primary_key) != 1:
            report(
                "FAIL",
                f"{table.name} needs a single-column primary key; "
                f"it has {table.primary_key or 'none'}",
            )
        for key in table.foreign_keys:
            if len(key.columns) != 1:
                report("FAIL", f"{table.name} link {key.columns} must use one column")
            if key.referred_table not in tables:
                report(
                    "WARN",
                    f"{table.name}.{key.columns[0]} points to {key.referred_table}, "
                    "which is not selected; the link is unused",
                )
        if not table.foreign_keys and table.name not in referenced:
            report("WARN", f"{table.name} is linked to nothing and will be ignored")
        registered = (source.get("time_columns") or {}).get(table.name)
        time_columns = [c.name for c in table.columns if is_time_type(c.data_type)]
        if registered:
            if registered not in {c.name for c in table.columns}:
                report("FAIL", f"{table.name}.{registered} does not exist")
            elif registered not in time_columns:
                report("FAIL", f"{table.name}.{registered} is not a date or timestamp")
            else:
                report("PASS", f"{table.name} is an event table on {registered}")
        elif time_columns and table.name != entity:
            report(
                "WARN",
                f"{table.name} has {', '.join(time_columns)} but no registered "
                "time column; register one if its rows are events, otherwise "
                "its rows are treated as immutable facts",
            )
    if entity:
        if entity not in tables:
            report("FAIL", f"entity table {entity} is not selected")
        else:
            table = tables[entity]
            created = [c.name for c in table.columns if is_time_type(c.data_type)]
            report(
                "PASS" if created else "WARN",
                f"{entity} creation-time candidates: {', '.join(created) or 'none'}"
                + ("" if created else " (add one: each entity's first event time)"),
            )
            direct = [
                t.name
                for t in schema.tables
                if t.time_column
                and any(k.referred_table == entity for k in t.foreign_keys)
            ]
            report(
                "PASS" if direct else "FAIL",
                f"event tables linked straight to {entity}: "
                f"{', '.join(direct) or 'none (a target needs one)'}",
            )
    return failures


def main() -> int:
    """Run the connection test, discovery and checks."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", type=Path, help="nori_rel_source.json")
    parser.add_argument("--entity", help="table the prediction is FOR EACH row of")
    arguments = parser.parse_args()
    source = load_json(arguments.source)
    with NoriRelClient() as client:
        database = connect_source(client, source)
        status = client.test_connection(database)
        print(f"connection: {status.status} ({status.latency_ms} ms)")
        if status.error:
            print(f"FAIL: {status.error}")
            return 1
        schema = client.discover(database)
    print_schema(schema)
    print()
    failures = check_schema(schema, source, arguments.entity)
    print(f"\n{len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
