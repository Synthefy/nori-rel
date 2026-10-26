---
name: nori-rel-connect
description: Connect a PostgreSQL (including AWS RDS or Aurora), Snowflake or Databricks database to Nori-Rel with a read-only identity, register each event table's time column, check keys, links and creation times with check_source.py, and write the nori_rel_source.json the other nori-rel-* skills read. Use when someone wants to point Nori-Rel at their database, asks why connect() or discover() fails, or needs to know whether their tables are ready for a prediction.
---

# Connect a database to Nori-Rel

Nori-Rel reads a bounded snapshot of linked tables through a read-only
identity, in your process. This skill gets the connection right, checks that
the schema can carry a prediction, and leaves a `nori_rel_source.json` for
[nori-rel-predict](../nori-rel-predict/SKILL.md). It changes nothing in the
database.

## What a schema needs

1. Every table has a one-column primary key.
2. Relationships are declared single-column foreign keys. Nori-Rel follows
   only declared keys; a table nothing links to is ignored.
3. Each **event table** (orders, visits, readings) has one time column, which
   you register in `time_columns`. Rows of unregistered tables are treated as
   facts that were always true.
4. The **entity table** you predict for (customers, machines) has a column
   that says when each row first existed, such as `created_at`.
5. The event table in the target links straight to the entity table.
6. Entity columns do not change over time. A `lifetime_value` or
   `is_active` column on the entity table leaks the answer; see the leakage
   section of [nori-rel-prepare-data](../nori-rel-prepare-data/SKILL.md).

Snowflake needs standard tables with Time Travel history; Databricks needs
Unity Catalog Delta tables with retained versions. Views are not supported on
any engine.

## 1. Ask for the connection, not the secret

Ask in one message:

- the engine, and the schema that holds the tables;
- the tables the prediction needs (a short allowlist reads faster and stays
  under the limits: 32 tables, 1,000,000 rows per table, 3,000,000 in all);
- which of them are event tables, and their time column;
- the entity table and its creation column, if known;
- how the credential is held: the name of an environment variable, or an AWS
  Secrets Manager secret id and region.

Have the user set the variable in the shell that will run Python. Never ask
for a password in chat, and never write one into a file. PostgreSQL requires
TLS: `sslmode=require` at least, `sslmode=verify-full` with `sslrootcert=` to
verify the server. The identity should be able to read the tables and nothing
more.

```bash
export DATABASE_URL='postgresql://reader:...@db.example.com:5432/shop?sslmode=require'
```

## 2. Write `nori_rel_source.json`

PostgreSQL, with the URL in an environment variable:

```json
{
  "engine": "postgresql",
  "credential": {"environment": "DATABASE_URL"},
  "schema_name": "public",
  "tables": ["users", "orders"],
  "time_columns": {"orders": "event_time"}
}
```

AWS RDS or Aurora PostgreSQL, with the connection in Secrets Manager (a
PostgreSQL URL, or AWS's RDS fields `host`, `port`, `dbname`, `username`,
`password`; a secret RDS manages itself holds only the user and password, so
store a complete one). The AWS identity in the shell needs read access to
that one secret:

```json
{
  "engine": "postgresql",
  "credential": {"aws_secret": {"secret_id": "prod/shop/reader", "region_name": "us-east-1"}},
  "schema_name": "public",
  "tables": ["users", "orders"],
  "time_columns": {"orders": "event_time"}
}
```

Snowflake. The variable holds a JSON object with one authentication shape:
key pair (`user`, `private_key` as PEM text or `private_key_file`,
`authenticator: "SNOWFLAKE_JWT"`), OAuth token (`authenticator: "oauth"`,
`token`) or client credentials. Snowflake records keys without enforcing them,
so declare them here; Nori-Rel checks them against the data:

```json
{
  "engine": "snowflake",
  "credential": {"environment": "SNOWFLAKE_AUTH_JSON"},
  "connection": {"account": "myorg-myaccount", "warehouse": "NORI_READ_WH", "database": "ANALYTICS", "role": "NORI_READER"},
  "schema_name": "PUBLIC",
  "tables": ["users", "orders"],
  "time_columns": {"orders": "event_time"},
  "primary_keys": {"users": ["user_id"], "orders": ["order_id"]},
  "foreign_keys": {"orders": [{"columns": ["user_id"], "referred_table": "users", "referred_columns": ["user_id"]}]}
}
```

Databricks. The variable holds `{"access_token": "..."}` or
`{"client_id": "...", "client_secret": "..."}`:

```json
{
  "engine": "databricks",
  "credential": {"environment": "DATABRICKS_AUTH_JSON"},
  "connection": {"server_hostname": "myworkspace.cloud.databricks.com", "http_path": "/sql/1.0/warehouses/abc123", "catalog": "analytics"},
  "schema_name": "customers",
  "tables": ["users", "orders"],
  "time_columns": {"orders": "event_time"},
  "primary_keys": {"users": ["user_id"], "orders": ["order_id"]},
  "foreign_keys": {"orders": [{"columns": ["user_id"], "referred_table": "users", "referred_columns": ["user_id"]}]}
}
```

Unquoted names match case-insensitively when exactly one table or column
matches (`orders` finds `ORDERS` on Snowflake); Databricks uses the exact
names. Table and column names are not secrets, but keep the file out of a
public repository if the schema is.

## 3. Run the check

```bash
python scripts/check_source.py nori_rel_source.json --entity users
```

The script tests the connection, discovers the schema the way a prediction
will see it, prints each table's key, time column, links and columns, and
reports `PASS`, `WARN` or `FAIL` for each item above. `--entity` names the
table the prediction is `FOR EACH` row of; the script then lists its
creation-time candidates and the event tables linked straight to it. Exit
code 1 means at least one `FAIL`.

Fix every `FAIL` and read every `WARN`:

- A missing or multi-column key, or an undeclared link, is fixed in the
  database (`ALTER TABLE ... ADD PRIMARY KEY`, `ADD FOREIGN KEY`) by someone
  with that right; the reader identity cannot. On Snowflake and Databricks,
  declare the keys in the source file instead.
- A table with a timestamp column but no registered time column is either an
  event table you forgot (register it) or a fact table (leave it, and make
  sure the column is not a changing state).
- An entity with no creation-time candidate needs one: usually each entity's
  first event time, added as a column.
- A table that is linked to nothing does nothing; drop it from `tables`.

When the data is files, or the schema needs reshaping (history columns on the
entity table, child tables with no key to the entity), hand over to
[nori-rel-prepare-data](../nori-rel-prepare-data/SKILL.md): it designs and
loads a schema that passes this check.

## 4. Report

Tell the user, in a short list: the tables and their roles (entity, event with
its time column, fact), the links, the check result, and the path of
`nori_rel_source.json`. Mention what the connection costs: warehouse reads are
billed by the warehouse, every prediction reads the selected tables once, and
the row limits can be raised with
`NoriRelClient(limits=RelationalLimits(max_rows_per_table=...))` at the price
of memory. Then continue with [nori-rel-predict](../nori-rel-predict/SKILL.md).

## Problems

| Message | Cause and fix |
|---|---|
| `provide exactly one of database_url or credential` | The source file needs a `credential` object. |
| `warehouse connections require an explicit schema_name` | Add `schema_name` for Snowflake or Databricks. |
| `database environment variable 'DATABASE_URL' is not set` | Export it in the shell that runs Python. |
| TLS or `sslmode` rejected | PostgreSQL URLs need `sslmode=require` or stricter; non-TLS servers are refused. |
| Ambiguous name on Snowflake | Two names differ only by case; write the exact one in backticks in PQL and exactly in the file. |
| `ConnectorError` naming a key check | A declared key is not unique or a foreign key points to missing parents; fix the data or the declaration. |
| `SnapshotLimitError` | A table exceeds the row or byte limit. Narrow `tables`, or raise `RelationalLimits`. |
