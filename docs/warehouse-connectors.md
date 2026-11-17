# Snowflake and Databricks sources

Nori-Rel supports Snowflake standard tables and Databricks Unity Catalog Delta
tables through SQL warehouses. It discovers columns and relationships, reads a
bounded source snapshot, builds historical labels and features locally, and
runs prediction through the same client used for PostgreSQL.

Install the driver for your source. Local inference also needs the pinned model
runtime:

```bash
python -m pip install "nori-rel[snowflake,local]"
# Or:
python -m pip install "nori-rel[databricks,local]"
```

## Credentials and permissions

Pass nonsecret connection settings in a `SnowflakeConnection` or
`DatabricksConnection`. Pass authentication as an `EnvironmentCredential` or
`AwsSecret` reference whose value is a JSON object. Keep the JSON secret outside
source code. The connector resolves it in your process; it never appears in the
registered source, prediction metadata, or hosted inference request.

Snowflake accepts these authentication shapes:

| Authentication | Secret JSON fields |
| --- | --- |
| Key pair from a file | `user`, `private_key_file`, optional `private_key_file_pwd`; `authenticator="SNOWFLAKE_JWT"` |
| Key pair from PEM | `user`, `private_key`, optional `private_key_password`; `authenticator="SNOWFLAKE_JWT"` |
| OAuth token | `authenticator="oauth"`, `token`, optional `user` |
| OAuth client credentials | `authenticator="OAUTH_CLIENT_CREDENTIALS"`, `oauth_client_id`, `oauth_client_secret`, optional `oauth_token_request_url`, `oauth_scope`, `user` |

Databricks accepts `access_token`, or `client_id` and `client_secret` for an OAuth
service principal. Driver authentication options are described in the official
[Snowflake connector documentation](https://docs.snowflake.com/en/developer-guide/python-connector/python-connector-connect)
and [Databricks SQL connector documentation](https://docs.databricks.com/aws/en/dev-tools/python-sql-connector).
Use exactly one authentication shape with nonempty string values. Unknown or
mixed fields fail local validation with field names and redacted values. For
Snowflake client-credentials OAuth, a supplied token request URL must use HTTPS.
Connection settings belong in the typed connection object, not the secret JSON;
session/TLS overrides are rejected.

Use a dedicated reader identity. Snowflake needs warehouse and database/schema
usage plus SELECT on the allowed tables; Databricks needs USE CATALOG, USE SCHEMA
and SELECT. These identities must not have write privileges. Warehouse connectors
use pinned read queries and reader permissions to enforce the read-only contract;
their capability metadata distinguishes this from PostgreSQL transaction-level
read-only enforcement. TLS verification stays enabled.

## Connect and predict

Both engines require an explicit schema. Set a table allowlist to bound which
tables the source reads. Supply explicit
primary keys and foreign keys when warehouse metadata does not declare them.
Keys are validated because warehouse declarations may be unenforced. Event-time
columns and entity creation times are separate settings.

This example assumes `users(user_id, created_at, ...)` and
`orders(order_id, user_id, event_time, amount, ...)`. Set `SNOWFLAKE_AUTH_JSON` to
one supported authentication JSON object before running it:

```python
from datetime import UTC, datetime

from nori_rel import (
    EnvironmentCredential,
    ForeignKeySchema,
    NoriRelClient,
    SnowflakeConnection,
)

with NoriRelClient() as client:
    source = client.connect(
        engine="snowflake",
        connection=SnowflakeConnection(
            account="your-org-your-account",
            warehouse="NORI_READ_WH",
            database="ANALYTICS",
            role="NORI_READER",
        ),
        credential=EnvironmentCredential(variable="SNOWFLAKE_AUTH_JSON"),
        schema_name="PUBLIC",
        tables=["users", "orders"],
        time_columns={"orders": "event_time"},
        primary_keys={"users": ["user_id"], "orders": ["order_id"]},
        foreign_keys={
            "orders": [
                ForeignKeySchema(
                    columns=["user_id"],
                    referred_table="users",
                    referred_columns=["user_id"],
                )
            ]
        },
    )
    print(client.test_connection(source))
    print(client.discover(source))
    # Example dates: use anchors and completeness verified for your own data.
    predictions = client.predict(
        source=source,
        query="PREDICT SUM(orders.amount,0,30,days) FOR EACH users.user_id",
        anchor_times=[datetime(2026, 1, 1, tzinfo=UTC)],
        anchor_time=datetime(2026, 3, 1, tzinfo=UTC),
        observed_through=datetime(2026, 3, 1, tzinfo=UTC),
        entity_created_at="created_at",
        return_details=True,
    )
    print(predictions["predictions"])
```

For Databricks, use the same mappings and prediction call with this connection:

```python
from nori_rel import DatabricksConnection, EnvironmentCredential

source = client.connect(
    engine="databricks",
    connection=DatabricksConnection(
        server_hostname="your-workspace.cloud.databricks.com",
        http_path="/sql/1.0/warehouses/your-warehouse-id",
        catalog="analytics",
    ),
    credential=EnvironmentCredential(variable="DATABRICKS_AUTH_JSON"),
    schema_name="customers",
    tables=["users", "orders"],
    time_columns={"orders": "event_time"},
    primary_keys={"users": ["user_id"], "orders": ["order_id"]},
    foreign_keys={
        "orders": [
            ForeignKeySchema(
                columns=["user_id"],
                referred_table="users",
                referred_columns=["user_id"],
            )
        ]
    },
)
```

Place that connection inside a `with NoriRelClient() as client:` block.
For Snowflake, exact discovered table and column names take precedence.
Otherwise, configured table/column names and unquoted PQL names match case-insensitively
only when exactly one discovered name matches: `orders` can find `ORDERS`, and
`user_id` can find `USER_ID`. Ambiguous matches fail. This applies to table
allowlists, primary/foreign keys, and event-time mappings as well. Generated
SQL preserves the physical spelling returned by discovery. Databricks uses the
exact discovered names in its configuration and PQL.

Backtick-quoted PQL identifiers require an exact match. Use backticks for names
containing spaces, dots, or punctuation, or to select a particular spelling
when Snowflake names differ only by case. Snowflake normally stores unquoted
names in uppercase and preserves quoted names; see its
[identifier rules](https://docs.snowflake.com/en/sql-reference/identifiers-syntax).

Unity Catalog table names can contain hyphens but cannot contain dots or spaces.
Delta column names containing spaces require column mapping. See the Databricks
[name rules](https://docs.databricks.com/aws/en/sql/language-manual/sql-ref-names)
for the platform's limits.

The default local mode supports regression targets. Binary predicate targets
require `NoriRelClient(mode="hosted", api_key=...)`. In hosted mode only generated
feature matrices and labels reach Nori; database credentials and source tables
remain in your process. Generated features and labels may still contain sensitive derived values.

## Write predictions back

`client.write_predictions` stores one run's predictions in a table you name,
on PostgreSQL, Snowflake or Databricks. It is a separate call, so `predict()`
stays read-only, and it connects with its own writer credential. Predictions
from a MySQL source can be written to any of the three.

```python
from nori_rel import EnvironmentCredential, PredictionDestination, SnowflakeConnection

destination = PredictionDestination(
    engine="snowflake",
    connection=SnowflakeConnection(
        account="acme-prod", warehouse="SCORING", database="ANALYTICS"
    ),
    credential=EnvironmentCredential(variable="SNOWFLAKE_WRITER_AUTH_JSON"),
    schema_name="FORECASTS",
)
frame = client.predict(source=source, pipeline=pipeline,
                       as_of=as_of, observed_through=as_of)
record = client.write_predictions(
    frame, destination=destination, table="CUSTOMER_SPEND", create_table=True
)
```

| Column | Contents |
| --- | --- |
| `entity_id` | The entity key: a 64-bit integer, a 38-digit decimal for wider ids, or text |
| `as_of` | The scoring time |
| `prediction` | The requested mean or median; a Boolean for a binary target |
| `probability` | Binary targets only |
| `quantiles` | A JSON list of the requested quantiles, with `output_type="quantiles"` |
| `run_id`, `pipeline_id` | The run and the pipeline that produced the rows |
| `written_at` | When the rows were written |

- Writing a run again replaces that run's rows, so retrying the write of the
  same predictions leaves one copy. A retry that predicts again is a new run
  with its own rows: pass `replace_earlier_runs=True` to also replace the rows
  other runs of the same pipeline wrote for the same `as_of`, so a scheduled
  replay leaves one batch per prediction time. The earlier runs' records stay
  in `nori_rel_runs`. PostgreSQL and Snowflake replace the
  rows and the run's record in one transaction, and PostgreSQL makes two
  writes of the same run take turns. Databricks replaces each table with one
  atomic `INSERT INTO … REPLACE WHERE run_id = …`; rows that do not fit one
  900 KB parameter are loaded into a staging table first, which is dropped
  after, so such a write also needs `CREATE TABLE`.
- `create_table=True` creates a missing table (on PostgreSQL with an index on
  `run_id`). An existing table must have exactly these columns, in any order,
  as the destination names them (uppercase on Snowflake); otherwise a
  `ValueError` names the missing and unexpected ones. The first write sets
  the `entity_id` type, so a run that scored nobody creates no table.
- Write to a schema no source reads, or connect the sources that read it with
  `tables=[...]`. A source without that list reads every table of its schema:
  it skips `nori_rel_runs` and the staging tables, but it reads the
  predictions table, and replaying a pipeline saved before refuses the new
  table as a schema change. `write_predictions` warns when a source of the
  same client reads every table of the destination schema.
- Times are written in UTC, also into `timestamp without time zone` columns of
  a table you created.
- The run's record, with `output_table` and `rows_written`, goes to
  `nori_rel_runs` in the same schema (`NORI_REL_RUNS` on Snowflake), which is
  created when missing. It is also put back on
  `predictions.attrs["run_record"]`, written to the client's `run_records`
  and returned. See [run records](pipeline-export.md#run-records).
- Names are used as given, as for a source. Snowflake columns are uppercase,
  so unquoted SQL reads them; pass an uppercase table name for the same.
- Use an identity that can only create and write tables in the output
  schema: PostgreSQL `USAGE` and `CREATE` on the schema; Snowflake warehouse
  `USAGE`, database and schema `USAGE`, and `CREATE TABLE` on the schema;
  Databricks `USE CATALOG`, `USE SCHEMA`, `CREATE TABLE` and `MODIFY`. Pass it
  as an `EnvironmentCredential` or `AwsSecret` with the same JSON shapes as a
  reader (a `DatabaseUrl` also works for PostgreSQL). It is resolved for each
  write and never stored, and errors never show it: warehouse errors give
  only the driver's code and SQL state.
- Source connections never write. `write_predictions` does not use them, and
  they still refuse every write statement.
- Values travel as parameters. Snowflake and Databricks receive each batch of
  rows as one JSON parameter of at most 900 KB, read with `FLATTEN` or
  `from_json`; Databricks allows 1 MB of parameters per statement.

To keep every run's record in the warehouse, failed and cancelled runs
included, pass a `TableRunRecordStore(destination)` as
`NoriRelClient(run_records=...)`.

## Snapshot and time behavior

Snowflake reads allowed standard tables at one extraction timestamp using Time
Travel. Databricks resolves each Delta table's version against one extraction
timestamp and uses `VERSION AS OF` for all subsequent reads. The same pinned
references are used for historical labels, scoring filters, and feature-table
reads. Databricks table identities are checked before and after extraction to
detect replacement under the same table name. Snapshot provenance is available
in detailed prediction metadata.

That extraction timestamp describes the physical database state. The supplied
historical `anchor_times`, scoring `anchor_time`, and verified `observed_through`
watermark describe the prediction problem. A label window is
`(anchor + start, anchor + end]`; feature construction only uses events known at
the row's anchor. `observed_through` is supplied by the caller rather than inferred
from the latest event. Entity attributes must be immutable or valid at every
requested anchor; the connectors do not reconstruct historical attribute values.

Time Travel must retain the required history. Views and unsupported table types,
unavailable Delta versions, retired history, and incompatible schema changes must
fail the extraction rather than mix current rows with a pinned snapshot. Review
the platform limitations in [Snowflake Time Travel](https://docs.snowflake.com/en/sql-reference/constructs/at-before)
and [Databricks table history](https://docs.databricks.com/aws/en/tables/history).

Returned tables, total rows, snapshot memory, and statement durations are bounded
by `RelationalLimits`. Exceeding a limit fails the prediction; data is not silently
truncated or sampled. Database query limits do not guarantee a fixed amount of
warehouse compute scanned or billed.

Prediction rejects excessive table counts or available metadata row estimates
before key validation. Missing estimates use a count probe with a row LIMIT;
actual extraction limits still apply. A large full-table metadata estimate can
reject a source even when a historical cutoff would expose fewer rows.
Primary-key checks use one grouped query per keyed table; foreign-key checks use one
anti-join per declared relationship with a parent inside the allowlist. They
read the same pinned source state as
labels and features. Event-table validation uses only rows at or before the
prediction's scoring cutoff; unmapped tables are validated in full. It is not
a separate certification of future rows or the whole database. These checks can
incur warehouse charges even when extraction fails. Snowflake enforces the
statement timeout in the reader session on the server.

## Run deterministic checks

From the repository root, install the full development environment:

```bash
uv sync --group dev --extra database --extra snowflake --extra databricks
uv run --no-sync pytest -m "not warehouse and not live_nori"
uv run --no-sync ruff format --check .
uv run --no-sync ruff check .
uv build --no-sources
```

Driver simulations cover configuration, binding, conversion, error handling,
optional dependencies, resource limits, and credential redaction without paid
services. PostgreSQL integration tests use `NORI_REL_TEST_DATABASE_URL` and the
existing isolated fixture schemas.

## Configure live warehouse tests

Live checks provision synthetic fixtures in a random schema named
`nori_rel_test_<uuid>`. A separate writer identity creates it, inserts fixtures,
optionally grants the reader access, and drops only that owned schema afterward.
The connector uses the reader identity. Do not point both credential references
to an identity that can write: the read-only test deliberately requires the
reader's INSERT to fail.

Each test configuration is a JSON object without credentials. It still names
your account, workspace, warehouse and roles, so store it as a CI secret, not a
plain variable: GitHub Actions prints variables unmasked in its logs, which are
public for a public repository. Set `writer_schema` to an
existing schema if Snowflake's `PUBLIC` or Databricks' `default` schema is
unavailable. The writer connects there before creating its isolated fixture
schema.

For Snowflake, set
`NORI_REL_TEST_SNOWFLAKE_CONFIG` to:

```json
{
  "connection": {
    "account": "your-org-your-account",
    "warehouse": "NORI_TEST_WH",
    "database": "NORI_TEST",
    "role": "NORI_TEST_READER"
  },
  "reader_credential_env": "NORI_REL_TEST_SNOWFLAKE_READER",
  "writer_credential_env": "NORI_REL_TEST_SNOWFLAKE_WRITER",
  "writer_connection": {"role": "NORI_TEST_WRITER"},
  "reader_role": "NORI_TEST_READER"
}
```

Set the two referenced environment variables to authentication JSON using the
supported shapes above. The writer role must be able to create/drop schemas and
grant reader usage/SELECT within the test database. Both roles need database and
warehouse usage. `reader_role` enables fixture grants; omit it only if equivalent
future grants are already configured by the administrator.

For Databricks, set `NORI_REL_TEST_DATABRICKS_CONFIG` to:

```json
{
  "connection": {
    "server_hostname": "your-workspace.cloud.databricks.com",
    "http_path": "/sql/1.0/warehouses/your-warehouse-id",
    "catalog": "nori_test"
  },
  "reader_credential_env": "NORI_REL_TEST_DATABRICKS_READER",
  "writer_credential_env": "NORI_REL_TEST_DATABRICKS_WRITER",
  "reader_principal": "reader-service-principal-application-id"
}
```

The writer needs CREATE SCHEMA and permission to grant USE SCHEMA/SELECT in the
test catalog. Both identities need USE CATALOG and SQL warehouse access.
`reader_principal` enables fixture grants; omit it when equivalent grants are
already managed externally. Keep credentials in environment variables or CI
secrets; tests do not print their values.

Run each selected service with required credentials:

```bash
uv run --no-sync pytest tests/connectors/test_warehouse_live.py \
  -m "warehouse and snowflake and not live_nori" --require-warehouse
uv run --no-sync pytest tests/connectors/test_warehouse_live.py \
  -m "warehouse and databricks and not live_nori" --require-warehouse
uv run --no-sync pytest tests/connectors/test_warehouse_live.py \
  -m "warehouse and parity and not live_nori" --require-warehouse
```

These checks cover discovery, explicit relationships, all supported aggregate
targets, arithmetic and binary labels, empty and all-null windows, ordered ties,
exact window boundaries, delayed windows, IDs above `2**53`, bounded snapshots,
reader write denial, duplicate/null keys, orphan relationships, quoted names,
scoring filters, unavailable snapshots, schema changes, and concurrent updates
after snapshot pinning. Public client
tests build real features and use deterministic hosted/local predictors. The
parity check compares labels, feature matrices, and results between both real
warehouses.

Ordinary runs skip absent service configurations. `--require-warehouse` makes
missing settings fail; include the explicit test-file path so pytest loads the
warehouse option before parsing arguments. The manually triggered
`Warehouse integration` GitHub Actions workflow uses that option and uploads
JUnit reports. Configure `warehouse` environment secrets with the JSON above
(`NORI_REL_TEST_SNOWFLAKE_CONFIG`, `NORI_REL_TEST_DATABRICKS_CONFIG`) and with
the corresponding `NORI_REL_TEST_*_READER`/`*_WRITER` values. Running both
engines also runs parity. This workflow is not a default PR check because it
requires external credentials and incurs warehouse charges.

The PostgreSQL integration suite also loads this exact fixture and checks the
same independent label expectations. These 14 parity cases run in ordinary CI
when `NORI_REL_TEST_DATABASE_URL` is configured:

```bash
uv run --no-sync pytest tests/connectors/test_warehouse_live.py \
  -m "parity and not warehouse"
```

A live inference smoke test runs the default Nori 1.5 checkpoint from the
Hugging Face cache. Install the `local` extra, populate the cache, and require the model
in release certification:

```bash
uv run --no-sync pytest tests/connectors/test_warehouse_live.py \
  -m "warehouse and snowflake and live_nori" --require-warehouse --require-local-model
# Substitute databricks for snowflake to use that service.
```

Deterministic inference checks do not certify live Nori service behavior or model
quality. Unavailable-version tests exercise failure without a current-data
fallback; they do not deliberately retire retention history. Actual history
retirement and backend cancellation of long-running queries require separately
controlled service checks. Live validation is complete only after actual service
runs pass. Skipped checks leave that validation outstanding.

## Synthetic example data

The [public examples](../examples/README.md) generate the same two-table fixture
for every connector. Run `python examples/synthetic_data.py /tmp/nori-rel-example`
to create `users.csv` and `orders.csv`. Load them into a disposable schema using
your warehouse's data-loading tools, preserving UTC timestamps, integer IDs,
and numeric amounts. Schema definitions are in
[`examples/snowflake.sql`](../examples/snowflake.sql) and
[`examples/databricks.sql`](../examples/databricks.sql).

Use the mappings above with `order_id` as the orders primary key and `user_id`
as the foreign key. For this fixture, set historical anchors to the eight weekly
cutoffs starting 37 days before `2026-03-02T00:00:00Z`, the scoring cutoff and
verified watermark to `2026-03-02T00:00:00Z`, and `entity_created_at="created_at"`.
Run a reader identity without write permissions. Remove only the disposable
schema you created when finished. These examples generate no research data and
need no LLM provider SDK.
