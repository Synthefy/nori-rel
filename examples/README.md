# Synthetic examples

These examples generate 48 users and 1,248 weekly orders with fixed dates and
amounts. There are no customer records, research dataset downloads, or LLM calls.
Data generation is deterministic; inference still depends on the selected model,
hardware, and runtime.

## PostgreSQL prediction and pipeline replay

Install the package, check out this repository for the example scripts, and set
`DATABASE_URL` privately to a disposable TLS-enabled PostgreSQL database. The
identity must be able to create and drop schemas. The example creates a random
schema, loads only synthetic data, and removes that schema in a `finally` block.
Its prediction reads use the client's read-only transactions.

```bash
python -m pip install "nori-rel[database,local]"
python examples/pipeline_reload.py
```

Local mode downloads model weights on first use and needs no Synthefy API key.
To use hosted inference instead:

```bash
python -m pip install "nori-rel[database]"
# Set SYNTHEFY_NORI_API_KEY privately before running.
python examples/pipeline_reload.py --mode hosted
```

The script predicts 30-day customer spend at March 2, 2026, using eight earlier
weekly anchors. It saves a FastDFS pipeline in a temporary directory, closes the
client, loads the pipeline into a fresh client, and checks that replay produces
the same context and feature hashes. The temporary pipeline file is cleaned up.
To schedule future predictions, retain a pipeline file and the persistent data
source, then supply a new `as_of` and verified `observed_through` watermark.

## Load the fixture into PostgreSQL

To keep the data and work with it by hand or through the
[agent skills](../skills/README.md#try-the-whole-path-on-synthetic-data):

```bash
python examples/load_fixture.py --schema nori_rel_example
```

It creates the schema (dropping it first if it exists) with the same tables,
plus a nullable `income` column on `users` that one user in six lacks, so a
static target has rows to score. Drop the schema yourself when finished.

## Snowflake and Databricks

Create the same portable data without installing any dependencies:

```bash
python examples/synthetic_data.py /tmp/nori-rel-example
```

Create a disposable schema with [snowflake.sql](snowflake.sql) or
[databricks.sql](databricks.sql). Load the CSVs with the warehouse's data-loading
tools, preserving header names, integer IDs, numeric amounts, and timezone-aware
UTC timestamps. Grant a dedicated reader identity usage and SELECT, without
write privileges. Follow the connection and key mappings in the
[warehouse guide](../docs/warehouse-connectors.md).

Use `CUSTOMER_SPEND_QUERY`, `SCORING_TIME`, and `HISTORICAL_ANCHOR_TIMES` from
[synthetic_data.py](synthetic_data.py) with `entity_created_at="created_at"`.
`SCORING_TIME` is also this synthetic fixture's completeness watermark. The same
prediction and pipeline API applies to all three connectors. Drop only the
schema you created after finishing. Warehouse loading and prediction may incur
service charges; examples do not provision compute automatically.
