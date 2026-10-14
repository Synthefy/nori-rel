# Nori-Rel

Predict from relational databases with [Nori](https://docs.synthefy.com/nori).
Nori-Rel reads a bounded database snapshot, constructs historical targets and
FastDFS features, and runs local or hosted inference through one Python client.

The supported package covers PostgreSQL, Snowflake standard tables, and
Databricks Unity Catalog Delta tables. Releases come from reviewed commits on
`main`.

Review the [support contract](docs/support.md) and
[dependency security assessment](docs/dependency-security.md) before deployment.
The assessment includes scoped, expiring exceptions for upstream dependencies.

## Install

Python 3.11 and 3.12 are supported. Install your connector and, for the default
local inference mode, the model runtime:

```bash
python -m pip install "nori-rel[database,local]"
# Other connectors:
python -m pip install "nori-rel[snowflake,local]"
python -m pip install "nori-rel[databricks,local]"
```

For hosted inference, omit `local` and set `SYNTHEFY_NORI_API_KEY` privately.
A hosted installation does not require the local model runtime or model weights.
See [supported capabilities](docs/support.md) for platform and target limits.

## Start from CSV or Excel files

If your data is a set of exported files rather than a database, the
[`nori-rel-prepare-data`](skills/nori-rel-prepare-data/SKILL.md) skill walks an
AI agent through converting them. The agent profiles the files and designs
entity and event tables with declared keys, leaving out columns that would leak
the future, such as a customer's lifetime total. It shows you every table,
column and relationship it plans to create and waits for your confirmation
before it touches the database. It then writes a load script you can rerun
when new files arrive, and proves the schema with Nori-Rel's own discovery and
one prediction.

With Claude Code, copy the folder into your skills, then ask for the conversion:

```bash
cp -r skills/nori-rel-prepare-data ~/.claude/skills/
```

With another agent, give it `SKILL.md` as its instructions.

## Predict customer spend

This example expects existing `users(user_id, created_at)` and
`orders(order_id, user_id, event_time, amount)` tables with declared primary and
foreign keys. Set `DATABASE_URL` to a TLS-enabled PostgreSQL connection string.
Use [the synthetic example](examples/README.md) for a complete disposable dataset.

```python
from datetime import UTC, datetime

from nori_rel import EnvironmentCredential, NoriRelClient

with NoriRelClient() as client:
    source = client.connect(
        engine="postgresql",
        credential=EnvironmentCredential(variable="DATABASE_URL"),
        tables=["users", "orders"],
        time_columns={"orders": "event_time"},
    )
    details = client.predict(
        source=source,
        query="PREDICT SUM(orders.amount,0,30,days) FOR EACH users.user_id",
        anchor_times=[datetime(2026, 1, 1, tzinfo=UTC)],
        anchor_time=datetime(2026, 3, 1, tzinfo=UTC),
        observed_through=datetime(2026, 3, 1, tzinfo=UTC),
        entity_created_at="created_at",
        return_details=True,
    )
    predictions = details.predictions
    print(predictions[["entity_id", "prediction"]])
```

The query predicts each user's total order amount over the next 30 days.
`anchor_times` supplies historical examples; `anchor_time` is the scoring time.
`observed_through` is your verified data-completeness watermark, not an estimate
of completeness from the latest event. Use dates appropriate to your data and
at least 32 usable historical rows. Register every event table's timestamp;
other entity attributes must be immutable or valid at every requested anchor.
See [PQL and time semantics](docs/pql.md).

The default mode runs Nori in your process. The first local prediction downloads
model weights; subsequent predictions can use the populated cache offline.
Use `NoriRelClient(mode="hosted")` to send the generated context/query feature
matrices and labels to hosted Nori. Database connections, credentials, and raw
source tables stay in the caller's process. Generated features can still contain
sensitive information derived from those tables.

## Save and reload

Request prediction details to obtain a pipeline, then reconnect when using it:

```python
from nori_rel import NoriRelPipeline

if details.pipeline is None:
    raise RuntimeError("This prediction cannot be exported as a pipeline")
details.pipeline.save("customer_spend.json")

pipeline = NoriRelPipeline.load("customer_spend.json")
with NoriRelClient() as client:
    source = client.connect(
        engine="postgresql",
        credential=EnvironmentCredential(variable="DATABASE_URL"),
        tables=["users", "orders"],
        time_columns={"orders": "event_time"},
    )
    predictions = client.predict(
        source=source,
        pipeline=pipeline,
        as_of=datetime(2026, 4, 1, tzinfo=UTC),
        observed_through=datetime(2026, 4, 1, tzinfo=UTC),
    )
```

A pipeline stores the task, feature recipe, schema contract, context policy, and
model settings. Each replay reads fresh rows. It contains no credentials,
source rows, or model weights. See [pipeline export](docs/pipeline-export.md)
and the runnable [save/reload example](examples/pipeline_reload.py).

## Operations and safety boundaries

Use the client as a context manager to close connection pools. PostgreSQL uses
TLS and read-only transactions; warehouse connectors use pinned reads and require
a dedicated identity without write privileges. Warehouse capabilities and
snapshot guarantees are documented in the [connector guide](docs/warehouse-connectors.md).

`RelationalLimits` bounds tables, rows, snapshot memory, feature size, and SQL
statement time. Exceeding an extraction limit raises an error. These limits do
not cap total warehouse billing or the process's total memory. See
[configuration and errors](docs/api.md) and [progress/cancellation](docs/run-control.md).

## Development and licensing

See the [public API inventory](docs/public-api.md),
[CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), and
[RELEASING.md](RELEASING.md). Public release changes are recorded in
[CHANGELOG.md](CHANGELOG.md).

Original Synthefy code is [MIT-licensed](LICENSE). The vendored PQL grammar and
derived parser retain Apache-2.0 licensing. Dependencies, downloaded model weights,
and external datasets have their own terms; see [ATTRIBUTION.md](ATTRIBUTION.md).
