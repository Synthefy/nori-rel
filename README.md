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

## Choose a feature search

Each prediction chooses its feature search with
`predict(..., feature_strategy=...)`: `"fastdfs"` (the default), `"noridfs"`, or
an `AgenticSearch`. `feature_strategy="noridfs"` plans a NoriDFS relationship
search over the foreign-key graph, with aggregation windows scaled to the
target's label interval. It needs only the core dependencies.

```python
details = client.predict(..., feature_strategy="noridfs")
```

`start_predict` takes `feature_strategy` too. `NoriRelClient(features=...)` is
deprecated: it still sets the search for predictions that pass none, with a
`DeprecationWarning`, and will be removed in 0.5.0.

`AgenticSearch` lets a language-model agent write SQL feature columns on top of
NoriDFS. Install `nori-rel[agentic]`. Each candidate program runs locally in an
in-memory DuckDB catalog and is scored with the client's own Nori predictor on
held-out historical rows. A program is kept only if it also beats NoriDFS alone
on a separate confirmation slice; otherwise the prediction uses NoriDFS alone.

```python
from nori_rel import AgenticSearch

search = AgenticSearch.claude()  # reads ANTHROPIC_API_KEY
details = client.predict(..., feature_strategy=search)
```

Because the choice is per prediction, one client and one source can compare
NoriDFS alone with an agent search, or agents on two models:

```python
opus = AgenticSearch.claude(model="claude-opus-5-5")
sonnet = AgenticSearch.claude(model="claude-sonnet-5-5")
for strategy in ("noridfs", opus, sonnet):
    details = client.predict(..., feature_strategy=strategy)
```

`AgenticSearch.openai(model=...)` uses an OpenAI-compatible endpoint. The agent
sends that model provider:

- your schema: table and column names, row counts, keys, and time columns;
- a profile of each column it asks about: the share of missing values, the
  number of distinct values, and either the 1st, 25th, 50th, 75th and 99th
  percentiles of a numeric column, the earliest and latest value of a date
  column, or the 10 most frequent values of any other column with their counts.
  Those values are raw cells, such as names, emails, or IDs;
- up to 50 rows of each exploratory query result;
- the error message of any query that fails, which can quote values.

Do not enable it for data that must not leave your environment for that
provider. A search also costs provider
tokens and extra Nori calls. Bound it with `max_cost_usd` (which also needs
`prices`) or the search's `limits`. It is not exercised against a live model provider in release certification. See
[run control](docs/run-control.md) for `stop` and `steer`.

The agent keeps a program only if it lowers the error on held-out history.
`predict(..., metric=...)` chooses that error for a regression target:

| `metric` | Error (lower is better) |
|---|---|
| `"mae"` (default) | Mean absolute error |
| `"mse"` | Mean squared error |
| `"rmse"` | Root mean squared error |
| `"nmae"` | MAE divided by the range (max minus min) of the actuals |
| `"nrmse"` | RMSE divided by the range of the actuals |
| `"medae"` | Median absolute error |
| `"wape"` | Total absolute error divided by total actual |
| `"mape"` | Mean absolute percentage error, over rows whose actual is not zero |
| `"smape"` | Symmetric MAPE: mean of 2\|error\| / (\|actual\| + \|forecast\|) |

On one holdout, NMAE and WAPE rank programs as MAE does and NRMSE as RMSE
does: each divides by the same constant for every candidate. MSE and RMSE
weigh large misses more. MAPE and SMAPE weigh every row equally in relative
terms, so small series count as much as large ones. The metric changes which
program is kept, not how the forecast is made. Trial and baseline events report
it, with relative and normalized errors as percentages
(`Trial 3/24: WAPE 11.20% ...`).

```python
client.predict(source=source, query=query, feature_strategy=search, metric="wape", ...)
```

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

A pipeline stores the task, feature recipe (a FastDFS or NoriDFS plan, plus the
agent's SQL program after an agent search), schema contract, context policy, and
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
