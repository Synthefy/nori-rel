# Client configuration and errors

The public entry point is `nori_rel.NoriRelClient`. It owns connection pools and
prediction runs; use `with NoriRelClient(...) as client:` or call `close()`.
Closing is idempotent. A closed client cannot be reused. Create a new client to
pick up rotated database credentials.

## Inference settings

| Setting | Default and behavior |
| --- | --- |
| `mode` | `"local"`; `"hosted"` opts into the hosted service. |
| `model` | `"nori-30m"`; local supports `nori-6m`, `nori-30m`, and `nori-100m`. |
| `api_key` | Hosted mode uses this value or `SYNTHEFY_NORI_API_KEY`. |
| `device` | Local only; automatic CUDA, then MPS, then CPU. An explicit PyTorch device can be supplied. |
| `features` | Deprecated; pass `feature_strategy=` to `predict` or `start_predict` instead. It still sets the feature search for predictions that pass none, with a `DeprecationWarning`, and will be removed in 0.5.0. `feature_strategy` takes `"fastdfs"` (default), `"noridfs"`, an `AgenticSearch`, or a custom `FeatureBackend` instance, which has no pipeline format. |
| `timeout` | 300 seconds for inference transport configuration. |
| `max_retries` | 2 for hosted transport retries. |
| `limits` | `RelationalLimits()` for extraction/context ceilings. |

`timeout` is not a hard deadline for the complete relational workflow.
`run.result(timeout=...)` bounds the caller's wait and leaves the run active;
see [run control](run-control.md).

The optional local runtime is pinned independently of the lightweight Synthefy
SDK. Initial prediction downloads the selected weights to the Hugging Face
cache. Populate that cache first to run with `HF_HUB_OFFLINE=1`. Missing or
corrupted required weights cause an actionable failure; local mode never falls
back to a hosted call. Hardware, runtime, and source changes can affect numerical
results. Pipeline hashes detect specific source/model changes, not arbitrary
numerical nondeterminism.

## Connections

`connect()` returns a `Database` handle. Provide `engine` and exactly one of
`database_url` or `credential`. Credential references are:

- `EnvironmentCredential(variable="DATABASE_URL")` for a PostgreSQL URL, or
  a variable holding warehouse authentication JSON.
- `AwsSecret(secret_id="your-secret-name", region_name="us-west-2")` for AWS
  Secrets Manager. PostgreSQL accepts a URL or standard RDS JSON fields.
- `DatabaseUrl(value=...)`, primarily for direct PostgreSQL URL use.

Each connector extra includes AWS credential support. An RDS or Aurora PostgreSQL
endpoint still uses `engine="postgresql"`; MySQL is unsupported.

Use `schema_name`, a `tables` allowlist, and `time_columns` mappings. Warehouses
require a typed `connection` object and an explicit schema. `primary_keys` and
`foreign_keys` supply relationships when warehouse metadata lacks them.
`client.test_connection(source)` checks connectivity, and `client.discover(source)`
returns the schema and connector capabilities. See [warehouse setup](warehouse-connectors.md).

## Resource limits

```python
from nori_rel import NoriRelClient, RelationalLimits

client = NoriRelClient(
    limits=RelationalLimits(
        max_rows_per_table=100_000,
        max_total_rows=300_000,
        max_snapshot_bytes=500_000_000,
        statement_timeout_ms=60_000,
    )
)
# Connect, predict, and close the client when finished.
```

| Limit | Default |
| --- | ---: |
| Tables | 32 |
| Rows per table | 1,000,000 |
| Total snapshot rows | 3,000,000 |
| Snapshot bytes | 4,000,000,000 |
| SQL statement timeout | 300,000 ms |
| Minimum training rows | 32 |
| Maximum context rows | 60,000 |
| Feature elements | 3,000,000 |
| Features | 512 |

The per-request `max_context_rows` defaults to 5,000 and remains subject to the
resource ceilings. Snapshot limits constrain extracted data; feature building
and model inference can allocate additional memory. SQL timeouts and row limits
do not guarantee a fixed warehouse scan cost. Exceeding an extraction limit
fails the operation instead of silently truncating source data.

## Prediction details

`predict(..., return_details=True)` returns `PredictionDetails`:

| Attribute | Contents |
| --- | --- |
| `predictions` | one row per scored entity: `entity_id`, `anchor_time`, `prediction` |
| `metadata` | run facts; `metadata["features"]` names the feature search that ran |
| `pipeline` | a `NoriRelPipeline` to save and replay, or `None` |
| `features` | the feature matrix Nori scored, one row per `entity_id`, one column per feature |
| `internal_feature_names` | the model's name for each column of `features`, in order |
| `feature_sql` | the SQL FastDFS ran for each feature, by internal name |
| `context` | the labeled rows Nori learned from: `entity_id`, `anchor_time` (a past date) and `label` |
| `context_features` | the feature matrix of the `context` rows, the one Nori was fitted on: same row order, the columns of `features` |

`features` names each column in dots: the tables the feature reads through
from the entity, then the column, then the aggregate and its window, such as
`results.grid.mean`, `results.count`, `qualifying.constructors.count` (the
constructors reached through the entity's qualifying rows) or
`orders.amount.mean_30d`. The entity's own columns keep their table name
(`drivers.dob`), anchor-time fields start with `anchor.`, FastDFS columns read
the same way (`results.grid.max`), and an agent's column is
`<query>.<column>`. Names are unique within a prediction: when two features
would share one, each table they reach through a different foreign key shows
that key in brackets (`matches[home_team_id].count`), and a name that still
collides stays the internal name. Only this copy of the matrix is renamed; the
model, the pipeline, a saved agent program and `metadata["feature_matrix_sha256"]`
use the internal names.

`get_feature_table()` returns one row per column of `features`, in the same
order: `feature`, `internal_name`, `source`, `sql`, `dtype`, `missing_share`,
`mean`, `min` and `max`. `sql` holds the agent's query for an agent column and
the statements FastDFS ran on its DuckDB copy of the snapshot for a FastDFS
column. NoriDFS computes in pandas, so its `sql` is `None`.

`get_context_and_scored_table()` returns every row Nori read: the context rows
it learned from and the scored rows it predicted. Columns are `entity_id`,
`anchor_time`, `label`, `prediction` (and `probability` for a yes/no target or
`quantiles` for a quantile forecast), then the columns of `features`. A context
row holds its label and the features Nori learned from, with the forecast
columns empty; a scored row holds its prediction and the features Nori
predicted from, with `label` empty. Rows are sorted by `entity_id`, then
`anchor_time`, so one entity's rows read as a timeline that ends in its
prediction. A past anchor time with no labeled row, such as one whose window
had no events, has no rows. The context rows are exactly what Nori was fitted
on, so `context_features` and `context["label"]` are the data a SHAP or
glass-box explainer needs.

```python
table = details.get_context_and_scored_table()
table[table["entity_id"] == entity_id]  # one entity's timeline
```

## Errors and compatibility

Invalid arguments raise `ValueError` or typed validation errors. Connector
failures raise `nori_rel.connectors.errors.ConnectorError`; extraction limits raise its
`SnapshotLimitError` subclass. Prediction runs expose `PredictionCancelled`,
`PredictionFailed`, and `RunTimeout`. Saved-pipeline incompatibilities raise a
`PipelineError` subclass, described in [pipeline export](pipeline-export.md).
Inspect error types and messages rather than matching full exception strings.

The old `entity`/`target` aggregate request form is a deprecated compatibility
bridge and emits `FutureWarning`. Use `query=` PQL for new code. Direct/NEXT
legacy targets and explicit relationship-path prediction are retired. RelArena
adapters, fine-tuning, and benchmark runners are outside the public client
contract and are not included in this package.
