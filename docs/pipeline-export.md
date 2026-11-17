# Save and reload a pipeline

A pipeline is a versioned prediction recipe. It stores the PQL task, source
schema, the feature recipe, ordered feature columns, model settings, and rules
for historical context. The recipe is FastDFS settings, a fitted NoriDFS plan, or
a NoriDFS plan with the SQL program an agent search kept. It contains no credentials, database rows, or
model weights.

First request details from a normal PQL prediction:

```python
# request is the complete source/query/time argument mapping for a prediction.
details = client.predict(**request, return_details=True)
pipeline = details.pipeline
if pipeline is None:
    raise RuntimeError("This prediction cannot be exported as a pipeline")
pipeline.save("customer_spend.json")
```

Reconnect to the source before replaying. Use the same local/hosted mode and model
as the saved pipeline:

```python
from datetime import UTC, datetime

from nori_rel import EnvironmentCredential, NoriRelClient, NoriRelPipeline

pipeline = NoriRelPipeline.load("customer_spend.json")
with NoriRelClient() as client:
    source = client.connect(
        engine="postgresql",
        credential=EnvironmentCredential(variable="DATABASE_URL"),
        tables=["users", "orders"],
        time_columns={"orders": "event_time"},
    )
    frame = client.predict(
        source=source,
        pipeline=pipeline,
        as_of=datetime(2026, 4, 1, tzinfo=UTC),
        observed_through=datetime(2026, 4, 1, tzinfo=UTC),
    )
```

See [the complete synthetic example](../examples/pipeline_reload.py) for dataset
creation, prediction, export, a fresh client, and cleanup.

## What replay preserves

The context policy stores each historical anchor's distance before the scoring
cutoff, and either `max_context_rows` or the `ContextPolicy` the first run used
(`row_policy`; see [context-policy.md](context-policy.md)). A new `as_of` shifts
those anchors, reads fresh historical rows and applies the saved rule to them.
Offsets are elapsed time, not calendar rules: a monthly schedule is not encoded
as a calendar-month offset. Late-arriving rows or corrected values may therefore
change a replay's context and predictions.

FastDFS replay restores the saved type hints and requires the same FastDFS
version and feature columns. NoriDFS replay runs the saved plan without planning
again and refuses a plan whose schema or digest no longer matches. An agent
pipeline replays its saved SQL program on that plan in DuckDB, so it needs the
`agentic` extra but sends nothing to a model provider and searches no further.
If no program beat NoriDFS alone, the pipeline holds the NoriDFS plan only. Schema checks cover tables, column types, keys,
and registered time columns. Local pipelines pin the checkpoint repository,
revision, file SHA-256, runtime version, and inference settings. Hosted pipelines
record model names and SDK version; they do not pin the server's underlying
weights or guarantee that a model name always serves identical weights.

Pass `source`, `as_of`, and `observed_through` on every replay. You may also
supply `query_filter`, a PQL `query` changing only the scoring `FOR` selection,
or `expected_context_sha256`. Arguments fixed by the pipeline, such as
`anchor_times`, `entity_created_at`, and output settings, cannot be overridden.

Every prediction records `context_rows_sha256` and `feature_matrix_sha256` in
its metadata. They describe that run. To require the same historical rows as a
previous run, pass its context hash as `expected_context_sha256`. This checks
context equality, not equality of all model behavior or future source snapshots.

## Format and failures

The JSON has `format="nori-rel.pipeline"`, an explicit `format_version`, and a
computed `pipeline_id`. Loading checks the format, typed fields, and content
identifier. The identifier detects edits but is not a cryptographic signature
from a trusted publisher. Pipeline files can reveal schema names, query text,
and model configuration even though they contain no source rows or credentials.

| Failure | Error from `nori_rel.pipeline` |
| --- | --- |
| Invalid format/version/content identifier | `PipelineFormatError` |
| Changed target, historical predicate, or entity | `PipelineTaskError` |
| Incompatible source schema/dtypes | `PipelineSchemaError` |
| Requested context hash differs | `PipelineContextError` |
| Feature plan, program, FastDFS version, or feature columns differ | `PipelineFeatureError` |
| Mode/model/runtime/checkpoint/settings differ | `PipelineModelError` |

These inherit from `PipelineError`, a `ValueError`. A different Nori-Rel version
emits `PipelineVersionWarning`; the remaining compatibility checks still apply.
Custom feature implementations and local checkpoint files
outside the supported cache layout may not produce an exportable pipeline.

## Run records

Every run builds a `RunRecord` when it ends, whether it completed, failed or
was cancelled. It names the pipeline, the snapshot, the package and model
versions, the database user the connector read as, and the outcome. A client
created with `run_records` writes each run's record twice: with status
`running` before the run does any work, and again before its final status.

```python
with NoriRelClient(run_records="/mnt/audit/nori_rel_runs") as client:
    source = client.connect(...)
    frame = client.predict(source=source, pipeline=pipeline,
                           as_of=..., observed_through=...)

record = client.run_records.read(frame.attrs["run_id"])
```

`run_records` takes a directory, where each run writes `<run_id>.json`, a
`TableRunRecordStore(destination)`, which keeps one row per run in the
`nori_rel_runs` table of a write-back destination, or any `RunRecordStore`
(an object with `write(record)` and `read(run_id)`). Without it the record is
still built. A completed prediction carries its run's id in
`attrs["run_id"]` and the record in `attrs["run_record"]` and
`details.run_record`; `details.metadata` leaves both out, so identical requests
keep equal metadata. `run.record` holds the record for every outcome. An
exception raised by `predict()` carries the note `Nori-Rel run <run_id>`.

| Field | Contents |
| --- | --- |
| `run_id`, `status` | `completed`, `failed` or `cancelled`; `running` until the run ends |
| `started_at`, `finished_at` | When the run started running and when it ended (`None` while running) |
| `error_class`, `error_message` | Failed runs only: the exception class, such as `PipelineSchemaError`, and the run's redacted error message |
| `pipeline_id`, `strategy` | The pipeline replayed or built, and the feature strategy used |
| `as_of`, `observed_through` | The prediction time and the completeness watermark |
| `engine`, `schema_name`, `principal` | The source, and the user the connector read as: `current_user` on PostgreSQL, `CURRENT_USER()` on MySQL, Snowflake and Databricks (`None` if a warehouse lookup fails, which is logged) |
| `snapshot` | `metadata["snapshot"]`: the extraction timestamp and each table's Time Travel or `VERSION AS OF` reference. `None` for PostgreSQL and MySQL |
| `nori_rel_version`, `synthefy_nori_version`, `synthefy_version` | Installed package versions |
| `transport`, `model`, `checkpoint_sha256`, `model_version` | Where Nori ran (`local` or `hosted`), the model name, the SHA-256 of the checkpoint local Nori loaded or the pipeline pins, and a reserved model digest |
| `context_rows_read`, `context_rows_used`, `entities_scored` | Candidate context rows read, labelled or not; rows Nori read as context; prediction rows returned |
| `context_rows_sha256`, `feature_matrix_sha256` | The audit hashes of metadata. Pass the first as `expected_context_sha256` to reproduce the run |
| `output_table`, `rows_written` | The table `write_predictions` wrote the run's predictions to, and how many rows; `None` until then |

- A field the run never reached stays `None`. A replay's pipeline id and
  pinned model are known before it reads anything. A replay that
  `PipelineSchemaError` refuses has already read the source, so its record has
  the principal, the snapshot and `context_rows_read`, but no
  `entities_scored`. A reported value the record cannot hold, such as a naive
  `as_of`, is left out and logged; the record is still written.
- With `run_records`, a run that cannot write its `running` record fails
  before it reads anything, and one that would complete but cannot write its
  final record fails with the write error, so no prediction is returned
  without its record. A failed or cancelled run whose record cannot be written
  keeps its status, and the write error is logged.
- A request refused before its run starts still raises at once, and with
  `run_records` it also leaves a `failed` record: invalid arguments, or a
  `query` whose task differs from the pipeline's (`PipelineTaskError`). The
  record names the error and what the request fixed (pipeline, pinned model,
  source, `as_of`, `observed_through`); nothing was read, so it has no
  principal or row counts.
- A run whose process dies keeps its `running` record, with no `finished_at`:
  one killed by `SIGKILL` or the out-of-memory killer, or a `start_predict`
  run left behind by a script that exits without `close()` or a `with` block.
  `close()` waits for runs, so their final records are written.
- The record holds no credentials and no row values. Its error message is the
  run's redacted one. The principal is the database's own name for the user
  and is not redacted, since redaction turns a long service-account name into
  `***`. Names, table references, versions, hashes and counts are kept.
- PostgreSQL and MySQL read one consistent transaction and cannot pin a past
  version, so their records have no snapshot; `started_at` and `finished_at`
  bound when they read.
- A file holds `format: "nori-rel.run-record"` and `format_version: 1`. It is
  replaced atomically, and parallel runs write separate files.

`client.write_predictions(frame, destination=..., table=...)` stores a
completed run's predictions next to its record: the record, with
`output_table` and `rows_written`, goes to `nori_rel_runs` in the same schema
in the same write ([writing predictions back](warehouse-connectors.md#write-predictions-back)).
A `nori_rel_runs` row holds `run_id`, `pipeline_id`, `status`, `as_of`,
`started_at`, `finished_at`, `output_table`, `rows_written`, `written_at` and
the whole record as JSON in `record` (`jsonb` on PostgreSQL, `VARIANT` on
Snowflake, JSON text on Databricks), so the table keeps its columns when the
record gains fields. Writing a run's record again replaces its row.
