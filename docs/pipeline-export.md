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
cutoff. A new `as_of` shifts those anchors and reads fresh historical rows.
Offsets are elapsed time, not calendar rules: a monthly schedule is not encoded
as a calendar-month offset. Late-arriving rows or corrected values may therefore
change a replay's context and predictions.

FastDFS replay restores the saved type hints and requires the same FastDFS
version and feature columns. NoriDFS replay runs the saved plan without planning
again and refuses a plan whose schema or digest no longer matches. It runs the
plan on DuckDB, as a new prediction does, so a plan saved by 0.4.3 or earlier,
whose features were computed in pandas, can replay with last-digit differences
in means, sums and ages and with corrected standard deviations. An agent
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
