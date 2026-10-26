---
name: nori-rel-schedule
description: Save a Nori-Rel prediction as a pipeline and run it again on a schedule from fresh rows with no feature search, using refresh.py under cron, Airflow or any scheduler, and handle the pipeline errors that follow a schema, model or runtime change. Use when someone wants predictions refreshed every day, week or month, asks how to deploy or productionize a Nori-Rel prediction, or a saved pipeline stops running.
---

# Schedule a Nori-Rel prediction

`pipeline.json`, written by [nori-rel-predict](../nori-rel-predict/SKILL.md),
is the complete recipe of one prediction: the query, the feature plan (and
the agent's SQL after an agent search), the schema it expects, the context
rule, the model and the runtime version. It holds no credentials, rows or
weights. A later run loads it, reads fresh rows, places the past dates the
same distance before the new date, and predicts without a feature search.
Static targets have no pipeline; schedule `predict.py` for them instead.

## 1. Prove the replay once

Run the pipeline at the same date as the original prediction and compare:

```bash
python scripts/refresh.py nori_rel_source.json nori_rel_output/pipeline.json \
  --as-of 2026-03-01 --observed-through 2026-03-01 \
  --task nori_rel_task.json --out nori_rel_output/replay.csv
```

The predictions must equal `predictions.csv`. `--task` passes the mode, model
and device the pipeline was saved with; the pipeline refuses a client with
another mode or model. Then run it at a later date to see the context move
forward.

## 2. Write the scheduled command

Each run needs two dates. `--as-of` is the date to predict from, normally the
run's date at a fixed hour in UTC. `--observed-through` is the date through
which the database is verified complete; it comes from the user's own
pipeline (the load that finished, the export date), not from the clock and
not from the newest row. Ask how they know it and write that rule into the
job. A run with `observed_through` later than the truth learns from
incomplete windows and underpredicts.

```bash
# Every Monday at 06:00 UTC, after the 05:00 load has finished.
0 6 * * 1  cd /srv/nori-rel && . .venv/bin/activate && \
  HF_HUB_OFFLINE=1 python refresh.py nori_rel_source.json pipeline.json \
    --as-of "$(date -u +%F)" --observed-through "$(date -u -d yesterday +%F)" \
    --task nori_rel_task.json --out "predictions_$(date -u +%F).csv" \
    >> refresh.log 2>&1
```

In Airflow, Dagster or a similar scheduler, run the same command in a task
that depends on the load, with the credential variable injected from the
scheduler's secret store. Copy `refresh.py`, `nori_rel_source.json`,
`nori_rel_task.json` and `pipeline.json` next to the job. The model weights
download on the first run; `HF_HUB_OFFLINE=1` afterwards keeps the job off
the network.

To write results into the database, use a separate identity that may write to
one results table, never the reader:

```python
import pandas as pd
from sqlalchemy import create_engine

frame = pd.read_csv("predictions_2026-03-09.csv")
frame["run_date"] = "2026-03-09"
engine = create_engine(os.environ["RESULTS_DATABASE_URL"])
frame.to_sql("nori_rel_predictions", engine, if_exists="append", index=False)
```

## 3. Watch the output

`refresh.py` prints the number of predictions, the context rows they were
learned from, the pipeline id and the context hash of the run. Keep these in
the log: the hash lets a past prediction be traced, and a drop in context
rows means the data behind the run changed. Exit code 2 means the pipeline
cannot run; the message names the error:

| Error | Meaning | What to do |
|---|---|---|
| `PipelineSchemaError` | A table or column changed, or a column's type did | Repair the source, or predict again and save a new pipeline |
| `PipelineModelError` | The client's mode or model, the `synthefy-nori` version or the weights differ from the saved ones | Run with the saved mode and model (`--task`), or predict again after an upgrade and save a new pipeline |
| `PipelineFeatureError` | The saved features no longer compute (another `fastdfs` version, a changed column) | Predict again and save |
| `PipelineTaskError` | A `--query` with a different target was passed | Only the `FOR` clause may change |
| `PipelineContextError` | `expected_context_sha256` was set and the context differs | Expected when the data changed; drop the check |
| `PipelineVersionWarning` | Saved by another `nori-rel` version; the run goes on | Re-save at the next deliberate upgrade |

An upgrade of `nori-rel` or the model is a deliberate step: predict again,
backtest ([nori-rel-backtest](../nori-rel-backtest/SKILL.md)), and save the
new pipeline next to the old one with the date in its name.

## 4. Keep it honest

Schedule a backtest too, monthly or once a quarter, so the error against the
references is tracked over time; a pipeline that was good in March can drift
by September. Report to the user: the command, where the outputs go, how
`observed_through` is set, and when the next backtest runs.
