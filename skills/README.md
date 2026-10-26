# Agent skills for Nori-Rel

A skill is a folder with a `SKILL.md` that an AI coding agent (Claude Code,
Codex CLI, Cursor and others that follow the
[Agent Skills](https://agentskills.io) format) loads when a task matches its
description. These skills take an agent from an empty environment to a
scheduled Nori-Rel prediction. Each one works on its own; `nori-rel-quickstart`
runs them in order.

| Skill | Use it when |
|---|---|
| [`nori-rel-quickstart`](nori-rel-quickstart/SKILL.md) | You want the whole path: install, connect, predict, check, schedule. |
| [`nori-rel-setup`](nori-rel-setup/SKILL.md) | Install Nori-Rel with the right extras, choose local or hosted Nori, verify the environment. |
| [`nori-rel-prepare-data`](nori-rel-prepare-data/SKILL.md) | Your data is CSV, Excel, Parquet or JSON files rather than a relational database. |
| [`nori-rel-connect`](nori-rel-connect/SKILL.md) | Connect PostgreSQL (including AWS RDS and Aurora), Snowflake or Databricks and check the schema. |
| [`nori-rel-predict`](nori-rel-predict/SKILL.md) | Turn a business question into PQL, choose the dates, run the prediction. |
| [`nori-rel-backtest`](nori-rel-backtest/SKILL.md) | Measure the prediction on windows whose outcome is known, next to simple references. |
| [`nori-rel-features`](nori-rel-features/SKILL.md) | Choose FastDFS, NoriDFS or the agent search; read and explain the features. |
| [`nori-rel-schedule`](nori-rel-schedule/SKILL.md) | Save the pipeline and predict again on a schedule from fresh rows. |

## Install

**Claude Code, as a plugin.** The repository is its own marketplace:

```text
/plugin marketplace add Synthefy/nori-rel
/plugin install nori-rel@synthefy-nori-rel
```

The skills then appear as `/nori-rel:nori-rel-quickstart`,
`/nori-rel:nori-rel-connect` and so on, and Claude also picks them up from a
plain request such as "connect my Snowflake tables to Nori-Rel". To try them
from a checkout without installing: `claude --plugin-dir .`

**Claude Code, as plain skills.** Copy the folders into your personal or
project skills directory. They are then `/nori-rel-connect` and so on:

```bash
cp -r skills/nori-rel-* ~/.claude/skills/      # every project
cp -r skills/nori-rel-* .claude/skills/        # this project only
```

**Codex CLI.** Copy the same folders into `~/.codex/skills/`.

**Any other agent.** Give it the `SKILL.md` as instructions and let it read
the folder's `scripts/` and `references/`.

Install only the skill folders; `README.md` is for people.

## How the skills hand work to each other

The skills pass three files between steps. Keep them next to your project, out
of version control if the table names are sensitive. None holds a credential.

**`nori_rel_source.json`**, written by `nori-rel-connect`, describes the
database. The credential is a reference, never a value:

```json
{
  "engine": "postgresql",
  "credential": {"environment": "DATABASE_URL"},
  "schema_name": "public",
  "tables": ["users", "orders"],
  "time_columns": {"orders": "event_time"},
  "primary_keys": {},
  "foreign_keys": {},
  "connection": null
}
```

`engine` is `postgresql`, `snowflake` or `databricks`. `credential` is
`{"environment": "VAR"}` (a PostgreSQL URL, or warehouse authentication JSON)
or `{"aws_secret": {"secret_id": "...", "region_name": "..."}}`. Snowflake
and Databricks fill `connection` with the fields of `SnowflakeConnection` or
`DatabricksConnection` and usually `primary_keys` and `foreign_keys`, as
`nori-rel-connect` explains.

**`nori_rel_task.json`**, written by `nori-rel-predict`, describes one
prediction:

```json
{
  "query": "PREDICT SUM(orders.amount,0,30,days) FOR EACH users.user_id",
  "window": {"end": 30, "unit": "days"},
  "anchor_time": "2026-03-01",
  "observed_through": "2026-03-01",
  "past_dates": 12,
  "entity_created_at": "created_at",
  "max_context_rows": 5000,
  "feature_strategy": "fastdfs",
  "output_type": null,
  "quantiles": null,
  "query_filter": null,
  "mode": "local",
  "model": null,
  "device": null
}
```

`window.end` and `window.unit` (`days` or `months`) repeat the window in the
query; the past dates are placed that far apart before `anchor_time`, unless
`anchor_spacing` (`{"value": 7, "unit": "days"}`) says otherwise. A static
target has `"anchor_time": null` and no window. `feature_strategy` is
`"fastdfs"`, `"noridfs"` or an agent object described in `nori-rel-features`.
`model` and `device` are `null` for the installed version's defaults.

**`nori_rel_output/`**, written by the scripts: `predictions.csv`,
`context.csv` (the labelled past rows Nori learned from), `features.csv`,
`feature_table.csv`, `metadata.json`, `pipeline.json` when the prediction can
be replayed, and `backtest.csv` with `backtest_rows.csv` after a backtest.

## Rules every skill follows

- Credentials stay in environment variables or AWS Secrets Manager. The agent
  never asks for a password in chat and never writes one to a file.
- `observed_through` is a fact the user states: the date through which the
  database is complete. It is never read off the newest row.
- Database rows stay in the process. Only in hosted mode do feature matrices
  and labels leave it, and only the agent feature search sends schema, column
  profiles and query samples to a model provider. Each skill says so before
  enabling either.
- The agent reads the installed version's README and `docs/` before working
  around a missing capability, and names the version in what it reports.

## Try the whole path on synthetic data

The repository's synthetic fixture (48 users, 1,248 weekly orders, no real
records) is enough to run every script. Everything below stays on your
machine; in local mode nothing leaves it.

```bash
docker run -d --name nori-rel-pg -e POSTGRES_PASSWORD=example \
  -e POSTGRES_DB=example -p 127.0.0.1:5432:5432 postgres:16-alpine
export DATABASE_URL='postgresql://postgres:example@127.0.0.1:5432/example'
python -m pip install "nori-rel[database,local]"
python examples/load_fixture.py
```

The loader creates the `nori_rel_example` schema and prints the newest order
time. Write the two handoff files. The source file is the one shown above
with `"schema_name": "nori_rel_example"`; the task file is the one shown above
with `"past_dates": 4`, `"output_type": "quantiles"` and
`"quantiles": [0.1, 0.5, 0.9]`. March 2, 2026 is the fixture's completeness
watermark, so `anchor_time` and `observed_through` both say `2026-03-02`.
Then run the scripts in the quickstart's order, from the skills directory you
installed them to:

```bash
S=skills
python $S/nori-rel-connect/scripts/check_source.py nori_rel_source.json --entity users
python $S/nori-rel-predict/scripts/predict.py nori_rel_source.json nori_rel_task.json
python $S/nori-rel-backtest/scripts/backtest.py nori_rel_source.json nori_rel_task.json --folds 2
python $S/nori-rel-schedule/scripts/refresh.py nori_rel_source.json \
  nori_rel_output/pipeline.json --as-of 2026-03-02 --observed-through 2026-03-02 \
  --task nori_rel_task.json
```

What to expect. The check prints `connection: ready`, a `PASS` for
`orders` as an event table on `event_time`, a `PASS` for `orders` as the event
table linked straight to `users`, and `0 failures`. The prediction downloads
Nori 1.5 on first use, learns from 192 context rows (48 users at four past
dates), scores 48 rows with a 10%, 50% and 90% quantile each, and saves
`nori_rel_output/pipeline.json`. The backtest predicts from January 31 and
January 1, 2026, and prints, per fold, the mean absolute error of Nori, of
repeating each user's previous window, and of the context median, with the
share of outcomes inside the 10%-90% band. The refresh reproduces the 48
predictions from the saved pipeline and prints the same context SHA-256 as
`nori_rel_output/metadata.json`. For a static target, change the task to
`PREDICT users.income FOR EACH users.user_id` with `"anchor_time": null`:
40 users have a stored income, 8 do not and are scored, and no pipeline is
saved. `docker rm -f nori-rel-pg` removes the database.

A database on another host must offer TLS: the PostgreSQL connector sets
`sslmode=require` for every non-loopback address.

## How the skills are tested

Two test files cover the skills; both run in the repository's CI job on every
pull request.

- [`tests/test_skills.py`](../tests/test_skills.py) checks the files: each
  `SKILL.md` has a `name` equal to its folder and a description under 1,024
  characters, every relative link resolves, every script compiles and
  documents its usage, every skill is in the table above, and the plugin
  manifests agree with each other.
- [`tests/test_skill_scripts.py`](../tests/test_skill_scripts.py) runs each
  script's `main()` on the synthetic fixture, in a PostgreSQL schema the test
  creates and drops, in the quickstart's order. With
  `NORI_REL_TEST_DATABASE_URL` set and no weights, the local Nori regressor is
  replaced by a stand-in, so the run takes seconds. It checks the `PASS` lines
  and the `FAIL` exit for a wrong time column; 48 predictions with quantiles,
  192 context rows, metadata, a feature table and a saved pipeline for both
  FastDFS and NoriDFS; 8 scored rows, 40 context rows and no pipeline for the
  static target; two backtest folds at the expected cutoffs with 96 scored
  rows and a coverage between 0 and 1; a five-row static holdout; a replay
  with the same predictions and context hash, a filtered replay of the 16
  Canadian users, and exit code 2 with `PipelineModelError` when the client
  names another model.

The same flow runs on real, cached Nori 1.5 weights with

```bash
uv run --no-sync pytest tests/test_skill_scripts.py -m live_nori --require-local-model
```

which takes a few minutes. Not covered automatically: the agent feature
search against a live model provider, Snowflake and Databricks sources (the
scripts build the same connection objects the connectors' own tests cover),
hosted mode, and the prose of the skills, which an agent reads and whose
quality shows in what it does.

The scripts were last run by hand on 2026-10-26 with `nori-rel` 0.4.7 from
PyPI, `synthefy-nori` 0.20.1 and `nori-30m` on a GPU, against the same
fixture in a TLS PostgreSQL 16 container. The backtest table came out as
follows; a fold where repeating the previous window beats Nori is reported as
it is, which is what `nori-rel-backtest` asks of the agent.

| fold | cutoff | MAE Nori | MAE repeat previous | MAE context median | coverage |
|---|---|---|---|---|---|
| 1 | 2026-01-31 | 13.57 | 5.00 | 17.93 | 0.40 |
| 2 | 2026-01-01 | 9.45 | 14.49 | 19.83 | 0.79 |

The static holdout of 8 incomes gave an MAE of 82 for Nori against 7,456 for
the context median, and the replay reproduced the 48 predictions exactly.
