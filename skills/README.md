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
