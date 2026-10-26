---
name: nori-rel-quickstart
description: Take a new user through Nori-Rel end to end - install it, connect their database or load their files, turn their question into a prediction, check it on windows whose outcome is known, and schedule it - calling the other nori-rel-* skills in order and stopping at each point that needs the user. Use when someone says they want to try Nori-Rel, predict something from their database, or asks where to start.
---

# Nori-Rel from zero to a scheduled prediction

Nori-Rel predicts a number per row of a table (each customer's spend next
month, each machine's downtime next week) from the linked tables around it.
Nothing is trained: Nori-Rel builds features at past dates from what had
happened by then, and Nori learns in context from those examples. Your job is
to get a user from "I have a database and a question" to a prediction they
have checked and can rerun. Do it in the six steps below, each with its own
skill, and keep the user in the loop at the gates.

Read [../README.md](../README.md) once for the files the steps pass between
them: `nori_rel_source.json`, `nori_rel_task.json` and `nori_rel_output/`.

## Before you start: ask three questions in one message

1. **What do you want to predict, for whom, over what period?** One sentence,
   such as "how much each customer spends in the next 30 days". This fixes the
   entity table, the target and the window.
2. **Where is the data?** PostgreSQL (including AWS RDS or Aurora), Snowflake,
   Databricks, or files. For a database: can the user give you a read-only
   login, held in an environment variable or an AWS secret?
3. **Through what date is the data complete?** This is `observed_through`.
   Nori-Rel cannot infer it; the user has to say.

If the target is yes/no ("will this customer churn"), say now that it needs
hosted Nori and a Synthefy API key; numeric targets run locally with no
account. Then proceed with what you have; ask for the rest when a step needs it.

## The steps

| Step | Skill | Done when | Gate |
|---|---|---|---|
| 1 | [nori-rel-setup](../nori-rel-setup/SKILL.md) | `nori-rel` imports in a fresh environment with the right extras; mode and model reported | none |
| 2a | [nori-rel-connect](../nori-rel-connect/SKILL.md) | `check_source.py` passes; `nori_rel_source.json` written | the user sets the credential variable |
| 2b | [nori-rel-prepare-data](../nori-rel-prepare-data/SKILL.md) (files only) | a PostgreSQL schema loaded from the files passes the same check | the user confirms every table and relationship before any load |
| 3 | [nori-rel-predict](../nori-rel-predict/SKILL.md) | `predictions.csv` and `pipeline.json` in `nori_rel_output/` | the user confirms the query, `anchor_time` and `observed_through` |
| 4 | [nori-rel-backtest](../nori-rel-backtest/SKILL.md) | `backtest.csv` compares Nori with repeat-previous-window and median references | report before anyone acts on step 3's numbers |
| 5 | [nori-rel-features](../nori-rel-features/SKILL.md) (if step 4 is weak) | a strategy chosen on earlier folds, confirmed on the last | the user approves the agent search's cost and data sharing |
| 6 | [nori-rel-schedule](../nori-rel-schedule/SKILL.md) | `refresh.py` runs the saved pipeline at a new date; the scheduled command is written down | the user confirms how `observed_through` is set on each run |

Run step 1 and 2 without waiting for more answers. Stop at every gate: a
schema the user has not confirmed is not loaded, a prediction whose dates the
user has not confirmed is not reported as a result, and the agent search is
not enabled without the user accepting what it sends to the model provider.

## What to report at the end

One message with:

- the query, `anchor_time`, `observed_through`, the past dates used, the
  context rows Nori learned from, the model and the `nori-rel` version;
- the backtest table: per fold, the mean absolute error of Nori, of repeating
  the previous window and of the context median, and quantile coverage if
  quantiles were requested;
- where `predictions.csv` and `pipeline.json` are, and the one command that
  refreshes the prediction;
- every assumption you made, in a list: which column is the creation time,
  which amount column, which rows you excluded, which entities fell outside
  the context cap.

Write the numbers as they are. A backtest where a simple reference beats Nori
is a finding to report, not something to tune away.
