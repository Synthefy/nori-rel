---
name: backtest
description: Check a Nori-Rel prediction before anyone acts on it, by predicting from earlier cutoffs whose outcomes are known and comparing with what happened, next to repeat-the-previous-window and median references and quantile coverage; static targets hold out stored values. Use when a prediction has been run (nori_rel_task.json exists) and someone asks how accurate it is, whether to trust it, or how Nori compares with a simple rule.
---

# Backtest a Nori-Rel prediction

A forecast from today cannot be checked yet. A forecast from an earlier date,
made with only the data up to that date, can be compared with what then
happened. This skill does that for the task in `nori_rel_task.json` and puts
Nori next to two references anyone could have used.

## 1. Run it

```bash
python scripts/backtest.py nori_rel_source.json nori_rel_task.json --folds 3 --out-dir nori_rel_output
```

For a **temporal** target the script takes `--folds` cutoffs, one window
before `anchor_time`, two windows before, and so on. For each cutoff it runs
the task exactly as written but anchored there, with `observed_through` moved
back to the cutoff, so nothing after the cutoff is visible. It reads the
actual outcomes of those windows in one extra prediction, and compares. The
data must cover `past_dates + folds + 1` windows before `anchor_time`; a
cutoff whose outcome window is not complete is skipped and reported.

For a **static** target (`"anchor_time": null`) it holds out `--holdout`
rows whose value is stored (at most all but 32), predicts them from the rest
with `FOR ... IN`, and compares with the stored values.

Each fold prints the number of entities compared and the mean absolute error
of three forecasts:

| Forecast | What it is |
|---|---|
| `nori` | Nori-Rel's prediction from the cutoff |
| `repeat_previous` | each entity's own value in the window before the cutoff (temporal only) |
| `context_median` | the median of the examples Nori learned from, the same number for every entity |

`backtest.csv` has one row per fold with MAE and RMSE of each forecast and,
when the task asks for quantiles, `coverage`: the share of actuals inside the
outer quantile band. `backtest_rows.csv` has every entity in every fold.

## 2. Read it

- Nori below both references on most folds: the prediction carries
  information beyond "same as last time" and "the typical value". Say by how
  much, in the target's units and as a share of the mean actual.
- Nori near `repeat_previous`: the entities are steady and last window is a
  strong rule; Nori adds little. Say so.
- Nori above a reference: before concluding, check the usual causes in
  order: too few examples (raise `past_dates` or `max_context_rows`), a
  window too short for the event rate (many zero labels), features that do
  not reach the tables that matter ([features](../features/SKILL.md)),
  a date with incomplete data among the past dates.
- Coverage of a 10-90 band should sit near 0.8. Much lower means the bands
  are too narrow to act on; much higher means they are too wide.
- Nori far better than both references on every fold, with errors near zero:
  suspect leakage (an entity column that encodes the future) before
  celebrating. See the troubleshooting table in
  [predict](../predict/references/troubleshooting.md).

One fold is one window; three folds on a seasonal business can all fall in
the same season. Run more folds when the history allows, and say how many
windows the verdict rests on.

## 3. Report

Give the user the fold table as printed, the pooled numbers, the number of
entities each fold compared against the number scored (entities outside the
context cap or without an outcome are not compared), and a one-sentence
verdict. Keep the numbers as they came out.

## Rules

- Choose the query, dates and feature strategy on earlier folds, and report
  the latest fold untouched. Do not rerun the backtest with changed settings
  and report only the best.
- The outcome read uses `max_context_rows`; if the script warns that it hit
  the cap, raise the cap in the task, or the check covers only the newest
  cutoffs' entities.
- A static backtest removes the held-out rows from the examples, so it needs
  at least 32 stored values beyond the holdout.
