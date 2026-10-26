---
name: predict
description: Turn a business question ("how much will each customer spend next month", "fill in the missing incomes") into a Nori-Rel PQL prediction - choose the entity, target and window, set anchor_time, observed_through, the past dates and the creation column, run predict.py, and read the result. Use when a database is connected (nori_rel_source.json exists) and someone wants a prediction, asks how to write PQL, or gets a PQL, date or context-row error from predict().
---

# Predict with Nori-Rel

A prediction is one PQL query plus the dates that place it in time. Nori-Rel
turns them into labelled examples at past dates (one per entity per date, the
label computed over the window after the date, the features from everything
before it) and scores every entity that exists at `anchor_time`. Nothing is
trained; the examples are Nori's context.

You need `nori_rel_source.json` from
[connect](../connect/SKILL.md). Read
[references/pql.md](references/pql.md) for the grammar before writing a query.

## 1. The question becomes a query

| The user asks | Query |
|---|---|
| How much will each customer spend in the next 30 days? | `PREDICT SUM(orders.amount,0,30,days) FOR EACH customers.customer_id` |
| How many tickets will each account open next week? | `PREDICT COUNT(tickets.*,0,7,days) FOR EACH accounts.account_id` |
| What will each store's sales be next calendar month? | `PREDICT SUM(sales.total,0,1,months) FOR EACH stores.store_id` |
| Average order value per customer over the next quarter | `PREDICT SUM(orders.amount,0,90,days)/COUNT(orders.*,0,90,days) FOR EACH customers.customer_id` |
| Spend 30 to 60 days out (a delayed window) | `PREDICT SUM(orders.amount,30,60,days) FOR EACH customers.customer_id` |
| Will each customer order at all next month? (yes/no, hosted only) | `PREDICT COUNT(orders.*,0,30,days)>0 FOR EACH customers.customer_id` |
| Fill in the missing `income` of each user (no dates) | `PREDICT users.income FOR EACH users.user_id` |

The `FOR EACH` table is the entity. The event table in the target must have a
declared foreign key straight to it and a registered time column. `0,30,days`
is the window `(anchor, anchor + 30 days]`; `months` counts UTC calendar
months and suits monthly data. Add `WHERE` to restrict the historical
examples, `query_filter` to restrict who is scored.

Ask before guessing when the question leaves a choice: which amount column,
gross or net of refunds, whether cancelled orders count. Write the query and
say in one sentence what it predicts.

## 2. The dates (temporal targets)

| Setting | How to choose it |
|---|---|
| `anchor_time` | The date to predict from. Usually the last date the data is complete; a past date for a check. The result is the window after it. |
| `observed_through` | The date through which every event table is complete. **The user states it.** Never set it from the newest row: a recent row does not prove earlier rows are all in. Past dates whose window ends after it give no examples. |
| `past_dates` | How many past dates Nori learns from. Each gives one example per entity that existed by then. Default 12, placed one window apart before `anchor_time` (`anchor_spacing` changes the gap). The oldest date must be after the data starts, and a seasonal business wants last year's season among them. The agent search needs at least 5. |
| `entity_created_at` | The entity table's creation column. An entity has no example before it and is not scored if created after `anchor_time`. |
| `max_context_rows` | Cap on examples, default 5,000, at most 60,000. Candidates are taken newest date first. With more entities than the cap, only the newest date counts: raise it (entities times past dates), and say what it costs in memory. |

Count the examples before running: entities created by the oldest date, times
past dates, against the cap. Fewer than 32 usable examples fails the run.

A **static target** (a column of the entity table) takes no dates: the rows
where the value is stored are the examples, the rows where it is NULL are
scored, and the column never enters the features. Set `"anchor_time": null`
and leave out `window`, `past_dates` and `observed_through`. Static targets
run with FastDFS only and have no pipeline.

Yes/no targets (a comparison, or a Boolean column) need
`"mode": "hosted"` and `SYNTHEFY_NORI_API_KEY`; local Nori is regression only.

## 3. Write the task and run it

Write `nori_rel_task.json` (the fields are listed in
[../README.md](../README.md)) and run:

```bash
python scripts/predict.py nori_rel_source.json nori_rel_task.json --out-dir nori_rel_output
```

It prints each stage, then a summary. Read these lines:

- `context rows: N usable of M candidates`: at least 32, and ideally
  hundreds. Many unknown labels dropped means windows with no events for an
  `AVG`, `MIN`, `MAX`, `FIRST`, `LAST` or ratio target; see
  [references/troubleshooting.md](references/troubleshooting.md).
- `labelled rows per past date`: a date with far fewer rows than the others
  had fewer entities or incomplete data. A missing date fell after
  `observed_through` minus the window, or before the data starts.
- `features N columns`: a handful means few tables link to the entity; see
  [features](../features/SKILL.md).

Outputs in `nori_rel_output/`: `predictions.csv` (`entity_id`,
`anchor_time`, `prediction`, plus `quantiles` or `probability`),
`context.csv` (the examples with their labels), `features.csv`,
`feature_table.csv`, `metadata.json` and `pipeline.json`.

## 4. Read the result

The default output is the median of Nori's predictive distribution; set
`"output_type": "mean"` when totals matter more than typical values, or
`"output_type": "quantiles"` with `"quantiles": [0.1, 0.5, 0.9]` for a
range. A count or sum can come back slightly negative; say so and clip at zero
in the report rather than in the file.

Before anyone acts on the numbers, run
[backtest](../backtest/SKILL.md). Then save the pipeline for
[schedule](../schedule/SKILL.md).

Report the query, `anchor_time`, `observed_through`, the past dates and how
many examples each gave, the cap, the model, the feature strategy and the
`nori-rel` version. Report every assumption as an assumption.

## Rules

- Do not infer `observed_through`, and do not move it to make more past
  dates fit. Ask.
- Do not change the query, the dates or the strategy after seeing a backtest
  on the windows you then report. Choose on earlier folds, report on the
  last.
- Local mode keeps rows in the process. Say what hosted mode sends (feature
  matrices and labels) before switching to it.
