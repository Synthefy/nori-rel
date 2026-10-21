# PQL predictions

PQL describes a future target for entities in a relational source. For example:

```text
PREDICT SUM(orders.amount,0,30,days) FOR EACH users.user_id
```

This predicts each user's total order amount during `(anchor, anchor + 30 days]`.
The source must describe the users-to-orders relationship through a foreign key.
`COUNT`, `SUM`, `AVG`, `MIN`, and `MAX` temporal aggregates are supported;
`NEXT` targets are not.

A **static** target is a column of the entity table, such as an account's
balance, with no time window. Nori learns it from the rows where the value is
known and predicts it for the rest. It needs no dates; see
[Static targets](#static-targets):

```text
PREDICT accounts.balance FOR EACH accounts.account_id
```

The entity table is the one named in `FOR EACH` (`users`). Each event table in
the target (`orders`) needs a foreign key straight to the entity table and a
time column registered in `time_columns` at `connect()`. This page covers the
arguments that set up a temporal prediction, the `entity_created_at` column,
rates written as a division of two counts, static targets, and the v0 limits.

## Historical examples and prediction time

Nori learns from labeled historical rows and then scores the entities that
exist at the prediction cutoff. Both come from one database snapshot:

- **Context rows (training).** One candidate per entity per historical anchor in
  `anchor_times`, if the entity existed at that anchor and the label window is
  fully observed. The label is the target computed over its window after the
  anchor. Candidates whose label is unknown are dropped. Features use only data
  up to the anchor.
- **Scoring rows.** One per entity that exists at `anchor_time`. Features use
  data up to `anchor_time`; the prediction is the target over the window after
  it.

Pass timezone-aware UTC datetimes:

| Argument | Meaning |
| --- | --- |
| `anchor_times` | Historical cutoffs. Each gives one candidate row per entity that existed by then. |
| `anchor_time` | Cutoff at which to predict; use an explicit value for repeatable work. |
| `observed_through` | Caller-verified completeness watermark for historical labels. |
| `entity_created_at` | Name of the column on the entity table that holds when each entity first existed. See [below](#entity_created_at-when-each-entity-first-existed). |
| `max_context_rows` | Cap on candidate context rows, default 5,000 and at most `RelationalLimits.max_context_rows` (60,000). Candidates are taken newest anchor first, then by primary key, before unknown labels are dropped. |
| `entity_time` | Optional, for one prediction per event: the entity table's timestamp column. Replaces `anchor_times` and `entity_created_at`. See [Event-time rows](#event-time-rows-one-prediction-per-event). |

A label window is `(anchor + start, anchor + end]`, counted in `days` (exact
24-hour periods) or `months` (UTC calendar months). With monthly data, use
months: `SUM(orders.amount,0,1,months)` from an anchor on the 1st covers that
calendar month, and `(1,2,months)` the month after it. A month added to the 31st
lands on the last day of a shorter month.

A label window must be fully observed before it can provide training context:
an anchor whose window ends after the earlier of `anchor_time` and
`observed_through` gives no training rows, and the client warns which anchors
it dropped. A 30-day window shows why months matter on monthly data: from
February 1 it ends on March 3, so a forecast dated March 1 loses every February
anchor. Features use only events available at
the corresponding row's anchor. Register each event table in `time_columns`
when connecting. Rows without an event timestamp and mutable entity attributes
are not automatically reconstructed as they existed historically: provide
immutable attributes or a source whose values are valid at all requested anchors.
The client assumes events are available at their recorded event time; delayed
arrival requires preprocessing into a valid source.

`observed_through` cannot be inferred from the largest event timestamp: a recent
event does not prove that earlier events are complete. Historical data corrections
can change later predictions even when the task and model stay the same.

## `entity_created_at`: when each entity first existed

`entity_created_at` names a date or timestamp column **on the `FOR EACH` table
itself**, holding when that entity first existed. Pass the bare column name,
such as `"created_at"`, not `"products.created_at"`. A column on an event table
does not work. Nori-Rel never creates this column and does not infer it.

It decides which entities take part:

- **Context.** An entity gets a candidate row at an anchor only if
  `created_at <= anchor`. Anchors before its creation give it no row.
- **Scoring.** An entity created after `anchor_time` is not scored. With
  `FOR EACH` it is left out of the result. Requesting it by ID
  (`FOR products.product_id=3`) raises an error.
- **Nulls.** An entity whose `created_at` is null gets no context rows and is
  never scored.
- **Dates.** A `DATE` column counts as midnight UTC of that day. Every connector
  runs its session in UTC.

The value must not use future information. The set of rows that exist at an
anchor is itself something the model learns from. If `created_at` were
`updated_at`, a product that sold again after an anchor would be missing from
that anchor's context, so the context would be built from the products that
went quiet. The time of the entity's first event is safe: it is known at every
anchor after it.

A value that is too late loses training rows and leaves real entities
unscored. One that is too early adds training rows from before the entity
existed, with features that describe no history.

If the name is wrong, the error says why and lists the entity table's
date/timestamp columns, or says the table has none.

### Worked example

A shop has a `products` entity table and two event tables, `page_views`
(`view_time`) and `purchases` (`purchased_at`). The products export has no
creation date, so use each product's first view.

```sql
ALTER TABLE shop.products ADD COLUMN created_at timestamptz;

UPDATE shop.products AS p
SET created_at = first_view.first_view_time
FROM (
    SELECT product_id, MIN(view_time) AS first_view_time
    FROM shop.page_views
    GROUP BY product_id
) AS first_view
WHERE first_view.product_id = p.product_id;
```

A product that appears in several event tables takes the earliest of them:
`LEAST(first_view_time, first_purchase_time)`. PostgreSQL's `LEAST` ignores
nulls. Nori-Rel discovers tables, not views, so the column must be on the table
itself. Products with no views keep a null `created_at` and drop out.

Check the result before predicting:

```sql
-- Entities that will be dropped: no creation time.
SELECT COUNT(*) FROM shop.products WHERE created_at IS NULL;

-- created_at after the product's first event: it is too late.
SELECT COUNT(*)
FROM shop.products AS p
JOIN (SELECT product_id, MIN(purchased_at) AS first_purchase_time
      FROM shop.purchases GROUP BY product_id) AS b USING (product_id)
WHERE p.created_at > b.first_purchase_time;
```

Then predict with weekly anchors on March 1, 8 and 15 and a cutoff of
March 22, with `(0,7,days)` windows:

```python
from datetime import UTC, datetime

from nori_rel import EnvironmentCredential, NoriRelClient

client = NoriRelClient()
source = client.connect(
    engine="postgresql",
    credential=EnvironmentCredential(variable="DATABASE_URL"),
    schema_name="shop",
    tables=["products", "page_views", "purchases"],
    time_columns={"page_views": "view_time", "purchases": "purchased_at"},
)
predictions = client.predict(
    source=source,
    query="PREDICT COUNT(purchases.*,0,7,days) FOR EACH products.product_id",
    anchor_times=[datetime(2026, 3, day, tzinfo=UTC) for day in (1, 8, 15)],
    anchor_time=datetime(2026, 3, 22, tzinfo=UTC),
    observed_through=datetime(2026, 3, 22, tzinfo=UTC),
    entity_created_at="created_at",
)
```

All three anchors are usable, since `March 15 + 7 days = March 22`. Four
products show each rule:

| `product_id` | `created_at` (first view) | Context rows from anchors | Scored at March 22 |
| --- | --- | --- | --- |
| 1 | 2026-02-20 09:12 | March 1, 8 and 15 | yes |
| 2 | 2026-03-10 14:00 | March 15 | yes |
| 3 | 2026-03-22 09:30 | none | no, created after `anchor_time` |
| 4 | null, never viewed | none | no |

## Rates: divide two counts

PQL has no rate aggregate. A rate such as purchases per view is written as one
count divided by another:

```text
PREDICT COUNT(purchases.*,0,7,days)
      / COUNT(page_views.* WHERE page_views.is_bot = false,0,7,days)
FOR EACH products.product_id
```

- Each count is computed separately, over its own event table and window, and
  then divided. A product's views never multiply its purchases.
- Arithmetic is decimal (`NUMERIC` on PostgreSQL), so 3 purchases over 40
  views is `0.075`, not integer `0`.
- **A zero denominator gives a NULL label, and that row is dropped.** A product
  with no views in the window gives `0 / 0`. Nori-Rel treats this as an unknown
  answer, not as a rate of zero: the row leaves the context and is counted in
  `dropped_unknown_labels`. Counting it as zero would mix "no traffic" with "no
  conversions", and give every product a row at every anchor. Only that
  (entity, anchor) row goes: the product still has rows at other anchors and
  is still scored.
- Use `output_type="mean"` for an expected rate.

In the example above, product 2 at the March 15 anchor had 40 views and 3
purchases in `(March 15, March 22]`, so its label is `0.075`. Product 1 had no
views between March 8 and March 15, so its March 8 row is `0 / 0` and is
dropped. Its March 1 and March 15 rows stay.

### An outcome that settles a fixed time after creation

Some outcomes resolve a fixed delay after the entity is created. Take a
`trials` entity table (`created_at` = trial start) and an `outcomes` event
table, where each trial gets one outcome row stamped exactly 7 days after it
started, with status `converted`, `cancelled` or `unknown`:

```text
PREDICT COUNT(outcomes.* WHERE outcomes.status = 'converted',6,7,days)
      / COUNT(outcomes.* WHERE outcomes.status IN ('converted', 'cancelled'),
              6,7,days)
FOR EACH trials.trial_id
```

Each trial has one outcome, so the label is `1 / 1 = 1` (converted),
`0 / 1 = 0` (cancelled), or `0 / 0` and dropped when no settled outcome falls
in the window. That covers `unknown` statuses and outcomes not yet recorded. A
comparison target such as `COUNT(...)>0` would instead label an unsettled trial
as not converted, since its count is 0. Use `output_type="mean"`: the mean of a
0/1 label is the conversion probability, and the default median is not.

**`(6,7,days)` keeps one training row per trial.** With daily anchors at
midnight UTC, take a trial that started on March 10 at 14:05 and settled on
March 17 at 14:05:

| Anchor | Trial exists? | Window | Label |
| --- | --- | --- | --- |
| March 10 00:00 | no | | no row |
| March 11 00:00 | yes | (March 17 00:00, March 18 00:00] | `1 / 1`, kept |
| March 12 00:00 | yes | (March 18 00:00, March 19 00:00] | `0 / 0`, dropped |
| later anchors | yes | later days | `0 / 0`, dropped |

The kept row is the one from the anchor at which the trial was less than a day
old, the same age as the trials you score on a run day. With `(0,7,days)`,
every anchor from March 11 to March 17 would contain the outcome. That gives
seven rows per trial, the older ones with features from days of the trial's own
activity that a new trial does not have yet.

Anchors less than 7 days before the cutoff are skipped, because their windows
have not been observed yet.

The general rule: for an outcome that settles exactly `L` days after creation,
with anchors every `s` days, use `(L - s, L, days)`. This needs `L >= s` and
evenly spaced anchors with no gaps: a trial created during a gap gets no row.

**The dropped rows still use up `max_context_rows`.** Each anchor contributes
every trial that exists by then as a candidate, but only about one day's trials
have a label. Candidates are capped newest anchor first before unknown labels
are dropped, and nothing refills them, so a small cap fills with unlabeled
rows from the newest anchors. On 300 synthetic trials with 29 daily anchors,
2,892 candidates held 161 labeled rows. Set the cap to the sum, over the
anchors you want, of the trials that exist at each one. A cap above
`RelationalLimits.max_context_rows` (60,000 by default) is refused; raise it
with `NoriRelClient(limits=RelationalLimits(max_context_rows=...))`. The INFO
log reports usable and candidate counts.

Every trial that exists at `anchor_time` is scored, including old ones whose
outcome is already known. With `anchor_time` at March 22 00:00, score only
the trials started on March 21 with
`query_filter="trials.created_at >= 2026-03-21"`.

A numeric outcome column gives the same rows more simply. If `outcomes`
has `converted` set to 1, 0 or null, the target is
`AVG(outcomes.converted,6,7,days)`. An average over no rows, or only
nulls, is NULL, so the same rows are dropped.

## Select entities

Use a `FOR` clause for particular scoring entities:

```text
PREDICT SUM(orders.amount,0,30,days) FOR users.user_id=42
```

Pass `query_filter="users.country='CA'"` to select scoring entities by an entity
predicate. This filter intersects the `FOR` selection and does not restrict
historical context. No matching entities returns an empty result. A NULL
predicate excludes that row. Historical target `WHERE` conditions and scoring
`query_filter` serve different purposes.

Use backticks for identifiers with spaces, dots, or punctuation:

```text
PREDICT SUM(`order.items`.amount,0,30,days) FOR `user.accounts`.`user.id`=42
```

Configure `tables`, keys, and time mappings with the original database names.
PostgreSQL and Databricks use the discovered names; Snowflake also accepts an
unambiguous case-insensitive match for unquoted names. Backticks require an exact
match. Do not interpolate untrusted strings into PQL.

## Results

A prediction is a pandas `DataFrame` with `entity_id`, `anchor_time`, and
`prediction`. `frame.attrs` records the task, model, feature/context counts, and
audit hashes. `return_details=True` returns `PredictionDetails` with
`predictions`, `metadata`, and `pipeline` attributes.

Regression accepts `output_type="mean"`, `"median"` (the default), or
`"quantiles"` with a list such as `quantiles=[0.1, 0.5, 0.9]`. Quantile results
include a `quantiles` column. A binary predicate target, for example
`COUNT(orders.*,0,30,days)=0`, requires hosted mode and also returns a
`probability` column. Local mode rejects binary targets.

Empty and all-null aggregate windows follow the target's SQL semantics; missing
labels are not silently replaced by zero. The default request context cap is
5,000 rows, additionally bounded by `RelationalLimits`. Too few usable labels
raises an error. Use [prediction metadata and limits](api.md) to inspect the
resulting context.

## Static targets

A static target predicts a column of the `FOR EACH` table, with no time
window: the rows where the value is known are the examples, and the rows where
it is NULL are scored. The names follow Kumo's static and temporal predictive
queries. A table with no dates works:

```python
source = client.connect(
    engine="postgresql",
    database_url=os.environ["DATABASE_URL"],
    tables=["branches", "accounts", "transactions"],
)
predictions = client.predict(
    source=source,
    query="PREDICT accounts.balance FOR EACH accounts.account_id",
)
```

- **Context rows.** Every row that passes the top-level `WHERE` and has a
  known value (not NULL, and not NaN), minus the rows being scored, up to
  `max_context_rows` (default 5,000), which counts these rows only. Above the
  cap they are taken by primary key, so on a large table the context is the
  lowest keys, often the oldest rows; with `entity_created_at`, newest first.
  For a uniform sample, pass
  `context=ContextPolicy(max_rows=5_000, time_sampling="random")`; the same
  seed keeps the same rows on every run. See
  [the context policy guide](context-policy.md). If the `FOR EACH` table has a
  registered time column, rows dated after the read are left out, as they are
  from the feature tables.
- **Scoring rows.** `FOR EACH` scores the rows whose value is NULL or NaN.
  `FOR accounts.account_id IN (3, 8)` scores those rows even when their value
  is stored, and they then leave the context, so you can check predictions
  against stored values without the model seeing them. `query_filter` narrows
  the scored rows; a `FOR` row it rejects stays in the context.
- **Features.** The target column is removed from the tables before features
  are built. No feature reads any row's value: not the row's own column, and
  not a path through related tables such as a branch's average balance.
  Columns that copy or encode the target, such as `balance_band`, are yours to
  remove: connect a view without them.

| Argument | Static target |
| --- | --- |
| `anchor_times`, `observed_through`, `anchor_time` | Not accepted. Values are read as stored at the time of the read, and there is no label window to place. For a per-row cutoff, use `entity_created_at`. |
| `entity_created_at` | Optional. Without it, every row's features use the time of the read: PostgreSQL's `CURRENT_TIMESTAMP`, or the warehouse's pinned extraction time. With it, each row's features are computed as of its own creation time, and rows with a null or later time are left out. |

Leave `entity_created_at` out when the value is the row's current state, such
as a balance. Pass it when the value describes the row when it was created,
such as fraud on a transaction: an old transaction's features then stop at its
own time and do not include later events, like a chargeback, that a new
transaction cannot have yet. The `anchor_time` column of the result is each
row's cutoff.

The task follows the column:

- A Boolean column, or a comparison such as `PREDICT users.income > 50000`, is
  binary: the result has `probability` and a Boolean `prediction`. Like all
  binary PQL, it needs `mode="hosted"`.
- A numeric column is regression, with the usual `output_type` and
  `quantiles`.
- A numeric column whose known values, all of them and not only the context
  rows, are 0 and 1 is ambiguous and raises until you pass
  `task="classification"` or `task="regression"`. Regression with
  `output_type="mean"` returns the probability of 1, which is how to get one
  with local Nori.
- A text column needs a comparison, such as `PREDICT users.plan = 'pro'`. The
  target must be a column of the `FOR EACH` table and cannot be its primary
  key.

Not supported yet:

- Backtests at a past cutoff. They need to know when each value was stored,
  through a label-availability column or versioned labels.
- Predictions for every stored row at once (leave-one-out). Name the rows in
  `FOR` instead.
- Feature searches other than FastDFS or a custom `FeatureBackend`: NoriDFS,
  the agent search and saved programs scale with a label window. Saved
  pipelines: `details["pipeline"]` is None.

## Event-time rows: one prediction per event

Some questions are about events, not entities: will this page view lead to a
signup, is this transaction fraud. Each row of the entity table is one event,
and its features should use everything known at the moment of the event,
including what happened minutes before it. A shared anchor cannot give that:
with `anchor_times`, every row is cut at the same midnight.

`entity_time` names the entity table's timestamp column. Each entity row then
becomes one example anchored at its own time `t`:

```python
details = client.predict(
    source=source,
    feature_strategy="noridfs",
    # 1 if a signup credited to this view comes within a day of it.
    query="PREDICT COUNT(signups.*, 0, 1, days) FOR EACH views.view_id",
    entity_time="view_time",         # each view is anchored at its own time
    anchor_time=run_time,            # the run: rows existing by then are scored
    observed_through=run_time,       # context labels must have settled by then
    query_filter="views.view_time >= 2026-07-25",  # score the run day's views
    context=ContextPolicy(balance=ClassBalance(positives="all"), max_rows=20_000),
    output_type="mean",
    return_details=True,
)
```

- **Label.** The window is measured from the row's own time: `(0, 1, days)` is
  `(t, t + 1 day]`.
- **Context rows.** One candidate per entity row, with no `anchor_times`. A row
  is eligible when `t + end <= min(anchor_time, observed_through)`, that is, its
  label had settled by the run. `ContextPolicy` chooses among them as usual.
- **Scoring rows.** Entity rows with `t <= anchor_time` that pass `query_filter`,
  each scored at its own time. Scoring a day's events in one batch gives the same
  features as scoring each event when it happens.
- **Features.** The row's own columns are visible at its own time. NoriDFS reads
  other rows strictly before `t`, so rows at the same instant are not each
  other's history; FastDFS keeps its rule of rows up to and including the
  cutoff.
- **Report.** `metadata["context_policy"]` counts the rows by UTC day, and
  `equal_per_cutoff` balances days.
- **Not yet.** The agent search, saved feature programs and pipelines raise; no
  pipeline is recorded. An entity at several times is a later step.

### History from the same parent: sibling paths

A view's most useful history is often its device's other views, which point at
the device rather than at the view. NoriDFS reaches them through an opt-in
sibling path: up one foreign key to a parent, then down to the parent's
time-stamped rows.

```python
from nori_rel.feature.dfs.noridfs import NativeFeatureConfig, NoriDFS, SiblingPathConfig

device_history = NoriDFS(
    NativeFeatureConfig(max_features=128, max_direct_features=32),
    horizon="1D",
    sibling_paths=SiblingPathConfig(parent_tables=("devices",)),
)
details = client.predict(
    source=source,
    query="PREDICT COUNT(signups.*, 0, 1, days) FOR EACH views.view_id",
    feature_strategy=device_history,
    entity_time="view_time",
    anchor_time=run_time,
    observed_through=run_time,
    return_details=True,
)
```

For each row, over the parent's rows strictly before the row's cutoff:

| Feature | Meaning |
| --- | --- |
| `count` | All earlier rows, and those in the last 1 hour, 6 hours, 24 hours and 7 days (`window_ns`). |
| `recency`, `oldest_age` | Days since the latest and the earliest earlier row. |
| `distinct` | Distinct values of each of their foreign keys, and distinct timestamps. |
| `match_count`, `match_recency`, `match_previous` | For each foreign key the row shares with them, such as the drug: earlier rows with this row's value, days since the latest one, and 1 when the latest earlier timestamp had it. |

With `entity_time`, the cutoff is each row's own time, so a page view counts its
device's views earlier the same day and never itself or a view logged at the
same instant. With `anchor_times`, the parent's rows before the shared anchor
include the entity's own row. `parent_tables` picks the parents to go through,
`max_paths` caps the paths, same-table siblings first. Sibling paths are off by
default, so `feature_strategy="noridfs"` is unchanged.

More `SiblingPathConfig` options, all off by default:

| Option | What it adds |
| --- | --- |
| `include_numeric_aggregates=True` | `sum` and `mean` of each numeric column of the earlier rows, over all of them and over each window, plus their all-time `min` and `max`. `numeric_columns=("claims.paid_claims",)` limits the columns; each adds two features per window. |
| `parent_tables=("drug_classes",)` | A table one step above a parent groups that parent's rows by it: a signup reaches the earlier rows of every drug in its drug's class, not just its drug's. |
| `excluded_paths=(("drug_classes", "views"),)` | Leaves out (parent, child table) pairs. Use it when a child table holds only part of a parent's rows, chosen by what happens later: a views table kept only for devices that sign up describes a device's history fully, but a drug class's only partly. |
| `max_features=40` | Caps the sibling family's share of `max_features`. Sibling features are planned before every other path, so without a cap they can take the whole budget. |

### The rows a path reaches, as a whole

`NoriDFS(include_composition_features=True)` describes the rows a descendant or
bridge path reaches, such as a signup's journey views and their drugs and
pharmacies, in the rows' own time order:

| Feature | Meaning |
| --- | --- |
| `first` | The first row's value, for every column. |
| `entropy` | How mixed a categorical column's values are, in nats. |
| `mode_share`, `last_share` | The share of rows with the most common value, and with the last row's value. |
| `distinct` | The number of distinct values. |

`NoriDFS(composition_pairs=(("drugs.drug_class", "pharmacies.company"),))` adds
the same features, and the mode, over pairs of values. Each pair comes from one
row both columns hang off, such as one view's drug class and pharmacy company.
A pair whose names reach no path plans nothing.

A value two steps above the rows, such as a view's drug class (view, drug,
class), needs `NativeFeatureConfig(max_depth=3)`. A column with many values also
needs `max_categorical_cardinality` (default 32) set above its number of values.

The order and ages of these rows follow each row's own time, not the path's.
A path's time is the latest along it, so every journey view of a signup carries
the signup's time on the path. `first` and `last` still follow the order of the
views, and `recency`, `mean_age` and `oldest_age` count from when each view
happened.

## v0 limits

- **No `OR`.** Filters combine conditions with `AND` only. Use `IN` for
  alternatives on one column, as the denominator above does. For anything else,
  add a column to the event table when loading the data.
- **No aggregate filters.** A `WHERE` clause cannot contain an aggregate, such
  as keeping only products with views. A ratio already drops the rows whose
  denominator is zero.
- **Windows** use `days` or `months`, with `0 <= start < end`. A window cannot
  look back before the anchor.
- **Arithmetic** combines at most 8 aggregates with numeric literals.
  Arithmetic over static targets is not supported.
- **Event tables** in the target must link to the entity table through one
  foreign key.
- **`query_filter`** selects scoring entities by entity columns, with
  comparisons, `IN`, null checks and `AND`. It does not change the context.
