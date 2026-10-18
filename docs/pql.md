# PQL predictions

PQL describes a future target for entities in a relational source. For example:

```text
PREDICT SUM(orders.amount,0,30,days) FOR EACH users.user_id
```

This predicts each user's total order amount during `(anchor, anchor + 30 days]`.
The source must describe the users-to-orders relationship through a foreign key.
`COUNT`, `SUM`, `AVG`, `MIN`, and `MAX` temporal aggregates are supported.
Direct column prediction and `NEXT` targets are unsupported.

The entity table is the one named in `FOR EACH` (`users`). Each event table in
the target (`orders`) needs a foreign key straight to the entity table and a
time column registered in `time_columns` at `connect()`. This page covers the
arguments that set up a prediction, the `entity_created_at` column, rates
written as a division of two counts, and the v0 limits.

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
  Direct-column targets and direct-column arithmetic are not supported.
- **Event tables** in the target must link to the entity table through one
  foreign key.
- **`query_filter`** selects scoring entities by entity columns, with
  comparisons, `IN`, null checks and `AND`. It does not change the context.
