# PQL predictions

PQL describes a future target for entities in a relational source. For example:

```text
PREDICT SUM(orders.amount,0,30,days) FOR EACH users.user_id
```

This predicts each user's total order amount during `(anchor, anchor + 30 days]`.
The source must describe the users-to-orders relationship through a foreign key.
`COUNT`, `SUM`, `AVG`, `MIN`, and `MAX` temporal aggregates are supported.
Direct column prediction and `NEXT` targets are unsupported.

## Historical examples and prediction time

Pass timezone-aware UTC datetimes:

| Argument | Meaning |
| --- | --- |
| `anchor_times` | Historical cutoffs used to construct labeled context rows. |
| `anchor_time` | Cutoff at which to predict; use an explicit value for repeatable work. |
| `observed_through` | Caller-verified completeness watermark for historical labels. |
| `entity_created_at` | Creation-time column on the entity table. |

A label window is `(anchor + start, anchor + end]`. It must be fully observed
before it can provide training context. Features use only events available at
the corresponding row's anchor. Register each event table in `time_columns`
when connecting. Rows without an event timestamp and mutable entity attributes
are not automatically reconstructed as they existed historically: provide
immutable attributes or a source whose values are valid at all requested anchors.
The client assumes events are available at their recorded event time; delayed
arrival requires preprocessing into a valid source.

`observed_through` cannot be inferred from the largest event timestamp: a recent
event does not prove that earlier events are complete. Historical data corrections
can change later predictions even when the task and model stay the same.

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
