# PQL, the parts you need

A query is `PREDICT <target> FOR EACH <table>.<key>`, with optional `WHERE`
before `FOR`. The full reference is `docs/pql.md` in the nori-rel repository
and the "Predict with PQL" section of its README.

## Targets

**Temporal:** an aggregate over an event table's rows in a window after the
anchor, `(anchor + start, anchor + end]`, `0 <= start < end`, unit `days`
(exact 24-hour periods) or `months` (UTC calendar months).

| Aggregate | Meaning | Empty window |
|---|---|---|
| `COUNT(orders.*,0,30,days)` | rows | 0 |
| `COUNT(orders.amount,0,30,days)` | rows where the column is not null | 0 |
| `COUNT_DISTINCT(orders.product_id,0,30,days)` | distinct non-null values | 0 |
| `SUM(orders.amount,0,30,days)` | sum | 0 |
| `AVG(...)`, `MIN(...)`, `MAX(...)` | average, minimum, maximum | unknown: the example is dropped |
| `FIRST(orders.amount,0,30,days)`, `LAST(...)` | the first or last event's value by time | unknown |

Arithmetic combines up to 8 aggregates with numbers: `+ - * /`, parentheses.
`SUM(a)/COUNT(b)` is a rate; division by zero is unknown, so examples with no
events are dropped. Each aggregate is computed on its own, so two event
tables in one target do not multiply each other's rows.

A comparison makes a yes/no target: `COUNT(orders.*,0,30,days)=0`,
`SUM(orders.amount,0,30,days)>100`, `FIRST(orders.approved,0,30,days)=true`.
Yes/no needs hosted Nori.

**Static:** a column of the entity table, `PREDICT users.income FOR EACH
users.user_id`. Numeric is regression; Boolean or a comparison
(`PREDICT users.plan = 'pro'`) is yes/no; a 0/1 numeric column needs
`task="regression"` or `task="classification"`. The target cannot be the key.

## Who is scored, who is learned from

- `FOR EACH users.user_id`: every entity that exists at `anchor_time` (static:
  every row whose value is NULL).
- `FOR users.user_id IN (3, 8)` or `FOR users.user_id=42`: those rows. For a
  static target they leave the examples, so stored values can be checked.
- `WHERE users.country='CA'` (before `FOR`): restricts the historical
  examples. Conditions join with `AND`; `IN`, comparisons and null checks are
  allowed; no `OR`, no aggregates in `WHERE`.
- `query_filter="users.country='CA' AND users.active=TRUE"` (a predict
  argument, not PQL text): restricts the scored rows and leaves the examples
  alone. Entity columns only.

## Names

Unquoted names are plain identifiers. Backticks quote names with dots, spaces
or hyphens, and pick an exact spelling on Snowflake:
`` PREDICT SUM(`order.items`.amount,0,30,days) FOR EACH `user.accounts`.`user.id` ``.
Date literals are `YYYY-MM-DD`.

## Limits

- The event table in a target links to the entity through one declared
  foreign key. A table two links away (order lines under orders) needs its
  own key to the entity.
- No `OR`; no aggregate conditions in `WHERE`; at most 8 aggregates; no
  arithmetic over static targets; windows do not look back before the anchor.
- `anchor_times`, `anchor_time` and `observed_through` are timezone-aware;
  the scripts take ISO dates and treat them as UTC.
