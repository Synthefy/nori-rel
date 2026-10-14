---
name: nori-rel-prepare-data
description: Turn CSV, Excel, Parquet or JSON files into a PostgreSQL database that Nori-Rel can predict from. Use when someone wants Nori-Rel predictions but has exported files, spreadsheets or a flat table instead of a relational database, or when an existing database fails Nori-Rel's checks.
---

# Prepare data for Nori-Rel

Nori-Rel predicts from a relational database: entity tables (customers,
products), event tables (orders, visits, tickets) and the foreign keys that
link them. For each past date it builds features only from what had happened
by then, and learns from what happened next. Most people have files instead: a
CRM export, a folder of monthly CSVs, a workbook with one sheet per year.

Your job is to turn those files into a PostgreSQL schema that meets the
contract below, prove it with Nori-Rel's own discovery and one prediction, and
leave the user a load script they can rerun when new files arrive. You do every
step yourself with pandas, SQL and the Nori-Rel client. There are no helper
scripts.

**The user decides the schema.** Never create, alter or load a table the user
has not confirmed. Step 3 ends with their explicit confirmation of every
table, column and relationship, and nothing touches the database before it.

## The contract

A schema Nori-Rel can use well has all of these. Step 5 checks each one.

1. **Every table has a single-column primary key.** Use the natural ID when
   it is unique and never null; otherwise add `id BIGINT GENERATED ALWAYS AS
   IDENTITY`. A composite natural key stays as a `UNIQUE` constraint next to
   the surrogate key.
2. **Relationships are declared foreign keys** (`REFERENCES`). Nori-Rel finds
   links only from the database catalog, and follows single-column foreign
   keys only. A table nothing links to is ignored.
3. **Each event table has one time column**: when the event happened, or when
   it became known if that is later. The user passes it to Nori-Rel as
   `time_columns={"orders": "ordered_at"}`. Use `timestamptz`, in UTC. A child
   of an event table (`order_items` under `orders`) is an event table too:
   copy its parent's time onto it and register it, so its rows appear no
   earlier than the parent's. When a parent's own rows disagree (an invoice
   whose lines carry different times), use the latest: the document is
   complete only then.
4. **Every event table you care about links straight to the entity.** A
   query such as `SUM(order_items.amount,0,30,days) FOR EACH
   customers.customer_id` needs a foreign key from `order_items` to
   `customers`; a path through `orders` fails with `NO_DIRECT_RELATIONSHIP`.
   The default features also use only tables one link from the entity (see
   step 6). Copy the entity's key onto the child rows
   (`order_items.customer_id`) and declare it, next to the link to the parent.
5. **The entity you predict for has a creation time** (`created_at`). Nori-Rel
   scores and learns only from entities that existed at each date. If the data
   has no creation date, use the first time the entity appears in any event
   table, and say so.
6. **Entity tables hold only facts that do not change.** See the next section.
   This rule matters more than all the others together.
7. **Types the default features use:** numbers, booleans (as 0/1), and text
   with at most 9 distinct values (as categories). They skip text with 10 or
   more distinct values (a 38-value `country`, a product description) and
   dates other than time columns. If such a column matters, group the text
   into fewer values or turn it into true/false flags (`is_uk`). Store dates
   as `timestamptz` in UTC, at midnight for plain dates. JSON, arrays and
   binary columns are ignored: split them into columns or child tables.
8. **Size:** by default at most 1,000,000 rows per table, 3,000,000 in total
   and 32 tables. These are `RelationalLimits` defaults that the user can
   raise (`NoriRelClient(limits=RelationalLimits(max_rows_per_table=...))`).
   Nori-Rel refuses to load over a limit. History rows for learning are capped
   separately; see step 6.
9. **Names are lowercase `snake_case`**, so PQL needs no quoting. Tables are
   plural nouns (`customers`, `order_items`); keys are `<singular>_id`.

## Leakage: values that change over time

Nori-Rel reads an entity table's columns as true at every past date. A
column that changed after the row was created gives the model the answer: a
customer's `lifetime_value` already contains the spending you are trying to
predict, and `is_active = 'N'` already says the customer churned.

- **Drop, or move to an event table, every column that summarizes history or
  holds the current state:** lifetime totals, order counts, averages,
  `last_*` and `first_*` dates, "days since", current status, tier, plan,
  segment, score, balance, churn flags. Nori-Rel builds its own features at
  each past date from the event rows before it (counts, means, extremes), so
  nothing about the future gets in.
- **If the files record changes**, such as plan upgrades or status changes,
  load them as an event table (`plan_changes(customer_id, changed_at, plan)`),
  not as a column.
- **Ask when you cannot tell.** Whether `region` or `plan_tier` ever changes is
  a business fact, so ask the user. If nobody can answer, decide by what the
  column describes. Drop it if it measures activity or names a current state
  (anything in the list above). Keep it if it says what the thing is, such as
  country, category, size or a creation date. Either way, list it in the
  design as an assumption.
- **Event time is when the information existed.** A monthly total is known
  only once the month is over: stamp it with the first instant after the
  period (`2024-02-01 00:00 UTC` for January), not with the month's first or
  last day.
- **Every column of an event row must be known at its time column.** Nori-Rel
  shows the whole row from that moment on. A ticket row timestamped when it
  opened must not carry `closed_at`, `resolution_hours` or the final
  `status`; an order row must not carry `returned = true`. Load each later
  fact as its own event table (`ticket_closures(ticket_id, closed_at)`,
  `returns(order_id, returned_at)`).
- **Never fill gaps with later data** (back-filling, whole-history means), and
  never pre-aggregate events into totals. Load the raw rows.

## Steps

### 1. Profile the files

```bash
pip install "nori-rel[database,local]" openpyxl pyarrow
```

This installs everything the later steps need: pandas, the PostgreSQL driver,
Excel and Parquet readers, and local Nori. For hosted Nori only, drop `local`.

Read every file and every Excel sheet with every cell as text
(`dtype=str`, and `header=None` until you know where the header is), so
pandas does not reshape IDs or guess dates. For a file over a few million
rows, look at a sample first (`nrows=500_000`), then confirm keys and
references on the full file before you trust them.

For each file or sheet, record the row count and, per column, the share of
empty cells, the number of distinct values, a few examples, and what the
values are: IDs, numbers, numbers stored as text, dates, booleans, or free
text. Then work out the relationships between files:

```python
def get_normalized_values(column):
    """Return the column's non-empty values, trimmed and lower-cased, as a set."""
    return set(column.dropna().astype(str).str.strip().str.casefold())


def is_candidate_key(column):
    """Return whether the column is never empty and never repeats a value."""
    return column.notna().all() and column.is_unique


def get_reference_match_shares(column, key_column):
    """Return the shares of a column's values found in a key, exactly and normalized."""
    values = set(column.dropna().astype(str))
    exact = len(values & set(key_column.dropna().astype(str))) / max(len(values), 1)
    normalized = get_normalized_values(column)
    folded = len(normalized & get_normalized_values(key_column))
    return exact, folded / max(len(normalized), 1)


def get_columns_fixed_by(frame, id_column):
    """Return the columns that hold one value per ID: entity columns in a flat file."""
    distinct_per_id = frame.groupby(id_column).nunique(dropna=False).max()
    return [name for name, count in distinct_per_id.items() if count == 1]
```

A column that matches another file's key only after normalizing tells you the
loader must trim and case-fold both sides the same way. List the values that
match nothing; they become orphans in step 4.

Look for each of these, and note what you find:

- a header below title rows, which your loader must skip;
- day/month order of slash dates: a first part over 12 means day first, a
  second part over 12 means month first, and neither means the order is
  ambiguous and you must ask;
- money stored as text, booleans as `Y`/`N`, numbers as text;
- columns holding lists (`SKU-1|SKU-7`), which become a child table;
- columns named by date (`2023-01`, `Jan 2024`), which are a wide layout to
  melt into an event table;
- each text file's delimiter and encoding, which your loader must use (files
  with the same columns can still differ);
- values in a date column that do not parse, which your loader must count;
- several date columns in one file, which usually means a later fact sits on
  an event row (see Leakage);
- exact duplicate rows, and IDs spelled several ways within one column
  (`85123a` and `85123A`);
- ID columns with missing values (see rows with no key, in step 4);
- negative numbers and letter prefixes on mostly numeric IDs (`C536379`),
  which often mark cancellations, returns or credit notes;
- files with the same columns, to stack into one table, their combined row
  count against Nori-Rel's limit, and rows that appear in more than one of
  them (sheets exported with overlapping dates), with those rows' date range;
- values that look like test data (`TEST001`, "This is a test product");
- text columns with 10 or more distinct values, which the default features
  skip (contract item 7);
- ID columns that fix other columns in a flat file, which mark entity and
  header tables to split out (per `invoice`, fixed: `customer_id`); a column
  that is only mostly fixed (one customer, two countries) stays on the rows,
  or takes its first value, which you then report.

### 2. Ask what the files cannot tell you

Ask in one message, and only what you need:

- **What to predict, for whom, over what window.** For example "how much each
  customer spends in the next 30 days". This picks the entity table, the event
  table and the PQL.
- Which date to use when an event has several (ordered, shipped, invoiced).
- The time zone of times without one, if it matters at the prediction's scale.
- Whether specific columns change over time (see Leakage).
- Whether rows that look wrong (negative quantities, test accounts) are real.
- For money targets: net of returns and cancellations, or gross, and whether
  non-product lines (postage, fees, discounts) count.

Wait for the answers. When the user cannot answer something, choose a default
and mark it as an assumption in the design you confirm in step 3.

### 3. Design the schema, and confirm it with the user

Write the full design and show it to the user before anything touches the
database. It has seven parts:

1. **Tables**, one row each: name, kind (entity, event, or lookup), primary
   key, time column (event tables), the files it comes from, and its expected
   row count.
2. **Columns** of each table: name, PostgreSQL type, the source file and
   column, and any conversion (day-first dates, currency symbols stripped,
   `Y`/`N` to boolean, keys trimmed and lower-cased).
3. **Relationships**: every foreign key as `orders.customer_id →
   customers.customer_id`, each with one plain sentence ("each order belongs
   to one customer; a customer can have many orders"). Mark the entity keys
   you copied onto child rows so they link straight to the entity (contract
   item 4).
4. **Left out or moved**: every source column you will drop or move to an
   event table, and why (leakage, an unused type, a duplicate of another
   column).
5. **Row rules**: how you will treat orphans, rows with no key, duplicates and
   overlapping files, with the counts you found in step 1.
6. **Assumptions**: every default you chose in step 2, such as the time zone,
   which date is the event time, or which columns you treated as fixed.
7. **The prediction**: the entity, target and window, written as PQL.

Common reshapes:

| The files have | Load it as |
|---|---|
| One flat export with customer and product columns repeated on every line | `customers` and `products` entity tables (one row per ID), an event table per line, and a header table (`invoices`) when lines share a document number. The line table keeps its own `customer_id` foreign key, not only the link to `invoices`, so you can predict from it (contract item 4). A column whose value varies across rows of one ID belongs on the event, not the entity. |
| The same columns in several files or sheets (one per year or region) | One table, stacked, after cutting any overlapping period from one file. Do not deduplicate the stacked table: that also removes repeats within a file. Keep a `source_file` column only if it carries meaning. |
| Columns named by period (`jan_sales`, `2023-01`) | An event table `(entity_id, period_end, value)`, one row per non-empty cell, where `period_end` is the first instant after the period. |
| A list in a cell (`a;b;c`) | A child table, one row per item, with a foreign key to the parent. |
| A reference by email or name instead of ID | Map it to the entity's ID during load; keep unmatched rows with a null foreign key and count them. |
| A lookup sheet (region codes, categories) | A small entity table, referenced by a foreign key. |
| Several dates on one record (opened, closed) | The earliest date as the record's time column, and one more event table for each later date that matters (`ticket_closures`). |

End the message with a direct question, such as "Should I create these tables
and relationships? Confirm, or tell me what to change." Then stop and wait.

- **Only an explicit yes confirms the design.** A question, a partial answer
  or silence does not. Answer the question, then ask again.
- **After every correction, show the whole revised design** and ask again.
  The user confirms the design as a whole, not a list of edits.
- **If nobody can confirm** (you are running unattended), stop after this
  step and hand over the design as your report. Do not load a schema nobody
  has approved.
- **The same rule holds later.** If loading or checking shows the design must
  change (a key that repeats, a table to add, a link that cannot be declared),
  describe the change, say why, and get a yes before you apply it.

### 4. Write one load script

Implement exactly the confirmed design. Write the conversion as a single
script the user can rerun when new files arrive (`load_<dataset>.py` in their
project), not as interactive steps. It should:

1. Read each file with explicit options: header row, delimiter, encoding,
   `dtype=str` for IDs (Excel turns `00123` into `123` and IDs into floats like
   `123.0`).
2. Clean with explicit rules: parse dates with an explicit `format=` (or
   `dayfirst=` from step 1), convert to UTC; strip currency symbols and
   thousands separators; treat `(12.50)` as negative; map `Y`/`N` to booleans;
   trim and case-normalize join keys the same way on both sides.
3. Count every row it drops or changes, with the reason: unparseable dates,
   rows whose key is missing, duplicates you decided are export errors. Never
   drop rows silently. For exact duplicate rows, look at where they sit:
   copies next to each other, or whole repeated documents, suggest an export
   error; repeats scattered inside a document are usually real (the same
   item entered twice). Keep or drop them, and say which and why.
4. Create the tables in a dedicated schema with explicit DDL, load with
   `COPY`, then add the foreign keys last, so an orphan fails loudly:

```python
import io

import psycopg

DDL = """
DROP SCHEMA IF EXISTS shop CASCADE;
CREATE SCHEMA shop;
CREATE TABLE shop.customers (
    customer_id TEXT PRIMARY KEY,
    created_at  TIMESTAMPTZ NOT NULL,
    region      TEXT
);
CREATE TABLE shop.orders (
    order_id    BIGINT PRIMARY KEY,
    customer_id TEXT,
    ordered_at  TIMESTAMPTZ NOT NULL,
    amount      NUMERIC(12, 2)
);
"""
FOREIGN_KEYS = """
ALTER TABLE shop.orders ADD FOREIGN KEY (customer_id)
    REFERENCES shop.customers (customer_id);
"""


def copy_frame(connection, table, frame):
    """Load a DataFrame into an existing table with COPY."""
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False, header=False, na_rep="\\N")
    buffer.seek(0)
    columns = ", ".join(frame.columns)
    with connection.cursor().copy(
        f"COPY {table} ({columns}) FROM STDIN WITH (FORMAT csv, NULL '\\N')"
    ) as copy:
        copy.write(buffer.read())


with psycopg.connect(DATABASE_URL, autocommit=True) as connection:
    connection.execute(DDL)
    copy_frame(connection, "shop.customers", customers)
    copy_frame(connection, "shop.orders", orders)
    connection.execute(FOREIGN_KEYS)
```

Keep timestamp columns timezone-aware in pandas (`utc=True`). `to_csv` then
writes the offset, and PostgreSQL stores the right instant whatever its
session time zone.

**Orphans** (an event whose customer ID is not in the customer file): look at
the rows before deciding. IDs in the same format and range as real ones,
spread over time with ordinary values, are usually real customers missing
from the export: add each with only its ID and `created_at` = its first event,
and count them. IDs outside the real range, test-like names, or rows that
copy other rows are usually junk: drop them and count them. The rule you
apply is the one the user confirmed in step 3. Do not leave the foreign key
undeclared to make the load pass.

**Rows with no key** (a sale with no customer ID): they cannot count toward
any customer. Keep them, with a null foreign key, only if another prediction
needs them, such as store-level totals; otherwise drop them and count them.
Kept rows still count toward the row limits.

### 5. Check the schema

Read the schema back through Nori-Rel's own discovery, which sees the tables,
keys and time columns exactly as a prediction will:

```python
from nori_rel import NoriRelClient

with NoriRelClient() as client:
    source = client.connect(
        engine="postgresql",
        database_url=DATABASE_URL,
        schema_name="shop",
        time_columns={"orders": "ordered_at"},
    )
    schema = client.discover(source)
for table in schema.tables:
    links = [f"{key.columns} -> {key.referred_table}" for key in table.foreign_keys]
    print(table.name, "key", table.primary_key, "time", table.time_column, links)
```

If `connect` or `discover` raises, fix that first. Then check each item, with
the discovered schema or a SQL query, and fix every failure:

- **Keys and links.** Every table has a one-column primary key. Every foreign
  key has one column. Every table has a foreign key or is referenced by one;
  an unlinked table is ignored. Every event table you will predict from has a
  foreign key straight to the entity.
- **Time columns.** Every event table has one, registered in `time_columns`,
  of type `timestamp with time zone`, never null. Note each event table's
  earliest and latest time; step 6 needs them.
- **Creation time.** No entity is created after its first event:

  ```sql
  SELECT COUNT(*)
  FROM shop.customers AS c
  JOIN (SELECT customer_id, MIN(ordered_at) AS first_event
        FROM shop.orders GROUP BY customer_id) AS e USING (customer_id)
  WHERE c.created_at > e.first_event;
  ```

- **Later facts on event rows.** No other date on an event row falls after its
  time column (`SELECT COUNT(*) FROM shop.tickets WHERE closed_at >
  opened_at`). A nonzero count means the row carries a fact known only later.
- **Leakage by name.** Read every entity column whose name sounds like a total,
  a latest date or a current state, and every event column that sounds
  settled after the event (a final status, an outcome). Keep one only if it
  is fixed when the row starts; renaming it does not help.
- **Types.** List the columns with
  `SELECT table_name, column_name, data_type FROM information_schema.columns
  WHERE table_schema = 'shop'`. JSON, arrays and binary columns are unused.
  Count the distinct values of each text column on the entity and the tables
  one link from it; 10 or more and the default features skip it.
- **Size.** Row counts against the limits in contract item 8.

If a fix changes the confirmed design, ask first (step 3).

### 6. Run one prediction

Run a query that matches the user's goal from the date to predict from. Choose
the anchor from the time ranges in step 5: the end of the last complete day
that every event table covers. `observed_through` is the instant through
which every table is complete; Nori-Rel learns only from past outcome windows
that end by then. If the files were exported at different times, use the
earliest export.

```python
from datetime import UTC, datetime, timedelta

from nori_rel import NoriRelClient

anchor = datetime(2024, 6, 1, tzinfo=UTC)
with NoriRelClient() as client:
    source = client.connect(
        engine="postgresql",
        database_url=DATABASE_URL,
        schema_name="shop",
        time_columns={"orders": "ordered_at"},
    )
    details = client.predict(
        source=source,
        query="PREDICT SUM(orders.amount,0,30,days) FOR EACH customers.customer_id",
        # Learn from 12 past dates, 30 days apart.
        anchor_times=[anchor - timedelta(days=30 * k) for k in range(1, 13)],
        anchor_time=anchor,
        observed_through=anchor,
        entity_created_at="created_at",
        return_details=True,
    )
predictions = details["predictions"]
print(len(predictions), "predictions")
print(details["metadata"]["usable_rows"], "history rows,",
      details["metadata"]["feature_count"], "features")
print(predictions["prediction"].describe())
print(details["pipeline"].feature_names)
```

This runs local Nori with FastDFS features. Each past date needs its whole
outcome window before the anchor, and Nori-Rel needs at least 32 usable
history rows. Binary targets such as `COUNT(orders.*,0,30,days)=0` need
hosted Nori: use `NoriRelClient(mode="hosted")` and set
`SYNTHEFY_NORI_API_KEY`.

**History is capped.** Nori-Rel learns from at most 5,000 history rows by
default (`max_context_rows`), filling them from the most recent past date
backwards and, within a date, from the lowest entity IDs. Each past date
adds about one row per entity created by then. Count those entities date by
date, newest first (`SELECT COUNT(*) FROM shop.customers WHERE created_at <=
'2024-05-02'`), to see which dates fit under the cap. With more than 5,000
entities, the most recent past date alone fills it and the other dates are
unused. Raise the cap with `max_context_rows=12_000` so several past dates
count. For a seasonal business, say whether last year's season is among them.
More rows need more GPU memory: if the run fails with `OutOfMemoryError`,
lower the cap. `CUDACachingAllocator ... OOM` warnings alone are retries and
harmless when the run finishes. Tell the user which cap you used.

Each feature name says which table and column it came from. The default
features (FastDFS) use only tables one link from the predicted entity: the
entity's own columns, the tables it points to (`regions`), and summaries over
the history before each date (count, mean, min, max, standard deviation, most
common value) of the tables that point to it (`orders`). Tables further out
contribute nothing: rows of a child's child (`order_items` reached only
through `orders`), and parents of a child (`products` reached through
`order_items`). If a table or column the user cares about produced no
feature, give its rows a direct key to the entity, copy fixed columns of
farther parents onto them (`order_items.category`), or see contract item 7.
Any of these changes the design, so confirm it first.

Predictions can come back slightly negative; when the target cannot be
negative (a count, gross spend), tell the user to clip them at 0.

A run that returns predictions proves the schema works. It says nothing about
accuracy.

### 7. Report

Give the user:

- the design they confirmed, with the loaded row counts, and every change
  they approved after it;
- every row dropped or changed, with counts and reasons;
- every assumption: time zone, date order, which date is the event time,
  which columns you treated as fixed;
- columns left out for leakage, and why;
- any check from step 5 you left unfixed, and why it is acceptable;
- the load script's path and how to rerun it;
- the Nori-Rel call to use:

```python
from datetime import UTC, datetime, timedelta

from nori_rel import NoriRelClient

anchor = datetime(2024, 6, 1, tzinfo=UTC)  # the date to predict from
with NoriRelClient() as client:
    source = client.connect(
        engine="postgresql",
        database_url=DATABASE_URL,
        schema_name="shop",
        time_columns={"orders": "ordered_at"},
    )
    predictions = client.predict(
        source=source,
        query="PREDICT SUM(orders.amount,0,30,days) FOR EACH customers.customer_id",
        anchor_times=[anchor - timedelta(days=30 * k) for k in range(1, 13)],
        anchor_time=anchor,
        observed_through=anchor,
        entity_created_at="created_at",
        max_context_rows=12_000,  # the cap you used in step 6
    )
# A DataFrame: entity_id, anchor_time, prediction. Spend cannot be negative.
predictions["prediction"] = predictions["prediction"].clip(lower=0)
```

## Where the database lives

- **The user has PostgreSQL:** create a dedicated schema in it. Loading needs
  a role that can create tables; Nori-Rel later needs only read access.
  Remote connections must use TLS (`?sslmode=require`); Nori-Rel refuses
  plain-text connections to anything but localhost.
- **No database, Docker available:**

  ```bash
  docker run -d --name nori-rel-db -e POSTGRES_PASSWORD=nori -p 5432:5432 postgres:16
  # DATABASE_URL=postgresql://postgres:nori@localhost:5432/postgres
  ```

- **No database, no Docker:** `pip install pgserver` runs PostgreSQL from a
  folder, with no install or root access:

  ```python
  import pgserver

  server = pgserver.get_server("nori-rel-db", cleanup_mode=None)
  DATABASE_URL = server.get_uri()  # keeps running after Python exits
  ```

Ask before creating a database or schema in the user's own PostgreSQL; that
is part of the design they confirm.

## Mistakes to avoid

- Creating, altering or loading tables before the user confirms the design,
  or applying a later change they have not seen.
- Declaring foreign keys only in your design, not in the database. Nori-Rel
  reads the catalog; an undeclared link does not exist.
- Keeping a customer file's current-state columns because they look useful.
  They are the most common way to get a model that is perfect in testing and
  useless in production.
- Letting pandas guess date order. `03/04/2024` is March or April depending
  on the parser; use the day/month finding from step 1 and an explicit format.
- Loading IDs as floats. `123.0` and `123` do not join.
- Normalizing a key on one side only. If orders say `cust-0001 ` and the CRM
  says `CUST-0001`, apply the same trim and case rule to both.
- Dropping rows to make a foreign key pass, without counting them.
- Pre-aggregating events (monthly totals per customer) when the raw events
  exist. Nori-Rel needs the events to build features at any date.
- Splitting one dataset across schemas. Nori-Rel reads one schema at a time.
