# Changelog

## Unreleased

- `predict(entity_time="view_time")` makes each entity row one example,
  anchored at its own time, for predictions about events rather than entities
  ("will this page view lead to a signup?"). The label window is measured from
  the row's own time, a row is a context candidate once its label has settled
  (`time + end <= min(anchor_time, observed_through)`), and its features read
  only rows from before that time plus its own columns, so same-day history is
  visible. `anchor_times` must then be omitted. The context report counts the
  rows by UTC day, and `ContextPolicy(time_sampling="equal_per_cutoff")`
  balances days. The agent search, saved feature programs and pipelines reject
  `entity_time` for now, and static targets reject it with a pointer to
  `entity_created_at`. See [event-time rows](docs/pql.md#event-time-rows-one-prediction-per-event).
- NoriDFS builds features from a row's history through a shared parent and
  describes the rows a path reaches as a whole. All of it is opt-in, so
  `feature_strategy="noridfs"` plans the same features as before:
  - `NoriDFS(sibling_paths=SiblingPathConfig(...))` adds sibling paths: up one
    foreign key to a parent, then down to its time-stamped rows strictly before
    each row's cutoff, such as a page view's device's earlier views. They give
    windowed counts, recency, oldest age, distinct values, and match features on
    the row's own values. `include_numeric_aggregates` adds sums, means, minima
    and maxima. A table one step above a parent in `parent_tables` groups by
    that parent's attribute (every drug in a drug's class). `excluded_paths`
    leaves out (parent, child table) pairs, and `max_features` caps the
    family's share of the budget.
  - `NoriDFS(include_composition_features=True)` adds, over the rows a
    descendant or bridge path reaches in their own time order, first values,
    entropy, the most common and the last value's shares, and distinct values.
    `composition_pairs` adds the same over pairs of columns from two bridge
    paths, such as the most common drug class and pharmacy company together.
  - Recency, mean age and oldest age of the rows a path reaches count from each
    row's own time rather than the latest time along the path, so an event
    row's earlier rows (a signup's journey views) have their real ages.
  See [sibling paths](docs/pql.md#history-from-the-same-parent-sibling-paths).

## 0.4.6 — 2026-10-20

- Release certification for 0.4.6 ran the live warehouse suite on Snowflake
  only: the Databricks test workspace rejected the certification credentials.
  The Databricks connector module is unchanged since 0.4.5, which was certified
  on Databricks; the shared warehouse code it uses was certified on Snowflake.
- `predict(context=ContextPolicy(...))` chooses the context among every
  historical (entity, cutoff) row with a known label. Without it, Nori-Rel keeps
  the newest `max_context_rows` candidates and then drops unknown labels, which
  can leave few rows for a rare or delayed label. A policy caps rows per entity
  (`max_rows_per_entity`), balances a 0/1 label
  (`balance=ClassBalance(positives="all", negatives_per_positive=1)`) and caps
  the total (`max_rows`). Each quota fills newest first, at random, or equally
  per cutoff (`time_sampling`), keyed by `seed`. Balanced negatives follow the
  kept positives through time, at random within each class, so the date says
  nothing about the class. Balanced probabilities and 0/1 means are mapped back
  to the population rate unless `correct_probabilities=False`. See
  [context policy](docs/context-policy.md).
- `ContextPolicy(candidate_sql=...)` narrows the candidates with one read-only
  `SELECT` over `candidates(entity_id, anchor_time)`, for filters a PQL `WHERE`
  cannot express, such as a join to another table.
- When a label is unknown unless an aggregate has an event (a ratio's
  denominator, or AVG, MIN, MAX, FIRST and LAST), a policy's candidates start
  from those events instead of every entity at every cutoff.
- `metadata["context_policy"]` is a `ContextPolicyReport` for every
  prediction, with or without a policy: who chose the rule, labelled and kept
  rows and positives per cutoff, label counts, the probability correction, and,
  with an agent search, its validation split and confirmation slice.
- `predict(agent_instructions="...")` sends text verbatim at the end of an
  agent search's request. It needs an `AgenticSearch` as the prediction's
  `feature_strategy` (or the client's search) and replaces
  `AgenticSearch(task_hint=...)`, which now raises a `FutureWarning`.
- Pipelines save the policy in `context_policy.row_policy` and replay it at
  each run's `as_of` with the same seed. Pipelines saved before keep their
  `pipeline_id`.
- A foreign key from a selected table to a table the source did not select,
  such as `signups.device_id` when `devices` is too large to load, no longer
  stops NoriDFS or an agent search with "referred key references missing
  table". Both connectors leave such keys out of the snapshot, FastDFS as
  before, and every prediction lists them in `metadata["dropped_foreign_keys"]`
  (for example `"signups.device_id -> devices.device_id"`); pipelines keep the
  list in their provenance.

- Static PQL targets. `PREDICT users.income FOR EACH users.user_id` predicts a
  column of the entity table, learned from the rows where it is known, for the
  rows where it is NULL. It needs no dates: it takes no `anchor_times`,
  `anchor_time` or `observed_through`, and `entity_created_at` is optional (with
  it, each row's features stop at its own creation time). A Boolean column or a
  comparison is binary, a numeric column is regression, and a 0/1 numeric
  column needs `task=`. The target column never reaches the features.
  `FOR ... IN (...)` scores stored rows too, and they leave the context, so
  predictions can be checked against stored values. Static targets use FastDFS
  or a custom `FeatureBackend` and make no saved pipeline. See
  [static targets](docs/pql.md#static-targets).
- `expected_context_sha256` reproduces a run without a pipeline too: pass a
  prediction's `metadata["context_rows_sha256"]` to `predict`, and a run on
  other context rows raises `PipelineContextError`. Without
  `entity_created_at`, a static target's hash leaves out the read time, so an
  unchanged source gives the same hash on every run.
- A static target no longer raises "PQL inference currently requires a temporal
  aggregate target". A temporal query without `anchor_times`,
  `observed_through` or `entity_created_at` now raises "temporal PQL requires
  anchor_times, observed_through and entity_created_at; a static target such as
  PREDICT users.income takes none of them" instead of "PQL requires
  anchor_times, observed_through and entity_created_at".

## 0.4.5 — 2026-10-18

- A wrong `entity_created_at` now says why it is wrong (no such column on the
  entity table, an ambiguous match, or a column that is not a date or
  timestamp) and lists the entity table's date/timestamp columns, or says it
  has none and one must be added. Context, scoring-filter and scoring-row
  preparation share one check and one message.
- [PQL and time semantics](docs/pql.md) explains `entity_created_at` (a
  column on the entity table holding when each entity first existed, and how
  to derive it from the first event), rates written as
  `COUNT(...)/COUNT(...)` with zero denominators dropped, the `(L - s, L, days)`
  window that keeps one training row per entity for an outcome that settles a
  fixed delay after creation, and the v0 limits.

## 0.4.4 — 2026-10-16

- `details.context` holds the labeled rows Nori learned from (`entity_id`,
  `anchor_time`, `label`), and `details.context_features` their feature
  matrix, the one Nori was fitted on, under the dotted names of
  `details.features`. Labels are the floats the model read, so a database's
  decimal labels arrive as numbers.
- `details.get_context_and_scored_table()` returns every row Nori read, one
  per entity and anchor time: the context rows with their `label` and the
  scored rows with their `prediction`, each with its features. An entity's
  rows read as a timeline that ends in its prediction.

## 0.4.3 — 2026-10-15

- `AgenticSearch` keeps a program that beats the baseline by any amount. A
  trial becomes the best when it scores higher than the current best, and the
  best is kept when it scores higher than NoriDFS alone on the confirmation
  rows, which the search never scored it on. The 1% minimum gain and the
  margin of twice the scorer's seed-to-seed noise are gone; previously a 1%
  gain on rel-f1 fell back to NoriDFS.
- The fallback when the best program loses on the confirmation rows says so:
  "The best program did not beat the baseline on the confirmation rows". It
  used to say "No program beat the baseline", even when one had on the
  search's holdout.
- `details.features` names its columns in dots instead of the internal
  encoding: `results.grid.mean` instead of
  `nr|path|drivers%3C-results.driverId|grid|mean|all`,
  `qualifying.constructors.count` for a two-step path, `results.grid.max`
  instead of `d2__drivers.MAX(results.grid)`, and `<query>.<column>` for an
  agent's column. This changes the column names from 0.4.x. The model,
  pipelines, saved programs and `feature_matrix_sha256` keep the internal
  names, now in `details.internal_feature_names`.
- `get_feature_table()` returns `feature` (the dotted name), `internal_name`,
  `source`, `sql`, `dtype`, `missing_share`, `mean`, `min` and `max`. The
  `definition` column is gone; `sql` holds the agent's query or the statements
  FastDFS ran for each feature, and `None` for NoriDFS, which computes in
  pandas.

## 0.4.2 — 2026-10-15

- Support Python 3.13. `requires-python` is now `>=3.11,<3.14`; every extra
  installs on 3.13 and the test suite passes on it. The dependency audit,
  dependency compatibility check and release matrix run on 3.13 too.

## 0.4.1 — 2026-10-15

- `predict(..., feature_strategy=...)` and `start_predict(...,
  feature_strategy=...)` choose the feature search for one prediction:
  `"fastdfs"` (the default), `"noridfs"`, a `FeatureBackend`, or an
  `AgenticSearch`. The value is checked before the run starts. Passing it with
  `pipeline=` or `feature_program=` raises `ValueError`, since a pipeline fixes
  its features and a saved program always replays on NoriDFS.
- `NoriRelClient(features=...)` is deprecated in favor of
  `predict(..., feature_strategy=...)`. It still sets the search for
  predictions that pass none, with a `DeprecationWarning`, and will be removed
  in 0.5.0.
- `predict(return_details=True)` keeps the scored rows' feature matrix in
  `PredictionDetails.features` (one row per `entity_id`), and
  `PredictionDetails.get_feature_table()` describes each feature column: its
  source (an agent query, NoriDFS, or the run's feature strategy), its
  definition (the agent query that produced it; for a NoriDFS feature on a
  one-step path, the aggregation, column, window and table in words, such as
  "mean of QTY_CANS over the last 30 days, in SHIPMENTS matched on
  CUSTOMER_ID"; two-step and bridge paths show their path text), and its
  missing share, mean, min and max over the scored entities.

## 0.4.0 — 2026-10-14

- PQL label windows accept `months`: `SUM(orders.amount,0,1,months)` covers one
  UTC calendar month from the anchor (a whole month from an anchor on the 1st),
  rendered as calendar-month arithmetic on PostgreSQL, Snowflake, and
  Databricks. Day windows are unchanged.
- Warn when `anchor_times` give no training rows because their label window ends
  after the earlier of `anchor_time` and `observed_through`, naming the dropped
  anchors. Previously they were dropped silently, which with a 30-day window on
  monthly data removed every February anchor.
- Add the NoriDFS relationship feature search (`features="noridfs"`) beside
  FastDFS. It needs only the core dependencies.
- Add `AgenticSearch` (`nori-rel[agentic]`). A language-model agent writes SQL
  feature columns on top of NoriDFS. Each candidate runs locally in DuckDB and
  is kept only if it beats NoriDFS alone on held-out rows. It is opt-in. It
  sends the model provider you configure the schema, column profiles (which
  include up to 10 raw values of each profiled categorical column), up to 50
  rows of each exploratory query result, and failed queries' error messages.
  The README lists exactly what.
- Save NoriDFS plans, and NoriDFS plans with an agent's SQL program, in
  pipelines. Replay does not plan again, search again, or call a model provider.
- Add `stop` and `steer` run commands, trial, spend, and fallback events, and
  `max_runtime_seconds` and `max_spend_usd` limits for agent searches.
- Add the `agentic` extra: `anthropic`, `duckdb`, and `openai`.
- Add `predict(..., metric=...)`: an agent search on a regression target scores
  its candidate programs with `"mae"` (the default), `"mse"`, `"rmse"`,
  `"nmae"`, `"nrmse"`, `"medae"`, `"wape"`, `"mape"` or `"smape"`
  (`AgentSearchMetric`). Events report relative and range-normalized errors as
  percentages. An unknown metric, or one passed without an agent search, for a
  binary target, or with a saved program or pipeline, raises `ValueError`.

`features` still defaults to `"fastdfs"`. One behavior changes: in 0.3.0,
`RunLimits.max_runtime_seconds`, passed to `nori_rel.run.LocalExecutor`,
cancelled any run at its next stage boundary. Run limits now bound only the
agent search, which they end as `stop` does; a FastDFS or NoriDFS run ignores
them.

## 0.3.0 — 2026-10-14

Initial public database-client release.

- Focus the distribution on PQL, FastDFS, PostgreSQL, Snowflake, Databricks,
  local/hosted inference, pipeline export/reload, and prediction lifecycle.
- Exclude research adapters, experimental feature search, benchmarks, notebooks,
  their extras, and the research CLI from the supported distribution.
- Add an independent `local` extra, public license/attribution, synthetic
  examples, contribution/security guidance, and a documented support contract.
- Validate saved model settings when reusing a pipeline predictor.
- Add isolated package/connector checks, release certification, and protected
  Trusted Publishing workflows using the tested distribution artifacts.

Migration: use `query=` PQL and `features="fastdfs"`. Research feature names,
agent programs, and research pipeline variants are outside the supported package.
Install `nori-rel[<connector>,local]` for local inference (`database` is the
PostgreSQL extra). The old research `nori-rel` command is no longer installed.

Dependency findings and their expiring, version-specific
assessments are documented in [the security review](docs/dependency-security.md).
