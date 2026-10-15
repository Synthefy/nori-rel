# Changelog

## Unreleased

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
