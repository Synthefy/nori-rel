# Changelog

## Unreleased

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
