# Changelog

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
