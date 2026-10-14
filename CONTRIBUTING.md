# Contributing

Use a feature branch and open a pull request against `main`. Describe the user
behavior being changed, the reason, and validation performed. Include a small
reproduction for bugs, using synthetic data and redacted configuration.

`main` contains the supported PQL client, FastDFS, database connectors, pipeline
replay, and prediction lifecycle.

Maintainers keep research code, tests, and earlier history in the private
`Synthefy/nori-rel-internal` repository. Promote reviewed patches or selected
files, with relevant public regression tests, through focused PRs. Do not merge
internal branches or import their full history or collaboration records.

## Development setup

Install Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/), then:

```bash
uv sync --group dev --extra database --extra snowflake --extra databricks
uv run --no-sync ruff format --check .
uv run --no-sync ruff check .
uv run --no-sync pytest -m "not warehouse and not live_nori"
uv build --no-sources
```

See the [test-to-contract inventory](docs/testing.md) for what the retained
suite covers, research-test preservation, and commands matching release CI.

PostgreSQL tests need `NORI_REL_TEST_DATABASE_URL` pointing at a disposable,
TLS-enabled database where the test user can create schemas. Test fixtures
create and remove owned schemas. An unconfigured PostgreSQL run skips those
checks; it does not establish database correctness. See
[warehouse integration](docs/warehouse-connectors.md) for the separate live
service tests and [RELEASING.md](RELEASING.md) for mandatory release gates.

Install `--extra local --group cpu` for local inference development on CPU.
Resolve the latest stable `synthefy` SDK for new SDK work, update the constraint
and lockfile, and verify the version in the interpreter running the code. Read
the installed version's API before writing a compatibility workaround; an old
SDK is not evidence that a capability is missing. `synthefy` is the SDK, and
`synthefy-nori` is the separate local runtime that `pyproject.toml` pins to one
version. An SDK upgrade does not change the pinned runtime, model weights, or
inference settings.

## Review expectations

Tests should protect a user-visible behavior or failure mode. Keep temporal
leakage, key precision, credential redaction, SQL read-only enforcement,
resource ceilings, cancellation, and pipeline compatibility covered. Consolidate
duplicate cases; do not remove tests to meet a deletion or coverage target.
Research implementations and their tests remain together in the private
internal repository.

Local inference is the default; never fall back to hosted inference unless the
caller asks for it. Keep database drivers and the local model runtime optional.
Check the relevant extra in a clean installed package when modifying dependency
boundaries. Preserve vendored PQL notices and regenerate the parser/checksums
together when changing the grammar. Document public API changes and migration
guidance in `CHANGELOG.md`.

Do not commit credentials, `.env` files, raw customer data, model weights, cached
features, or generated build artifacts. Report security issues privately through
[SECURITY.md](SECURITY.md). Maintainers review compatibility and approve merges;
release maintainers additionally verify the release evidence and protected
publishing environment.
