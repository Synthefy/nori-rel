# Nori-Rel agent guide

Read [README.md](README.md) and [CONTRIBUTING.md](CONTRIBUTING.md) before editing.
`main` supports the database/PQL client, FastDFS, connectors, pipeline replay,
and generic prediction lifecycle. Research and earlier repository history remain
private in `Synthefy/nori-rel-internal`. This release repository starts from a
reviewed source snapshot with fresh history and stays private until release
gates pass.

Promote only reviewed patches or selected files through a focused client PR.
Do not merge internal branches or import their full history. Keep research code,
dependencies, artifacts, and internal collaboration records out of this
repository. Changes to this boundary require an explicit scope decision.

## Latest Synthefy SDK

Always use the **latest published stable `synthefy` SDK** for new development,
demos, tests and evaluations, with either local or hosted inference. This rule
applies to the database client, demos, tests, and research evaluations.

- Resolve the current release from PyPI when starting new SDK-related work or
  preparing an environment. Do not assume an installed version, cached wheel,
  old lockfile or historical minimum requirement is current.
- Upgrade the relevant dependency constraints and lockfile with the project's
  package manager, install that resolution, and verify the `synthefy` version
  in the interpreter that will actually run the code. Resolve compatibility
  issues instead of silently falling back to an older SDK.
- Read the installed version's API before implementing compatibility workarounds
  or concluding that a capability is unavailable. A stale SDK is not evidence
  that local inference, memory reporting or another feature is unsupported.
- Record the resolved SDK version in experiment provenance and freeze it across
  every arm of that comparison. If an SDK upgrade is needed after an experiment
  starts, treat it as a new versioned comparison and rerun affected controls;
  never mix versions silently or rewrite historical results. An explicitly
  requested reproduction of an older frozen run retains its recorded version.
- `synthefy` is the SDK; `synthefy-nori` is the separate local model runtime.
  Updating the SDK does not authorize changing pinned model weights, runtime
  versions, inference settings or evaluation protocols.

Do not hard-code today's release number into this policy: resolve the latest
published stable SDK for each new task.

## Commands

```bash
uv sync --group dev --extra database --extra snowflake --extra databricks
uv run --no-sync ruff format --check .
uv run --no-sync ruff check .
uv run --no-sync pytest -m "not warehouse and not live_nori"
uv build --no-sources
```

Set `NORI_REL_TEST_DATABASE_URL` to a disposable TLS-enabled PostgreSQL database
for real database tests. See `RELEASING.md` for required live certification;
skipped integration checks are not passing certification.

## Contracts

- Default to local Nori inference; never silently fall back to hosted mode.
  Hosted mode sends only generated feature matrices and labels to Nori.
- Use public `synthefy` and `synthefy-nori` APIs. Keep the pinned runtime and
  model identity; an SDK upgrade does not authorize changing model weights.
- Keep connector extras independently installable and local weights lazy.
  Base imports must not require database drivers, Torch, or research modules.
- Preserve temporal target/feature isolation, entity IDs, null semantics,
  row ordering, snapshot consistency, and all resource limits.
- Use read-only PostgreSQL transactions. Warehouse read-only enforcement
  requires reader identities; report the distinction in capabilities.
- Never log or serialize resolved credentials. Test sanitization with synthetic
  secrets, including exception chains, without reading real credential files.
- Preserve exact vendored license/provenance files and parser checksums.
  Retain regression tests for supported behavior; remove research tests only
  together with code already preserved in the private internal repository.
- Use feature branches and reviewed PRs. Follow `RELEASING.md` for publishing;
  only this release repository may publish the supported package.
