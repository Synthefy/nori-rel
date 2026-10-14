# Releasing Nori-Rel

Only reviewed commits on `main` in `Synthefy/nori-rel` can produce supported
releases. The initial public version is `0.3.0`. Complete the release gates below
before its first publication and repository visibility change. Research, earlier
history, and collaboration records remain in the private
`Synthefy/nori-rel-internal` repository.

The initial candidate has scoped, expiring
[dependency security assessments](docs/dependency-security.md). The dependency
gate requires complete audit results and zero unassessed findings before either
TestPyPI or PyPI upload; raw findings remain in the audit artifacts.
Each release also requires live-service certification and verified publisher setup.

## One-time repository and index setup

Maintainers must verify these settings in GitHub and PyPI; committing workflow
files does not activate repository protections or index ownership.

1. Protect `main`: require reviewed pull requests and the stable
   `release-readiness` check, require resolved discussions, and prevent force
   pushes/deletion. Require the branch to be current before merge. Restrict
   `v*` release-tag creation to release maintainers.
2. Set default Actions permissions to read. Keep untrusted PR code away from
   credentials, deployment environments, and publishing permissions. Enable
   secret scanning, push protection where available, and private vulnerability
   reporting before changing visibility.
3. Configure protected `warehouse` and `pypi` environments, plus `testpypi` when
   using a TestPyPI rehearsal. Set reviewers and deployment rules allowing only
   the `main` branch.
   Environment branch rules do not restrict workflow identity; the index's
   Trusted Publisher binds the exact publishing workflow in the next step.
   Scope live-service credentials to their test steps and environments.
4. Verify ownership or create a pending publisher for the `nori-rel` project on
   PyPI and, when rehearsing there, TestPyPI. Configure a GitHub Trusted Publisher
   with owner `Synthefy`, repository `nori-rel`, workflow `publish.yml`, and
   environment `pypi` on PyPI or `testpypi` on TestPyPI. These are independent
   registrations.
5. Use OIDC from the publisher job. Do not copy an account password, `.env.pypi`,
   or a personal API token into the repository or workflow. Account/2FA access
   is handled by the project owner outside the workflow.
6. Keep publishing disabled during private migration. Enable the
   `Publish` workflow only after publisher registration and release gates have
   been verified. Keep the internal repository's publisher disabled.

See [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/) and
[adding a publisher](https://docs.pypi.org/trusted-publishers/adding-a-publisher/).

## Before the first public release

Start this release repository from a reviewed source snapshot with fresh Git
history. Verify that the private internal repository retains the original
research, history, pull requests, and issues. Keep migration and preservation
evidence internal. Promote only reviewed patches or selected files; never merge
internal branches or import their full history into this repository.

Review the entire surface of the new repository that will become public:
source, all branches and history, PR/issue bodies and attachments, release assets,
wiki pages, and Actions logs/artifacts/caches. Record the scope and redacted
findings of secret/license scans. Resolve findings before visibility changes;
rotate exposed credentials before relying on deletion. A clean current checkout
alone does not certify repository history or collaboration records.

Preserve the root MIT license and the unchanged upstream PQL Apache-2.0
license/notice. Verify rights to every included dataset or artifact separately
from the source-code license.

## Prepare a release candidate

1. Merge the version/changelog and reviewed changes through a PR. Replace the
   candidate's `Unreleased` heading with its release date when ready. Ensure
   `pyproject.toml`, the changelog, and `vX.Y.Z` agree. Tag the exact approved
   `main` commit; do not move or reuse a published tag/version.
2. Require `release-readiness` to pass for that exact commit. Check Python
   3.11/3.12, each connector extra independently, real PostgreSQL tests, clean
   wheel/sdist installation outside the checkout, public import/typing checks,
   license/package-content checks, lock consistency, and secret scanning.
3. Run protected live warehouse and real inference certification for the exact
   candidate. Missing credentials, model weights, or services must fail the
   certification; a skipped test is not a pass. Record the actual database,
   SDK, runtime, and model versions alongside JUnit/test reports.
4. Record the source commit, tag, package version, workflow run IDs, dependency
   resolution, test results, and SHA-256 hashes of the wheel and sdist. Confirm
   that research modules, extras, credentials, notebooks, and results are absent
   from both artifacts.

Warehouse configuration and reader/writer separation are documented in
[the connector guide](docs/warehouse-connectors.md). Each live test creates and
removes only its own synthetic resources. The release certification uses the
`warehouse-tests.yml` workflow through the protected `warehouse` environment.
It must test Snowflake, Databricks, parity, and real inference. Long-query
cancellation and history-retirement claims need controlled service evidence;
do not infer them from mocks or unrelated successful tests.

For an offline local-model check, install the local runtime, populate the pinned
model cache, and run:

```bash
uv sync --group dev --group cpu --extra database --extra local
uv run --no-sync pytest \
  tests/pql/test_local_postgres.py::test_offline_postgres_to_cached_local_nori \
  --require-local-model
```

This command also needs `NORI_REL_TEST_DATABASE_URL` for a disposable TLS-enabled
PostgreSQL database. Hosted certification needs a real authorized Synthefy API
key and a small synthetic prediction; mocked transport does not certify it.
Do not print credentials in release evidence.

## Publish

Use `publish.yml` from `main` with the existing `vX.Y.Z` tag. Select one of these
paths explicitly:

| Path | `target` | `direct_publish` | `candidate_run` |
| --- | --- | --- | --- |
| TestPyPI rehearsal | `testpypi` | `false` (default) | Leave empty |
| Promote rehearsed files | `pypi` | `false` (default) | Successful TestPyPI run ID |
| Direct PyPI release | `pypi` | `true` | Leave empty |

For the initial release, use direct PyPI publication without a TestPyPI
rehearsal. This path needs the PyPI Trusted Publisher and
protected `pypi` environment; no TestPyPI registration is needed. The direct
option cannot be combined with a TestPyPI target or an existing candidate run.
Leaving it disabled preserves the requirement to supply a successful TestPyPI
run before production promotion.

Both fresh-candidate paths require the exact protected-main CI result and
dependency assessment, then build the wheel and sdist once. The workflow checks
package contents and licenses, installs both files outside the checkout using
the audited dependency resolution, and records their SHA-256 hashes and the
originating repository, workflow, run ID, and intended index. Mandatory live
warehouse, local-inference, and hosted-inference certification then uses that
exact candidate wheel. Direct publication skips only the TestPyPI upload and
rehearsal; every other release gate still applies.

When promoting from TestPyPI, the workflow downloads the exact files from
`candidate_run` and requires their manifest to identify that TestPyPI run at the
same source commit and package version. It checks their hashes and repeats the
candidate installation and live certification before PyPI upload. It does not
rebuild the files. After a rehearsal, verify the downloaded TestPyPI files and
install the candidate with dependencies from PyPI; avoid a mixed index as an
unrestricted dependency resolver.

Approve the protected `pypi` environment only after its candidate evidence is
complete. The publishing job downloads and verifies the certified files, then
uses OIDC to upload them. It never checks out or builds project code;
`id-token: write` belongs only to that job. Keep manifests, attestations, audit
results, and certification reports with the release evidence.

After publishing, independently download/install the exact production version
from PyPI in a fresh environment. Verify metadata and distribution digests
against the manifest, and run the package smoke checks outside the checkout.
Check the release tag, installed version, source commit, artifact hashes,
attestations, and release notes agree. Publication is complete only after this
verification passes. Repository visibility is a separate step and requires the
public-exposure review above.

## A bad release

Stop new publishing while investigating. Preserve the original artifacts and
run evidence. Revoke a compromised credential or Trusted Publisher immediately
if needed, and use the private security process for sensitive details.

Yank a broken PyPI version when appropriate and explain the affected behavior,
versions, and recovery path. PyPI filenames/versions are immutable: fix forward
with a new version, repeat the gates, and advise users to upgrade or temporarily
pin the last working version. Do not overwrite a tag or pretend that deleting a
release entry recalls packages already installed. Record the incident and the
regression check that prevents recurrence.
