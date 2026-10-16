# Dependency security review

The release gate requires **zero unassessed findings**, complete audit results,
and a current applicability review. It does not claim that every installed
package is free of advisories. This review covers Nori-Rel's supported client
and release paths.

## Reviewed findings, 2026-10-14

The locked candidate reports three advisory IDs across three dependencies.
Repeated PyArrow and Setuptools findings remain in the raw scanner output.
Compatible upstream bounds still exclude the reported fixes:

| Locked dependency | Reported fix | Upstream constraint |
| --- | --- | --- |
| PyArrow 20.0.0 | 23.0.1 | FastDFS 1.1 and 1.2 require `pyarrow<21`. |
| OAuthlib 3.3.1 | 4.0.0 | Databricks SQL Connector 4.6.0 requires `oauthlib<4`. |
| Setuptools 80.10.2 | 83.0.0 | `synthefy-nori==0.20.1` requires `setuptools<81`. |

No resolver overrides, runtime changes, or model changes bypass these bounds.
The stable `synthefy` SDK was checked against PyPI and remains 7.2.0 for this
review.

The isolated build backend is pinned to `uv_build==0.12.24` and included in the
audited build group. Artifact installation and publication smoke checks use
constraints exported from the same reviewed lockfile. This keeps their actual
dependency versions aligned with the applicability assessment.

### PyArrow: Python bindings exclude the affected API

[GHSA-rgxp-2hwp-jwgg / CVE-2026-25087](https://github.com/advisories/GHSA-rgxp-2hwp-jwgg)
affects Arrow C++ IPC **file** reading with explicit metadata pre-buffering.
The advisory excludes Python bindings and IPC streams. The reviewed PyArrow
20.0.0 Python API exposes no affected pre-buffering method. Nori-Rel passes
in-memory data to FastDFS; Databricks SQL Connector 4.6.0 uses
`pyarrow.ipc.open_stream`.

The exception covers this finding on the reviewed Python client paths. It does
not cover applications linking Arrow C++ directly and enabling
`RecordBatchFileReader::PreBufferMetadata`.

### OAuthlib: supported authentication uses client flows

[GHSA-xpv3-w29h-x7cv / CVE-2026-49265](https://github.com/advisories/GHSA-xpv3-w29h-x7cv)
affects server-side authorization-code PKCE verification. Nori-Rel accepts only
Databricks PAT or service-principal M2M authentication. Reviewed SQL connector
4.6.0 and SDK 0.151.0 paths select token authentication or the SDK's
`client_credentials` flow. Source inspection and offline provider/header probes
did not invoke the affected server grant implementation. These probes do not
replace live authentication certification.

The [upstream advisory](https://github.com/oauthlib/oauthlib/security/advisories/GHSA-xpv3-w29h-x7cv)
names 3.3.2 as patched; the newer global record names 4.0.0. PyPI's release
catalog and version endpoint contain no 3.3.2 release as of this review, so no
published patched upgrade currently satisfies the connector's `<4` bound.

The separate [revocation-callback advisory](https://github.com/oauthlib/oauthlib/security/advisories/GHSA-hj66-6f7g-4r5v)
also concerns an OAuth server endpoint. It is absent from the current scanner
report and is **not** preemptively exempted. A new reported finding requires
review. The exception does not cover applications hosting an OAuth server in
the same environment.

### Setuptools: release builds use a different backend

[GHSA-h35f-9h28-mq5c / CVE-2026-59890](https://github.com/advisories/GHSA-h35f-9h28-mq5c)
affects Unicode filename matching for `MANIFEST.in` exclusions when producing
source distributions, especially on macOS filesystems. The installed version's
matcher exhibits the reported exclusion failure in a synthetic probe.

Nori-Rel builds with isolated `uv_build` on Ubuntu, has no `MANIFEST.in` or
`setup.py`, and validates package contents and licenses. The pinned model
runtime uses Setuptools compatibility APIs but does not build source
distributions in the inspected prediction paths. Its isolated build
requirements do not carry the runtime's `<81` bound.

This accepts the finding for Nori-Rel release and prediction; it does not fix
Setuptools. Installing the local runtime selects the affected Setuptools range.
Users building other projects in that environment, including on macOS, remain
exposed to the packaging issue. Use a separate build environment with patched
Setuptools for those projects. The newer model runtime retains the same upper
bound and would not resolve this finding.

## Agent-search dependencies

The optional `agentic` extra adds `anthropic` 1.13.0, `openai` 2.54.0, and
`duckdb` 1.5.5, with their locked transitive packages (including `httpx2` 2.13.1
and `jiter` 0.17.0). The audit of the lock that contains them on 2026-10-14
reports the same three advisory IDs as above and nothing new, so no exception
covers these packages. The base package does not import them. `anthropic` and
`openai` install only with the `agentic` extra. `duckdb` also installs with the
`database`, `snowflake` and `databricks` extras, because NoriDFS runs its
features on DuckDB; those extras already installed it through `fastdfs`. The
agent sends data to a model provider you configure; that is a data-handling
decision, described in the [support scope](support.md), and not a dependency
finding. The lock is audited with every extra, so a future advisory on these
packages blocks a release like any other.

## Enforced review scope

[`dependency-exceptions.json`](dependency-exceptions.json) records the owner,
review date, expiry, exact versions, advisory identities, scanner description
fingerprints, and rationale. The review expires on **2026-11-13 (UTC)**. SHA-256
hashes bind the complete package source inventory, dependency lockfile, project
configuration, audit checker, artifact validator, and release workflows. A
relevant edit requires another review.

The checker runs unfiltered `pip-audit --strict` on all retained locked extras
and development groups, evaluating markers on Python 3.11/3.12/3.13 on Linux, macOS,
and Windows. Every applicable pinned requirement must appear in the raw result.
Missing, skipped, duplicate, malformed, or unexpected dependency records,
scanner errors, and inconsistent exit statuses fail the gate. New advisories,
changed descriptions or aliases, changed versions or reviewed files, and expired
reviews also fail. Raw reports, diagnostics, and assessments remain available as
Actions artifacts.

This evaluates the checked-in resolution, not arbitrary downstream resolutions
or unrelated uses of the same environment. Prefer compatible upstream fixes;
remove exceptions after upgrading and repeating affected checks. Runtime changes
require separate compatibility review.

Assessment is required on PRs, on `main`, and before each TestPyPI upload or PyPI
promotion. Passing does not replace live connector/inference certification,
publisher registration, or the final public exposure review.
