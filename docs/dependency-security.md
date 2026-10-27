# Dependency security review

The release gate requires **zero unassessed findings**, complete audit results,
and a current applicability review. It does not claim that every installed
package is free of advisories. This review covers Nori-Rel's supported client
and release paths.

## Reviewed findings, 2026-10-27

The locked candidate reports three advisory IDs across two dependencies.
Repeated PyArrow and Setuptools findings remain in the raw scanner output.
Compatible upstream bounds still exclude the reported fixes:

| Locked dependency | Reported fix | Upstream constraint |
| --- | --- | --- |
| PyArrow 20.0.0 | 23.0.1 | FastDFS 1.1 and 1.2 require `pyarrow<21`. |
| Setuptools 80.10.2 | 83.0.0 and 84.1.0 | `synthefy-nori==0.22.0` requires `setuptools<81`. |

Filelock 3.32.4 (CVE-2026-112907, CVE-2026-112913, CVE-2026-112940; fix
4.2.1) was reported on 2026-10-26 and is upgraded rather than excepted: the
Snowflake connector's 4.8.1 release lifted its `filelock<4` bound, so the lock
now holds `snowflake-connector-python` 4.8.1 and `filelock` 4.2.4. No other
locked version changed.

OAuthlib 3.3.1 (GHSA-3jp2-3mrg-2vxc / CVE-2026-49272; fix 4.0.0) was reported
on 2026-10-27 and is upgraded rather than excepted: Databricks SQL Connector
4.8.0 allows `oauthlib<5`, so the lock now holds `databricks-sql-connector`
4.8.0 and `oauthlib` 4.0.0. The upgrade also fixes GHSA-xpv3-w29h-x7cv
(CVE-2026-49265), whose exception is removed. OAuthlib 4.0.0's breaking
changes are in its OAuth server classes; the connector's browser login uses
`WebApplicationClient`, and Nori-Rel accepts only Databricks PAT or M2M
authentication. No other locked version changed.

No resolver overrides, runtime changes, or model changes bypass these bounds.
The stable `synthefy` SDK was checked against PyPI on 2026-10-27 and is 7.4.0,
the locked version.

The isolated build backend is pinned to `uv_build==0.12.24` and included in the
audited build group. Artifact installation and publication smoke checks use
constraints exported from the same reviewed lockfile. This keeps their actual
dependency versions aligned with the applicability assessment.

### PyArrow: Python bindings exclude the affected API

[GHSA-rgxp-2hwp-jwgg / CVE-2026-25087](https://github.com/advisories/GHSA-rgxp-2hwp-jwgg)
affects Arrow C++ IPC **file** reading with explicit metadata pre-buffering.
The advisory excludes Python bindings and IPC streams. The reviewed PyArrow
20.0.0 Python API exposes no affected pre-buffering method. Nori-Rel passes
in-memory data to FastDFS; Databricks SQL Connector 4.8.0 uses
`pyarrow.ipc.open_stream`.

The exception covers this finding on the reviewed Python client paths. It does
not cover applications linking Arrow C++ directly and enabling
`RecordBatchFileReader::PreBufferMetadata`.

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

### Setuptools: the legacy wheel installer is not on any supported path

[GHSA-xmj3-9gf7-h52w](https://github.com/advisories/GHSA-xmj3-9gf7-h52w)
(published 2026-10-23, low severity, fixed in 84.1.0) affects
`Wheel.install_as_egg` on Windows: a wheel's `namespace_packages.txt` can name
a drive-qualified path and make the legacy wheel-to-egg converter create an
`__init__.py` outside the destination egg directory. Only the file's location
is attacker-influenced, not its contents.

Nori-Rel and its locked dependencies install as wheels through uv or pip,
which never call setuptools' wheel-to-egg converter; `easy_install` and
`setup.py install` are not used in the release workflows, the smoke install,
or any prediction path. The same `synthefy-nori` upper bound that keeps the
earlier Setuptools finding in place keeps this one. The exception covers
Nori-Rel release and prediction; users who run the legacy installer on Windows
in the same environment remain exposed and should use a patched Setuptools
there.

## Agent-search dependencies

The optional `agentic` extra adds `anthropic` 1.13.0, `openai` 2.54.0, and
`duckdb` 1.5.5, with their locked transitive packages (including `httpx2` 2.13.1
and `jiter` 0.17.0). The audit of the lock that contains them on 2026-10-27
reports the same three advisory IDs as above and nothing new, so no exception
covers these packages. They install only with the extra and are not imported by
the base package. The agent sends data to a model provider you configure; that is
a data-handling decision, described in the [support scope](support.md), and not
a dependency finding. The lock is audited with every extra, so a future advisory
on these packages blocks a release like any other.

## Enforced review scope

[`dependency-exceptions.json`](dependency-exceptions.json) records the owner,
review date, expiry, exact versions, advisory identities, scanner description
fingerprints, and rationale. The review expires on **2026-11-26 (UTC)**. SHA-256
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
