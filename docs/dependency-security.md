# Dependency release blockers

The first-public-release audit on 2026-10-14 found upstream dependency bounds
that prevent installing patched versions. These are release blockers. Passing
client tests does not resolve a dependency advisory, and the publishing workflow
must pass the separate dependency audit before uploading any files.

| Dependency | Patched version | Blocking upstream requirement |
| --- | --- | --- |
| PyArrow | 23.0.1 | FastDFS 1.1 and the latest 1.2 require `pyarrow<21`. |
| OAuthlib | 4.0.0 | Databricks SQL Connector 4.6.0 requires `oauthlib<4`. |
| Setuptools | 83.0.0 | The pinned local runtime `synthefy-nori==0.20.1` requires `setuptools<81`. |

The applicable advisories are
[PyArrow IPC pre-buffering](https://github.com/advisories/GHSA-rgxp-2hwp-jwgg),
[OAuthlib PKCE comparison](https://github.com/advisories/GHSA-xpv3-w29h-x7cv),
[OAuthlib revocation callbacks](https://github.com/advisories/GHSA-hj66-6f7g-4r5v),
and [Setuptools source archive exclusions](https://github.com/advisories/GHSA-h35f-9h28-mq5c).
The dependency findings do not by themselves establish that a particular
Nori-Rel call reaches each vulnerable path.

The candidate raises compatible security floors for urllib3 to 2.8.0 and fsspec
to 2026.6.0. Research-only dependencies were removed from the supported package.
There are no advisory suppressions or overrides of upstream version bounds.

Before publication, obtain upstream releases that accept patched dependencies,
update the relevant constraints and lockfile, and repeat the installed-artifact,
client, and live certification checks. Any change to the pinned local runtime
also requires explicit runtime/checkpoint compatibility review. Keep this record
current as blockers are resolved and retain the final audit report with the
release evidence.
