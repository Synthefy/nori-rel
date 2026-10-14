# Security

Report a suspected vulnerability through
[GitHub's private vulnerability reporting form](https://github.com/Synthefy/nori-rel/security/advisories/new).
If private reporting is unavailable, use the contact channel on
[Synthefy's official website](https://synthefy.com/) to request a private security
contact. Do not put exploit details, credentials, or customer data in a public
issue while arranging that contact.

Include the affected package version, connector and inference mode, impact,
and a minimal reproduction with synthetic data. Useful reports cover credential
exposure, unintended database writes, temporal leakage, unsafe SQL construction,
and escapes from documented resource or pipeline validation boundaries.

Security fixes target the latest public release and `main`. Private research
code is outside the supported production distribution. A maintainer will
triage the report, coordinate a fix and disclosure, and document affected versions
in an advisory when appropriate. No response-time commitment is implied.

Keep database identities scoped to the intended tables. PostgreSQL enforces
read-only transactions; Snowflake and Databricks additionally require identities
without write privileges. Hosted inference receives generated matrices and
labels, which can contain sensitive derived values. See the connector and
pipeline documentation before sharing snapshots, logs, or exported recipes.
