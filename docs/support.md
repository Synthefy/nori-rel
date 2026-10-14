# Supported scope

The 0.3.0 candidate defines the initial public database-client contract.

| Area | Scope |
| --- | --- |
| Python | 3.11 and 3.12. |
| Primary CI platform | Linux. Other operating systems require their own dependency/runtime validation. |
| PostgreSQL | TLS-enabled PostgreSQL; read-only transactions and bounded extraction. CI server versions are specified in `.github/workflows/ci.yml`. |
| Snowflake | Standard tables with available Time Travel history. |
| Databricks | Unity Catalog Delta tables on a SQL warehouse, with retained versions. |
| Features | FastDFS. |
| Local prediction | Temporal aggregate regression; point and quantile output. |
| Hosted prediction | Temporal aggregate regression and binary predicate targets. |
| Pipelines | FastDFS recipes, schema/model checks, relative context anchors, fresh data on replay. |
| Operations | Synchronous/background runs, progress, cooperative cancellation, pool cleanup. |

Database views, arbitrary SQL queries as sources, MySQL, direct column/NEXT
prediction, and the research feature engines are outside this release's support
scope. Nori Thinking is not a local model option. Connector support does not imply
that every vendor server/runtime version has been tested; release certification
records the actual server and service versions exercised.

A passing deterministic test suite establishes behavior against synthetic
fixtures. It does not establish model quality for a customer's dataset. Live
warehouse and real inference checks are separate release gates in
[RELEASING.md](../RELEASING.md); skipped or unconfigured checks are not certification.

## Compatibility policy

During the 0.x series, a minor release may change a documented API or pipeline
format. Such changes must be called out in the changelog with migration guidance.
Patch releases are intended for compatible fixes. Public imports and documented
arguments define the supported API; names beginning with `_` are implementation
details.

Where practical, removing a supported argument should first produce a warning
and replacement guidance for at least one minor release. Security or correctness
fixes may require rejecting a previously accepted request immediately. Saved
pipeline schemas are explicitly versioned and refuse unsupported formats rather
than silently changing a prediction's meaning.

## Release status

Version 0.3.0 has not been published. This repository stays private while
[upstream dependency blockers](dependency-security.md), live-service
certification, and publisher setup remain unresolved. See the
[release gates](../RELEASING.md).
