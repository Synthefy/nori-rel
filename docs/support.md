# Supported scope

Version 0.3.0 defines the initial public database-client contract.

| Area | Scope |
| --- | --- |
| Python | 3.11 and 3.12. |
| Primary CI platform | Linux. CI also installs the wheel with the `database` extra on macOS and Windows and runs import checks only. Other operating systems require their own dependency/runtime validation. |
| PostgreSQL | TLS-enabled PostgreSQL; read-only transactions and bounded extraction. CI server versions are specified in `.github/workflows/ci.yml`. |
| Snowflake | Standard tables with available Time Travel history. |
| Databricks | Unity Catalog Delta tables on a SQL warehouse, with retained versions. |
| Features | FastDFS (default) and NoriDFS. |
| Agent search | Opt-in `AgenticSearch` (`agentic` extra), after 0.3.0. Anthropic models or an OpenAI-compatible endpoint, with local or hosted Nori. It sends the schema, aggregate column profiles, and up to 50 rows of each exploratory query result to the configured provider. Offline tests use fake agents; release certification does not call a live model provider, so the agent's value on your data is not established. |
| Synthefy SDK | `synthefy>=7.2.0,<8`, installed with each connector extra. |
| Local prediction | Temporal aggregate regression; point and quantile output. |
| Hosted prediction | Temporal aggregate regression and binary predicate targets. |
| Local runtime | `synthefy-nori==0.20.1`, pinned in the `local` extra. A newer runtime release is not supported until a Nori-Rel release raises the pin. |
| Models | Local: `nori-6m`, `nori-30m` (default), and `nori-100m`. Hosted: the default `nori-30m`. Release certification exercises `nori-30m` only. |
| Hardware | Release certification runs local inference on CPU (`device="cpu"`) on a Linux CI runner. By default, `device` selects CUDA, then MPS, then CPU; CUDA and MPS are not release-certified. Hosted inference needs no local accelerator or model weights. |
| Pipelines | FastDFS and NoriDFS recipes, and saved agent SQL programs; schema/model checks, relative context anchors, fresh data on replay. |
| Operations | Synchronous/background runs, progress, cooperative cancellation, agent-search stop and steer, pool cleanup. |

Database views, arbitrary SQL queries as sources, MySQL, direct column/NEXT
prediction, RelArena adapters, and fine-tuning are outside this release's support
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

Version 0.3.0 was published to PyPI on 2026-10-14. Its release run passed the
protected live-service certification for PostgreSQL, Snowflake, and
Databricks, with local and hosted inference. Upstream version bounds still
block some dependency advisory fixes; the scoped, expiring assessments are in
the [dependency security review](dependency-security.md). See the
[release gates](../RELEASING.md).
