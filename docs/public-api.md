# Public API inventory

This inventory defines the public API boundary. Installation extras are
`database` (PostgreSQL), `snowflake`, `databricks`, `local`, and `agentic`. The
package has no command-line entry point. See [support and versioning](support.md).

## Root imports

These names are exported from `nori_rel`:

| Contract | Exports |
| --- | --- |
| Prediction client | `NoriRelClient` |
| Credentials | `AwsSecret`, `EnvironmentCredential`, `DatabaseUrl`, `CredentialReference` |
| Connection configuration | `DatabaseCreateRequest`, `DatabaseEngine`, `SnowflakeConnection`, `DatabricksConnection` |
| Connection/schema results | `Database`, `DatabaseStatus`, `ConnectionStatus`, `ConnectorCapabilities`, `SchemaGraph`, `TableSchema`, `ForeignKeySchema` |
| Prediction settings | `RelationalLimits`, `RelationalOutputType`, `PredictionTask` |
| Feature search | `AgenticSearch`, loaded on first use so the `agentic` extra stays optional |
| Pipelines | `NoriRelPipeline`, `PipelineError` |
| Lifecycle | `PredictionRun`, `RunEvent`, `PredictionCancelled`, `PredictionFailed`, `RunTimeout` |
| Deprecated request typing | `PredictionOperation`, for the aggregate-argument compatibility bridge |

The client supports context management, `connect`, `test_connection`, `discover`,
`predict`, `start_predict`, `get_run`, and `close`. Prediction uses either PQL
with explicit historical/scoring/completeness settings or a saved pipeline with
`as_of` and `observed_through`. Background runs expose `progress`, `status`,
`events`, `result`, `cancel`, `stop`, `steer`, `pause`, and `resume`. `stop`
and `steer` apply to agent searches; see [run control](run-control.md).

## Additional documented modules

- `nori_rel.pipeline`: `PredictionDetails`, pipeline save/load and typed pipeline
  errors/warnings. `PipelineManifest`, `PipelinePredictRequest`, format constants,
  and task-signature helpers describe the versioned serialization contract;
  applications should save/load through `NoriRelPipeline`.
- `nori_rel.feature`: `FeatureBackend`, `FeatureMatrices`, and `RelationalData`
  describe the custom-backend extension contract. FastDFS and NoriDFS are the
  supported built-in implementations; `nori_rel.feature.agentic` holds the agent
  search. Callers sharing custom backends across runs must
  provide their own concurrency safety; custom backends have no pipeline format.
- `nori_rel.connectors`: connector classes, `RelationalConnector`, `Snapshot`,
  and `create_connector` expose the connector extension contract. Import
  `ConnectorError` and `SnapshotLimitError` from `nori_rel.connectors.errors`.

Private modules and names beginning with `_` are implementation details. A
public import alone does not guarantee that every optional operation works
without installing its connector or local runtime dependencies.

## Compatibility and excluded interfaces

Use `query=` PQL for new code. The legacy aggregate `entity`/`target` arguments
remain as a deprecated bridge with `FutureWarning`; direct/NEXT legacy targets
and explicit relationship-path prediction are retired.

RelArena adapters, fine-tuning, benchmark runners, notebooks, and the research
CLI are absent from the public package. FastDFS is the default feature backend.
NoriDFS and the opt-in `AgenticSearch` are also supported, and their plans and
programs are saved in pipelines.
