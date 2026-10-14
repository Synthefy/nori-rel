"""Nori-Rel database prediction client."""

from typing import TYPE_CHECKING

from nori_rel.client import NoriRelClient, PredictionOperation, PredictionTask
from nori_rel.contracts import (
    AwsSecret,
    ConnectionStatus,
    ConnectorCapabilities,
    CredentialReference,
    Database,
    DatabaseCreateRequest,
    DatabaseEngine,
    DatabaseStatus,
    DatabaseUrl,
    DatabricksConnection,
    EnvironmentCredential,
    ForeignKeySchema,
    RelationalOutputType,
    SchemaGraph,
    SnowflakeConnection,
    TableSchema,
)
from nori_rel.limits import RelationalLimits
from nori_rel.pipeline import PipelineError
from nori_rel.run import (
    PredictionCancelled,
    PredictionFailed,
    PredictionRun,
    RunEvent,
    RunTimeout,
)

if TYPE_CHECKING:
    from nori_rel.pipeline import NoriRelPipeline


def __getattr__(name: str):
    """Load pipeline records only when requested."""
    if name == "NoriRelPipeline":
        from nori_rel.pipeline import NoriRelPipeline

        return NoriRelPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AwsSecret",
    "ConnectionStatus",
    "ConnectorCapabilities",
    "CredentialReference",
    "Database",
    "DatabaseCreateRequest",
    "DatabaseEngine",
    "DatabaseStatus",
    "DatabaseUrl",
    "DatabricksConnection",
    "EnvironmentCredential",
    "ForeignKeySchema",
    "NoriRelClient",
    "NoriRelPipeline",
    "PipelineError",
    "PredictionCancelled",
    "PredictionFailed",
    "PredictionOperation",
    "PredictionRun",
    "PredictionTask",
    "RelationalLimits",
    "RelationalOutputType",
    "RunEvent",
    "RunTimeout",
    "SchemaGraph",
    "SnowflakeConnection",
    "TableSchema",
]
