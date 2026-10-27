"""Relational database connectors."""

from typing import TYPE_CHECKING

from nori_rel.connectors.base import RelationalConnector, Snapshot
from nori_rel.contracts import DatabaseCreateRequest, DatabaseEngine
from nori_rel.limits import RelationalLimits

if TYPE_CHECKING:
    from nori_rel.connectors.databricks import DatabricksConnector
    from nori_rel.connectors.mysql import MySQLConnector
    from nori_rel.connectors.postgres import PostgreSQLConnector
    from nori_rel.connectors.snowflake import SnowflakeConnector


def create_connector(
    config: DatabaseCreateRequest, settings: RelationalLimits
) -> RelationalConnector:
    """Select a driver lazily so unrelated warehouse extras remain optional."""
    connector_name = {
        DatabaseEngine.POSTGRESQL: "PostgreSQLConnector",
        DatabaseEngine.SNOWFLAKE: "SnowflakeConnector",
        DatabaseEngine.DATABRICKS: "DatabricksConnector",
        DatabaseEngine.MYSQL: "MySQLConnector",
    }[config.engine]
    return __getattr__(connector_name)(config, settings)


def __getattr__(name: str):
    """Import a concrete connector only when the caller selects it."""
    if name == "PostgreSQLConnector":
        from nori_rel.connectors.postgres import PostgreSQLConnector

        return PostgreSQLConnector
    if name == "SnowflakeConnector":
        from nori_rel.connectors.snowflake import SnowflakeConnector

        return SnowflakeConnector
    if name == "DatabricksConnector":
        from nori_rel.connectors.databricks import DatabricksConnector

        return DatabricksConnector
    if name == "MySQLConnector":
        from nori_rel.connectors.mysql import MySQLConnector

        return MySQLConnector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DatabricksConnector",
    "MySQLConnector",
    "PostgreSQLConnector",
    "RelationalConnector",
    "Snapshot",
    "SnowflakeConnector",
    "create_connector",
]
