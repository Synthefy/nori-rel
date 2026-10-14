from datetime import UTC, datetime

from nori_rel.contracts import ColumnSchema, ForeignKeySchema, TableSchema
from nori_rel.pipeline.task import get_pql_task_signature

QUERY = "PREDICT COUNT(orders.*,0,30,days) FOR EACH users.user_id"


def manifest_fields() -> dict:
    """Return the sections of a small, valid NoriDFS pipeline."""
    users = TableSchema(
        name="users",
        columns=[
            ColumnSchema(name="user_id", data_type="BIGINT", nullable=False),
            ColumnSchema(name="created_at", data_type="TIMESTAMP", nullable=True),
        ],
        primary_key=["user_id"],
    )
    orders = TableSchema(
        name="orders",
        columns=[
            ColumnSchema(name="id", data_type="INTEGER", nullable=False),
            ColumnSchema(name="buyer_id", data_type="BIGINT", nullable=True),
            ColumnSchema(name="event_time", data_type="TIMESTAMP", nullable=True),
        ],
        primary_key=["id"],
        foreign_keys=[
            ForeignKeySchema(
                columns=["buyer_id"],
                referred_table="users",
                referred_columns=["user_id"],
            )
        ],
        time_column="event_time",
    )
    path = {
        "entity_table": "users",
        "edges": [
            {
                "child_table": "orders",
                "child_column": "buyer_id",
                "parent_table": "users",
                "parent_column": "user_id",
            }
        ],
        "direction": "descendant",
    }
    return {
        "created_at": datetime(2026, 10, 8, tzinfo=UTC),
        "nori_rel_version": "0.2.0",
        "task": {
            "pql_query": QUERY,
            "signature": get_pql_task_signature(QUERY),
            "entity_created_at": "created_at",
            "task": "regression",
        },
        "source": {
            "engine": "postgresql",
            "schema_name": "public",
            "tables": [users, orders],
            "column_dtypes": {
                "users": {"user_id": "Int64", "created_at": "datetime64[ns, UTC]"},
                "orders": {
                    "id": "Int64",
                    "buyer_id": "Int64",
                    "event_time": "datetime64[ns, UTC]",
                },
            },
        },
        "features": {
            "kind": "noridfs",
            "plan": {
                "schema_digest": "a" * 32,
                "anchors": {
                    "entity_table": "users",
                    "entity_column": "user_id",
                    "cutoff_column": "__nori_rel_cutoff_time",
                    "horizon_ns": 30 * 86_400 * 10**9,
                },
                "config": {
                    "max_depth": 2,
                    "max_paths": 24,
                    "max_features": 256,
                    "max_direct_features": 64,
                    "max_direct_categorical_cardinality": None,
                    "max_categorical_cardinality": 32,
                    "window_multipliers": [1, 4, 16],
                    "include_missing_in_modes": False,
                    "include_missing_fractions": False,
                },
                "paths": [path],
                "features": [
                    {
                        "name": "nr|path|orders|rows|count|all",
                        "operation": "count",
                        "kind": "numeric",
                        "table": "orders",
                        "path": path,
                    }
                ],
                "digest": "b" * 32,
            },
        },
        "columns": {
            "names": ["nr|path|orders|rows|count|all"],
            "dtypes": ["float64"],
            "categorical": [],
        },
        "model": {
            "mode": "local",
            "model": "nori-30m",
            "repo_id": "Synthefy/Nori-30M",
            "revision": "c" * 40,
            "filename": "model.pt",
            "sha256": "d" * 64,
            "synthefy_nori_version": "0.20.1",
            "regressor_params": {"augmentations": ["yj"]},
        },
        "prediction": {"output_type": "median"},
        "context_policy": {
            "anchor_offsets": ["P90D", "P60D", "P30D"],
            "max_context_rows": 5000,
        },
        "provenance": {"features": "noridfs"},
    }
