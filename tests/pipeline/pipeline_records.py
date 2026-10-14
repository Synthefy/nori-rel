from datetime import UTC, datetime

from nori_rel.contracts import ColumnSchema, ForeignKeySchema, TableSchema
from nori_rel.pipeline.task import get_pql_task_signature

QUERY = "PREDICT COUNT(orders.*,0,30,days) FOR EACH users.user_id"


def manifest_fields() -> dict:
    """Return the sections of a small, valid FastDFS pipeline."""
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
            "kind": "fastdfs",
            "max_depth": 2,
            "max_features": 48,
            "fastdfs_version": "1.1",
            "column_types": {"orders": {"event_time": "datetime"}},
        },
        "columns": {
            "names": ["COUNT(orders)"],
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
        "provenance": {"features": "fastdfs"},
    }
