"""Write predictions and run records back to a schema the caller names.

See ``docs/warehouse-connectors.md`` ("Write predictions back") and
``docs/pipeline-export.md`` ("Run records"). Database drivers load only when
a write connects.
"""

from nori_rel.writeback.rows import RUN_RECORDS_TABLE
from nori_rel.writeback.write import TableRunRecordStore, write_predictions
from nori_rel.writeback.writers import PredictionWriteError

__all__ = [
    "RUN_RECORDS_TABLE",
    "PredictionWriteError",
    "TableRunRecordStore",
    "write_predictions",
]
