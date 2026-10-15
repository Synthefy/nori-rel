"""Snapshot-to-matrix interface around the existing NoriDFS planner."""

from __future__ import annotations

import pandas as pd

from .._basedfs import CUTOFF_COLUMN, BaseDFS, FeatureMatrices, RelationalData
from .contracts import AnchorDefinition, DatabaseView, FeaturePlan, NativeFeatureConfig
from .engine import NativeRelationalFeaturizer
from .sql import SQLSchema, get_plan_sql, get_sql_schema


class NoriDFS(BaseDFS):
    """Use the native planner through the common DFS matrix interface.

    Callers that need the fitted plan itself can use
    ``NativeRelationalFeaturizer`` directly.
    """

    def __init__(
        self,
        config: NativeFeatureConfig | None = None,
        *,
        horizon: str | pd.Timedelta | None = None,
    ) -> None:
        """Set the native feature budget and optional aggregation-window unit."""
        self.config = config or NativeFeatureConfig()
        self.horizon = horizon
        #: A fitted plan to run instead of planning again; see ``from_plan``.
        self.saved_plan: FeaturePlan | None = None
        #: The plan from the last ``compute``, for callers that describe or
        #: extend its columns rather than only read the matrix.
        self.last_plan: FeaturePlan | None = None
        #: What the last plan's SQL needs to know about the snapshot.
        self._last_sql_schema: SQLSchema | None = None

    @property
    def last_feature_sql(self) -> dict[str, str]:
        """Return the DuckDB query behind each feature of the last ``compute``.

        Each query reads the snapshot tables and ``eval_table`` (``row_id``,
        the entity key and ``cutoff``, one row per context or query row) and
        gives the values the pandas executor computed; ``noridfs.sql``
        describes the two helper columns some queries read. Empty before the
        first ``compute``.
        """
        if self.last_plan is None or self._last_sql_schema is None:
            return {}
        return get_plan_sql(self.last_plan, self._last_sql_schema)

    @classmethod
    def from_plan(cls, plan: FeaturePlan) -> NoriDFS:
        """Run a saved plan on each snapshot without planning again.

        The snapshot must have the plan's schema, including column dtypes,
        and the rows must use the plan's entity table, key and cutoff column.
        """
        backend = cls(plan.config)
        backend.saved_plan = plan
        return backend

    def categorical_columns(self) -> tuple[str, ...] | None:
        """Return the fitted plan's categories, independent of matrix dtypes."""
        return None if self.last_plan is None else self.last_plan.categorical_columns

    def compute(
        self,
        snapshot: RelationalData,
        train_rows: pd.DataFrame,
        query_rows: pd.DataFrame,
        entity_table: str,
        *,
        row_id_column: str,
    ) -> FeatureMatrices:
        """Plan once and return features in the supplied context/query order."""
        database = DatabaseView.from_parts(
            tables=snapshot.tables,
            primary_keys=snapshot.primary_keys,
            foreign_keys=snapshot.foreign_keys,
            time_columns=snapshot.time_columns,
        )
        if entity_table not in database.primary_keys:
            raise ValueError("NoriDFS requires a single-column entity primary key")
        entity_column = database.primary_keys[entity_table]
        combined = pd.concat([train_rows, query_rows], ignore_index=True)
        if row_id_column not in combined or combined[row_id_column].isna().any():
            raise ValueError("feature rows require non-null row identifiers")
        if combined[row_id_column].duplicated().any():
            raise ValueError("feature rows require unique row identifiers")
        definition = AnchorDefinition.create(
            entity_table=entity_table,
            entity_column=entity_column,
            cutoff_column=CUTOFF_COLUMN,
            horizon=self.horizon,
        )
        if self.saved_plan is None:
            featurizer = NativeRelationalFeaturizer.fit(
                database, definition, self.config
            )
        else:
            saved = self.saved_plan.anchors
            if (saved.entity_table, saved.entity_column, saved.cutoff_column) != (
                definition.entity_table,
                definition.entity_column,
                definition.cutoff_column,
            ):
                raise ValueError("the saved NoriDFS plan reads another entity table")
            featurizer = NativeRelationalFeaturizer(self.saved_plan)
        self.last_plan = featurizer.plan
        anchors = combined[[entity_column, CUTOFF_COLUMN]]
        self._last_sql_schema = get_sql_schema(
            database, featurizer.plan, anchors[entity_column]
        )
        frame = featurizer.transform(database, anchors).frame
        split = len(train_rows)
        return FeatureMatrices(
            train=frame.iloc[:split].reset_index(drop=True),
            query=frame.iloc[split:].reset_index(drop=True),
            feature_names=list(frame.columns),
        )
