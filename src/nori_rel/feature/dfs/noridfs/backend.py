"""Snapshot-to-matrix interface around the existing NoriDFS planner."""

from __future__ import annotations

import pandas as pd

from .._basedfs import CUTOFF_COLUMN, BaseDFS, FeatureMatrices, RelationalData
from .contracts import (
    AnchorDefinition,
    DatabaseView,
    FeaturePlan,
    NativeFeatureConfig,
    SiblingPathConfig,
)
from .engine import NativeRelationalFeaturizer


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
        sibling_paths: SiblingPathConfig | None = None,
        include_composition_features: bool = False,
        composition_pairs: tuple[tuple[str, str], ...] = (),
    ) -> None:
        """Set the native feature budget, optional window unit and sibling paths.

        ``sibling_paths`` opts into features from a parent's other rows, such as a
        page view's device's earlier views; see ``SiblingPathConfig``.
        ``include_composition_features`` opts into the composition of the rows a
        path reaches, in their own time order: first values, and how mixed each
        categorical column is (entropy, the most common and last value's shares,
        distinct values), such as a signup's journey's first pharmacy and its
        drug-class mix. ``composition_pairs`` adds the same over pairs of
        columns named ``("table.column", "table.column")``, such as a journey's
        most common drug class and pharmacy company together.
        """
        self.config = config or NativeFeatureConfig()
        self.horizon = horizon
        self.sibling_paths = sibling_paths
        self.include_composition_features = include_composition_features
        self.composition_pairs = tuple(composition_pairs)
        #: A fitted plan to run instead of planning again; see ``from_plan``.
        self.saved_plan: FeaturePlan | None = None
        #: The plan from the last ``compute``, for callers that describe or
        #: extend its columns rather than only read the matrix.
        self.last_plan: FeaturePlan | None = None

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
                database,
                definition,
                self.config,
                sibling_paths=self.sibling_paths,
                include_composition_features=self.include_composition_features,
                composition_pairs=self.composition_pairs,
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
        frame = featurizer.transform(database, anchors).frame
        split = len(train_rows)
        return FeatureMatrices(
            train=frame.iloc[:split].reset_index(drop=True),
            query=frame.iloc[split:].reset_index(drop=True),
            feature_names=list(frame.columns),
        )
