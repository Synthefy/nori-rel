"""Relational feature generation at entity-specific temporal cutoffs."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from nori_rel._compat import install_pkg_resources_compatibility
from nori_rel._names import build_aliases

from ._basedfs import CUTOFF_COLUMN, BaseDFS, FeatureMatrices, RelationalData

install_pkg_resources_compatibility()

from fastdfs import DFSConfig, compute_dfs_features, create_rdb  # noqa: E402

_CUTOFF = CUTOFF_COLUMN
#: Column types that depend on the data; keys have a fixed type.
_DATA_TYPES = frozenset({"float", "category", "datetime", "text", "timestamp"})


def _get_column_types(rdb, originals: Mapping[str, str]) -> dict[str, dict[str, str]]:
    """Return the type of each non-key column, by database table and column name."""
    return {
        originals.get(table.name, table.name): {
            originals.get(column.name, column.name): _get_type_name(column.dtype)
            for column in table.columns
            if _get_type_name(column.dtype) in _DATA_TYPES
        }
        for table in rdb.metadata.tables
    }


def _get_type_name(dtype) -> str:
    """Return a FastDFS column type as its string, whether it is an enum or not."""
    return str(getattr(dtype, "value", dtype))


class FastDFS(BaseDFS):
    """Build cutoff-aware feature matrices with the upstream FastDFS package."""

    def __init__(
        self,
        *,
        max_depth: int = 2,
        max_features: int = -1,
        type_hints: Mapping[str, Mapping[str, str]] | None = None,
    ) -> None:
        """Keep the existing database-client search defaults.

        FastDFS reads a text column with fewer than ten distinct values as a
        category and any other as text, and the type decides which features it
        builds. ``type_hints`` maps database table and column names to the
        types ``last_column_types`` reported earlier and replaces that
        inference, so a later snapshot builds the same features.
        """
        self.max_depth = max_depth
        self.max_features = max_features
        self.type_hints = type_hints
        #: Types FastDFS gave the columns in the latest ``compute``, by
        #: database table and column name. Key columns are not listed.
        self.last_column_types: dict[str, dict[str, str]] | None = None

    def compute(
        self,
        snapshot: RelationalData,
        train_rows: pd.DataFrame,
        query_rows: pd.DataFrame,
        entity_table: str,
        *,
        row_id_column: str,
    ) -> FeatureMatrices:
        """Build aligned features for explicit context and scoring rows.

        train_rows/query_rows contain the entity PK, __nori_rel_cutoff_time and
        a unique row_id_column (e.g. context:1/query:1). Labels are never inputs.
        Returns FeatureMatrices(train, query, feature_names): identical feature
        columns, rows in input order, with PK/cutoff/row ID excluded. Table and
        column names FastDFS cannot carry are aliased here and restored in
        feature_names, so callers only ever see original database names.
        """
        primary_keys = {
            table: columns[0]
            for table, columns in snapshot.primary_keys.items()
            if len(columns) == 1
        }
        if entity_table not in primary_keys:
            raise RuntimeError(
                "relational feature search requires a single-column entity primary key"
            )
        entity_column = primary_keys[entity_table]
        usable_foreign_keys = [
            relationship
            for relationship in snapshot.foreign_keys
            if relationship[0] in snapshot.tables
            and relationship[2] in snapshot.tables
            and relationship[2] in primary_keys
        ]
        aliases = build_aliases(snapshot)
        alias = aliases.apply
        # Rename rather than rebuild: nullable integer keys must keep their
        # dtype, and therefore their exact identity above 2**53.
        tables = {
            alias(name): frame.rename(
                columns={column: alias(str(column)) for column in frame.columns}
            )
            for name, frame in snapshot.tables.items()
        }
        rdb = create_rdb(
            tables=tables,
            name="nori-rel-snapshot",
            primary_keys={
                alias(table): alias(column) for table, column in primary_keys.items()
            },
            foreign_keys=[
                (alias(child), alias(key), alias(parent), alias(referred))
                for child, key, parent, referred in usable_foreign_keys
            ],
            time_columns={
                alias(table): alias(column)
                for table, column in snapshot.time_columns.items()
            },
            type_hints=None
            if self.type_hints is None
            else {
                alias(table): {alias(column): kind for column, kind in hints.items()}
                for table, hints in self.type_hints.items()
            },
        )
        self.last_column_types = _get_column_types(rdb, aliases.to_original)
        combined = pd.concat([train_rows, query_rows], ignore_index=True).rename(
            columns={entity_column: alias(entity_column)}
        )
        features = compute_dfs_features(
            rdb=rdb,
            target_dataframe=combined,
            key_mappings={
                alias(entity_column): f"{alias(entity_table)}.{alias(entity_column)}"
            },
            cutoff_time_column=_CUTOFF,
            config=DFSConfig(
                max_depth=self.max_depth,
                use_cutoff_time=True,
                include_cutoff_time=True,
                max_features=self.max_features,
                engine="dfs2sql",
                n_jobs=1,
            ),
        )
        if row_id_column not in features or features[row_id_column].duplicated().any():
            raise RuntimeError("feature rows lost their unique row identifiers")
        expected = combined[row_id_column]
        if set(features[row_id_column]) != set(expected):
            raise RuntimeError("feature row identifiers do not match requested rows")
        features = features.set_index(row_id_column).loc[expected].reset_index()
        drop = [
            column
            for column in (alias(entity_column), _CUTOFF, row_id_column)
            if column in features
        ]
        features = features.drop(columns=drop)
        features = features.loc[:, ~features.columns.duplicated()].replace(
            [float("inf"), float("-inf")], pd.NA
        )
        features.columns = [aliases.restore(str(column)) for column in features.columns]
        if features.shape[1] == 0:
            raise RuntimeError("relational feature search generated no model features")
        split = len(train_rows)
        train = features.iloc[:split].reset_index(drop=True)
        query = features.iloc[split:].reset_index(drop=True)
        return FeatureMatrices(
            train=train,
            query=query,
            feature_names=[str(column) for column in features.columns],
        )
