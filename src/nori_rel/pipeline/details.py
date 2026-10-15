"""What ``predict(return_details=True)`` returns."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import unquote

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, SkipValidation, model_validator

from nori_rel.pipeline.artifact import NoriRelPipeline


class PredictionDetails(BaseModel):
    """A prediction with its metadata and the pipeline that produced it.

    ``details["predictions"]``, ``details["metadata"]`` and
    ``details["pipeline"]`` work as they did when ``return_details`` returned
    a dict.

    ``features`` names each column by what it reads, joined with dots: the
    tables the feature reads through from the entity, then the column, then
    the aggregate, which ends in a window such as ``_60d`` when it covers
    only the last 60 days:

    - ``results.grid.mean``: the mean of ``results.grid`` over the entity's
      ``results`` rows. ``results.count`` counts those rows; ``recency``,
      ``mean_age`` and ``oldest_age`` are their age in days.
    - ``qualifying.constructors.count``: the ``constructors`` rows reached
      through the entity's ``qualifying`` rows.
    - ``drivers.dob``: the entity's own column. ``teams.name``: a column of
      the parent row the entity points to.
    - ``anchor.cutoff_year``: a field of the anchor time predicted from.
    - FastDFS reads the same way: ``results.grid.max``. A nested aggregate
      reads inside out: ``results.pit_stops.duration.max.mean`` is the mean,
      over the entity's results, of each result's longest pit stop.
    - An agent's column is ``<query>.<column>``.

    When two features would share a name, each table they reach through a
    different foreign key carries that key in brackets, as in
    ``matches[home_team_id].count``. A name that still collides stays the
    internal name. ``get_feature_table`` maps every name to the internal one.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    predictions: pd.DataFrame
    metadata: dict[str, Any]
    #: ``None`` when the prediction cannot be exported (a custom feature
    #: backend, or a model that cannot be pinned).
    pipeline: SkipValidation[NoriRelPipeline | None]
    #: The scored rows' feature matrix the model predicted from: one row per
    #: entity_id, one column per feature, under the dotted names above.
    #: ``None`` when not kept.
    features: pd.DataFrame | None = None
    #: The model's own name for each column of ``features``, in order: the
    #: names the pipeline, a saved agent program and
    #: ``metadata["feature_matrix_sha256"]`` use. Filled from the columns of
    #: ``features`` when the details are built from internal names.
    internal_feature_names: tuple[str, ...] | None = None
    #: The SQL behind each FastDFS or NoriDFS feature, by internal name: the
    #: statements FastDFS ran, or a DuckDB query that gives NoriDFS's values.
    feature_sql: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _get_fields_with_dotted_feature_names(cls, fields: Any) -> Any:
        """Return the constructor's fields with ``features`` under dotted names.

        ``features`` arrives with the model's internal names, which move to
        ``internal_feature_names``. Fields that already carry
        ``internal_feature_names`` are returned unchanged once it has one name
        per column.
        """
        if not isinstance(fields, dict):
            return fields
        features = fields.get("features")
        if not isinstance(features, pd.DataFrame):
            return fields
        internal_names = fields.get("internal_feature_names")
        if internal_names is not None:
            if len(internal_names) != features.shape[1]:
                raise ValueError(
                    "internal_feature_names needs one name per column of features"
                )
            return fields
        internal_names = tuple(str(name) for name in features.columns)
        dotted_names = _get_dotted_feature_names(
            internal_names, fields.get("metadata") or {}
        )
        return {
            **fields,
            "features": features.set_axis(dotted_names, axis=1),
            "internal_feature_names": internal_names,
        }

    def __getitem__(self, key: str) -> Any:
        """Return a field by name, for code written against the old dict."""
        if key not in type(self).model_fields:
            raise KeyError(key)
        return getattr(self, key)

    def get_feature_table(self) -> pd.DataFrame:
        """Return one row per column of ``features``, in the same order.

        Columns:
            feature: the column's name in ``features``.
            internal_name: the model's name for the column, which the
                pipeline, a saved agent program and
                ``metadata["feature_matrix_sha256"]`` use.
            source: "agent" for a column an agent's SQL query produced,
                "noridfs", or the prediction's feature strategy otherwise
                (for example "fastdfs").
            sql: the SQL behind the column: the agent's query for an agent
                column, the statements FastDFS ran on its DuckDB copy of the
                snapshot for a FastDFS column, and for a NoriDFS column a
                DuckDB query over the snapshot tables and ``eval_table`` that
                gives the values NoriDFS computed in pandas.
            dtype: the column's data type.
            missing_share: the share of scored entities with no value.
            mean, min, max: over the scored entities, for numeric columns.

        Raises:
            ValueError: the details hold no features.
        """
        if self.features is None:
            raise ValueError(
                "these details hold no features; call predict(return_details=True)"
            )
        agent_query_sql = _get_agent_query_sql(self.metadata)
        strategy = str(self.metadata.get("features", "unknown"))
        internal_names = self.internal_feature_names or tuple(
            str(name) for name in self.features.columns
        )
        rows = []
        for position, internal_name in enumerate(internal_names):
            values = self.features.iloc[:, position]
            query = _get_agent_query_name(internal_name, agent_query_sql)
            if query is not None:
                source, sql = "agent", agent_query_sql[query]
            else:
                source = "noridfs" if internal_name.startswith("nr|") else strategy
                sql = self.feature_sql.get(internal_name)
            numeric = pd.api.types.is_numeric_dtype(
                values
            ) and not pd.api.types.is_bool_dtype(values)
            rows.append(
                {
                    "feature": self.features.columns[position],
                    "internal_name": internal_name,
                    "source": source,
                    "sql": sql,
                    "dtype": str(values.dtype),
                    "missing_share": float(values.isna().mean()),
                    "mean": float(values.mean()) if numeric else None,
                    "min": float(values.min()) if numeric else None,
                    "max": float(values.max()) if numeric else None,
                }
            )
        return pd.DataFrame(
            rows,
            columns=[
                "feature",
                "internal_name",
                "source",
                "sql",
                "dtype",
                "missing_share",
                "mean",
                "min",
                "max",
            ],
        )


#: Units of a NoriDFS window suffix, largest first; a window takes the largest
#: unit that divides it, so 60 days is ``_60d`` and 36 hours ``_36h``.
_WINDOW_UNITS = (
    ("d", 86_400 * 10**9),
    ("h", 3_600 * 10**9),
    ("min", 60 * 10**9),
    ("s", 10**9),
)

#: FastDFS's prefix on a feature name: ``d`` and the Featuretools depth.
_FASTDFS_DEPTH_PREFIX = re.compile(r"d\d+__")

#: One Featuretools primitive applied to its argument, such as ``MEAN(x.y)``.
_FEATURETOOLS_CALL = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\((.*)\)", re.DOTALL)


@dataclass(frozen=True)
class _DottedName:
    """A feature's dotted name in parts, so a collision can add foreign keys.

    ``steps`` are the tables the feature reads through from the entity, each
    with the foreign-key column that joins it; ``tail`` follows them.
    """

    steps: tuple[tuple[str, str], ...] = ()
    tail: tuple[str, ...] = ()

    def get_name(self, steps_with_keys: frozenset[int] = frozenset()) -> str:
        """Return the dotted name; the steps at ``steps_with_keys`` show their key.

        With ``steps_with_keys={0}``, a ``results`` step joined on
        ``driverId`` is written ``results[driverId]``.
        """
        tables = [
            f"{table}[{key}]" if position in steps_with_keys else table
            for position, (table, key) in enumerate(self.steps)
        ]
        return ".".join([*tables, *self.tail])


def _get_dotted_feature_names(
    internal_names: Sequence[str], metadata: Mapping[str, Any]
) -> list[str]:
    """Return a unique dotted name for each internal feature name, in order.

    Features whose names collide show the foreign key of each table step at
    which they differ; any name still shared falls back to the internal name.
    """
    agent_query_sql = _get_agent_query_sql(metadata)
    strategy = str(metadata.get("features", ""))
    name_parts = [
        _get_dotted_name_parts(name, agent_query_sql, strategy)
        for name in internal_names
    ]
    names = [parts.get_name() for parts in name_parts]
    for positions in _get_colliding_positions(names):
        steps_with_keys = _get_step_positions_with_different_keys(
            [name_parts[position] for position in positions]
        )
        for position in positions:
            names[position] = name_parts[position].get_name(steps_with_keys)
    for positions in _get_colliding_positions(names):
        for position in positions:
            names[position] = internal_names[position]
    if len(set(names)) != len(names):
        return list(internal_names)
    return names


def _get_colliding_positions(names: Sequence[str]) -> list[list[int]]:
    """Return the positions of each name that occurs more than once."""
    positions_by_name: dict[str, list[int]] = defaultdict(list)
    for position, name in enumerate(names):
        positions_by_name[name].append(position)
    return [positions for positions in positions_by_name.values() if len(positions) > 1]


def _get_step_positions_with_different_keys(
    parts: Sequence[_DottedName],
) -> frozenset[int]:
    """Return the step positions whose foreign keys differ among ``parts``.

    Names with different numbers of steps show the key of every step.
    """
    step_counts = {len(part.steps) for part in parts}
    if len(step_counts) > 1:
        return frozenset(range(max(step_counts)))
    (step_count,) = step_counts
    return frozenset(
        position
        for position in range(step_count)
        if len({part.steps[position][1] for part in parts}) > 1
    )


def _get_dotted_name_parts(
    internal_name: str, agent_query_sql: Mapping[str, str], strategy: str
) -> _DottedName:
    """Return the dotted name of one feature, in parts.

    A name of a form this module does not know is kept as it is.
    """
    query = _get_agent_query_name(internal_name, agent_query_sql)
    if query is not None:
        return _DottedName(tail=(query, internal_name.removeprefix(f"{query}__")))
    if internal_name.startswith("nr|"):
        parts = _get_noridfs_dotted_name(internal_name)
        if parts is not None:
            return parts
    if strategy.lower() == "fastdfs":
        name = _get_fastdfs_dotted_name(internal_name)
        if name is not None:
            return _DottedName(tail=(name,))
    return _DottedName(tail=(internal_name,))


def _get_agent_query_sql(metadata: Mapping[str, Any]) -> dict[str, str]:
    """Return the SQL of each query of the prediction's agent program, by name."""
    program = metadata.get("agentic_program") or {}
    return {query["name"]: query["sql"] for query in program.get("queries", [])}


def _get_agent_query_name(
    internal_name: str, agent_query_sql: Mapping[str, str]
) -> str | None:
    """Return the agent query that produced a column, or None.

    An agent query's columns are named ``<query>__<column>``.
    """
    for query in agent_query_sql:
        if internal_name.startswith(f"{query}__"):
            return query
    return None


def _get_noridfs_dotted_name(internal_name: str) -> _DottedName | None:
    """Return a NoriDFS column's dotted name in parts, or None for another form.

    Names are ``nr|anchor|<component>``, ``nr|direct|<table>.<column>`` or
    ``nr|path|<path>|<column or "rows">|<operation>|<window>``, each part
    percent-escaped (``FeatureSpec`` in ``feature.dfs.noridfs.contracts``).
    """
    parts = [unquote(part) for part in internal_name.split("|")]
    if parts[1:2] == ["anchor"] and len(parts) == 3:
        return _DottedName(tail=("anchor", parts[2]))
    if parts[1:2] == ["direct"] and len(parts) == 3:
        return _DottedName(tail=(parts[2],))
    if parts[1:2] != ["path"] or len(parts) != 6:
        return None
    _, _, path, column, operation, window = parts
    steps = _get_noridfs_path_steps(path)
    if steps is None:
        return None
    aggregate = operation + _get_window_suffix(window)
    if operation == "lookup":
        tail = (column,)
    elif column == "rows":
        tail = (aggregate,)
    else:
        tail = (column, aggregate)
    return _DottedName(steps=steps, tail=tail)


def _get_noridfs_path_steps(path: str) -> tuple[tuple[str, str], ...] | None:
    """Return the tables a NoriDFS path reads through, each with its join key.

    ``path`` is a ``RelationPath.identity``:

    - descendant ``E<-orders.user_id<-items.order_id``: orders by user_id,
      then items by order_id;
    - parent ``E=>E.team_id->teams.id/teams.org_id->orgs.id``: teams by
      team_id, then orgs by org_id;
    - bridge ``E|D:results.driverId->E.driverId|U:results.raceId->races.raceId``:
      results by driverId, then races by raceId.

    Returns None for a path of another form.
    """
    if "|D:" in path:
        _, _, edges = path.partition("|D:")
        descendant, _, parents = edges.partition("|")
        child_table, child_column, _ = _get_foreign_key_parts(descendant)
        steps = [(child_table, child_column)]
        for edge in parents.split("/"):
            if not edge.startswith("U:"):
                return None
            _, child_column, parent_table = _get_foreign_key_parts(edge[2:])
            steps.append((parent_table, child_column))
    elif "=>" in path:
        _, _, edges = path.partition("=>")
        steps = [
            (parent_table, child_column)
            for _, child_column, parent_table in map(
                _get_foreign_key_parts, edges.split("/")
            )
        ]
    else:
        _, separator, edges = path.partition("<-")
        if not separator:
            return None
        steps = [edge.rpartition(".")[::2] for edge in edges.split("<-")]
    if not all(table and key for table, key in steps):
        return None
    return tuple(steps)


def _get_foreign_key_parts(identity: str) -> tuple[str, str, str]:
    """Return ``child.column->parent.key`` as (child, column, parent).

    Any part is empty when ``identity`` is not of that form.
    """
    child, _, parent = identity.partition("->")
    child_table, _, child_column = child.rpartition(".")
    return child_table, child_column, parent.rpartition(".")[0]


def _get_window_suffix(window: str) -> str:
    """Return a NoriDFS window as a name suffix.

    ``"all"`` gives ``""``; ``"<nanoseconds>ns"`` gives the largest whole
    unit, such as ``"_60d"`` or ``"_12h"``.
    """
    if window == "all":
        return ""
    if window.endswith("ns") and window[:-2].isdigit():
        nanoseconds = int(window[:-2])
        for unit, unit_nanoseconds in _WINDOW_UNITS:
            if nanoseconds % unit_nanoseconds == 0:
                return f"_{nanoseconds // unit_nanoseconds}{unit}"
    return f"_{window}"


def _get_fastdfs_dotted_name(internal_name: str) -> str | None:
    """Return a FastDFS column's dotted name, or None for another form.

    FastDFS names a feature ``d<depth>__<entity>.<Featuretools name>``, such
    as ``d2__drivers.MAX(results.grid)``, which reads ``results.grid.max``.
    The entity is dropped unless the feature is one of its own columns
    (``d1__drivers.nationality`` stays ``drivers.nationality``). A name
    Featuretools builds from a primitive with several arguments or a WHERE
    clause is returned without its depth prefix.
    """
    prefix = _FASTDFS_DEPTH_PREFIX.match(internal_name)
    if prefix is None:
        return None
    expression = internal_name[prefix.end() :]
    segments = _get_top_level_parts(expression, ".")
    if len(segments) > 2 or (
        len(segments) == 2 and _FEATURETOOLS_CALL.fullmatch(segments[1])
    ):
        segments = segments[1:]
    parts = _get_featuretools_name_parts(segments)
    if parts is None or not all(parts):
        return expression
    return ".".join(parts)


def _get_featuretools_name_parts(segments: Sequence[str]) -> list[str] | None:
    """Return a Featuretools name's dotted segments as name parts.

    Each primitive moves after its argument, lowercase:
    ``MEAN(results.MAX(pit_stops.duration))`` gives results, pit_stops,
    duration, max, mean. Returns None for a primitive with several arguments
    or a WHERE clause.
    """
    *tables, last = segments
    call = _FEATURETOOLS_CALL.fullmatch(last)
    if call is None:
        return list(segments)
    primitive, argument = call.groups()
    if len(_get_top_level_parts(argument, ",")) > 1 or " WHERE " in argument:
        return None
    inner = _get_featuretools_name_parts(_get_top_level_parts(argument, "."))
    if inner is None:
        return None
    return [*tables, *inner, primitive.lower()]


def _get_top_level_parts(text: str, separator: str) -> list[str]:
    """Return ``text`` split at ``separator`` outside brackets and backticks.

    Featuretools wraps arguments in parentheses and relationship keys in
    square brackets; nori-rel puts backticks around a restored name that
    contains a dot.
    """
    parts = []
    depth = 0
    quoted = False
    start = 0
    for position, character in enumerate(text):
        if character == "`":
            quoted = not quoted
        elif quoted:
            continue
        elif character in "([":
            depth += 1
        elif character in ")]":
            depth -= 1
        elif character == separator and depth == 0:
            parts.append(text[start:position])
            start = position + 1
    parts.append(text[start:])
    return parts


__all__ = ["PredictionDetails"]
