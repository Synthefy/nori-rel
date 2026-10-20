"""Compose context and label SQL into one read-only statement."""

from nori_rel.contracts import SchemaGraph
from nori_rel.pql._generated.PQLGrammarParser import PQLGrammarParser as Grammar
from nori_rel.pql.context_sql import compile_context_sql
from nori_rel.pql.dialects import SQLDialect
from nori_rel.pql.label_sql import compile_label_sql
from nori_rel.pql.sql_expressions import Scalar, namespace_sql

_PREPARED_CONTEXT_CTE_NAME = "__nori_context"


def compile_prepared_context_sql(
    tree: Grammar.ProgContext,
    schema: SchemaGraph,
    *,
    dialect: SQLDialect | None = None,
    **options,
) -> tuple[str, dict[str, Scalar]]:
    """Return one SELECT joining candidate context to labels by context_row_id.

    Inputs: parsed PQL, registered SchemaGraph, and compile_context_sql keyword
    options (anchor_times, anchor_time, observed_through, entity_created_at,
    optional max_context_rows, where None reads every candidate that can have
    a known label). Values stay bound, including overlapping filter parameter
    names in the two compilers.

    Output: (sql, params), returning ordered columns
    (context_row_id, entity_id, anchor_time, label), including NULL labels.
    The executor counts/drops those unknown labels. PostgreSQL materializes the
    context CTE; warehouse optimizers choose their own execution strategy while
    all CTEs read the same pinned source relations. No source tables are made.
    """
    dialect = dialect or SQLDialect()
    context_sql, context_params = namespace_sql(
        *compile_context_sql(tree, schema, dialect=dialect, **options), "context"
    )
    label_sql, label_params = namespace_sql(
        *compile_label_sql(
            tree,
            schema,
            context_schema=None,
            context_table=_PREPARED_CONTEXT_CTE_NAME,
            dialect=dialect,
        ),
        "label",
    )
    # Snowflake folds unquoted names to uppercase. Match the quoted source
    # relation emitted by compile_label_sql while preserving existing PG SQL.
    context_cte_reference = (
        _PREPARED_CONTEXT_CTE_NAME
        if dialect.is_postgresql
        else dialect.identifier(_PREPARED_CONTEXT_CTE_NAME)
    )
    sql = f"""WITH {context_cte_reference} AS {dialect.materialized_cte}({context_sql}),
__nori_labels AS ({label_sql})
SELECT c.context_row_id, c.entity_id, c.anchor_time, l.label
FROM {context_cte_reference} AS c
JOIN __nori_labels AS l USING (context_row_id)
ORDER BY c.context_row_id"""
    return sql, context_params | label_params
