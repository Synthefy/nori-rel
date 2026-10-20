"""Choose which labelled rows Nori learns from, and report what was chosen.

A prediction's context is the set of historical (entity, cutoff) rows, with
their labels, that Nori reads as examples. Without a policy, nori-rel keeps the
newest candidates up to ``max_context_rows`` and then drops the ones whose
label is unknown, so a rare label can leave almost nothing. A ``ContextPolicy``
selects among labelled rows only, can rebalance a 0/1 label, spreads the rows
over time, caps rows per entity, and maps balanced probabilities back to the
population rate.

``ContextPolicyReport`` is the record of what a prediction actually used: the
policy (or the default), who chose it, the class counts before and after
selection, the rows kept at each cutoff, and, when an agent searched for
features, how the agent split its rows. Predictions return it as
``metadata["context_policy"]``.
"""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
)

TimeSampling = Literal["newest_first", "random", "equal_per_cutoff"]
ContextPolicyChooser = Literal["default", "caller", "pipeline"]

#: Labels with at most this many distinct values report a count per value.
MAX_REPORTED_LABEL_VALUES = 20


class _FrozenModel(BaseModel):
    """Reject unknown fields and keep instances immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ClassBalance(_FrozenModel):
    """Resample a 0/1 label to a fixed mix of positives and negatives.

    ``positives="all"`` keeps every labelled positive from every training
    cutoff; an integer keeps at most that many, spread over cutoffs by the
    policy's ``time_sampling`` and drawn at random within a cutoff. Negatives
    kept are ``floor(positives kept * negatives_per_positive)``, split over
    cutoffs (anchor times, or days for event-time rows) in proportion to the
    positives kept at each and drawn at random within a cutoff. The class
    ratio is then the same at every cutoff, so the date says nothing about
    the class. A cutoff with fewer negatives than its
    share keeps all of them; the shortfall is not taken from other cutoffs,
    and a cutoff with no kept positive contributes no negatives.

    Example:
        ``ClassBalance(positives="all", negatives_per_positive=1)`` keeps every
        signup and the same number of non-signups.
    """

    #: The label value counted as positive: 1 or 0 (True or False for a
    #: comparison target).
    positive_label: Literal[0, 1] = 1
    #: ``"all"`` for every labelled positive, or the most positives to keep.
    positives: Literal["all"] | Annotated[StrictInt, Field(ge=1)] = "all"
    #: Negatives kept for each positive kept, rounded down.
    negatives_per_positive: float = Field(default=1.0, gt=0, allow_inf_nan=False)


class ContextPolicy(_FrozenModel):
    """Which labelled historical rows a prediction gives Nori as examples.

    Passing a policy to ``predict(context=...)`` changes two things. First,
    rows are chosen only among (entity, cutoff) candidates whose label is
    known, so an unknown label never takes a row. Second, the fields below
    choose among those rows, in this order: the per-entity cap, the class
    balance, then ``max_rows``. Leaving ``context`` out keeps the older rule:
    the newest candidates up to ``max_context_rows``, unknown labels dropped
    afterwards.

    ``time_sampling`` decides which rows fill a quota whenever a pool holds
    more rows than the quota. With ``balance`` it applies to the positives
    only; negatives follow the kept positives (see ``ClassBalance``):

    - ``"newest_first"``: the newest cutoff first, then the entity key, the
      same order the older rule uses.
    - ``"random"``: a uniform random subset. Each row's random key depends only
      on ``seed``, its entity and its cutoff, so adding new days does not
      reshuffle the older rows.
    - ``"equal_per_cutoff"``: the same number of rows from each cutoff, chosen
      at random within the cutoff; a cutoff that runs out hands its unused
      share to the others.

    ``candidate_sql`` narrows the candidates with your own SELECT before the
    other fields choose among them, for filters the query's WHERE cannot
    express, such as a join to another table. Most predictions leave it out:
    ``anchor_times`` already picks the cutoffs, WHERE the entities, and the
    labelled-first read every labelled row.

    ``correct_probabilities`` applies only with ``balance``. Balancing moves
    the share of positives Nori sees from the population rate ``pi`` to the
    sample rate ``s``, which inflates a predicted probability ``p``. The
    correction multiplies its odds by ``k = (pi / (1 - pi)) / (s / (1 - s))``:
    ``p' = p * k / (p * k + 1 - p)``. It applies to binary probabilities and to
    the mean output of a 0/1 regression label (clipped to [0, 1] first); a
    median or quantile output cannot be corrected and is refused.
    """

    #: ``None`` keeps the natural class mix; a ``ClassBalance`` resamples a
    #: 0/1 label.
    balance: ClassBalance | None = None
    #: Which rows fill a quota: see the class docstring.
    time_sampling: TimeSampling = "newest_first"
    #: The most rows any one entity keeps, taken in ``time_sampling`` order
    #: with its positives first when ``balance`` is set; ``None`` for no cap.
    max_rows_per_entity: Annotated[StrictInt, Field(ge=1)] | None = None
    #: The most rows kept in total. With ``balance``, positives shrink to
    #: ``floor(max_rows / (1 + negatives_per_positive))`` so the ratio holds.
    #: ``None`` keeps every row the other fields allow, and the prediction
    #: fails if that is more than ``RelationalLimits.max_context_rows``.
    max_rows: Annotated[StrictInt, Field(ge=1)] | None = None
    #: The prediction fails when the kept rows hold fewer positives than this.
    #: Checked only for a 0/1 label; ``None`` turns the check off.
    min_positives: Annotated[StrictInt, Field(ge=0)] | None = 10
    #: Fixes the ``"random"`` and ``"equal_per_cutoff"`` choices.
    seed: StrictInt = 0
    #: With ``balance``, map outputs back to the population's positive rate.
    correct_probabilities: StrictBool = True
    #: Optional. One read-only SELECT, in the source's SQL dialect, over
    #: ``candidates(entity_id, anchor_time)`` (nori-rel's eligible candidates:
    #: the entity existed by the cutoff and the label window is observed)
    #: that returns the pairs to keep. It may join any registered table, and
    #: it can only narrow the candidates: pairs it returns that are not
    #: candidates are ignored. The other fields apply to what it keeps.
    #: ``None`` keeps every candidate.
    candidate_sql: str | None = None

    @field_validator("candidate_sql")
    @classmethod
    def validate_candidate_sql(cls, value: str | None) -> str | None:
        """Return the SQL stripped, requiring one SELECT or WITH statement.

        Raises:
            ValueError: Blank text, a semicolon, or another kind of statement.
        """
        if value is None:
            return None
        stripped = value.strip()
        if not re.match(r"(?is)^(select|with)\b", stripped):
            raise ValueError("candidate_sql must be one SELECT (or WITH ... SELECT)")
        if ";" in stripped:
            raise ValueError("candidate_sql must be one statement, without semicolons")
        return stripped


class CutoffContextCounts(_FrozenModel):
    """Rows and positives at one training cutoff, before and after selection.

    The positive counts are ``None`` when the label is not 0/1.
    """

    anchor_time: AwareDatetime
    #: Candidates at this cutoff whose label is known.
    labelled_rows: int
    labelled_positives: int | None
    #: Rows kept at this cutoff as Nori's context.
    kept_rows: int
    kept_positives: int | None


class AgentSearchFoldReport(_FrozenModel):
    """One validation window of an agent feature search.

    The agent scores each candidate feature program by fitting on the
    ``context`` rows and predicting the ``validation`` rows; both are slices of
    the prediction's own context rows, ordered by cutoff.
    """

    context_rows: int
    context_last_cutoff: AwareDatetime | None
    validation_rows: int
    validation_first_cutoff: AwareDatetime | None
    validation_last_cutoff: AwareDatetime | None
    #: Mean label of the validation rows; for a 0/1 label, the positive rate.
    validation_label_mean: float | None


class AgentSearchSplitReport(_FrozenModel):
    """How an agent feature search split the context rows it scored on."""

    #: ``"cutoff"`` when every slice ends at a change of cutoff, otherwise
    #: ``"row_count"``.
    split_boundary: Literal["cutoff", "row_count"]
    holdout_fraction: float
    confirmation_fraction: float
    requested_folds: int
    folds: tuple[AgentSearchFoldReport, ...]
    #: The newest rows, held back to re-check the winning program.
    confirmation_rows: int
    confirmation_first_cutoff: AwareDatetime | None
    confirmation_last_cutoff: AwareDatetime | None


class ContextPolicyReport(_FrozenModel):
    """What one prediction used as Nori's context, and who decided it.

    Returned as ``metadata["context_policy"]``. ``chosen_by`` is ``"default"``
    when ``predict`` got no policy (the older newest-candidates rule, capped at
    ``max_context_rows``), ``"caller"`` for a policy passed to ``predict``, and
    ``"pipeline"`` for a saved pipeline's policy. Totals are the sums of
    ``cutoffs``; the rows the database returned are
    ``metadata["candidate_rows"]``.
    """

    chosen_by: ContextPolicyChooser
    #: The policy applied; ``None`` for the older newest-candidates rule.
    policy: ContextPolicy | None
    #: The older rule's candidate cap; ``None`` when a policy was applied.
    max_context_rows: int | None
    #: The label value counted as positive, or ``None`` when the label is
    #: not 0/1.
    positive_label: int | None
    #: Share of positives among the labelled rows (pi) and the kept rows (s).
    population_positive_rate: float | None
    sample_positive_rate: float | None
    #: The odds multiplier applied to the outputs, or ``None`` when no
    #: correction ran.
    probability_correction_factor: float | None
    #: Rows per label value, when the label has at most
    #: ``MAX_REPORTED_LABEL_VALUES`` distinct values; keys are the values.
    labelled_label_counts: dict[str, int] | None
    kept_label_counts: dict[str, int] | None
    #: One entry per training cutoff that could be labelled, newest first.
    cutoffs: tuple[CutoffContextCounts, ...]
    #: How an agent feature search split these rows; ``None`` without one.
    agent_search_split: AgentSearchSplitReport | None = None
    #: The caller's ``agent_instructions``, verbatim; ``None`` without them.
    agent_instructions: str | None = None
    #: The agent stages whose requests carried them; empty when no agent ran.
    agent_instruction_stages: tuple[Literal["feature_search"], ...] = ()


__all__ = [
    "MAX_REPORTED_LABEL_VALUES",
    "AgentSearchFoldReport",
    "AgentSearchSplitReport",
    "ClassBalance",
    "ContextPolicy",
    "ContextPolicyChooser",
    "ContextPolicyReport",
    "CutoffContextCounts",
    "TimeSampling",
]
