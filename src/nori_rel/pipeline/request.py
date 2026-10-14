"""The request that runs a saved pipeline."""

from __future__ import annotations

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field


class PipelinePredictRequest(BaseModel):
    """What one run of a saved pipeline takes from the caller.

    The pipeline fixes everything else. It follows the saved
    request, which has ``as_of``, and adds ``observed_through`` because the
    two cutoffs stay separate here: ``as_of`` is the prediction time the
    scoring rows are built at, and ``observed_through`` is the time up to
    which the database is complete, which decides which context labels are
    known.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: Prediction time. The context anchors are saved as offsets before it.
    as_of: AwareDatetime
    #: The database holds every event up to this time.
    observed_through: AwareDatetime
    #: A PQL query for other entities; only its FOR clause may differ.
    query: str | None = Field(default=None, min_length=1)
    #: A PQL entity predicate that selects the entities to score.
    query_filter: str | None = Field(default=None, min_length=1)
    #: Reproduce an earlier run: its ``context_rows_sha256``. Without it, the
    #: run uses whatever context the database gives now.
    expected_context_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


__all__ = ["PipelinePredictRequest"]
