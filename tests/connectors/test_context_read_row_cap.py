"""The row cap of a connector's context read."""

import pytest

from nori_rel.connectors.base import get_context_read_row_cap
from nori_rel.connectors.errors import SnapshotLimitError
from nori_rel.limits import RelationalLimits


def test_an_uncapped_read_has_no_row_cap_even_with_small_limits():
    limits = RelationalLimits(
        max_context_rows=1, max_rows_per_table=1, max_total_rows=1
    )
    assert get_context_read_row_cap(limits, None) is None


def test_an_integer_cap_is_kept_when_it_fits_every_row_limit():
    assert get_context_read_row_cap(RelationalLimits(), 60_000) == 60_000


@pytest.mark.parametrize("max_context_rows", [0, 60_001, 1.5, True])
def test_an_integer_cap_outside_the_row_limits_fails(max_context_rows):
    with pytest.raises(SnapshotLimitError, match="max_context_rows must be"):
        get_context_read_row_cap(RelationalLimits(), max_context_rows)
