"""Which aggregates must have an event for a target's label to be known."""

import pytest

from nori_rel.pql import parse_pql
from nori_rel.pql.label_events import get_label_event_aggregates
from nori_rel.pql.tree_access import target_aggregates

A = "COUNT(orders.* WHERE orders.channel='web',0,30,days)"
B = "COUNT(orders.*,0,30,days)"
C = "COUNT(orders.* WHERE orders.channel='store',0,30,days)"
S = "SUM(orders.amount,0,30,days)"
V = "AVG(orders.amount,0,30,days)"


@pytest.mark.parametrize(
    ("target", "required_positions"),
    [
        # COUNT and SUM are 0 over an empty window: always known.
        (B, []),
        (S, []),
        (f"{B}=0", []),
        # AVG, MIN, MAX, FIRST, LAST are NULL over an empty window.
        (V, [0]),
        ("MIN(orders.amount,0,30,days)", [0]),
        (f"{V}>3", [0]),
        # A ratio is unknown when its denominator is 0, whatever it counts.
        (f"{A}/{B}", [1]),
        (f"{S}/{B}", [1]),
        (f"{A}/{B}>0.5", [1]),
        (f"{V}/{B}", [0, 1]),
        # A sum is nonzero when either side is: neither event is required.
        (f"{A}/({B}+{C})", []),
        (f"{A}/({B}-{C})", []),
        # A product is nonzero only when both sides are.
        (f"{A}/({B}*{C})", [1, 2]),
        # Sum and difference of known values are known.
        (f"{A}+{B}", []),
        (f"-{V}+{B}", [0]),
        # A constant denominator needs nothing.
        (f"{A}/2", []),
        # The numerator of a ratio is not required for a known label.
        (f"({A}/{B})*100", [1]),
    ],
)
def test_required_label_events(target, required_positions):
    tree = parse_pql(f"PREDICT {target} FOR EACH users.user_id")
    aggregates = target_aggregates(tree)
    assert get_label_event_aggregates(tree) == [
        aggregates[position] for position in required_positions
    ]
