# Context policy: which labelled rows Nori learns from

Nori learns a prediction from its context: historical (entity, cutoff) rows
with their labels. `predict(context=ContextPolicy(...))` chooses those rows.
`metadata["context_policy"]` reports what every prediction used, with or
without a policy.

## Why the default rule fails rare or delayed labels

Without a policy, nori-rel crosses every entity that exists by each cutoff
with every cutoff, keeps the newest `max_context_rows` of those candidates,
and then drops the ones whose label is unknown. Nothing refills the dropped
rows. A label is unknown when its window holds nothing to compute it from, for
example 0 / 0 in a ratio target.

A trial-to-paid task shows the size of the problem. Each signup's outcome
settles once, so crossing every signup with 29 daily cutoffs gives millions of
candidates, and only the few percent whose outcome settles in that cutoff's
window have a label. When the newest cutoff alone holds more candidates than a
60,000-row cap, the cap fills from that one day and keeps a few hundred
labelled rows. With a policy, the same read returns the labelled rows of all
29 cutoffs and nothing else.

## What an unknown label is

A candidate is one entity at one cutoff, and its label is computed over that
cutoff's window. An unknown label means the window holds nothing the label
can be computed from. It is not a missing outcome.

In the trial-to-paid task, the label is paid outcomes divided by settled
outcomes in the window 6 to 7 days after the cutoff. Each signup has one
outcome, which settles 7 days after the signup, so it falls in that window
for exactly one cutoff: the midnight after the signup day. A signup at
18 Jul 14:05 settles at 25 Jul 14:05, so the 19 Jul cutoff labels it, 1 or 0.
At the 20 Jul cutoff its window is empty and the label is 0 / 0. Counting that
as "not paid" would be wrong: the signup may have paid. So every settled
signup gets one label, and pairing it with any other cutoff gives an unknown
one. The division is what drops those pairs. Counting paid outcomes alone
would label every older signup 0 at every cutoff.

A count over an empty window is a real 0, so COUNT targets have no unknown
labels.

## The rule with a policy

```python
from nori_rel import ClassBalance, ContextPolicy

details = client.predict(
    source=source,
    query=QUERY,
    anchor_times=anchor_times,
    anchor_time=run_end,
    observed_through=run_end,
    entity_created_at="created_at",
    context=ContextPolicy(
        balance=ClassBalance(positives="all", negatives_per_positive=1),
        time_sampling="equal_per_cutoff",
        seed=42,
    ),
    output_type="mean",
    return_details=True,
)
details["metadata"]["context_policy"]  # what was kept, per cutoff
```

1. **Labelled rows only.** The database returns every candidate whose label is
   known, so an unknown label never takes a row. When the label is unknown
   unless one aggregate has an event in its window, candidates start from that
   aggregate's events instead of every entity at every cutoff. That covers a
   ratio's denominator, and AVG, MIN, MAX, FIRST and LAST. For the task above
   that is only the signups whose outcome settles in each cutoff's window.
   COUNT and SUM labels are always known, so those targets keep the
   entity-by-cutoff candidates, without a cap. The read has no row cap of its own: memory
   (`max_snapshot_bytes`) and the statement timeout bound it, and it fails
   instead of truncating.
2. **Per-entity cap** (`max_rows_per_entity`): an entity that appears at every
   cutoff keeps at most this many rows.
3. **Class balance** (`balance`): for a 0/1 label, keep `positives` ("all" or
   a count) and `floor(positives kept * negatives_per_positive)` negatives.
   The negatives are split over cutoffs in proportion to the positives kept
   at each, and every class is drawn at random within a cutoff, so the class
   ratio is the same at every cutoff and the date carries no information
   about the class. A cutoff with too few negatives keeps all it
   has; one with no kept positive contributes no negatives.
4. **Row cap** (`max_rows`): at most this many rows. With a balance the
   positives shrink so the ratio holds. Without `max_rows`, a policy that keeps
   more than `RelationalLimits.max_context_rows` (60,000) rows fails.
5. **Minimum positives** (`min_positives`, default 10): the prediction fails
   when the kept rows hold fewer positives. Checked only for a 0/1 label.

Rows are chosen in Python after one read, so the choice is the same on every
SQL engine. `time_sampling` decides which rows fill each quota:

| `time_sampling` | Rows kept first |
|---|---|
| `"newest_first"` (default) | Newest cutoff, then entity key: the default rule's order |
| `"random"` | A uniform random subset, keyed by `seed` |
| `"equal_per_cutoff"` | One row from each cutoff in turn; a cutoff that runs out hands its share to the others |

A row's random key is a hash of the seed, its entity and its anchor time. It
does not depend on the other rows, so a daily run that adds a new cutoff keeps
the same choice among the older cutoffs. Keys are computed in a fraction of a
second per million rows.

## Optional: narrow the candidates with SQL

Most predictions need no SQL: `anchor_times` picks the cutoffs, the query's
`WHERE` picks the entities, and the read finds every labelled row. For a filter
`WHERE` cannot express, such as a join to another table, `candidate_sql` is one
SELECT over `candidates(entity_id, anchor_time)` that returns the pairs to
keep:

```python
ContextPolicy(
    candidate_sql=(
        "SELECT c.entity_id, c.anchor_time FROM candidates c "
        "JOIN core.signups s ON s.signup_id = c.entity_id "
        "WHERE s.signup_time::date NOT BETWEEN '2026-06-01' AND '2026-06-03'"
    ),
    balance=ClassBalance(positives="all"),
)
```

- It can only narrow the candidates. Pairs it returns that are not candidates
  are ignored, and a pair it returns twice is kept once, so the time rules
  still hold.
- It runs inside the same read-only snapshot, in the source's SQL dialect, as
  one statement: a `SELECT` or `WITH ... SELECT`, without semicolons.
- The other fields choose among what it keeps. The per-cutoff report counts
  the rows that passed it, and pipelines save it and replay it.

## Probability correction

Balancing moves the share of positives Nori sees from the population rate
`pi` (all labelled rows) to the sample rate `s` (kept rows). A model trained
on half positives predicts probabilities near 0.5 for traffic that converts at
1%. With `correct_probabilities=True` (the default) and a balance, each
output's odds are multiplied by `k = (pi / (1 - pi)) / (s / (1 - s))`:

```text
p' = p * k / (p * k + 1 - p)
```

An output equal to the sample rate becomes the population rate. The correction
applies to binary probabilities, where the decision threshold then applies to
the corrected value. It also applies to the mean output of a 0/1 regression
label, clipped to [0, 1] first. A median or quantile output of a balanced
label cannot be corrected, and the request is refused before any data is read.
Ranking metrics such as ROC-AUC and average precision do not change.

## The report

`metadata["context_policy"]` is a `ContextPolicyReport`:

| Field | Meaning |
|---|---|
| `chosen_by` | `default` (no policy), `caller` (passed to `predict`) or `pipeline` (a saved pipeline) |
| `policy` | The `ContextPolicy` applied, or `null` for the default rule |
| `max_context_rows` | The default rule's cap, or `null` with a policy |
| `positive_label` | The label value counted as positive, for a 0/1 label; `null` otherwise |
| `population_positive_rate`, `sample_positive_rate` | `pi` and `s` above |
| `probability_correction_factor` | `k`, or `null` when no correction ran |
| `labelled_label_counts`, `kept_label_counts` | Rows per label value, for labels with at most 20 distinct values |
| `cutoffs` | Per training cutoff, newest first: labelled and kept rows and positives. Totals are their sums |
| `agent_search_split` | With an agent feature search: its folds (context rows, validation rows, their cutoffs, the validation mean label) and its confirmation slice |
| `agent_instructions`, `agent_instruction_stages` | The `predict(agent_instructions=...)` text, verbatim, and the agent stages whose requests carried it (`feature_search` today) |

The agent's split is reported because its validation rows are slices of
these context rows. With a balanced policy, its holdout is balanced too, so
its scores are not rates on real traffic. A fixed validation set drawn at the
natural rate is planned.

## Pipelines

A pipeline saves the policy in `context_policy.row_policy` and replays it at
each run's `as_of`, with the same seed. Pipelines saved before this field
existed keep their `pipeline_id`. `predict(pipeline=..., context=...)` is
refused: the pipeline fixes the policy.
