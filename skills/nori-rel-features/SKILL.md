---
name: nori-rel-features
description: Choose and inspect the relational features Nori-Rel builds - FastDFS (default, one link from the entity), NoriDFS (two links, windowed), or the opt-in agent search that writes SQL features with Claude or an OpenAI-compatible model - read feature_table.csv, and explain one prediction with SHAP. Use when a backtest is weak, someone asks what Nori learned from, wants to add or exclude features, or asks whether the agent search is worth its cost and data sharing.
---

# Features in Nori-Rel

Nori-Rel builds every feature itself at each example's date, from the rows
before it. You choose the search in `nori_rel_task.json` under
`feature_strategy`, per prediction. Read `feature_table.csv` after every run.

## The three searches

| | `"fastdfs"` (default) | `"noridfs"` | agent search |
|---|---|---|---|
| Reads | the entity's own columns, the tables it points to, and summaries (count, mean, min, max, standard deviation, most common value) of the tables that point to it | paths up to two links away, with windows scaled to the label interval (recent and older activity), recency and distinct counts | NoriDFS, plus SQL columns a language model writes and tests |
| Needs | a connector extra | nothing more | `nori-rel[agentic]`, an API key, at least 5 past dates |
| Static targets | yes | no | no |
| Pipeline (replay without a search) | yes | yes | yes, the winning SQL is saved |
| Cost | seconds | seconds to minutes | model tokens plus up to 28 extra Nori calls per prediction |

Text columns with more than nine distinct values and JSON, array and binary
columns produce no feature. Dates other than registered time columns produce
none either. A table reached only through another table (order lines under
orders) is invisible to FastDFS; NoriDFS sees it. Give the rows a direct key
to the entity when a table two links away matters.

Switch by editing the task and rerunning [nori-rel-predict](../nori-rel-predict/SKILL.md)
and [nori-rel-backtest](../nori-rel-backtest/SKILL.md). Decide on earlier
folds, report the last.

## The agent search

The agent proposes SQL feature programs, runs each locally in DuckDB on the
snapshot, scores it with the client's own Nori on held-out past rows, and
keeps a program only if it beats NoriDFS by more than 1% on a selection slice
and again on a confirmation slice. Otherwise the prediction uses NoriDFS and
`metadata.agentic_fallback_reason` says why. Three of the past dates train,
one selects, one confirms, so the task needs at least five.

Before enabling it, tell the user what the provider receives, and get a yes:

- the schema: table and column names, row counts, keys, time columns;
- a profile of each column the agent asks about, including up to ten raw
  values of a text column with their counts, and percentiles of numbers;
- up to 50 rows of each exploratory query result;
- the error text of a failed query, which can quote values.

Do not enable it on data that must not reach that provider. Then set, in the
task:

```json
"feature_strategy": {
  "agent": "claude",
  "model": "claude-sonnet-5-5",
  "max_cost_usd": 3.0,
  "prices": {"input": 3.0, "output": 15.0, "cache_read": 0.3, "cache_write_5m": 3.75, "cache_write_1h": 6.0}
}
```

`prices` are US dollars per million tokens from the provider's current price
list; `max_cost_usd` needs them and caps the agent's tokens per prediction
(not Nori calls, and not the next prediction). The key comes from
`ANTHROPIC_API_KEY`. For an OpenAI-compatible endpoint use
`"agent": "openai"`, `"model"`, optionally `"base_url"`, and `OPENAI_API_KEY`;
it reports no cost, so it has no cap. `"agent_instructions": "..."` in the
task sends your text to the agent verbatim ("signups on 1-3 June had broken
tracking; do not build features from them"); it cannot loosen the leak checks.
`"metric": "wape"` (or `mae`, `rmse`, `mape`, ...) changes which program is
kept, not how Nori predicts.

After the run, `metadata.json` holds `agentic_trials`, `agentic_cost_usd`,
`agentic_stop_reason` and `agentic_fallback_reason`; `pipeline.json` carries
the winning SQL, so scheduled runs replay it without another search.

## Read the features

`feature_table.csv` has one row per feature column, in the order Nori saw
them: `feature` (a dotted path from the entity: `orders.amount.mean`,
`orders.count`, `orders.amount.mean_30d`, `orders.products.count`, the
entity's own `users.country`, `anchor.cutoff_month`), `source` (`fastdfs`,
`noridfs` or the agent's query name), `sql` for FastDFS and agent columns,
`dtype`, `missing_share`, `mean`, `min`, `max` over the scored entities.

Look for: `missing_share` near 1 (a feature that is empty for nearly everyone
and carries nothing), a `max` that could only be known after the anchor (a
leak; find the column and drop it from `tables` through a view), and whether
the tables the user talks about appear at all. A handful of features with the
tables missing means the links are not declared.

## Explain one prediction

Nori is fitted on `details.context_features` with `details.context["label"]`
and predicts `details.features`. SHAP on that fit splits one entity's
prediction into one contribution per feature. It needs the live prediction
objects and `pip install shapiq "synthefy-nori[interpretability]"`:

```python
import pandas as pd
import shapiq
from synthefy_nori import NoriRegressor

details = client.predict(..., return_details=True)   # the same request
columns = details.context_features.columns
output_type = details.metadata.get("output_type") or "median"
output_type = getattr(output_type, "value", output_type)
model = NoriRegressor(model=details.metadata["model"])
model.fit(details.context_features, details.context["label"])


def predict(rows):
    """Return Nori's predictions for feature rows given as an array."""
    frame = pd.DataFrame(rows, columns=columns)
    frame = frame.astype(details.context_features.dtypes.to_dict())
    return model.predict(frame, output_type=output_type)


entity = details.predictions["entity_id"].iloc[0]
row = details.features.loc[[entity], columns].to_numpy()
explainer = shapiq.TabularExplainer(
    model=predict, data=details.context_features.to_numpy(),
    imputer="baseline", index="SV", max_order=1, random_state=0,
)
values = explainer.explain(row, budget=1024)
contributions = {
    columns[key[0]]: value
    for key, value in values.dict_values.items() if len(key) == 1
}
print(f"average prediction {values.baseline_value:.3f}")
for name, value in sorted(contributions.items(), key=lambda item: -abs(item[1]))[:8]:
    print(f"  {name:40s} {value:+.3f}")
```

Check first that the refitted model reproduces the reported prediction for
that entity; the fit depends on the order of the examples, which
`context_features` keeps. Report the top contributions as pushes from the
average, in the target's units, and say they describe this fit, not a causal
effect.
