# When a prediction fails or looks wrong

| Message or symptom | Cause and fix |
|---|---|
| `only N usable context rows; at least 32 required` | Too few examples. More past dates, an earlier `anchor_time`, a larger `max_context_rows`, or a target whose label is known for more entities (`COUNT` and `SUM` are always known; `AVG`, `MIN`, `MAX`, ratios and `FIRST`/`LAST` are unknown without events). |
| The client warns that it dropped anchors | Their window ends after `min(anchor_time, observed_through)`. Expected for the newest ones; if all are dropped, `observed_through` is earlier than the dates assume. |
| `NO_DIRECT_RELATIONSHIP` or a message that the event table does not link to the entity | The target's event table has no declared single-column foreign key to the `FOR EACH` table. Declare one, or copy the entity key onto the child rows ([prepare-data](../../prepare-data/SKILL.md)). |
| `requires mode='hosted'` for a binary target | Comparisons and Boolean columns need hosted Nori. Switch mode and set `SYNTHEFY_NORI_API_KEY`, or predict the count instead of the yes/no. |
| PQL parse error | Check the window `(table.column,start,end,days)`, the unit, `FOR EACH` with `table.key`, backticks around odd names, and that `WHERE` comes before `FOR`. |
| `context exceeds the model feature-element budget` | Lower `max_context_rows` or `RelationalLimits.max_features`. |
| `SnapshotLimitError` | A table is over the row or byte limit. Narrow `tables` or raise `RelationalLimits`. |
| `OutOfMemoryError` on a GPU | Lower `max_context_rows`, or set `"device": "cpu"` in the task. |
| Every prediction is almost the same number | Few features carry information: the entity has little history before the anchors, or the tables that matter are not linked straight to it. Look at `feature_table.csv` and [features](../../features/SKILL.md). |
| Predictions look too good in a backtest | Leakage. An entity column that changes over time (a lifetime total, a status) or an event row carrying a later fact. Remove the column from `tables` by connecting a view without it, or fix the schema. |
| A date with far fewer examples than the others | Fewer entities existed then, or the data for that period is incomplete. Drop it from the past dates or extend `observed_through` only if the user confirms the data is complete. |
| `ambiguous: pass task=` for a 0/1 column | Set `"task"` in the request: `regression` with mean output gives the probability of 1 locally. |
| Replay refuses a saved program or pipeline | The schema, query, creation column or model changed. Run a new search or save a new pipeline; see [schedule](../../schedule/SKILL.md). |
