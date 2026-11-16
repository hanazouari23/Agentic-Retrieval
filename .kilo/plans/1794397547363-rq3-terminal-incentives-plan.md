# RQ3 — Terminal Incentives Plan

## Scope
Implement only the section **“Do terminal incentives change stopping?”** from `RQs/RQ3/readme.md`. Cost and latency sweeps are out of scope for now.

## Data files
From `RQs/relevant_csvs/`:

| Display label | File |
|---|---|
| Base | `cql_full_tree_traces_50_msmarco_dev.csv` |
| Stop penalty | `cql_full_tree_stop_penalty_traces_50_msmarco_dev.csv` |
| Reduced terminal reward | `cql_full_tree_50_msmarco_traces_low_stop.csv` |
| Terminal reward | `cql_full_tree_50_msmarco_traces_terminal_reward.csv` |

Display order for figure x-axis and table rows: **Base → Stop penalty → Reduced terminal reward → Terminal reward**.

## Task 1: First-action distribution figure

Add a new cell group to `RQs/RQ3/plots.ipynb`.

### Steps
1. Load the four CSVs above.
2. For each variant, take the first row per `query_id` (i.e., `step == 1` or `first()` after sorting by `step`).
3. Classify the first action into four categories:
   - `STOP`
   - `Rerank`
   - `PRF` (`action_name == "PseudoRelevanceFeedback"`)
   - `Reformulate` (`action_name == "QueryReform"`)
4. Compute the proportion of queries in each category per variant (so each variant sums to 1.0 / 100%).
5. Draw a stacked bar chart:
   - **x-axis**: reward configuration / variant label
   - **y-axis**: proportion of queries
   - **segments**: the four action categories
   - Use the same category order and colors as the RQ2 action-distribution pie charts for consistency.

### Output
A figure in `RQs/RQ3/plots.ipynb` showing how the first action changes across reward variants.

## Task 2: Reward-variant comparison table

Append the LaTeX source to `RQs/RQ3/latex_table.tex`. Match the RQ2 table style (`table` environment, `tabularx`, `\small`, `\toprule`/`*rule`).

### Rows
One row per variant: Base, Stop penalty, Reduced terminal reward, Terminal reward.

### Columns and definitions

| Column | Definition |
|---|---|
| Reward variant | Variant label |
| Mean NDCG | Mean of the final `ndcg_after` value per query (last trace row) |
| Mean Recall | Mean of the final `recall_after` value per query (last trace row) |
| Mean cost | Mean of the final `cumulative_cost` per query (last trace row) |
| Mean latency | Mean of the final `cumulative_latency_ms` per query (last trace row) |
| Immediate STOP (%) | Percentage of queries whose first action is `STOP` |
| Later explicit STOP (%) | Percentage of queries whose trace contains a `STOP` action with `step > 1` |
| Horizon reached (%) | Percentage of queries with `timeout == 1` (reached `SIM_MAX_STEPS` without explicit STOP) |
| Mean enhancement count | Mean number of non-`STOP` actions per query |

### Final-row handling for effectiveness / cost / latency means
For each query, take the **last row of its trace** (sorted by `step`, then `groupby("query_id").last()`), regardless of whether the episode ended by explicit `STOP` (`terminal == 1`) or by hitting the horizon (`timeout == 1`). This ensures the means include every evaluated query.

- If a query ended with `terminal == 1`, the last row is the `STOP` row.
- If a query ended with `timeout == 1`, the last row is the final non-`STOP` row at `SIM_MAX_STEPS`.

Stopping-behavior percentages are still computed from the full trace (first action, any later `STOP`, and `timeout` flag).

### Notes / edge cases
- The alpha-1.5 sweep CSV had 48 terminal queries while the others had 50; these reward-variant CSVs may also differ. Do not drop or pad queries — report whatever number of terminal rows each CSV contains, and keep per-query denominators consistent within a variant.
- If a query has no `timeout == 1` rows, it is not counted as horizon-reached.
- If a query has no `STOP` with `step > 1`, it contributes 0 to the later-explicit-STOP percentage.

## Validation
- After implementation, the figure should show four stacked bars summing to 100% each.
- The table values should be reproducible from the CSVs using the definitions above.
- A quick sanity check: the Base variant immediate-STOP rate should match the `2.0` point of the section-1 immediate-STOP panel.

## Out of scope
Cost and latency sweeps (`reward_gamma` / `reward_delta` coefficient matrix) are not part of this plan.
