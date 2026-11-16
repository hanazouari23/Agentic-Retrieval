# RQ3 Plots and Tables Plan

## Goal
Complete the figures and tables for RQ3 as specified in `RQs/RQ3/readme.md`.

## Current state
- **Section 1 — NDCG weighting sweep:** already implemented in `RQs/RQ3/plots.ipynb` (six-panel figure, weights 0.5 / 1.0 / 1.5 / 2.0).
- **Section 2 — Reward variants:** CSVs now available.
- **Section 3 — Cost / latency sweeps:** coefficient matrix is clear from `src/simulation.py`, but the corresponding rollout CSVs do not exist yet.

## Section 2: Terminal incentives / reward variants

### Figure: first-action distribution
Add a new cell group to `RQs/RQ3/plots.ipynb`:

1. Load these reward-variant CSVs from `RQs/relevant_csvs/`:
   - **Base**: `cql_full_tree_traces_50_msmarco_dev.csv`
   - **Stop penalty**: `cql_full_tree_stop_penalty_traces_50_msmarco_dev.csv`
   - **Low stop**: `cql_full_tree_50_msmarco_traces_low_stop.csv`
   - **Terminal reward**: `cql_full_tree_50_msmarco_traces_terminal_reward.csv`
2. For each variant, take the first action per `query_id`.
3. Classify into four categories: `STOP`, `Rerank`, `PRF` (`PseudoRelevanceFeedback`), `Reformulate` (`QueryReform`).
4. Compute proportion of queries in each category.
5. Plot a stacked bar chart:
   - x-axis: reward configuration
   - y-axis: proportion of queries
   - segments: the four action categories

### Table: reward-variant comparison
Append to `RQs/RQ3/latex_table.tex` a table titled *“Impact of Reward Variants on Retrieval Performance and Stopping Behavior”* with rows for each reward variant and these columns:

| Column | Computation |
|---|---|
| Mean NDCG | mean of terminal `ndcg_after` |
| Mean Recall | mean of terminal `recall_after` |
| Mean cost | mean of terminal `cumulative_cost` |
| Mean latency | mean of terminal `cumulative_latency_ms` |
| Immediate STOP (%) | first action == `STOP` |
| Later explicit STOP (%) | `STOP` actions with `step > 1`, averaged per query then across queries |
| Horizon reached (%) | terminal == 1 but last action ≠ `STOP` (equivalently `timeout == 1`) |
| Mean enhancement count | non-`STOP` actions per query |

Use the RQ2 LaTeX style (`tabularx`, `\small`, `\toprule`/`*rule`).

## Section 3: Cost and latency sweeps

From `src/simulation.py`:
- `reward_alpha` = ΔNDCG weight
- `reward_beta` = ΔRecall weight
- `reward_gamma` = cost penalty weight
- `reward_delta` = latency penalty weight

Base configuration (used in `src/main.py`):
- `reward_gamma = 0.2`
- `reward_delta = 0.1`

Required sweep configurations (hold other coefficients fixed at Base):

| Configuration | `reward_gamma` (cost) | `reward_delta` (latency) |
|---|---|---|
| Base | 0.2 | 0.1 |
| Cost = 0 | 0.0 | 0.1 |
| Cost = 2× Base | 0.4 | 0.1 |
| Latency = 0 | 0.2 | 0.0 |
| Latency = 2× Base | 0.2 | 0.2 |

Five configurations total; Base is shared.

### Missing dependency
The rollout CSVs for these five configurations do not exist in `RQs/relevant_csvs/`. They must either be generated with `generate_cql_traces.py` (by varying `SimConfig.reward_gamma` / `reward_delta`) or provided by the user.

## Open decision
Do you want the implementation agent to generate the five cost/latency sweep CSVs by running `generate_cql_traces.py` with the coefficient matrix above, or will you provide those CSVs?

**Recommended:** Generate them now, because the coefficient values and generation script are already defined.
