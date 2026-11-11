# RQ2

## Quality–cost and quality–latency trade-offs

**Table: one row per policy(Title: Comparison of Retrieval Policies: Effectiveness, Cost, and Latency)**

| Policy                          | Mean final NDCG@50 | Mean final Recall@50 | Mean cumulative abstract cost | Mean cumulative latency |
| ------------------------------- | ------------------ | -------------------- | ----------------------------- | ----------------------- |
| Always STOP                     | —                  | —                    | —                             | —                       |
| Rerank → STOP                   | —                  | —                    | —                             | —                       |
| PRF → STOP                      | —                  | —                    | —                             | —                       |
| Reformulate → STOP              | —                  | —                    | —                             | —                       |
| Random                          | —                  | —                    | —                             | —                       |
| Greedy oracle / two-step expert | —                  | —                    | —                             | —                       |
| Learned CQL policy              | —                  | —                    | —                             | —                       |

**two-panel scatter. x = mean cumulative cost or mean cumulative latency(calculate cumulative value per query then average across the total which in this case = 50); y = NDCG; marker/color = policy**:
The policies included here are: Greedy oracle, rerank, immediatly stop, reformulate and CQL

## Query level gains and losses

**One figure with four vertically stacked panels:**
Learned CQL policy.
Always rerank
The greedy oracle
Random

    For each panel:
    x-axis: evaluation queries.
    y-axis:
    		ΔNDCGq​=NDCGq,policy​−NDCGq,initial BM25​.
    Draw bars above or below a horizontal zero line.
    Use the same query ordering in every panel.
    Use the same y-axis limits.
    A useful ordering is initial BM25 NDCG, from low to high. This keeps the    ordering independent of which evaluated policy performs best. Avoid sorting each   panel separately, because that destroys query alignment.

**Pair the figure with a compact table covering all policies:**
Title: Query-Level Impact of Retrieval Policies

| Policy | Queries improved (%) | Unchanged (%) | Worsened (%) | Mean ΔNDCG |
| ------ | -------------------- | ------------- | ------------ | ---------- |
| ...    | ...                  | ...           | ...          | ...        |

**Action distribution plot**
A pizza slice kind of plot
Learned CQL policy.
The greedy oracle
Random
Immediate STOP (%), Later explicit STOP (%), Reformulate, Rerank, PRF,
