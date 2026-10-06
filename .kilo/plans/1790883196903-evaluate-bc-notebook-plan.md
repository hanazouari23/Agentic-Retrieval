# Plan: evaluate_BC.ipynb — BC Policy NDCG Improvement vs Cost Plot

## Goal
Create `notebooks/evaluate_BC.ipynb` that loads the trained `bc_policy.pt` model, runs it in the
retrieval simulation environment on 30 unseen MS MARCO queries (mirroring the approach in
`evaluate_cql_unseen_queries.ipynb`), and plots **cumulative cost** (x-axis) vs
**NDCG improvement percentage** (y-axis) per query.

## Context from codebase
- **`train_bc.py`**: Trains a `BCPolicy` (PyTorch `nn.Sequential` MLP, input_dim=399,
  hidden_dims=(256,128), num_actions=4). Saves `bc_policy.pt` containing `model_state_dict`,
  `optimizer_state_dict`, `best_test_accuracy`, `epoch`. Also saves `metrics_bc.json`.
- **`train_cql.py`**: Same structure but using d3rlpy `DiscreteCQLConfig`. Saves `cql_policy.pt`.
- **`evaluate_cql_unseen_queries.ipynb`**: Existing evaluation notebook for CQL policy on 30
  unseen MS MARCO dev queries. Uses SentenceTransformer encoder, OpenSearch `Retriever`, three
  agent instances (`ReformulationAgent`, `RerankingAgent`, `PRFAgent`), and `Simulation`.
  Records per-step: ndcg_before, ndcg_after, ndcg_gain, cost, cumulative_cost, latency_ms.
- **`src/simulation.py`**: `Simulation` class with `build_state()`, `compute_effects()`,
  `compute_ndcg()`, `compute_recall()`, `generate_trajectory()`. State dim = 399.
  Actions: 0=QR, 1=RR, 2=PRF, 3=STOP. `ACTION_COSTS = {0:1, 1:0.3, 2:0.3, 3:0.0}`.
- **`src/utils/retriever.py`**: `Retriever` needs OpenSearch endpoint + credentials from `.env`.
  `create_retriever_callable()` wraps it into the `query → (doc_ids, scores, corpus)` interface.
- **Data files**: `data/topics.ms-marco-dev.tsv`, `data/qrels.ms-marco-dev.tsv`,
  `data/trajectories_expert_5k.csv`. Model file: `bc_policy.pt`.

## Key Design Decisions

### 1. BC Policy Loading
- Reconstruct `BCPolicy(input_dim=399, hidden_dims=(256,128), num_actions=4)` inline in the
  notebook (copy the class from `train_bc.py`).
- Load `model_state_dict` from `bc_policy.pt` via `torch.load()`.
- Set `model.eval()` and wrap inference in `torch.no_grad()`.
- Apply the operational validity mask (same as CQL notebook): if the predicted action is
  invalid (same agent already used in previous step), override to `ACTION_STOP`.

### 2. Evaluation Loop (mirrors `evaluate_cql_unseen_queries.ipynb`)
- Load first 30 queries from `data/topics.ms-marco-dev.tsv`, filter to those with qrels.
- For each query:
  1. Retrieve top-50 docs via `Retriever` (BM25 baseline).
  2. Record initial NDCG@50 (baseline).
  3. Run up to `MAX_STEPS=3` simulation steps:
     - Build state vector (399-dim) via `sim.build_state()`.
     - Get BC policy prediction: `action = model(state_tensor).argmax().item()`.
     - Validate action against validity mask; override to STOP if invalid.
     - Call `sim.compute_effects()` to get new query, docs, scores, metrics, cost.
     - Record: ndcg_before, ndcg_after, cost, cumulative_cost, action, latency.
  4. Compute final NDCG (after all steps / STOP).
- Save per-step records to `eval_bc_30_unseen_queries.csv`.

### 3. NDCG Improvement Percentage
- **Definition**: `ndcg_improvement_pct = (final_ndcg - initial_ndcg) / max(initial_ndcg, 1e-9) * 100`
  computed **per query** (one value per query).
- This measures: "what percentage did NDCG improve from the baseline BM25 retrieval to the
  final state after the BC policy's actions?"

### 4. Cost
- **Definition**: `cumulative_cost` per query — the sum of all step costs
  (`ACTION_COSTS` + reformulation token cost) across the trajectory.

### 5. Plot Axes
- **x-axis**: cumulative cost (resource investment / independent variable)
- **y-axis**: NDCG improvement percentage (outcome / dependent variable)
- **Rationale**: Cost is the input (resource spent); NDCG improvement is the output (benefit
  gained). This is the conventional "for a given cost, what is the NDCG return?" framing.
- **Plot type**: Scatter plot, one point per query, with a horizontal line at 0% (baseline).
- Additional: histogram of NDCG improvement percentages.

## Notebook Structure (cells)

| # | Type    | Content |
|---|---------|---------|
| 1 | markdown | Title + description |
| 2 | code    | Imports, path setup, config constants (`MAX_STEPS=3`, `TOP_K=50`, `NDCG_K=50`) |
| 3 | code    | Load topics (30) + qrels; validate 30 queries have qrels |
| 4 | code    | Build simulation: encoder, retriever, agents, `Simulation` |
| 5 | code    | Define `BCPolicy` class; load `bc_policy.pt`; set eval mode |
| 6 | code    | Define `evaluate(query_id, query_text)` function (mirrors CQL notebook's `evaluate`) |
| 7 | code    | Run evaluation loop over 30 queries; collect records + errors |
| 8 | code    | Build DataFrame; save to `eval_bc_30_unseen_queries.csv` |
| 9 | code    | Compute per-query: initial_ndcg, final_ndcg, ndcg_improvement_pct, cumulative_cost |
| 10 | markdown | Plot heading |
| 11 | code    | Scatter plot: cost (x) vs NDCG improvement % (y); histogram of NDCG improvement % |

## Risks & Dependencies
- **OpenSearch endpoint**: The notebook requires `.env` with `RETRIEVAL_ENDPOINT`,
  `MY_USERNAME`, `MY_PASSWORD` for the `Retriever` to work. If no endpoint is available,
  the notebook will fail at the retrieval step. This is the same dependency as the existing
  CQL notebook.
- **Model file**: `bc_policy.pt` must exist (produced by `train_bc.py`).
- **Data files**: `data/topics.ms-marco-dev.tsv`, `data/qrels.ms-marco-dev.tsv` must exist.

## Validation
- Notebook runs end-to-end without errors.
- Scatter plot shows 30 points (one per query).
- CSV file `eval_bc_30_unseen_queries.csv` is saved with per-step records.
- NDCG improvement percentages and cumulative costs are computed and displayed.

## Skipped (per Lazy Senior Dev)
- No new `BCPolicy` import from `train_bc.py` — the class is defined inline in the notebook
  to keep the notebook self-contained (mirrors how the CQL notebook is self-contained).
- No comparison plot against CQL policy — out of scope (user only asked for BC).
- No per-step plots — aggregated per-query scatter is the primary deliverable.
