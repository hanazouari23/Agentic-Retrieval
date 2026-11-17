# Extract STOP Baseline Traces for 150-Query Set

## Goal
Create `RQs/relevant_csvs/stop_traces_150_msmarco_dev.csv` from the CQL rollout traces by keeping only the initial-retrieval step for each query and relabeling it as a STOP action.

## Input
`results/cql_base_uncorrected_traces_150_msmarco_dev.csv` (the full CQL rollout for the final 150-query reference set).

## Output
`RQs/relevant_csvs/stop_traces_150_msmarco_dev.csv`

Columns to keep, in order:
```
query_id, query, new_query, action_name, query_length, score_spread, score_entropy, step, ndcg_before, ndcg_after, recall_before, recall_after
```

## Implementation

Create `helper_scripts/extract_stop_baseline_traces.py` that:
1. Loads the input CQL trace CSV with pandas.
2. Keeps only rows where `step == 1` (one row per query, the initial BM25 retrieval).
3. Sets `action_name = "STOP"` for every row.
4. Sets `new_query = query` for every row.
5. Sets `ndcg_after = ndcg_before` and `recall_after = recall_before` for every row, because a STOP action leaves the retrieval state unchanged.
6. Selects only the columns listed above.
7. Writes the output CSV.

Usage:
```powershell
poetry run python helper_scripts/extract_stop_baseline_traces.py --input-csv results/cql_base_uncorrected_traces_150_msmarco_dev.csv --output-csv RQs/relevant_csvs/stop_traces_150_msmarco_dev.csv
```

## Validation
- [ ] Output file has exactly 150 rows (one per query).
- [ ] Every row has `action_name == "STOP"`.
- [ ] Every row has `step == 1`.
- [ ] Every row has `new_query == query`.
- [ ] Every row has `ndcg_after == ndcg_before` and `recall_after == recall_before`.
- [ ] Output contains only the 12 requested columns.
