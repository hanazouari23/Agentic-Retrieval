# Expand always-rerank->stop traces to 150 MS MARCO dev queries

## Goal
Produce `RQs/relevant_csvs/rerank_traces_150_msmarco_dev.csv` containing **always rerank→STOP** trajectories for all 150 query IDs in `msmarco_dev_reference_success_150.csv`, reusing the 50 queries already present in `rerank_traces_50_msmarco_dev.csv` and only rolling out the missing 100.

## Constraints
- Do **not** overwrite `rerank_traces_50_msmarco_dev.csv`.
- Do **not** rerun the already-logged 50 queries.
- Do **not** use adaptive retrieval fallback for this run; use `size=50` consistently.
- Preserve the exact CSV schema and column order of the existing file.
- No downstream reference updates (user handles manually).

## Context
- The `rerank` policy in `src/simulation.py` forces `Rerank` on step 1 and `STOP` on step 2, producing exactly the trajectories in `rerank_traces_50_msmarco_dev.csv`.
- `evaluation/generate_baseline_traces.py` already supports `--query-ids`, `--resume-from`, `--output-path`, and `--append` from the expert expansion.
- `src/utils/retriever.py` currently has adaptive fallback enabled in `generate_baseline_traces.py` (`adaptive_top_k=True`). For this run it must be disabled.

## Implementation plan

### 1. Make adaptive fallback configurable in `generate_baseline_traces.py`
Add a CLI flag so the same script can be used with or without adaptive fallback:
- `--adaptive-top-k` / `--no-adaptive-top-k` (default: enabled, to preserve prior expert/baseline behavior).
- Pass the flag value to `create_retriever_callable(..., adaptive_top_k=...)`.

For the rerank expansion, run with `--no-adaptive-top-k`.

### 2. Generate the missing 100 rerank traces
Run only the `rerank` policy for the missing 100 query IDs:

```powershell
poetry run python evaluation/generate_baseline_traces.py `
    --policies rerank `
    --no-adaptive-top-k `
    --query-ids RQs/relevant_csvs/msmarco_dev_reference_success_150.csv `
    --resume-from RQs/relevant_csvs/rerank_traces_50_msmarco_dev.csv `
    --output-path RQs/relevant_csvs/rerank_traces_150_msmarco_dev_new.csv
```

This writes only the 100 new trajectories. Because the API is reportedly stable now, adaptive fallback is not needed; if transient `500` errors do appear, the standard 3 retries at `size=50` will apply.

### 3. Concatenate old + new into the final 150-query file
After step 2 succeeds, concatenate `rerank_traces_50_msmarco_dev.csv` (header + rows) with `rerank_traces_150_msmarco_dev_new.csv` (rows only, same schema) into `RQs/relevant_csvs/rerank_traces_150_msmarco_dev.csv`.

### 4. Clean up intermediate files
Remove the temporary `rerank_traces_150_msmarco_dev_new.csv` (and any `.tmp` or log files created during generation) once the final file is validated.

## Validation
1. `rerank_traces_150_msmarco_dev.csv` has exactly 150 unique `query_id`s.
2. Its columns and column order match `rerank_traces_50_msmarco_dev.csv`.
3. The first 50 query IDs and their rows match the original 50-query file.
4. The remaining 100 query IDs are exactly the set difference between `msmarco_dev_reference_success_150.csv` and the original 50.
5. Each query has a `Rerank` transition followed by a `STOP` transition (2 rows per query, unless a failure produced a different pattern — flag if so).

## Open questions / none
All decisions resolved: same expansion pattern as the expert traces, adaptive fallback disabled for this run, no reference updates.
