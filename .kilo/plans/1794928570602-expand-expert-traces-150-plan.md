# Expand two-step expert traces to 150 MS MARCO dev queries

## Goal
Produce `RQs/relevant_csvs/expert_traces_150_msmarco_dev.csv` containing two-step-ahead greedy expert trajectories for **all 150 query IDs in `msmarco_dev_reference_success_150.csv`**, reusing the 50 queries already present in `expert_traces_50_msmarco_dev.csv` and only rolling out the missing 100.

## Constraints
- Do **not** overwrite `expert_traces_50_msmarco_dev.csv`.
- Do **not** rerun the already-logged 50 queries (the expert policy is deterministic given the same state, so reuse is safe).
- Preserve the exact CSV schema and column order of the existing file.

## Context verified
- `msmarco_dev_reference_success_150.csv` contains 150 `query_id`s. The first 50 IDs match the unique IDs in `expert_traces_50_msmarco_dev.csv`.
- The remaining 100 IDs run from `1095942` to `126821`.
- Existing `evaluation/generate_baseline_traces.py` derives its query list from `CQL_TRACES_PATH` and writes to `OUTPUT_DIR / f"{policy}_traces_50_msmarco_dev.csv`.
- The two-step expert policy is implemented in `src/simulation.py` (`generate_trajectory(policy="expert")` → `_policy_expert_two_step`).

## Implementation plan

### 1. Extend `evaluation/generate_baseline_traces.py`
Add CLI options so the same rollout code can be reused for the expansion:
- `--query-ids PATH` – path to a single-column `query_id` CSV (default: keep existing `CQL_TRACES_PATH` behavior for backward compatibility).
- `--resume-from PATH` – path to an existing traces CSV. If provided, skip any `query_id` already present in that file.
- `--output-path PATH` – optional explicit output path. If omitted, fall back to the current `OUTPUT_DIR / f"{policy}_traces_50_msmarco_dev.csv` behavior.

When `--query-ids` is provided, resolve query text and qrels the same way the script currently does. When `--resume-from` is provided, read its unique `query_id`s and remove them from the rollout list. Print the counts: already-present, to-roll-out, total target.

### 2. Generate the missing 100 expert traces
Run only the `expert` policy for the missing 100 query IDs:

```powershell
poetry run python evaluation/generate_baseline_traces.py `
    --policies expert `
    --query-ids RQs/relevant_csvs/msmarco_dev_reference_success_150.csv `
    --resume-from RQs/relevant_csvs/expert_traces_50_msmarco_dev.csv `
    --output-path RQs/relevant_csvs/expert_traces_150_msmarco_dev.csv
```

This should write only the 100 new trajectories to the specified output file.

### 3. Concatenate old + new into the final 150-query file
After step 2 succeeds, concatenate the existing 50-query CSV (header + rows) with the new 100-query CSV (rows only, same schema). Verify the result has exactly 150 unique `query_id`s and the same columns as the original.

This can be done with a short pandas one-liner or by adding a `--concat-with PATH` option to the script so it happens automatically. Recommended: keep it explicit and reproducible in a helper command rather than manual copy/paste.

### 4. Output location
Write the final 150-query file to `RQs/relevant_csvs/expert_traces_150_msmarco_dev.csv`. Leave `RQs/relevant_csvs/expert_traces_50_msmarco_dev.csv` untouched. The user will manually update any downstream references (e.g., `results/` copies, notebooks, comparison scripts) outside this task.

## Validation
1. `expert_traces_150_msmarco_dev.csv` has 150 unique `query_id`s.
2. Its columns and column order match `expert_traces_50_msmarco_dev.csv` exactly.
3. The first 50 query IDs and their rows are byte-for-byte identical to the original file (or at least produce the same per-query aggregates).
4. The remaining 100 query IDs are exactly the set difference between `msmarco_dev_reference_success_150.csv` and the original 50.
5. No reference updates are performed as part of this task (user will handle manually).

## Open questions / none
All decisions resolved: new file `RQs/relevant_csvs/expert_traces_150_msmarco_dev.csv`, reuse existing 50 queries, roll out only the missing 100, no reference updates.
