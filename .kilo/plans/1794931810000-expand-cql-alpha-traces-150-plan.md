# Expand CQL full-tree traces to 150 MS MARCO dev queries (alpha05/10/15)

## Goal
Produce three separate 150-query CQL rollout files:
- `RQs/relevant_csvs/cql_full_tree_150_msmarco_traces_alpha05.csv`
- `RQs/relevant_csvs/cql_full_tree_150_msmarco_traces_alpha10.csv`
- `RQs/relevant_csvs/cql_full_tree_150_msmarco_traces_alpha15.csv`

Each file must cover all 150 query IDs in `msmarco_dev_reference_success_150.csv`, reusing the 50 queries already present in the corresponding `cql_full_tree_50_msmarco_traces_alpha*.csv` files and only rolling out the remaining 100.

## Constraints
- Do **not** overwrite the existing 50-query files.
- Do **not** rerun the already-logged 50 queries.
- Preserve the exact CSV schema and column order of the existing CQL trace files.
- Use the same model architecture and reward hyperparameters that produced the existing 50-query files.
- No downstream reference updates (user handles manually).

## Context verified
- Trace generator: `evaluation/generate_cql_traces_param.py`.
- The script already supports resume: it reads existing `query_id`s from the output file, skips them, and appends new rows.
- It uses `--query-ids-from` to define the target query set and `--num-queries` to cap the total.
- Existing 50-query source files:
  - `RQs/relevant_csvs/cql_full_tree_50_msmarco_traces_alpha05.csv`
  - `RQs/relevant_csvs/cql_full_tree_50_msmarco_traces_alpha10.csv`
  - `RQs/relevant_csvs/cql_full_tree_50_msmarco_traces_alpha15.csv`
- Model files:
  - `models/cql_on_full_tree_dataset_alpha05.pt`
  - `models/cql_on_full_tree_dataset_alpha10.pt`
  - `models/cql_on_full_tree_dataset_alpha15.pt`
- `regenerate_traces.ps1` uses `--hidden-dims 256 128` for these full-tree CQL models.
- The CQL script uses index `msmarco-v2.1-segmented` and does **not** enable adaptive retrieval fallback (matches the user's preference).

## Implementation plan

### 1. Prepare resumable output files
For each alpha, copy the existing 50-query CSV to the new 150-query target path. This seeds the output file so the generator skips the first 50 query IDs.

```powershell
Copy-Item RQs/relevant_csvs/cql_full_tree_50_msmarco_traces_alpha05.csv RQs/relevant_csvs/cql_full_tree_150_msmarco_traces_alpha05.csv
Copy-Item RQs/relevant_csvs/cql_full_tree_50_msmarco_traces_alpha10.csv RQs/relevant_csvs/cql_full_tree_150_msmarco_traces_alpha10.csv
Copy-Item RQs/relevant_csvs/cql_full_tree_50_msmarco_traces_alpha15.csv RQs/relevant_csvs/cql_full_tree_150_msmarco_traces_alpha15.csv
```

### 2. Roll out the remaining 100 queries per model
Run the CQL trace generator for each model. The three runs are independent and can be launched in parallel.

```powershell
# alpha05
poetry run python -m evaluation.generate_cql_traces_param `
    --model-path models/cql_on_full_tree_dataset_alpha05.pt `
    --hidden-dims 256 128 `
    --output-path RQs/relevant_csvs/cql_full_tree_150_msmarco_traces_alpha05.csv `
    --num-queries 150 `
    --query-ids-from RQs/relevant_csvs/msmarco_dev_reference_success_150.csv

# alpha10
poetry run python -m evaluation.generate_cql_traces_param `
    --model-path models/cql_on_full_tree_dataset_alpha10.pt `
    --hidden-dims 256 128 `
    --output-path RQs/relevant_csvs/cql_full_tree_150_msmarco_traces_alpha10.csv `
    --num-queries 150 `
    --query-ids-from RQs/relevant_csvs/msmarco_dev_reference_success_150.csv

# alpha15
poetry run python -m evaluation.generate_cql_traces_param `
    --model-path models/cql_on_full_tree_dataset_alpha15.pt `
    --hidden-dims 256 128 `
    --output-path RQs/relevant_csvs/cql_full_tree_150_msmarco_traces_alpha15.csv `
    --num-queries 150 `
    --query-ids-from RQs/relevant_csvs/msmarco_dev_reference_success_150.csv
```

The generator will:
1. Load the 150 reference IDs.
2. Read existing IDs from the output file (the copied 50).
3. Skip those 50 and roll out the remaining 100.
4. Append the new trajectories to the output file.

### 3. Clean up intermediate files
Remove any `.tmp` files or per-run logs created during generation once the final files are validated. Do **not** delete the original 50-query files.

## Validation
For each of the three output files:
1. File has exactly 150 unique `query_id`s.
2. All 150 IDs are present in `msmarco_dev_reference_success_150.csv` and no extras.
3. Columns and column order match the corresponding 50-query source file.
4. The first 50 query IDs in the output file match the source 50-query file exactly.
5. Report per-file row counts and any failed queries logged by the generator.

## Risks / notes
- CQL inference can be slow, especially with model loading. The three runs can execute in parallel to save wall-clock time.
- If a run is interrupted, rerunning the same command will resume from the output file because of the built-in skip logic.
- The output file order will be: original 50 in their original order, followed by the new 100 in the order of `msmarco_dev_reference_success_150.csv`.

## Open questions / none
All decisions resolved: output filenames are the clean `cql_full_tree_150_msmarco_traces_alpha*.csv` names, existing 50-query files are reused as seeds and left untouched, no reference updates.
