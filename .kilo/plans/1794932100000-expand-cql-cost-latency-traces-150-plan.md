# Expand CQL full-tree traces to 150 MS MARCO dev queries (cost / latency variants)

## Goal
Produce four separate 150-query CQL rollout files:
- `RQs/relevant_csvs/cql_full_tree_150_msmarco_traces_cost_2x.csv`
- `RQs/relevant_csvs/cql_full_tree_150_msmarco_traces_cost_zero.csv`
- `RQs/relevant_csvs/cql_full_tree_150_msmarco_traces_latency_2x.csv`
- `RQs/relevant_csvs/cql_full_tree_150_msmarco_traces_latency_zero.csv`

Each file must cover all 150 query IDs in `msmarco_dev_reference_success_150.csv`, reusing the 50 queries already present in the corresponding `cql_full_tree_50_msmarco_traces_{cost,latency}_{2x,zero}.csv` files and only rolling out the remaining 100.

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
  - Cost: `cql_full_tree_50_msmarco_traces_cost_2x.csv`, `cql_full_tree_50_msmarco_traces_cost_zero.csv`
  - Latency: `cql_full_tree_50_msmarco_traces_latency_2x.csv`, `cql_full_tree_50_msmarco_traces_latency_zero.csv`
- Model files:
  - Cost: `models/cql_on_full_tree_dataset_cost_2x.pt`, `models/cql_on_full_tree_dataset_cost_zero.pt`
  - Latency: `models/cql_on_full_tree_dataset_latency_2x.pt`, `models/cql_on_full_tree_dataset_latency_zero.pt`
- `regenerate_traces.ps1` uses `--hidden-dims 256 128` for these full-tree CQL models.
- The CQL script uses index `msmarco-v2.1-segmented` and does **not** enable adaptive retrieval fallback (matches the user's preference).

## Implementation plan

### 1. Prepare resumable output files
For each variant, copy the existing 50-query CSV to the new 150-query target path. This seeds the output file so the generator skips the first 50 query IDs.

```powershell
$variants = @("cost_2x", "cost_zero", "latency_2x", "latency_zero")
foreach ($v in $variants) {
    Copy-Item "RQs/relevant_csvs/cql_full_tree_50_msmarco_traces_$v.csv" `
        "RQs/relevant_csvs/cql_full_tree_150_msmarco_traces_$v.csv"
}
```

### 2. Roll out the remaining 100 queries per model
Run the CQL trace generator for each variant. The four runs are independent and can be launched in parallel.

```powershell
$variants = @(
    @{ name = "cost_2x";     model = "cql_on_full_tree_dataset_cost_2x.pt" },
    @{ name = "cost_zero";   model = "cql_on_full_tree_dataset_cost_zero.pt" },
    @{ name = "latency_2x";  model = "cql_on_full_tree_dataset_latency_2x.pt" },
    @{ name = "latency_zero"; model = "cql_on_full_tree_dataset_latency_zero.pt" }
)
foreach ($v in $variants) {
    poetry run python -m evaluation.generate_cql_traces_param `
        --model-path "models/$($v.model)" `
        --hidden-dims 256 128 `
        --output-path "RQs/relevant_csvs/cql_full_tree_150_msmarco_traces_$($v.name).csv" `
        --num-queries 150 `
        --query-ids-from RQs/relevant_csvs/msmarco_dev_reference_success_150.csv
}
```

The generator will:
1. Load the 150 reference IDs.
2. Read existing IDs from the output file (the copied 50).
3. Skip those 50 and roll out the remaining 100.
4. Append the new trajectories to the output file.

### 3. Clean up intermediate files
Remove any `.tmp` files or per-run logs created during generation once the final files are validated. Do **not** delete the original 50-query files.

## Validation
For each of the four output files:
1. File has exactly 150 unique `query_id`s.
2. All 150 IDs are present in `msmarco_dev_reference_success_150.csv` and no extras.
3. Columns and column order match the corresponding 50-query source file.
4. The first 50 query IDs in the output file match the source 50-query file exactly.
5. Report per-file row counts and any failed queries logged by the generator.

## Risks / notes
- CQL inference can be slow, especially with model loading. The four runs can execute in parallel to save wall-clock time, but that increases GPU/CPU and memory pressure; if a run OOMs, run the variants sequentially instead.
- If a run is interrupted, rerunning the same command will resume from the output file because of the built-in skip logic.
- The output file order will be: original 50 in their original order, followed by the new 100 in the order of `msmarco_dev_reference_success_150.csv`.

## Open questions / none
All decisions resolved: output filenames are the clean `cql_full_tree_150_msmarco_traces_<variant>.csv` names, existing 50-query files are reused as seeds and left untouched, no reference updates.
