# CQL Multi-Dataset Training Plan

## Goal
Train one Discrete CQL policy per corrected tree-branches dataset using `training/train_cql.py`, producing models in `models/` and metrics in `training/metrics/`. Start with the `alpha15_corrected` dataset and run each dataset individually, followed by a separate manual evaluation step.

## Decisions

| Topic | Decision |
|-------|----------|
| `alpha` in dataset names | Refers to the **reward NDCG weight**, not the CQL conservative-Q coefficient. `CQL_ALPHA` remains `1.0`. |
| `gamma` in dataset names (e.g. `gamma04`) | Refers to the **reward cost weight**, not the RL discount factor. `GAMMA` remains `0.99`. |
| Training workflow | Convert `train_cql.py` to CLI args and run **one dataset at a time** manually. |
| Evaluation workflow | Separate manual step using `evaluation/evaluate_cql_traces.py` on the generated rollout CSV. |
| Model naming | Replace the prefix `mdp_dataset_tree_branches_5000` with `cql`, keep the suffix. Example: `mdp_dataset_tree_branches_5000_alpha15_corrected.h5` → `models/cql_alpha15_corrected.pt`. |
| Metrics naming | `training/metrics/train_cql_metrics_<suffix>.json`. Example: `training/metrics/train_cql_metrics_alpha15_corrected.json`. |
| Hyperparameters | Keep current defaults unless overridden via CLI: `hidden_dims=(256,128)`, `batch_size=256`, `gamma=0.99`, `lr=3e-4`, `n_critics=2`, `target_update_interval=8000`, `cql_alpha=1.0`, `n_steps=50000`, `n_steps_per_epoch=10000`, `seed=42`. |
| Backward compatibility | `train_cql.py` should still run with no arguments and reproduce the current behavior (`mdp_dataset_tree_branches_5000_stop_penalty_coeff005.h5` → `cql_on_full_tree_dataset_5000_stop_penalty_coeff005.pt`). |

## Datasets to Train (start at alpha15)

The following 10 datasets are included. The 6 excluded datasets are not listed here:
`alpha05_gamma04_corrected`, `alpha05_gamma04_stop_penalty_corrected`, `stop_penalty_corrected`, `stop_penalty_coeff005_corrected`, `stop_penalty_coeff02_corrected`, `stop_rewarded_corrected`.

| # | Dataset | Model | Metrics |
|---|---------|-------|---------|
| 1 | `mdp_dataset_tree_branches_5000_alpha15_corrected.h5` | `models/cql_alpha15_corrected.pt` | `training/metrics/train_cql_metrics_alpha15_corrected.json` |
| 2 | `mdp_dataset_tree_branches_5000_alpha10_corrected.h5` | `models/cql_alpha10_corrected.pt` | `training/metrics/train_cql_metrics_alpha10_corrected.json` |
| 3 | `mdp_dataset_tree_branches_5000_alpha05_corrected.h5` | `models/cql_alpha05_corrected.pt` | `training/metrics/train_cql_metrics_alpha05_corrected.json` |
| 4 | `mdp_dataset_tree_branches_5000_cost_2x_corrected.h5` | `models/cql_cost_2x_corrected.pt` | `training/metrics/train_cql_metrics_cost_2x_corrected.json` |
| 5 | `mdp_dataset_tree_branches_5000_cost_zero_corrected.h5` | `models/cql_cost_zero_corrected.pt` | `training/metrics/train_cql_metrics_cost_zero_corrected.json` |
| 6 | `mdp_dataset_tree_branches_5000_latency_2x_corrected.h5` | `models/cql_latency_2x_corrected.pt` | `training/metrics/train_cql_metrics_latency_2x_corrected.json` |
| 7 | `mdp_dataset_tree_branches_5000_latency_zero_corrected.h5` | `models/cql_latency_zero_corrected.pt` | `training/metrics/train_cql_metrics_latency_zero_corrected.json` |
| 8 | `mdp_dataset_tree_branches_5000_low_stop_corrected.h5` | `models/cql_low_stop_corrected.pt` | `training/metrics/train_cql_metrics_low_stop_corrected.json` |
| 9 | `mdp_dataset_tree_branches_5000_stop_penalty_coeff01_corrected.h5` | `models/cql_stop_penalty_coeff01_corrected.pt` | `training/metrics/train_cql_metrics_stop_penalty_coeff01_corrected.json` |
| 10 | `mdp_dataset_tree_branches_5000_terminal_reward_corrected.h5` | `models/cql_terminal_reward_corrected.pt` | `training/metrics/train_cql_metrics_terminal_reward_corrected.json` |

## Implementation Steps

1. **Modify `training/train_cql.py` to accept CLI arguments.**
   - Add `--dataset` (path to `.h5`).
   - Add `--model-name` (basename only, e.g. `cql_alpha15_corrected`); default to current behavior.
   - Add optional overrides: `--hidden-dims`, `--batch-size`, `--gamma`, `--learning-rate`, `--n-critics`, `--target-update-interval`, `--cql-alpha`, `--n-steps`, `--n-steps-per-epoch`, `--seed`.
   - Derive model save path as `models/{model_name}.pt`.
   - Derive metrics save path as `training/metrics/train_cql_metrics_{suffix}.json`, where `suffix` is `model_name` with the leading `cql_` stripped.
   - Create `training/metrics/` if it does not exist.
   - Keep the current constants as defaults so the script still runs with zero arguments and reproduces the current output names.

2. **The user runs the training commands below one at a time.**
   - These commands are not to be executed by the agent.
   - Verify after each run that `models/cql_<suffix>.pt` and `training/metrics/train_cql_metrics_<suffix>.json` were created.

3. **The user runs the evaluation commands below for each trained model.**
   - These commands are not to be executed by the agent.
   - Generate traces with `generate_cql_traces.py` at the repo root.
   - The 50 evaluation queries come from `RQs/relevant_csvs/cql_full_tree_traces_50_msmarco_dev.csv` (the default `--query-ids-from` value).
   - Run `evaluation/evaluate_cql_traces.py` on the resulting CSV.

## Training Commands

Run these one at a time from the repo root. Do not run these automatically:

```powershell
poetry run python training/train_cql.py --dataset data/training/mdp_dataset_tree_branches_5000_alpha15_corrected.h5 --model-name cql_alpha15_corrected
poetry run python training/train_cql.py --dataset data/training/mdp_dataset_tree_branches_5000_alpha10_corrected.h5 --model-name cql_alpha10_corrected
poetry run python training/train_cql.py --dataset data/training/mdp_dataset_tree_branches_5000_alpha05_corrected.h5 --model-name cql_alpha05_corrected
poetry run python training/train_cql.py --dataset data/training/mdp_dataset_tree_branches_5000_cost_2x_corrected.h5 --model-name cql_cost_2x_corrected
poetry run python training/train_cql.py --dataset data/training/mdp_dataset_tree_branches_5000_cost_zero_corrected.h5 --model-name cql_cost_zero_corrected
poetry run python training/train_cql.py --dataset data/training/mdp_dataset_tree_branches_5000_latency_2x_corrected.h5 --model-name cql_latency_2x_corrected
poetry run python training/train_cql.py --dataset data/training/mdp_dataset_tree_branches_5000_latency_zero_corrected.h5 --model-name cql_latency_zero_corrected
poetry run python training/train_cql.py --dataset data/training/mdp_dataset_tree_branches_5000_low_stop_corrected.h5 --model-name cql_low_stop_corrected
poetry run python training/train_cql.py --dataset data/training/mdp_dataset_tree_branches_5000_stop_penalty_coeff01_corrected.h5 --model-name cql_stop_penalty_coeff01_corrected
poetry run python training/train_cql.py --dataset data/training/mdp_dataset_tree_branches_5000_terminal_reward_corrected.h5 --model-name cql_terminal_reward_corrected
```

## Evaluation Commands

For each trained model, generate traces and evaluate. Run these manually; do not run them automatically. The 50 reference queries are fixed via `--query-ids-from`.

```powershell
poetry run python generate_cql_traces.py --model-path models/cql_alpha15_corrected.pt --hidden-dims 256 128 --output-path results/cql_alpha15_corrected_traces.csv --num-queries 50 --query-ids-from RQs/relevant_csvs/cql_full_tree_traces_50_msmarco_dev.csv
poetry run python -m evaluation.evaluate_cql_traces results/cql_alpha15_corrected_traces.csv

poetry run python generate_cql_traces.py --model-path models/cql_alpha10_corrected.pt --hidden-dims 256 128 --output-path results/cql_alpha10_corrected_traces.csv --num-queries 50 --query-ids-from RQs/relevant_csvs/cql_full_tree_traces_50_msmarco_dev.csv
poetry run python -m evaluation.evaluate_cql_traces results/cql_alpha10_corrected_traces.csv

poetry run python generate_cql_traces.py --model-path models/cql_alpha05_corrected.pt --hidden-dims 256 128 --output-path results/cql_alpha05_corrected_traces.csv --num-queries 50 --query-ids-from RQs/relevant_csvs/cql_full_tree_traces_50_msmarco_dev.csv
poetry run python -m evaluation.evaluate_cql_traces results/cql_alpha05_corrected_traces.csv

poetry run python generate_cql_traces.py --model-path models/cql_cost_2x_corrected.pt --hidden-dims 256 128 --output-path results/cql_cost_2x_corrected_traces.csv --num-queries 50 --query-ids-from RQs/relevant_csvs/cql_full_tree_traces_50_msmarco_dev.csv
poetry run python -m evaluation.evaluate_cql_traces results/cql_cost_2x_corrected_traces.csv

poetry run python generate_cql_traces.py --model-path models/cql_cost_zero_corrected.pt --hidden-dims 256 128 --output-path results/cql_cost_zero_corrected_traces.csv --num-queries 50 --query-ids-from RQs/relevant_csvs/cql_full_tree_traces_50_msmarco_dev.csv
poetry run python -m evaluation.evaluate_cql_traces results/cql_cost_zero_corrected_traces.csv

poetry run python generate_cql_traces.py --model-path models/cql_latency_2x_corrected.pt --hidden-dims 256 128 --output-path results/cql_latency_2x_corrected_traces.csv --num-queries 50 --query-ids-from RQs/relevant_csvs/cql_full_tree_traces_50_msmarco_dev.csv
poetry run python -m evaluation.evaluate_cql_traces results/cql_latency_2x_corrected_traces.csv

poetry run python generate_cql_traces.py --model-path models/cql_latency_zero_corrected.pt --hidden-dims 256 128 --output-path results/cql_latency_zero_corrected_traces.csv --num-queries 50 --query-ids-from RQs/relevant_csvs/cql_full_tree_traces_50_msmarco_dev.csv
poetry run python -m evaluation.evaluate_cql_traces results/cql_latency_zero_corrected_traces.csv

poetry run python generate_cql_traces.py --model-path models/cql_low_stop_corrected.pt --hidden-dims 256 128 --output-path results/cql_low_stop_corrected_traces.csv --num-queries 50 --query-ids-from RQs/relevant_csvs/cql_full_tree_traces_50_msmarco_dev.csv
poetry run python -m evaluation.evaluate_cql_traces results/cql_low_stop_corrected_traces.csv

poetry run python generate_cql_traces.py --model-path models/cql_stop_penalty_coeff01_corrected.pt --hidden-dims 256 128 --output-path results/cql_stop_penalty_coeff01_corrected_traces.csv --num-queries 50 --query-ids-from RQs/relevant_csvs/cql_full_tree_traces_50_msmarco_dev.csv
poetry run python -m evaluation.evaluate_cql_traces results/cql_stop_penalty_coeff01_corrected_traces.csv

poetry run python generate_cql_traces.py --model-path models/cql_terminal_reward_corrected.pt --hidden-dims 256 128 --output-path results/cql_terminal_reward_corrected_traces.csv --num-queries 50 --query-ids-from RQs/relevant_csvs/cql_full_tree_traces_50_msmarco_dev.csv
poetry run python -m evaluation.evaluate_cql_traces results/cql_terminal_reward_corrected_traces.csv
```

## Validation

- [ ] `train_cql.py --help` prints all new arguments.
- [ ] Running with no arguments still produces the original default model and metrics names.
- [ ] Each dataset trains to completion (50000 steps) and saves exactly one `.pt` file and one `.json` metrics file.
- [ ] Metrics JSON contains `config`, `data_stats`, and `epoch_metrics`.
- [ ] Evaluation CSV is generated and `evaluate_cql_traces.py` runs without errors.

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Training each dataset takes significant time; queue is manual. | Commands are listed in order; run one at a time. |
| Existing `models/` already contains old variants with similar names. | New names use the `cql_<suffix>.pt` pattern and will not overwrite `cql_on_full_tree_dataset_*.pt` files. |
| `evaluate_cql_traces.py` requires a pre-generated CSV, not the model directly. | Evaluation commands generate the CSV first. |
| Metrics directory does not exist. | Trainer creates `training/metrics/` automatically. |
