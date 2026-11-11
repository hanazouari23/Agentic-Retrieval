"""Aggregate CQL and baseline evaluation metrics into one comparison table."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation.evaluate_cql_traces import evaluate_traces


RESULTS = {
    # Baselines
    "STOP": "results/stop_traces_50_msmarco_dev.csv",
    "Rerank": "results/rerank_traces_50_msmarco_dev.csv",
    "Reformulate": "results/reformulate_traces_50_msmarco_dev.csv",
    "Greedy Oracle": "results/expert_traces_50_msmarco_dev.csv",
    "standalone_BC": "results/expert/standalone_bc_full_expert_msmarco_dev.csv",
    "Expert Boolean top20": "results/new_setup/expert_boolean_top20/expert_boolean_traces_50_msmarco_dev_top20.csv",
    # CQL trained models (50 dev queries)
    "CQL full tree": "results/cql_full_tree_traces_50_msmarco_dev.csv",
    "CQL terminal reward": "results/cql_full_tree_terminal_reward_traces_50_msmarco_dev.csv",
    "CQL stop penalty": "results/cql_full_tree_stop_penalty_traces_50_msmarco_dev.csv",
    "CQL BC warmstart (expert)": "results/expert/cql_full_expert_traces_50_msmarco_dev.csv",
    # CQL validation-selected models (20 dev queries)
    "CQL terminal alpha05 val best (20q)": "results/cql_terminal_reward_alpha05_val_64_32.csv",
    "CQL terminal alpha10 val best (20q)": "results/cql_terminal_reward_alpha10_val_64_32.csv",
    "CQL terminal alpha15 val best (20q)": "results/cql_terminal_reward_alpha15_val_64_32.csv",
    "CQL tree val best (20q)": "results/cql_full_tree_traces_20_msmarco_dev_64_32_best.csv",
    "CQL tree expert BC warmstart val (20q)": "results/cql_tree_expert_bc_warmstart_val_64_32.csv",
    "CQL stop penalty BC val (20q)": "results/cql_tree_stop_penalty_bc_val_20_msmarco_dev_64_32.csv",
    # CQL validation-selected models (partial 50 dev queries)
    "CQL terminal alpha15 val best (22/50q)": "results/cql_terminal_reward_alpha15_val_64_32_best_50_msmarco_dev.csv",
}


def capture_metrics(path: Path) -> dict:
    df = pd.read_csv(path)
    grouped = df.groupby("query_id")
    per_query = pd.DataFrame({
        "cumulative_cost": grouped["cumulative_cost"].max(),
        "ndcg_final": grouped["ndcg_after"].last(),
        "recall_final": grouped["recall_after"].last(),
        "ndcg_improvement": grouped.apply(lambda g: g["ndcg_after"].iloc[-1] - g["ndcg_before"].iloc[0]),
        "recall_improvement": grouped.apply(lambda g: g["recall_after"].iloc[-1] - g["recall_before"].iloc[0]),
        "total_reward": grouped["reward"].sum(),
        "n_steps": grouped.size(),
    }).reset_index()

    return {
        "queries": df["query_id"].nunique(),
        "mean_cost": per_query["cumulative_cost"].mean(),
        "mean_ndcg_final": per_query["ndcg_final"].mean(),
        "mean_recall_final": per_query["recall_final"].mean(),
        "mean_ndcg_improvement": per_query["ndcg_improvement"].mean(),
        "mean_recall_improvement": per_query["recall_improvement"].mean(),
        "mean_total_reward": per_query["total_reward"].mean(),
        "mean_steps": per_query["n_steps"].mean(),
        "positive_reward_queries": (per_query["total_reward"] > 0).sum(),
        "negative_reward_queries": (per_query["total_reward"] < 0).sum(),
        "zero_reward_queries": (per_query["total_reward"] == 0).sum(),
    }


def main() -> None:
    records = []
    for name, rel_path in RESULTS.items():
        p = Path(rel_path)
        if not p.exists():
            continue
        try:
            metrics = capture_metrics(p)
        except Exception as e:
            print(f"Skipping {name}: {e}", file=sys.stderr)
            continue
        records.append({"model": name, **metrics})

    summary = pd.DataFrame(records)
    cols = [
        "model",
        "queries",
        "mean_cost",
        "mean_ndcg_final",
        "mean_recall_final",
        "mean_ndcg_improvement",
        "mean_recall_improvement",
        "mean_total_reward",
        "mean_steps",
        "positive_reward_queries",
        "negative_reward_queries",
        "zero_reward_queries",
    ]
    summary = summary[cols].sort_values("mean_total_reward", ascending=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
