"""Evaluate CQL rollout traces CSV."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def evaluate_traces(csv_path: Path) -> None:
    df = pd.read_csv(csv_path)

    print("=== Basic stats ===")
    print(f"Rows: {len(df)}")
    print(f"Queries: {df['query_id'].nunique()}")
    print()

    print("=== Action distribution ===")
    print(df["action_name"].value_counts())
    print()

    print("=== Step distribution ===")
    print(df["step"].value_counts().sort_index())
    print()

    print("=== Terminal/timeout counts ===")
    print(f"terminal=1: {(df['terminal'] == 1).sum()}")
    print(f"timeout=1: {(df['timeout'] == 1).sum()}")
    print()

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

    def reward_category(r: float) -> str:
        if r > 0:
            return "positive"
        if r < 0:
            return "negative"
        return "zero"

    per_query["reward_category"] = per_query["total_reward"].apply(reward_category)

    print("=== Reward categories ===")
    print(per_query["reward_category"].value_counts())
    print()

    print("=== Aggregate metrics ===")
    print(f"Mean cumulative cost: {per_query['cumulative_cost'].mean():.4f}")
    print(f"Mean NDCG@50 final: {per_query['ndcg_final'].mean():.4f}")
    print(f"Mean Recall@100 final: {per_query['recall_final'].mean():.4f}")
    print(f"Mean NDCG improvement: {per_query['ndcg_improvement'].mean():.4f}")
    print(f"Mean Recall improvement: {per_query['recall_improvement'].mean():.4f}")
    print(f"Mean total reward: {per_query['total_reward'].mean():.4f}")
    print(f"Mean steps per query: {per_query['n_steps'].mean():.2f}")
    print()

    print("=== Cost breakdown ===")
    print(f"Total retrieval latency (ms): {df['cumulative_latency_ms'].max():.2f}")
    print(f"Mean latency per query (ms): {grouped['cumulative_latency_ms'].max().mean():.2f}")
    print()

    print("=== Per-query summary (first 15) ===")
    print(per_query.sort_values("total_reward", ascending=False).head(15).to_string(index=False))
    print()

    print("=== Worst 10 by total reward ===")
    print(per_query.sort_values("total_reward", ascending=True).head(10).to_string(index=False))
    print()

    print("=== Correlation: NDCG improvement vs cumulative cost ===")
    print(per_query[["ndcg_improvement", "cumulative_cost", "total_reward"]].corr().to_string())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate CQL rollout traces.")
    parser.add_argument("csv", type=Path, default=Path("results/expert/cql_full_expert_traces_50_msmarco_dev_segmented_corpus.csv"))
    args = parser.parse_args(argv)
    evaluate_traces(args.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
