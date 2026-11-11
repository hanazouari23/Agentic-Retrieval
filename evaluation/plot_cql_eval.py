"""Plot CQL rollout cost vs NDCG improvement, colored by reward."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_cost_vs_ndcg(csv_path: Path, output_path: Path) -> None:
    df = pd.read_csv(csv_path)
    grouped = df.groupby("query_id")
    per_query = pd.DataFrame({
        "cumulative_cost": grouped["cumulative_cost"].max(),
        "ndcg_improvement": grouped.apply(lambda g: g["ndcg_after"].iloc[-1] - g["ndcg_before"].iloc[0]),
        "total_reward": grouped["reward"].sum(),
    }).reset_index()

    def reward_category(r: float) -> str:
        if r > 0:
            return "positive"
        if r < 0:
            return "negative"
        return "zero"

    per_query["reward_category"] = per_query["total_reward"].apply(reward_category)

    point_counts = (
        per_query
        .groupby(["ndcg_improvement", "cumulative_cost", "reward_category"])
        .size()
        .reset_index(name="n_queries")
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"positive": "green", "zero": "gray", "negative": "red"}

    for category, color in colors.items():
        subset = point_counts[point_counts["reward_category"] == category]
        if subset.empty:
            continue
        ax.scatter(
            subset["ndcg_improvement"],
            subset["cumulative_cost"],
            s=80 + 40 * subset["n_queries"],
            c=color,
            alpha=0.6,
            edgecolors="black",
            label=f"{category} reward",
        )

    for _, row in point_counts.iterrows():
        ax.annotate(
            str(int(row["n_queries"])),
            (row["ndcg_improvement"], row["cumulative_cost"]),
            textcoords="offset points",
            xytext=(8, 5),
            fontsize=9,
        )

    ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("NDCG overall improvement")
    ax.set_ylabel("Cumulative trajectory cost")
    ax.set_title("CQL (BC warm-start) on 50 MS MARCO dev queries")
    ax.legend(title="Total reward", loc="upper left")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot CQL traces cost vs NDCG improvement.")
    parser.add_argument("--csv", type=Path, default=Path("results/expert/cql_full_expert_traces_50_msmarco_dev_segmented_corpus.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/expert/cql_eval_cost_vs_ndcg.png"))
    args = parser.parse_args(argv)
    plot_cost_vs_ndcg(args.csv, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
