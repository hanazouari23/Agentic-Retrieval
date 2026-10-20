import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

policy_files = {
    "CQL": "../results/cql_full_tree_traces_50_msmarco_dev.csv",
    "STOP": "../results/stop_traces_50_msmarco_dev.csv",
    "Rerank": "../results/rerank_traces_50_msmarco_dev.csv",
    "Reformulate": "../results/reformulate_traces_50_msmarco_dev.csv",
    "Expert": "../results/expert_traces_50_msmarco_dev.csv",
}

records = []
for policy_name, path in policy_files.items():
    p = Path(path)
    if not p.exists():
        print(f"{policy_name}: missing")
        continue
    df = pd.read_csv(p)
    if df.empty:
        print(f"{policy_name}: empty")
        continue
    grouped = df.groupby("query_id")
    per_query = pd.DataFrame({
        "cumulative_cost": grouped["cumulative_cost"].max(),
        "ndcg_final": grouped["ndcg_after"].last(),
        "recall_final": grouped["recall_after"].last(),
    })
    records.append({
        "policy": policy_name,
        "mean_cost": per_query["cumulative_cost"].mean(),
        "mean_ndcg": per_query["ndcg_final"].mean(),
        "mean_recall": per_query["recall_final"].mean(),
        "n_queries": len(per_query),
    })

summary = pd.DataFrame(records)
print(summary)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

palette = {
    "CQL": "tab:green",
    "STOP": "tab:gray",
    "Rerank": "tab:blue",
    "Reformulate": "tab:purple",
    "Expert": "tab:orange",
}

ax = axes[0]
for _, row in summary.iterrows():
    policy = row["policy"]
    color = palette.get(policy, "black")
    ax.scatter(
        row["mean_cost"],
        row["mean_ndcg"],
        s=200,
        c=color,
        alpha=0.8,
        edgecolors="black",
        label=policy,
        zorder=3 if policy == "CQL" else 2,
    )
    ax.annotate(
        policy,
        (row["mean_cost"], row["mean_ndcg"]),
        textcoords="offset points",
        xytext=(8, 5),
        fontsize=9,
    )
ax.set_xlabel("Mean cumulative action cost per query")
ax.set_ylabel("Mean NDCG@50")
ax.set_title("Panel A: Cost vs NDCG@50 (CQL highlighted)")
ax.grid(True, alpha=0.3)

ax = axes[1]
for _, row in summary.iterrows():
    policy = row["policy"]
    color = palette.get(policy, "black")
    ax.scatter(
        row["mean_cost"],
        row["mean_recall"],
        s=200,
        c=color,
        alpha=0.8,
        edgecolors="black",
        label=policy,
        zorder=3 if policy == "STOP" else 2,
    )
    ax.annotate(
        policy,
        (row["mean_cost"], row["mean_recall"]),
        textcoords="offset points",
        xytext=(8, 5),
        fontsize=9,
    )
ax.set_xlabel("Mean cumulative action cost per query")
ax.set_ylabel("Mean Recall@100")
ax.set_title("Panel B: Cost vs Recall@100 (STOP / BM25 highlighted)")
ax.grid(True, alpha=0.3)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=len(summary), bbox_to_anchor=(0.5, -0.05))

plt.tight_layout()
plt.savefig("../results/cost_effectiveness_plot.png", dpi=150, bbox_inches="tight")
print("Plot saved to ../results/cost_effectiveness_plot.png")
plt.close()
