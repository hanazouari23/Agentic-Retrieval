r"""Relabel rewards in a tree-branches HDF5 dataset using new SimConfig weights.

The expensive part of dataset generation (retrieval, agent calls, embeddings) is
already done and stored in the HDF5 and CSV. This script only recomputes the
reward scalar for each transition using the logged NDCG/Recall/cost/latency
values, then writes a new HDF5 with the updated rewards.

Usage:
    .venv\Scripts\python relabel_tree_rewards.py \
        --input-h5 data/training/mdp_dataset_tree_branches_5000.h5 \
        --input-csv data/training/trajectories_tree_branches_5000_with_query.csv \
        --output-h5 data/training/mdp_dataset_tree_branches_5000_alpha15.h5 \
        --reward-alpha 1.5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from d3rlpy.dataset import MDPDataset, ReplayBuffer
from d3rlpy.dataset.buffers import InfiniteBuffer

# Default reward hyperparameters matching src/simulation.SimConfig.
DEFAULT_REWARD_ALPHA = 2.0
DEFAULT_REWARD_BETA = 0.5
DEFAULT_REWARD_GAMMA = 0.2
DEFAULT_REWARD_DELTA = 0.1
DEFAULT_ELAPSED_TIME_NORM = 3000.0


def load_buffer(path: str) -> ReplayBuffer:
    with open(path, "rb") as f:
        return ReplayBuffer.load(f, InfiniteBuffer())


def recompute_reward(
    row: pd.Series,
    reward_alpha: float,
    reward_beta: float,
    reward_gamma: float,
    reward_delta: float,
    elapsed_time_norm: float,
) -> float:
    """Recompute reward from logged CSV columns.

    Mirrors src.simulation.Simulation._compute_reward.
    STOP actions always receive 0 reward.
    """
    action_name = row["action_name"]
    if action_name == "STOP":
        return 0.0

    ndcg_gain = float(row["ndcg_after"]) - float(row["ndcg_before"])
    recall_gain = float(row["recall_after"]) - float(row["recall_before"])
    action_cost = float(row["cost"]) if pd.notna(row["cost"]) else 0.0
    elapsed_ms = float(row["latency_ms"]) if pd.notna(row["latency_ms"]) else 0.0

    quality_reward = reward_alpha * ndcg_gain
    recall_reward = reward_beta * recall_gain
    cost_penalty = reward_gamma * action_cost
    time_penalty = reward_delta * (elapsed_ms / max(elapsed_time_norm, 1e-12))

    return float(quality_reward + recall_reward - cost_penalty - time_penalty)


def relabel_rewards(
    input_h5: Path,
    input_csv: Path,
    output_h5: Path,
    reward_alpha: float,
    reward_beta: float,
    reward_gamma: float,
    reward_delta: float,
    elapsed_time_norm: float,
) -> None:
    print(f"Loading HDF5: {input_h5}")
    buffer = load_buffer(str(input_h5))
    print(f"  Episodes: {buffer.size():,}")

    total_transitions = sum(ep.size() for ep in buffer.episodes)
    print(f"  Transitions: {total_transitions:,}")

    print(f"Loading CSV: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"  Rows: {len(df):,}")

    if len(df) != total_transitions:
        raise ValueError(
            f"Mismatch: {total_transitions} transitions in HDF5 vs {len(df)} rows in CSV"
        )

    # Ensure the CSV contains the numeric columns needed for reward recomputation.
    required_cols = ["ndcg_before", "ndcg_after", "recall_before", "recall_after", "cost", "latency_ms"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV is missing required column: {col}")
        if df[col].isna().all():
            raise ValueError(
                f"Column {col} is entirely NaN. "
                "Use a CSV variant that logs NDCG/Recall/cost/latency, "
                "e.g. trajectories_tree_branches_5000_terminal_reward.csv"
            )

    # Compute new rewards from CSV columns.
    print("Recomputing rewards...")
    new_rewards = df.apply(
        lambda row: recompute_reward(
            row,
            reward_alpha=reward_alpha,
            reward_beta=reward_beta,
            reward_gamma=reward_gamma,
            reward_delta=reward_delta,
            elapsed_time_norm=elapsed_time_norm,
        ),
        axis=1,
    ).astype(np.float32).values

    # Build episode-level arrays from the original buffer.
    all_observations: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    all_rewards: list[np.ndarray] = []
    all_terminals: list[np.ndarray] = []
    all_timeouts: list[np.ndarray] = []

    csv_idx = 0
    for ep in buffer.episodes:
        ep_len = ep.size()
        ep_obs = ep.observations
        ep_actions = ep.actions.reshape(-1)

        # The CSV timeout column is not reliable for the base tree dataset
        # (generate_tree_branches_csv.py sets timeout=0 for all rows). Use the
        # episode's terminated flag from the HDF5 instead.
        ep_terminals = np.zeros(ep_len, dtype=np.float32)
        ep_timeouts = np.zeros(ep_len, dtype=np.float32)
        if ep.terminated:
            ep_terminals[-1] = 1.0
        else:
            ep_timeouts[-1] = 1.0

        all_observations.append(ep_obs)
        all_actions.append(ep_actions)
        all_rewards.append(new_rewards[csv_idx : csv_idx + ep_len])
        all_terminals.append(ep_terminals)
        all_timeouts.append(ep_timeouts)

        csv_idx += ep_len

    observations = np.concatenate(all_observations, axis=0).astype(np.float32)
    actions = np.concatenate(all_actions, axis=0).astype(np.int64)
    rewards = np.concatenate(all_rewards, axis=0).astype(np.float32)
    terminals = np.concatenate(all_terminals, axis=0).astype(np.float32)
    timeouts = np.concatenate(all_timeouts, axis=0).astype(np.float32)

    print(f"Built arrays: obs={observations.shape}, actions={actions.shape}, rewards={rewards.shape}")

    dataset = MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        timeouts=timeouts,
    )

    output_h5.parent.mkdir(parents=True, exist_ok=True)
    # h5py needs read/write access during dump, so use w+b mode.
    with open(output_h5, "w+b") as f:
        dataset.dump(f)
    print(f"Saved relabeled dataset to {output_h5}")

    # Print reward statistics.
    old_rewards = df["reward"].astype(np.float32).values
    print("\nReward statistics:")
    print(f"  Old mean: {old_rewards.mean():.6f}, std: {old_rewards.std():.6f}")
    print(f"  New mean: {rewards.mean():.6f}, std: {rewards.std():.6f}")
    print(f"  Old min/max: {old_rewards.min():.6f} / {old_rewards.max():.6f}")
    print(f"  New min/max: {rewards.min():.6f} / {rewards.max():.6f}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Relabel rewards in a tree-branches HDF5 using new SimConfig weights."
    )
    parser.add_argument("--input-h5", type=Path, required=True,
                        help="Input HDF5 dataset path.")
    parser.add_argument("--input-csv", type=Path, required=True,
                        help="Input CSV with logged NDCG/Recall/cost/latency per transition.")
    parser.add_argument("--output-h5", type=Path, required=True,
                        help="Output HDF5 dataset path.")
    parser.add_argument("--reward-alpha", type=float, default=DEFAULT_REWARD_ALPHA,
                        help="Weight for NDCG gain (default: 2.0).")
    parser.add_argument("--reward-beta", type=float, default=DEFAULT_REWARD_BETA,
                        help="Weight for Recall gain (default: 0.5).")
    parser.add_argument("--reward-gamma", type=float, default=DEFAULT_REWARD_GAMMA,
                        help="Weight for action cost penalty (default: 0.2).")
    parser.add_argument("--reward-delta", type=float, default=DEFAULT_REWARD_DELTA,
                        help="Weight for latency penalty (default: 0.1).")
    parser.add_argument("--elapsed-time-norm", type=float, default=DEFAULT_ELAPSED_TIME_NORM,
                        help="Elapsed time normalization divisor in ms (default: 3000.0).")
    args = parser.parse_args(argv)

    relabel_rewards(
        input_h5=args.input_h5,
        input_csv=args.input_csv,
        output_h5=args.output_h5,
        reward_alpha=args.reward_alpha,
        reward_beta=args.reward_beta,
        reward_gamma=args.reward_gamma,
        reward_delta=args.reward_delta,
        elapsed_time_norm=args.elapsed_time_norm,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
