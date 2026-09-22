r"""Train a Discrete CQL policy on simulated expert trajectories.

Usage:
    .venv\Scripts\python train_cql.py
    # or
    poetry run python train_cql.py
"""
from __future__ import annotations

import ast
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from d3rlpy.algos import DiscreteCQLConfig
from d3rlpy.constants import ActionSpace
from d3rlpy.dataset import MDPDataset
from d3rlpy.models.encoders import VectorEncoderFactory

# -----------------------------
# Configuration
# -----------------------------
TRAJECTORIES_PATH: str = "data/trajectories_expert_5k.csv"

RANDOM_SEED = 42
TEST_RATIO = 0.2
HIDDEN_DIMS = (256, 128)
BATCH_SIZE = 256
GAMMA = 0.99
LEARNING_RATE = 3e-4
N_CRITICS = 2
TARGET_UPDATE_INTERVAL = 8000
CQL_ALPHA = 1.0
N_STEPS = 50_000
N_STEPS_PER_EPOCH = 10_000

MODEL_SAVE_PATH = "cql_policy.pt"
METRICS_SAVE_PATH = "train_cql_metrics.json"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(path: str):
    df = pd.read_csv(path)
    states = np.vstack(df["state"].apply(ast.literal_eval).tolist())
    actions = df["action"].to_numpy(dtype=np.int64)
    rewards = df["reward"].to_numpy(dtype=np.float32)
    terminals = df["terminal"].to_numpy(dtype=np.float32)
    timeouts = df["timeout"].to_numpy(dtype=np.float32)
    query_ids = df["query_id"].to_numpy()
    return states, actions, rewards, terminals, timeouts, query_ids


def split_by_query(query_ids: np.ndarray, test_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    unique = np.unique(query_ids)
    rng.shuffle(unique)
    n_test = max(1, int(len(unique) * test_ratio))
    test_ids = set(unique[:n_test].tolist())
    train_mask = np.array([q not in test_ids for q in query_ids])
    return train_mask, test_ids


def build_dataset(
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    terminals: np.ndarray,
    timeouts: np.ndarray,
    num_actions: int,
) -> MDPDataset:
    return MDPDataset(
        observations=states.astype(np.float32),
        actions=actions.astype(np.int64),
        rewards=rewards.astype(np.float32),
        terminals=terminals.astype(np.float32),
        timeouts=timeouts.astype(np.float32),
        action_space=ActionSpace.DISCRETE,
        action_size=num_actions,
    )


def main():
    set_seed(RANDOM_SEED)
    states, actions, rewards, terminals, timeouts, query_ids = load_data(
        TRAJECTORIES_PATH
    )
    print(f"Loaded {len(states):,} transitions from {TRAJECTORIES_PATH}")
    print(f"State dim: {states.shape[1]}, actions: {sorted(np.unique(actions).tolist())}")

    train_mask, test_ids = split_by_query(query_ids, TEST_RATIO, RANDOM_SEED)
    num_actions = int(np.max(actions) + 1)

    train_dataset = build_dataset(
        states[train_mask],
        actions[train_mask],
        rewards[train_mask],
        terminals[train_mask],
        timeouts[train_mask],
        num_actions,
    )
    print(
        f"Train transitions: {len(actions[train_mask]):,}, "
        f"held-out test queries: {len(test_ids)}"
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    config = DiscreteCQLConfig(
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        learning_rate=LEARNING_RATE,
        encoder_factory=VectorEncoderFactory(hidden_units=list(HIDDEN_DIMS)),
        n_critics=N_CRITICS,
        target_update_interval=TARGET_UPDATE_INTERVAL,
        alpha=CQL_ALPHA,
    )
    algo = config.create(device=device)

    print(f"Training DiscreteCQL for {N_STEPS:,} steps ({N_STEPS_PER_EPOCH:,} steps/epoch)")
    epoch_metrics = algo.fit(
        train_dataset,
        n_steps=N_STEPS,
        n_steps_per_epoch=N_STEPS_PER_EPOCH,
        experiment_name="cql_policy",
        save_interval=10_000,  # no intermediate saves (total epochs < this)
    )

    algo.save_model(MODEL_SAVE_PATH)
    print(f"Saved model to {MODEL_SAVE_PATH}")

    metrics = {
        "config": {
            "trajectories_path": TRAJECTORIES_PATH,
            "random_seed": RANDOM_SEED,
            "test_ratio": TEST_RATIO,
            "hidden_dims": list(HIDDEN_DIMS),
            "batch_size": BATCH_SIZE,
            "gamma": GAMMA,
            "learning_rate": LEARNING_RATE,
            "n_critics": N_CRITICS,
            "target_update_interval": TARGET_UPDATE_INTERVAL,
            "cql_alpha": CQL_ALPHA,
            "n_steps": N_STEPS,
            "n_steps_per_epoch": N_STEPS_PER_EPOCH,
        },
        "data_stats": {
            "total_transitions": int(len(states)),
            "train_transitions": int(len(actions[train_mask])),
            "num_test_queries": len(test_ids),
            "num_actions": int(num_actions),
        },
        "epoch_metrics": [
            {"epoch": int(epoch), "metrics": {k: float(v) for k, v in m.items()}}
            for epoch, m in epoch_metrics
        ],
    }
    with open(METRICS_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Training metrics saved to {METRICS_SAVE_PATH}")


if __name__ == "__main__":
    main()
