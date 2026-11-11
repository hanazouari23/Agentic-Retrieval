r"""Train a Discrete CQL policy on simulated expert trajectories.

Usage:
    .venv\Scripts\python train_cql.py
    # or
    poetry run python train_cql.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from d3rlpy.algos import DiscreteCQLConfig
from d3rlpy.dataset import ReplayBuffer
from d3rlpy.dataset.buffers import InfiniteBuffer
from d3rlpy.models.encoders import VectorEncoderFactory

# -----------------------------
# Configuration
# -----------------------------
DATASET_PATH: str = "data/training/mdp_dataset_tree_branches_5000_terminal_reward.h5"
MODEL_SAVE_DIR = Path("models")
MODEL_SAVE_PATH = str(MODEL_SAVE_DIR / "cql_on_full_tree_terminal_reward_dataset.pt")
METRICS_SAVE_PATH = "train_cql_metrics_terminal_reward.json"

RANDOM_SEED = 42
HIDDEN_DIMS = (256, 128)
BATCH_SIZE = 256
GAMMA = 0.99
LEARNING_RATE = 3e-4
N_CRITICS = 2
TARGET_UPDATE_INTERVAL = 8000
CQL_ALPHA = 1.0
N_STEPS = 50_000
N_STEPS_PER_EPOCH = 10_000


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(path: str) -> ReplayBuffer:
    with open(path, "rb") as f:
        dataset = ReplayBuffer.load(f, InfiniteBuffer())
    print(f"Loaded {dataset.size():,} episodes from {path}")
    return dataset


def main():
    set_seed(RANDOM_SEED)

    train_dataset = load_data(DATASET_PATH)
    num_actions = int(max(int(ep.actions.max()) for ep in train_dataset.episodes) + 1)

    print(f"Action size: {num_actions}")
    print(f"Total episodes used for training: {train_dataset.size():,}")

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
        experiment_name="cql_on_full_tree_dataset",
        save_interval=10_000,  # no intermediate saves (total epochs < this)
    )

    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    algo.save_model(MODEL_SAVE_PATH)
    print(f"Saved model to {MODEL_SAVE_PATH}")

    metrics = {
        "config": {
            "dataset_path": DATASET_PATH,
            "random_seed": RANDOM_SEED,
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
            "train_episodes": int(train_dataset.size()),
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
