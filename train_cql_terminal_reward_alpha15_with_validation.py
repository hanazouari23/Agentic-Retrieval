r"""Train a Discrete CQL policy on the terminal_reward tree dataset with reward_alpha=1.5.

Uses a train/validation split and early stopping. No BC warm-start.

Usage:
    .venv\Scripts\python train_cql_terminal_reward_alpha15_with_validation.py
    # or
    poetry run python train_cql_terminal_reward_alpha15_with_validation.py
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import List

import numpy as np
import torch
from d3rlpy.algos import DiscreteCQLConfig
from d3rlpy.dataset import ReplayBuffer
from d3rlpy.dataset.buffers import InfiniteBuffer
from d3rlpy.dataset.components import EpisodeBase
from d3rlpy.metrics import TDErrorEvaluator
from d3rlpy.models.encoders import VectorEncoderFactory

# -----------------------------
# Configuration
# -----------------------------
DATASET_PATH: str = "data/training/mdp_dataset_tree_branches_5000_terminal_reward_alpha05.h5"
QUERY_CSV_PATH: str = "data/training/trajectories_tree_branches_5000_terminal_reward.csv"
MODEL_SAVE_DIR = Path("models")
MODEL_SAVE_PATH = str(MODEL_SAVE_DIR / "cql_terminal_reward_alpha05_val_64_32.pt")
BEST_MODEL_SAVE_PATH = str(MODEL_SAVE_DIR / "cql_terminal_reward_alpha05_val_64_32_best.pt")
METRICS_SAVE_PATH = "train_cql_terminal_reward_alpha05_with_validation_metrics.json"

RANDOM_SEED = 42
HIDDEN_DIMS = (64, 32)
BATCH_SIZE = 256
GAMMA = 0.99
LEARNING_RATE = 3e-4
N_CRITICS = 2
TARGET_UPDATE_INTERVAL = 8000
CQL_ALPHA = 1.0
N_STEPS = 50_000
N_STEPS_PER_EPOCH = 10_000
VAL_SPLIT = 0.1  # fraction of queries held out for validation
EARLY_STOPPING_PATIENCE = 2  # stop if validation TD error does not improve for N epochs


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


def map_episodes_to_query_ids(csv_path: str) -> List[str]:
    """Return the query_id for each episode/branch in HDF5 episode order.

    The CSV rows are written in the same order as the HDF5 episodes. Each
    episode corresponds to one branch (trajectory), which starts at step==1.
    """
    query_ids: List[str] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["step"]) == 1:
                query_ids.append(row["query_id"])
    return query_ids


def split_episodes_by_query(
    episodes: List[EpisodeBase],
    episode_query_ids: List[str],
    val_fraction: float,
    seed: int,
) -> tuple[List[EpisodeBase], List[EpisodeBase], set[str], set[str]]:
    """Split episodes so all branches from one query stay in one split."""
    rng = np.random.default_rng(seed)
    unique_qids = list(dict.fromkeys(episode_query_ids))  # preserve order
    shuffled_qids = unique_qids.copy()
    rng.shuffle(shuffled_qids)

    n_val = max(1, int(len(shuffled_qids) * val_fraction))
    val_qids = set(shuffled_qids[:n_val])

    train_eps = [
        ep for ep, qid in zip(episodes, episode_query_ids) if qid not in val_qids
    ]
    val_eps = [
        ep for ep, qid in zip(episodes, episode_query_ids) if qid in val_qids
    ]
    train_qids = set(unique_qids) - val_qids
    return train_eps, val_eps, train_qids, val_qids


def build_buffer_from_episodes(episodes: List[EpisodeBase]) -> ReplayBuffer:
    """Create a new ReplayBuffer containing only the given episodes."""
    return ReplayBuffer(
        buffer=InfiniteBuffer(),
        episodes=episodes,
        transition_picker=None,
        trajectory_slicer=None,
        writer_preprocessor=None,
    )


def main():
    set_seed(RANDOM_SEED)

    full_dataset = load_data(DATASET_PATH)
    num_actions = int(max(int(ep.actions.max()) for ep in full_dataset.episodes) + 1)

    episode_query_ids = map_episodes_to_query_ids(QUERY_CSV_PATH)
    if len(episode_query_ids) != len(full_dataset.episodes):
        raise ValueError(
            f"Episode/query mismatch: {len(full_dataset.episodes)} episodes "
            f"vs {len(episode_query_ids)} query_ids from CSV"
        )

    train_eps, val_eps, train_qids, val_qids = split_episodes_by_query(
        full_dataset.episodes, episode_query_ids, VAL_SPLIT, RANDOM_SEED
    )
    train_dataset = build_buffer_from_episodes(train_eps)
    val_dataset = build_buffer_from_episodes(val_eps)

    print(f"Action size: {num_actions}")
    print(f"Training queries: {len(train_qids):,}")
    print(f"Validation queries: {len(val_qids):,}")
    print(f"Training episodes: {train_dataset.size():,}")
    print(f"Validation episodes: {val_dataset.size():,}")

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

    print(f"Training DiscreteCQL for up to {N_STEPS:,} steps ({N_STEPS_PER_EPOCH:,} steps/epoch)")
    print(f"Early stopping patience: {EARLY_STOPPING_PATIENCE} epoch(s)")

    # Validation evaluator. It will evaluate on val_eps regardless of the
    # dataset passed to __call__ (we pass train_dataset only to provide the
    # transition picker).
    val_evaluator = TDErrorEvaluator(episodes=val_eps)
    best_val_td_error = float("inf")
    epochs_without_improvement = 0
    should_stop = False

    def epoch_callback(algo, epoch, total_step):
        nonlocal best_val_td_error, epochs_without_improvement, should_stop
        val_td_error = val_evaluator(algo, train_dataset)
        print(f"  Epoch {epoch} validation TD error: {val_td_error:.6f}")

        if val_td_error < best_val_td_error:
            best_val_td_error = val_td_error
            epochs_without_improvement = 0
            MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
            algo.save_model(BEST_MODEL_SAVE_PATH)
            print(f"  -> New best model saved (TD error {best_val_td_error:.6f})")
        else:
            epochs_without_improvement += 1
            print(f"  -> No improvement for {epochs_without_improvement} epoch(s)")
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                should_stop = True
                print(f"  -> Early stopping triggered after epoch {epoch}")

    epoch_metrics = []
    for epoch, metrics in algo.fitter(
        train_dataset,
        n_steps=N_STEPS,
        n_steps_per_epoch=N_STEPS_PER_EPOCH,
        experiment_name="cql_terminal_reward_alpha15_with_validation_64_32",
        save_interval=10_000,  # no intermediate saves (total epochs < this)
        epoch_callback=epoch_callback,
    ):
        epoch_metrics.append((epoch, metrics))
        if should_stop:
            break

    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    algo.save_model(MODEL_SAVE_PATH)
    print(f"Saved final model to {MODEL_SAVE_PATH}")
    print(f"Saved best validation model to {BEST_MODEL_SAVE_PATH}")

    metrics = {
        "config": {
            "dataset_path": DATASET_PATH,
            "query_csv_path": QUERY_CSV_PATH,
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
            "val_split": VAL_SPLIT,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        },
        "data_stats": {
            "total_episodes": int(full_dataset.size()),
            "train_episodes": int(train_dataset.size()),
            "val_episodes": int(val_dataset.size()),
            "train_queries": len(train_qids),
            "val_queries": len(val_qids),
            "num_actions": int(num_actions),
        },
        "epoch_metrics": [
            {"epoch": int(epoch), "metrics": {k: float(v) for k, v in m.items()}}
            for epoch, m in epoch_metrics
        ],
        "best_val_td_error": float(best_val_td_error),
    }
    with open(METRICS_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Training metrics saved to {METRICS_SAVE_PATH}")


if __name__ == "__main__":
    main()
