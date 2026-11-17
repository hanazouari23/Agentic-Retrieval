r"""Train a Discrete CQL policy on simulated expert trajectories.

Usage:
    poetry run python train_cql.py
    poetry run python train_cql.py --dataset data/training/mdp_dataset_tree_branches_5000_alpha15_corrected.h5 --model-name cql_alpha15_corrected
"""
from __future__ import annotations

import argparse
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
DEFAULT_DATASET_PATH: str = "data/training/mdp_dataset_tree_branches_5000_stop_penalty_coeff005.h5"
DEFAULT_MODEL_NAME: str = "cql_on_full_tree_dataset_5000_stop_penalty_coeff005"
DEFAULT_METRICS_PATH: str = "train_cql_metrics_stop_penalty_coeff005.json"
DEFAULT_EXPERIMENT_NAME: str = "cql_on_full_tree_dataset"

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Discrete CQL policy.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET_PATH,
        help="Path to the HDF5 dataset.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Base name for the saved model (e.g. cql_alpha15_corrected).",
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=list(HIDDEN_DIMS),
        help="Hidden layer dimensions for the encoder.",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--n-critics", type=int, default=N_CRITICS)
    parser.add_argument(
        "--target-update-interval", type=int, default=TARGET_UPDATE_INTERVAL
    )
    parser.add_argument("--cql-alpha", type=float, default=CQL_ALPHA)
    parser.add_argument("--n-steps", type=int, default=N_STEPS)
    parser.add_argument("--n-steps-per-epoch", type=int, default=N_STEPS_PER_EPOCH)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser


def resolve_paths(model_name: str | None) -> tuple[str, str, str]:
    """Return (model_save_path, metrics_save_path, experiment_name)."""
    if model_name is None:
        # Backward-compatible defaults when no --model-name is supplied.
        return (
            str(Path("models") / f"{DEFAULT_MODEL_NAME}.pt"),
            DEFAULT_METRICS_PATH,
            DEFAULT_EXPERIMENT_NAME,
        )

    model_save_path = str(Path("models") / f"{model_name}.pt")
    suffix = model_name.removeprefix("cql_")
    metrics_dir = Path("training") / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_save_path = str(metrics_dir / f"train_cql_metrics_{suffix}.json")
    experiment_name = model_name
    return model_save_path, metrics_save_path, experiment_name


def main():
    parser = build_parser()
    args = parser.parse_args()

    set_seed(args.seed)

    model_save_path, metrics_save_path, experiment_name = resolve_paths(args.model_name)
    model_name = args.model_name or DEFAULT_MODEL_NAME

    train_dataset = load_data(args.dataset)
    num_actions = int(max(int(ep.actions.max()) for ep in train_dataset.episodes) + 1)

    print(f"Action size: {num_actions}")
    print(f"Total episodes used for training: {train_dataset.size():,}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    config = DiscreteCQLConfig(
        batch_size=args.batch_size,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        encoder_factory=VectorEncoderFactory(hidden_units=list(args.hidden_dims)),
        n_critics=args.n_critics,
        target_update_interval=args.target_update_interval,
        alpha=args.cql_alpha,
    )
    algo = config.create(device=device)

    print(f"Training DiscreteCQL for {args.n_steps:,} steps ({args.n_steps_per_epoch:,} steps/epoch)")
    epoch_metrics = algo.fit(
        train_dataset,
        n_steps=args.n_steps,
        n_steps_per_epoch=args.n_steps_per_epoch,
        experiment_name=experiment_name,
        save_interval=10_000,  # no intermediate saves (total epochs < this)
    )

    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
    algo.save_model(model_save_path)
    print(f"Saved model to {model_save_path}")

    metrics = {
        "config": {
            "dataset_path": args.dataset,
            "model_name": model_name,
            "random_seed": args.seed,
            "hidden_dims": list(args.hidden_dims),
            "batch_size": args.batch_size,
            "gamma": args.gamma,
            "learning_rate": args.learning_rate,
            "n_critics": args.n_critics,
            "target_update_interval": args.target_update_interval,
            "cql_alpha": args.cql_alpha,
            "n_steps": args.n_steps,
            "n_steps_per_epoch": args.n_steps_per_epoch,
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
    with open(metrics_save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Training metrics saved to {metrics_save_path}")


if __name__ == "__main__":
    main()
