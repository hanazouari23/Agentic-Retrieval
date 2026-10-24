r"""Train a Discrete CQL policy on expert trajectories with a BC warmstart.

This is research/diagnosis code. It does not overwrite train_cql.py.

Usage:
    poetry run python train_cql_with_BC_warmstart.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from d3rlpy.algos import DiscreteBCConfig, DiscreteCQLConfig
from d3rlpy.dataset import ReplayBuffer
from d3rlpy.dataset.buffers import InfiniteBuffer
from d3rlpy.models.encoders import VectorEncoderFactory

# -----------------------------
# Configuration
# -----------------------------
DATASET_PATH: str = "data/training/mdp_dataset_trajectories_expert_5k.h5"
MODEL_SAVE_DIR = Path("models")
BC_SAVE_PATH = str(MODEL_SAVE_DIR / "bc_expert_warmstart.pt")
MODEL_SAVE_PATH = str(MODEL_SAVE_DIR / "cql_on_full_expert_BC_warmstart.pt")
METRICS_SAVE_PATH = "train_cql_with_bc_warmstart_metrics.json"

RANDOM_SEED = 42
HIDDEN_DIMS = (256, 128)
BATCH_SIZE = 256
GAMMA = 0.99
LEARNING_RATE = 3e-4
N_CRITICS = 2
TARGET_UPDATE_INTERVAL = 8000
CQL_ALPHA = 1.0

BC_N_STEPS = 10_000
BC_N_STEPS_PER_EPOCH = 5_000

CQL_N_STEPS = 50_000
CQL_N_STEPS_PER_EPOCH = 10_000


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

    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Phase 1: Behavior Cloning
    # -----------------------------
    bc_config = DiscreteBCConfig(
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        encoder_factory=VectorEncoderFactory(hidden_units=list(HIDDEN_DIMS)),
    )
    bc_algo = bc_config.create(device=device)
    bc_algo.build_with_dataset(train_dataset)

    print(f"Pretraining BC for {BC_N_STEPS:,} steps ({BC_N_STEPS_PER_EPOCH:,} steps/epoch)")
    bc_metrics = bc_algo.fit(
        train_dataset,
        n_steps=BC_N_STEPS,
        n_steps_per_epoch=BC_N_STEPS_PER_EPOCH,
        experiment_name="bc_expert_warmstart",
        save_interval=BC_N_STEPS + 1,  # no intermediate saves
    )

    bc_algo.save_model(BC_SAVE_PATH)
    print(f"Saved BC model to {BC_SAVE_PATH}")

    # -----------------------------
    # Phase 2: CQL with BC-warmstarted encoders
    # -----------------------------
    cql_config = DiscreteCQLConfig(
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        learning_rate=LEARNING_RATE,
        encoder_factory=VectorEncoderFactory(hidden_units=list(HIDDEN_DIMS)),
        n_critics=N_CRITICS,
        target_update_interval=TARGET_UPDATE_INTERVAL,
        alpha=CQL_ALPHA,
    )
    cql_algo = cql_config.create(device=device)
    cql_algo.build_with_dataset(train_dataset)

    # Warmstart CQL Q-network encoders from the trained BC encoder.
    # This is critical for the experiment: if it fails, we crash rather than
    # silently training a randomly-initialized CQL model.
    bc_encoder = bc_algo.impl.modules.imitator._encoder
    cql_modules = cql_algo.impl.modules
    cql_q_funcs = list(cql_modules.q_funcs)
    cql_targ_q_funcs = list(cql_modules.targ_q_funcs)

    if len(cql_q_funcs) != N_CRITICS:
        raise RuntimeError(
            f"CQL has {len(cql_q_funcs)} online Q-functions, expected {N_CRITICS}. "
            "Cannot verify BC warmstart."
        )
    if len(cql_targ_q_funcs) != N_CRITICS:
        raise RuntimeError(
            f"CQL has {len(cql_targ_q_funcs)} target Q-functions, expected {N_CRITICS}. "
            "Cannot verify BC warmstart."
        )

    print(
        f"Copying BC encoder weights into {len(cql_q_funcs)} online Q-function encoders "
        f"and {len(cql_targ_q_funcs)} target Q-function encoders"
    )
    bc_encoder_state = bc_encoder.state_dict()
    for name, q_funcs in [("online", cql_q_funcs), ("target", cql_targ_q_funcs)]:
        for i, q_func in enumerate(q_funcs):
            missing, unexpected = q_func._encoder.load_state_dict(
                bc_encoder_state, strict=False
            )
            if missing or unexpected:
                raise RuntimeError(
                    f"BC->CQL encoder warmstart mismatch for {name} Q-function {i}: "
                    f"missing={missing}, unexpected={unexpected}"
                )
            print(f"  Warmstarted CQL {name} Q-function {i}")
    print("BC warmstart verified.")

    print(f"Training DiscreteCQL for {CQL_N_STEPS:,} steps ({CQL_N_STEPS_PER_EPOCH:,} steps/epoch)")
    cql_metrics = cql_algo.fit(
        train_dataset,
        n_steps=CQL_N_STEPS,
        n_steps_per_epoch=CQL_N_STEPS_PER_EPOCH,
        experiment_name="cql_on_full_expert_BC_warmstart",
        save_interval=CQL_N_STEPS + 1,  # no intermediate saves
    )

    cql_algo.save_model(MODEL_SAVE_PATH)
    print(f"Saved CQL model to {MODEL_SAVE_PATH}")

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
            "bc_n_steps": BC_N_STEPS,
            "bc_n_steps_per_epoch": BC_N_STEPS_PER_EPOCH,
            "cql_n_steps": CQL_N_STEPS,
            "cql_n_steps_per_epoch": CQL_N_STEPS_PER_EPOCH,
        },
        "data_stats": {
            "train_episodes": int(train_dataset.size()),
            "num_actions": int(num_actions),
        },
        "bc_epoch_metrics": [
            {"epoch": int(epoch), "metrics": {k: float(v) for k, v in m.items()}}
            for epoch, m in bc_metrics
        ],
        "cql_epoch_metrics": [
            {"epoch": int(epoch), "metrics": {k: float(v) for k, v in m.items()}}
            for epoch, m in cql_metrics
        ],
    }
    with open(METRICS_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Training metrics saved to {METRICS_SAVE_PATH}")


if __name__ == "__main__":
    main()
