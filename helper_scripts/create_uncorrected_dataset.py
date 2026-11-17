r"""Create an uncorrected variant of a tree-branches HDF5 dataset.

All episodes are written to d3rlpy as terminal episodes (terminal=1 on the last
transition, timeout=0 everywhere).  This keeps the original rewards,
observations, and actions unchanged.

Usage:
    poetry run python helper_scripts/create_uncorrected_dataset.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
from d3rlpy.dataset import MDPDataset, ReplayBuffer
from d3rlpy.dataset.buffers import InfiniteBuffer

DEFAULT_INPUT_H5: Path = Path("data/training/mdp_dataset_tree_branches_5000.h5")
DEFAULT_OUTPUT_H5: Path = Path(
    "data/training/mdp_dataset_tree_branches_5000_uncorrected.h5"
)


def load_buffer(path: Path) -> ReplayBuffer:
    with open(path, "rb") as f:
        return ReplayBuffer.load(f, InfiniteBuffer())


def create_uncorrected_dataset(input_h5: Path, output_h5: Path) -> None:
    print(f"Loading dataset from {input_h5}")
    buffer = load_buffer(input_h5)
    print(f"  Episodes: {buffer.size():,}")

    all_observations: List[np.ndarray] = []
    all_actions: List[np.ndarray] = []
    all_rewards: List[np.ndarray] = []
    all_terminals: List[np.ndarray] = []
    all_timeouts: List[np.ndarray] = []

    for ep in buffer.episodes:
        ep_len = ep.size()
        all_observations.append(ep.observations)
        all_actions.append(ep.actions.reshape(-1))
        all_rewards.append(ep.rewards.reshape(-1))

        # Force terminal=1 on the last transition and timeout=0 everywhere.
        terminals = np.zeros(ep_len, dtype=np.float32)
        terminals[-1] = 1.0
        timeouts = np.zeros(ep_len, dtype=np.float32)

        all_terminals.append(terminals)
        all_timeouts.append(timeouts)

    observations = np.concatenate(all_observations, axis=0).astype(np.float32)
    actions = np.concatenate(all_actions, axis=0).astype(np.int64)
    rewards = np.concatenate(all_rewards, axis=0).astype(np.float32)
    terminals = np.concatenate(all_terminals, axis=0).astype(np.float32)
    timeouts = np.concatenate(all_timeouts, axis=0).astype(np.float32)

    print(f"Built arrays: obs={observations.shape}, actions={actions.shape}, rewards={rewards.shape}")
    print(f"  terminal flags sum: {terminals.sum():.0f} (should equal {buffer.size():,})")
    print(f"  timeout flags sum:  {timeouts.sum():.0f} (should be 0)")

    dataset = MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        timeouts=timeouts,
    )

    output_h5.parent.mkdir(parents=True, exist_ok=True)
    with open(output_h5, "w+b") as f:
        dataset.dump(f)
    print(f"Saved uncorrected dataset to {output_h5}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create an uncorrected variant of a tree-branches HDF5 dataset."
    )
    parser.add_argument(
        "--input-h5",
        type=Path,
        default=DEFAULT_INPUT_H5,
        help="Input HDF5 dataset path.",
    )
    parser.add_argument(
        "--output-h5",
        type=Path,
        default=DEFAULT_OUTPUT_H5,
        help="Output uncorrected HDF5 dataset path.",
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    create_uncorrected_dataset(args.input_h5, args.output_h5)
    return 0


if __name__ == "__main__":
    sys.exit(main())
