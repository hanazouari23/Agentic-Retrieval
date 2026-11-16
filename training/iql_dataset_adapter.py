"""IQL dataset adapter for the tree-branches MDP dataset.

Loads `mdp_dataset_tree_branches_5000.h5` and extracts transitions with
`d3rlpy`'s default `BasicTransitionPicker`, exactly matching how the existing
CQL dataloader samples the data.  Query IDs are attached from the aligned
`trajectories_tree_branches_5000_terminal_reward.csv` metadata file, but the
reward values come from the HDF5 (i.e. the same rewards CQL sees).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from d3rlpy.dataset import ReplayBuffer
from d3rlpy.dataset.buffers import InfiniteBuffer

# Keep d3rlpy quiet on import.
os.environ.setdefault("D3RLPY_LOGGING_LEVEL", "ERROR")

DATASET_PATH: str = "data/training/mdp_dataset_tree_branches_5000.h5"
QUERY_CSV_PATH: str = "data/training/trajectories_tree_branches_5000_terminal_reward.csv"

ACTION_NAMES = ["QueryReform", "Rerank", "PseudoRelevanceFeedback", "STOP"]


def load_buffer(path: str) -> ReplayBuffer:
    with open(path, "rb") as f:
        return ReplayBuffer.load(f, InfiniteBuffer())


def _verify_alignment(buffer: ReplayBuffer, df: pd.DataFrame) -> None:
    """Check that HDF5 episodes and CSV rows are in the same order."""
    if buffer.size() == 0 or len(df) == 0:
        raise ValueError("Empty dataset")

    csv_idx = 0
    for ep_idx, ep in enumerate(buffer.episodes):
        ep_len = ep.size()
        for t in range(ep_len):
            expected_action = ACTION_NAMES[int(ep.actions[t, 0])]
            actual_action = str(df.iloc[csv_idx]["action_name"])
            if expected_action != actual_action:
                raise ValueError(
                    f"Action mismatch at episode {ep_idx}, step {t} "
                    f"(CSV row {csv_idx}): HDF5={expected_action}, CSV={actual_action}"
                )
            csv_idx += 1

    if csv_idx != len(df):
        raise ValueError(
            f"CSV/HDF5 length mismatch: {csv_idx} HDF5 rows vs {len(df)} CSV rows"
        )

    print(f"Alignment verified: {buffer.size():,} episodes, {len(df):,} CSV rows")


def build_iql_dataset(
    h5_path: str = DATASET_PATH,
    csv_path: str | None = QUERY_CSV_PATH,
) -> Dict[str, np.ndarray]:
    """Build a sampleable transition dataset matching CQL's picker exactly.

    ``csv_path`` is optional. When provided, the CSV is used only for
    alignment verification and metadata (``query_ids``, ``row_indices``).
    Rewards, terminals, and observations always come from the HDF5.

    Returns a dictionary with:
      - observations        (N, 399) float32
      - actions             (N,)     int64
      - rewards             (N,)     float32  (from the HDF5)
      - next_observations   (N, 399) float32
      - terminals           (N,)     float32
      - action_masks        (N, 4)   float32  (observations[:, 395:399])
      - query_ids           (N,)     int64    (only if csv_path is given)
      - row_indices         (N,)     int64    (only if csv_path is given)
    """
    buffer = load_buffer(h5_path)
    df: pd.DataFrame | None = None
    if csv_path:
        df = pd.read_csv(csv_path)
        _verify_alignment(buffer, df)

    picker = buffer.transition_picker
    expected_count = buffer.transition_count

    observations: list[np.ndarray] = []
    next_observations: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    terminals: list[float] = []
    action_masks: list[np.ndarray] = []
    query_ids: list[int] = []
    row_indices: list[int] = []

    csv_idx = 0
    for ep_idx, ep in enumerate(buffer.episodes):
        for t in range(ep.transition_count):
            transition = picker(ep, t)

            obs = transition.observation
            next_obs = transition.next_observation

            observations.append(obs)
            next_observations.append(next_obs)
            actions.append(int(transition.action[0]))
            rewards.append(float(transition.reward[0]))
            terminals.append(float(transition.terminal))
            action_masks.append(obs[395:399].copy())
            if df is not None:
                query_ids.append(int(df.iloc[csv_idx]["query_id"]))
                row_indices.append(csv_idx)

            csv_idx += 1

        # d3rlpy excludes the final transition of timeout episodes.
        if df is not None and not ep.terminated and ep.transition_count < ep.size():
            csv_idx += 1

    dataset: Dict[str, np.ndarray] = {
        "observations": np.stack(observations).astype(np.float32),
        "actions": np.array(actions, dtype=np.int64),
        "rewards": np.array(rewards, dtype=np.float32),
        "next_observations": np.stack(next_observations).astype(np.float32),
        "terminals": np.array(terminals, dtype=np.float32),
        "action_masks": np.stack(action_masks).astype(np.float32),
    }
    if df is not None:
        dataset["query_ids"] = np.array(query_ids, dtype=np.int64)
        dataset["row_indices"] = np.array(row_indices, dtype=np.int64)

    if dataset["observations"].shape[0] != expected_count:
        raise ValueError(
            f"Transition count mismatch: got {dataset['observations'].shape[0]}, "
            f"expected {expected_count}"
        )

    return dataset


def verify_against_picker(
    dataset: Dict[str, np.ndarray], buffer: ReplayBuffer
) -> None:
    """Re-check every array entry against the live BasicTransitionPicker."""
    picker = buffer.transition_picker
    pos = 0
    for ep_idx, ep in enumerate(buffer.episodes):
        for t in range(ep.transition_count):
            tr = picker(ep, t)
            np.testing.assert_allclose(
                dataset["observations"][pos], tr.observation, rtol=1e-6, atol=1e-6
            )
            np.testing.assert_allclose(
                dataset["next_observations"][pos],
                tr.next_observation,
                rtol=1e-6,
                atol=1e-6,
            )
            assert dataset["actions"][pos] == int(tr.action[0])
            assert np.isclose(dataset["rewards"][pos], float(tr.reward[0]))
            assert np.isclose(dataset["terminals"][pos], float(tr.terminal))
            pos += 1

    if pos != len(dataset["observations"]):
        raise ValueError(f"Verification length mismatch: {pos} vs {len(dataset['observations'])}")

    print(f"Verified {pos:,} transitions against BasicTransitionPicker.")


def main() -> None:
    print("Building IQL dataset...")
    dataset = build_iql_dataset()

    print(f"Observations shape: {dataset['observations'].shape}")
    print(f"Actions shape:      {dataset['actions'].shape}")
    print(f"Rewards shape:      {dataset['rewards'].shape}")
    print(f"Next obs shape:     {dataset['next_observations'].shape}")
    print(f"Terminals shape:    {dataset['terminals'].shape}")
    print(f"Action masks shape: {dataset['action_masks'].shape}")
    print(f"Query IDs shape:    {dataset['query_ids'].shape}")
    print(f"Row indices shape:  {dataset['row_indices'].shape}")
    print(f"Terminal flags sum: {dataset['terminals'].sum():.0f}")
    print(f"Unique queries:     {np.unique(dataset['query_ids']).size:,}")

    buffer = load_buffer(DATASET_PATH)
    print("Running picker verification...")
    verify_against_picker(dataset, buffer)
    print("Done.")


if __name__ == "__main__":
    main()
