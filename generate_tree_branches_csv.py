#!/usr/bin/env python3
"""
Convert the tree-branches MDP datasets in ``data/training`` from HDF5 to CSV.

The HDF5 files are d3rlpy ``MDPDataset`` dumps. The newer d3rlpy API exposes
 them as ``ReplayBuffer`` episodes. This script walks every episode/transition
and writes the fields that are still recoverable from the stored state vectors.

Recoverable fields:
  * action / action_name
  * reward
  * terminal (inferred from episode-level ``terminated`` flag)
  * step (recovered from the normalised step feature in the state)
  * query_length, score_spread, score_entropy
  * rank_overlap, query_drift
  * cumulative_cost, normalised elapsed time
  * last-action one-hot and valid-action mask

Fields that were present in the original trajectory CSVs but are *not* stored
in the HDF5 (query_id, raw query text, NDCG/Recall, latency, etc.) are emitted
as empty columns so the schema stays aligned with the existing CSVs.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from pathlib import Path
from typing import List

import numpy as np

# d3rlpy is noisy on import; keep the console readable.
logging.basicConfig(level=logging.WARNING)
os.environ.setdefault("D3RLPY_LOGGING_LEVEL", "ERROR")

from d3rlpy.dataset import ReplayBuffer  # noqa: E402
from d3rlpy.dataset.buffers import InfiniteBuffer  # noqa: E402


ACTION_NAMES = {
    0: "QueryReform",
    1: "Rerank",
    2: "PseudoRelevanceFeedback",
    3: "STOP",
}

# State vector indices taken from src/simulation.py::build_state.
IDX_QUERY_LENGTH = 0
IDX_SCORE_SPREAD = 385
IDX_SCORE_ENTROPY = 386
IDX_LAST_ACTION_START = 387
IDX_NORMALIZED_STEP = 390
IDX_RANK_OVERLAP = 391
IDX_QUERY_DRIFT = 392
IDX_NORMALIZED_ELAPSED = 393
IDX_CUMULATIVE_COST = 394
IDX_VALID_MASK_START = 395

CSV_FIELDS = [
    "query_id",
    "query",
    "new_query",
    "action_name",
    "query_length",
    "score_spread",
    "score_entropy",
    "step",
    "rank_overlap",
    "query_drift",
    "ndcg_before",
    "ndcg_after",
    "recall_before",
    "recall_after",
    "cost",
    "latency_ms",
    "cumulative_cost",
    "cumulative_latency_ms",
    "reward",
    "terminal",
    "timeout",
]


def _clean(value: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if not np.isfinite(v) else v


def _recover_step(normalized_step: float, max_steps: int) -> int:
    """Return a 1-based step number from the normalised step feature."""
    raw = int(round(_clean(normalized_step) * max_steps))
    return max(1, raw + 1)


def h5_to_csv(h5_path: Path, csv_path: Path, max_steps: int = 3, force: bool = False) -> int:
    """Convert one HDF5 MDPDataset to CSV. Returns the number of rows written."""
    if csv_path.exists() and not force:
        print(f"  skipping {csv_path.name} (already exists; use --force to overwrite)")
        return 0

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = csv_path.with_suffix(".tmp")

    rows_written = 0
    with open(h5_path, "rb") as f:
        buffer = ReplayBuffer.load(f, InfiniteBuffer())

    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for episode_idx, episode in enumerate(buffer.episodes):
            obs = episode.observations
            actions = episode.actions.reshape(-1)
            rewards = episode.rewards.reshape(-1)
            episode_len = episode.size()

            for i in range(episode_len):
                state = obs[i]
                is_last = i == episode_len - 1
                terminal = 1 if (is_last and episode.terminated) else 0
                # Timeouts are not preserved in the new d3rlpy episode format.
                timeout = 0

                row = {
                    "query_id": "",
                    "query": "",
                    "new_query": "",
                    "action_name": ACTION_NAMES.get(int(actions[i]), ""),
                    "query_length": _clean(state[IDX_QUERY_LENGTH]),
                    "score_spread": _clean(state[IDX_SCORE_SPREAD]),
                    "score_entropy": _clean(state[IDX_SCORE_ENTROPY]),
                    "step": _recover_step(state[IDX_NORMALIZED_STEP], max_steps),
                    "rank_overlap": _clean(state[IDX_RANK_OVERLAP]),
                    "query_drift": _clean(state[IDX_QUERY_DRIFT]),
                    "ndcg_before": "",
                    "ndcg_after": "",
                    "recall_before": "",
                    "recall_after": "",
                    "cost": "",
                    "latency_ms": "",
                    "cumulative_cost": _clean(state[IDX_CUMULATIVE_COST]),
                    "cumulative_latency_ms": "",
                    "reward": _clean(rewards[i]),
                    "terminal": terminal,
                    "timeout": timeout,
                }
                writer.writerow(row)
                rows_written += 1

            if (episode_idx + 1) % 2000 == 0:
                print(f"  processed {episode_idx + 1} episodes...")

    os.replace(tmp_path, csv_path)
    return rows_written


def find_tree_branch_variants(directory: Path) -> List[Path]:
    """Return the four tree-branches HDF5 datasets sorted by filename."""
    pattern = "mdp_dataset_tree_branches_5000*.h5"
    files = sorted(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No files matched '{pattern}' in {directory}"
        )
    return files


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert tree-branches HDF5 datasets to CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/training"),
        help="Directory containing the tree-branches HDF5 files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for CSV output. Defaults to --input-dir.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=3,
        help="Maximum steps per trajectory used during dataset generation.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing CSV files instead of skipping them.",
    )
    args = parser.parse_args(argv)

    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir

    h5_files = find_tree_branch_variants(input_dir)
    print(f"Found {len(h5_files)} tree-branches variant(s) in {input_dir}")

    for h5_path in h5_files:
        suffix = h5_path.stem.replace("mdp_dataset_tree_branches_5000", "").lstrip("_")
        csv_name = (
            f"trajectories_tree_branches_5000_{suffix}.csv"
            if suffix
            else "trajectories_tree_branches_5000.csv"
        )
        csv_path = output_dir / csv_name

        print(f"\n{h5_path.name} -> {csv_path.name}")
        rows = h5_to_csv(
            h5_path, csv_path, max_steps=args.max_steps, force=args.force
        )
        if rows:
            print(f"  wrote {rows:,} transition rows")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
