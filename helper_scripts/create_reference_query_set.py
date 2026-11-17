r"""Create a fixed 200-query MS MARCO dev reference set.

Keeps the existing 50 query IDs from the legacy CQL traces and adds 150 new
judged queries selected deterministically with seed 42.

Usage:
    poetry run python helper_scripts/create_reference_query_set.py
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import List, Set

import pandas as pd

DEFAULT_TOPICS_PATH: Path = Path("data/topics.ms-marco-dev.tsv")
DEFAULT_QRELS_PATH: Path = Path("data/qrels.ms-marco-dev.tsv")
DEFAULT_EXISTING_TRACES_PATH: Path = Path(
    "RQs/relevant_csvs/cql_full_tree_traces_50_msmarco_dev.csv"
)
DEFAULT_OUTPUT_PATH: Path = Path("RQs/relevant_csvs/msmarco_dev_reference_200.csv")
RANDOM_SEED = 42


def load_judged_query_ids(topics_path: Path, qrels_path: Path) -> List[str]:
    """Return judged query IDs in the order they appear in the topics file."""
    qrels_df = pd.read_csv(qrels_path, sep="\t", header=0, dtype=str)
    judged: Set[str] = set(qrels_df["query_id"].astype(str).unique())

    topics_df = pd.read_csv(topics_path, sep="\t", header=0, dtype=str)
    topics_df["query_id"] = topics_df["query_id"].astype(str)

    ordered = [
        str(qid)
        for qid in topics_df["query_id"]
        if qid in judged
    ]
    return ordered


def load_existing_query_ids(traces_path: Path) -> List[str]:
    """Return ordered unique query IDs from an existing trace/reference CSV."""
    df = pd.read_csv(traces_path, dtype=str)
    ids = [str(qid) for qid in df["query_id"].unique()]
    return ids


def create_reference_set(
    topics_path: Path,
    qrels_path: Path,
    existing_traces_path: Path,
    output_path: Path,
    seed: int = RANDOM_SEED,
) -> None:
    print(f"Loading judged queries from {topics_path} + {qrels_path}")
    judged_ordered = load_judged_query_ids(topics_path, qrels_path)
    judged_set: Set[str] = set(judged_ordered)
    print(f"  Total judged queries: {len(judged_set)}")

    print(f"Loading existing reference IDs from {existing_traces_path}")
    existing_ids = load_existing_query_ids(existing_traces_path)
    print(f"  Existing reference IDs: {len(existing_ids)}")

    # Validate that existing IDs are judged.
    missing = [qid for qid in existing_ids if qid not in judged_set]
    if missing:
        raise ValueError(
            f"Existing reference contains {len(missing)} query IDs not in qrels: {missing[:5]}"
        )

    # Remove existing IDs from the pool and sample 150 more.
    remaining = [qid for qid in judged_ordered if qid not in existing_ids]
    if len(remaining) < 150:
        raise ValueError(
            f"Only {len(remaining)} new judged queries available, need 150"
        )

    rng = random.Random(seed)
    new_ids = rng.sample(remaining, 150)

    reference_ids = existing_ids + new_ids
    assert len(reference_ids) == 200
    assert len(set(reference_ids)) == 200

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"query_id": reference_ids}).to_csv(output_path, index=False)
    print(f"Wrote {len(reference_ids)} reference query IDs to {output_path}")
    print(f"  Existing IDs: {len(existing_ids)}")
    print(f"  New IDs:      {len(new_ids)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a fixed 200-query MS MARCO dev reference set."
    )
    parser.add_argument("--topics", type=Path, default=DEFAULT_TOPICS_PATH)
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS_PATH)
    parser.add_argument(
        "--existing-traces",
        type=Path,
        default=DEFAULT_EXISTING_TRACES_PATH,
        help="CSV whose query_id column provides the first 50 reference IDs.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    create_reference_set(
        topics_path=args.topics,
        qrels_path=args.qrels,
        existing_traces_path=args.existing_traces,
        output_path=args.output,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
