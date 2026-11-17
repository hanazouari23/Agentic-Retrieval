r"""Build a success-filtered reference set by probing new candidate queries.

Starts from a seed set of known-good query IDs, samples new candidates from the
judged MS MARCO dev pool, probes them with a lightweight retrieval check, and
keeps those that pass until a target size is reached.

Usage:
    poetry run python helper_scripts/build_success_reference_set.py --target 150 --batch-size 100 --min-results 50 --seed 42 --output RQs/relevant_csvs/msmarco_dev_reference_success_150.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd

# Add project paths for absolute imports.
root_path = Path(__file__).resolve().parent.parent
src_path = root_path / "src"
sys.path.insert(0, str(root_path))
sys.path.insert(0, str(src_path))

from helper_scripts.probe_retriever import (
    load_judged_queries,
    probe_candidates,
)
from src.utils.retriever import Retriever, create_retriever_callable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_SEED_PATH: Path = Path("RQs/relevant_csvs/successful_query_ids_71.csv")
DEFAULT_TRIED_PATH: Path = Path("RQs/relevant_csvs/msmarco_dev_reference_200.csv")
DEFAULT_TOPICS_PATH: Path = Path("data/topics.ms-marco-dev.tsv")
DEFAULT_QRELS_PATH: Path = Path("data/qrels.ms-marco-dev.tsv")
DEFAULT_OUTPUT_PATH: Path = Path("RQs/relevant_csvs/msmarco_dev_reference_success_150.csv")
DEFAULT_TARGET: int = 150
DEFAULT_BATCH_SIZE: int = 100
DEFAULT_MIN_RESULTS: int = 50
DEFAULT_SEED: int = 42


def load_query_ids(csv_path: Path) -> List[str]:
    """Load ordered unique query IDs from a single-column CSV."""
    df = pd.read_csv(csv_path, dtype=str)
    if "query_id" not in df.columns:
        raise ValueError(f"No query_id column in {csv_path}")
    return [str(qid) for qid in df["query_id"].unique()]


def sample_batches(
    candidates: List[str],
    batch_size: int,
    rng: random.Random,
) -> List[List[str]]:
    """Split candidates into random batches of size batch_size."""
    shuffled = candidates.copy()
    rng.shuffle(shuffled)
    batches: List[List[str]] = []
    for i in range(0, len(shuffled), batch_size):
        batches.append(shuffled[i : i + batch_size])
    return batches


def build_reference_set(
    seed_ids: List[str],
    tried_ids: List[str],
    topics_path: Path,
    qrels_path: Path,
    output_path: Path,
    target: int,
    batch_size: int,
    min_results: int,
    seed: int,
) -> None:
    queries_map = load_judged_queries(topics_path, qrels_path)
    all_judged_ids = set(queries_map.keys())

    # Validate seed IDs.
    missing_seed = [qid for qid in seed_ids if qid not in all_judged_ids]
    if missing_seed:
        raise ValueError(
            f"{len(missing_seed)} seed query IDs are not judged: {missing_seed[:5]}"
        )

    # Build candidate pool: judged queries not already tried.
    tried_set: Set[str] = set(tried_ids)
    candidates = [qid for qid in queries_map.keys() if qid not in tried_set]
    logger.info(
        "Starting with %d seed IDs; %d already tried; %d candidates available",
        len(seed_ids),
        len(tried_set),
        len(candidates),
    )

    success_pool: List[str] = list(seed_ids)
    probe_log: Dict[str, object] = {
        "seed_ids": seed_ids,
        "tried_ids": tried_ids,
        "target": target,
        "min_results": min_results,
        "seed": seed,
        "batches": [],
    }

    if len(success_pool) >= target:
        logger.info("Seed pool already meets target of %d; no new candidates needed", target)
    else:
        rng = random.Random(seed)
        batches = sample_batches(candidates, batch_size, rng)

        logger.info("Initializing ChatNoir BM25 retriever (msmarco-v2.1)")
        retriever_instance = Retriever(index="msmarco-v2.1")
        retriever_func = create_retriever_callable(retriever_instance)

        for batch_idx, batch in enumerate(batches, start=1):
            needed = target - len(success_pool)
            if needed <= 0:
                break

            # Only probe as many as needed to avoid extra API calls.
            batch_to_probe = batch[:needed] if len(batch) > needed else batch
            logger.info(
                "Batch %d/%d: probing %d candidates (need %d more successes)",
                batch_idx,
                len(batches),
                len(batch_to_probe),
                needed,
            )

            rows = probe_candidates(
                candidate_ids=batch_to_probe,
                queries_map=queries_map,
                retriever_func=retriever_func,
                top_k=50,
                min_results=min_results,
                max_retries=3,
                retry_backoff_s=5.0,
            )

            batch_successes = [row["query_id"] for row in rows if row["success"]]
            success_pool.extend(batch_successes)

            probe_log["batches"].append({
                "batch_index": batch_idx,
                "candidate_ids": batch_to_probe,
                "results": rows,
            })

            logger.info(
                "Batch %d yielded %d new successes; pool size now %d/%d",
                batch_idx,
                len(batch_successes),
                len(success_pool),
                target,
            )

            if len(success_pool) >= target:
                logger.info("Reached target of %d successful queries", target)
                break

    if len(success_pool) < target:
        logger.warning(
            "Candidate pool exhausted. Final pool size is %d (target was %d).",
            len(success_pool),
            target,
        )

    # Preserve order: seed IDs first, then new successes in the order they were found.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"query_id": success_pool}).to_csv(output_path, index=False)

    log_path = output_path.with_suffix(".json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(probe_log, f, indent=2)

    print(f"Final success pool: {len(success_pool)} query IDs")
    print(f"Wrote reference CSV to {output_path}")
    print(f"Wrote probe log to {log_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a success-filtered reference query set."
    )
    parser.add_argument(
        "--seed-ids",
        type=Path,
        default=DEFAULT_SEED_PATH,
        help="CSV with seed query IDs (not re-probed).",
    )
    parser.add_argument(
        "--tried-ids",
        type=Path,
        default=DEFAULT_TRIED_PATH,
        help="CSV with all previously tried query IDs (excluded from candidate pool).",
    )
    parser.add_argument("--topics", type=Path, default=DEFAULT_TOPICS_PATH)
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--min-results", type=int, default=DEFAULT_MIN_RESULTS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    seed_ids = load_query_ids(args.seed_ids)
    tried_ids = load_query_ids(args.tried_ids)

    build_reference_set(
        seed_ids=seed_ids,
        tried_ids=tried_ids,
        topics_path=args.topics,
        qrels_path=args.qrels,
        output_path=args.output,
        target=args.target,
        batch_size=args.batch_size,
        min_results=args.min_results,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
