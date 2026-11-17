r"""Lightweight ChatNoir probe that tests whether queries return enough results.

Usage:
    poetry run python helper_scripts/probe_retriever.py --candidate-csv CANDIDATES.csv --min-results 50 --output-csv RESULTS.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Add project paths for absolute imports.
root_path = Path(__file__).resolve().parent.parent
src_path = root_path / "src"
sys.path.insert(0, str(root_path))
sys.path.insert(0, str(src_path))

from src.utils.retriever import Retriever, create_retriever_callable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_TOPICS_PATH: Path = Path("data/topics.ms-marco-dev.tsv")
DEFAULT_QRELS_PATH: Path = Path("data/qrels.ms-marco-dev.tsv")
DEFAULT_MIN_RESULTS: int = 50
DEFAULT_TOP_K: int = 50
DEFAULT_MAX_RETRIES: int = 3
DEFAULT_RETRY_BACKOFF_S: float = 5.0


def load_judged_queries(topics_path: Path, qrels_path: Path) -> Dict[str, str]:
    """Return a mapping from judged query_id to query text."""
    qrels_df = pd.read_csv(qrels_path, sep="\t", header=0, dtype=str)
    judged_ids = set(qrels_df["query_id"].astype(str).unique())

    topics_df = pd.read_csv(topics_path, sep="\t", header=0, dtype=str)
    topics_df["query_id"] = topics_df["query_id"].astype(str)
    query_col = "query" if "query" in topics_df.columns else "query_text"

    return {
        str(qid): str(qtext)
        for qid, qtext in zip(topics_df["query_id"], topics_df[query_col])
        if qid in judged_ids
    }


def probe_query(
    query: str,
    retriever_func: callable,
    top_k: int,
    min_results: int,
    max_retries: int,
    retry_backoff_s: float,
) -> Tuple[bool, int]:
    """Probe one query and return (success, num_results)."""
    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            doc_ids, _, _ = retriever_func(query, top_k)
            num_results = len(doc_ids) if doc_ids is not None else 0
            success = num_results >= min_results
            return success, num_results
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait = retry_backoff_s * (2 ** attempt)
                logger.warning(
                    "Probe failed for query %r (attempt %d/%d); retrying in %.1fs: %s",
                    query[:80],
                    attempt + 1,
                    max_retries,
                    wait,
                    e,
                )
                time.sleep(wait)
            else:
                break

    logger.warning(
        "Probe failed for query %r after %d attempts: %s",
        query[:80],
        max_retries,
        last_error,
    )
    return False, 0


def probe_candidates(
    candidate_ids: List[str],
    queries_map: Dict[str, str],
    retriever_func: callable,
    top_k: int,
    min_results: int,
    max_retries: int,
    retry_backoff_s: float,
) -> List[Dict[str, object]]:
    """Probe all candidate queries and return result rows."""
    rows: List[Dict[str, object]] = []
    for idx, qid in enumerate(candidate_ids, start=1):
        query = queries_map.get(qid)
        if query is None:
            logger.warning("[%d/%d] Query ID %s not found in topics; skipping", idx, len(candidate_ids), qid)
            rows.append({"query_id": qid, "query": "", "success": False, "num_results": 0})
            continue

        logger.info("[%d/%d] Probing query_id=%s | %s", idx, len(candidate_ids), qid, query[:80])
        success, num_results = probe_query(
            query=query,
            retriever_func=retriever_func,
            top_k=top_k,
            min_results=min_results,
            max_retries=max_retries,
            retry_backoff_s=retry_backoff_s,
        )
        rows.append({
            "query_id": qid,
            "query": query,
            "success": success,
            "num_results": num_results,
        })
        logger.info(
            "[%d/%d] query_id=%s success=%s num_results=%d",
            idx,
            len(candidate_ids),
            qid,
            success,
            num_results,
        )
    return rows


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Probe ChatNoir retrieval for a list of candidate queries."
    )
    parser.add_argument(
        "--candidate-csv",
        type=Path,
        required=True,
        help="CSV with a query_id column listing candidates to probe.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Output CSV path for probe results.",
    )
    parser.add_argument("--topics", type=Path, default=DEFAULT_TOPICS_PATH)
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS_PATH)
    parser.add_argument(
        "--min-results",
        type=int,
        default=DEFAULT_MIN_RESULTS,
        help="Minimum number of retrieved documents required for success.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of documents to request from ChatNoir.",
    )
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--retry-backoff-s", type=float, default=DEFAULT_RETRY_BACKOFF_S)
    args = parser.parse_args(argv)

    if not args.candidate_csv.exists():
        raise FileNotFoundError(f"Candidate CSV not found: {args.candidate_csv}")

    candidate_df = pd.read_csv(args.candidate_csv, dtype=str)
    if "query_id" not in candidate_df.columns:
        raise ValueError(f"No query_id column in {args.candidate_csv}")
    candidate_ids = [str(qid) for qid in candidate_df["query_id"].unique()]

    queries_map = load_judged_queries(args.topics, args.qrels)

    logger.info("Initializing ChatNoir BM25 retriever (msmarco-v2.1)")
    retriever_instance = Retriever(index="msmarco-v2.1")
    retriever_func = create_retriever_callable(retriever_instance)

    rows = probe_candidates(
        candidate_ids=candidate_ids,
        queries_map=queries_map,
        retriever_func=retriever_func,
        top_k=args.top_k,
        min_results=args.min_results,
        max_retries=args.max_retries,
        retry_backoff_s=args.retry_backoff_s,
    )

    output_df = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output_csv, index=False)

    success_count = int(output_df["success"].sum())
    print(f"Probed {len(rows)} queries; {success_count} succeeded with >= {args.min_results} results")
    print(f"Wrote probe results to {args.output_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
