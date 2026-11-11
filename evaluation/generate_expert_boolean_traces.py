r"""Generate two-step greedy oracle traces using Boolean query formulation.

The script reads the MS MARCO dev2 topics, runs a two-step greedy oracle with
Boolean query reformulation, and writes the resulting transitions to CSV.  All
first-branch agent outputs and retriever results are cached under ``cache/`` so
re-generating the dataset (or extending it to more queries) re-uses previous
work.  Runs are resumable: completed query IDs are skipped if the output CSV
already exists.

Usage:
    poetry run python generate_expert_boolean_traces.py --num-queries 500

To extend to 1000 queries later:
    poetry run python generate_expert_boolean_traces.py --num-queries 1000 --resume
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

root_path = Path(__file__).resolve().parent
src_path = root_path / "src"
sys.path.insert(0, str(root_path))
sys.path.insert(0, str(src_path))

from src.simulation import Simulation, SimConfig, ACTION_NAMES, ACTION_STOP
from src.agents.reformulate import ReformulationAgent
from src.agents.rerank import RerankingAgent
from src.agents.prf import PRFAgent
from src.utils.retriever import Retriever
from src.utils.cache import DiskCache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# -----------------------------
# Configuration
# -----------------------------
QUERIES_PATH: str = "data/topics.ms-marco-dev2.tsv"
QRELS_PATH: str = "data/qrels.ms-marco-dev2.tsv"
OUTPUT_PATH: str = "results/expert_boolean_traces_dev2_top20.csv"
CACHE_DIR: str = "cache/expert_boolean_oracle"
CACHE_VERSION: str = "v1"

RANDOM_SEED = 42

SIM_MAX_STEPS = 3
SIM_TOP_K_RERANK = 20
SIM_TOP_K_PRF = 10
SIM_NDCG_K = 20
SIM_RECALL_K = 20

QR_FORMULATION = "boolean"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_cache_version(
    base_version: str,
    index: str,
    search_method: str,
    min_top_k: int,
) -> str:
    """Return a version string that invalidates the cache when config changes."""
    return (
        f"{base_version}"
        f"-tr{SIM_TOP_K_RERANK}"
        f"-tp{SIM_TOP_K_PRF}"
        f"-ndcg{SIM_NDCG_K}"
        f"-recall{SIM_RECALL_K}"
        f"-{QR_FORMULATION}"
        f"-idx{index}"
        f"-sm{search_method}"
        f"-min{min_top_k}"
    )


def load_queries_and_qrels(
    queries_path: Path,
    qrels_path: Path,
) -> Tuple[Dict[str, str], Dict[str, Dict[str, int]]]:
    """Load dev2 topics and qrels."""
    queries_df = pd.read_csv(queries_path, sep="\t", header=0, dtype={"query_id": str})
    queries_df["query_id"] = queries_df["query_id"].astype(str)
    query_col = "query" if "query" in queries_df.columns else "query_text"
    queries_map = {
        str(qid): qtext
        for qid, qtext in zip(queries_df["query_id"], queries_df[query_col])
    }

    qrels_df = pd.read_csv(qrels_path, sep="\t", header=0)
    qrels: Dict[str, Dict[str, int]] = {}
    for _, row in qrels_df.iterrows():
        qid = str(row["query_id"])
        doc_id = str(row["doc_id"])
        rel = int(row["relevance"])
        qrels.setdefault(qid, {})[doc_id] = rel

    return queries_map, qrels


def get_completed_query_ids(output_path: Path) -> Set[str]:
    """Read already-processed query IDs from an existing output CSV."""
    if not output_path.exists():
        return set()
    try:
        df = pd.read_csv(output_path, usecols=["query_id"], dtype={"query_id": str})
        return set(df["query_id"].unique())
    except Exception as exc:
        logger.warning("Could not read existing output %s: %s", output_path, exc)
        return set()


def transition_to_row(
    query_id: str,
    query: str,
    step_idx: int,
    transition: Any,
) -> Dict[str, Any]:
    """Flatten one Transition into a CSV row."""
    state = transition.state
    info = transition.info
    action = int(transition.action)

    return {
        "query_id": query_id,
        "query": query,
        "new_query": info.get("new_query", query),
        "action_name": info.get("action_name", ACTION_NAMES.get(action, "")),
        "query_length": float(state[0]),
        "score_spread": float(state[385]),
        "score_entropy": float(state[386]),
        "step": step_idx + 1,
        "rank_overlap": float(info.get("rank_overlap", state[391])),
        "query_drift": float(info.get("query_drift", state[392])),
        "ndcg_before": float(info.get("ndcg_before", 0.0)),
        "ndcg_after": float(info.get("ndcg_after", 0.0)),
        "recall_before": float(info.get("recall_before", 0.0)),
        "recall_after": float(info.get("recall_after", 0.0)),
        "cost": float(info.get("cost", 0.0)),
        "latency_ms": float(info.get("elapsed_ms", 0.0)),
        "cumulative_cost": float(info.get("cum_cost", 0.0)),
        "cumulative_latency_ms": float(info.get("cum_elapsed_ms", 0.0)),
        "reward": float(transition.reward),
        "terminal": int(action == ACTION_STOP),
        "timeout": int(step_idx == SIM_MAX_STEPS - 1 and action != ACTION_STOP),
    }


# ---------------------------------------------------------------------------
# Cached components
# ---------------------------------------------------------------------------
def create_cached_retriever_callable(
    retriever_instance: Retriever,
    cache: DiskCache,
    search_method: str = "bm25",
    adaptive_top_k: bool = True,
    min_top_k: int = 1,
) -> callable:
    """Return a (query, top_k) callable that caches ChatNoir responses on disk."""

    def retriever_func(query: str, top_k: int) -> Tuple[List[str], np.ndarray, Dict[str, str]]:
        key = f"retrieve|q:{query}|k:{top_k}|idx:{retriever_instance.index}|sm:{search_method}"
        cached = cache.get(key)
        if cached is not None:
            return cached

        results = retriever_instance.retrieve(
            query,
            top_k=top_k,
            search_method=search_method,
            adaptive_top_k=adaptive_top_k,
            min_top_k=min_top_k,
        )

        doc_ids = [doc["id"] for doc in results]
        scores = np.array([doc["score"] for doc in results], dtype=np.float32)
        corpus_data = {doc["id"]: doc["text"] for doc in results}

        value = (doc_ids, scores, corpus_data)
        cache.put(key, value)
        return value

    return retriever_func


class CachedReformulationAgent(ReformulationAgent):
    """ReformulationAgent wrapper that memoises first-branch outputs."""

    def __init__(
        self,
        embed_model: SentenceTransformer,
        formulation_mode: str,
        cache: DiskCache,
    ):
        super().__init__(embed_model=embed_model, formulation_mode=formulation_mode)
        self._cache = cache

    def compute_effects(self, query_features: Dict[str, Any]) -> Dict[str, Any]:
        query_text = query_features["query_text"]
        top_k = query_features.get("top_k", SIM_TOP_K_RERANK)
        key = f"reformulate|q:{query_text}|m:{self.formulation_mode}|k:{top_k}"

        cached = self._cache.get(key)
        if cached is not None:
            return cached

        effects = super().compute_effects(query_features)
        self._cache.put(key, effects)
        return effects


class CachedPRFAgent(PRFAgent):
    """PRFAgent wrapper that memoises first-branch outputs."""

    def __init__(
        self,
        embed_model: SentenceTransformer,
        num_expansion_terms: int,
        cache: DiskCache,
    ):
        super().__init__(embed_model=embed_model, num_expansion_terms=num_expansion_terms)
        self._cache = cache

    def compute_effects(self, query_features: Dict[str, Any]) -> Dict[str, Any]:
        query_text = query_features["query_text"]
        top_k_rerank = query_features.get("top_k_rerank", SIM_TOP_K_RERANK)
        top_k_prf = query_features.get("top_k_prf", SIM_TOP_K_PRF)
        key = (
            f"prf|q:{query_text}"
            f"|kr:{top_k_rerank}"
            f"|kp:{top_k_prf}"
            f"|ne:{self.num_expansion_terms}"
        )

        cached = self._cache.get(key)
        if cached is not None:
            return cached

        effects = super().compute_effects(query_features)
        self._cache.put(key, effects)
        return effects


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args: argparse.Namespace) -> None:
    set_seed(RANDOM_SEED)

    queries_path = Path(args.queries_path)
    qrels_path = Path(args.qrels_path)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    queries_map, qrels = load_queries_and_qrels(queries_path, qrels_path)

    # Stable ordering: take the first N queries after the offset that have qrels.
    all_qids = list(queries_map.keys())
    selected = [
        (qid, queries_map[qid])
        for qid in all_qids[args.offset : args.offset + args.num_queries]
        if qid in qrels and qrels[qid]
    ]
    if not selected:
        raise ValueError("No queries selected; check offset/num-queries and qrel coverage")

    completed: Set[str] = set()
    if args.resume:
        completed = get_completed_query_ids(output_path)
        if completed:
            logger.info("Resuming: %d query IDs already in %s", len(completed), output_path)

    logger.info("Loading query encoder: all-MiniLM-L6-v2")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    logger.info("Initializing ChatNoir BM25 retriever")
    retriever_instance = Retriever()

    cache_version = _build_cache_version(
        args.cache_version,
        index=retriever_instance.index,
        search_method=args.search_method,
        min_top_k=args.min_top_k,
    )
    cache = DiskCache(Path(args.cache_dir), cache_version)
    logger.info("Cache directory: %s", cache.cache_dir)

    retriever = create_cached_retriever_callable(
        retriever_instance,
        cache,
        search_method=args.search_method,
        adaptive_top_k=args.adaptive_top_k,
        min_top_k=args.min_top_k,
    )

    logger.info("Using Boolean query formulation for the reformulation agent")
    qr_agent = CachedReformulationAgent(
        embed_model=encoder,
        formulation_mode=QR_FORMULATION,
        cache=cache,
    )
    rr_agent = RerankingAgent(embed_model=encoder)
    prf_agent = CachedPRFAgent(
        embed_model=encoder,
        num_expansion_terms=5,
        cache=cache,
    )

    config = SimConfig(
        max_steps=SIM_MAX_STEPS,
        top_k_rerank=SIM_TOP_K_RERANK,
        top_k_prf=SIM_TOP_K_PRF,
        ndcg_k=SIM_NDCG_K,
        recall_k=SIM_RECALL_K,
        reward_alpha=2.0,
        reward_beta=0.5,
        reward_gamma=0.2,
        reward_delta=0.1,
    )

    sim = Simulation(
        encoder=encoder,
        retriever=retriever,
        agents=[qr_agent, rr_agent, prf_agent],
        config=config,
        policy_model=None,
    )

    fieldnames = [
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

    # Write a config snapshot next to the output for reproducibility.
    config_path = output_path.with_suffix(".config.json")
    config_path.write_text(
        json.dumps(
            {
                "queries_path": str(queries_path),
                "qrels_path": str(qrels_path),
                "output_path": str(output_path),
                "cache_dir": args.cache_dir,
                "cache_version": cache_version,
                "num_queries": args.num_queries,
                "offset": args.offset,
                "random_seed": RANDOM_SEED,
                "sim_max_steps": SIM_MAX_STEPS,
                "sim_top_k_rerank": SIM_TOP_K_RERANK,
                "sim_top_k_prf": SIM_TOP_K_PRF,
                "sim_ndcg_k": SIM_NDCG_K,
                "sim_recall_k": SIM_RECALL_K,
                "qr_formulation": QR_FORMULATION,
                "adaptive_top_k": args.adaptive_top_k,
                "min_top_k": args.min_top_k,
                "search_method": args.search_method,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    write_header = not output_path.exists() or output_path.stat().st_size == 0
    rows_written = 0
    queries_failed = 0
    queries_succeeded = 0

    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for idx, (query_id, query) in enumerate(selected, start=1):
            if query_id in completed:
                logger.info(
                    "[expert_boolean] [%d/%d] SKIPPING already-completed query_id=%s",
                    idx,
                    len(selected),
                    query_id,
                )
                queries_succeeded += 1
                continue

            logger.info(
                "[expert_boolean] [%d/%d] Rolling out query_id=%s | %s",
                idx,
                len(selected),
                query_id,
                query[:80],
            )
            try:
                doc_ids, doc_scores, corpus_data = retriever(query, sim.cfg.top_k_rerank)
                if not doc_ids:
                    raise RuntimeError("Initial retrieval returned no documents")

                trajectory = sim.generate_trajectory(
                    query=query,
                    doc_ids=doc_ids,
                    doc_scores=doc_scores,
                    qrels=qrels.get(query_id, {}),
                    policy="expert",
                    corpus_data=corpus_data,
                )

                for step_idx, transition in enumerate(trajectory):
                    writer.writerow(transition_to_row(query_id, query, step_idx, transition))
                    rows_written += 1

                f.flush()
                queries_succeeded += 1
                logger.info(
                    "[expert_boolean] [%d/%d] SAVED %d transitions",
                    idx,
                    len(selected),
                    len(trajectory),
                )

            except Exception as e:
                queries_failed += 1
                logger.exception(
                    "[expert_boolean] [%d/%d] FAILED query_id=%s: %s",
                    idx,
                    len(selected),
                    query_id,
                    e,
                )

    logger.info("=" * 70)
    logger.info("[expert_boolean] rollout complete")
    logger.info("Output: %s", output_path)
    logger.info("Config: %s", config_path)
    logger.info("Total rows written this run: %d", rows_written)
    logger.info("Queries succeeded: %d", queries_succeeded)
    logger.info("Queries failed: %d", queries_failed)
    logger.info("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate two-step greedy oracle traces with Boolean query reformulation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--queries-path",
        type=str,
        default=QUERIES_PATH,
        help="TSV file with query_id and query/query_text columns.",
    )
    parser.add_argument(
        "--qrels-path",
        type=str,
        default=QRELS_PATH,
        help="TSV qrels file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT_PATH,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=500,
        help="Number of queries to process.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Offset into the topic file.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=CACHE_DIR,
        help="Root cache directory (a versioned sub-directory is created inside it).",
    )
    parser.add_argument(
        "--cache-version",
        type=str,
        default=CACHE_VERSION,
        help="Cache version prefix.  Bump this to invalidate the cache.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip query IDs already present in the output CSV.",
    )
    parser.add_argument(
        "--adaptive-top-k",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Halve the requested ChatNoir size on 500 errors down to --min-top-k.",
    )
    parser.add_argument(
        "--min-top-k",
        type=int,
        default=1,
        help="Minimum retrieval depth when adaptive fallback is used.",
    )
    parser.add_argument(
        "--search-method",
        type=str,
        default="bm25",
        help="ChatNoir search method.",
    )
    args = parser.parse_args()
    main(args)
