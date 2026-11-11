r"""Lightweight query-first evaluation of query formulation strategies.

Compares three baselines on the same queries:
  1. Original query -> retrieval.
  2. Natural-language LLM reformulation -> retrieval.
  3. Boolean concept formulation -> retrieval.

Formulation baselines do NOT retrieve the original query first; they issue a
single retrieval call with the formulated query.  The script logs generated
queries, formulation mode, fallback/failure reasons, retrieval-call counts,
and timing.

Usage:
    poetry run python evaluate_formulation.py
    poetry run python evaluate_formulation.py --max-queries 10 --qr-formulation boolean
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

root_path = Path(__file__).resolve().parent
src_path = root_path / "src"
sys.path.insert(0, str(root_path))
sys.path.insert(0, str(src_path))

from src.simulation import Simulation, SimConfig
from src.agents.reformulate import ReformulationAgent
from src.utils.retriever import Retriever, create_retriever_callable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ── Configuration ────────────────────────────────────────────────────────────
OUTPUT_DIR: str = "results"
OUTPUT_CSV: str = "formulation_baseline_50_msmarco_dev.csv"
CQL_TRACES_PATH: str = "results/cql_full_tree_traces_50_msmarco_dev.csv"

RANDOM_SEED = 42

SIM_MAX_STEPS = 3
SIM_TOP_K_RERANK = 50
SIM_TOP_K_PRF = 10
SIM_NDCG_K = 50
SIM_RECALL_K = 100


# ── Environment loading ──────────────────────────────────────────────────────
def load_env_file() -> None:
    """Load environment variables from .env.txt in the repository root."""
    env_file = root_path / ".env.txt"
    if not env_file.exists():
        return
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip()


load_env_file()


# ── Data loading ─────────────────────────────────────────────────────────────
def load_qrels(qrels_path: str) -> Dict[str, Dict[str, int]]:
    qrels: Dict[str, Dict[str, int]] = {}
    df = pd.read_csv(qrels_path, sep="\t", header=0)
    for _, row in df.iterrows():
        qid = str(row["query_id"])
        doc_id = str(row["doc_id"])
        rel = int(row["relevance"])
        qrels.setdefault(qid, {})[doc_id] = rel
    return qrels


def load_queries(queries_path: str) -> Dict[str, str]:
    df = pd.read_csv(queries_path, sep="\t", header=0, dtype={"query_id": str})
    df["query_id"] = df["query_id"].astype(str)
    query_col = "query" if "query" in df.columns else "query_text"
    return {
        str(qid): qtext
        for qid, qtext in zip(df["query_id"], df[query_col])
    }


# ── Metrics ──────────────────────────────────────────────────────────────────
def compute_metrics(
    doc_ids: List[str],
    qrels: Dict[str, int],
    ndcg_k: int,
    recall_k: int,
) -> Dict[str, float]:
    return {
        "ndcg": Simulation.compute_ndcg(doc_ids, qrels, ndcg_k),
        "recall": Simulation.compute_recall(doc_ids, qrels, recall_k),
    }


# ── Evaluation helpers ───────────────────────────────────────────────────────
def evaluate_original(
    query: str,
    retriever: callable,
    top_k: int,
    qrels: Dict[str, int],
    ndcg_k: int,
    recall_k: int,
) -> Dict[str, Any]:
    """Original-query baseline: exactly one retrieval call."""
    t0 = time.perf_counter()
    doc_ids, doc_scores, corpus_data = retriever(query, top_k)
    elapsed_ms = (time.perf_counter() - t0) * 1_000.0
    metrics = compute_metrics(doc_ids, qrels, ndcg_k, recall_k)
    return {
        "mode": "original",
        "generated_query": query,
        "fallback_reason": None,
        "retrieval_count": 1,
        "num_results": len(doc_ids),
        "ndcg": metrics["ndcg"],
        "recall": metrics["recall"],
        "elapsed_ms": elapsed_ms,
    }


def evaluate_reformulation(
    query: str,
    agent: ReformulationAgent,
    retriever: callable,
    top_k: int,
    qrels: Dict[str, int],
    ndcg_k: int,
    recall_k: int,
) -> Dict[str, Any]:
    """Reformulation baseline: one LLM call + one retrieval call."""
    query_features = {
        "query_text": query,
        "retriever": retriever,
        "top_k": top_k,
    }
    t0 = time.perf_counter()
    effects = agent.compute_effects(query_features)
    elapsed_ms = (time.perf_counter() - t0) * 1_000.0

    new_query = effects.get("new_query_text", query)
    new_doc_ids = effects.get("new_doc_ids", [])
    fallback_reason = effects.get("fallback_reason")
    formulation_mode = effects.get("formulation_mode", "unknown")
    retrieval_error = bool(effects.get("retrieval_error", False))

    metrics = compute_metrics(new_doc_ids, qrels, ndcg_k, recall_k)

    # Retrieval count inside the agent is always 1 for the formulation path.
    status = "ok"
    if fallback_reason:
        status = "fallback"
    elif retrieval_error:
        status = "retrieval_error"
    elif len(new_doc_ids) == 0:
        status = "zero_results"

    return {
        "mode": formulation_mode,
        "generated_query": new_query,
        "fallback_reason": fallback_reason,
        "retrieval_count": 1,
        "num_results": len(new_doc_ids),
        "ndcg": metrics["ndcg"],
        "recall": metrics["recall"],
        "elapsed_ms": elapsed_ms,
        "status": status,
        "boolean_query": effects.get("boolean_query"),
        "validated_concepts": effects.get("validated_concepts"),
    }


# ── Main evaluation loop ─────────────────────────────────────────────────────
def main(args: argparse.Namespace) -> None:
    np.random.seed(RANDOM_SEED)

    # Resolve query IDs used by the CQL traces for comparability.
    cql_path = Path(args.cql_traces_path)
    if not cql_path.exists():
        raise FileNotFoundError(f"CQL traces not found at {cql_path}")

    df_cql = pd.read_csv(cql_path)
    query_ids = [str(qid) for qid in df_cql["query_id"].unique().tolist()]
    logger.info("Loaded %d query IDs from %s", len(query_ids), cql_path)

    queries = load_queries(args.queries_path)
    qrels = load_qrels(args.qrels_path)

    selected = [
        (qid, queries[qid])
        for qid in query_ids
        if qid in queries and qid in qrels and qrels[qid]
    ]
    if args.max_queries is not None:
        selected = selected[: args.max_queries]
    logger.info("Evaluating %d queries", len(selected))

    logger.info("Loading query encoder: all-MiniLM-L6-v2")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    logger.info("Initializing ChatNoir BM25 retriever")
    retriever_instance = Retriever()
    retriever = create_retriever_callable(retriever_instance)

    config = SimConfig(
        max_steps=SIM_MAX_STEPS,
        top_k_rerank=SIM_TOP_K_RERANK,
        top_k_prf=SIM_TOP_K_PRF,
        ndcg_k=SIM_NDCG_K,
        recall_k=SIM_RECALL_K,
    )

    agents = {
        "natural": ReformulationAgent(
            embed_model=encoder,
            formulation_mode="natural",
        ),
        "boolean": ReformulationAgent(
            embed_model=encoder,
            formulation_mode="boolean",
        ),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_csv
    temporary_path = output_path.with_suffix(".tmp")

    fieldnames = [
        "query_id",
        "query",
        "mode",
        "generated_query",
        "boolean_query",
        "fallback_reason",
        "status",
        "retrieval_count",
        "num_results",
        "ndcg",
        "recall",
        "elapsed_ms",
    ]

    rows: List[Dict[str, Any]] = []
    stats: Dict[str, Dict[str, Any]] = {
        "original": {"count": 0, "ndcg_sum": 0.0, "recall_sum": 0.0, "retrievals": 0},
        "natural": {"count": 0, "ndcg_sum": 0.0, "recall_sum": 0.0, "retrievals": 0},
        "boolean": {"count": 0, "ndcg_sum": 0.0, "recall_sum": 0.0, "retrievals": 0},
    }

    for idx, (query_id, query) in enumerate(selected, start=1):
        logger.info(
            "[%d/%d] Evaluating query_id=%s | %s",
            idx,
            len(selected),
            query_id,
            query[:80],
        )
        qrels_for_query = qrels.get(query_id, {})

        # Baseline 1: original query.
        original_result = evaluate_original(
            query=query,
            retriever=retriever,
            top_k=config.top_k_rerank,
            qrels=qrels_for_query,
            ndcg_k=config.ndcg_k,
            recall_k=config.recall_k,
        )
        original_result["query_id"] = query_id
        original_result["query"] = query
        rows.append(original_result)
        _accumulate(stats, "original", original_result)

        # Baseline 2: natural-language reformulation.
        natural_result = evaluate_reformulation(
            query=query,
            agent=agents["natural"],
            retriever=retriever,
            top_k=config.top_k_rerank,
            qrels=qrels_for_query,
            ndcg_k=config.ndcg_k,
            recall_k=config.recall_k,
        )
        natural_result["query_id"] = query_id
        natural_result["query"] = query
        rows.append(natural_result)
        _accumulate(stats, "natural", natural_result)

        # Baseline 3: Boolean concept formulation.
        boolean_result = evaluate_reformulation(
            query=query,
            agent=agents["boolean"],
            retriever=retriever,
            top_k=config.top_k_rerank,
            qrels=qrels_for_query,
            ndcg_k=config.ndcg_k,
            recall_k=config.recall_k,
        )
        boolean_result["query_id"] = query_id
        boolean_result["query"] = query
        rows.append(boolean_result)
        _accumulate(stats, "boolean", boolean_result)

        logger.info(
            "[%d/%d] original_ndcg=%.4f natural_ndcg=%.4f boolean_ndcg=%.4f",
            idx,
            len(selected),
            original_result["ndcg"],
            natural_result["ndcg"],
            boolean_result["ndcg"],
        )

    with open(temporary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        f.flush()

    temporary_path.replace(output_path)

    logger.info("=" * 70)
    logger.info("Formulation evaluation complete")
    logger.info("Output: %s", output_path)
    _log_summary(stats)
    logger.info("=" * 70)


def _accumulate(
    stats: Dict[str, Dict[str, Any]],
    key: str,
    row: Dict[str, Any],
) -> None:
    stats[key]["count"] += 1
    stats[key]["ndcg_sum"] += float(row["ndcg"])
    stats[key]["recall_sum"] += float(row["recall"])
    stats[key]["retrievals"] += int(row["retrieval_count"])


def _log_summary(stats: Dict[str, Dict[str, Any]]) -> None:
    for key, vals in stats.items():
        count = vals["count"]
        if count == 0:
            logger.info("[%s] no queries evaluated", key)
            continue
        avg_ndcg = vals["ndcg_sum"] / count
        avg_recall = vals["recall_sum"] / count
        logger.info(
            "[%s] queries=%d avg_ndcg=%.4f avg_recall=%.4f total_retrievals=%d",
            key,
            count,
            avg_ndcg,
            avg_recall,
            vals["retrievals"],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate query formulation baselines",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--queries-path",
        type=str,
        default="data/topics.ms-marco-dev.tsv",
        help="Path to queries TSV",
    )
    parser.add_argument(
        "--qrels-path",
        type=str,
        default="data/qrels.ms-marco-dev.tsv",
        help="Path to qrels TSV",
    )
    parser.add_argument(
        "--cql-traces-path",
        type=str,
        default=CQL_TRACES_PATH,
        help="CQL traces CSV used to select query IDs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help="Output directory",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=OUTPUT_CSV,
        help="Output CSV filename",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Evaluate only the first N queries",
    )
    args = parser.parse_args()
    main(args)
