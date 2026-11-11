r"""Generate CQL rollout traces with full parameterization.

Usage example:
    python -m evaluation.generate_cql_traces_param \
        --model-path models/cql_on_full_tree_dataset_val_64_32_best.pt \
        --hidden-dims 64 32 \
        --reward-alpha 2.0 \
        --output-path results/cql_on_full_tree_dataset_val_64_32_best_50_msmarco_dev.csv \
        --num-queries 50
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
from d3rlpy.algos import DiscreteCQLConfig
from d3rlpy.dataset import MDPDataset
from d3rlpy.models.encoders import VectorEncoderFactory
from sentence_transformers import SentenceTransformer

root_path = Path(__file__).resolve().parent
src_path = root_path / "src"
sys.path.insert(0, str(root_path))
sys.path.insert(0, str(src_path))

from src.simulation import Simulation, SimConfig, ACTION_NAMES, ACTION_STOP
from src.agents.reformulate import ReformulationAgent
from src.agents.rerank import RerankingAgent
from src.agents.prf import PRFAgent
from src.utils.retriever import Retriever, create_retriever_callable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

REFERENCE_QUERY_IDS_PATH: str = "RQs/relevant_csvs/cql_full_tree_traces_50_msmarco_dev.csv"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_qrels(qrels_path: str) -> Dict[str, Dict[str, int]]:
    from collections import defaultdict
    qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
    with open(qrels_path, "r", encoding="utf-8") as f:
        next(f, None)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            query_id = parts[0].strip()
            doc_id = parts[2].strip()
            try:
                grade = int(parts[3].strip())
            except ValueError:
                continue
            qrels[query_id][doc_id] = grade
    return dict(qrels)


def load_queries(queries_path: str) -> List[Tuple[str, str]]:
    queries: List[Tuple[str, str]] = []
    with open(queries_path, "r", encoding="utf-8") as f:
        next(f, None)
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                query_id = parts[0].strip()
                query_text = parts[1].strip()
            else:
                query_id = str(len(queries))
                query_text = parts[0].strip()
            queries.append((query_id, query_text))
    return queries


def load_reference_query_ids(reference_path: str) -> List[str]:
    """Load the ordered list of query IDs to evaluate from a reference CSV."""
    df = pd.read_csv(reference_path)
    ids = [str(qid).strip() for qid in df["query_id"].unique().tolist()]
    logger.info("Loaded %d reference query IDs from %s", len(ids), reference_path)
    return ids


def load_existing_query_ids(output_path: Path) -> Set[str]:
    if not output_path.exists():
        return set()
    try:
        with open(output_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return {row["query_id"].strip() for row in reader if row.get("query_id")}
    except Exception as e:
        logger.warning("Could not read existing output file %s: %s", output_path, e)
        return set()


def load_initial_retrieval(
    query: str,
    retriever_func: callable,
    top_k: int = 50,
    max_retries: int = 5,
) -> Tuple[List[str], np.ndarray, Dict[str, str]]:
    last_error: Exception = RuntimeError("Retriever returned no documents")
    for attempt in range(max_retries):
        doc_ids, doc_scores, corpus_data = retriever_func(query, top_k)
        if doc_ids is not None and len(doc_ids) > 0:
            return doc_ids, doc_scores, corpus_data
        if attempt < max_retries - 1:
            wait = min(2 ** attempt, 16)
            logger.warning(
                "Retriever returned no documents for query %r (attempt %d/%d); retrying in %ds",
                query, attempt + 1, max_retries, wait,
            )
            time.sleep(wait)
    raise last_error


def load_cql_model(model_path: str, device: str, hidden_dims: Tuple[int, ...]):
    config = DiscreteCQLConfig(
        batch_size=256,
        gamma=0.99,
        learning_rate=3e-4,
        encoder_factory=VectorEncoderFactory(hidden_units=list(hidden_dims)),
        n_critics=2,
        target_update_interval=8000,
        alpha=1.0,
    )
    algo = config.create(device=device)
    dummy_obs = np.zeros((10, 399), dtype=np.float32)
    dummy_actions = np.arange(10, dtype=np.int64) % 4
    dummy_rewards = np.zeros((10,), dtype=np.float32)
    dummy_terminals = np.zeros((10,), dtype=np.float32)
    dummy_terminals[-1] = 1.0
    dummy_timeouts = np.zeros((10,), dtype=np.float32)
    dummy_dataset = MDPDataset(
        observations=dummy_obs,
        actions=dummy_actions,
        rewards=dummy_rewards,
        terminals=dummy_terminals,
        timeouts=dummy_timeouts,
    )
    algo.build_with_dataset(dummy_dataset)
    algo.load_model(model_path)
    return algo


def transition_to_row(
    query_id: str,
    query: str,
    step_idx: int,
    transition: Any,
) -> Dict[str, Any]:
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
        "timeout": int(step_idx == 2 and action != ACTION_STOP),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate CQL rollout traces with full parameterization.")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--hidden-dims", type=int, nargs="+", required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--num-queries", type=int, default=50)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--reward-alpha", type=float, default=2.0)
    parser.add_argument("--reward-beta", type=float, default=0.5)
    parser.add_argument("--reward-gamma", type=float, default=0.2)
    parser.add_argument("--reward-delta", type=float, default=0.1)
    parser.add_argument("--queries-path", type=str, default="data/topics.ms-marco-dev.tsv")
    parser.add_argument("--qrels-path", type=str, default="data/qrels.ms-marco-dev.tsv")
    parser.add_argument("--query-ids-from", type=str, default=REFERENCE_QUERY_IDS_PATH,
                        help="Reference CSV whose query_id column defines the exact queries to evaluate.")
    args = parser.parse_args()

    set_seed(args.random_seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info("Loading CQL model from %s on %s", args.model_path, device)
    policy_model = load_cql_model(args.model_path, device, tuple(args.hidden_dims))

    qrels = load_qrels(args.qrels_path)
    queries = load_queries(args.queries_path)
    judged_queries = [(qid, qtext) for qid, qtext in queries if qid in qrels and qrels[qid]]

    if args.query_ids_from:
        reference_ids = load_reference_query_ids(args.query_ids_from)
        judged_map = {qid: qtext for qid, qtext in judged_queries}
        shuffled = [(qid, judged_map[qid]) for qid in reference_ids if qid in judged_map]
        logger.info("Using %d reference query IDs from %s", len(shuffled), args.query_ids_from)
    else:
        rng = random.Random(args.random_seed)
        shuffled = list(judged_queries)
        rng.shuffle(shuffled)

    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    retriever_instance = Retriever(index="msmarco-v2.1-segmented")
    retriever = create_retriever_callable(retriever_instance)

    qr_agent = ReformulationAgent(embed_model=encoder)
    rr_agent = RerankingAgent(embed_model=encoder)
    prf_agent = PRFAgent(embed_model=encoder, num_expansion_terms=5)

    config = SimConfig(
        max_steps=3,
        top_k_rerank=50,
        top_k_prf=10,
        ndcg_k=50,
        recall_k=100,
        reward_alpha=args.reward_alpha,
        reward_beta=args.reward_beta,
        reward_gamma=args.reward_gamma,
        reward_delta=args.reward_delta,
    )

    sim = Simulation(
        encoder=encoder,
        retriever=retriever,
        agents=[qr_agent, rr_agent, prf_agent],
        config=config,
        policy_model=policy_model,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_qids = load_existing_query_ids(output_path)

    fieldnames = [
        "query_id", "query", "new_query", "action_name", "query_length",
        "score_spread", "score_entropy", "step", "rank_overlap", "query_drift",
        "ndcg_before", "ndcg_after", "recall_before", "recall_after", "cost",
        "latency_ms", "cumulative_cost", "cumulative_latency_ms", "reward",
        "terminal", "timeout",
    ]

    rows_written = 0
    queries_failed = 0
    queries_succeeded = len(existing_qids)
    output_exists = output_path.exists() and output_path.stat().st_size > 0

    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not output_exists:
            writer.writeheader()

        for query_id, query in shuffled:
            if queries_succeeded >= args.num_queries:
                break
            if query_id in existing_qids:
                queries_succeeded += 1
                continue

            logger.info("Rolling out query_id=%s | %s", query_id, query[:80])
            try:
                doc_ids, doc_scores, corpus_data = load_initial_retrieval(
                    query=query, retriever_func=retriever, top_k=config.top_k_rerank,
                )
                trajectory = sim.generate_trajectory(
                    query=query,
                    doc_ids=doc_ids,
                    doc_scores=doc_scores,
                    qrels=qrels.get(query_id, {}),
                    policy="cql",
                    corpus_data=corpus_data,
                )
                for step_idx, transition in enumerate(trajectory):
                    writer.writerow(transition_to_row(query_id, query, step_idx, transition))
                    rows_written += 1
                f.flush()
                queries_succeeded += 1
                logger.info("SAVED %d transitions", len(trajectory))
                time.sleep(0.5)
            except Exception as e:
                queries_failed += 1
                logger.exception("FAILED query_id=%s: %s", query_id, e)
                time.sleep(5.0)

    logger.info("=" * 70)
    logger.info("CQL rollout complete")
    logger.info("Output: %s", output_path)
    logger.info("Total rows written this run: %d", rows_written)
    logger.info("Queries succeeded total: %d", queries_succeeded)
    logger.info("Queries failed this run: %d", queries_failed)


if __name__ == "__main__":
    main()
