r"""Generate trajectories using a trained CQL policy on MS MARCO dev queries.

Usage:
    poetry run python generate_cql_traces.py

Produces:
    results/cql_full_tree_traces_50_msmarco_dev.csv
"""
from __future__ import annotations

import csv
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from d3rlpy.algos import DiscreteCQLConfig
from d3rlpy.dataset import MDPDataset
from d3rlpy.models.encoders import VectorEncoderFactory
from sentence_transformers import SentenceTransformer

# Add project paths for absolute imports.
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

# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH: str = "models/cql_on_full_tree_dataset.pt"
DATASET_PATH: str = "data/training/mdp_dataset_tree_branches_5000.h5"
QRELS_PATH: str = "data/qrels.ms-marco-dev.tsv"
QUERIES_PATH: str = "data/topics.ms-marco-dev.tsv"
OUTPUT_PATH: str = "results/cql_full_tree_traces_50_msmarco_dev.csv"

RANDOM_SEED = 42
NUM_QUERIES = 50
HIDDEN_DIMS = (256, 128)
BATCH_SIZE = 256
GAMMA = 0.99
LEARNING_RATE = 3e-4
N_CRITICS = 2
TARGET_UPDATE_INTERVAL = 8000
CQL_ALPHA = 1.0

SIM_MAX_STEPS = 3
SIM_TOP_K_RERANK = 50
SIM_TOP_K_PRF = 10
SIM_NDCG_K = 50
SIM_RECALL_K = 100


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_qrels(qrels_path: str) -> Dict[str, Dict[str, int]]:
    """Load qrels in TSV format: query_id iteration doc_id relevance_grade."""
    from collections import defaultdict

    qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
    with open(qrels_path, "r", encoding="utf-8") as f:
        next(f, None)  # skip header
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
    """Load queries as a list of (query_id, query_text)."""
    queries: List[Tuple[str, str]] = []
    with open(queries_path, "r", encoding="utf-8") as f:
        next(f, None)  # skip header
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


def load_initial_retrieval(
    query: str,
    retriever_func: callable,
    top_k: int = 50,
    max_retries: int = 3,
) -> Tuple[List[str], np.ndarray, Dict[str, str]]:
    """Retrieve initial BM25 results for one query with retries."""
    import time

    last_error: Exception = RuntimeError("Retriever returned no documents")
    for attempt in range(max_retries):
        doc_ids, doc_scores, corpus_data = retriever_func(query, top_k)
        if doc_ids is not None and len(doc_ids) > 0:
            return doc_ids, doc_scores, corpus_data
        if attempt < max_retries - 1:
            wait = 2 ** attempt
            logger.warning("Retriever returned no documents for query %r (attempt %d/%d); retrying in %ds", query, attempt + 1, max_retries, wait)
            time.sleep(wait)
    raise last_error


def load_cql_model(model_path: str, device: str):
    """Rebuild a DiscreteCQL algo and load trained weights."""
    config = DiscreteCQLConfig(
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        learning_rate=LEARNING_RATE,
        encoder_factory=VectorEncoderFactory(hidden_units=list(HIDDEN_DIMS)),
        n_critics=N_CRITICS,
        target_update_interval=TARGET_UPDATE_INTERVAL,
        alpha=CQL_ALPHA,
    )
    algo = config.create(device=device)

    # Build the network with a dummy dataset that has the correct shapes.
    dummy_obs = np.zeros((10, 399), dtype=np.float32)
    dummy_actions = np.arange(10, dtype=np.int64) % 4  # ensure all 4 actions appear
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


def main():
    set_seed(RANDOM_SEED)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info("Loading CQL model from %s on %s", MODEL_PATH, device)
    policy_model = load_cql_model(MODEL_PATH, device)

    logger.info("Loading qrels from %s", QRELS_PATH)
    qrels = load_qrels(QRELS_PATH)

    logger.info("Loading queries from %s", QUERIES_PATH)
    queries = load_queries(QUERIES_PATH)

    judged_queries = [
        (qid, qtext)
        for qid, qtext in queries
        if qid in qrels and qrels[qid]
    ]
    logger.info("Found %d judged queries", len(judged_queries))

    rng = random.Random(RANDOM_SEED)
    shuffled = list(judged_queries)
    rng.shuffle(shuffled)
    logger.info("Shuffled %d judged queries; will stop after %d successful rollouts", len(shuffled), NUM_QUERIES)

    logger.info("Loading query encoder: all-MiniLM-L6-v2")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    logger.info("Initializing ChatNoir BM25 retriever")
    retriever_instance = Retriever()
    retriever = create_retriever_callable(retriever_instance)

    qr_agent = ReformulationAgent(embed_model=encoder)
    rr_agent = RerankingAgent(embed_model=encoder)
    prf_agent = PRFAgent(embed_model=encoder, num_expansion_terms=5)

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
        policy_model=policy_model,
    )

    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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

    temporary_path = output_path.with_suffix(".tmp")
    rows_written = 0
    queries_failed = 0
    queries_succeeded = 0
    attempted = 0

    with open(temporary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for query_id, query in shuffled:
            if queries_succeeded >= NUM_QUERIES:
                break
            attempted += 1
            logger.info("[%d] Rolling out query_id=%s | %s", attempted, query_id, query[:80])
            try:
                doc_ids, doc_scores, corpus_data = load_initial_retrieval(
                    query=query,
                    retriever_func=retriever,
                    top_k=config.top_k_rerank,
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

                queries_succeeded += 1
                logger.info("[%d] SAVED %d transitions (success %d/%d)", attempted, len(trajectory), queries_succeeded, NUM_QUERIES)

            except Exception as e:
                queries_failed += 1
                logger.exception("[%d] FAILED query_id=%s: %s", attempted, query_id, e)

        f.flush()

    import os
    os.replace(temporary_path, output_path)

    logger.info("=" * 70)
    logger.info("CQL rollout complete")
    logger.info("Output: %s", output_path)
    logger.info("Total rows written: %d", rows_written)
    logger.info("Queries succeeded: %d", queries_succeeded)
    logger.info("Queries failed: %d", queries_failed)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
