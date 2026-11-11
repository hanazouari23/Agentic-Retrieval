r"""Generate trajectories using the standalone BC policy on the same 50 MS MARCO dev queries used by CQL.

Usage:
    poetry run python generate_bc_traces.py

Produces:
    results/expert/standalone_bc_full_expert_msmarco_dev.csv
"""
from __future__ import annotations

import csv
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
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
BC_MODEL_PATH: str = "models/standalone_bc_on_full_expert.pt"
CQL_TRACES_PATH: str = "results/expert/cql_full_expert_traces_50_msmarco_dev.csv"
QRELS_PATH: str = "data/qrels.ms-marco-dev.tsv"
QUERIES_PATH: str = "data/topics.ms-marco-dev.tsv"
OUTPUT_PATH: str = "results/expert/standalone_bc_full_expert_msmarco_dev.csv"

RANDOM_SEED = 42
HIDDEN_DIMS = (256, 128)

SIM_MAX_STEPS = 3
SIM_TOP_K_RERANK = 50
SIM_TOP_K_PRF = 10
SIM_NDCG_K = 50
SIM_RECALL_K = 100

RETRIEVER_MAX_RETRIES = 5
QUERY_DELAY_S = 0.5
FAILURE_BACKOFF_S = 5.0


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


def load_queries(queries_path: str) -> Dict[str, str]:
    """Load queries as a dict {query_id: query_text}."""
    queries: Dict[str, str] = {}
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
            queries[query_id] = query_text
    return queries


def load_initial_retrieval(
    query: str,
    retriever_func: callable,
    top_k: int = 50,
    max_retries: int = RETRIEVER_MAX_RETRIES,
) -> Tuple[List[str], np.ndarray, Dict[str, str]]:
    """Retrieve initial BM25 results for one query with retries and backoff."""
    import time

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


class StandaloneBCPolicy:
    """Wrapper around the raw PyTorch BC model so it behaves like d3rlpy's predict API."""

    def __init__(self, model_path: str, device: str):
        from training.train_bc import BCPolicy

        checkpoint = torch.load(model_path, map_location=device)
        # The checkpoint may be the full dict saved by train_bc.py or just state_dict.
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # Infer dimensions from the first layer weight.
        input_dim = int(state_dict["net.0.weight"].shape[1])
        output_dim = int(state_dict[list(state_dict.keys())[-1]].shape[0])

        self.model = BCPolicy(input_dim, HIDDEN_DIMS, output_dim).to(device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.device = device

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Return greedy action indices for a batch of observations."""
        with torch.no_grad():
            obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).to(self.device)
            logits = self.model(obs_t)
            actions = logits.argmax(dim=1).cpu().numpy()
        return actions


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
    logger.info("Loading standalone BC model from %s on %s", BC_MODEL_PATH, device)
    policy_model = StandaloneBCPolicy(BC_MODEL_PATH, device)

    logger.info("Loading qrels from %s", QRELS_PATH)
    qrels = load_qrels(QRELS_PATH)

    logger.info("Loading queries from %s", QUERIES_PATH)
    queries_map = load_queries(QUERIES_PATH)

    # Use the exact same query IDs as the CQL expert traces.
    cql_df = pd.read_csv(CQL_TRACES_PATH)
    target_query_ids = [str(qid) for qid in cql_df["query_id"].unique()]
    logger.info("Loaded %d target query IDs from %s", len(target_query_ids), CQL_TRACES_PATH)

    selected = [
        (qid, queries_map[qid])
        for qid in target_query_ids
        if qid in queries_map and qid in qrels and qrels[qid]
    ]
    if len(selected) < len(target_query_ids):
        logger.warning(
            "Could only resolve %d/%d target query IDs with both text and qrels",
            len(selected), len(target_query_ids)
        )

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

    rows_written = 0
    queries_failed = 0
    queries_succeeded = 0

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, (query_id, query) in enumerate(selected, start=1):
            logger.info("[%d/%d] Rolling out query_id=%s | %s", idx, len(selected), query_id, query[:80])
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
                f.flush()

                queries_succeeded += 1
                logger.info("[%d/%d] SAVED %d transitions", idx, len(selected), len(trajectory))

                if QUERY_DELAY_S > 0:
                    import time
                    time.sleep(QUERY_DELAY_S)

            except Exception as e:
                queries_failed += 1
                logger.exception("[%d/%d] FAILED query_id=%s: %s", idx, len(selected), query_id, e)
                import time
                time.sleep(FAILURE_BACKOFF_S)

    logger.info("=" * 70)
    logger.info("Standalone BC rollout complete")
    logger.info("Output: %s", output_path)
    logger.info("Total rows written: %d", rows_written)
    logger.info("Queries succeeded: %d", queries_succeeded)
    logger.info("Queries failed: %d", queries_failed)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
