r"""Generate baseline trajectories for the same 50 MS MARCO dev queries used by CQL.

Usage:
    poetry run python generate_baseline_traces.py

Produces, for each baseline policy, a CSV in results/:
    - stop_traces_50_msmarco_dev.csv
    - rerank_traces_50_msmarco_dev.csv
    - reformulate_traces_50_msmarco_dev.csv
    - expert_traces_50_msmarco_dev.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
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
CQL_TRACES_PATH: str = "RQs/relevant_csvs/cql_full_tree_traces_50_msmarco_dev.csv"
OUTPUT_DIR: str = "RQs/relevant_csvs"

RANDOM_SEED = 42

SIM_MAX_STEPS = 3
SIM_TOP_K_RERANK = 50
SIM_TOP_K_PRF = 10
SIM_NDCG_K = 50
SIM_RECALL_K = 100

POLICIES = ["stop", "rerank", "reformulate", "expert", "random"]
DEFAULT_QR_FORMULATION = "natural"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_query_ids(csv_path: Path) -> List[str]:
    """Load ordered unique query IDs from a single-column CSV."""
    import pandas as pd
    df = pd.read_csv(csv_path, dtype=str)
    if "query_id" not in df.columns:
        raise ValueError(f"No query_id column in {csv_path}")
    return [str(qid) for qid in df["query_id"].unique()]


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
            logger.warning(
                "Retriever returned no documents for query %r (attempt %d/%d); retrying in %ds",
                query, attempt + 1, max_retries, wait
            )
            time.sleep(wait)
    raise last_error


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


def generate_policy_traces(
    policy: str,
    queries: List[Tuple[str, str]],
    qrels: Dict[str, Dict[str, int]],
    sim: Simulation,
    retriever: callable,
    output_path: Path,
    append: bool = False,
) -> None:
    """Generate and save traces for one baseline policy.

    If ``append`` is True and ``output_path`` already exists, new rows are
    appended to the existing file instead of overwriting it. This is useful
    for resuming a crashed or interrupted run.
    """
    import os

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

    if append and output_path.exists():
        # Resume: append directly to the existing output file.
        file_mode = "a"
        write_header = False
        logger.info("Appending to existing output file: %s", output_path)
    else:
        # Fresh run: write to a temp file and atomically replace the output.
        file_mode = "w"
        write_header = True

    temporary_path = output_path.with_suffix(".tmp") if not append else output_path

    with open(temporary_path, file_mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for idx, (query_id, query) in enumerate(queries, start=1):
            logger.info("[%s] [%d/%d] Rolling out query_id=%s | %s", policy, idx, len(queries), query_id, query[:80])
            try:
                doc_ids, doc_scores, corpus_data = load_initial_retrieval(
                    query=query,
                    retriever_func=retriever,
                    top_k=sim.cfg.top_k_rerank,
                )

                trajectory = sim.generate_trajectory(
                    query=query,
                    doc_ids=doc_ids,
                    doc_scores=doc_scores,
                    qrels=qrels.get(query_id, {}),
                    policy=policy,
                    corpus_data=corpus_data,
                )

                for step_idx, transition in enumerate(trajectory):
                    writer.writerow(transition_to_row(query_id, query, step_idx, transition))
                    rows_written += 1

                f.flush()
                queries_succeeded += 1
                logger.info("[%s] [%d/%d] SAVED %d transitions", policy, idx, len(queries), len(trajectory))

            except Exception as e:
                queries_failed += 1
                logger.exception("[%s] [%d/%d] FAILED query_id=%s: %s", policy, idx, len(queries), query_id, e)

        f.flush()

    if not append:
        os.replace(temporary_path, output_path)

    logger.info("=" * 70)
    logger.info("[%s] baseline rollout complete", policy)
    logger.info("Output: %s", output_path)
    logger.info("Total rows written: %d", rows_written)
    logger.info("Queries succeeded: %d", queries_succeeded)
    logger.info("Queries failed: %d", queries_failed)
    logger.info("=" * 70)


def main(args: argparse.Namespace) -> None:
    set_seed(RANDOM_SEED)

    import pandas as pd

    # Load target query IDs either from an explicit list or from the CQL traces.
    if args.query_ids:
        query_ids_path = Path(args.query_ids)
        if not query_ids_path.exists():
            raise FileNotFoundError(f"Query IDs file not found at {query_ids_path}")
        query_ids = load_query_ids(query_ids_path)
        logger.info("Loaded %d query IDs from %s", len(query_ids), query_ids_path)
    else:
        cql_path = Path(CQL_TRACES_PATH)
        if not cql_path.exists():
            raise FileNotFoundError(f"CQL traces not found at {cql_path}")
        df_cql = pd.read_csv(cql_path)
        query_ids = [str(qid) for qid in df_cql["query_id"].unique().tolist()]
        logger.info("Loaded %d query IDs from %s", len(query_ids), CQL_TRACES_PATH)

    # Skip query IDs already present in the resume-from file.
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume-from file not found at {resume_path}")
        existing_ids = set(load_query_ids(resume_path))
        logger.info("Found %d query IDs already in %s", len(existing_ids), resume_path)
        query_ids = [qid for qid in query_ids if qid not in existing_ids]
        logger.info("%d query IDs remaining to roll out", len(query_ids))

    # Load queries and filter to the selected query IDs.
    queries_path = Path("data/topics.ms-marco-dev.tsv")
    qrels_path = Path("data/qrels.ms-marco-dev.tsv")

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

    selected = [(qid, queries_map[qid]) for qid in query_ids if qid in queries_map and qid in qrels and qrels[qid]]
    if len(selected) < len(query_ids):
        logger.warning("Could only resolve %d/%d query IDs with both text and qrels", len(selected), len(query_ids))

    logger.info("Loading query encoder: all-MiniLM-L6-v2")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    logger.info("Initializing ChatNoir BM25 retriever (msmarco-v2.1)")
    retriever_instance = Retriever(index="msmarco-v2.1")
    retriever = create_retriever_callable(
        retriever_instance,
        adaptive_top_k=True,
        min_top_k=1,
    )

    qr_agent = ReformulationAgent(
        embed_model=encoder,
        formulation_mode=args.qr_formulation,
    )
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
        policy_model=None,
    )

    output_dir = Path(OUTPUT_DIR)
    for policy in args.policies:
        if args.output_path:
            output_path = Path(args.output_path)
        else:
            output_path = output_dir / f"{policy}_traces_50_msmarco_dev.csv"
        generate_policy_traces(
            policy=policy,
            queries=selected,
            qrels=qrels,
            sim=sim,
            retriever=retriever,
            output_path=output_path,
            append=args.append,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate baseline policy traces for MAESTRO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--qr-formulation",
        type=str,
        default=DEFAULT_QR_FORMULATION,
        choices=["natural", "boolean"],
        help="Query-reformulation mode for the reformulate baseline",
    )
    parser.add_argument(
        "--policies",
        type=str,
        nargs="+",
        default=POLICIES,
        choices=POLICIES,
        help="Baseline policies to run",
    )
    parser.add_argument(
        "--query-ids",
        type=str,
        default=None,
        help="Path to a single-column CSV of query IDs to roll out (overrides CQL traces default)",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to an existing traces CSV; query IDs already present are skipped",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Explicit output CSV path (use with a single policy)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to the output file if it already exists (for resuming)",
    )
    args = parser.parse_args()
    if args.output_path and len(args.policies) > 1:
        parser.error("--output-path can only be used with a single policy")
    main(args)
