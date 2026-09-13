#!/usr/bin/env python3
"""
MAESTRO: Multi-Agent Ensemble for Search Through Reinforcement Optimization

Offline RL trajectory collection with resumable, per-query checkpoints.

Each successfully generated trajectory is saved as:
    <experiment-dir>/checkpoints/<query-id-hash>.json

The experiment directory must contain an ``experiment.json`` manifest before
this script is invoked. If the program stops, rerun the same command.
Completed query IDs are detected from checkpoint files and skipped automatically.
"""

import os
import sys
import csv
import json
import logging
import argparse
import hashlib
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import defaultdict

import urllib3
import numpy as np
from d3rlpy.dataset import MDPDataset
from sentence_transformers import SentenceTransformer


# -----------------------------------------------------------------------------
# Logging and paths
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

src_path = Path(__file__).resolve().parent
root_path = src_path.parent

sys.path.insert(0, str(root_path))
sys.path.insert(0, str(src_path))


# -----------------------------------------------------------------------------
# Environment loading
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Project imports
# -----------------------------------------------------------------------------

try:
    from .simulation import Simulation, SimConfig, Transition, ACTION_STOP, ACTION_NAMES
    from .core.agents import AgentBase
    from .agents.reformulate import ReformulationAgent
    from .agents.rerank import RerankingAgent
    from .agents.prf import PRFAgent
    from .utils.retriever import Retriever, create_retriever_callable
    from .utils.notifications import send_email
    from .experiments import (
        require_experiment_dir,
        log_artifact,
        EXPERIMENTS_DIR,
    )

except ImportError:
    from simulation import Simulation, SimConfig, Transition, ACTION_STOP, ACTION_NAMES
    from core.agents import AgentBase
    from agents.reformulate import ReformulationAgent
    from agents.rerank import RerankingAgent
    from agents.prf import PRFAgent
    from utils.retriever import Retriever, create_retriever_callable
    from utils.notifications import send_email
    from experiments import (
        require_experiment_dir,
        log_artifact,
        EXPERIMENTS_DIR,
    )


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def load_qrels(qrels_path: str) -> Dict[str, Dict[str, int]]:
    """
    Load qrels in format such as:

        query_id <tab> iteration <tab> doc_id <tab> relevance_grade
    """
    qrels = defaultdict(dict)

    with open(qrels_path, "r", encoding="utf-8") as f:
        next(f, None)  # Skip header.

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

    logger.info("Loaded qrels for %d queries from %s", len(qrels), qrels_path)
    return dict(qrels)


def load_queries(
    queries_path: str,
    num_queries: Optional[int] = None,
) -> List[Tuple[str, str]]:
    """Load queries as a list of (query_id, query_text)."""
    queries = []

    with open(queries_path, "r", encoding="utf-8") as f:
        next(f, None)  # Skip header.

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

            if num_queries is not None and len(queries) >= num_queries:
                break

    logger.info("Loaded %d queries from %s", len(queries), queries_path)
    return queries


def load_initial_retrieval(
    query: str,
    retriever_func: callable,
    top_k: int = 50,
) -> Tuple[List[str], np.ndarray, Dict[str, str]]:
    """Retrieve initial BM25/OpenSearch results for one query."""
    doc_ids, doc_scores, corpus_data = retriever_func(query, top_k)

    if doc_ids is None or len(doc_ids) == 0:
        raise RuntimeError("Retriever returned no documents")

    return doc_ids, doc_scores, corpus_data


# -----------------------------------------------------------------------------
# Checkpoint helpers
# -----------------------------------------------------------------------------

def make_run_name(args: argparse.Namespace) -> str:
    """
    Separate checkpoints when important generation settings differ.

    Do not resume the same directory after changing policy, max steps,
    or reranking/PRF parameters.
    """
    return (
        f"policy-{args.policy}"
        f"_n-{args.num_trajectories}"
        f"_steps-{args.max_steps}"
        f"_rerank-{args.top_k_rerank}"
        f"_prf-{args.top_k_prf}"
        f"_ndcg-{args.ndcg_k}"
        f"_recall-{args.recall_k}"
    )


def resolve_experiment_dir(args: argparse.Namespace) -> Path:
    """Return the experiment directory from --experiment-dir or deprecated --run-name."""
    if args.experiment_dir:
        return Path(args.experiment_dir)

    if args.run_name:
        logger.warning(
            "--run-name is deprecated and will be removed. "
            "Use --experiment-dir instead."
        )
        return EXPERIMENTS_DIR / args.run_name

    raise ValueError(
        "Either --experiment-dir or the deprecated --run-name must be provided."
    )


def checkpoint_path(
    checkpoint_dir: Path,
    query_id: str,
    branch_id: str = "main",
) -> Path:
    """
    Generate a deterministic, filesystem-safe checkpoint filename.

    Query IDs can contain characters unsuitable for filenames, so a
    SHA-256 prefix is used while the original ID remains inside JSON.
    The optional branch_id supports multiple trajectories per query.
    """
    query_hash = hashlib.sha256(query_id.encode("utf-8")).hexdigest()[:20]
    safe_branch = branch_id.replace(" ", "_")
    return checkpoint_dir / f"{query_hash}_{safe_branch}.json"


def transition_to_dict(
    transition: Transition,
    step: int,
    max_steps: int,
) -> Dict[str, Any]:
    """Convert one Transition into JSON-serializable checkpoint data."""
    is_terminal = transition.action == ACTION_STOP
    is_timeout = step == max_steps - 1 and not is_terminal
    info = transition.info
    state = transition.state

    return {
        # Classic offline RL fields (kept for d3rlpy / MDPDataset reconstruction).
        "state": np.asarray(state, dtype=np.float32).tolist(),
        "action": int(transition.action),
        "reward": float(transition.reward),
        "terminal": int(is_terminal),
        "timeout": int(is_timeout),

        # Reward-objective fields for human-readable CSV export.
        "ndcg_before": float(info.get("ndcg_before", 0.0)),
        "ndcg_after": float(info.get("ndcg_after", 0.0)),
        "recall_before": float(info.get("recall_before", 0.0)),
        "recall_after": float(info.get("recall_after", 0.0)),
        "cost": float(info.get("cost", 0.0)),
        "latency_ms": float(info.get("elapsed_ms", 0.0)),
        "cumulative_cost": float(info.get("cum_cost", 0.0)),
        "cumulative_latency_ms": float(info.get("cum_elapsed_ms", 0.0)),

        # Query / action metadata.
        "new_query": info.get("new_query", ""),
        "action_name": info.get("action_name", ACTION_NAMES.get(transition.action, "")),

        # Observable state features (embedding excluded).
        "query_length": float(state[0]),
        "score_spread": float(state[385]),
        "score_entropy": float(state[386]),
        # 1-based step number: which action number this is in the trajectory.
        "step": int(step) + 1,

        # Post-action values from info, so they align with the action row:
        # - rank_overlap: overlap between the ranking before and after this
        #   action (1.0 for STOP, since the ranking is unchanged). state[391]
        #   is the pre-action feature, which is 0.0 at step 0 by definition.
        # - query_drift: drift of new_query vs the original query, caused by
        #   this action. state[392] only shows it in the following row.
        "rank_overlap": float(info.get("rank_overlap", state[391])),
        "query_drift": float(info.get("query_drift", state[392])),
    }


def trajectory_to_checkpoint(
    query_id: str,
    query: str,
    trajectory: List[Transition],
    config: SimConfig,
    branch_id: str = "main",
) -> Dict[str, Any]:
    """Build the durable record for one fully completed trajectory."""
    return {
        "query_id": query_id,
        "query": query,
        "branch_id": branch_id,
        "num_steps": len(trajectory),
        "transitions": [
            transition_to_dict(
                transition=transition,
                step=step,
                max_steps=config.max_steps,
            )
            for step, transition in enumerate(trajectory)
        ],
    }


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    """
    Safely write a checkpoint.

    The final .json path appears only after the complete JSON content was
    written, flushed, and atomically replaced.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, temporary_filename = tempfile.mkstemp(
        prefix=f".{path.stem}_",
        suffix=".tmp",
        dir=path.parent,
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())

        os.replace(temporary_filename, path)

    except Exception:
        try:
            os.unlink(temporary_filename)
        except FileNotFoundError:
            pass

        raise


def read_checkpoint(path: Path) -> Dict[str, Any]:
    """Read and minimally validate one checkpoint."""
    with open(path, "r", encoding="utf-8") as f:
        record = json.load(f)

    required_keys = {"query_id", "query", "num_steps", "transitions"}

    if not required_keys.issubset(record):
        missing = required_keys - set(record.keys())
        raise ValueError(f"Missing checkpoint fields: {sorted(missing)}")

    if not isinstance(record["transitions"], list):
        raise ValueError("'transitions' must be a list")

    return record


def find_completed_query_ids(checkpoint_dir: Path) -> Set[str]:
    """
    Return IDs with valid, fully committed checkpoint files.

    Invalid JSON is ignored and that query is regenerated on resume.
    """
    completed_query_ids: Set[str] = set()

    if not checkpoint_dir.exists():
        return completed_query_ids

    for path in checkpoint_dir.glob("*.json"):
        try:
            record = read_checkpoint(path)
            completed_query_ids.add(record["query_id"])
        except (OSError, json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning("Ignoring invalid checkpoint %s: %s", path, e)

    return completed_query_ids


def find_completed_branches(checkpoint_dir: Path, query_id: str) -> Set[str]:
    """
    Return branch IDs that have a valid checkpoint for a given query.

    Used by the `tree` policy, which writes one checkpoint per valid first
    action (e.g. STOP, QueryReform, Rerank, PseudoRelevanceFeedback).
    """
    completed_branches: Set[str] = set()

    if not checkpoint_dir.exists():
        return completed_branches

    query_hash = hashlib.sha256(query_id.encode("utf-8")).hexdigest()[:20]
    for path in checkpoint_dir.glob(f"{query_hash}_*.json"):
        try:
            record = read_checkpoint(path)
            if record.get("query_id") == query_id:
                completed_branches.add(record.get("branch_id", "main"))
        except (OSError, json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning("Ignoring invalid checkpoint %s: %s", path, e)

    return completed_branches


def iter_checkpoint_records(checkpoint_dir: Path):
    """Yield valid checkpoint records sorted by filename."""
    for path in sorted(checkpoint_dir.glob("*.json")):
        try:
            yield read_checkpoint(path)
        except (OSError, json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning("Skipping invalid checkpoint %s: %s", path, e)


# -----------------------------------------------------------------------------
# Final artifact export
# -----------------------------------------------------------------------------

def build_mdp_dataset_from_checkpoints(checkpoint_dir: Path) -> MDPDataset:
    """
    Reconstruct an MDPDataset from all durable checkpoints.

    This is intentionally done from checkpoints so mdp_dataset.h5 can always
    be recreated, even if the program stops during its final export.
    """
    observations = []
    actions = []
    rewards = []
    terminals = []
    timeouts = []

    for record in iter_checkpoint_records(checkpoint_dir):
        for transition in record["transitions"]:
            observations.append(transition["state"])
            actions.append(transition["action"])
            rewards.append(transition["reward"])
            terminals.append(transition["terminal"])
            timeouts.append(transition["timeout"])

    if not observations:
        raise RuntimeError(
            f"No completed trajectory checkpoints found in: {checkpoint_dir}"
        )

    return MDPDataset(
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.int64),
        rewards=np.asarray(rewards, dtype=np.float32),
        terminals=np.asarray(terminals, dtype=np.int64),
        timeouts=np.asarray(timeouts, dtype=np.int64),
    )


def export_checkpoints_to_csv(checkpoint_dir: Path, csv_path: Path) -> None:
    """
    Export every completed trajectory checkpoint into one human-readable CSV.

    The HDF5 output keeps the compact state/action/reward/terminal/timeout
    format required by d3rlpy; this CSV is for analysis only.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        # Query / action metadata.
        "query_id",
        "query",
        "new_query",
        "action_name",
        # Observable state features (embedding excluded).
        "query_length",
        "score_spread",
        "score_entropy",
        "step",
        "rank_overlap",
        "query_drift",
        # Reward objectives.
        "ndcg_before",
        "ndcg_after",
        "recall_before",
        "recall_after",
        "cost",
        "latency_ms",
        "cumulative_cost",
        "cumulative_latency_ms",
        "reward",
        # Episode markers.
        "terminal",
        "timeout",
    ]

    temporary_path = csv_path.with_suffix(".tmp")

    def state_feature(state: Any, index: int, default: float = 0.0) -> float:
        """Read one scalar feature from a stored state vector, if present."""
        try:
            return float(state[index])
        except (TypeError, IndexError, ValueError):
            return default

    def clean_number(value: Any, default: float = 0.0) -> float:
        """Coerce to float, replacing missing/NaN/inf with a finite default."""
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default
        if not np.isfinite(number):
            return default
        return number

    legacy_rows = 0

    with open(temporary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for record in iter_checkpoint_records(checkpoint_dir):
            for step_idx, transition in enumerate(record["transitions"]):
                state = transition.get("state")
                action = transition.get("action")

                # Legacy checkpoints (only state/action/reward/terminal/timeout)
                # lack the diagnostic fields. Recover what is possible from the
                # stored state vector and fill the rest with finite defaults so
                # the CSV never contains empty/NaN cells.
                if "ndcg_before" not in transition:
                    legacy_rows += 1

                row = {
                    "query_id": record["query_id"],
                    "query": record["query"],
                    "new_query": transition.get("new_query") or record["query"],
                    "action_name": (
                        transition.get("action_name")
                        or ACTION_NAMES.get(action, "")
                    ),
                    # Observable state features (recoverable from `state`).
                    "query_length": clean_number(
                        transition.get("query_length", state_feature(state, 0))
                    ),
                    "score_spread": clean_number(
                        transition.get("score_spread", state_feature(state, 385))
                    ),
                    "score_entropy": clean_number(
                        transition.get("score_entropy", state_feature(state, 386))
                    ),
                    # 1-based step number for the current trajectory. The
                    # stored state vector holds the *normalized pre-action*
                    # step index, so we use the transition order instead.
                    "step": step_idx + 1,
                    "rank_overlap": clean_number(
                        transition.get("rank_overlap", state_feature(state, 391))
                    ),
                    "query_drift": clean_number(
                        transition.get("query_drift", state_feature(state, 392))
                    ),
                    # Reward objectives (NOT stored in `state` by design; legacy
                    # checkpoints cannot recover them and get 0.0 placeholders).
                    "ndcg_before": clean_number(transition.get("ndcg_before")),
                    "ndcg_after": clean_number(transition.get("ndcg_after")),
                    "recall_before": clean_number(transition.get("recall_before")),
                    "recall_after": clean_number(transition.get("recall_after")),
                    "cost": clean_number(transition.get("cost")),
                    "latency_ms": clean_number(transition.get("latency_ms")),
                    "cumulative_cost": clean_number(
                        transition.get("cumulative_cost")
                    ),
                    "cumulative_latency_ms": clean_number(
                        transition.get("cumulative_latency_ms")
                    ),
                    "reward": clean_number(transition.get("reward")),
                    # Episode markers.
                    "terminal": int(transition.get("terminal", 0)),
                    "timeout": int(transition.get("timeout", 0)),
                }
                writer.writerow(row)

        f.flush()
        os.fsync(f.fileno())

    os.replace(temporary_path, csv_path)

    if legacy_rows:
        logger.warning(
            "%d rows came from legacy checkpoints without diagnostic fields. "
            "ndcg_*/recall_*/cost/latency were exported as 0.0 placeholders; "
            "regenerate those runs to obtain real values.",
            legacy_rows,
        )


def count_checkpoint_stats(checkpoint_dir: Path) -> Tuple[int, int]:
    """Return (number_of_trajectories, total_number_of_transitions)."""
    num_trajectories = 0
    total_transitions = 0

    for record in iter_checkpoint_records(checkpoint_dir):
        num_trajectories += 1
        total_transitions += len(record["transitions"])

    return num_trajectories, total_transitions


# -----------------------------------------------------------------------------
# Trajectory generation
# -----------------------------------------------------------------------------

def generate_trajectories(
    config: SimConfig,
    qrels: Dict[str, Dict[str, int]],
    queries: List[Tuple[str, str]],
    encoder: SentenceTransformer,
    agents: List[AgentBase],
    retriever: callable,
    num_trajectories: int,
    policy: str,
    checkpoint_dir: Path,
) -> None:
    """
    Generate missing trajectories only.

    Every successful query is checkpointed immediately. This function does
    not need to keep all trajectories in RAM for the full multi-day run.
    """
    logger.info("=" * 70)
    logger.info("TRAJECTORY GENERATION PIPELINE")
    logger.info("=" * 70)
    if policy == "tree":
        logger.info(
            "Requested queries: %d (tree policy yields up to 4 trajectories per query)",
            num_trajectories,
        )
    else:
        logger.info("Requested trajectories: %d", num_trajectories)
    logger.info("Policy: %s", policy)
    logger.info("Checkpoint directory: %s", checkpoint_dir)
    logger.info(
        "Config: max_steps=%d, top_k_rerank=%d",
        config.max_steps,
        config.top_k_rerank,
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    completed_query_ids = find_completed_query_ids(checkpoint_dir)
    logger.info(
        "Resume mode: found %d completed trajectories",
        len(completed_query_ids),
    )

    sim = Simulation(
        encoder=encoder,
        retriever=retriever,
        agents=agents,
        config=config,
    )

    # Only generate trajectories for queries that have relevance judgments.
    # Unjudged queries produce all-zero NDCG/recall/reward and teach the policy
    # nothing useful.
    judged_queries = [
        (query_id, query)
        for query_id, query in queries
        if query_id in qrels and qrels[query_id]
    ]
    selected_queries = judged_queries[: min(num_trajectories, len(judged_queries))]
    generated_this_run = 0
    skipped_this_run = 0
    failed_this_run = 0

    for trajectory_index, (query_id, query) in enumerate(selected_queries, start=1):
        # ── Tree policy: one checkpoint per valid first action ──
        if policy == "tree":
            expected_branches = {ACTION_NAMES[a] for a in range(4)}
            completed_branches = find_completed_branches(checkpoint_dir, query_id)

            if completed_branches >= expected_branches:
                skipped_this_run += 1
                logger.info(
                    "[%d/%d] SKIP all branches completed query_id=%s",
                    trajectory_index,
                    len(selected_queries),
                    query_id,
                )
                continue

            logger.info(
                "[%d/%d] Generating tree branches query_id=%s | %s",
                trajectory_index,
                len(selected_queries),
                query_id,
                query[:80],
            )

            try:
                doc_ids, doc_scores, corpus_data = load_initial_retrieval(
                    query=query,
                    retriever_func=retriever,
                    top_k=config.top_k_rerank,
                )

                qrels_for_query = qrels.get(query_id, {})

                branch_trajectories = sim.generate_expert_branches(
                    query=query,
                    doc_ids=doc_ids,
                    doc_scores=doc_scores,
                    qrels=qrels_for_query,
                    corpus_data=corpus_data,
                )

                saved_any = False
                for action, traj in enumerate(branch_trajectories):
                    branch_name = ACTION_NAMES[action]
                    if branch_name in completed_branches:
                        continue

                    if traj is None or not traj:
                        logger.warning(
                            "Empty/missing branch %s for query_id=%s; skipping",
                            branch_name,
                            query_id,
                        )
                        continue

                    checkpoint = trajectory_to_checkpoint(
                        query_id=query_id,
                        query=query,
                        trajectory=traj,
                        config=config,
                        branch_id=branch_name,
                    )
                    output_path = checkpoint_path(
                        checkpoint_dir, query_id, branch_id=branch_name
                    )
                    atomic_write_json(output_path, checkpoint)

                    saved_any = True
                    generated_this_run += 1
                    logger.info(
                        "[%d/%d] SAVED branch=%s %d transitions -> %s",
                        trajectory_index,
                        len(selected_queries),
                        branch_name,
                        len(traj),
                        output_path.name,
                    )

                if not saved_any:
                    logger.info(
                        "[%d/%d] All branches already existed query_id=%s",
                        trajectory_index,
                        len(selected_queries),
                        query_id,
                    )

                total_completed = generated_this_run + skipped_this_run
                if total_completed > 0 and total_completed % 1000 == 0:
                    send_email(
                        subject=f"[MAESTRO] Milestone: {total_completed} trajectories completed",
                        body=(
                            f"Policy: {policy}\n"
                            f"Run: {checkpoint_dir.name}\n"
                            f"Total completed: {total_completed}\n"
                            f"Latest query: {query_id}\n"
                            f"Timestamp: {datetime.now().isoformat()}"
                        ),
                    )

            except KeyboardInterrupt:
                logger.warning(
                    "Interrupted. Previously saved checkpoints are safe. "
                    "Run the same command again to resume."
                )
                raise

            except Exception as e:
                failed_this_run += 1
                logger.exception(
                    "[%d/%d] FAILED tree branches query_id=%s: %s",
                    trajectory_index,
                    len(selected_queries),
                    query_id,
                    e,
                )
                send_email(
                    subject=f"[MAESTRO] Failure: query {query_id} failed",
                    body=(
                        f"Policy: {policy}\n"
                        f"Run: {checkpoint_dir.name}\n"
                        f"Query ID: {query_id}\n"
                        f"Query text: {query[:200]}\n"
                        f"Error: {e}\n"
                        f"Timestamp: {datetime.now().isoformat()}"
                    ),
                )

            continue

        # ── All other policies: single trajectory per query ──
        output_path = checkpoint_path(checkpoint_dir, query_id)

        if query_id in completed_query_ids and output_path.exists():
            skipped_this_run += 1
            logger.info(
                "[%d/%d] SKIP completed query_id=%s",
                trajectory_index,
                len(selected_queries),
                query_id,
            )
            continue

        logger.info(
            "[%d/%d] Generating query_id=%s | %s",
            trajectory_index,
            len(selected_queries),
            query_id,
            query[:80],
        )

        try:
            doc_ids, doc_scores, corpus_data = load_initial_retrieval(
                query=query,
                retriever_func=retriever,
                top_k=config.top_k_rerank,
            )

            qrels_for_query = qrels.get(query_id, {})

            trajectory = sim.generate_trajectory(
                query=query,
                doc_ids=doc_ids,
                doc_scores=doc_scores,
                qrels=qrels_for_query,
                policy=policy,
                corpus_data=corpus_data,
            )

            if not trajectory:
                raise RuntimeError("Simulation returned an empty trajectory")

            checkpoint = trajectory_to_checkpoint(
                query_id=query_id,
                query=query,
                trajectory=trajectory,
                config=config,
            )

            atomic_write_json(output_path, checkpoint)

            completed_query_ids.add(query_id)
            generated_this_run += 1

            total_completed = len(completed_query_ids)
            if total_completed > 0 and total_completed % 1000 == 0:
                send_email(
                    subject=f"[MAESTRO] Milestone: {total_completed} trajectories completed",
                    body=(
                        f"Policy: {policy}\n"
                        f"Run: {checkpoint_dir.name}\n"
                        f"Total completed: {total_completed}\n"
                        f"Latest query: {query_id}\n"
                        f"Timestamp: {datetime.now().isoformat()}"
                    ),
                )

            logger.info(
                "[%d/%d] SAVED %d transitions -> %s",
                trajectory_index,
                len(selected_queries),
                len(trajectory),
                output_path.name,
            )

        except KeyboardInterrupt:
            logger.warning(
                "Interrupted. Previously saved checkpoints are safe. "
                "Run the same command again to resume."
            )
            raise

        except Exception as e:
            failed_this_run += 1
            logger.exception(
                "[%d/%d] FAILED query_id=%s: %s",
                trajectory_index,
                len(selected_queries),
                query_id,
                e,
            )
            send_email(
                subject=f"[MAESTRO] Failure: query {query_id} failed",
                body=(
                    f"Policy: {policy}\n"
                    f"Run: {checkpoint_dir.name}\n"
                    f"Query ID: {query_id}\n"
                    f"Query text: {query[:200]}\n"
                    f"Error: {e}\n"
                    f"Timestamp: {datetime.now().isoformat()}"
                ),
            )

    logger.info("=" * 70)
    logger.info("Generation run finished")
    logger.info("Generated this run: %d", generated_this_run)
    logger.info("Skipped from checkpoint: %d", skipped_this_run)
    logger.info("Failed this run: %d", failed_this_run)
    logger.info("=" * 70)


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    """Load components, resume generation, and export final artifacts."""
    logger.info("Starting MAESTRO offline RL trajectory collection")

    # -------------------------------------------------------------------------
    # Step 0: Validate experiment directory
    # -------------------------------------------------------------------------

    experiment_dir = resolve_experiment_dir(args)
    if not experiment_dir.is_absolute():
        experiment_dir = root_path / experiment_dir

    experiment = require_experiment_dir(experiment_dir)
    logger.info(
        "Experiment: %s (%s)", experiment.slug, experiment.purpose
    )

    # -------------------------------------------------------------------------
    # Step 1: Load data
    # -------------------------------------------------------------------------

    logger.info("[Step 1] Loading data")

    qrels_path = Path(args.qrels_path)
    if not qrels_path.is_absolute():
        qrels_path = root_path / qrels_path

    queries_path = Path(args.queries_path)
    if not queries_path.is_absolute():
        queries_path = root_path / queries_path

    qrels = load_qrels(str(qrels_path))
    queries = load_queries(str(queries_path), num_queries=args.num_queries)

    if not queries:
        raise RuntimeError("No queries were loaded")

    # -------------------------------------------------------------------------
    # Step 2: Initialize dependencies
    # -------------------------------------------------------------------------

    logger.info("[Step 2] Initializing components")

    logger.info("Loading query encoder: all-MiniLM-L6-v2")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    logger.info("Initializing OpenSearch BM25 retriever")
    retriever_instance = Retriever()
    retriever = create_retriever_callable(retriever_instance)

    qr_agent = ReformulationAgent(embed_model=encoder)
    rr_agent = RerankingAgent(embed_model=encoder)
    prf_agent = PRFAgent(embed_model=encoder, num_expansion_terms=5)

    agents = [qr_agent, rr_agent, prf_agent]

    config = SimConfig(
        max_steps=args.max_steps,
        top_k_rerank=args.top_k_rerank,
        top_k_prf=args.top_k_prf,
        ndcg_k=args.ndcg_k,
        recall_k=args.recall_k,
        reward_alpha=2.0,
        reward_beta=0.5,
        reward_gamma=0.2,
        reward_delta=0.1,
    )

    # -------------------------------------------------------------------------
    # Step 3: Resume/generate checkpoints
    # -------------------------------------------------------------------------

    checkpoint_dir = experiment_dir / "checkpoints"

    generate_trajectories(
        config=config,
        qrels=qrels,
        queries=queries,
        encoder=encoder,
        agents=agents,
        retriever=retriever,
        num_trajectories=args.num_trajectories,
        policy=args.policy,
        checkpoint_dir=checkpoint_dir,
    )

    # -------------------------------------------------------------------------
    # Step 4: Rebuild final artifacts from durable checkpoints
    # -------------------------------------------------------------------------

    logger.info("[Step 4] Rebuilding final outputs from checkpoints")

    experiment_dir.mkdir(parents=True, exist_ok=True)

    csv_path = experiment_dir / "trajectories.csv"
    h5_path = experiment_dir / "dataset.h5"

    export_checkpoints_to_csv(checkpoint_dir, csv_path)

    mdp_dataset = build_mdp_dataset_from_checkpoints(checkpoint_dir)
    mdp_dataset.dump(str(h5_path))

    # Record produced artifacts in the manifest. The manifest update is
    # atomic; a crash here leaves the artifacts on disk, just unlisted.
    log_artifact(experiment.slug, "dataset", "dataset.h5")
    log_artifact(experiment.slug, "trajectories", "trajectories.csv")

    completed_trajectories, total_transitions = count_checkpoint_stats(
        checkpoint_dir
    )

    expected_trajectories = min(args.num_trajectories, len(queries))

    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info("Completed trajectories: %d / %d", completed_trajectories, expected_trajectories)
    logger.info("Total transitions: %d", total_transitions)
    logger.info("Queries loaded: %d", len(queries))
    logger.info("Qrels loaded: %d", len(qrels))
    logger.info("Checkpoint directory: %s", checkpoint_dir)
    logger.info("CSV output: %s", csv_path)
    logger.info("HDF5 output: %s", h5_path)
    logger.info("=" * 70)

    if completed_trajectories < expected_trajectories:
        logger.warning(
            "Some trajectories are still missing. Run the exact same command "
            "again to retry failed or unfinished queries."
        )
    else:
        logger.info("Pipeline complete. All requested trajectories exist.")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate resumable offline RL trajectories for MAESTRO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--qrels-path",
        type=str,
        default="notebooks/qrels/qrels.ms-marco-dev2.tsv",
        help="Path to qrels file",
    )
    parser.add_argument(
        "--queries-path",
        type=str,
        default="notebooks/queries/topics.ms-marco-dev2.tsv",
        help="Path to queries file",
    )

    parser.add_argument(
        "--num-trajectories",
        type=int,
        default=10,
        help="Number of trajectories to generate",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=None,
        help="Load only the first N queries",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=3,
        help="Maximum steps per trajectory",
    )
    parser.add_argument(
        "--top-k-rerank",
        type=int,
        default=50,
        help="Top-k documents for reranking",
    )
    parser.add_argument(
        "--top-k-prf",
        type=int,
        default=10,
        help="Top-k documents for PRF term extraction",
    )
    parser.add_argument(
        "--ndcg-k",
        type=int,
        default=50,
        help="NDCG evaluation cutoff",
    )
    parser.add_argument(
        "--recall-k",
        type=int,
        default=100,
        help="Recall evaluation cutoff",
    )

    parser.add_argument(
        "--policy",
        type=str,
        default="random",
        choices=["random", "expert", "stop", "prf", "rerank", "tree"],
        help="Action selection policy",
    )

    parser.add_argument(
        "--experiment-dir",
        type=str,
        default=None,
        help=(
            "Path to the experiment directory. The directory must already "
            "contain an experiment.json manifest."
        ),
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help=(
            "Deprecated. Use --experiment-dir instead; if provided without "
            "--experiment-dir, the value is treated as a slug under experiments/."
        ),
    )

    args = parser.parse_args()
    main(args)