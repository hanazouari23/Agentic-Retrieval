"""Standalone Boolean-formulation retrieval experiment.

This script formulates each query into Boolean form using the existing
BooleanFormulationAgent, retrieves with ChatNoir using the Boolean query,
and records one trace row per requested query ID (including failures).

Usage:
    .venv\\Scripts\\python.exe experiments\\new_setup\\run_boolean_formulate_experiment.py ^
        --query-source results\\stop_traces_50_msmarco_dev.csv ^
        --topics-file data\\topics.ms-marco-dev.tsv ^
        --qrels-file data\\qrels.ms-marco-dev.tsv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import openai
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables before importing project modules that read them at
# import time.
load_dotenv()

# Make project sources importable when running from the repository root.
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.simulation import Simulation  # noqa: E402
from src.agents.boolean_formulate import (  # noqa: E402
    BooleanFormulationAgent,
    BooleanFormulationError,
    parse_and_render,
    BOOLEAN_SYSTEM_PROMPT,
    BOOLEAN_USER_PROMPT_TEMPLATE,
)

# Reuse query-loading helpers from the original-query cache script.
from experiments.new_setup.cache_original_retrieval import (  # noqa: E402
    load_query_source,
    load_topics_map,
    select_queries,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


CHATNOIR_API_URL = os.getenv(
    "CHATNOIR_API_URL",
    "https://www.chatnoir.eu/api/v1/_search",
).strip()
CHATNOIR_API_KEY = os.getenv("CHATNOIR_API_KEY", "").strip()

LLM_MODEL_NAME = os.getenv("MODEL_NAME_HPC", "").strip()
LLM_BASE_URL = os.getenv("BASE_URL_HPC", "").strip()

DEFAULT_INDEX = "msmarco-v2.1"
DEFAULT_SEARCH_METHOD = "bm25"
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_NDCG_K = 50
DEFAULT_RECALL_K = 100
DEFAULT_TOP_K = 50
OUTPUT_DIR = Path("results/new_setup/boolean_formulate")


LLM_RETRY_EXCEPTIONS = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
    openai.APIError,
)


# ---------------------------------------------------------------------------
# Qrels and evaluation
# ---------------------------------------------------------------------------


def load_qrels(path: Path) -> Dict[str, Dict[str, int]]:
    """Load a TSV qrels file into {query_id: {doc_id: relevance}}."""
    qrels: Dict[str, Dict[str, int]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            qid = str(row["query_id"]).strip()
            doc_id = str(row["doc_id"]).strip()
            rel = int(row["relevance"])
            qrels.setdefault(qid, {})[doc_id] = rel
    return qrels


def evaluate(
    doc_ids: List[str],
    qrels: Dict[str, int],
    ndcg_k: int = DEFAULT_NDCG_K,
    recall_k: int = DEFAULT_RECALL_K,
) -> Dict[str, float]:
    """Compute NDCG and recall for a single ranking."""
    return {
        f"ndcg@{ndcg_k}": Simulation.compute_ndcg(doc_ids, qrels, ndcg_k),
        f"recall@{recall_k}": Simulation.compute_recall(doc_ids, qrels, recall_k),
    }


# ---------------------------------------------------------------------------
# Boolean formulation with retries
# ---------------------------------------------------------------------------


def formulate_boolean(
    query_text: str,
    agent: BooleanFormulationAgent,
    max_attempts: int,
) -> Tuple[Optional[str], Optional[Dict[str, Any]], int, float, Optional[str]]:
    """Formulate a Boolean query, retrying only transient LLM failures.

    Returns (boolean_query, concepts, attempts_used, elapsed_ms, error_message).
    """
    last_error: Optional[str] = None
    attempts = 0
    t0 = time.perf_counter()

    for attempt in range(max_attempts):
        attempts = attempt + 1
        try:
            raw = agent._call_llm(query_text)  # type: ignore[misc]
            boolean_query, concepts = parse_and_render(raw)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return boolean_query, concepts, attempts, elapsed_ms, None
        except LLM_RETRY_EXCEPTIONS as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < max_attempts - 1:
                wait = 2 ** attempt
                logger.warning(
                    "Transient LLM failure (attempt %d/%d) for query %r; retrying in %ds: %s",
                    attempts,
                    max_attempts,
                    query_text[:80],
                    wait,
                    last_error,
                )
                time.sleep(wait)
                continue
            break
        except BooleanFormulationError as exc:
            # Non-transient: malformed/invalid LLM output. Do not retry.
            last_error = f"BooleanFormulationError: {exc}"
            break
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            break

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return None, None, attempts, elapsed_ms, last_error


# ---------------------------------------------------------------------------
# Retrieval with retries
# ---------------------------------------------------------------------------


def _is_transient(exc: requests.RequestException) -> bool:
    if exc.response is not None:
        return exc.response.status_code >= 500 or exc.response.status_code == 429
    return True


def _adaptive_sizes(top_k: int, min_top_k: int) -> List[int]:
    """Return a descending list of retrieval depths to try on 500 errors."""
    sizes: List[int] = []
    size = top_k
    while size >= min_top_k:
        sizes.append(size)
        size = size // 2
    if min_top_k not in sizes:
        sizes.append(min_top_k)
    return sizes


def retrieve_boolean(
    boolean_query: str,
    top_k: int,
    index: str,
    search_method: str,
    timeout: float,
    max_attempts: int,
    adaptive_top_k: bool = False,
    min_top_k: int = 1,
) -> Tuple[List[str], np.ndarray, Dict[str, str], int, int, float, Optional[str]]:
    """Retrieve using the Boolean query, retrying transient failures.

    If ``adaptive_top_k`` is True and ChatNoir returns a 500, the requested
    ``size`` is halved down to ``min_top_k`` before giving up. This works
    around ChatNoir's behaviour of returning 500 when ``size`` exceeds the
    number of matching documents for a restrictive Boolean query.

    Returns (doc_ids, scores, snippets, result_count, attempts, elapsed_ms, error_message).
    A successful response with zero results has error_message=None.
    """
    if not CHATNOIR_API_KEY:
        raise ValueError("CHATNOIR_API_KEY is not set.")

    payload = {
        "query": boolean_query,
        "index": [index],
        "search_method": search_method,
    }
    headers = {
        "Authorization": f"Bearer {CHATNOIR_API_KEY}",
        "Content-Type": "application/json",
    }

    candidate_sizes = (
        _adaptive_sizes(top_k, min_top_k) if adaptive_top_k else [top_k]
    )

    last_error: Optional[str] = None
    attempts = 0
    t0 = time.perf_counter()

    for size in candidate_sizes:
        payload["size"] = size
        for attempt in range(max_attempts):
            attempts += 1
            try:
                resp = requests.post(
                    CHATNOIR_API_URL,
                    json=payload,
                    headers=headers,
                    timeout=timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                results = data.get("results", []) if isinstance(data, dict) else []

                doc_ids: List[str] = []
                scores_list: List[float] = []
                snippets: Dict[str, str] = {}
                for result in results:
                    doc_id = result.get("trec_id")
                    if doc_id is None:
                        continue
                    doc_ids.append(str(doc_id))
                    scores_list.append(float(result.get("score", 0.0)))
                    snippets[str(doc_id)] = result.get("snippet", "")

                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                if size != top_k:
                    logger.info(
                        "Adaptive fallback succeeded for query %r at size=%d (requested %d)",
                        boolean_query[:80],
                        size,
                        top_k,
                    )
                return (
                    doc_ids,
                    np.array(scores_list, dtype=np.float32),
                    snippets,
                    len(doc_ids),
                    attempts,
                    elapsed_ms,
                    None,
                )
            except requests.exceptions.RequestException as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if _is_transient(exc) and attempt < max_attempts - 1:
                    wait = 2 ** attempt
                    logger.warning(
                        "Transient retrieval failure (attempt %d/%d, size=%d) for query %r; retrying in %ds: %s",
                        attempt + 1,
                        max_attempts,
                        size,
                        boolean_query[:80],
                        wait,
                        last_error,
                    )
                    time.sleep(wait)
                    continue
                # Move to the next smaller candidate size if adaptive fallback is enabled.
                break
            except json.JSONDecodeError as exc:
                last_error = f"JSONDecodeError: {exc}"
                break

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return (
        [],
        np.array([], dtype=np.float32),
        {},
        0,
        attempts,
        elapsed_ms,
        last_error,
    )


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def ensure_output_dir(output_dir: Path, force: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [p for p in output_dir.iterdir() if p.is_file()]
    if existing and not force:
        names = ", ".join(p.name for p in existing)
        raise FileExistsError(
            f"Output directory already contains files: {names}. "
            "Remove them or pass --force to overwrite."
        )


def save_traces_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "query_id",
        "original_query",
        "boolean_query",
        "status",
        "formulation_attempts",
        "retrieval_attempts",
        "result_count",
        "ndcg@50",
        "recall@100",
        "formulation_time_ms",
        "retrieval_time_ms",
        "total_time_ms",
        "error_message",
        "boolean_concepts",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_existing_traces(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load a trace CSV into an ordered dict keyed by query_id."""
    rows: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = str(row["query_id"]).strip()
            rows[qid] = row
    return rows


def save_docs_json(path: Path, docs_data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(docs_data, f, indent=2, ensure_ascii=False)


def save_config(path: Path, config: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def prompt_hash() -> str:
    combined = BOOLEAN_SYSTEM_PROMPT + BOOLEAN_USER_PROMPT_TEMPLATE
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]


def normalize_segment_ids(
    doc_ids: List[str],
    scores: np.ndarray,
    snippets: Dict[str, str],
) -> Tuple[List[str], np.ndarray, Dict[str, str], int]:
    """Collapse ChatNoir segment IDs to their parent document IDs.

    Segmented IDs look like ``msmarco_v2.1_doc_36_1245610104#1_2568230487``.
    Document IDs have no ``#`` and are returned unchanged. When several
    segments belong to the same parent, the highest-scoring (first) segment
    is kept.
    """
    parent_ids: List[str] = []
    parent_scores: List[float] = []
    parent_snippets: Dict[str, str] = {}
    seen: set[str] = set()

    for doc_id, score in zip(doc_ids, scores):
        parent_id = doc_id.split("#", 1)[0]
        if parent_id in seen:
            continue
        seen.add(parent_id)
        parent_ids.append(parent_id)
        parent_scores.append(float(score))
        parent_snippets[parent_id] = snippets.get(doc_id, "")

    return (
        parent_ids,
        np.array(parent_scores, dtype=np.float32),
        parent_snippets,
        len(parent_ids),
    )


def process_one_query(
    qid: str,
    query_text: str,
    agent: BooleanFormulationAgent,
    qrels: Dict[str, Dict[str, int]],
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Optional[Tuple[str, str]]]:
    """Formulate + retrieve + evaluate one query.

    Returns (trace_row, docs_entry_or_none, failure_or_none).
    """
    boolean_query: Optional[str] = None
    concepts: Optional[Dict[str, Any]] = None
    formulation_attempts = 0
    retrieval_attempts = 0
    formulation_time_ms = 0.0
    retrieval_time_ms = 0.0
    error_message: Optional[str] = None
    status = "success"
    doc_ids: List[str] = []
    scores = np.array([], dtype=np.float32)
    snippets: Dict[str, str] = {}
    result_count = 0
    metrics: Dict[str, float] = {}
    docs_entry: Optional[Dict[str, Any]] = None
    failure: Optional[Tuple[str, str]] = None

    # Step 1: Boolean formulation.
    (
        boolean_query,
        concepts,
        formulation_attempts,
        formulation_time_ms,
        form_error,
    ) = formulate_boolean(query_text, agent, args.max_attempts)

    if form_error is not None:
        status = "failed_formulation"
        error_message = form_error
        failure = (qid, error_message)
    else:
        # Step 2: retrieval with the Boolean query.
        (
            doc_ids,
            scores,
            snippets,
            result_count,
            retrieval_attempts,
            retrieval_time_ms,
            retr_error,
        ) = retrieve_boolean(
            boolean_query=boolean_query,  # type: ignore[arg-type]
            top_k=args.top_k,
            index=args.index,
            search_method=args.search_method,
            timeout=args.timeout,
            max_attempts=args.max_attempts,
            adaptive_top_k=args.adaptive_top_k,
            min_top_k=args.min_top_k,
        )

        if retr_error is not None:
            status = "failed_retrieval"
            error_message = retr_error
            failure = (qid, error_message)
        elif result_count == 0:
            status = "success_zero"
        else:
            # Collapse segment IDs to parent doc IDs before scoring. This is a
            # no-op when the index returns document IDs (no ``#`` suffix).
            doc_ids, scores, snippets, result_count = normalize_segment_ids(
                doc_ids, scores, snippets
            )

            if result_count == 0:
                status = "success_zero"
            else:
                status = "success"
                query_qrels = qrels.get(qid, {})
                if query_qrels:
                    metrics = evaluate(
                        doc_ids,
                        query_qrels,
                        ndcg_k=args.ndcg_k,
                        recall_k=args.recall_k,
                    )
                else:
                    metrics = {
                        f"ndcg@{args.ndcg_k}": 0.0,
                        f"recall@{args.recall_k}": 0.0,
                    }

                docs_entry = {
                    "original_query": query_text,
                    "boolean_query": boolean_query,
                    "doc_ids": doc_ids,
                    "scores": scores.tolist(),
                    "snippets": snippets,
                    "result_count": result_count,
                }

    total_time_ms = formulation_time_ms + retrieval_time_ms

    row = {
        "query_id": qid,
        "original_query": query_text,
        "boolean_query": boolean_query if boolean_query is not None else "",
        "status": status,
        "formulation_attempts": formulation_attempts,
        "retrieval_attempts": retrieval_attempts,
        "result_count": result_count,
        "ndcg@50": metrics.get(f"ndcg@{args.ndcg_k}", ""),
        "recall@100": metrics.get(f"recall@{args.recall_k}", ""),
        "formulation_time_ms": f"{formulation_time_ms:.2f}",
        "retrieval_time_ms": f"{retrieval_time_ms:.2f}",
        "total_time_ms": f"{total_time_ms:.2f}",
        "error_message": error_message if error_message is not None else "",
        "boolean_concepts": json.dumps(concepts) if concepts is not None else "",
    }
    return row, docs_entry, failure


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    source_path = Path(args.query_source)
    topics_path = Path(args.topics_file)
    qrels_path = Path(args.qrels_file)

    topics_map = load_topics_map(topics_path)
    loaded = load_query_source(source_path, args.query_id_column, topics_map)
    loaded_map = {qid: text for qid, text in loaded}
    qrels = load_qrels(qrels_path)

    logger.info("Loaded %d unique query IDs from %s", len(loaded), source_path)

    trace_path = output_dir / "boolean_formulate_traces_50_msmarco_dev.csv"
    docs_path = output_dir / "boolean_formulate_docs_50_msmarco_dev.json"
    config_path = output_dir / "config.json"

    # Prepare output container.
    if args.retry_failed:
        if not trace_path.exists():
            raise FileNotFoundError(
                f"--retry-failed requested but no existing trace file at {trace_path}"
            )
        existing_rows = load_existing_traces(trace_path)
        with docs_path.open("r", encoding="utf-8") as f:
            docs_data: Dict[str, Any] = json.load(f)
        failed_ids = [
            qid
            for qid, row in existing_rows.items()
            if row.get("status") in ("failed_formulation", "failed_retrieval")
        ]
        if not failed_ids:
            logger.info("No failed queries to retry.")
            return
        selected = [(qid, loaded_map[qid]) for qid in failed_ids if qid in loaded_map]
        if len(selected) < len(failed_ids):
            missing = set(failed_ids) - {qid for qid, _ in selected}
            logger.warning(
                "Could not resolve %d failed query IDs in the source: %s",
                len(missing),
                ", ".join(sorted(missing)),
            )
        requested_count = len(selected)
        rows_by_id = existing_rows
        logger.info("Retrying %d failed queries", len(selected))
    else:
        ensure_output_dir(output_dir, args.force)
        selected, requested_count = select_queries(loaded, args.num_queries)
        rows_by_id: Dict[str, Dict[str, Any]] = {}
        docs_data = {}

    logger.info("Selected %d queries (requested %d)", len(selected), requested_count)
    logger.info("Loading encoder: %s", args.encoder)
    encoder = SentenceTransformer(args.encoder)

    logger.info("Initializing BooleanFormulationAgent (model=%s)", LLM_MODEL_NAME or "default")
    agent = BooleanFormulationAgent(embed_model=encoder)

    if not args.retry_failed:
        config = {
            "model": LLM_MODEL_NAME,
            "endpoint": LLM_BASE_URL,
            "prompt_module": "src/agents/boolean_formulate.py",
            "prompt_hash": prompt_hash(),
            "chatnoir_endpoint": CHATNOIR_API_URL,
            "index": args.index,
            "top_k": args.top_k,
            "adaptive_top_k": args.adaptive_top_k,
            "min_top_k": args.min_top_k,
            "search_method": args.search_method,
            "timeout": args.timeout,
            "max_attempts": args.max_attempts,
            "ndcg_k": args.ndcg_k,
            "recall_k": args.recall_k,
            "query_source": str(source_path),
            "topics_file": str(topics_path),
            "qrels_file": str(qrels_path),
            "num_queries_requested": requested_count,
            "num_queries_selected": len(selected),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        save_config(config_path, config)

    failed_queries: List[Tuple[str, str]] = []

    for idx, (qid, query_text) in enumerate(selected, start=1):
        logger.info(
            "[%d/%d] query_id=%s | %s",
            idx,
            len(selected),
            qid,
            query_text[:80],
        )

        row, docs_entry, failure = process_one_query(
            qid=qid,
            query_text=query_text,
            agent=agent,
            qrels=qrels,
            args=args,
        )

        rows_by_id[qid] = row
        if docs_entry is not None:
            docs_data[qid] = docs_entry
        elif qid in docs_data:
            # If a previous run produced docs but this retry failed, keep stale
            # docs out of the final file.
            del docs_data[qid]
        if failure is not None:
            failed_queries.append(failure)

        # Save incremental progress (preserve original order for non-retry rows).
        save_traces_csv(trace_path, list(rows_by_id.values()))
        if docs_data:
            save_docs_json(docs_path, docs_data)

    if args.retry_failed and config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
        config["retry_timestamp"] = datetime.now(timezone.utc).isoformat()
        config["retry_failed_count"] = len(selected)
        config["retry_max_attempts"] = args.max_attempts
        save_config(config_path, config)

    # Final summary from the full set of rows.
    trace_rows = list(rows_by_id.values())
    counts = {
        "success": sum(1 for r in trace_rows if r["status"] == "success"),
        "success_zero": sum(1 for r in trace_rows if r["status"] == "success_zero"),
        "failed_formulation": sum(
            1 for r in trace_rows if r["status"] == "failed_formulation"
        ),
        "failed_retrieval": sum(
            1 for r in trace_rows if r["status"] == "failed_retrieval"
        ),
    }
    requested = len(trace_rows)
    successful = counts["success"] + counts["success_zero"]
    failed = counts["failed_formulation"] + counts["failed_retrieval"]

    logger.info("=" * 70)
    if args.retry_failed:
        logger.info("Boolean formulation retry complete")
    else:
        logger.info("Boolean formulation experiment complete")
    logger.info("Output directory: %s", output_dir)
    logger.info("Total rows in trace: %d", requested)
    logger.info("Successful (with results): %d", counts["success"])
    logger.info("Successful (zero results): %d", counts["success_zero"])
    logger.info("Failed formulation: %d", counts["failed_formulation"])
    logger.info("Failed retrieval: %d", counts["failed_retrieval"])
    logger.info("Total failed: %d", failed)
    if counts["success"] > 0:
        ndcgs = [
            float(row["ndcg@50"])
            for row in trace_rows
            if row["status"] == "success" and row["ndcg@50"] != ""
        ]
        recalls = [
            float(row["recall@100"])
            for row in trace_rows
            if row["status"] == "success" and row["recall@100"] != ""
        ]
        logger.info(
            "Average NDCG@50 over successful-with-results queries: %.4f",
            sum(ndcgs) / len(ndcgs) if ndcgs else 0.0,
        )
        logger.info(
            "Average Recall@100 over successful-with-results queries: %.4f",
            sum(recalls) / len(recalls) if recalls else 0.0,
        )
    logger.info(
        "Aggregate averages above exclude failed and zero-result queries; "
        "their metrics are recorded as empty in the trace CSV."
    )
    if failed_queries:
        logger.info("Failed query IDs in this run:")
        for qid, reason in failed_queries:
            logger.info("  %s: %s", qid, reason)
    logger.info("=" * 70)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a standalone Boolean-formulation retrieval experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--query-source", type=Path, default="results/stop_traces_50_msmarco_dev.csv")
    parser.add_argument("--topics-file", type=Path, default="data/topics.ms-marco-dev.tsv")
    parser.add_argument("--qrels-file", type=Path, default="data/qrels.ms-marco-dev.tsv")
    parser.add_argument("--num-queries", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument("--query-id-column", type=str, default="query_id")
    parser.add_argument("--index", type=str, default=DEFAULT_INDEX)
    parser.add_argument("--search-method", type=str, default=DEFAULT_SEARCH_METHOD)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--ndcg-k", type=int, default=DEFAULT_NDCG_K)
    parser.add_argument("--recall-k", type=int, default=DEFAULT_RECALL_K)
    parser.add_argument("--encoder", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files without prompting.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help=(
            "Load the existing trace CSV, re-run only queries that previously "
            "failed formulation or retrieval, and merge the results back in."
        ),
    )
    parser.add_argument(
        "--adaptive-top-k",
        action="store_true",
        help=(
            "On ChatNoir 500 errors, progressively halve the requested size "
            "down to --min-top-k before giving up."
        ),
    )
    parser.add_argument(
        "--min-top-k",
        type=int,
        default=1,
        help="Minimum retrieval depth when --adaptive-top-k is used.",
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
