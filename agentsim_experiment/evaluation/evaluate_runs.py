"""Compute ranking metrics and cost summaries for MS MARCO run files."""

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from evaluation.io_utils import load_qrels


def _load_run(run_path: str | Path) -> Dict[str, List[Tuple[str, float]]]:
    """Load a run file as {qid: [(passage_id, score), ...]}."""
    ranking: Dict[str, List[Tuple[str, float]]] = {}
    with open(run_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            qid, _, doc_id, rank, score, _ = parts[:6]
            ranking.setdefault(qid, []).append((doc_id, float(score)))
    return ranking


def _dcg(gains: List[float], k: int) -> float:
    r = np.asarray(gains[:k], dtype=float)
    if r.size == 0:
        return 0.0
    positions = np.arange(2, r.size + 2)
    return float(np.sum(r / np.log2(positions)))


def _ndcg_at_k(
    ranked_doc_ids: List[str],
    qrels: Dict[str, int],
    k: int,
) -> float:
    gains = [qrels.get(doc_id, 0) for doc_id in ranked_doc_ids[:k]]
    ideal = sorted((rel for rel in qrels.values() if rel > 0), reverse=True)[:k]
    dcg = _dcg(gains, k)
    idcg = _dcg(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


def _mrr_at_k(
    ranked_doc_ids: List[str],
    qrels: Dict[str, int],
    k: int,
) -> float:
    for rank, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if qrels.get(doc_id, 0) > 0:
            return 1.0 / rank
    return 0.0


def _recip_rank_at_k(
    ranked_doc_ids: List[str],
    qrels: Dict[str, int],
    k: int,
) -> float:
    """Alias for MRR@k for a single query."""
    return _mrr_at_k(ranked_doc_ids, qrels, k)


def compute_run_metrics(
    run_path: str | Path,
    qrels: Dict[str, Dict[str, int]],
    k: int = 10,
) -> Dict[str, List[float]]:
    """Return per-query MRR@k and nDCG@k values."""
    ranking = _load_run(run_path)
    metrics = {"mrr": [], "ndcg": [], "qid": []}
    for qid, ranked in ranking.items():
        doc_ids = [doc_id for doc_id, _ in ranked]
        qrel = qrels.get(qid, {})
        if not qrel:
            continue
        metrics["qid"].append(qid)
        metrics["mrr"].append(_mrr_at_k(doc_ids, qrel, k))
        metrics["ndcg"].append(_ndcg_at_k(doc_ids, qrel, k))
    return metrics


def compute_cost_summary(diagnostics_path: str | Path) -> Dict[str, float]:
    """Aggregate cost/latency diagnostics into means and failure rate."""
    Path(diagnostics_path).parent.mkdir(parents=True, exist_ok=True)
    if not Path(diagnostics_path).exists():
        return {
            "retrieval_calls": 0.0,
            "filter_calls": 0.0,
            "drafter_calls": 0.0,
            "finalizer_calls": 0.0,
            "latency_seconds": 0.0,
            "fallback_rate": 0.0,
        }

    counts = {
        "chatnoir_retriever": [],
        "filter": [],
        "answer_drafter": [],
        "finalizer": [],
        "latency": [],
        "failed": [],
    }

    with open(diagnostics_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            calls = record.get("agent_calls", {})
            for tool in counts:
                if tool == "latency":
                    latencies = record.get("latencies", {})
                    total = sum(latencies.values()) if isinstance(latencies, dict) else 0.0
                    counts["latency"].append(total)
                elif tool == "failed":
                    counts["failed"].append(1 if record.get("failed") else 0)
                else:
                    counts[tool].append(calls.get(tool, 0))

    n = len(counts["failed"]) or 1
    return {
        "retrieval_calls": float(np.mean(counts["chatnoir_retriever"])) if counts["chatnoir_retriever"] else 0.0,
        "filter_calls": float(np.mean(counts["filter"])) if counts["filter"] else 0.0,
        "drafter_calls": float(np.mean(counts["answer_drafter"])) if counts["answer_drafter"] else 0.0,
        "finalizer_calls": float(np.mean(counts["finalizer"])) if counts["finalizer"] else 0.0,
        "latency_seconds": float(np.mean(counts["latency"])) if counts["latency"] else 0.0,
        "fallback_rate": float(sum(counts["failed"]) / n),
    }


def trec_eval_metric(
    qrels_path: str | Path,
    run_path: str | Path,
    metric: str = "recip_rank",
    k: int = 10,
) -> Optional[Dict[str, float]]:
    """Run trec_eval and parse the averaged metric value if available."""
    try:
        result = subprocess.run(
            [
                "trec_eval",
                "-c",
                "-M",
                str(k),
                "-m",
                metric,
                str(qrels_path),
                str(run_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        for line in result.stdout.strip().splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[0] == metric:
                return {"metric": metric, "value": float(parts[2])}
    except Exception:
        pass
    return None


def evaluate_pair(
    behavior_run: str | Path,
    cql_run: str | Path,
    qrels_path: str | Path,
    behavior_diagnostics: str | Path,
    cql_diagnostics: str | Path,
    output_summary: str | Path,
    k: int = 10,
) -> Dict[str, Dict[str, float]]:
    """Compute and save summary metrics for behavior and CQL runs."""
    qrels = load_qrels(qrels_path)

    behavior_metrics = compute_run_metrics(behavior_run, qrels, k=k)
    cql_metrics = compute_run_metrics(cql_run, qrels, k=k)
    behavior_cost = compute_cost_summary(behavior_diagnostics)
    cql_cost = compute_cost_summary(cql_diagnostics)

    def _mean(values: List[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    summary = {
        "behavior": {
            "mrr_at_10": _mean(behavior_metrics["mrr"]),
            "ndcg_at_10": _mean(behavior_metrics["ndcg"]),
            "mean_retrieval_calls": behavior_cost["retrieval_calls"],
            "mean_filter_calls": behavior_cost["filter_calls"],
            "mean_drafter_calls": behavior_cost["drafter_calls"],
            "mean_finalizer_calls": behavior_cost["finalizer_calls"],
            "mean_latency_seconds": behavior_cost["latency_seconds"],
            "fallback_rate": behavior_cost["fallback_rate"],
        },
        "cql": {
            "mrr_at_10": _mean(cql_metrics["mrr"]),
            "ndcg_at_10": _mean(cql_metrics["ndcg"]),
            "mean_retrieval_calls": cql_cost["retrieval_calls"],
            "mean_filter_calls": cql_cost["filter_calls"],
            "mean_drafter_calls": cql_cost["drafter_calls"],
            "mean_finalizer_calls": cql_cost["finalizer_calls"],
            "mean_latency_seconds": cql_cost["latency_seconds"],
            "fallback_rate": cql_cost["fallback_rate"],
        },
    }

    Path(output_summary).parent.mkdir(parents=True, exist_ok=True)
    with open(output_summary, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "policy",
            "mrr_at_10",
            "ndcg_at_10",
            "mean_retrieval_calls",
            "mean_filter_calls",
            "mean_drafter_calls",
            "mean_finalizer_calls",
            "mean_latency_seconds",
            "fallback_rate",
        ])
        for policy, row in summary.items():
            writer.writerow([
                policy,
                f"{row['mrr_at_10']:.6f}",
                f"{row['ndcg_at_10']:.6f}",
                f"{row['mean_retrieval_calls']:.3f}",
                f"{row['mean_filter_calls']:.3f}",
                f"{row['mean_drafter_calls']:.3f}",
                f"{row['mean_finalizer_calls']:.3f}",
                f"{row['mean_latency_seconds']:.6f}",
                f"{row['fallback_rate']:.6f}",
            ])

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate behavior and CQL run files."
    )
    parser.add_argument("--behavior-run", default="runs/behavior.run")
    parser.add_argument("--cql-run", default="runs/cql.run")
    parser.add_argument("--qrels", default="data/qrels.tsv")
    parser.add_argument("--behavior-diagnostics", default="runs/behavior_diagnostics.jsonl")
    parser.add_argument("--cql-diagnostics", default="runs/cql_diagnostics.jsonl")
    parser.add_argument("--output", default="results/summary.csv")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    summary = evaluate_pair(
        args.behavior_run,
        args.cql_run,
        args.qrels,
        args.behavior_diagnostics,
        args.cql_diagnostics,
        args.output,
        k=args.k,
    )

    for policy, metrics in summary.items():
        print(f"{policy}: MRR@{args.k}={metrics['mrr_at_10']:.4f}, "
              f"nDCG@{args.k}={metrics['ndcg_at_10']:.4f}, "
              f"latency={metrics['mean_latency_seconds']:.3f}s, "
              f"fallback={metrics['fallback_rate']:.3f}")


if __name__ == "__main__":
    main()
