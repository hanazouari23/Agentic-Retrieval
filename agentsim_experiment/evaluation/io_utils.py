"""Shared I/O helpers for evaluation."""

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


def load_eval_config(config_path: str | Path) -> dict:
    """Load eval_config.yaml."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_queries(queries_path: str | Path) -> List[Dict[str, str]]:
    """Load queries as a list of {'qid': ..., 'text': ...} dicts."""
    queries = []
    with open(queries_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            qid = row.get("query_id") or row.get("qid")
            text = row.get("query") or row.get("query_text")
            if qid and text:
                queries.append({"qid": qid.strip(), "text": text.strip()})
    return queries


def load_qrels(qrels_path: str | Path) -> Dict[str, Dict[str, int]]:
    """Load qrels as {qid: {doc_id: relevance_grade}}."""
    qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
    with open(qrels_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            qid = row.get("query_id") or row.get("qid")
            doc_id = row.get("doc_id")
            rel = row.get("relevance")
            if qid and doc_id and rel is not None:
                try:
                    qrels[qid.strip()][doc_id.strip()] = int(rel)
                except ValueError:
                    continue
    return dict(qrels)


def load_behavior_actions(
    episode_manifest_path: str | Path,
    fallback_action: int = 4,
) -> Dict[str, int]:
    """Map query_id -> historical action_id from the episode manifest."""
    behavior: Dict[str, int] = {}
    if not Path(episode_manifest_path).exists():
        return behavior

    with open(episode_manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            episode = json.loads(line)
            qid = episode.get("query")
            # The manifest stores the raw query text as identifier in these files.
            # We also keep a lookup by the provenance group key if present.
            transitions = episode.get("transitions", [])
            if not transitions:
                continue
            action_id = transitions[0].get("action_id", fallback_action)
            behavior[qid] = int(action_id)
    return behavior


def load_training_queries(
    episode_manifest_path: str | Path,
    split: str = "train",
) -> set:
    """Return a set of query texts that appear in the training split."""
    training: set = set()
    if not Path(episode_manifest_path).exists():
        return training

    with open(episode_manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            episode = json.loads(line)
            if episode.get("split") == split:
                training.add(episode.get("query", ""))
    return training


def select_held_out_queries(
    queries: List[Dict[str, str]],
    qrels: Dict[str, Dict[str, int]],
    behavior_actions: Dict[str, int],
    num_queries: Optional[int] = None,
    seed: int = 42,
    exclude_training: Optional[set] = None,
) -> List[Dict[str, str]]:
    """Select judged queries that also have a recorded behavior action."""
    judged = [q for q in queries if q["qid"] in qrels and qrels[q["qid"]]]
    if exclude_training:
        judged = [q for q in judged if q["text"] not in exclude_training]

    # Prefer queries with a behavior action for a fair comparison.
    with_behavior = [q for q in judged if q["text"] in behavior_actions]
    candidates = with_behavior if with_behavior else judged

    rng = random.Random(seed)
    rng.shuffle(candidates)

    if num_queries is not None:
        candidates = candidates[:num_queries]
    return candidates


def write_run_file(
    run_path: str | Path,
    query_results: List[Tuple[str, List[Tuple[str, float]]]],
    run_id: str,
    max_rank: int = 1000,
) -> None:
    """Write MS MARCO run file with six whitespace-separated fields."""
    Path(run_path).parent.mkdir(parents=True, exist_ok=True)
    with open(run_path, "w", encoding="utf-8") as f:
        for qid, ranked in query_results:
            for rank, (passage_id, score) in enumerate(ranked[:max_rank], start=1):
                f.write(
                    f"{qid} Q0 {passage_id} {rank} {score:.6f} {run_id}\n"
                )


def write_diagnostics(
    diagnostics_path: str | Path,
    records: List[dict],
) -> None:
    """Append diagnostic records as JSON lines."""
    Path(diagnostics_path).parent.mkdir(parents=True, exist_ok=True)
    with open(diagnostics_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
