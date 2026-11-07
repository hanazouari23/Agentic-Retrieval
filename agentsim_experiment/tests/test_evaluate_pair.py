"""End-to-end smoke test with synthetic data and mock agents."""

import json
import sys
import tempfile
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from evaluation.evaluate_runs import evaluate_pair


def test_evaluate_pair():
    tmp = Path(tempfile.mkdtemp(prefix="eval_test_"))
    run_dir = tmp / "runs"
    results_dir = tmp / "results"
    run_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Synthetic behavior run: relevant doc at rank 2.
    with open(run_dir / "behavior.run", "w", encoding="utf-8") as f:
        f.write("q1 Q0 d1 1 1.0 behavior\n")
        f.write("q1 Q0 d2 2 0.5 behavior\n")

    # Synthetic CQL run: relevant doc at rank 1.
    with open(run_dir / "cql.run", "w", encoding="utf-8") as f:
        f.write("q1 Q0 d2 1 1.0 cql\n")
        f.write("q1 Q0 d1 2 0.5 cql\n")

    # Synthetic qrels.
    with open(results_dir / "qrels_test.tsv", "w", encoding="utf-8") as f:
        f.write("query_id\tdoc_id\trelevance\n")
        f.write("q1\td2\t1\n")

    # Synthetic diagnostics.
    for name in ("behavior", "cql"):
        with open(run_dir / f"{name}_diagnostics.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "qid": "q1",
                "action_id": 4,
                "pipeline_steps": ["chatnoir_retriever", "filter", "answer_drafter", "finalizer"],
                "agent_calls": {
                    "chatnoir_retriever": 1,
                    "filter": 1,
                    "answer_drafter": 1,
                    "finalizer": 1,
                },
                "num_ranked": 2,
                "latencies": {"latency_chatnoir_retriever": 0.1},
                "failed": False,
                "failure_reason": None,
            }) + "\n")

    summary = evaluate_pair(
        behavior_run=run_dir / "behavior.run",
        cql_run=run_dir / "cql.run",
        qrels_path=results_dir / "qrels_test.tsv",
        behavior_diagnostics=run_dir / "behavior_diagnostics.jsonl",
        cql_diagnostics=run_dir / "cql_diagnostics.jsonl",
        output_summary=results_dir / "summary_test.csv",
        k=10,
    )

    assert summary["behavior"]["mrr_at_10"] == 0.5
    assert summary["cql"]["mrr_at_10"] == 1.0
    assert summary["cql"]["mean_retrieval_calls"] == 1.0
    print("evaluate_pair OK")


if __name__ == "__main__":
    test_evaluate_pair()
