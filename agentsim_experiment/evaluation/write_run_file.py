"""Convert a diagnostics JSONL file into a MS MARCO run file."""

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from evaluation.io_utils import write_run_file


def diagnostics_to_run(diagnostics_path: str | Path, run_path: str | Path, run_id: str) -> None:
    """Read diagnostics and write a run file."""
    diagnostics_path = Path(diagnostics_path)
    query_results = []
    with open(diagnostics_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            qid = record["qid"]
            ranked = [
                (passage_id, float(score))
                for passage_id, score in record.get("ranked_passages", [])
            ]
            query_results.append((qid, ranked))

    write_run_file(run_path, query_results, run_id)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert diagnostics JSONL to a MS MARCO run file."
    )
    parser.add_argument("diagnostics", help="Path to diagnostics JSONL")
    parser.add_argument("run", help="Output run file path")
    parser.add_argument("--run-id", default="cql_policy", help="Run identifier")
    args = parser.parse_args()

    diagnostics_to_run(args.diagnostics, args.run, args.run_id)


if __name__ == "__main__":
    main()
