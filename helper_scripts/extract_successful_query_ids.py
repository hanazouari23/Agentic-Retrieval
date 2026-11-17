r"""Extract the unique query IDs that already succeeded in a trace CSV.

Usage:
    poetry run python helper_scripts/extract_successful_query_ids.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd

DEFAULT_TRACE_PATH: Path = Path(
    "results/cql_base_uncorrected_traces_200_msmarco_dev.csv"
)
DEFAULT_OUTPUT_PATH: Path = Path("RQs/relevant_csvs/successful_query_ids_71.csv")


def extract_successful_query_ids(trace_path: Path) -> List[str]:
    df = pd.read_csv(trace_path, dtype=str)
    if "query_id" not in df.columns:
        raise ValueError(f"No query_id column in {trace_path}")
    ids = [str(qid) for qid in df["query_id"].unique()]
    return ids


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract unique query IDs from a trace CSV."
    )
    parser.add_argument(
        "--trace-path",
        type=Path,
        default=DEFAULT_TRACE_PATH,
        help="Path to the rollout trace CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output CSV path for the extracted query IDs.",
    )
    args = parser.parse_args(argv)

    if not args.trace_path.exists():
        raise FileNotFoundError(f"Trace file not found: {args.trace_path}")

    query_ids = extract_successful_query_ids(args.trace_path)
    print(f"Extracted {len(query_ids)} unique query IDs from {args.trace_path}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"query_id": query_ids}).to_csv(args.output, index=False)
    print(f"Wrote query IDs to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
