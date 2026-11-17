r"""Extract STOP baseline traces from a CQL rollout CSV.

Keeps only the initial-retrieval step (step == 1) for each query, relabels the
action as STOP, and sets the "after" metrics equal to the "before" metrics
because STOP does not change the retrieval state.

Usage:
    poetry run python helper_scripts/extract_stop_baseline_traces.py \
        --input-csv results/cql_base_uncorrected_traces_150_msmarco_dev.csv \
        --output-csv RQs/relevant_csvs/stop_traces_150_msmarco_dev.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd

DEFAULT_INPUT_CSV: Path = Path("results/cql_base_uncorrected_traces_150_msmarco_dev.csv")
DEFAULT_OUTPUT_CSV: Path = Path("RQs/relevant_csvs/stop_traces_150_msmarco_dev.csv")

OUTPUT_COLUMNS: List[str] = [
    "query_id",
    "query",
    "new_query",
    "action_name",
    "query_length",
    "score_spread",
    "score_entropy",
    "step",
    "ndcg_before",
    "ndcg_after",
    "recall_before",
    "recall_after",
    "cumulative_cost",
    "cumulative_latency_ms",
]


def extract_stop_traces(input_csv: Path, output_csv: Path) -> None:
    df = pd.read_csv(input_csv, dtype={"query_id": str})

    # Keep only the initial retrieval step for each query.
    df = df[df["step"] == 1].copy()

    # Relabel as STOP and keep the original query as the "new" query.
    df["action_name"] = "STOP"
    df["new_query"] = df["query"]

    # STOP does not change the retrieval state.
    df["ndcg_after"] = df["ndcg_before"]
    df["recall_after"] = df["recall_before"]

    # STOP incurs no cost or latency.
    df["cumulative_cost"] = 0.0
    df["cumulative_latency_ms"] = 0.0

    # Reorder and select only the requested columns.
    df = df[OUTPUT_COLUMNS]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"Extracted {len(df)} STOP baseline rows from {input_csv}")
    print(f"Wrote {output_csv}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract STOP baseline traces from a CQL rollout CSV."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Path to the CQL rollout trace CSV.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Path to the output STOP baseline trace CSV.",
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    extract_stop_traces(args.input_csv, args.output_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
