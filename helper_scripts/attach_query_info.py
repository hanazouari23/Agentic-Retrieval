"""Re-attach query_id and query from the terminal_reward CSV to the base CSV.

Both files have the same row order (verified by matching action sequences and
step patterns per query), so the missing columns can be copied row-by-row.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def attach_query_info(base_csv: Path, source_csv: Path, output_csv: Path) -> int:
    with open(base_csv, newline="", encoding="utf-8") as f:
        base_rows = list(csv.DictReader(f))
    with open(source_csv, newline="", encoding="utf-8") as f:
        source_rows = list(csv.DictReader(f))

    if len(base_rows) != len(source_rows):
        raise ValueError(
            f"Row count mismatch: {len(base_rows)} vs {len(source_rows)}"
        )

    for b, s in zip(base_rows, source_rows):
        b["query_id"] = s.get("query_id", "")
        b["query"] = s.get("query", "")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=base_rows[0].keys())
        writer.writeheader()
        writer.writerows(base_rows)

    return len(base_rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Attach query_id/query from a source CSV to a base CSV by row index."
    )
    parser.add_argument("--base", type=Path, default=Path("data/training/trajectories_tree_branches_5000.csv"))
    parser.add_argument("--source", type=Path, default=Path("data/training/trajectories_tree_branches_5000_terminal_reward.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/training/trajectories_tree_branches_5000_with_query.csv"))
    args = parser.parse_args(argv)

    rows = attach_query_info(args.base, args.source, args.output)
    print(f"Wrote {rows:,} rows to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
