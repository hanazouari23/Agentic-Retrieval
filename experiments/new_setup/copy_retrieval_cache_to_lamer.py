"""Copy cached original-query retrievals from the Boolean oracle cache to the LameR cache.

This seeds the LameR experiment with retrieval results that are already known
from a previous ``generate_expert_boolean_traces.py`` run.  Only retrieval
entries for the original query text are copied; augmented-query retrievals
(for example LameR's ``q + answer`` query) will be fetched and cached during
the LameR run itself.

Usage:
    poetry run python copy_retrieval_cache_to_lamer.py --num-queries 500
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Match the constants used by the Boolean and LameR trace generators
# ---------------------------------------------------------------------------
QUERIES_PATH = "data/topics.ms-marco-dev2.tsv"
QRELS_PATH = "data/qrels.ms-marco-dev2.tsv"

SOURCE_CACHE_DIR = (
    "cache/expert_boolean_oracle/"
    "v1-tr20-tp10-ndcg20-recall20-boolean-idxmsmarco-v2.1-smbm25-min1"
)
DEST_CACHE_DIR = (
    "cache/expert_lamer_oracle/"
    "v1-tr20-tp10-ndcg20-recall20-lamer-idxmsmarco-v2.1-smbm25-min1"
)

INDEX = "msmarco-v2.1"  # same as Retriever().index used by the generators
SEARCH_METHOD = "bm25"
TOP_KS_TO_COPY = [10, 20]  # 20: rollout + PRF + LameR final; 10: LameR initial demos


def load_selected_queries(
    queries_path: Path, qrels_path: Path, num_queries: int, offset: int
) -> list[tuple[str, str]]:
    queries_df = pd.read_csv(queries_path, sep="\t", header=0, dtype={"query_id": str})
    query_col = "query" if "query" in queries_df.columns else "query_text"
    queries_df[query_col] = queries_df[query_col].astype(str)

    qrels_df = pd.read_csv(qrels_path, sep="\t", header=0, dtype={"query_id": str})
    qids_with_qrels = set(qrels_df["query_id"].unique())

    selected = []
    for _, row in queries_df.iterrows():
        qid = str(row["query_id"])
        if qid not in qids_with_qrels:
            continue
        selected.append((qid, row[query_col]))

    return selected[offset : offset + num_queries]


def cache_path(key: str, cache_dir: Path) -> Path:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.pkl"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed the LameR cache with useful Boolean retrieval results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=500,
        help="Number of queries to copy cache entries for.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Offset into the topic file.",
    )
    args = parser.parse_args()

    src_dir = Path(SOURCE_CACHE_DIR)
    dst_dir = Path(DEST_CACHE_DIR)
    dst_dir.mkdir(parents=True, exist_ok=True)

    if not src_dir.exists():
        raise FileNotFoundError(f"Source cache directory does not exist: {src_dir}")

    selected = load_selected_queries(
        Path(QUERIES_PATH), Path(QRELS_PATH), args.num_queries, args.offset
    )

    copied = 0
    missing = 0

    for _qid, query in selected:
        for top_k in TOP_KS_TO_COPY:
            key = f"retrieve|q:{query}|k:{top_k}|idx:{INDEX}|sm:{SEARCH_METHOD}"
            src = cache_path(key, src_dir)
            dst = cache_path(key, dst_dir)
            if src.exists():
                shutil.copy2(src, dst)
                copied += 1
            else:
                missing += 1

    print(f"Copied {copied} cached retrieval files to {dst_dir}")
    print(f"Missing keys (will be fetched live if needed): {missing}")


if __name__ == "__main__":
    main()
