"""Cache original-query ChatNoir retrieval responses with resumable, per-query files.

Usage:
    python experiments/new_setup/cache_original_retrieval.py \
        --num-queries 50 \
        --top-k 50 \
        --query-source results/stop_traces_50_msmarco_dev.csv \
        --topics-file data/topics.ms-marco-dev.tsv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import shlex
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


DEFAULT_API_URL = os.getenv(
    "CHATNOIR_API_URL",
    "https://www.chatnoir.eu/api/v1/_search",
).strip()
DEFAULT_API_KEY = os.getenv("CHATNOIR_API_KEY", "").strip()
DEFAULT_INDEX = "msmarco-v2.1"
DEFAULT_SEARCH_METHOD = "bm25"
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_CACHE_ROOT = Path("results/new_setup/original_retrieval_cache")


# ---------------------------------------------------------------------------
# Loading and selecting queries
# ---------------------------------------------------------------------------


def _detect_header(first_line: str, delimiter: str) -> Optional[List[str]]:
    """Return parsed header tokens if the first line looks like a header."""
    tokens = [t.strip() for t in first_line.split(delimiter)]
    known_id = {"query_id", "qid", "id"}
    known_text = {"query", "query_text", "text"}
    has_id = any(tok.lower() in known_id for tok in tokens)
    has_text = any(tok.lower() in known_text for tok in tokens)
    if has_id or has_text:
        return tokens
    return None


def _resolve_text_col(header: List[str]) -> Optional[str]:
    for tok in header:
        if tok.lower() == "query_text":
            return tok
        if tok.lower() == "query":
            return tok
    return None


def _resolve_id_col(header: List[str], preferred: str) -> str:
    lowered = [h.lower() for h in header]
    if preferred.lower() in lowered:
        return header[lowered.index(preferred.lower())]
    for candidate in ("query_id", "qid", "id"):
        if candidate in lowered:
            return header[lowered.index(candidate)]
    return header[0]


def _file_delimiter(path: Path) -> str:
    suffix = path.suffix.lower()
    return "," if suffix == ".csv" else "\t"


def load_query_source(
    path: Path,
    query_id_column: str = "query_id",
    topics_map: Optional[Dict[str, str]] = None,
) -> List[Tuple[str, str]]:
    """Load query IDs (and text when present) from a CSV or TSV source.

    Returns a list of (query_id, query_text) preserving first occurrence order
    and deduplicated by query_id. IDs and text are strings.
    """
    if not path.exists():
        raise FileNotFoundError(f"Query source not found: {path}")

    delimiter = _file_delimiter(path)

    with path.open("r", encoding="utf-8", newline="") as f:
        # Skip empty lines to inspect the first real row.
        first_line = ""
        for line in f:
            stripped = line.strip()
            if stripped:
                first_line = stripped
                break
        if not first_line:
            raise ValueError(f"Query source is empty: {path}")

    header = _detect_header(first_line, delimiter)

    rows: List[Tuple[str, str]] = []
    seen: set = set()

    with path.open("r", encoding="utf-8", newline="") as f:
        if header is not None:
            reader = csv.DictReader(f, delimiter=delimiter)
            actual_header = reader.fieldnames or []
            id_col = _resolve_id_col(actual_header, query_id_column)
            text_col = _resolve_text_col(actual_header)
            for row in reader:
                qid = str(row.get(id_col, "")).strip()
                text = str(row.get(text_col, "")).strip() if text_col else ""
                if qid and qid not in seen:
                    seen.add(qid)
                    rows.append((qid, text))
        else:
            reader = csv.reader(f, delimiter=delimiter)
            for parts in reader:
                if not parts or not parts[0].strip():
                    continue
                qid = str(parts[0]).strip()
                text = str(parts[1]).strip() if len(parts) > 1 else ""
                if qid not in seen:
                    seen.add(qid)
                    rows.append((qid, text))

    if topics_map is not None:
        resolved: List[Tuple[str, str]] = []
        unresolved = 0
        for qid, text in rows:
            effective_text = text or topics_map.get(qid, "")
            if effective_text:
                resolved.append((qid, effective_text))
            else:
                unresolved += 1
        if unresolved:
            logger.warning(
                "Could not resolve query text for %d/%d IDs from topics file",
                unresolved,
                len(rows),
            )
        return resolved

    # No topics map: keep rows that already carry text.
    textless = [qid for qid, text in rows if not text]
    if textless:
        raise ValueError(
            f"{len(textless)} query IDs have no text and no --topics-file was provided. "
            "Provide --topics-file or a source that includes query text."
        )
    return rows


def load_topics_map(path: Path) -> Dict[str, str]:
    """Load a topics TSV into a query_id -> query_text map."""
    if not path.exists():
        raise FileNotFoundError(f"Topics file not found: {path}")

    delimiter = "\t"
    with path.open("r", encoding="utf-8", newline="") as f:
        first_line = ""
        for line in f:
            stripped = line.strip()
            if stripped:
                first_line = stripped
                break
    if not first_line:
        raise ValueError(f"Topics file is empty: {path}")

    header = _detect_header(first_line, delimiter)
    mapping: Dict[str, str] = {}

    with path.open("r", encoding="utf-8", newline="") as f:
        if header is not None:
            reader = csv.DictReader(f, delimiter=delimiter)
            actual_header = reader.fieldnames or []
            id_col = _resolve_id_col(actual_header, "query_id")
            text_col = _resolve_text_col(actual_header)
            if text_col is None:
                raise ValueError(f"Topics file has no recognizable text column: {path}")
            for row in reader:
                qid = str(row.get(id_col, "")).strip()
                text = str(row.get(text_col, "")).strip()
                if qid and qid not in mapping:
                    mapping[qid] = text
        else:
            reader = csv.reader(f, delimiter=delimiter)
            for parts in reader:
                if len(parts) < 2:
                    continue
                qid = str(parts[0]).strip()
                text = str(parts[1]).strip()
                if qid and qid not in mapping:
                    mapping[qid] = text

    return mapping


def select_queries(
    queries: List[Tuple[str, str]], num_queries: int
) -> Tuple[List[Tuple[str, str]], int]:
    """Return the first N unique queries and the originally requested count."""
    selected = queries[:num_queries]
    if len(selected) < num_queries:
        logger.warning(
            "Source contains only %d unique queries; %d were requested.",
            len(selected),
            num_queries,
        )
    return selected, num_queries


# ---------------------------------------------------------------------------
# Settings fingerprinting and cache paths
# ---------------------------------------------------------------------------


def retrieval_settings(
    api_url: str,
    index: str,
    top_k: int,
    search_method: str,
    timeout: float,
) -> Dict[str, Any]:
    """Return the retrieval-affecting settings used for fingerprinting."""
    return {
        "api_url": api_url,
        "index": index,
        "top_k": top_k,
        "search_method": search_method,
        "timeout": timeout,
    }


def settings_fingerprint(settings: Dict[str, Any], length: int = 16) -> str:
    canonical = json.dumps(settings, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:length]


def query_text_hash(query_text: str, length: int = 16) -> str:
    return hashlib.sha256(query_text.encode("utf-8")).hexdigest()[:length]


def cache_path_for_query(
    cache_root: Path, query_text: str, fingerprint: str
) -> Path:
    return cache_root / "per_query" / query_text_hash(query_text) / f"{fingerprint}.json"


# ---------------------------------------------------------------------------
# Retrieval and cache I/O
# ---------------------------------------------------------------------------


def _is_transient(exc: requests.RequestException) -> bool:
    """Decide whether a request exception is worth retrying."""
    if exc.response is not None:
        code = exc.response.status_code
        return code >= 500 or code == 429
    return True


def retrieve_raw(
    query_text: str,
    api_url: str,
    api_key: str,
    index: str,
    top_k: int,
    search_method: str,
    timeout: float,
    max_attempts: int,
) -> Tuple[str, Optional[Dict[str, Any]], int, Optional[int], Optional[str], Optional[str]]:
    """Query ChatNoir and return the raw response with status metadata.

    Returns:
        (status, response_or_none, attempts_used, result_count, error_message, retrieved_at)
    """
    if not api_key:
        raise ValueError(
            "A ChatNoir API key is required. Set CHATNOIR_API_KEY in the environment."
        )

    payload = {
        "query": query_text,
        "index": [index],
        "size": top_k,
        "search_method": search_method,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_error: Optional[str] = None
    attempts_used = 0
    retrieved_at: Optional[str] = None

    for attempt in range(max_attempts):
        attempts_used = attempt + 1
        try:
            resp = requests.post(
                api_url,
                json=payload,
                headers=headers,
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            retrieved_at = datetime.now(timezone.utc).isoformat()
            results = data.get("results", []) if isinstance(data, dict) else []
            result_count = len(results)
            return "success", data, attempts_used, result_count, None, retrieved_at
        except (requests.exceptions.RequestException, json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, requests.exceptions.RequestException) and _is_transient(exc):
                if attempt < max_attempts - 1:
                    wait = 2 ** attempt
                    logger.warning(
                        "Transient failure for query %r (attempt %d/%d); retrying in %ds: %s",
                        query_text[:80],
                        attempt + 1,
                        max_attempts,
                        wait,
                        last_error,
                    )
                    time.sleep(wait)
                    continue
            break

    return "failed", None, attempts_used, None, last_error, retrieved_at


def load_cached_response(
    cache_path: Path,
    query_text: str,
    fingerprint: str,
) -> Optional[Dict[str, Any]]:
    """Load a cached raw response if it exists and matches the query/settings."""
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("r", encoding="utf-8") as f:
            wrapper = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read cache file %s: %s", cache_path, exc)
        return None

    if wrapper.get("query_text") != query_text:
        return None
    if wrapper.get("settings_fingerprint") != fingerprint:
        return None
    return wrapper


def save_cached_response(
    cache_path: Path,
    query_text: str,
    settings: Dict[str, Any],
    fingerprint: str,
    response: Dict[str, Any],
    retrieved_at: str,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = {
        "query_text": query_text,
        "settings": settings,
        "settings_fingerprint": fingerprint,
        "retrieved_at": retrieved_at,
        "response": response,
    }
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(wrapper, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# Run output and manifest
# ---------------------------------------------------------------------------


def source_selection_dir(
    source_path: Path,
    topics_path: Optional[Path],
    selected_ids: List[str],
) -> Path:
    """Build a directory name that is unique to source + topics + selection."""
    source_stem = source_path.stem
    topics_stem = topics_path.stem if topics_path else "self_contained"
    id_digest = hashlib.sha256(
        json.dumps(selected_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    return Path(source_stem) / f"{topics_stem}_{id_digest}"


def run_filenames(num_selected: int, top_k: int) -> Tuple[str, str]:
    base = f"{num_selected}queries_top{top_k}"
    return f"manifest_{base}.json", f"README_{base}.md"


def classify_result(status: str, result_count: Optional[int], top_k: int) -> str:
    if status == "failed":
        return "failed"
    if result_count == 0:
        return "success_zero"
    if result_count is not None and result_count < top_k:
        return "success_incomplete"
    return "success"


def build_readme(
    manifest: Dict[str, Any],
    output_dir: Path,
    cache_root: Path,
    command: str,
) -> str:
    settings = manifest["settings"]
    queries = manifest["queries"]

    counts: Dict[str, int] = {
        "success": 0,
        "success_zero": 0,
        "success_incomplete": 0,
        "failed": 0,
    }
    for q in queries:
        counts[q["status"]] = counts.get(q["status"], 0) + 1

    cache_reused = sum(1 for q in queries if q.get("cache_reused"))

    snippets = False
    full_text = False
    for q in queries[:5]:
        cache_path = cache_root / q["cache_location"]
        if not cache_path.exists():
            continue
        try:
            with cache_path.open("r", encoding="utf-8") as f:
                wrapper = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        resp = wrapper.get("response", {}) if isinstance(wrapper, dict) else {}
        if not isinstance(resp, dict):
            continue
        for result in resp.get("results", [])[:1]:
            if isinstance(result, dict):
                if "snippet" in result and result.get("snippet"):
                    snippets = True
                if "text" in result and result.get("text"):
                    full_text = True

    lines = [
        "# Original-query retrieval cache",
        "",
        "This run caches raw ChatNoir responses for a deterministic selection of queries.",
        "",
        "## Source and selection",
        "",
        f"- **Query source:** `{manifest['query_source']}`",
    ]
    if manifest.get("topics_file"):
        lines.append(f"- **Topics file:** `{manifest['topics_file']}`")
    lines.extend(
        [
            f"- **Requested query count:** {manifest['requested_count']}",
            f"- **Selected query count:** {manifest['selected_count']}",
            "- **Selection rule:** preserve first occurrence order, deduplicate by query ID, "
            "then take the first N unique queries.",
            "",
            "## Retrieval settings",
            "",
            f"- **Endpoint:** `{settings['api_url']}`",
            f"- **Index:** `{settings['index']}`",
            f"- **Top-k:** {settings['top_k']}",
            f"- **Search method:** `{settings['search_method']}`",
            f"- **Timeout:** {settings['timeout']}s",
            f"- **Settings fingerprint:** `{manifest['settings_fingerprint']}`",
            "",
            "## Timestamps",
            "",
            f"- **Started:** {manifest['started_at']}",
            f"- **Completed:** {manifest.get('completed_at', 'N/A')}",
            f"- **Cache entries reused:** {cache_reused}",
            "",
            "## Result summary",
            "",
            f"- Successful (≥1 result, complete): {counts['success']}",
            f"- Successful with zero results: {counts['success_zero']}",
            f"- Successful but fewer than top-k: {counts['success_incomplete']}",
            f"- Failed: {counts['failed']}",
            "",
            "## Response contents",
            "",
            f"- Responses include snippets: {'yes' if snippets else 'no'}",
            f"- Responses include full document text: {'yes' if full_text else 'no'}",
            "",
            "## Reproduce / resume",
            "",
            "Run the exact command below to resume or reproduce this cache run:",
            "",
            "```bash",
            command,
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def write_manifest(manifest_path: Path, manifest: Dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())


def strip_secrets_from_command(command: str) -> str:
    """Remove any API key that may appear in the printed command."""
    if DEFAULT_API_KEY and DEFAULT_API_KEY in command:
        return command.replace(DEFAULT_API_KEY, "***")
    return command


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    if args.run_tests:
        run_tests()
        return

    if args.query_source is None:
        raise ValueError("--query-source is required unless --run-tests is set.")

    if not DEFAULT_API_KEY:
        raise ValueError(
            "CHATNOIR_API_KEY is not set. Add it to a .env file or environment."
        )

    source_path = Path(args.query_source)
    topics_path = Path(args.topics_file) if args.topics_file else None

    topics_map = load_topics_map(topics_path) if topics_path else None
    loaded = load_query_source(source_path, args.query_id_column, topics_map)
    logger.info("Loaded %d unique query IDs from %s", len(loaded), source_path)

    selected, requested_count = select_queries(loaded, args.num_queries)
    logger.info("Selected %d queries (requested %d)", len(selected), requested_count)

    settings = retrieval_settings(
        api_url=args.api_url,
        index=args.index,
        top_k=args.top_k,
        search_method=args.search_method,
        timeout=args.timeout,
    )
    fingerprint = settings_fingerprint(settings)

    run_dir = (
        Path(args.cache_root)
        / source_selection_dir(source_path, topics_path, [qid for qid, _ in selected])
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_name, readme_name = run_filenames(len(selected), args.top_k)
    manifest_path = run_dir / manifest_name

    started_at = datetime.now(timezone.utc).isoformat()
    manifest: Dict[str, Any] = {
        "requested_count": requested_count,
        "selected_count": len(selected),
        "query_source": str(source_path),
        "topics_file": str(topics_path) if topics_path else None,
        "settings": settings,
        "settings_fingerprint": fingerprint,
        "started_at": started_at,
        "completed_at": None,
        "cache_reuse_count": 0,
        "queries": [],
    }

    failed_ids: List[str] = []
    cache_reuse_count = 0

    for idx, (qid, query_text) in enumerate(selected, start=1):
        logger.info("[%d/%d] query_id=%s | %s", idx, len(selected), qid, query_text[:80])
        cache_path = cache_path_for_query(
            Path(args.cache_root), query_text, fingerprint
        )
        cached = load_cached_response(cache_path, query_text, fingerprint)

        if cached is not None:
            logger.info("[%d/%d] Cache hit for query_id=%s", idx, len(selected), qid)
            response = cached["response"]
            result_count = (
                len(response.get("results", []))
                if isinstance(response, dict)
                else 0
            )
            status = classify_result("success", result_count, args.top_k)
            entry = {
                "query_id": qid,
                "query_text": query_text,
                "status": status,
                "attempts": 0,
                "result_count": result_count,
                "error": None,
                "cache_location": str(cache_path.relative_to(Path(args.cache_root))),
                "cache_reused": True,
                "retrieved_at": cached.get("retrieved_at"),
            }
            cache_reuse_count += 1
        else:
            status, response, attempts, result_count, error, retrieved_at = retrieve_raw(
                query_text=query_text,
                api_url=args.api_url,
                api_key=DEFAULT_API_KEY,
                index=args.index,
                top_k=args.top_k,
                search_method=args.search_method,
                timeout=args.timeout,
                max_attempts=args.max_attempts,
            )
            final_status = classify_result(status, result_count, args.top_k)
            entry = {
                "query_id": qid,
                "query_text": query_text,
                "status": final_status,
                "attempts": attempts,
                "result_count": result_count,
                "error": error,
                "cache_location": str(cache_path.relative_to(Path(args.cache_root))),
                "cache_reused": False,
                "retrieved_at": retrieved_at,
            }
            if status != "failed":
                save_cached_response(
                    cache_path=cache_path,
                    query_text=query_text,
                    settings=settings,
                    fingerprint=fingerprint,
                    response=response,  # type: ignore[arg-type]
                    retrieved_at=retrieved_at or datetime.now(timezone.utc).isoformat(),
                )
            else:
                failed_ids.append(qid)

        manifest["queries"].append(entry)
        manifest["cache_reuse_count"] = cache_reuse_count
        write_manifest(manifest_path, manifest)

    completed_at = datetime.now(timezone.utc).isoformat()
    manifest["completed_at"] = completed_at
    write_manifest(manifest_path, manifest)

    # Build and write README from the final manifest.
    command_parts = [
        "python",
        str(Path(__file__).relative_to(Path.cwd())),
        "--num-queries",
        str(requested_count),
        "--top-k",
        str(args.top_k),
        "--query-source",
        str(source_path),
    ]
    if topics_path is not None:
        command_parts.extend(["--topics-file", str(topics_path)])
    command_parts.extend(
        [
            "--max-attempts",
            str(args.max_attempts),
            "--query-id-column",
            args.query_id_column,
            "--index",
            args.index,
            "--search-method",
            args.search_method,
            "--timeout",
            str(args.timeout),
        ]
    )
    command = strip_secrets_from_command(shlex.join(command_parts))
    readme = build_readme(manifest, run_dir, Path(args.cache_root), command)
    readme_path = run_dir / readme_name
    with readme_path.open("w", encoding="utf-8") as f:
        f.write(readme)
        f.flush()
        os.fsync(f.fileno())

    logger.info("=" * 70)
    logger.info("Run complete")
    logger.info("Manifest: %s", manifest_path)
    logger.info("README: %s", readme_path)
    logger.info("Selected queries: %d", len(selected))
    logger.info("Successful: %d", sum(1 for q in manifest["queries"] if q["status"] == "success"))
    logger.info("Zero results: %d", sum(1 for q in manifest["queries"] if q["status"] == "success_zero"))
    logger.info("Incomplete: %d", sum(1 for q in manifest["queries"] if q["status"] == "success_incomplete"))
    logger.info("Failed: %d", len(failed_ids))
    if failed_ids:
        logger.info("Failed query IDs: %s", ", ".join(failed_ids))
    logger.info("=" * 70)


# ---------------------------------------------------------------------------
# Lightweight tests
# ---------------------------------------------------------------------------


def run_tests() -> None:
    """Run a small deterministic test suite and exit."""
    logger.info("Running lightweight tests...")
    failures = []

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        # 1. CSV loading with ID-only and topics resolution.
        csv_path = root / "ids.csv"
        csv_path.write_text("query_id\n1048642\n524332\n1048642\n999999\n", encoding="utf-8")
        topics_path = root / "topics.tsv"
        topics_path.write_text(
            "query_id\tquery_text\n"
            "1048642\twhat is paranoid sc\n"
            "524332\ttreating tension headaches without medication\n",
            encoding="utf-8",
        )
        topics_map = load_topics_map(topics_path)
        loaded = load_query_source(csv_path, query_id_column="query_id", topics_map=topics_map)
        expected = [("1048642", "what is paranoid sc"), ("524332", "treating tension headaches without medication")]
        if loaded != expected:
            failures.append(f"CSV loading failed: {loaded}")

        # 2. TSV loading with embedded text.
        tsv_path = root / "topics.tsv"
        tsv_path.write_text(
            "query_id\tquery_text\n"
            "2\tAndrogen receptor define\n"
            "3\tsome query\n"
            "2\tduplicate\n",
            encoding="utf-8",
        )
        tsv_loaded = load_query_source(tsv_path)
        if tsv_loaded != [("2", "Androgen receptor define"), ("3", "some query")]:
            failures.append(f"TSV loading failed: {tsv_loaded}")

        # 3. Deterministic selection and discrepancy reporting.
        selected, requested = select_queries(loaded, 10)
        if len(selected) != 2 or requested != 10:
            failures.append(f"Selection failed: {selected}, requested={requested}")

        # 4. Settings fingerprint changes with top-k, not with query count.
        s50 = retrieval_settings("url", "idx", 50, "bm25", 30.0)
        s100 = retrieval_settings("url", "idx", 100, "bm25", 30.0)
        if settings_fingerprint(s50) == settings_fingerprint(s100):
            failures.append("Fingerprint did not change with top-k")

        # 5. Cache path does not include num-queries.
        path50 = cache_path_for_query(root, "hello", settings_fingerprint(s50))
        if "50" in str(path50.relative_to(root)) and "queries" in str(path50.relative_to(root)):
            failures.append("Cache path leaks query count")

        # 6. Cache compatibility (reuse on matching settings, reject on mismatch).
        settings = retrieval_settings("url", "idx", 50, "bm25", 30.0)
        fp = settings_fingerprint(settings)
        cache_path = cache_path_for_query(root, "hello", fp)
        save_cached_response(
            cache_path,
            "hello",
            settings,
            fp,
            {"results": [{"trec_id": "d1", "score": 1.0}]},
            "2024-01-01T00:00:00+00:00",
        )
        hit = load_cached_response(cache_path, "hello", fp)
        if hit is None or hit["response"]["results"][0]["trec_id"] != "d1":
            failures.append("Cache hit failed")
        miss = load_cached_response(cache_path, "hello", settings_fingerprint(s100))
        if miss is not None:
            failures.append("Cache should miss for mismatched settings")

        # 7. Numbered filenames use selected count and top-k.
        mname, rname = run_filenames(50, 50)
        if mname != "manifest_50queries_top50.json" or rname != "README_50queries_top50.md":
            failures.append(f"Wrong filenames: {mname}, {rname}")

        # 8. Resume from 50 to 100 reuses the first 50 cache entries.
        # Build a fake run with 50 selected and the same cache root.
        source_selection = root / "resume_test"
        run_dir = source_selection / "selection_hash"
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest_50 = {
            "requested_count": 50,
            "selected_count": 50,
            "query_source": str(csv_path),
            "topics_file": str(topics_path),
            "settings": settings,
            "settings_fingerprint": fp,
            "queries": [
                {
                    "query_id": str(i),
                    "query_text": f"query {i}",
                    "status": "success",
                    "cache_reused": True,
                    "cache_location": str(
                        cache_path_for_query(root, f"query {i}", fp).relative_to(root)
                    ),
                }
                for i in range(50)
            ],
        }
        # Create cache files for first 50.
        for i in range(50):
            save_cached_response(
                cache_path_for_query(root, f"query {i}", fp),
                f"query {i}",
                settings,
                fp,
                {"results": [{"trec_id": f"d{i}", "score": float(i)}]},
                "2024-01-01T00:00:00+00:00",
            )
        # Simulate 100-query selection: the first 50 must be reusable.
        hundred_ids = [str(i) for i in range(100)]
        reused = 0
        for qid in hundred_ids[:50]:
            text = f"query {qid}"
            cp = cache_path_for_query(root, text, fp)
            if load_cached_response(cp, text, fp) is not None:
                reused += 1
        if reused != 50:
            failures.append(f"Resume reuse failed: only {reused}/50 reusable")

        manifest_path = run_dir / "manifest_50queries_top50.json"
        write_manifest(manifest_path, manifest_50)
        if not manifest_path.exists():
            failures.append("Manifest write failed")

    if failures:
        for msg in failures:
            logger.error("TEST FAILED: %s", msg)
        sys.exit(1)

    logger.info("All lightweight tests passed.")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cache original-query ChatNoir retrieval responses.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-queries", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--query-source", type=Path, default=None)
    parser.add_argument("--topics-file", type=Path, default=None)
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument("--query-id-column", type=str, default="query_id")
    parser.add_argument("--index", type=str, default=DEFAULT_INDEX)
    parser.add_argument("--search-method", type=str, default=DEFAULT_SEARCH_METHOD)
    parser.add_argument("--api-url", type=str, default=DEFAULT_API_URL)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=DEFAULT_CACHE_ROOT,
    )
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Run lightweight tests and exit without calling ChatNoir.",
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
