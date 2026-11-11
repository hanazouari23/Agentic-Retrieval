import json
import logging
import os
import time
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import requests
from dotenv import load_dotenv


logger = logging.getLogger(__name__)


def _is_transient(exc: requests.RequestException) -> bool:
    """Return True if the request exception looks retryable."""
    if exc.response is not None:
        return exc.response.status_code >= 500 or exc.response.status_code == 429
    return True


# def _adaptive_sizes(top_k: int, min_top_k: int) -> List[int]:
#     """Return a descending list of retrieval depths to try on 500 errors."""
#     sizes: List[int] = []
#     size = top_k
#     while size >= min_top_k:
#         sizes.append(size)
#         size = size // 2
#     if min_top_k not in sizes:
#         sizes.append(min_top_k)
#     return sizes


load_dotenv()


CHATNOIR_API_URL = os.getenv(
    "CHATNOIR_API_URL",
    "https://www.chatnoir.eu/api/v1/_search",
).strip()
CHATNOIR_API_KEY = os.getenv("CHATNOIR_API_KEY", "").strip()


class Retriever:
    """
    Retriever that queries the ChatNoir REST API and returns the top-k
    documents from the MS MARCO V2.1 segmented index using BM25.
    """

    def __init__(
        self,
        api_url: str = CHATNOIR_API_URL,
        api_key: str = CHATNOIR_API_KEY,
        index: str = "msmarco-v2.1",
        top_k: int = 50,
        timeout: float = 30.0,
    ):
        """
        Initialize the ChatNoir retriever.

        Args:
            api_url: ChatNoir search endpoint.
            api_key: ChatNoir API key.
            index: ChatNoir index short name.
            top_k: Default number of documents to retrieve.
            timeout: Request timeout in seconds.
        """
        if not api_key:
            raise ValueError(
                "A ChatNoir API key is required. Set CHATNOIR_API_KEY in your environment."
            )

        self.api_url = api_url
        self.api_key = api_key
        self.index = index
        self.top_k = top_k
        self.timeout = timeout

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        search_method: str = "bm25",
        adaptive_top_k: bool = False,
        min_top_k: int = 1,
        max_attempts: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top documents for a query using ChatNoir.

        Args:
            query: The search query string.
            top_k: Number of documents to retrieve; overrides the instance default.
            search_method: ChatNoir retrieval model; use "bm25" for plain BM25.
            adaptive_top_k: If True, halve the requested ``size`` on 500 errors
                until ``min_top_k`` is reached.  This works around ChatNoir
                returning 500 when ``size`` exceeds the number of matches for a
                restrictive Boolean query.
            min_top_k: Minimum retrieval depth when adaptive fallback is used.
            max_attempts: Retries per candidate size for transient failures.

        Returns:
            A list of dictionaries with keys:
              - id: ChatNoir document UUID
              - text: document snippet
              - score: ranking score
              - source: full ChatNoir result object
        """
        if top_k is None:
            top_k = self.top_k

        # Adaptive sizing disabled for consistent retrieval depth across queries.
        # candidate_sizes = (
        #     _adaptive_sizes(top_k, min_top_k) if adaptive_top_k else [top_k]
        # )
        # candidate_sizes = [top_k]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Optional[Exception] = None

        # Adaptive sizing disabled: use the requested top_k only.
        # for size in candidate_sizes:
        size = top_k
        payload = {
            "query": query,
            "index": [self.index],
            "size": size,
            "search_method": search_method,
        }

        for attempt in range(max_attempts):
            try:
                response = requests.post(
                    self.api_url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()

                data = response.json()
                results = data.get("results", []) if isinstance(data, dict) else []

                documents = []
                for result in results:
                    documents.append(
                        {
                            "id": result.get("trec_id"),
                            "text": result.get("snippet", ""),
                            "score": float(result.get("score", 0.0)),
                            "source": result,
                        }
                    )

                # Adaptive fallback logging disabled.
                # if size != top_k:
                #     logger.info(
                #         "Adaptive fallback succeeded for query %r at size=%d (requested %d)",
                #         query[:80],
                #         size,
                #         top_k,
                #     )
                return documents

            except requests.exceptions.RequestException as exc:
                last_error = exc
                if _is_transient(exc) and attempt < max_attempts - 1:
                    wait = 2 ** attempt
                    logger.warning(
                        "Transient retrieval failure (attempt %d/%d, size=%d) for query %r; retrying in %ds: %s",
                        attempt + 1,
                        max_attempts,
                        size,
                        query[:80],
                        wait,
                        exc,
                    )
                    time.sleep(wait)
                    continue
                # Give up on this attempt.
                break
            except json.JSONDecodeError as exc:
                last_error = exc
                break

        logger.warning(
            "All retrieval attempts failed for query %r: %s",
            query[:80],
            last_error,
        )
        return []


def create_retriever_callable(retriever_instance: Retriever) -> callable:
    """
    Create a callable matching the Simulation's expected retriever interface.

    Args:
        retriever_instance: An instance of Retriever.

    Returns:
        A callable taking (query, top_k) and returning:
          - doc_ids: list of ChatNoir document UUIDs
          - scores: NumPy array of ranking scores
          - corpus_data: mapping from document UUID to snippet text
    """
    def retriever_func(
        query: str, top_k: int
    ) -> Tuple[List[str], np.ndarray, Dict[str, str]]:
        results = retriever_instance.retrieve(query, top_k=top_k)

        doc_ids = [doc["id"] for doc in results]
        scores = np.array([doc["score"] for doc in results], dtype=np.float32)
        corpus_data = {doc["id"]: doc["text"] for doc in results}

        return doc_ids, scores, corpus_data

    return retriever_func