import logging
import os
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import requests
from dotenv import load_dotenv


logger = logging.getLogger(__name__)


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
        index: str = "msmarco-v2.1-segmented",
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
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top documents for a query using ChatNoir.

        Args:
            query: The search query string.
            top_k: Number of documents to retrieve; overrides the instance default.
            search_method: ChatNoir retrieval model; use "bm25" for plain BM25.

        Returns:
            A list of dictionaries with keys:
              - id: ChatNoir document UUID
              - text: document snippet
              - score: ranking score
              - source: full ChatNoir result object
        """
        if top_k is None:
            top_k = self.top_k

        payload = {
            "query": query,
            "index": [self.index],
            "size": top_k,
            "search_method": search_method,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                self.api_url,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()

            data = response.json()
            results = data.get("results", [])

            documents = []
            for result in results:
                documents.append(
                    {
                        "id": result.get("uuid"),
                        "text": result.get("snippet", ""),
                        "score": float(result.get("score", 0.0)),
                        "source": result,
                    }
                )

            return documents

        except requests.exceptions.RequestException as exc:
            logger.warning("Error retrieving documents from ChatNoir: %s", exc)
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