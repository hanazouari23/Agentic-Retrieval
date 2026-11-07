"""ChatNoir retriever agent."""

import os
import time
from typing import Any, Dict, List, Optional

import numpy as np

from .base import AgentInterface

# Re-use the project retriever if it is importable; otherwise fall back to a
# minimal ChatNoir client. The project root is two levels above agentsim_experiment.
try:
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.utils.retriever import Retriever, create_retriever_callable
except Exception:  # pragma: no cover - fallback for standalone use
    Retriever = None
    create_retriever_callable = None


class ChatNoirRetrieverAgent(AgentInterface):
    """Retrieve top-k passages from ChatNoir for the current query."""

    name = "chatnoir_retriever"

    def __init__(
        self,
        top_k: int = 20,
        index: str = "msmarco-v2.1",
        search_method: str = "bm25",
        adaptive_top_k: bool = True,
        min_top_k: int = 1,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.top_k = top_k
        self.index = index
        self.search_method = search_method
        self.adaptive_top_k = adaptive_top_k
        self.min_top_k = min_top_k

        self._api_url = api_url
        self._api_key = api_key
        self._index = index
        self._retrieve = None

    def _ensure_retriever(self):
        if self._retrieve is not None:
            return
        if Retriever is None or create_retriever_callable is None:
            raise RuntimeError(
                "ChatNoir retriever unavailable: src.utils.retriever could not be imported"
            )
        retriever_instance = Retriever(
            api_url=self._api_url,
            api_key=self._api_key,
            index=self._index,
            top_k=self.top_k,
        )
        self._retrieve = create_retriever_callable(retriever_instance)

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_retriever()

        query = state.get("query", "")
        if isinstance(query, dict):
            query = query.get("text", "")
        query = str(query)

        start = time.perf_counter()
        doc_ids, scores, corpus = self._retrieve(query, self.top_k)
        elapsed = time.perf_counter() - start

        retrieved = [
            {
                "id": doc_id,
                "text": corpus.get(doc_id, ""),
                "score": float(score),
            }
            for doc_id, score in zip(doc_ids, scores)
        ]

        state["retrieved"] = retrieved
        state.setdefault("corpus", {}).update(corpus)
        state.setdefault("latency_chatnoir_retriever", []).append(elapsed)

        return state
