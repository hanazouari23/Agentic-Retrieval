"""Finalizer agent: produce the final ranked passage list."""

import time
from typing import Any, Dict, List, Tuple

from .base import AgentInterface


class FinalizerAgent(AgentInterface):
    """Finalize a ranking of passages for the run file.

    The finalizer combines the filtered passages with any drafted answers and
    returns a ranked list of (passage_id, score) tuples. When no passages are
    available it returns an empty ranking so the run file can still record the
    query as attempted.
    """

    name = "finalizer"

    def __init__(self, top_k: int = 20):
        self.top_k = top_k

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        filtered = state.get("filtered", [])
        drafted = state.get("drafted_answers", [])

        start = time.perf_counter()
        ranked: List[Tuple[str, float]] = []

        if drafted:
            # Score passages by how many answer candidates they overlap with.
            for rank, passage in enumerate(filtered):
                text = passage.get("text", "").lower()
                overlap = sum(1 for ans in drafted if ans.lower() in text)
                score = float(len(drafted) - rank + overlap)
                ranked.append((passage.get("id", f"passage_{rank}"), score))
        else:
            # No drafted answers: fall back to retrieval score ordering.
            for rank, passage in enumerate(filtered):
                score = passage.get("score", float(len(filtered) - rank))
                ranked.append((passage.get("id", f"passage_{rank}"), score))

        # Sort by descending score.
        ranked.sort(key=lambda x: x[1], reverse=True)
        ranked = ranked[: self.top_k]

        state["ranked_passages"] = ranked
        state["final_answer"] = drafted[0] if drafted else ""
        state.setdefault("latency_finalizer", []).append(time.perf_counter() - start)

        return state
