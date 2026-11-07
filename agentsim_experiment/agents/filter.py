"""Passage filter agent."""

import time
from typing import Any, Dict

from .base import AgentInterface


class FilterAgent(AgentInterface):
    """Filter retrieved passages by length and retrieval score."""

    name = "filter"

    def __init__(
        self,
        min_length: int = 30,
        max_length: int = 10000,
        min_score: float = 0.0,
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.min_score = min_score

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        retrieved = state.get("retrieved", [])
        if not retrieved:
            state["filtered"] = []
            return state

        start = time.perf_counter()
        filtered = []
        for passage in retrieved:
            text = passage.get("text", "")
            score = passage.get("score", 0.0)
            length = len(text)
            if (
                self.min_length <= length <= self.max_length
                and score >= self.min_score
            ):
                filtered.append(passage)

        state["filtered"] = filtered
        state.setdefault("latency_filter", []).append(time.perf_counter() - start)

        return state
