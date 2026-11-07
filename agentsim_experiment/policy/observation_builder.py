"""Observation builder that reproduces the 525-dimensional training feature vector."""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer


class ObservationBuilder:
    """Build a (525,) float32 observation from the pipeline state.

    Layout (must match the offline H5 dataset exactly):
        [0:512]   query hash (HashingVectorizer, word unigrams+bigrams, l2,
                             alternate_sign=False)
        [512]     log1p(evidence_count) / log(101)
        [513]     has_evidence
        [514]     rounds_used / max_rounds
        [515]     rounds_remaining / max_rounds
        [516]     log1p(history_count:chatnoir_retriever) / log(11)
        [517]     log1p(history_count:filter) / log(11)
        [518]     log1p(history_count:answer_drafter) / log(11)
        [519]     log1p(history_count:finalizer) / log(11)
        [520]     last_tool == none
        [521]     last_tool == chatnoir_retriever
        [522]     last_tool == filter
        [523]     last_tool == answer_drafter
        [524]     last_tool == finalizer
    """

    def __init__(
        self,
        query_hash_dim: int = 512,
        max_rounds: int = 2,
        tools: Optional[List[str]] = None,
    ):
        self.query_hash_dim = query_hash_dim
        self.max_rounds = max_rounds
        self.tools = tools or [
            "chatnoir_retriever",
            "filter",
            "answer_drafter",
            "finalizer",
        ]
        self._log_evidence_norm = math.log(101)
        self._log_history_norm = math.log(11)

        # Match the task_config feature description exactly.
        self._vectorizer = HashingVectorizer(
            n_features=query_hash_dim,
            analyzer="word",
            ngram_range=(1, 2),
            norm="l2",
            alternate_sign=False,
            dtype=np.float32,
        )

    @classmethod
    def from_task_config(cls, task_config_path: str | Path) -> "ObservationBuilder":
        """Create an ObservationBuilder from task_config.json."""
        with open(task_config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        return cls(
            query_hash_dim=config.get("query_hash_dim", 512),
            max_rounds=config.get("max_rounds", 2),
            tools=config.get("tools", None),
        )

    def _hash_query(self, query: str) -> np.ndarray:
        """Return the L2-normalized query hash vector."""
        vector = self._vectorizer.transform([query]).toarray().astype(np.float32)
        return vector.reshape(-1)

    def _numeric_features(self, state: Dict[str, Any]) -> np.ndarray:
        """Compute the 13 numeric features from the state dictionary."""
        evidence_count = len(state.get("filtered", []))
        has_evidence = float(evidence_count > 0)
        evidence_norm = math.log1p(evidence_count) / self._log_evidence_norm

        rounds_used = state.get("rounds_used", 0)
        rounds_remaining = state.get("rounds_remaining", self.max_rounds - rounds_used)
        rounds_used_norm = rounds_used / max(self.max_rounds, 1)
        rounds_remaining_norm = rounds_remaining / max(self.max_rounds, 1)

        agent_calls = state.get("agent_calls", {})
        history_counts = [
            agent_calls.get(tool, 0) for tool in self.tools
        ]
        history_norms = [
            math.log1p(count) / self._log_history_norm for count in history_counts
        ]

        previous_tools = state.get("previous_tools", [])
        last_tool = previous_tools[-1] if previous_tools else "none"
        last_tool_onehot = [
            float(last_tool == label)
            for label in ["none"] + self.tools
        ]

        features = [
            evidence_norm,
            has_evidence,
            rounds_used_norm,
            rounds_remaining_norm,
            *history_norms,
            *last_tool_onehot,
        ]
        return np.asarray(features, dtype=np.float32)

    def build(self, state: Dict[str, Any]) -> np.ndarray:
        """Build the full 525-dimensional observation vector."""
        query = state.get("query", "")
        if isinstance(query, dict):
            query = query.get("text", "")
        query = str(query)

        query_hash = self._hash_query(query)
        numeric = self._numeric_features(state)

        observation = np.concatenate([query_hash, numeric]).astype(np.float32)
        assert observation.shape == (525,), (
            f"Observation shape mismatch: expected (525,), got {observation.shape}"
        )
        return observation


if __name__ == "__main__":
    import sys

    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/task_config.json"
    builder = ObservationBuilder.from_task_config(config_path)

    state = {
        "query": "how old is eva link",
        "filtered": [],
        "rounds_used": 0,
        "rounds_remaining": 2,
        "agent_calls": {},
        "previous_tools": [],
    }
    obs = builder.build(state)
    print("shape:", obs.shape)
    print("last13:", obs[512:525])
    print("nonzero indices:", np.flatnonzero(obs).tolist())
