"""Unit tests for the Gymnasium pipeline environment."""

import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from env.action_registry import load_action_registry
from env.pipeline_env import MSMARCOOnlinePipelineEnv
from policy.observation_builder import ObservationBuilder


class MockAgent:
    name = "mock"

    def __init__(self, name: str):
        self.name = name

    def run(self, state):
        # The environment counts calls; agents only update tool history.
        state["previous_tools"].append(self.name)
        if self.name == "finalizer":
            state["ranked_passages"] = [("doc_1", 1.0), ("doc_2", 0.5)]
        return state


def test_env_step():
    registry = load_action_registry(project_root / "configs" / "task_config.json")
    builder = ObservationBuilder.from_task_config(project_root / "configs" / "task_config.json")
    agents = {
        "chatnoir_retriever": MockAgent("chatnoir_retriever"),
        "filter": MockAgent("filter"),
        "answer_drafter": MockAgent("answer_drafter"),
        "finalizer": MockAgent("finalizer"),
    }

    env = MSMARCOOnlinePipelineEnv(
        query={"qid": "1048575", "text": "what is a business plan"},
        agents=agents,
        observation_builder=builder,
        action_registry=registry,
    )
    obs, info = env.reset(seed=42)
    assert obs.shape == (525,)
    assert info["query_id"] == "1048575"

    obs, reward, terminated, truncated, info = env.step(4)
    assert terminated is True
    assert info["selected_action"] == 4
    assert info["pipeline_steps"] == registry[4]
    assert info["agent_calls"]["chatnoir_retriever"] == 1
    assert len(info["ranked_passages"]) == 2
    assert obs.shape == (525,)
    print("pipeline env OK")


if __name__ == "__main__":
    test_env_step()
