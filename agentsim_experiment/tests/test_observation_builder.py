"""Unit tests for the observation builder."""

import json
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from d3rlpy.dataset import load, Episode
from policy.observation_builder import ObservationBuilder


def test_observation_matches_h5():
    config_path = project_root / "configs" / "task_config.json"
    builder = ObservationBuilder.from_task_config(config_path)

    manifest_path = project_root / "episode_manifest.jsonl"
    with open(manifest_path, "r", encoding="utf-8") as f:
        episode = json.loads(f.readline())

    query = episode["states"][0]["query"]
    state = {
        "query": query,
        "filtered": [],
        "rounds_used": 0,
        "rounds_remaining": 2,
        "agent_calls": {},
        "previous_tools": [],
    }
    built = builder.build(state)

    h5_path = project_root / "msmarco_train_d3rlpy.h5"
    with open(h5_path, "rb") as f:
        episodes = load(Episode, f)
    expected = episodes[0].observations[0]

    assert built.shape == (525,), built.shape
    assert expected.shape == (525,), expected.shape
    assert np.allclose(built, expected), (
        f"Observation mismatch: max diff {np.max(np.abs(built - expected))}"
    )
    print("observation builder matches H5 dataset")


if __name__ == "__main__":
    test_observation_matches_h5()
