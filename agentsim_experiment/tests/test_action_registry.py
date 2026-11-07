"""Unit tests for the action registry."""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from env.action_registry import load_action_registry


def test_registry():
    registry = load_action_registry(project_root / "configs" / "task_config.json")
    assert max(registry) + 1 == 9
    assert len(registry) == 9
    assert registry[4] == ["chatnoir_retriever", "filter", "answer_drafter", "finalizer"]
    assert registry[2] == ["chatnoir_retriever"]
    assert all(isinstance(v, list) for v in registry.values())
    print("action registry OK")


if __name__ == "__main__":
    test_registry()
