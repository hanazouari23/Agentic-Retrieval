"""Load a frozen d3rlpy DiscreteCQL policy for greedy action selection."""

from pathlib import Path
from typing import Any

import numpy as np


def load_cql_policy(model_path: str | Path) -> Any:
    """Load a trained d3rlpy learnable model from disk."""
    import d3rlpy

    return d3rlpy.load_learnable(str(model_path))


def select_action(cql: Any, observation: np.ndarray) -> int:
    """Return the greedy action selected by the CQL policy."""
    obs = np.asarray(observation, dtype=np.float32).reshape(1, -1)
    action = int(np.asarray(cql.predict(obs)).reshape(-1)[0])
    return action
