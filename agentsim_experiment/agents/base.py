"""Common agent interface for the MS MARCO online pipeline."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class AgentInterface(ABC):
    """Base interface all pipeline agents implement."""

    name: str = "agent"

    @abstractmethod
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent on the shared state and return the updated state."""
        raise NotImplementedError

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(state)
