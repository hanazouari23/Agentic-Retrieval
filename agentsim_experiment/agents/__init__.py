"""Pipeline agents for the MS MARCO online evaluation."""

from .base import AgentInterface
from .chatnoir_retriever import ChatNoirRetrieverAgent
from .filter import FilterAgent
from .answer_drafter import AnswerDrafterAgent
from .finalizer import FinalizerAgent

__all__ = [
    "AgentInterface",
    "ChatNoirRetrieverAgent",
    "FilterAgent",
    "AnswerDrafterAgent",
    "FinalizerAgent",
]


AGENT_REGISTRY = {
    "chatnoir_retriever": ChatNoirRetrieverAgent,
    "filter": FilterAgent,
    "answer_drafter": AnswerDrafterAgent,
    "finalizer": FinalizerAgent,
}


def build_agents(config: dict) -> dict:
    """Build all agents from an eval_config dictionary."""
    agents = {}
    for name, cls in AGENT_REGISTRY.items():
        agent_cfg = config.get("agents", {}).get(name, {})
        agents[name] = cls(**agent_cfg)
    return agents
