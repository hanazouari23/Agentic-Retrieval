from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer


class AgentBase(ABC):
    """
    Base class for query optimization agents.
    
    Agents are pure domain-logic transformers: they receive query features
    and return modified artifacts (query text, document rankings, embeddings).
    
    MDP concerns (state management, metric computation, reward calculation)
    are handled by the Simulation class, not by agents.
    """
    
    def __init__(self, agent_id: int, embed_model: SentenceTransformer):
        """
        Parameters
        ----------
        agent_id : int
            Agent identifier (0=QueryReform, 1=Rerank, 2=PRF, 3=STOP)
        embed_model : SentenceTransformer
            Shared query encoder for embedding consistency
        """
        self.agent_id = agent_id
        self.embed_model = embed_model
    
    @abstractmethod
    def compute_effects(self, query_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply agent-specific transformation to query/documents.
        
        This method should contain only domain logic. The agent modifies
        inputs and returns results; Simulation handles metric computation,
        state updates, and reward calculation.
        
        Parameters
        ----------
        query_features : Dict[str, Any]
            Input features (varies by agent). May include:
            - 'query_text': current query string
            - 'embedding': current query embedding (768-d)
            - 'doc_ids': current ranked document IDs
            - 'doc_scores': current retrieval scores
            - 'corpus_data': mapping of doc_id to document text
        
        Returns
        -------
        Dict[str, Any]
            Agent-produced effects. Should include any/all of:
            - 'new_query_text': reformulated query (if applicable)
            - 'new_embedding': re-embedded query (if query changed)
            - 'new_doc_ids': reordered document list (if applicable)
            - 'new_doc_scores': updated scores (if applicable)
            - 'elapsed_time': wall-clock time for this agent call (seconds)
            - Any other agent-specific metadata for logging
        """
        pass
