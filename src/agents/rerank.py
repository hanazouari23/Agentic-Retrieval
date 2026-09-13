from sentence_transformers import CrossEncoder
from typing import List, Dict, Any
from ..core.agents import AgentBase
import numpy as np
import time

class RerankingAgent(AgentBase):
    """
    A re-ranking agent that uses a cross-encoder to re-score documents.
    """
    def __init__(self, embed_model, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        super().__init__(agent_id=1, embed_model=embed_model)
        self.model = CrossEncoder(model_name)

    def compute_effects(self, query_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rerank documents using cross-encoder.
        
        Args:
            query_features: Dict containing:
                - 'query_text': str - the query
                - 'doc_ids': List[str] - current document IDs
                - 'doc_scores': np.ndarray - current document scores
                - 'corpus': Dict[str, str] - document text corpus
                - 'top_k_rerank': int - how many top documents to rerank
                
        Returns:
            Dict with:
                - 'new_doc_ids': List[str] - reranked document IDs
                - 'new_doc_scores': np.ndarray - reranked document scores
                - 'elapsed_time': float - time taken for reranking
        """
        query = query_features['query_text']
        doc_ids = query_features['doc_ids']
        doc_scores = query_features['doc_scores']
        corpus = query_features['corpus']
        top_k = query_features.get('top_k_rerank', min(10, len(doc_ids)))
        
        # Limit reranking to top_k documents
        rerank_ids = doc_ids[:top_k]
        rerank_scores = doc_scores[:top_k]
        
        # Create query-document pairs
        pairs = [(query, corpus.get(doc_id, "")) for doc_id in rerank_ids]
        
        # Rerank with cross-encoder
        start_time = time.time()
        ce_scores = self.model.predict(pairs, show_progress_bar=False)
        elapsed_time = time.time() - start_time
        
        # Sort by cross-encoder scores (descending)
        order = np.argsort(ce_scores)[::-1]
        reranked_ids = [rerank_ids[i] for i in order]
        reranked_scores = np.asarray(ce_scores)[order].astype(np.float32)
        
        # Combine reranked top_k with remaining documents
        remaining_ids = doc_ids[top_k:]
        remaining_scores = doc_scores[top_k:]
        
        new_doc_ids = reranked_ids + remaining_ids
        new_doc_scores = np.concatenate([reranked_scores, remaining_scores])
        
        return {
            'new_doc_ids': new_doc_ids,
            'new_doc_scores': new_doc_scores,
            'elapsed_time': elapsed_time,
            'cost': 0.3
        }
