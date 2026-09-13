import logging
import os
from typing import List, Dict, Any, Tuple

import numpy as np
import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

ENDPOINT = os.getenv("RETRIEVAL_ENDPOINT")
MY_USERNAME = os.getenv("MY_USERNAME")
MY_PASSWORD = os.getenv("MY_PASSWORD")


class Retriever:
    """
    Retriever that accesses OpenSearch endpoint to retrieve top 50 documents
    from the msmarco-v2.1-segmented index using BM25 matching.
    """
    
    def __init__(self, endpoint: str = ENDPOINT,
                 username: str = MY_USERNAME,
                 password: str = MY_PASSWORD,
                 index_field: str = "segment",
                 top_k: int = 50):
        """
        Initialize the retriever with OpenSearch credentials and parameters.
        
        Args:
            endpoint: OpenSearch endpoint URL
            username: Username for authentication
            password: Password for authentication
            index_field: Field to search in (default: "segment")
            top_k: Number of top documents to retrieve (default: 50)
        """
        self.endpoint = endpoint
        self.username = username
        self.password = password
        self.index_field = index_field
        self.top_k = top_k
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve top documents for a given query using BM25 matching.
        
        Args:
            query: The search query string
            top_k: Number of top documents to retrieve (overrides instance default if provided)
        
        Returns:
            List of documents with id, text (segment), and score
        """
        if top_k is None:
            top_k = self.top_k
        
        # Construct OpenSearch query
        search_body = {
            "query": {
                "match": {
                    self.index_field: query
                }
            },
            "size": top_k
        }
        
        try:
            # Make authenticated request to OpenSearch
            response = requests.get(
                self.endpoint,
                json=search_body,
                auth=(self.username, self.password),
                headers={"Content-Type": "application/json"},
                verify=False  # Note: Set to True in production with proper certificates
            )
            response.raise_for_status()
            
            # Parse results
            results = response.json()
            documents = []
            
            if "hits" in results and "hits" in results["hits"]:
                for hit in results["hits"]["hits"]:
                    doc = {
                        "id": hit.get("_id"),
                        "text": hit.get("_source", {}).get(self.index_field, ""),
                        "score": hit.get("_score", 0.0),
                        "source": hit.get("_source", {})  # Full document source
                    }
                    documents.append(doc)
            
            return documents
        
        except requests.exceptions.RequestException as e:
            logger.warning("Error retrieving documents: %s", e)
            return []


def create_retriever_callable(retriever_instance: Retriever) -> callable:
    """
    Create a callable that matches the Simulation's expected retriever interface.
    
    Args:
        retriever_instance: An instance of the Retriever class
        
    Returns:
        A callable that takes a query string and returns (doc_ids, scores, corpus_data) where:
        - doc_ids: List of document IDs
        - scores: np.ndarray of BM25 scores
        - corpus_data: Dict mapping doc_id -> text
    """
    def retriever_func(query: str, top_k: int) -> Tuple[List[str], np.ndarray, Dict[str, str]]:
        results = retriever_instance.retrieve(query, top_k=top_k)
        doc_ids = [doc['id'] for doc in results]
        scores = np.array([doc['score'] for doc in results], dtype=np.float32)
        corpus_data = {doc['id']: doc['text'] for doc in results}
        return doc_ids, scores, corpus_data
    
    return retriever_func
