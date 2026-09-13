from openai import OpenAI
from sentence_transformers import SentenceTransformer
from ..core.agents import AgentBase
import os
import numpy as np
import time
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv

# system message sent to the LLM when reformulating queries
SYSTEM_PROMPT = (
    "You are a query rewriting assistant. "
    "Given a user query, you rewrite it into a clearer, more complete search query. "
    "Preserve the original intent and return only the rewritten query."
)
load_dotenv()
API_KEY = os.getenv("LLMAPI_KEY")
BASE_URL = os.getenv("BASE_URL_HPC")  # or BASE_URL_UNI depending on the environment
MODEL_NAME = os.getenv("MODEL_NAME_HPC")  # or MODEL_NAME_UNI depending on the environment
if not MODEL_NAME:
    raise RuntimeError("Missing environment variable: MODEL_NAME_HPC")


class ReformulationAgent(AgentBase):
    def __init__(self, embed_model: SentenceTransformer):
        super().__init__(agent_id=0, embed_model=embed_model)
        self.client = OpenAI(
            base_url=BASE_URL,
            api_key=API_KEY,
            default_headers={
                "HTTP-Referer": "MAESTRO-Query-Reformulator",
                "X-Title": "Query Reformulator",
            }
        )
    
    
    def compute_effects(self, query_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reformulate the query and retrieve new documents.
        
        Args:
            query_features: Dict containing:
                - 'query_text': str - the original query
                - 'retriever': callable - function to retrieve documents
                
        Returns:
            Dict with:
                - 'new_query_text': str - reformulated query
                - 'new_doc_ids': List[str] - new document IDs
                - 'new_doc_scores': np.ndarray - new document scores
                - 'elapsed_time': float - time taken for reformulation + retrieval
        """
        original_query = query_features['query_text']
        retriever = query_features['retriever']
        top_k = query_features['top_k']
        
        # 1. Reformulate query with LLM
        start_time = time.time()
        reformulated_query = self._call_llm(original_query)
        reformulation_time = time.time() - start_time
        
        # 2. Retrieve new documents with reformulated query
        retrieval_start = time.time()
        raw_results = retriever(reformulated_query, top_k)
        retrieval_time = time.time() - retrieval_start
        
        # 3. Extract doc_ids and scores
        new_doc_ids = raw_results[0] if raw_results else []
        new_doc_scores = np.array([result for result in raw_results[1]], dtype=np.float32)
        
        total_elapsed = reformulation_time + retrieval_time
        
        return {
            'new_query_text': reformulated_query,
            'new_doc_ids': new_doc_ids,
            'new_doc_scores': new_doc_scores,
            'elapsed_time': total_elapsed,
            'cost': 1.0  # Assuming a fixed cost for the LLM call; adjust as needed
        }
    
    def _call_llm(self, query: str) -> str:
        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.4,
            max_tokens=140,
        )
        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError("No content returned from LLM")
        content = content.strip()
       
        return content