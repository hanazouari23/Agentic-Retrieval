from openai import OpenAI
from sentence_transformers import SentenceTransformer
from ..core.agents import AgentBase
from .boolean_formulate import BooleanFormulationAgent
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
    def __init__(
        self,
        embed_model: SentenceTransformer,
        formulation_mode: str = "natural",
    ):
        """
        Args:
            embed_model: Shared query encoder.
            formulation_mode: 'natural' for the original LLM rewrite, or
                'boolean' for concept-based Boolean formulation.
        """
        super().__init__(agent_id=0, embed_model=embed_model)
        self.formulation_mode = formulation_mode.lower()
        if self.formulation_mode not in ("natural", "boolean"):
            raise ValueError(
                f"Unknown formulation_mode: {formulation_mode!r}. "
                "Use 'natural' or 'boolean'."
            )

        # Natural-language path keeps the original client.
        self.client = OpenAI(
            base_url=BASE_URL,
            api_key=API_KEY,
            default_headers={
                "HTTP-Referer": "MAESTRO-Query-Reformulator",
                "X-Title": "Query Reformulator",
            }
        )

        # Boolean path is delegated to a dedicated agent.
        self._boolean_agent: BooleanFormulationAgent | None = None
        if self.formulation_mode == "boolean":
            self._boolean_agent = BooleanFormulationAgent(embed_model=embed_model)

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
                - 'new_corpus_data': Dict[str, str] - document text mapping
                - 'elapsed_time': float - time taken for reformulation + retrieval
                - 'formulation_mode': str - which path was used
        """
        if self.formulation_mode == "boolean" and self._boolean_agent is not None:
            return self._boolean_agent.compute_effects(query_features)

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

        # 3. Extract doc_ids, scores and document text
        if not raw_results:
            new_doc_ids: List[str] = []
            new_doc_scores = np.array([], dtype=np.float32)
            new_corpus_data: Dict[str, str] = {}
        else:
            new_doc_ids = raw_results[0]
            new_doc_scores = np.array([result for result in raw_results[1]], dtype=np.float32)
            new_corpus_data = raw_results[2] if len(raw_results) > 2 else {}

        total_elapsed = reformulation_time + retrieval_time

        return {
            'new_query_text': reformulated_query,
            'new_doc_ids': new_doc_ids,
            'new_doc_scores': new_doc_scores,
            'new_corpus_data': new_corpus_data,
            'elapsed_time': total_elapsed,
            'cost': 1.0,
            'formulation_mode': 'natural',
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