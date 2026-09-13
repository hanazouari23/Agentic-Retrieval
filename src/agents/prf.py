
import time
import re
from typing import Any, Dict, List, Tuple

import numpy as np

from src.core.agents import AgentBase
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from collections import defaultdict

from src.utils import retriever



class PRFAgent(AgentBase):

    """Pseudo-Relevance Feedback (PRF) agent: Takes the top-k BM25 documents and extracts the most 
    discriminative terms from their content using TF-IDF weighting. These terms are used to compute 
    PRF-based scores for documents, which are then re-ranked to surface documents the initial search missed. """

    def __init__(self, embed_model, num_expansion_terms):
        super().__init__(agent_id=2, embed_model=embed_model)
        self.num_expansion_terms = num_expansion_terms
    
    def _reciprocal_rank_fusion(
    self,
    result_lists: List[List[str]],
    top_k: int,
    rrf_k: int = 60,
    ) -> Tuple[List[str], np.ndarray]:
        """
        Fuse ranked document-ID lists using Reciprocal Rank Fusion.

        A document receives 1 / (rrf_k + rank) for every list in which
        it occurs. Ranks start at 1.
        """
        fused_scores = defaultdict(float)

        for doc_ids in result_lists:
            for rank, doc_id in enumerate(doc_ids, start=1):
                fused_scores[doc_id] += 1.0 / (rrf_k + rank)

        ranked_docs = sorted(
            fused_scores.items(),
            key=lambda item: (-item[1], item[0]),  # deterministic tie-break
        )[:top_k]

        doc_ids = [doc_id for doc_id, _ in ranked_docs]
        scores = np.asarray(
            [score for _, score in ranked_docs],
            dtype=np.float32,
        )

        return doc_ids, scores

    def compute_effects(self, query_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply PRF: extract expansion terms from top-k documents, expand query, re-retrieve.
        
        Args:
            query_features: Dict containing:
                - 'query_text': str - the original query
                - 'retriever': callable - function to retrieve documents
                - 'top_k_rerank': int - TOP_K: number of initial documents to retrieve
                - 'top_k_prf': int - TOP_K_PRF: number of top documents to extract expansion terms from
                
        Returns:
            Dict with:
                - 'new_query_text': str - expanded query
                - 'new_doc_ids': List[str] - document IDs from re-retrieval
                - 'new_doc_scores': np.ndarray - scores from re-retrieval
                - 'elapsed_time': float - time taken for PRF process
        """
        start_time = time.time()
        
        retriever = query_features['retriever']
        original_query = query_features['query_text']
        top_k = query_features['top_k_rerank']
        top_k_prf = query_features['top_k_prf']
        
        # 1. Initial retrieval: get TOP_K documents
        initial_results = retriever(original_query, top_k)
        
        if not initial_results or len(initial_results[0]) == 0:
            elapsed_time = time.time() - start_time
            return {
                'new_query_text': original_query,
                'new_doc_ids': [],
                'new_doc_scores': np.array([], dtype=np.float32),
                'elapsed_time': elapsed_time,
            }
        
        initial_doc_ids = initial_results[0]
        initial_scores = initial_results[1]
        segments_dict = initial_results[2] if len(initial_results) > 2 else {}
        
        # 2. Extract TOP_K_PRF documents from initial TOP_K results
        prf_doc_ids = initial_doc_ids[:min(top_k_prf, len(initial_doc_ids))]
        prf_segments = [segments_dict.get(doc_id, "") for doc_id in prf_doc_ids]
        
        # 3. Extract NUM_EXPANSION_TERMS expansion terms using TF-IDF
        expansion_terms = self._extract_expansion_terms(original_query, prf_segments)
        
        # 4. Expand query with extracted terms
        new_query_text = original_query + " " + " ".join(expansion_terms) if expansion_terms else original_query
        
        # 5. Re-retrieve TOP_K documents using the expanded query
        expanded_results = retriever(new_query_text, top_k)

        # If PRF retrieval fails, keep the original ranking unchanged.
        if not expanded_results or len(expanded_results[0]) == 0:
            new_doc_ids = initial_doc_ids
            new_doc_scores = np.asarray(initial_scores, dtype=np.float32)
        else:
            expanded_doc_ids = expanded_results[0]

            # Fuse the original and PRF-expanded top-K rankings.
            new_doc_ids, new_doc_scores = self._reciprocal_rank_fusion(
                result_lists=[initial_doc_ids, expanded_doc_ids],
                top_k=top_k,
                rrf_k=60,
            )
        
        elapsed_time = time.time() - start_time
        
        return {
            'new_query_text': new_query_text,
            'new_doc_ids': new_doc_ids,
            'new_doc_scores': new_doc_scores.astype(np.float32) if isinstance(new_doc_scores, np.ndarray) else np.array(new_doc_scores, dtype=np.float32),
            'elapsed_time': elapsed_time,
            'cost': 0.3
        }
    
    def _extract_expansion_terms(self, original_query: str, segments: List[str]) -> List[str]:
        """
        Extract discriminative expansion terms from top-k documents using TF-IDF,
        with aggressive cleanup to avoid URL fragments, boilerplate, and junk tokens.
        """
        if not segments or all(not s or not s.strip() for s in segments):
            return []

        boilerplate_stopwords = {
            "http", "https", "www", "com", "org", "net", "html", "htm",
            "amp", "utm", "utm_source", "utm_medium", "utm_campaign",
            "january", "february", "march", "april", "may", "june", "july",
            "august", "september", "october", "november", "december",
            "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
            "oct", "nov", "dec"
        }
        stop_words = ENGLISH_STOP_WORDS.union(boilerplate_stopwords)

        def clean_text(text: str) -> str:
            text = text.lower()

            # Remove URLs
            text = re.sub(r"https?://\S+|www\.\S+", " ", text)

            # Remove emails
            text = re.sub(r"\b\S+@\S+\b", " ", text)

            # Keep letters/spaces only; drop digits, punctuation, underscores, etc.
            text = re.sub(r"[^a-z\s]", " ", text)

            # Collapse repeated whitespace
            text = re.sub(r"\s+", " ", text).strip()
            return text

        cleaned_segments = [clean_text(s) for s in segments if s and s.strip()]
        cleaned_segments = [s for s in cleaned_segments if s]

        if not cleaned_segments:
            return []

        def normalize_query_tokens(text: str) -> set:
            cleaned = clean_text(text)
            return {
                tok for tok in cleaned.split()
                if len(tok) >= 3 and tok not in stop_words
            }

        query_tokens = normalize_query_tokens(original_query)

        vectorizer = TfidfVectorizer(
            stop_words=list(stop_words),
            preprocessor=clean_text,
            token_pattern=r"(?u)\b[a-z]{3,}\b",  # alphabetic tokens only, len >= 3
            lowercase=True,
            max_features=5000,
            ngram_range=(1, 1),
            min_df=1,
            max_df=0.8,  # suppress very common boilerplate across PRF docs
        )

        try:
            tfidf_matrix = vectorizer.fit_transform(cleaned_segments)
        except ValueError:
            return []

        mean_scores = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
        terms = vectorizer.get_feature_names_out()

        ranked = sorted(zip(terms, mean_scores), key=lambda x: x[1], reverse=True)
        base_len = len(query_tokens)
        num_expansion_terms = max(1, min(5, round(0.2 * base_len)))
        expansion_terms = []
        for term, _ in ranked:
            if term in query_tokens:
                continue
            if term in stop_words:
                continue
            if len(term) < 3:
                continue
            if term.isdigit():
                continue
            expansion_terms.append(term)
            if len(expansion_terms) >= num_expansion_terms:
                break

        return expansion_terms
    