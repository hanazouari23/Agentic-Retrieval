from __future__ import annotations

import csv
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.special import softmax
from scipy.stats import entropy as scipy_entropy

from .core.agents import AgentBase

logger = logging.getLogger(__name__)

# ── Action constants ──────────────────────────────────────────────────────────
ACTION_QR   = 0
ACTION_RR   = 1
ACTION_PRF   = 2
ACTION_STOP = 3

ACTION_NAMES: Dict[int, str] = {
    ACTION_QR:   "QueryReform",
    ACTION_RR:   "Rerank",
    ACTION_PRF:   "PseudoRelevanceFeedback",
    ACTION_STOP: "STOP",
}
ACTION_COSTS: Dict[int, float] = {
    ACTION_QR:   1,  # Reformulate cost is based on token count, handled separately
    ACTION_RR:   0.3,
    ACTION_PRF:  0.3,
    ACTION_STOP: 0.0,
}
N_AGENTS = 3  # QR, RR, PRF — excludes STOP


# ── Configuration ─────────────────────────────────────────────────────────────
@dataclass
class SimConfig:
    """Hyper-parameters for the simulation environment."""

    # MDP
    max_steps:    int   = 3
    n_actions:    int   = 4

    # Retrieval windows
    top_k_rerank: int   = 50
    top_k_prf:    int   = 10   # top docs to extract PRF expansion terms from
    ndcg_k:       int   = 50
    recall_k:     int   = 100

    # Reward weights:   r = α·ΔNDCG + β·ΔRecall − γ·cost − δ·step_penalty
    reward_alpha: float = 2.0   # ΔNDCG@k weight
    reward_beta:  float = 0.5   # ΔRecall@k weight
    reward_gamma: float = 0.2   # cost penalty weight
    reward_delta: float = 0.1   # latency penalty 

    # Elapsed-time normalisation divisor (ms). Divide raw ms by this.
    elapsed_time_norm: float = 3000.0

    # Data paths
    qrels_path: Optional[str] = None      # Path to qrels file {doc_id: relevance_grade}
    queries_path: Optional[str] = None    # Path to queries file or list of query strings

    # State dimensions
    query_emb_dim: int  = 384  # all-MiniLM-L6-v2 returns 384-d embeddings
    # ┌─ Layout ──────────────────────────────────────────────────────────────┐
    # │ [0]       query_length     1                                          │
    # │ [1:385]   query_embedding  384                                        │
    # │ [385]     score_spread     1                                          │
    # │ [386]     score_entropy    1                                          │
    # │ [387:390] last_action      3  (one-hot: QR, RR, PRF)                  │
    # │ [390]     step             1  (normalised)                            │
    # │ [391]     rank_overlap     1  (Jaccard with previous top-k)           │
    # │ [392]     query_drift      1  (cosine distance from original query)   │
    # │ [393]     elapsed_time     1  (normalised)                            │
    # │ [394]     cost             1  (cumulative)                            │
    # │ [395:399] valid_actions    4                                          │
    # │ Total = 1+384+1+1+3+1+1+1+1+1+4 = 399                                 │
    # └───────────────────────────────────────────────────────────────────────┘
    state_dim: int = 399


# ── Transition data structure ─────────────────────────────────────────────────
@dataclass
class Transition:
    """
    Single (s, a, r, s′, done) experience tuple for offline RL training.

    Attributes
    ----------
    state      : float32 ndarray (state_dim,)
    action     : int in {0, 1, 2, 3}
    reward     : scalar float
    next_state : float32 ndarray (state_dim,)
    done       : True when episode ends (STOP chosen or max_steps reached)
    info       : auxiliary diagnostics for logging / analysis
    """
    state:      np.ndarray
    action:     int
    reward:     float
    next_state: np.ndarray
    done:       bool
    info:       Dict[str, Any] = field(default_factory=dict)




# ── Main class ────────────────────────────────────────────────────────────────
class Simulation:
    """
    MDP trajectory simulator for multi-agent retrieval offline RL.

    Parameters
    ----------
    encoder       : SentenceTransformer
                    Query encoder (msmarco-distilbert-base-v2 → 768-d).
    retriever     : Callable[[str], List[Tuple[str, float]]]
                    Base retriever: query → [(doc_id, score), …].
    agents        : List[AgentBase]
                    The three agents: [ReformulationAgent, RerankingAgent]
   
    config        : SimConfig  (defaults applied if None)
    """

    def __init__(
        self,
        encoder,
        retriever:    Callable[[str], List[Tuple[str, float]]],
        agents:       List[AgentBase],
        config:       Optional[SimConfig] = None,
    ) -> None:
        self.encoder       = encoder
        self.retriever     = retriever
        self.agents        = agents  # [qr_agent, rr_agent, cp_agent]
        self.cfg           = config or SimConfig()

    # ── Metrics (MDP-level evaluation) ────────────────────────────────────────
    @staticmethod
    def normalize_doc_id(doc_id: str) -> str:
        return doc_id.split('#', 1)[0] if '#' in doc_id else doc_id

    @staticmethod
    def deduplicate_doc_ids(doc_ids: List[str]) -> List[str]:
        seen = set()
        deduped = []
        for doc_id in doc_ids:
            normalized = Simulation.normalize_doc_id(doc_id)
            if normalized not in seen:
                deduped.append(normalized)
                seen.add(normalized)
        return deduped
    @staticmethod
    def _dcg(gains: np.ndarray, k: int) -> float:
        """Discounted cumulative gain at rank k."""
        r = np.asarray(gains[:k], dtype=float)
        if r.size == 0:
            return 0.0
        positions = np.arange(2, r.size + 2)
        return float(np.sum(r / np.log2(positions)))
    
    @staticmethod
    def compute_ndcg(
        ranked_doc_ids: List[str],
        qrels: Dict[str, int],
        k: int = 50,
    ) -> float:
        ranked_docs = Simulation.deduplicate_doc_ids(ranked_doc_ids)[:k]
        gains = [qrels.get(doc_id, 0) for doc_id in ranked_docs]
        ideal = sorted((rel for rel in qrels.values() if rel > 0), reverse=True)[:k]
        dcg = Simulation._dcg(np.array(gains, dtype=float), k)
        idcg = Simulation._dcg(np.array(ideal, dtype=float), k)
        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def compute_recall(
        ranked_doc_ids: List[str],
        qrels: Dict[str, int],
        k: int = 50,
    ) -> float:
        ranked_docs = Simulation.deduplicate_doc_ids(ranked_doc_ids)[:k]
        relevant = {d for d, r in qrels.items() if r > 0}
        if not relevant:
            return 0.0
        return len(set(ranked_docs) & relevant) / len(relevant)
 
    @staticmethod
    def _valid_actions_mask(last_action_agent: List[bool]) -> List[int]:
        """
        Compute which actions are currently available.

        The mask is qrel-agnostic: it only prevents the same agent from being
        used twice in a row. STOP is always available.

        Parameters
        ----------
        last_action_agent : List[bool]
            One-hot vector [last_was_qr, last_was_rr, last_was_prf].

        Returns
        -------
        List[int]  – binary mask of length 4: [qr, rr, prf, stop]
        """
        return [
            int(not last_action_agent[0]),  # QR valid if not last QR
            int(not last_action_agent[1]),  # RR valid if not last RR
            int(not last_action_agent[2]),  # PRF valid if not last PRF
            1,                              # STOP always available
        ]

    @staticmethod
    def _score_spread(scores: np.ndarray) -> float:
        """Compute standard deviation of scores."""
        return float(np.std(scores)) if len(scores) > 1 else 0.0

    @staticmethod
    def _score_entropy(scores: np.ndarray) -> float:
        """Compute entropy of softmax-normalized scores."""
        probs = softmax(scores.astype(float))
        return float(scipy_entropy(probs))

    @staticmethod
    def _rank_overlap(
        previous_docids: Optional[List[str]],
        current_docids: List[str],
        k: int = 50,
    ) -> np.float32:
        """
        Jaccard overlap of the previous and current top-k result sets.

        Returns 0.0 at the first state because no previous ranking exists.
        """
        if previous_docids is None:
            return np.float32(0.0)

        previous = {
            Simulation.normalize_doc_id(docid)
            for docid in previous_docids[:k]
        }
        current = {
            Simulation.normalize_doc_id(docid)
            for docid in current_docids[:k]
        }

        union = previous | current
        if not union:
            return np.float32(0.0)

        return np.float32(len(previous & current) / len(union))


    @staticmethod
    def _query_drift(
        original_query_embedding: np.ndarray,
        current_query_embedding: np.ndarray,
    ) -> np.float32:
        """
        Cosine distance from the original query embedding.

        0.0 means the current query has not changed semantically.
        Larger values indicate greater semantic drift.
        """
        original = np.asarray(original_query_embedding, dtype=np.float32)
        current = np.asarray(current_query_embedding, dtype=np.float32)

        denominator = np.linalg.norm(original) * np.linalg.norm(current)
        if denominator <= 1e-12:
            return np.float32(0.0)

        cosine_similarity = np.dot(original, current) / denominator
        cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)

        return np.float32(1.0 - cosine_similarity)
    # ── 1. build_state ────────────────────────────────────────────────────────
    def build_state(
        self,
        query: str,
        docids: List[str],
        docscores: np.ndarray,
        step: int,
        last_action_agent: List[bool],
        previous_docids: Optional[List[str]],
        original_query_embedding: np.ndarray,
        elapsed_ms: float = 0.0,
        cumulative_cost: float = 0.0,
        query_length: Optional[np.ndarray] = None,
        query_embedding: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Build the 399-dimensional observable policy state.

        State layout:
            0       : current query word count
            1:385   : current query embedding (384 dimensions)
            385     : score spread
            386     : score entropy
            387:390 : last action one-hot [QR, RR, PRF]
            390     : normalized step
            391     : top-k rank overlap with previous ranking
            392     : semantic query drift from original query
            393     : cumulative elapsed time, normalized
            394     : cumulative action cost
            395:399 : operational action mask [QR, RR, PRF, STOP]

        No qrels, NDCG, Recall, reward, or oracle-quality features are used.
        """
        cfg = self.cfg

        if query_length is None:
            query_length = np.float32(len(query.split()))
        else:
            query_length = np.float32(query_length)

        if query_embedding is None:
            query_embedding = self.encoder.encode(
                query,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype(np.float32)
        else:
            query_embedding = np.asarray(query_embedding, dtype=np.float32)

        if query_embedding.shape != (cfg.query_emb_dim,):
            raise ValueError(
                f"Expected query embedding shape ({cfg.query_emb_dim},), "
                f"got {query_embedding.shape}"
            )

        score_spread = np.float32(self._score_spread(docscores))
        score_entropy = np.float32(self._score_entropy(docscores))

        last_action_vector = np.asarray(
            last_action_agent,
            dtype=np.float32,
        )
        if last_action_vector.shape != (3,):
            raise ValueError(
                "last_action_agent must be [last_qr, last_rr, last_prf]"
            )

        normalized_step = np.float32(step / max(cfg.max_steps, 1))

        rank_overlap = self._rank_overlap(
            previous_docids=previous_docids,
            current_docids=docids,
            k=cfg.top_k_rerank,
        )

        query_drift = self._query_drift(
            original_query_embedding=original_query_embedding,
            current_query_embedding=query_embedding,
        )

        normalized_elapsed = np.float32(
            elapsed_ms / max(cfg.elapsed_time_norm, 1e-12)
        )
        cumulative_cost = np.float32(cumulative_cost)

        valid_action_mask = np.asarray(
            self._valid_actions_mask(last_action_agent),
            dtype=np.float32,
        )

        state = np.concatenate([
            np.asarray([query_length], dtype=np.float32),     # 1
            query_embedding,                                  # 384
            np.asarray([score_spread], dtype=np.float32),     # 1
            np.asarray([score_entropy], dtype=np.float32),    # 1
            last_action_vector,                               # 3
            np.asarray([normalized_step], dtype=np.float32),  # 1
            np.asarray([rank_overlap], dtype=np.float32),     # 1
            np.asarray([query_drift], dtype=np.float32),      # 1
            np.asarray([normalized_elapsed], dtype=np.float32), # 1
            np.asarray([cumulative_cost], dtype=np.float32),  # 1
            valid_action_mask,                                # 4
        ]).astype(np.float32)

        assert state.shape == (cfg.state_dim,), (
            f"State dimension mismatch: expected {cfg.state_dim}, "
            f"got {state.shape}"
        )

        return state

    # ── 2. compute_effects ────────────────────────────────────────────────────
    def compute_effects(
        self,
        action:     int,
        query:      str,
        doc_ids:    List[str],
        doc_scores: np.ndarray,
        qrels:      Dict[str, int],
        corpus_data: Optional[Dict[str, str]] = None,
    ) -> Tuple[str, List[str], np.ndarray, Dict[str, float], float, float]:
        """
        Apply the chosen agent action; return updated retrieval state + metrics.

        Parameters
        ----------
        action     : ACTION_QR / ACTION_RR / ACTION_PRF / ACTION_STOP
        query      : current query string
        doc_ids    : current ranked document IDs
        doc_scores : retrieval scores aligned with doc_ids
        qrels      : {doc_id: relevance_grade}

        Returns
        -------
        new_query      : (possibly reformulated) query string
        new_doc_ids    : updated ranked document list
        new_doc_scores : updated scores
        metrics        : {"ndcg": float, "recall": float}
        elapsed_ms     : wall-clock time taken by this agent call (ms)
        """
        t0 = time.perf_counter()

        if action == ACTION_STOP:
            metrics = {
                "ndcg":         Simulation.compute_ndcg(doc_ids, qrels, self.cfg.ndcg_k),
                "recall":       Simulation.compute_recall(doc_ids, qrels, self.cfg.recall_k),
            }
            elapsed_ms = (time.perf_counter() - t0) * 1_000.0
            return query, list(doc_ids), doc_scores.copy(), metrics, elapsed_ms, 0.0

        # Get the appropriate agent
        if action not in [ACTION_QR, ACTION_RR, ACTION_PRF]:
            raise ValueError(f"Unknown action id: {action}")
        
        agent = self.agents[action]  # ACTION_QR=0, ACTION_RR=1, ACTION_PRF=2
    
        # Prepare query features for the agent
        query_features = {
            'query_text': query,
            'embedding': self.encoder.encode(query, convert_to_numpy=True, show_progress_bar=False),
            'doc_ids': doc_ids,
            'doc_scores': doc_scores,
            'retriever': self.retriever,
            'corpus': corpus_data or {},  # Document text mapping for RerankingAgent
            'top_k': self.cfg.top_k_rerank,           # ReformulationAgent expects 'top_k'
            'top_k_rerank': self.cfg.top_k_rerank,   # RerankingAgent expects 'top_k_rerank'
            'top_k_prf': self.cfg.top_k_prf,         # PRFAgent expects 'top_k_prf'
        }
        
        # Call agent
        effects = agent.compute_effects(query_features)
        
        # Extract results
        new_query = effects.get('new_query_text', query)
        new_doc_ids = effects.get('new_doc_ids', doc_ids)
        new_doc_scores = effects.get('new_doc_scores', doc_scores)
        elapsed_ms = effects.get('elapsed_time', 0.0) * 1000.0  # Convert to ms
        cost = effects.get('cost')  # Agent-calculated cost
        
        # Compute metrics on the new results
        metrics = {
            "ndcg":         Simulation.compute_ndcg(new_doc_ids, qrels, self.cfg.ndcg_k),
            "recall":       Simulation.compute_recall(new_doc_ids, qrels, self.cfg.recall_k),
        }
        
        return new_query, new_doc_ids, new_doc_scores, metrics, elapsed_ms, cost

    # ── 3. Reward ─────────────────────────────────────────────────────────────
    def _compute_reward(
        self,
        ndcg_before:   float,
        ndcg_after:    float,
        recall_before: float,
        recall_after:  float,
        action:        int,
        elapsed_ms:    float,
        action_cost:   float = 0.0,
    ) -> float:
        """
        Multi-objective reward:

            r = α·ΔNDCG + β·ΔRecall − γ·cost(a) − δ·step_penalty

        where:
            Δx           = x_after − x_before  (change due to this action)
            cost(a)      = cost returned by agent (e.g., token count for QueryReform)
        """
        if action == ACTION_STOP:
            return float(0.0)

        cfg = self.cfg
        ndcg_gain = ndcg_after - ndcg_before
        recall_gain = recall_after - recall_before

        # Agents may return None as cost; treat it as zero.
        action_cost = float(action_cost) if action_cost is not None else 0.0

        quality_reward = cfg.reward_alpha * ndcg_gain
        recall_reward = cfg.reward_beta * recall_gain
        cost_penalty = cfg.reward_gamma * action_cost
        time_penalty = cfg.reward_delta * (
            elapsed_ms / max(cfg.elapsed_time_norm, 1e-12)
        )

        total = quality_reward + recall_reward - cost_penalty - time_penalty

        logger.debug(
            "reward action=%s: ndcg_gain=%+.6f, recall_gain=%+.6f, "
            "quality=%+.6f, recall_term=%+.6f, cost=-%.6f, time=-%.6f, total=%+.6f",
            ACTION_NAMES[action],
            ndcg_gain,
            recall_gain,
            quality_reward,
            recall_reward,
            cost_penalty,
            time_penalty,
            total,
        )

        return float(total)
        

    # ── 4. generate_trajectory ────────────────────────────────────────────────
    def generate_trajectory(
        self,
        query: str,
        doc_ids: List[str],
        doc_scores: np.ndarray,
        qrels: Dict[str, int],
        policy: str = "random",
        corpus_data: Optional[Dict[str, str]] = None,
        forced_actions: Optional[List[int]] = None,
    ) -> List[Transition]:
        """
        Generate one trajectory.

        forced_actions, if provided, overrides the policy for the first N
        steps. After the forced prefix the trajectory continues with the
        selected policy. This is used to generate tree-style datasets where
        every valid first action is explored from the same initial state.
    
        Qrels are used only to:
        - calculate transition rewards,
        - log evaluation metrics,
        - support the qrel-aware expert/oracle behavior policy.
    
        They are not included in the policy observation/state.
        """
        cfg = self.cfg
        trajectory: List[Transition] = []
    
        if corpus_data is None:
            raise ValueError(
                "corpus_data must be provided with actual document text for agents"
            )
    
        cur_corpus = corpus_data.copy()
        cur_query = query
        cur_ids = list(doc_ids)
        cur_scores = np.asarray(doc_scores, dtype=np.float32)
    
        # This is a last-action one-hot vector, not cumulative action history:
        # [last_was_qr, last_was_rr, last_was_prf].
        # It enforces: QR->QR, RR->RR, and PRF->PRF are not permitted.
        last_action_agent = [False, False, False]
    
        # The initial ranking has no predecessor. This gives rank_overlap = 0.0
        # in the initial state.
        previous_docids: Optional[List[str]] = None
    
        original_query_emb = self.encoder.encode(
            query,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
    
        cur_query_emb = original_query_emb.copy()
        cur_query_length = np.float32(len(cur_query.split()))
    
        cum_elapsed_ms = 0.0
        cum_cost = 0.0
    
        # For the expert policy, cache effects evaluated during two-step planning.
        # This prevents QR from calling the LLM again after planning selected it.
        cached_plan: List[int] = []
        cached_effects: List[Tuple] = []
    
        for step in range(cfg.max_steps):
            # State contains only observable retrieval, query, history, cost,
            # latency, and operational eligibility features.
            state = self.build_state(
                query=cur_query,
                docids=cur_ids,
                docscores=cur_scores,
                step=step,
                last_action_agent=last_action_agent,
                previous_docids=previous_docids,
                original_query_embedding=original_query_emb,
                elapsed_ms=cum_elapsed_ms,
                cumulative_cost=cum_cost,
                query_length=cur_query_length,
                query_embedding=cur_query_emb,
            )
    
            # Qrels remain available to the simulator for reward calculation.
            # They must not be passed to build_state or _valid_actions_mask.
            ndcg_before = Simulation.compute_ndcg(
                cur_ids,
                qrels,
                cfg.ndcg_k,
            )
            recall_before = Simulation.compute_recall(
                cur_ids,
                qrels,
                cfg.recall_k,
            )
    
            # Operational mask only: no immediate repeat of the previous agent.
            valid = self._valid_actions_mask(last_action_agent)
    
            if forced_actions and step < len(forced_actions):
                # Override the policy for the first N steps. This lets us
                # branch a single initial state into multiple trajectories.
                action = forced_actions[step]
                if not valid[action]:
                    raise ValueError(
                        f"forced action {action} ({ACTION_NAMES.get(action, '?')}) "
                        f"is not valid at step {step}; valid mask={valid}"
                    )

                (
                    new_query,
                    new_ids,
                    new_scores,
                    metrics,
                    elapsed_ms,
                    action_cost,
                ) = self.compute_effects(
                    action,
                    cur_query,
                    cur_ids,
                    cur_scores,
                    qrels,
                    corpus_data=cur_corpus,
                )

            elif policy == "expert":
                # A cached second planning action was evaluated during the
                # preceding two-step look-ahead, so do not execute its agent again.
                if cached_plan:
                    action = cached_plan.pop(0)
                    (
                        new_query,
                        new_ids,
                        new_scores,
                        metrics,
                        elapsed_ms,
                        action_cost,
                    ) = cached_effects.pop(0)
    
                else:
                    remaining_steps = cfg.max_steps - step
                    max_plan_steps = min(2, remaining_steps)
    
                    # _policy_expert_two_step recomputes the eligibility mask
                    # from last_action_agent and propagates it through the
                    # look-ahead so plans like QR -> QR are never considered.
                    plan, effects = self._policy_expert_two_step(
                        cur_query,
                        cur_ids,
                        cur_scores,
                        qrels,
                        last_action_agent,
                        corpus_data=cur_corpus,
                        max_plan_steps=max_plan_steps,
                    )
    
                    if not plan or not effects:
                        raise RuntimeError(
                            "Expert planner returned an empty plan/effects list"
                        )
    
                    cached_plan = list(plan)
                    cached_effects = list(effects)
    
                    action = cached_plan.pop(0)
                    (
                        new_query,
                        new_ids,
                        new_scores,
                        metrics,
                        elapsed_ms,
                        action_cost,
                    ) = cached_effects.pop(0)
    
            else:
                action = self._select_action(
                    policy,
                    valid,
                    cur_query,
                    cur_ids,
                    cur_scores,
                    qrels,
                    last_action_agent,
                )
    
                (
                    new_query,
                    new_ids,
                    new_scores,
                    metrics,
                    elapsed_ms,
                    action_cost,
                ) = self.compute_effects(
                    action,
                    cur_query,
                    cur_ids,
                    cur_scores,
                    qrels,
                    corpus_data=cur_corpus,
                )
    
            done = (
                action == ACTION_STOP
                or step == cfg.max_steps - 1
            )
    
            ndcg_after = metrics["ndcg"]
            recall_after = metrics["recall"]
    
            reward = self._compute_reward(
                ndcg_before,
                ndcg_after,
                recall_before,
                recall_after,
                action,
                elapsed_ms,
                action_cost,
            )
    
            cum_elapsed_ms += elapsed_ms
            cum_cost += action_cost
    
            # Preserve the old ranking before replacing it. build_state uses this
            # list and new_ids to calculate rank overlap in the next observation.
            next_previous_docids = list(cur_ids)
    
            # Update last-action one-hot state. This is used both in the next
            # observation and in the operational no-immediate-repeat mask.
            if action == ACTION_QR:
                next_last_action_agent = [True, False, False]
            elif action == ACTION_RR:
                next_last_action_agent = [False, True, False]
            elif action == ACTION_PRF:
                next_last_action_agent = [False, False, True]
            else:
                # STOP terminates the episode, but keeping a zero vector makes
                # the terminal next_state well-defined.
                next_last_action_agent = [False, False, False]
    
            new_query_length = np.float32(len(new_query.split()))
            new_query_emb = self.encoder.encode(
                new_query,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype(np.float32)
    
            next_state = self.build_state(
                query=new_query,
                docids=new_ids,
                docscores=np.asarray(new_scores, dtype=np.float32),
                step=step + 1,
                last_action_agent=next_last_action_agent,
                previous_docids=next_previous_docids,
                original_query_embedding=original_query_emb,
                elapsed_ms=cum_elapsed_ms,
                cumulative_cost=cum_cost,
                query_length=new_query_length,
                query_embedding=new_query_emb,
            )
    
            trajectory.append(
                Transition(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                    info={
                        "query": cur_query,
                        "new_query": new_query,
                        "step": step,
                        "action_name": ACTION_NAMES[action],
                        "cost": action_cost,
                        "cum_cost": cum_cost,
                        "elapsed_ms": elapsed_ms,
                        "cum_elapsed_ms": cum_elapsed_ms,
    
                        # Keep qrel-based metrics for diagnosis/evaluation only.
                        # They are intentionally absent from `state` and
                        # `next_state`.
                        "ndcg_before": ndcg_before,
                        "ndcg_after": ndcg_after,
                        "recall_before": recall_before,
                        "recall_after": recall_after,
    
                        "valid_actions": list(valid),
                        "last_action_agent": list(next_last_action_agent),
                        "rank_overlap": float(
                            self._rank_overlap(
                                previous_docids=next_previous_docids,
                                current_docids=new_ids,
                                k=cfg.top_k_rerank,
                            )
                        ),
                        "query_drift": float(
                            self._query_drift(
                                original_query_embedding=original_query_emb,
                                current_query_embedding=new_query_emb,
                            )
                        ),
                    },
                )
            )
    
            cur_query = new_query
            cur_ids = list(new_ids)
            cur_scores = np.asarray(new_scores, dtype=np.float32)
            cur_query_length = new_query_length
            cur_query_emb = new_query_emb
            previous_docids = next_previous_docids
            last_action_agent = next_last_action_agent
    
            if done:
                break
            
        return trajectory

    def generate_expert_branches(
        self,
        query: str,
        doc_ids: List[str],
        doc_scores: np.ndarray,
        qrels: Dict[str, int],
        corpus_data: Optional[Dict[str, str]] = None,
    ) -> List[Optional[List[Transition]]]:
        """
        Generate one trajectory per valid first action from the same state.

        Each branch starts with a different action (QR, RR, PRF, or STOP) and
        then continues with the two-step expert policy. This harvests the
        oracle's exploration of the initial action space while keeping every
        trajectory high-return because the expert recovers afterwards.

        Returns a list indexed by action id; entries are None for actions that
        are invalid or failed.
        """
        valid = self._valid_actions_mask([False, False, False])
        trajectories: List[Optional[List[Transition]]] = [None] * self.cfg.n_actions

        for action in range(self.cfg.n_actions):
            if not valid[action]:
                continue

            try:
                traj = self.generate_trajectory(
                    query=query,
                    doc_ids=doc_ids,
                    doc_scores=doc_scores,
                    qrels=qrels,
                    policy="expert",
                    corpus_data=corpus_data,
                    forced_actions=[action],
                )
                trajectories[action] = traj
            except Exception as exc:
                logger.warning(
                    "generate_expert_branches: action %s failed for query %r: %s",
                    ACTION_NAMES.get(action, action),
                    query,
                    exc,
                )

        return trajectories

    # ── 5. Action-selection policies ─────────────────────────────────────────
    def _select_action(
            self,
            policy:     str,
            valid:      List[int],
            query:      str,
            doc_ids:    List[str],
            doc_scores: np.ndarray,
            qrels:      Dict[str, int],
            agents_used: List[bool]
        ) -> int:
            if policy == "random":
                return self._policy_random(valid)
            if policy == "expert":
                return self._policy_expert(query, doc_ids, doc_scores, qrels, valid)
            if policy == "stop":
                return ACTION_STOP
            if policy == "prf":
                return ACTION_PRF
            if policy == "rerank":
                # First decision: run the reranker.
                if not agents_used[1]:
                    return ACTION_RR
                # Second decision: explicitly end the episode.
                return ACTION_STOP
            raise ValueError(f"Unknown policy: {policy!r} valid options are: random, expert, stop, prf, rerank")
            

    def _policy_random(self, valid: List[int]) -> int:
            """
            Uniformly sample a valid action.

            Parameters
            ----------
            valid : List[int]
                Binary mask of valid actions [qr_valid, rr_valid, prf_valid, stop_valid]

            Returns
            -------
            int
                Action index (0-3)
            """
            valid_indices = [i for i, v in enumerate(valid) if v == 1]
            return random.choice(valid_indices)

    def _policy_expert(
        self,
        query:      str,
        doc_ids:    List[str],
        doc_scores: np.ndarray,
        qrels:      Dict[str, int],
        valid:      List[int],
    ) -> int:
        """
        Greedy oracle policy: try each valid action and pick the one with best ΔNDCG.

        If multiple actions tie for best ΔNDCG, prefer STOP (to terminate early and save cost).

        Parameters
        ----------
        query      : query string
        doc_ids    : current ranked doc IDs
        doc_scores : retrieval scores
        qrels      : relevance judgments {doc_id: grade}
        valid      : binary mask of valid actions

        Returns
        -------
        int
            Action with best ΔNDCG improvement (or STOP if tied)
        """
        cfg = self.cfg

        # Current NDCG baseline
        ndcg_current = Simulation.compute_ndcg(doc_ids, qrels, cfg.ndcg_k)

        # Try each action and compute ΔNDCG
        action_deltas = {}
        effects = {}
        for action in range(N_AGENTS):  # QR, RR, PRF (excluding STOP)
            if not valid[action]:
                continue  # Skip invalid actions

            try:
                new_query, new_ids, new_scores, metrics, elapsed_ms, cost = \
                    self.compute_effects(action, query, doc_ids, doc_scores, qrels)
                ndcg_new = metrics["ndcg"]
                action_deltas[action] = ndcg_new - ndcg_current
                effects[action] = (new_query, new_ids, new_scores, metrics, elapsed_ms, cost)
            except Exception as e:
                # If an agent fails, penalize that action heavily
                logger.warning(f"Expert policy: action {ACTION_NAMES[action]} failed: {e}")
                action_deltas[action] = float('-inf')

        # STOP is always an option (ΔNDCG = 0, no change)
        if valid[ACTION_STOP]:
            action_deltas[ACTION_STOP] = 0.0
            effects[ACTION_STOP] = (
                query,
                doc_ids,
                doc_scores,
                {
                    "ndcg": ndcg_current,
                    "recall": Simulation.compute_recall(doc_ids, qrels, cfg.recall_k),
                },
                0.0,
                0.0,
            )

        # Pick action with best ΔNDCG; if tied, prefer STOP
        if not action_deltas:
            return ACTION_STOP  # Fallback: no valid actions

        best_delta = max(action_deltas.values())
        best_actions = [a for a, d in action_deltas.items() if d == best_delta]

        # Prefer STOP if it's tied for best
        if ACTION_STOP in best_actions:
            best_action = ACTION_STOP
        else:
            best_action = best_actions[0]

        return best_action  # Pick the first best action (deterministic)


    def _policy_expert_two_step(
        self,
        query: str,
        doc_ids: List[str],
        doc_scores: np.ndarray,
        qrels: Dict[str, int],
        last_action_agent: List[bool],
        corpus_data: Optional[Dict[str, str]] = None,
        max_plan_steps: int = 2,
    ) -> Tuple[List[int], List[Tuple]]:
        cfg = self.cfg
        eps = 1e-12

        valid = self._valid_actions_mask(last_action_agent)

        ndcg_current = Simulation.compute_ndcg(
            doc_ids,
            qrels,
            cfg.ndcg_k,
        )
        recall_current = Simulation.compute_recall(
            doc_ids,
            qrels,
            cfg.recall_k,
        )

        stop_effect = (
            query,
            list(doc_ids),
            np.asarray(doc_scores, dtype=np.float32).copy(),
            {
                "ndcg": ndcg_current,
                "recall": recall_current,
            },
            0.0,
            0.0,
        )

        best_plan = [ACTION_STOP]
        best_effects = [stop_effect]
        best_value = ndcg_current

        for action1 in range(N_AGENTS):
            if not valid[action1]:
                continue

            try:
                query1, ids1, scores1, met1, time1, cost1 = (
                    self.compute_effects(
                        action1,
                        query,
                        doc_ids,
                        doc_scores,
                        qrels,
                        corpus_data=corpus_data,
                    )
                )
            except Exception as exc:
                logger.warning(
                    "Expert two-step: first-step %s failed: %s",
                    ACTION_NAMES[action1],
                    exc,
                )
                continue

            step1_effect = (
                query1,
                ids1,
                scores1,
                met1,
                time1,
                cost1,
            )
            step1_value = met1["ndcg"]

            if step1_value > best_value + eps:
                best_value = step1_value
                best_plan = [action1]
                best_effects = [step1_effect]

            # No available rollout step for a second action.
            if max_plan_steps < 2:
                continue

            # The same agent cannot be used twice in a row. Build the
            # last-action vector that results from action1 and recompute
            # the eligibility mask for the second step.
            next_last_action_agent1 = [False, False, False]
            if action1 == ACTION_QR:
                next_last_action_agent1 = [True, False, False]
            elif action1 == ACTION_RR:
                next_last_action_agent1 = [False, True, False]
            elif action1 == ACTION_PRF:
                next_last_action_agent1 = [False, False, True]

            valid2 = self._valid_actions_mask(next_last_action_agent1)

            for action2 in range(N_AGENTS):
                if not valid2[action2]:
                    continue

                try:
                    query2, ids2, scores2, met2, time2, cost2 = (
                        self.compute_effects(
                            action2,
                            query1,
                            ids1,
                            scores1,
                            qrels,
                            corpus_data=corpus_data,
                        )
                    )
                except Exception as exc:
                    logger.warning(
                        "Expert two-step: %s after %s failed: %s",
                        ACTION_NAMES[action2],
                        ACTION_NAMES[action1],
                        exc,
                    )
                    continue

                step2_effect = (
                    query2,
                    ids2,
                    scores2,
                    met2,
                    time2,
                    cost2,
                )
                step2_value = met2["ndcg"]

                if step2_value > best_value + eps:
                    best_value = step2_value
                    best_plan = [action1, action2]
                    best_effects = [step1_effect, step2_effect]

        logger.debug(
            "Expert selected plan=%s; start_ndcg=%.6f; final_ndcg=%.6f",
            [ACTION_NAMES[action] for action in best_plan],
            ndcg_current,
            best_value,
        )

        return best_plan, best_effects


    # ── 7. Serialisation ──────────────────────────────────────────────────────
    @staticmethod
    def to_arrays(
        dataset: List[Transition],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Stack a list of Transition objects into numpy arrays for training.

        Returns
        -------
        states      : float32  (N, state_dim)
        actions     : int64    (N,)
        rewards     : float32  (N,)
        next_states : float32  (N, state_dim)
        dones       : bool     (N,)
        """
        states      = np.stack([t.state      for t in dataset]).astype(np.float32)
        actions     = np.array([t.action     for t in dataset], dtype=np.int64)
        rewards     = np.array([t.reward     for t in dataset], dtype=np.float32)
        next_states = np.stack([t.next_state for t in dataset]).astype(np.float32)
        dones       = np.array([t.done       for t in dataset], dtype=bool)
        return states, actions, rewards, next_states, dones

    @staticmethod
    def export_trajectories_to_csv(
        trajectories: List[List[Transition]],
        filename: str = "trajectories.csv",
    ) -> Path:
        """
        Export trajectories to a CSV file in the trajectories folder.
        Includes classic offline RL dataset fields: state, action, reward, next_state, done.

        Parameters
        ----------
        trajectories : List[List[Transition]]
            List of trajectories, where each trajectory is a list of Transition objects.
        filename : str
            Name of the CSV file to create (default: "trajectories.csv")

        Returns
        -------
        Path
            Absolute path to the created CSV file
        """
        # Determine trajectories folder (relative to this file)
        sim_file = Path(__file__).resolve()
        traj_folder = sim_file.parent.parent / "trajectories"
        traj_folder.mkdir(exist_ok=True)
        
        csv_path = traj_folder / filename
        
        # Flatten all transitions across all trajectories
        all_transitions = []
        for traj_idx, traj in enumerate(trajectories):
            for step_idx, trans in enumerate(traj):
                all_transitions.append((traj_idx, step_idx, trans))
        
        # Write to CSV
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                # Trajectory tracking
                'trajectory_id',
                'step',
                # Query information
                'query',
                'new_query',
                # Action tracking
                'action',
                'action_name',
                # Classic offline RL fields
                #'state',
                'reward',
                #'next_state',
                'done',
                # Metrics and diagnostics
                'ndcg_before',
                'ndcg_after',
                'recall_before',
                'recall_after',
                'cost',
                'cum_cost',
                'elapsed_ms',
                'cum_elapsed_ms',
            ])
            writer.writeheader()
            
            for traj_id, step, trans in all_transitions:
                info = trans.info
                # Serialize state vectors as comma-separated values (within the CSV cell)
                state_str = ','.join([f"{x:.6f}" for x in trans.state])
                next_state_str = ','.join([f"{x:.6f}" for x in trans.next_state])
                
                writer.writerow({
                    # Trajectory tracking
                    'trajectory_id': traj_id,
                    'step': step,
                    # Query information
                    'query': info.get('query', ''),
                    'new_query': info.get('new_query', ''),
                    # Action tracking
                    'action': int(trans.action),
                    'action_name': info.get('action_name', ''),
                    # Classic offline RL fields
                    #'state': state_str,
                    'reward': f"{trans.reward:.6f}",
                    #'next_state': next_state_str,
                    'done': trans.done,
                    # Metrics and diagnostics
                    'ndcg_before': f"{info.get('ndcg_before', 0.0):.6f}",
                    'ndcg_after': f"{info.get('ndcg_after', 0.0):.6f}",
                    'recall_before': f"{info.get('recall_before', 0.0):.6f}",
                    'recall_after': f"{info.get('recall_after', 0.0):.6f}",
                    'cost': f"{info.get('cost', 0.0):.6f}",
                    'cum_cost': f"{info.get('cum_cost', 0.0):.6f}",
                    'elapsed_ms': f"{info.get('elapsed_ms', 0.0):.2f}",
                    'cum_elapsed_ms': f"{info.get('cum_elapsed_ms', 0.0):.2f}",
                })
        
        logger.info(f"Trajectories exported to {csv_path}")
        return csv_path