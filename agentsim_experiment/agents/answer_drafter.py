"""Answer drafter agent."""

import os
import re
import time
import traceback
from typing import Any, Dict, List, Optional

from .base import AgentInterface

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful reading assistant. Given a question and a set of passages "
    "retrieved by a search engine, generate up to {n_candidates} concise, plausible "
    "answers to the question. Each answer should be a short phrase or sentence that "
    "directly answers the question. Return exactly one answer per line. Do not "
    "enumerate them. If the passages do not contain enough information, generate "
    "likely answers based on the question alone."
)

DEFAULT_USER_TEMPLATE = (
    "Question: {query}\n\n"
    "Retrieved passages:\n{passages}\n\n"
    "Generate up to {n_candidates} plausible answers (one per line):"
)


def _clean_answer(text: str) -> str:
    """Strip markdown and truncate to the first sentence, max 25 words."""
    if not text:
        return ""
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`[^`]*`", "", text)
    text = text.replace("`", "")
    text = text.replace("**", "").replace("__", "")
    text = text.replace("*", "").replace("_", "")
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text).strip()
    m = re.match(r"([^.!?]+[.!?:])\s*", text)
    first_sentence = m.group(1).strip() if m else text
    words = first_sentence.split()
    if len(words) > 25:
        first_sentence = " ".join(words[:25]) + "."
    return first_sentence


def _dedup_key(text: str) -> str:
    """Aggressive normalization for answer deduplication."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class AnswerDrafterAgent(AgentInterface):
    """Draft plausible answer candidates from filtered passages."""

    name = "answer_drafter"

    def __init__(
        self,
        n_candidates: int = 3,
        top_k_passages: int = 10,
        passage_chars: int = 500,
        system_prompt: Optional[str] = None,
        user_template: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 200,
    ):
        self.n_candidates = n_candidates
        self.top_k_passages = top_k_passages
        self.passage_chars = passage_chars
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.user_template = user_template or DEFAULT_USER_TEMPLATE
        self.temperature = temperature
        self.max_tokens = max_tokens

        if OpenAI is None:
            raise RuntimeError("openai package is required for AnswerDrafterAgent")

        self.model_name = model_name or os.getenv("MODEL_NAME_HPC") or os.getenv("MODEL_NAME")
        self.base_url = base_url or os.getenv("BASE_URL_HPC") or os.getenv("OPENAI_BASE_URL")
        self.api_key = api_key or os.getenv("LLMAPI_KEY") or os.getenv("OPENAI_API_KEY")
        self._client = None

    @property
    def client(self):
        if self._client is None:
            if not self.model_name or not self.api_key:
                raise RuntimeError(
                    "AnswerDrafterAgent requires MODEL_NAME_HPC/MODEL_NAME and "
                    "LLMAPI_KEY/OPENAI_API_KEY environment variables or explicit arguments"
                )

            headers = {}
            if self.base_url:
                headers["HTTP-Referer"] = "AgentSim-MSMarco-Eval"
                headers["X-Title"] = "Answer Drafter"

            client_kwargs = {"api_key": self.api_key}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            if headers:
                client_kwargs["default_headers"] = headers

            self._client = OpenAI(**client_kwargs)
        return self._client

    def _build_prompt(self, query: str, passages: List[Dict[str, Any]]) -> str:
        numbered = []
        for rank, passage in enumerate(passages[: self.top_k_passages], start=1):
            text = passage.get("text", "").strip()
            if text:
                numbered.append(f"{rank}.{text[: self.passage_chars]}")
        passage_block = " ".join(numbered)
        return self.user_template.format(
            query=query,
            passages=passage_block,
            n_candidates=self.n_candidates,
        )

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query = state.get("query", "")
        if isinstance(query, dict):
            query = query.get("text", "")
        query = str(query)

        filtered = state.get("filtered", [])
        if not filtered:
            state["drafted_answers"] = []
            return state

        start = time.perf_counter()
        candidates = self._generate_candidates(query, filtered)
        elapsed = time.perf_counter() - start

        state["drafted_answers"] = candidates
        state.setdefault("latency_answer_drafter", []).append(elapsed)

        return state

    def _generate_candidates(
        self, query: str, passages: List[Dict[str, Any]]
    ) -> List[str]:
        system_msg = self.system_prompt.format(n_candidates=self.n_candidates)
        user_msg = self._build_prompt(query, passages)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:
            traceback.print_exc()
            return []

        content = response.choices[0].message.content or ""
        seen = set()
        candidates = []
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            answer = _clean_answer(line)
            if len(answer) < 3:
                continue
            key = _dedup_key(answer)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(answer)
            if len(candidates) >= self.n_candidates:
                break

        return candidates
