"""Query-only Boolean concept-formulation agent.

This agent reformulates a natural-language query into an explicit Boolean
query using one LLM call.  The LLM only sees the query text; no retrieved
documents are used for formulation.  Generation, validation and rendering are
kept separately testable.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from ..core.agents import AgentBase

logger = logging.getLogger(__name__)


# ── Configuration defaults ───────────────────────────────────────────────────
def _load_env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


API_KEY = _load_env("LLMAPI_KEY")
BASE_URL = _load_env("BASE_URL_HPC")
MODEL_NAME = _load_env("MODEL_NAME_HPC")


# ── Validation / rendering limits ────────────────────────────────────────────
MAX_CONCEPT_GROUPS = 8
MAX_ALTERNATIVES_PER_GROUP = 8
MAX_ALTERNATIVE_LENGTH = 80


# ── Prompts ──────────────────────────────────────────────────────────────────
BOOLEAN_SYSTEM_PROMPT = (
    "You are a query concept extractor for Boolean retrieval. "
    "Given a user query, identify the essential concepts required to "
    "satisfy the information need. Return a single JSON object with one "
    "key 'required_concepts'. Its value is a list of concept groups. "
    "Each concept group is an object with one key 'alternatives', whose "
    "value is a short list of synonymous or closely related ways to "
    "express that required concept. "
    f"Use at most {MAX_CONCEPT_GROUPS} concept groups and at most "
    f"{MAX_ALTERNATIVES_PER_GROUP} alternatives per group. "
    "Use only the query text. Preserve the information need. "
    "Do not include question words (who, what, where, when, why, how) or "
    "optional context as required concepts. Do not invent constraints, "
    "exclusions, or concepts not grounded in the query. "
    "Keep alternatives conservative and concise. "
    "Do not include Markdown formatting or explanation; return raw JSON only."
)

BOOLEAN_USER_PROMPT_TEMPLATE = (
    "Extract required concepts for Boolean retrieval from this query:\n\n{}"
)


class BooleanFormulationError(Exception):
    """Raised when Boolean formulation cannot produce a valid query."""


def _make_client() -> OpenAI:
    if not MODEL_NAME:
        raise RuntimeError("Missing environment variable: MODEL_NAME_HPC")
    return OpenAI(
        base_url=BASE_URL or None,
        api_key=API_KEY or None,
        default_headers={
            "HTTP-Referer": "MAESTRO-Boolean-Formulator",
            "X-Title": "Boolean Query Formulator",
        },
    )


def _strip_code_fences(text: str) -> str:
    """Remove optional Markdown JSON fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _is_valid_alternative(alt: Any) -> bool:
    return isinstance(alt, str) and alt.strip() != ""


def validate_concepts(
    concepts: Any,
    max_groups: int = MAX_CONCEPT_GROUPS,
    max_alternatives: int = MAX_ALTERNATIVES_PER_GROUP,
) -> Dict[str, List[Dict[str, List[str]]]]:
    """Validate and normalise the LLM concept structure.

    Returns a normalised dict of the form::

        {"required_concepts": [{"alternatives": [str, ...]}, ...]}

    Raises BooleanFormulationError on any structural problem.
    """
    if not isinstance(concepts, dict):
        raise BooleanFormulationError(
            f"Concepts must be a dict, got {type(concepts).__name__}"
        )

    groups = concepts.get("required_concepts")
    if not isinstance(groups, list) or not groups:
        raise BooleanFormulationError("'required_concepts' must be a non-empty list")

    if len(groups) > max_groups:
        raise BooleanFormulationError(
            f"Too many concept groups: {len(groups)} > {max_groups}"
        )

    normalised_groups: List[Dict[str, List[str]]] = []
    for idx, group in enumerate(groups):
        if not isinstance(group, dict):
            raise BooleanFormulationError(
                f"Concept group {idx} must be a dict, got {type(group).__name__}"
            )

        alternatives = group.get("alternatives")
        if not isinstance(alternatives, list) or not alternatives:
            raise BooleanFormulationError(
                f"Group {idx} must contain a non-empty 'alternatives' list"
            )

        if len(alternatives) > max_alternatives:
            raise BooleanFormulationError(
                f"Group {idx} has too many alternatives: "
                f"{len(alternatives)} > {max_alternatives}"
            )

        cleaned: List[str] = []
        for alt in alternatives:
            if not _is_valid_alternative(alt):
                raise BooleanFormulationError(
                    f"Group {idx} contains an invalid alternative: {alt!r}"
                )
            alt = alt.strip()
            if len(alt) > MAX_ALTERNATIVE_LENGTH:
                raise BooleanFormulationError(
                    f"Group {idx} alternative too long: {alt[:40]!r}..."
                )
            cleaned.append(alt)

        normalised_groups.append({"alternatives": cleaned})

    return {"required_concepts": normalised_groups}


def render_boolean_query(
    concepts: Dict[str, List[Dict[str, List[str]]]],
) -> str:
    """Render validated concepts into a ChatNoir-style Boolean query.

    Rendering rules:
      * OR within a concept group's alternatives.
      * AND between required concept groups.
      * Multi-word alternatives are wrapped in double quotes (phrases).
      * Each group is parenthesised so precedence is explicit.
      * Internal double quotes are stripped to avoid escaping issues.
    """
    groups = concepts["required_concepts"]
    rendered_groups: List[str] = []

    for group in groups:
        terms: List[str] = []
        for alt in group["alternatives"]:
            # Strip characters that would break the query parser.
            safe = alt.replace('"', "")
            if not safe:
                raise BooleanFormulationError(
                    f"Alternative became empty after escaping: {alt!r}"
                )
            if " " in safe:
                terms.append(f'"{safe}"')
            else:
                terms.append(safe)
        rendered_groups.append("(" + " OR ".join(terms) + ")")

    return " AND ".join(rendered_groups)


def parse_and_render(raw_text: str) -> Tuple[str, Dict[str, Any]]:
    """Parse raw LLM output, validate it, and render a Boolean query.

    Returns (boolean_query, validated_concepts).
    """
    text = _strip_code_fences(raw_text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise BooleanFormulationError(f"LLM output is not valid JSON: {exc}") from exc

    concepts = validate_concepts(parsed)
    query = render_boolean_query(concepts)
    if not query:
        raise BooleanFormulationError("Rendered Boolean query is empty")
    return query, concepts


class BooleanFormulationAgent(AgentBase):
    """Agent that reformulates a query into an explicit Boolean expression."""

    def __init__(
        self,
        embed_model: SentenceTransformer,
        max_groups: int = MAX_CONCEPT_GROUPS,
        max_alternatives: int = MAX_ALTERNATIVES_PER_GROUP,
    ):
        super().__init__(agent_id=0, embed_model=embed_model)
        self.client = _make_client()
        self.max_groups = max_groups
        self.max_alternatives = max_alternatives

    def _call_llm(self, query: str) -> str:
        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": BOOLEAN_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": BOOLEAN_USER_PROMPT_TEMPLATE.format(query),
                },
            ],
            temperature=0.2,
            max_tokens=256,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError("No content returned from LLM")
        return content.strip()

    def compute_effects(self, query_features: Dict[str, Any]) -> Dict[str, Any]:
        """Reformulate the query into Boolean form and retrieve.

        Compatible with the ReformulationAgent return contract.  On any
        generation, validation, or rendering failure, falls back to the
        original query and logs the reason explicitly.
        """
        original_query = query_features["query_text"]
        retriever = query_features["retriever"]
        top_k = query_features["top_k"]

        start_time = time.time()
        formulation_time = 0.0
        boolean_query: str | None = None
        validated_concepts: Dict[str, Any] | None = None
        fallback_reason: str | None = None

        try:
            raw = self._call_llm(original_query)
            formulation_time = time.time() - start_time
            boolean_query, validated_concepts = parse_and_render(raw)
        except Exception as exc:
            fallback_reason = f"Boolean formulation failed: {exc}"
            logger.warning(
                "Boolean formulation failed for query %r: %s",
                original_query,
                exc,
                exc_info=False,
            )

        query_to_run = boolean_query if boolean_query else original_query
        mode = (
            "boolean" if boolean_query else "boolean_fallback_to_original"
        )

        retrieval_start = time.time()
        raw_results = retriever(query_to_run, top_k)
        retrieval_time = time.time() - retrieval_start

        if not raw_results:
            # Distinguish retrieval errors from valid zero-result queries.
            new_doc_ids: List[str] = []
            new_doc_scores = np.array([], dtype=np.float32)
            new_corpus_data: Dict[str, str] = {}
            retrieval_error = True
        else:
            new_doc_ids = raw_results[0]
            new_doc_scores = np.asarray(raw_results[1], dtype=np.float32)
            new_corpus_data = raw_results[2] if len(raw_results) > 2 else {}
            retrieval_error = False

        total_elapsed = formulation_time + retrieval_time

        result: Dict[str, Any] = {
            "new_query_text": query_to_run,
            "new_doc_ids": new_doc_ids,
            "new_doc_scores": new_doc_scores,
            "new_corpus_data": new_corpus_data,
            "elapsed_time": total_elapsed,
            "cost": 1.0,
            "formulation_mode": mode,
            "boolean_query": boolean_query,
            "validated_concepts": validated_concepts,
            "fallback_reason": fallback_reason,
            "retrieval_error": retrieval_error,
            "num_results": len(new_doc_ids),
        }
        return result
