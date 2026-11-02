"""Unit tests for Boolean query formulation.

These tests use unittest.mock so they do not require a live ChatNoir API key
or a running LLM endpoint.  Live backend tests are intentionally omitted.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np

root_path = Path(__file__).resolve().parent.parent
src_path = root_path / "src"
sys.path.insert(0, str(root_path))
sys.path.insert(0, str(src_path))

from src.agents.boolean_formulate import (
    BooleanFormulationAgent,
    BooleanFormulationError,
    parse_and_render,
    render_boolean_query,
    validate_concepts,
)
from src.agents.reformulate import ReformulationAgent


class TestConceptValidation(unittest.TestCase):
    """Validation of LLM concept structures."""

    def test_valid_concepts_normalised(self):
        raw = {
            "required_concepts": [
                {"alternatives": ["electric vehicle", "electric car"]},
                {"alternatives": ["battery", "batteries"]},
                {"alternatives": ["recycling"]},
            ]
        }
        out = validate_concepts(raw)
        self.assertEqual(len(out["required_concepts"]), 3)
        self.assertEqual(out["required_concepts"][0]["alternatives"][0], "electric vehicle")

    def test_empty_concept_list_rejected(self):
        with self.assertRaises(BooleanFormulationError):
            validate_concepts({"required_concepts": []})

    def test_missing_key_rejected(self):
        with self.assertRaises(BooleanFormulationError):
            validate_concepts({"required_concepts": [{"terms": ["a"]}]})

    def test_empty_alternative_rejected(self):
        raw = {"required_concepts": [{"alternatives": ["", "battery"]}]}
        with self.assertRaises(BooleanFormulationError):
            validate_concepts(raw)

    def test_non_string_alternative_rejected(self):
        raw = {"required_concepts": [{"alternatives": [123]}]}
        with self.assertRaises(BooleanFormulationError):
            validate_concepts(raw)

    def test_too_many_groups_rejected(self):
        raw = {"required_concepts": [{"alternatives": ["a"]} for _ in range(10)]}
        with self.assertRaises(BooleanFormulationError):
            validate_concepts(raw, max_groups=8)

    def test_too_many_alternatives_rejected(self):
        raw = {"required_concepts": [{"alternatives": ["a", "b", "c", "d", "e", "f"]}]}
        with self.assertRaises(BooleanFormulationError):
            validate_concepts(raw, max_alternatives=5)


class TestRendering(unittest.TestCase):
    """Deterministic rendering of validated concepts to Boolean strings."""

    def test_simple_rendering(self):
        concepts = {
            "required_concepts": [
                {"alternatives": ["electric vehicle", "electric car"]},
                {"alternatives": ["battery", "batteries"]},
                {"alternatives": ["recycling"]},
            ]
        }
        query = render_boolean_query(concepts)
        self.assertIn('"electric vehicle" OR "electric car"', query)
        self.assertIn("(battery OR batteries)", query)
        self.assertIn("(recycling)", query)
        self.assertIn(" AND ", query)
        self.assertEqual(query.count("("), 3)
        self.assertEqual(query.count(")"), 3)

    def test_single_concept(self):
        concepts = {"required_concepts": [{"alternatives": ["foo"]}]}
        self.assertEqual(render_boolean_query(concepts), "(foo)")

    def test_quotes_stripped(self):
        concepts = {"required_concepts": [{"alternatives": ['"hello world"']}]}
        query = render_boolean_query(concepts)
        self.assertEqual(query, '("hello world")')

    def test_parse_and_render_strips_fences(self):
        raw = '```json\n{"required_concepts": [{"alternatives": ["a", "b"]}]}\n```'
        query, concepts = parse_and_render(raw)
        self.assertEqual(query, "(a OR b)")
        self.assertIn("required_concepts", concepts)

    def test_parse_and_render_rejects_invalid_json(self):
        with self.assertRaises(BooleanFormulationError):
            parse_and_render("not json")


class TestAgentCompatibility(unittest.TestCase):
    """ReformulationAgent contract with both formulation modes."""

    def _make_encoder(self) -> MagicMock:
        encoder = MagicMock()
        encoder.encode.return_value = np.zeros(384, dtype=np.float32)
        return encoder

    def _make_retriever(self, doc_ids: List[str] | None = None) -> MagicMock:
        retriever = MagicMock()
        doc_ids = doc_ids or ["doc1", "doc2"]
        scores = np.array([1.0, 0.5], dtype=np.float32)
        corpus = {"doc1": "text one", "doc2": "text two"}
        retriever.return_value = (doc_ids, scores, corpus)
        return retriever

    @patch("src.agents.boolean_formulate.MODEL_NAME", "test-model")
    @patch("src.agents.boolean_formulate.OpenAI")
    def test_boolean_agent_returns_contract_fields(self, mock_openai: MagicMock):
        client = MagicMock()
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = json.dumps({
            "required_concepts": [
                {"alternatives": ["electric vehicle", "electric car"]},
                {"alternatives": ["battery"]},
            ]
        })
        client.chat.completions.create.return_value = response
        mock_openai.return_value = client

        agent = BooleanFormulationAgent(embed_model=self._make_encoder())
        retriever = self._make_retriever()
        effects = agent.compute_effects({
            "query_text": "electric vehicle battery recycling",
            "retriever": retriever,
            "top_k": 10,
        })

        self.assertIn("new_query_text", effects)
        self.assertIn("new_doc_ids", effects)
        self.assertIn("new_doc_scores", effects)
        self.assertIn("new_corpus_data", effects)
        self.assertIn("elapsed_time", effects)
        self.assertIn("cost", effects)
        self.assertIn("formulation_mode", effects)
        self.assertIn("boolean_query", effects)
        self.assertEqual(effects["formulation_mode"], "boolean")
        self.assertEqual(retriever.call_count, 1)
        self.assertIn("AND", effects["new_query_text"])

    @patch("src.agents.boolean_formulate.MODEL_NAME", "test-model")
    @patch("src.agents.boolean_formulate.OpenAI")
    def test_boolean_agent_fallback_on_invalid_llm_output(self, mock_openai: MagicMock):
        client = MagicMock()
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = "not valid json"
        client.chat.completions.create.return_value = response
        mock_openai.return_value = client

        agent = BooleanFormulationAgent(embed_model=self._make_encoder())
        retriever = self._make_retriever()
        effects = agent.compute_effects({
            "query_text": "electric vehicle battery recycling",
            "retriever": retriever,
            "top_k": 10,
        })

        self.assertEqual(effects["formulation_mode"], "boolean_fallback_to_original")
        self.assertIsNotNone(effects["fallback_reason"])
        self.assertEqual(retriever.call_count, 1)
        self.assertEqual(effects["new_query_text"], "electric vehicle battery recycling")

    @patch("src.agents.reformulate.MODEL_NAME", "test-model")
    @patch("src.agents.reformulate.OpenAI")
    def test_reformulation_agent_natural_mode(self, mock_openai: MagicMock):
        client = MagicMock()
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = "rewritten query"
        client.chat.completions.create.return_value = response
        mock_openai.return_value = client

        agent = ReformulationAgent(
            embed_model=self._make_encoder(),
            formulation_mode="natural",
        )
        retriever = self._make_retriever()
        effects = agent.compute_effects({
            "query_text": "original query",
            "retriever": retriever,
            "top_k": 10,
        })

        self.assertEqual(effects["new_query_text"], "rewritten query")
        self.assertEqual(effects["formulation_mode"], "natural")
        self.assertEqual(retriever.call_count, 1)

    @patch("src.agents.boolean_formulate.OpenAI")
    @patch("src.agents.boolean_formulate.MODEL_NAME", "test-model")
    @patch("src.agents.reformulate.MODEL_NAME", "test-model")
    @patch("src.agents.reformulate.OpenAI")
    def test_reformulation_agent_boolean_mode_delegates(
        self, mock_reformulate_openai: MagicMock, mock_bool_openai: MagicMock
    ):
        client = MagicMock()
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = json.dumps({
            "required_concepts": [{"alternatives": ["foo", "bar"]}]
        })
        client.chat.completions.create.return_value = response
        mock_bool_openai.return_value = client

        agent = ReformulationAgent(
            embed_model=self._make_encoder(),
            formulation_mode="boolean",
        )
        retriever = self._make_retriever()
        effects = agent.compute_effects({
            "query_text": "foo query",
            "retriever": retriever,
            "top_k": 10,
        })

        self.assertEqual(effects["formulation_mode"], "boolean")
        self.assertEqual(retriever.call_count, 1)

    def test_reformulation_agent_rejects_unknown_mode(self):
        with self.assertRaises(ValueError):
            ReformulationAgent(embed_model=self._make_encoder(), formulation_mode="fancy")


class TestRetrievalCallCounts(unittest.TestCase):
    """Formulation baselines issue exactly one retrieval call."""

    def _make_encoder(self) -> MagicMock:
        encoder = MagicMock()
        encoder.encode.return_value = np.zeros(384, dtype=np.float32)
        return encoder

    @patch("src.agents.reformulate.MODEL_NAME", "test-model")
    @patch("src.agents.reformulate.OpenAI")
    def test_natural_baseline_one_retrieval(self, mock_openai: MagicMock):
        client = MagicMock()
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = "rewritten"
        client.chat.completions.create.return_value = response
        mock_openai.return_value = client

        retriever = MagicMock()
        retriever.return_value = (["d1"], np.array([1.0], dtype=np.float32), {"d1": "x"})

        agent = ReformulationAgent(
            embed_model=self._make_encoder(),
            formulation_mode="natural",
        )
        agent.compute_effects({
            "query_text": "q",
            "retriever": retriever,
            "top_k": 10,
        })
        self.assertEqual(retriever.call_count, 1)

    @patch("src.agents.reformulate.MODEL_NAME", "test-model")
    @patch("src.agents.boolean_formulate.MODEL_NAME", "test-model")
    @patch("src.agents.boolean_formulate.OpenAI")
    def test_boolean_baseline_one_retrieval(self, mock_openai: MagicMock):
        client = MagicMock()
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = json.dumps({
            "required_concepts": [{"alternatives": ["a"]}]
        })
        client.chat.completions.create.return_value = response
        mock_openai.return_value = client

        retriever = MagicMock()
        retriever.return_value = (["d1"], np.array([1.0], dtype=np.float32), {"d1": "x"})

        agent = ReformulationAgent(
            embed_model=self._make_encoder(),
            formulation_mode="boolean",
        )
        agent.compute_effects({
            "query_text": "q",
            "retriever": retriever,
            "top_k": 10,
        })
        self.assertEqual(retriever.call_count, 1)


if __name__ == "__main__":
    unittest.main()
