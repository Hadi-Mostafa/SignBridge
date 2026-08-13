"""Contracts for context-aware, offline next-word suggestions."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import backend.main as main
from backend.routers import word_suggestions
from backend.services.context_word_predictor import ContextWordPredictor


class ContextWordPredictorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.predictor = ContextWordPredictor()
        cls.predictor.load()

    def test_assets_are_pinned_and_runtime_is_offline(self) -> None:
        status = ContextWordPredictor.asset_status(verify=True)
        self.assertTrue(status["ready"])
        self.assertTrue(status["neural_asset_ready"])
        self.assertTrue(status["integrity_verified"])
        self.assertTrue(status["offline"])
        self.assertFalse(status["downloads_on_request"])

    def test_general_context_changes_ranked_predictions(self) -> None:
        happy = self.predictor.suggest(sentence="Happy", prefix="", limit=8)
        happy_words = [item["word"] for item in happy["suggestions"]]
        self.assertIn("birthday", happy_words)
        self.assertIn("new", happy_words)

        birthday = self.predictor.suggest(sentence="Happy Birthday", prefix="", limit=8)
        birthday_words = [item["word"] for item in birthday["suggestions"]]
        self.assertNotEqual(happy_words, birthday_words)
        self.assertIn("to", birthday_words)
        self.assertTrue({"party", "cake", "gift"}.intersection(birthday_words))

        want_to = self.predictor.suggest(sentence="I want to", prefix="", limit=8)
        self.assertEqual(want_to["suggestions"][0]["word"], "be")

    def test_prefix_filters_contextual_candidates(self) -> None:
        result = self.predictor.suggest(
            context=["Happy"], sentence="ignored because explicit context is supplied", prefix="b", limit=8
        )
        words = [item["word"] for item in result["suggestions"]]
        self.assertEqual(words[0], "birthday")
        self.assertTrue(all(word.startswith("b") for word in words))
        self.assertEqual(result["context"], ["happy"])

    def test_optional_neural_absence_has_deterministic_fallback(self) -> None:
        original_session, original_tokenizer = self.predictor._session, self.predictor._tokenizer
        try:
            self.predictor._session = None
            self.predictor._tokenizer = None
            first = self.predictor.suggest(sentence="I am", prefix="", limit=8)
            second = self.predictor.suggest(sentence="I am", prefix="", limit=8)
        finally:
            self.predictor._session, self.predictor._tokenizer = original_session, original_tokenizer
        self.assertEqual(first["engine"], "google_web_1t_bigram")
        self.assertEqual(first["suggestions"], second["suggestions"])

    def test_empty_match_is_safe(self) -> None:
        result = self.predictor.suggest(sentence="hello", prefix="zzzzzzzz", limit=8)
        self.assertEqual(result["suggestions"], [])


class ContextWordSuggestionAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        word_suggestions.predictor = ContextWordPredictor()
        word_suggestions.predictor.load()
        cls.client = TestClient(main.app)

    def test_stable_endpoint_contract(self) -> None:
        response = self.client.post(
            "/api/word-suggestions",
            json={"context": ["Happy"], "sentence": "Happy", "prefix": "b", "limit": 5},
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["context"], ["happy"])
        self.assertEqual(payload["prefix"], "b")
        self.assertLessEqual(len(payload["suggestions"]), 5)
        self.assertEqual(payload["suggestions"][0]["rank"], 1)
        self.assertEqual(payload["suggestions"][0]["word"], "birthday")
        self.assertGreater(payload["suggestions"][0]["score"], 0)
        self.assertTrue(payload["offline"])
        self.assertIn(payload["engine"], {"distilgpt2_int8_hybrid", "google_web_1t_bigram"})

    def test_request_validation_is_bounded(self) -> None:
        invalid_prefix = self.client.post(
            "/api/word-suggestions", json={"context": [], "prefix": "a1", "limit": 8}
        )
        self.assertEqual(invalid_prefix.status_code, 422)
        invalid_limit = self.client.post(
            "/api/word-suggestions", json={"context": [], "prefix": "a", "limit": 100}
        )
        self.assertEqual(invalid_limit.status_code, 422)

    def test_health_reports_honest_local_readiness(self) -> None:
        response = self.client.get("/health")
        payload = response.json()
        capability = payload["capabilities"]["word_suggestions"]
        self.assertTrue(capability["ready"])
        self.assertTrue(capability["metadata"]["offline"])
        self.assertFalse(capability["metadata"]["downloads_on_request"])
        self.assertTrue(payload["modes"]["word_suggestions"])

    def test_request_never_calls_downloader(self) -> None:
        # The runtime module does not import huggingface_hub at all. The patch
        # is an explicit regression guard if that ever changes.
        with patch("huggingface_hub.hf_hub_download", side_effect=AssertionError("network call")):
            response = self.client.post(
                "/api/word-suggestions", json={"sentence": "thank", "prefix": "", "limit": 4}
            )
        self.assertEqual(response.status_code, 200, response.text)


if __name__ == "__main__":
    unittest.main(verbosity=2)

