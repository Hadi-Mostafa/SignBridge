"""Contract tests for checksum-locked alphabet and WLASL inference models.

These tests intentionally run offline. They verify model integrity, strict
schemas, label order, finite outputs, rejection metadata, no-hand handling, and
the absence of mutable cross-client temporal state in the shared pipeline.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
os.environ.setdefault("SIGNBRIDGE_OFFLINE", "1")

from models.word_model import WordModel, validate_and_reshape_55  # noqa: E402
from services.model_assets import (  # noqa: E402
    ModelAsset,
    ModelAssetError,
    get_asset,
    list_assets,
    resolve_asset,
    sha256_file,
)
import services.model_assets as model_assets  # noqa: E402
from services.realtime_pipeline import RealtimeASLPipeline  # noqa: E402


class ModelAssetTests(unittest.TestCase):
    def test_all_locked_assets_resolve_with_exact_hashes(self) -> None:
        for name in list_assets():
            with self.subTest(asset=name):
                asset = get_asset(name)
                path = resolve_asset(name, allow_download=False)
                self.assertEqual(sha256_file(path), asset.sha256)

    def test_wlasl_labels_match_original_label_encoder_order(self) -> None:
        labels = resolve_asset("wlasl100_labels", allow_download=False).read_text(
            encoding="utf-8"
        ).splitlines()
        glossary = json.loads((PROJECT_ROOT / "data" / "WLASL_v0.3.json").read_text(encoding="utf-8"))
        self.assertEqual(labels, sorted(entry["gloss"] for entry in glossary[:100]))
        self.assertEqual(len(labels), 100)
        self.assertEqual(labels[77], "school")

    def test_runtime_download_policy_fails_closed(self) -> None:
        missing = ModelAsset(
            name="missing-test-asset",
            filename="missing.bin",
            sha256="0" * 64,
            repo_id="owner/repository",
            revision="1" * 40,
        )
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(model_assets, "get_asset", return_value=missing), \
                patch.object(model_assets, "_candidate_paths", return_value=[]), \
                patch.object(model_assets, "MODEL_CACHE_DIR", Path(directory)), \
                patch.object(model_assets.app_config, "MODEL_DOWNLOAD_ENABLED", False), \
                patch.dict(os.environ, {"SIGNBRIDGE_OFFLINE": "0"}):
            with self.assertRaisesRegex(ModelAssetError, "MODEL_DOWNLOAD_ENABLED is false"):
                resolve_asset(missing.name, allow_download=True)


class AlphabetPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pipeline = RealtimeASLPipeline()
        cls.pipeline.load()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.pipeline.close()

    def test_classifier_schema_and_self_test(self) -> None:
        classifier = self.pipeline.classifier
        self.assertEqual(tuple(int(value) for value in classifier.class_ids), tuple(range(26)))
        self.assertEqual(classifier.labels, tuple(chr(ord("A") + i) for i in range(26)))
        self.assertEqual(classifier.model.n_features_in_, 42)
        self.assertEqual(classifier.self_test()["status"], "ok")

    def test_static_model_never_accepts_dynamic_j_or_z(self) -> None:
        classifier = self.pipeline.classifier
        for label in ("J", "Z"):
            accepted, reason = classifier._acceptance(label, 1.0)
            self.assertFalse(accepted)
            self.assertEqual(reason, "dynamic_letter_requires_motion")

    def test_blank_frame_returns_no_hand(self) -> None:
        blank = np.zeros((224, 224, 3), dtype=np.uint8)
        self.assertIsNone(self.pipeline.predict(blank, smooth=False))

    def test_pipeline_has_no_cross_client_temporal_state(self) -> None:
        metadata = self.pipeline.metadata()
        self.assertTrue(metadata["stateless"])
        self.assertTrue(metadata["thread_safe"])
        self.assertNotIn("recent", vars(self.pipeline))
        self.pipeline.reset()
        self.assertNotIn("recent", vars(self.pipeline))

    def test_invalid_frame_shape_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            self.pipeline.predict(np.zeros((224, 224), dtype=np.uint8))


class WordModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = WordModel()

    def test_strict_model_and_label_contract(self) -> None:
        metadata = self.model.metadata()
        self.assertEqual(metadata["revision"], WordModel.REVISION)
        self.assertEqual(metadata["classes"], 100)
        self.assertEqual(metadata["input_shape"], [1, 55, 100])
        self.assertEqual(self.model.vocab[77], "school")
        self.assertEqual(self.model.self_test()["status"], "ok")

    def test_prediction_returns_calibratable_acceptance_fields(self) -> None:
        base = np.linspace(-0.4, 0.4, 110, dtype=np.float32)
        frames = [
            np.clip(base + 0.08 * np.sin(index / 4.0), -0.9, 0.9).tolist()
            for index in range(50)
        ]
        result = self.model.predict_word(frames)
        required = {"label", "class_id", "confidence", "margin", "accepted", "reason", "top5"}
        self.assertTrue(required.issubset(result))
        self.assertEqual(len(result["top5"]), 5)
        self.assertIn(result["reason"], {"accepted", "low_confidence", "low_margin"})
        self.assertTrue(np.isfinite(result["confidence"]))
        self.assertTrue(0.0 <= result["confidence"] <= 1.0)

    def test_idle_sequence_is_rejected_before_closed_set_inference(self) -> None:
        frame = np.zeros(110, dtype=np.float32).tolist()
        result = self.model.predict_word([frame for _ in range(50)])
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "insufficient_motion")
        self.assertEqual(result["top5"], [])
        self.assertEqual(result["confidence"], 0.0)

    def test_input_validation_rejects_bad_shape_nonfinite_and_range(self) -> None:
        with self.assertRaises(ValueError):
            self.model.predict_word([[0.0] * 110 for _ in range(49)])
        with self.assertRaises(ValueError):
            validate_and_reshape_55([float("nan")] + [0.0] * 109)
        with self.assertRaises(ValueError):
            validate_and_reshape_55([2.0] + [0.0] * 109)


if __name__ == "__main__":
    unittest.main(verbosity=2)
