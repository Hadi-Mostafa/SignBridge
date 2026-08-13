"""Offline API contract tests runnable with ``python -m unittest``."""

from __future__ import annotations

import base64
import io
import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image
from starlette.websockets import WebSocketDisconnect

import backend.main as main
from backend.config import CORS_ALLOWED_ORIGINS, MAX_IMAGE_BYTES, WORD_SEQUENCE_LENGTH
from backend.routers import sign_to_text
from backend.utils.latency_logger import LatencyLogger


class FakePipeline:
    def __init__(self, result=None):
        self.result = result
        self.calls = 0

    def predict(self, _frame, smooth=False):
        self.calls += 1
        return self.result

    def metadata(self):
        return {
            "model": "fake-alphabet",
            "experimental": False,
            "path": "C:/private/alphabet.joblib",
            "nested": {"manifest_path": "C:/private/manifest.json", "safe": "visible"},
        }

    def close(self):
        return None


class FakeWordModel:
    def __init__(self):
        self.calls = 0

    def predict_word(self, frames):
        self.calls += 1
        if len(frames) != WORD_SEQUENCE_LENGTH:
            raise AssertionError("wrong sequence length")
        return {
            "label": "hello",
            "confidence": 0.91,
            "margin": 0.42,
            "accepted": True,
            "reason": "accepted",
            "top5": [{"label": "hello", "confidence": 0.91}],
        }

    def metadata(self):
        return {"model": "fake-word", "experimental": True}

    def self_test(self):
        return {"ok": True}


def jpeg_bytes(width=32, height=24):
    output = io.BytesIO()
    Image.new("RGB", (width, height), color=(20, 80, 120)).save(output, format="JPEG")
    return output.getvalue()


class APITestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(main.app)
        cls.origin = CORS_ALLOWED_ORIGINS[0]

    def setUp(self):
        self.original_pipeline = sign_to_text.pipeline
        self.original_pipeline_error = sign_to_text.pipeline_error
        self.original_word_model = main.word_model
        self.original_word_error = main.word_model_error
        sign_to_text._set_pipeline_for_tests(FakePipeline(), None)
        main.word_model = FakeWordModel()
        main.word_model_error = None

    def tearDown(self):
        sign_to_text._set_pipeline_for_tests(self.original_pipeline, self.original_pipeline_error)
        main.word_model = self.original_word_model
        main.word_model_error = self.original_word_error

    def test_liveness_and_readiness_contract(self):
        live = self.client.get("/health/live")
        self.assertEqual(live.status_code, 200)
        self.assertEqual(live.json()["status"], "alive")

        ready = self.client.get("/health")
        self.assertEqual(ready.status_code, 200)
        payload = ready.json()
        self.assertEqual(payload["status"], "ready")
        self.assertTrue(payload["capabilities"]["alphabet"]["ready"])
        self.assertTrue(payload["capabilities"]["words"]["ready"])
        self.assertTrue(payload["capabilities"]["words"]["metadata"]["experimental"])
        serialized = json.dumps(payload)
        self.assertNotIn("C:/private", serialized)
        self.assertEqual(
            payload["capabilities"]["alphabet"]["metadata"]["nested"]["safe"],
            "visible",
        )

        sign_to_text._set_pipeline_for_tests(None, "missing fixture")
        unavailable = self.client.get("/health")
        self.assertEqual(unavailable.status_code, 503)
        self.assertEqual(unavailable.json()["status"], "not_ready")

    def test_predict_is_bounded_and_runs_shared_pipeline(self):
        fake = FakePipeline(
            {
                "label": "A",
                "sign": "A",
                "confidence": 0.93,
                "accepted": True,
                "reason": "accepted",
                "top5": [{"label": "A", "confidence": 0.93}],
            }
        )
        sign_to_text._set_pipeline_for_tests(fake)
        response = self.client.post(
            "/predict",
            files={"file": ("hand.jpg", jpeg_bytes(), "image/jpeg")},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["sign"], "A")
        self.assertEqual(response.json()["type"], "prediction")
        self.assertEqual(fake.calls, 1)

        wrong_type = self.client.post(
            "/predict",
            files={"file": ("hand.txt", b"not an image", "text/plain")},
        )
        self.assertEqual(wrong_type.status_code, 415)
        self.assertEqual(wrong_type.json()["detail"]["code"], "unsupported_image_type")

        too_large = self.client.post(
            "/predict",
            files={"file": ("large.jpg", b"x" * (MAX_IMAGE_BYTES + 1), "image/jpeg")},
        )
        self.assertEqual(too_large.status_code, 413)

    def test_websocket_rejects_untrusted_origin(self):
        with self.assertRaises(WebSocketDisconnect) as caught:
            with self.client.websocket_connect(
                "/ws/sign-to-text", headers={"origin": "https://attacker.invalid"}
            ):
                pass
        self.assertEqual(caught.exception.code, 1008)

    def test_websocket_alphabet_contract_and_request_id(self):
        fake = FakePipeline(
            {
                "label": "B",
                "confidence": 0.40,
                "accepted": False,
                "reason": "below_confidence_threshold",
                "top5": [{"label": "B", "confidence": 0.40}],
                "latency": {"classify_ms": 1.0},
            }
        )
        sign_to_text._set_pipeline_for_tests(fake)
        encoded = base64.b64encode(jpeg_bytes()).decode("ascii")
        with self.client.websocket_connect(
            "/ws/sign-to-text", headers={"origin": self.origin}
        ) as websocket:
            connected = websocket.receive_json()
            self.assertEqual(connected["type"], "status")
            websocket.send_json(
                {"type": "frame", "mode": "alphabet", "request_id": "alphabet-1", "image": encoded}
            )
            prediction = websocket.receive_json()
        self.assertEqual(prediction["type"], "prediction")
        self.assertEqual(prediction["request_id"], "alphabet-1")
        self.assertFalse(prediction["accepted"])
        self.assertIsNone(prediction["label"])
        self.assertEqual(prediction["candidate_label"], "B")
        self.assertIn("top_predictions", prediction)
        self.assertIn("server_latency_ms", prediction)

    def test_word_capture_predicts_once_and_requires_reset(self):
        fake_word = FakeWordModel()
        main.word_model = fake_word
        landmarks = [0.5] * 110
        with self.client.websocket_connect(
            "/ws/sign-to-text", headers={"origin": self.origin}
        ) as websocket:
            websocket.receive_json()  # connected
            websocket.send_json(
                {
                    "type": "capture_start",
                    "mode": "words",
                    "request_id": "word-1",
                    "total": WORD_SEQUENCE_LENGTH,
                }
            )
            started = websocket.receive_json()
            self.assertEqual(started["captured"], 0)

            for index in range(WORD_SEQUENCE_LENGTH):
                websocket.send_json(
                    {
                        "type": "frame",
                        "mode": "words",
                        "request_id": "word-1",
                        "frame_index": index,
                        "total": WORD_SEQUENCE_LENGTH,
                        "landmarks": landmarks,
                    }
                )
                progress = websocket.receive_json()
                self.assertEqual(progress["type"], "progress")
                self.assertEqual(progress["captured"], index + 1)

            prediction = websocket.receive_json()
            self.assertEqual(prediction["type"], "prediction")
            self.assertTrue(prediction["accepted"])
            self.assertEqual(prediction["label"], "hello")
            self.assertEqual(fake_word.calls, 1)

            websocket.send_json(
                {
                    "type": "capture_end",
                    "mode": "words",
                    "request_id": "word-1",
                    "total": WORD_SEQUENCE_LENGTH,
                }
            )
            ended = websocket.receive_json()
            self.assertTrue(ended["awaiting_reset"])

            websocket.send_json(
                {
                    "type": "frame",
                    "mode": "words",
                    "request_id": "word-1",
                    "frame_index": 0,
                    "landmarks": landmarks,
                }
            )
            rejected = websocket.receive_json()
            self.assertEqual(rejected["error"]["code"], "reset_required")
            self.assertEqual(fake_word.calls, 1)

    def test_invalid_landmarks_use_structured_error(self):
        with self.client.websocket_connect(
            "/ws/sign-to-text", headers={"origin": self.origin}
        ) as websocket:
            websocket.receive_json()
            websocket.send_json(
                {
                    "type": "frame",
                    "mode": "words",
                    "request_id": "bad-landmarks",
                    "landmarks": [float("nan")] * 110,
                }
            )
            response = websocket.receive_json()
        self.assertEqual(response["type"], "error")
        self.assertEqual(response["request_id"], "bad-landmarks")
        self.assertEqual(response["error"]["code"], "invalid_landmarks")

    def test_latency_total_is_not_double_counted_or_sensitive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latency.jsonl"
            logger = LatencyLogger(path)
            entry = logger.log(
                "test",
                {"stage_a_ms": 5.0, "stage_b_ms": 7.0, "total_ms": 12.0},
                {"transcript": "private", "token_count": 2},
            )
            self.assertEqual(entry["total_ms"], 12.0)
            self.assertNotIn("transcript", entry.get("metadata", {}))
            stored = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(stored["total_ms"], 12.0)


if __name__ == "__main__":
    unittest.main()
