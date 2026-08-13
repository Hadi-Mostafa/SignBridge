"""Thread-safe, stateless realtime ASL alphabet inference.

REST and WebSocket callers share this exact service.  Temporal smoothing belongs
to a per-client session layer; keeping it here would leak one user's predictions
into another user's stream.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

try:
    from .. import config as app_config
    from .keypoint_classifier import KeypointClassifier
except ImportError:  # pragma: no cover - direct backend-path execution
    import config as app_config
    from services.keypoint_classifier import KeypointClassifier


class RealtimeASLPipeline:
    """Single verified MediaPipe-landmark → Random Forest inference path."""

    def __init__(self, confidence_threshold: float | None = None) -> None:
        threshold = (
            app_config.ALPHABET_MIN_CONFIDENCE
            if confidence_threshold is None
            else confidence_threshold
        )
        self.classifier = KeypointClassifier(confidence_threshold=float(threshold))
        # MediaPipe Tasks objects are not documented as concurrently callable.
        # The lock protects only model lifecycle/inference, not temporal state.
        self._lock = threading.RLock()
        self._loaded = False

    def load(self) -> None:
        with self._lock:
            if self._loaded:
                return
            self.classifier.load_model()
            self.classifier.self_test()
            self._loaded = True

    def reset(self) -> None:
        """Compatibility no-op: the shared pipeline deliberately has no stream state."""

    def predict(self, frame: np.ndarray, smooth: bool = False) -> dict[str, Any] | None:
        """Classify one frame, or return ``None`` when no hand is detected.

        ``smooth`` is accepted for backward API compatibility but intentionally
        ignored. Callers that need smoothing must keep it in per-client state.
        """

        del smooth
        started = time.perf_counter()
        with self._lock:
            if not self._loaded:
                self.load()
            output = self.classifier.predict_frame(frame)
        if output is None:
            return None

        latency = dict(output.get("latency", {}))
        latency["pipeline_total_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
        return {
            "sign": output["label"],
            "label": output["label"],
            "class_id": output["class_id"],
            "confidence": round(float(output["confidence"]), 4),
            "accepted": bool(output["accepted"]),
            "reason": output["reason"],
            "box": output["box"],
            "top5": output["top5"],
            "probabilities": output["probabilities"],
            "landmarks": output["landmarks"],
            "handedness": output.get("handedness"),
            "detector": "mediapipe-hand-landmarker",
            "model": "keypoint-random-forest",
            "latency": latency,
        }

    def metadata(self) -> dict[str, Any]:
        with self._lock:
            return {
                "name": "realtime_asl_alphabet",
                "loaded": self._loaded,
                "stateless": True,
                "thread_safe": True,
                "classifier": self.classifier.metadata(),
            }

    def close(self) -> None:
        with self._lock:
            self.classifier.close()
            self._loaded = False

    def __enter__(self) -> "RealtimeASLPipeline":
        self.load()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
