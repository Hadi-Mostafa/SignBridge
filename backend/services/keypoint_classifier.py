"""Static ASL alphabet recognition from MediaPipe hand landmarks.

The classifier is intentionally conservative: it only predicts after a hand is
detected, verifies every model asset against the project lock file, validates
the Random Forest schema, and maps probability columns through
``model.classes_`` rather than assuming their order.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

import cv2
import joblib
import mediapipe as mp
import numpy as np

try:
    from .. import config as app_config
    from .model_assets import asset_metadata, get_asset, resolve_asset, verify_sha256
except ImportError:  # pragma: no cover - direct backend-path execution
    import config as app_config
    from services.model_assets import asset_metadata, get_asset, resolve_asset, verify_sha256


EXPECTED_LANDMARKS = 21
EXPECTED_FEATURES = EXPECTED_LANDMARKS * 2
EXPECTED_CLASS_IDS = tuple(range(26))
DYNAMIC_LETTERS = frozenset({"J", "Z"})


def _class_id_to_label(class_id: Any) -> str:
    """Map the exact estimator class ID to its public alphabet label."""

    if isinstance(class_id, np.generic):
        class_id = class_id.item()
    if isinstance(class_id, (int, np.integer)) and 0 <= int(class_id) < 26:
        return chr(ord("A") + int(class_id))
    if isinstance(class_id, str) and len(class_id.strip()) == 1 and class_id.strip().isalpha():
        return class_id.strip().upper()
    raise ValueError(f"Unsupported alphabet class ID in checkpoint: {class_id!r}")


class KeypointClassifier:
    """MediaPipe Tasks hand detector plus a verified Random Forest classifier."""

    def __init__(
        self,
        checkpoints_dir: str | Path | None = None,
        confidence_threshold: float | None = None,
    ) -> None:
        # ``checkpoints_dir`` remains only as an explicit test override. Normal
        # application resolution always follows models.lock.json.
        self.checkpoints_dir = Path(checkpoints_dir).resolve() if checkpoints_dir else None
        self.confidence_threshold = float(
            app_config.ALPHABET_MIN_CONFIDENCE
            if confidence_threshold is None
            else confidence_threshold
        )
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("Alphabet confidence threshold must be in [0, 1].")

        self.model: Any | None = None
        self.hands: Any | None = None
        self.model_path: Path | None = None
        self.hand_model_path: Path | None = None
        self.class_ids: tuple[Any, ...] = ()
        self.labels: tuple[str, ...] = ()
        self._closed = False

    def _resolve_paths(self) -> tuple[Path, Path]:
        if self.checkpoints_dir is not None:
            model_path = self.checkpoints_dir / "keypoint_model.joblib"
            hand_path = self.checkpoints_dir / "hand_landmarker.task"
            if not model_path.is_file() or not hand_path.is_file():
                raise FileNotFoundError(
                    f"Expected keypoint_model.joblib and hand_landmarker.task in {self.checkpoints_dir}."
                )
            verify_sha256(model_path, get_asset("alphabet_keypoint_rf").sha256)
            verify_sha256(hand_path, get_asset("hand_landmarker").sha256)
            return model_path, hand_path
        return resolve_asset("alphabet_keypoint_rf"), resolve_asset("hand_landmarker")

    def load_model(self) -> None:
        if self.model is not None and self.hands is not None:
            return
        if self._closed:
            raise RuntimeError("Cannot reload a closed KeypointClassifier instance.")

        model_path, hand_path = self._resolve_paths()
        model = joblib.load(model_path)
        self._validate_estimator(model)

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(hand_path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.55,
            min_hand_presence_confidence=0.55,
        )
        hands = mp.tasks.vision.HandLandmarker.create_from_options(options)

        self.model = model
        self.hands = hands
        self.model_path = model_path
        self.hand_model_path = hand_path
        self.class_ids = tuple(model.classes_.tolist())
        self.labels = tuple(_class_id_to_label(class_id) for class_id in self.class_ids)
        self.self_test()

    @staticmethod
    def _validate_estimator(model: Any) -> None:
        required = ("predict", "predict_proba", "classes_", "n_features_in_")
        missing = [name for name in required if not hasattr(model, name)]
        if missing:
            raise ValueError(f"Alphabet checkpoint is not a compatible classifier; missing {missing}.")
        if int(model.n_features_in_) != EXPECTED_FEATURES:
            raise ValueError(
                f"Alphabet checkpoint expects {model.n_features_in_} features; expected {EXPECTED_FEATURES}."
            )
        classes = tuple(model.classes_.tolist())
        if len(classes) != 26 or len(set(classes)) != 26:
            raise ValueError(f"Alphabet checkpoint must contain 26 unique classes; got {classes!r}.")
        normalized = tuple(int(value) for value in classes)
        if normalized != EXPECTED_CLASS_IDS:
            raise ValueError(
                "Alphabet checkpoint classes must be the ordered IDs 0..25; "
                f"got {normalized!r}."
            )
        # Validate that every exact estimator class ID can be represented.
        tuple(_class_id_to_label(value) for value in classes)

    @staticmethod
    def pre_process_landmark(landmark_list: Sequence[Sequence[float]]) -> np.ndarray:
        points = np.asarray(landmark_list, dtype=np.float32)
        if points.shape != (EXPECTED_LANDMARKS, 2):
            raise ValueError(
                f"Expected landmark shape ({EXPECTED_LANDMARKS}, 2), got {points.shape}."
            )
        if not np.isfinite(points).all():
            raise ValueError("Hand landmarks contain NaN or infinity.")

        relative = points - points[0]
        flat = relative.reshape(EXPECTED_FEATURES)
        scale = float(np.max(np.abs(flat)))
        if scale <= np.finfo(np.float32).eps:
            raise ValueError("Degenerate hand landmarks cannot be normalized.")
        normalized = flat / scale
        if not np.isfinite(normalized).all() or np.max(np.abs(normalized)) > 1.00001:
            raise ValueError("Normalized hand landmarks violated the [-1, 1] feature contract.")
        return normalized.astype(np.float32, copy=False)

    @staticmethod
    def _validate_frame(frame_bgr: np.ndarray) -> np.ndarray:
        if not isinstance(frame_bgr, np.ndarray):
            raise TypeError("frame_bgr must be a numpy array.")
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError(f"Expected a BGR frame shaped (H, W, 3), got {frame_bgr.shape}.")
        if frame_bgr.size == 0 or frame_bgr.shape[0] < 2 or frame_bgr.shape[1] < 2:
            raise ValueError("Frame is empty or too small for hand detection.")
        if frame_bgr.dtype != np.uint8:
            raise ValueError(f"Expected uint8 BGR pixels, got {frame_bgr.dtype}.")
        return np.ascontiguousarray(frame_bgr)

    def _predict_features(self, features: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
        if self.model is None:
            raise RuntimeError("Alphabet model is not loaded.")
        matrix = np.asarray(features, dtype=np.float32).reshape(1, EXPECTED_FEATURES)
        probabilities = np.asarray(self.model.predict_proba(matrix), dtype=np.float64)
        if probabilities.shape != (1, len(self.class_ids)):
            raise ValueError(
                f"Alphabet probability shape {probabilities.shape} does not match "
                f"{len(self.class_ids)} estimator classes."
            )
        probabilities = probabilities[0]
        if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
            raise ValueError("Alphabet checkpoint returned invalid probabilities.")
        if not np.isclose(float(probabilities.sum()), 1.0, atol=1e-5):
            raise ValueError("Alphabet checkpoint probabilities do not sum to one.")

        top_indices = np.argsort(probabilities)[::-1][:5]
        top5 = [
            {
                "label": _class_id_to_label(self.class_ids[int(index)]),
                "class_id": int(self.class_ids[int(index)]),
                "confidence": float(probabilities[int(index)]),
            }
            for index in top_indices
        ]
        return probabilities, top5

    def _acceptance(self, label: str, confidence: float) -> tuple[bool, str]:
        # J and Z are trajectories in ASL. A single-frame model can emit them as
        # candidates, but accepting them would make a claim the input cannot
        # support scientifically.
        if label in DYNAMIC_LETTERS:
            return False, "dynamic_letter_requires_motion"
        if confidence < self.confidence_threshold:
            return False, "low_confidence"
        return True, "accepted"

    def predict_frame(self, frame_bgr: np.ndarray) -> dict[str, Any] | None:
        if self.model is None or self.hands is None:
            self.load_model()
        assert self.hands is not None

        started = time.perf_counter()
        frame = self._validate_frame(frame_bgr)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        detection = self.hands.detect(image)
        detect_ms = (time.perf_counter() - started) * 1000.0
        if not detection.hand_landmarks:
            return None

        landmarks = detection.hand_landmarks[0]
        if len(landmarks) != EXPECTED_LANDMARKS:
            raise ValueError(
                f"MediaPipe returned {len(landmarks)} landmarks; expected {EXPECTED_LANDMARKS}."
            )
        normalized_points = np.asarray([[point.x, point.y] for point in landmarks], dtype=np.float32)
        landmark_points = np.asarray(
            [[point.x, point.y, getattr(point, "z", 0.0)] for point in landmarks], dtype=np.float32
        )
        features = self.pre_process_landmark(normalized_points)

        classified_at = time.perf_counter()
        probabilities, top5 = self._predict_features(features)
        classify_ms = (time.perf_counter() - classified_at) * 1000.0
        best = top5[0]
        confidence = float(best["confidence"])
        accepted, reason = self._acceptance(str(best["label"]), confidence)

        xs = np.clip(normalized_points[:, 0], 0.0, 1.0)
        ys = np.clip(normalized_points[:, 1], 0.0, 1.0)
        xmin, xmax = float(xs.min()), float(xs.max())
        ymin, ymax = float(ys.min()), float(ys.max())
        box_width, box_height = xmax - xmin, ymax - ymin
        box = [xmin + box_width / 2.0, ymin + box_height / 2.0, box_width, box_height]

        return {
            "label": best["label"],
            "class_id": best["class_id"],
            "confidence": confidence,
            "accepted": accepted,
            "reason": reason,
            "box": box,
            "top5": top5,
            "probabilities": probabilities.astype(float).tolist(),
            "landmarks": landmark_points.astype(float).tolist(),
            "handedness": self._handedness(detection),
            "latency": {
                "detect_ms": round(detect_ms, 2),
                "classify_ms": round(classify_ms, 2),
                "total_ms": round((time.perf_counter() - started) * 1000.0, 2),
            },
        }

    @staticmethod
    def _handedness(detection: Any) -> str | None:
        try:
            return str(detection.handedness[0][0].category_name)
        except (AttributeError, IndexError, TypeError):
            return None

    def self_test(self) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Load the alphabet model before running its self-test.")
        # Non-degenerate deterministic points exercise normalization, estimator
        # shape, class mapping, probability finiteness, and top-k generation.
        points = np.stack(
            [np.linspace(0.0, 1.0, EXPECTED_LANDMARKS), np.linspace(1.0, 0.0, EXPECTED_LANDMARKS)],
            axis=1,
        )
        _, top5 = self._predict_features(self.pre_process_landmark(points))
        if len(top5) != 5 or any(item["label"] not in self.labels for item in top5):
            raise ValueError("Alphabet self-test produced an invalid top-5 result.")
        return {"status": "ok", "features": EXPECTED_FEATURES, "classes": len(self.labels)}

    def metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "name": "asl_alphabet_keypoint_random_forest",
            "loaded": self.model is not None and self.hands is not None,
            "features": EXPECTED_FEATURES,
            "landmarks": EXPECTED_LANDMARKS,
            "confidence_threshold": self.confidence_threshold,
            "classes": list(self.labels),
            "accepted_static_classes": [label for label in self.labels if label not in DYNAMIC_LETTERS],
            "dynamic_classes_requiring_motion": sorted(DYNAMIC_LETTERS),
            "threshold_status": "provisional_cross_domain_audit",
            "preprocessing": "wrist-relative_xy_maxabs_v1",
        }
        if self.model_path is not None:
            metadata["classifier_asset"] = asset_metadata("alphabet_keypoint_rf", self.model_path)
        if self.hand_model_path is not None:
            metadata["detector_asset"] = asset_metadata("hand_landmarker", self.hand_model_path)
        return metadata

    def close(self) -> None:
        if self.hands is not None:
            close = getattr(self.hands, "close", None)
            if callable(close):
                close()
        self.hands = None
        self.model = None
        self._closed = True

    def __enter__(self) -> "KeypointClassifier":
        self.load_model()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
