"""Hand localisation for the realtime ASL pipeline.

YOLO is used only with a checkpoint trained with a ``hand`` class.  The
generic COCO checkpoint does not recognise hands and is therefore never used
as a misleading substitute.  MediaPipe is retained solely as a local fallback
so the application remains usable before custom YOLO weights are trained.
"""

from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

try:
    from ..config import HAND_YOLO_CHECKPOINT, YOLO_CONFIDENCE_THRESHOLD, YOLO_NMS_IOU
except ImportError:  # pragma: no cover - direct backend-path execution
    from config import HAND_YOLO_CHECKPOINT, YOLO_CONFIDENCE_THRESHOLD, YOLO_NMS_IOU


class HandDetector:
    def __init__(self, weights: Path = HAND_YOLO_CHECKPOINT):
        self.weights = Path(weights)
        self.model = None
        self.backend = "mediapipe-fallback"
        task_path = next((Path(__file__).resolve().parent.parent / "checkpoints").rglob("hand_landmarker.task"), None)
        if task_path is None:
            raise FileNotFoundError("MediaPipe hand-landmarker.task is missing.")
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(task_path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.55,
            min_hand_presence_confidence=0.55,
        )
        self.fallback = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def load(self) -> None:
        if not self.weights.exists():
            print(f"[HandDetector] Hand YOLO weights not found: {self.weights}; using MediaPipe fallback.")
            return
        from ultralytics import YOLO
        model = YOLO(str(self.weights))
        labels = {str(name).lower() for name in model.names.values()}
        if not any("hand" in label for label in labels):
            raise ValueError(f"{self.weights} has no hand class: {labels}")
        self.model = model
        self.backend = "yolo11"
        print(f"[HandDetector] Loaded hand-trained YOLO checkpoint: {self.weights}")

    def detect(self, frame_bgr: np.ndarray) -> Optional[dict]:
        if self.model is not None:
            result = self.model.predict(frame_bgr, conf=YOLO_CONFIDENCE_THRESHOLD, iou=YOLO_NMS_IOU, imgsz=416, verbose=False)[0]
            if len(result.boxes):
                index = int(result.boxes.conf.argmax())
                box = result.boxes[index]
                x, y, w, h = box.xywhn[0].cpu().numpy().tolist()
                return {"box": [float(x), float(y), float(w), float(h)], "detector_confidence": float(box.conf)}

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.fallback.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not result.hand_landmarks:
            return None
        points = result.hand_landmarks[0]
        xs, ys = [point.x for point in points], [point.y for point in points]
        xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
        return {"box": [(xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin], "detector_confidence": 1.0}
