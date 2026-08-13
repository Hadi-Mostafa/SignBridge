"""
YOLOv11 Sign Language Detection Service.

Wraps the Ultralytics YOLO model for real-time inference.
"""

import time
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

from ultralytics import YOLO

try:
    from ..config import YOLO_CHECKPOINT, YOLO_CONFIDENCE_THRESHOLD, YOLO_NMS_IOU
except ImportError:  # pragma: no cover - direct backend-path execution
    from config import YOLO_CHECKPOINT, YOLO_CONFIDENCE_THRESHOLD, YOLO_NMS_IOU

class YoloClassifier:
    """
    Real-time gesture detection from RGB frames using YOLO11.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model: Optional[YOLO] = None
        self.labels = {}
        self.last_inference_time = 0.0

    def load_model(self, checkpoint_path: Optional[str] = None):
        """
        Load YOLO model weights.
        """
        if checkpoint_path is None:
            checkpoint_path = str(YOLO_CHECKPOINT)

        if not Path(checkpoint_path).is_file():
            raise FileNotFoundError(
                f"Sign-specific YOLO checkpoint not found: {checkpoint_path}. "
                "A generic COCO yolo11n checkpoint is not a sign-language model."
            )

        # Initialize model
        self.model = YOLO(checkpoint_path)
        self.labels = self.model.names
        if not self.labels:
            self.model = None
            raise ValueError("YOLO checkpoint exposes no class labels.")
        
        # Warmup
        dummy_img = np.zeros((416, 416, 3), dtype=np.uint8)
        self.model.predict(dummy_img, device=self.device, verbose=False)
        print(f"[YoloClassifier] Loaded YOLO model from {checkpoint_path}")

    def predict_frame(self, frame_bgr: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Run inference on a single BGR frame.

        Returns:
            Dict with {label, confidence, box} or None if no detection.
        """
        if self.model is None:
            return None

        start_time = time.time()
        
        # Run inference
        results = self.model.predict(
            source=frame_bgr, 
            device=self.device, 
            conf=YOLO_CONFIDENCE_THRESHOLD,
            iou=YOLO_NMS_IOU,
            verbose=False,
            imgsz=416, # Optimized for speed
        )

        self.last_inference_time = (time.time() - start_time) * 1000

        # We only care about the first image in batch
        result = results[0]
        
        if len(result.boxes) == 0:
            return None
            
        # Get the box with highest confidence
        # result.boxes.conf is a tensor of shape (N,)
        best_idx = int(result.boxes.conf.argmax())
        best_box = result.boxes[best_idx]
        
        confidence = float(best_box.conf)
        class_id = int(best_box.cls)
        label = self.labels[class_id]
        
        # xywhn format: [x_center, y_center, width, height] normalized 0-1
        # using xywh normalizes it relative to image dims. We want relative.
        xywh = best_box.xywhn[0].cpu().numpy().tolist()
        
        return {
            "label": label,
            "confidence": confidence,
            "box": [round(val, 4) for val in xywh]
        }
