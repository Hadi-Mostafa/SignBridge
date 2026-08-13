"""High-accuracy ASL alphabet image classifier.

The bundled checkpoint is a ResNet-18 trained for the 26 static ASL alphabet
classes.  It is deliberately kept separate from word-level recognition:
alphabet signs are static images, while words and emotions need temporal video
data and should not be guessed by an alphabet model.
"""

from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, resnet18
try:
    from ..config import ALPHABET_CHECKPOINT
except ImportError:  # pragma: no cover - direct backend-path execution
    from config import ALPHABET_CHECKPOINT


class AlphabetClassifier:
    """Classify a hand crop as an ASL alphabet letter (A–Z)."""

    LABELS = [chr(ord("A") + index) for index in range(26)]

    def __init__(self, checkpoint_path: str | Path | None = None):
        backend_dir = Path(__file__).resolve().parent.parent
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else (
            ALPHABET_CHECKPOINT if ALPHABET_CHECKPOINT.exists() else next(
                (backend_dir / "checkpoints").rglob("best_model.pth"), None
            )
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[nn.Module] = None

    def load_model(self) -> None:
        if self.checkpoint_path is None or not self.checkpoint_path.exists():
            raise FileNotFoundError("ASL alphabet checkpoint best_model.pth was not found.")

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        architecture = checkpoint.get("architecture", "legacy_resnet18") if isinstance(checkpoint, dict) else "legacy_resnet18"
        if architecture == "mobilenet_v3_large":
            self.LABELS = checkpoint["labels"]
            model = mobilenet_v3_large(weights=None)
            model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(self.LABELS))
            state_dict = checkpoint["model_state_dict"]
        else:
            model = resnet18(weights=None)
            model.fc = nn.Sequential(
                nn.Dropout(0.5), nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, 26)
            )
            state_dict = {key.removeprefix("model."): value for key, value in checkpoint["model_state_dict"].items()}
        model.load_state_dict(state_dict)
        self.model = model.to(self.device).eval()
        print(f"[AlphabetClassifier] Loaded ResNet-18 alphabet model from {self.checkpoint_path}")

    def predict_frame(self, frame_bgr: np.ndarray, box: Optional[list[float]] = None) -> dict[str, Any]:
        if self.model is None:
            self.load_model()

        image = self._crop_hand(frame_bgr, box)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(image).permute(2, 0, 1).float().div(255.0)
        # Standard ImageNet normalization used with ResNet-18 fine-tuning.
        tensor = (tensor - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / torch.tensor(
            [0.229, 0.224, 0.225]
        ).view(3, 1, 1)

        with torch.inference_mode():
            probabilities = torch.softmax(self.model(tensor.unsqueeze(0).to(self.device))[0], dim=0)
        top_indices = torch.topk(probabilities, k=5).indices.cpu().tolist()
        top5 = [{"label": self.LABELS[index], "confidence": float(probabilities[index])} for index in top_indices]
        return {"label": top5[0]["label"], "confidence": top5[0]["confidence"], "top5": top5}

    @staticmethod
    def _crop_hand(frame: np.ndarray, box: Optional[list[float]]) -> np.ndarray:
        """Use the landmark box with generous context, or whole frame as fallback."""
        if not box:
            return frame
        height, width = frame.shape[:2]
        center_x, center_y, box_w, box_h = box
        box_w = max(box_w * 2.2, 0.18)
        box_h = max(box_h * 2.2, 0.18)
        left = max(0, int((center_x - box_w / 2) * width))
        right = min(width, int((center_x + box_w / 2) * width))
        top = max(0, int((center_y - box_h / 2) * height))
        bottom = min(height, int((center_y + box_h / 2) * height))
        return frame[top:bottom, left:right] if right > left and bottom > top else frame
