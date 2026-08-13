"""Compatibility adapter for the canonical alphabet inference service.

The former implementation used generic Hugging Face Auto classes for a custom
29-class ASLResNet checkpoint.  That loader was incompatible and is retired.
New code should import :class:`services.realtime_pipeline.RealtimeASLPipeline`
directly; this adapter remains only for callers of the old class name.
"""

from __future__ import annotations

try:
    from ..services.realtime_pipeline import RealtimeASLPipeline
    from ..utils.video_utils import decode_image_bytes
except ImportError:  # pragma: no cover - direct backend-path execution
    from services.realtime_pipeline import RealtimeASLPipeline
    from utils.video_utils import decode_image_bytes


class AlphabetModel:
    """Deprecated wrapper around the verified static-alphabet pipeline."""

    def __init__(self) -> None:
        self.pipeline = RealtimeASLPipeline()
        self.pipeline.load()

    def predict_alphabet(self, image_bytes: bytes) -> dict:
        frame = decode_image_bytes(image_bytes)
        result = self.pipeline.predict(frame, smooth=False)
        if result is None:
            return {
                "label": None,
                "candidate_label": None,
                "confidence": 0.0,
                "accepted": False,
                "reason": "no_hand",
                "top5": [],
            }
        return result

    def close(self) -> None:
        self.pipeline.close()

    def __enter__(self) -> "AlphabetModel":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
