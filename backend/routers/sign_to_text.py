"""Bounded REST access to the shared realtime alphabet pipeline."""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile

try:
    from ..config import ALLOWED_IMAGE_CONTENT_TYPES, MAX_IMAGE_BYTES
    from ..services.realtime_pipeline import RealtimeASLPipeline
    from ..utils.latency_logger import LatencyLogger
    from ..utils.video_utils import ImageValidationError, decode_image_bytes
except ImportError:  # pragma: no cover - direct backend-path execution
    from config import ALLOWED_IMAGE_CONTENT_TYPES, MAX_IMAGE_BYTES
    from services.realtime_pipeline import RealtimeASLPipeline
    from utils.latency_logger import LatencyLogger
    from utils.video_utils import ImageValidationError, decode_image_bytes

logger = logging.getLogger(__name__)
router = APIRouter()

pipeline: RealtimeASLPipeline | None = None
pipeline_error: str | None = None
latency_logger: LatencyLogger | None = None
_state_lock = threading.RLock()
_inference_lock = threading.Lock()


def _public_error(exc: BaseException) -> str:
    """Return capability-safe startup context without local paths."""

    return f"{type(exc).__name__}: model initialization failed; see server logs"


def _metadata(instance: Any) -> dict:
    if instance is None:
        return {}
    provider = getattr(instance, "metadata", None)
    try:
        value = provider() if callable(provider) else provider
    except Exception:
        logger.exception("Alphabet metadata failed")
        return {}
    return dict(value) if isinstance(value, dict) else {}


def initialize_services() -> RealtimeASLPipeline:
    """Load the one alphabet pipeline used by both REST and WebSocket APIs."""
    global pipeline, pipeline_error, latency_logger
    with _state_lock:
        if pipeline is not None:
            return pipeline
        candidate = RealtimeASLPipeline()
        try:
            candidate.load()
        except Exception as exc:
            pipeline_error = _public_error(exc)
            try:
                candidate.close()
            except Exception:
                logger.exception("Partially loaded alphabet pipeline did not close cleanly")
            raise
        pipeline = candidate
        pipeline_error = None
        latency_logger = latency_logger or LatencyLogger()
        logger.info("Alphabet pipeline loaded")
        return candidate


def close_services() -> None:
    global pipeline
    with _state_lock:
        current, pipeline = pipeline, None
    if current is not None:
        try:
            current.close()
        except Exception:
            logger.exception("Alphabet pipeline shutdown failed")


def get_pipeline(*, required: bool = True) -> RealtimeASLPipeline | None:
    current = pipeline
    if required and current is None:
        raise RuntimeError("Alphabet recognition is unavailable.")
    return current


def predict_frame(frame: np.ndarray) -> dict | None:
    current = get_pipeline(required=True)
    assert current is not None
    with _inference_lock:
        return current.predict(frame, smooth=False)


def capability_status() -> dict:
    current = pipeline
    metadata = _metadata(current)
    return {
        "ready": current is not None,
        "status": "ready" if current is not None else "unavailable",
        "error": pipeline_error,
        "metadata": metadata,
        "mode": "alphabet",
    }


def _set_pipeline_for_tests(instance: RealtimeASLPipeline | None, error: str | None = None) -> None:
    """Test hook; never used by application code."""
    global pipeline, pipeline_error, latency_logger
    pipeline = instance
    pipeline_error = error
    latency_logger = latency_logger or LatencyLogger()


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict one alphabet sign from a bounded JPEG, PNG, or WebP upload."""
    content_type = (file.content_type or "").split(";", 1)[0].lower()
    if content_type not in ALLOWED_IMAGE_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail={"code": "unsupported_image_type", "message": "Upload a JPEG, PNG, or WebP image."},
        )
    data = await file.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail={"code": "image_too_large", "message": f"Image limit is {MAX_IMAGE_BYTES} bytes."},
        )
    try:
        frame = await asyncio.to_thread(decode_image_bytes, data)
    except ImageValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_image", "message": str(exc)},
        ) from None
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail={"code": "alphabet_unavailable", "message": "Alphabet recognition is unavailable."},
        )
    try:
        result = await asyncio.to_thread(predict_frame, frame)
    except Exception:
        logger.exception("Alphabet REST inference failed")
        raise HTTPException(
            status_code=503,
            detail={"code": "alphabet_inference_failed", "message": "Alphabet inference failed."},
        ) from None
    if result is None:
        return {
            "type": "no_hand",
            "mode": "alphabet",
            "sign": None,
            "label": None,
            "accepted": False,
            "reason": "no_hand",
            "confidence": 0.0,
            "top_predictions": [],
            "top5": [],
        }

    response = dict(result)
    response.pop("probabilities", None)
    top_predictions = response.get("top_predictions", response.get("top5", []))
    accepted = bool(response.get("accepted", False))
    candidate = response.get("candidate_label", response.get("label", response.get("sign")))
    response.update(
        {
            "type": "prediction",
            "mode": "alphabet",
            "accepted": accepted,
            "reason": response.get("reason") or ("accepted" if accepted else "below_confidence_threshold"),
            "candidate_label": candidate,
            "label": candidate if accepted else None,
            "sign": candidate if accepted else None,
            "result": candidate if accepted else None,
            "top_predictions": top_predictions,
            "top5": top_predictions,
        }
    )
    return response
