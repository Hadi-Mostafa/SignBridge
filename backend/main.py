"""FastAPI application for bounded sign-language inference."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Existing modules use top-level imports when the application is launched as
# ``python -m uvicorn backend.main:app``. Keep that mode compatible.
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

if __package__:
    from .config import (  # noqa: E402
        API_HOST,
        API_PORT,
        ALPHABET_MIN_CONFIDENCE,
        CORS_ALLOWED_ORIGINS,
        LANDMARK_MAX_VALUE,
        LANDMARK_MIN_VALUE,
        LANDMARK_VALUES_PER_FRAME,
        MAX_BASE64_CHARS,
        PROJECT_ROOT,
        WLASL_SIGN_VIDEOS_DIR,
        WORD_SEQUENCE_LENGTH,
        WS_ALLOWED_ORIGINS,
        WS_MAX_MESSAGE_BYTES,
    )
    from .models.word_model import WordModel  # noqa: E402
    from .routers import sign_to_text, speech_to_sign, word_suggestions  # noqa: E402
    from .services.alphabet_session import AlphabetSessionDecoder  # noqa: E402
    from .utils.video_utils import ImageValidationError, decode_base64_image  # noqa: E402
else:  # pragma: no cover - direct ``python backend/main.py`` compatibility
    from config import (  # noqa: E402
        API_HOST,
        API_PORT,
        ALPHABET_MIN_CONFIDENCE,
        CORS_ALLOWED_ORIGINS,
        LANDMARK_MAX_VALUE,
        LANDMARK_MIN_VALUE,
        LANDMARK_VALUES_PER_FRAME,
        MAX_BASE64_CHARS,
        PROJECT_ROOT,
        WLASL_SIGN_VIDEOS_DIR,
        WORD_SEQUENCE_LENGTH,
        WS_ALLOWED_ORIGINS,
        WS_MAX_MESSAGE_BYTES,
    )
    from models.word_model import WordModel  # noqa: E402
    from routers import sign_to_text, speech_to_sign, word_suggestions  # noqa: E402
    from services.alphabet_session import AlphabetSessionDecoder  # noqa: E402
    from utils.video_utils import ImageValidationError, decode_base64_image  # noqa: E402

logger = logging.getLogger(__name__)

word_model: WordModel | None = None
word_model_error: str | None = None


def _public_error(exc: BaseException) -> str:
    """Return capability-safe startup context without paths or remote details."""

    return f"{type(exc).__name__}: model initialization failed; see server logs"


def _model_metadata(instance: Any) -> dict:
    if instance is None:
        return {}
    provider = getattr(instance, "metadata", None)
    try:
        value = provider() if callable(provider) else provider
    except Exception:
        logger.exception("Model metadata failed")
        return {}
    return _public_metadata(value) if isinstance(value, dict) else {}


def _public_metadata(value: Any) -> Any:
    """Remove local filesystem details before readiness metadata leaves the API."""

    if isinstance(value, dict):
        return {
            str(key): _public_metadata(item)
            for key, item in value.items()
            if str(key).lower() not in {"path", "manifest_path", "source_manifest"}
            and not str(key).lower().endswith("_path")
        }
    if isinstance(value, list):
        return [_public_metadata(item) for item in value]
    if isinstance(value, tuple):
        return [_public_metadata(item) for item in value]
    return value


def _self_test_passed(result: Any) -> bool:
    if result is False:
        return False
    if isinstance(result, dict):
        for key in ("ok", "passed", "ready"):
            if key in result and not bool(result[key]):
                return False
    return True


def _load_word_model() -> WordModel:
    candidate = WordModel()
    self_test = getattr(candidate, "self_test", None)
    if callable(self_test) and not _self_test_passed(self_test()):
        raise RuntimeError("Word model self-test failed.")
    return candidate


@asynccontextmanager
async def lifespan(app: FastAPI):
    global word_model, word_model_error
    logger.info("Starting sign-language application")

    try:
        await asyncio.to_thread(sign_to_text.initialize_services)
    except Exception as exc:
        logger.exception("Alphabet pipeline failed to load")
        # The router preserves the same sanitized load error for readiness.
        if sign_to_text.pipeline_error is None:
            sign_to_text.pipeline_error = _public_error(exc)

    try:
        word_model = await asyncio.to_thread(_load_word_model)
        word_model_error = None
        logger.info("Word model loaded")
    except Exception as exc:
        word_model = None
        word_model_error = _public_error(exc)
        logger.exception("Word model failed to load")

    # Text/lookup services are lightweight and should be ready with the page.
    # Whisper weights still load only for an explicit voice request and never
    # download from inside that request.
    try:
        await asyncio.to_thread(speech_to_sign.initialize_services)
    except Exception:
        logger.exception("Text/voice-to-sign services failed to initialize")

    try:
        await asyncio.to_thread(word_suggestions.initialize_service)
    except Exception:
        logger.exception("Context-aware word suggestion service failed to initialize")

    logger.info("Application startup completed")
    try:
        yield
    finally:
        await asyncio.to_thread(sign_to_text.close_services)
        current_word, word_model = word_model, None
        close_word = getattr(current_word, "close", None)
        if callable(close_word):
            try:
                await asyncio.to_thread(close_word)
            except Exception:
                logger.exception("Word model shutdown failed")
        logger.info("Application shutdown completed")


app = FastAPI(title="Sign Language Translation System", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(CORS_ALLOWED_ORIGINS),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Accept"],
)
app.include_router(sign_to_text.router, tags=["Sign to Text"])
app.include_router(speech_to_sign.router, tags=["Speech to Sign"])
app.include_router(word_suggestions.router, tags=["Sign to Text"])


def _word_capability() -> dict:
    metadata = _model_metadata(word_model)
    metadata["experimental"] = True
    return {
        "ready": word_model is not None,
        "status": "ready" if word_model is not None else "unavailable",
        "error": word_model_error,
        "metadata": metadata,
        "mode": "words",
    }


def _capabilities() -> dict:
    alphabet = dict(sign_to_text.capability_status())
    alphabet["metadata"] = _public_metadata(alphabet.get("metadata", {}))
    return {
        "alphabet": alphabet,
        "words": _word_capability(),
        "word_suggestions": word_suggestions.capability_status(),
        "speech_to_sign": speech_to_sign.capability_status(),
    }


def _request_id(message: dict | None) -> str:
    value = message.get("request_id") if isinstance(message, dict) else None
    if isinstance(value, str) and 0 < len(value) <= 128:
        return value
    return uuid.uuid4().hex


def _ws_envelope(
    message_type: str,
    request_id: str,
    *,
    accepted: bool = False,
    reason: str,
    confidence: float = 0.0,
    top_predictions: list | None = None,
    server_latency_ms: float = 0.0,
    **extra: Any,
) -> dict:
    predictions = top_predictions or []
    return {
        "type": message_type,
        "request_id": request_id,
        "accepted": bool(accepted),
        "reason": reason,
        "confidence": round(float(confidence), 4),
        "top_predictions": predictions,
        "top5": predictions,
        "server_latency_ms": round(float(server_latency_ms), 1),
        **extra,
    }


async def _send_ws_error(
    websocket: WebSocket,
    request_id: str,
    code: str,
    message: str,
    *,
    server_latency_ms: float = 0.0,
) -> None:
    await websocket.send_json(
        _ws_envelope(
            "error",
            request_id,
            reason=code,
            server_latency_ms=server_latency_ms,
            error={"code": code, "message": message},
        )
    )


def _origin_allowed(websocket: WebSocket) -> bool:
    origin = websocket.headers.get("origin", "").rstrip("/")
    return bool(origin and origin in WS_ALLOWED_ORIGINS)


def _validate_landmarks(value: Any) -> list[float]:
    if not isinstance(value, list) or len(value) != LANDMARK_VALUES_PER_FRAME:
        raise ValueError(f"Expected {LANDMARK_VALUES_PER_FRAME} landmark values.")
    normalized: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError("Landmarks must contain only numbers.")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError("Landmarks must be finite.")
        if number < LANDMARK_MIN_VALUE or number > LANDMARK_MAX_VALUE:
            raise ValueError(
                f"Landmarks must be between {LANDMARK_MIN_VALUE} and {LANDMARK_MAX_VALUE}."
            )
        normalized.append(number)
    return normalized


def _prediction_response(result: dict, mode: str, request_id: str, elapsed_ms: float) -> dict:
    accepted = bool(result.get("accepted", False))
    candidate = result.get("candidate_label", result.get("label", result.get("sign")))
    confidence = float(result.get("confidence", 0.0))
    top_predictions = result.get("top_predictions", result.get("top5", [])) or []
    reason = result.get("reason") or ("accepted" if accepted else "below_confidence_threshold")
    latency = dict(result.get("latency", {})) if isinstance(result.get("latency"), dict) else {}
    latency["server_total_ms"] = round(elapsed_ms, 1)
    return _ws_envelope(
        "prediction",
        request_id,
        accepted=accepted,
        reason=str(reason),
        confidence=confidence,
        top_predictions=top_predictions,
        server_latency_ms=elapsed_ms,
        mode=mode,
        label=candidate if accepted else None,
        sign=candidate if accepted else None,
        result=candidate if accepted else None,
        candidate_label=candidate,
        box=result.get("box"),
        landmarks=result.get("landmarks"),
        handedness=result.get("handedness"),
        detector=result.get("detector"),
        model=result.get("model"),
        margin=result.get("margin"),
        quality=result.get("quality"),
        motion=result.get("motion"),
        latency=latency,
    )


@app.websocket("/ws/sign-to-text")
async def sign_to_text_ws(websocket: WebSocket):
    if not _origin_allowed(websocket):
        await websocket.close(code=1008, reason="Origin not allowed")
        return
    await websocket.accept()

    connection_id = uuid.uuid4().hex
    await websocket.send_json(
        _ws_envelope(
            "status",
            connection_id,
            reason="connected",
            status="connected",
            capabilities=_capabilities(),
        )
    )

    landmark_buffer: list[list[float]] = []
    alphabet_decoder = AlphabetSessionDecoder(ALPHABET_MIN_CONFIDENCE)
    word_capture_active = False
    word_awaiting_reset = False
    word_capture_request_id: str | None = None

    try:
        while True:
            try:
                raw = await websocket.receive_text()
            except WebSocketDisconnect:
                break
            except Exception:
                logger.info("WebSocket receive failed", exc_info=True)
                break

            if len(raw.encode("utf-8")) > WS_MAX_MESSAGE_BYTES:
                await _send_ws_error(
                    websocket,
                    connection_id,
                    "message_too_large",
                    f"WebSocket messages are limited to {WS_MAX_MESSAGE_BYTES} bytes.",
                )
                await websocket.close(code=1009, reason="Message too large")
                return

            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                await _send_ws_error(websocket, connection_id, "invalid_json", "Message must be valid JSON.")
                continue
            if not isinstance(message, dict):
                await _send_ws_error(websocket, connection_id, "invalid_message", "Message must be a JSON object.")
                continue

            request_id = _request_id(message)
            message_type = message.get("type")
            mode = message.get("mode")

            if message_type == "reset":
                landmark_buffer.clear()
                alphabet_decoder.reset()
                word_capture_active = False
                word_awaiting_reset = False
                word_capture_request_id = None
                await websocket.send_json(
                    _ws_envelope(
                        "status",
                        request_id,
                        reason="reset",
                        status="reset",
                        mode=mode,
                        captured=0,
                        total=WORD_SEQUENCE_LENGTH,
                        awaiting_reset=False,
                    )
                )
                continue

            if mode not in {"alphabet", "words"}:
                await _send_ws_error(websocket, request_id, "invalid_mode", "Mode must be 'alphabet' or 'words'.")
                continue

            if mode == "alphabet":
                if message_type != "frame":
                    await _send_ws_error(websocket, request_id, "invalid_message_type", "Alphabet mode expects a frame.")
                    continue
                if sign_to_text.get_pipeline(required=False) is None:
                    await _send_ws_error(
                        websocket, request_id, "alphabet_unavailable", "Alphabet recognition is unavailable."
                    )
                    continue
                image_b64 = message.get("image")
                if not isinstance(image_b64, str) or len(image_b64) > MAX_BASE64_CHARS:
                    await _send_ws_error(
                        websocket,
                        request_id,
                        "invalid_image",
                        f"Base64 image limit is {MAX_BASE64_CHARS} characters.",
                    )
                    continue
                started = time.perf_counter()
                try:
                    frame = await asyncio.to_thread(decode_base64_image, image_b64)
                    result = await asyncio.to_thread(sign_to_text.predict_frame, frame)
                except ImageValidationError as exc:
                    await _send_ws_error(
                        websocket,
                        request_id,
                        "invalid_image",
                        str(exc),
                        server_latency_ms=(time.perf_counter() - started) * 1000,
                    )
                    continue
                except Exception:
                    logger.exception("Alphabet WebSocket inference failed")
                    await _send_ws_error(
                        websocket,
                        request_id,
                        "alphabet_inference_failed",
                        "Alphabet inference failed.",
                        server_latency_ms=(time.perf_counter() - started) * 1000,
                    )
                    continue

                elapsed_ms = (time.perf_counter() - started) * 1000
                if result is None:
                    alphabet_decoder.reset()
                    await websocket.send_json(
                        _ws_envelope(
                            "no_hand",
                            request_id,
                            reason="no_hand",
                            server_latency_ms=elapsed_ms,
                            mode="alphabet",
                            label=None,
                            sign=None,
                            result=None,
                            candidate_label=None,
                            latency={"server_total_ms": round(elapsed_ms, 1)},
                        )
                    )
                else:
                    result = alphabet_decoder.update(dict(result))
                    await websocket.send_json(
                        _prediction_response(dict(result), "alphabet", request_id, elapsed_ms)
                    )
                continue

            # Word capture control messages never enter the frame buffer.
            if message_type == "capture_start":
                if word_model is None:
                    await _send_ws_error(websocket, request_id, "word_unavailable", "Word recognition is unavailable.")
                    continue
                if word_awaiting_reset:
                    await _send_ws_error(websocket, request_id, "reset_required", "Reset before starting another capture.")
                    continue
                if message.get("total", WORD_SEQUENCE_LENGTH) != WORD_SEQUENCE_LENGTH:
                    await _send_ws_error(
                        websocket,
                        request_id,
                        "invalid_sequence_length",
                        f"Word capture requires exactly {WORD_SEQUENCE_LENGTH} frames.",
                    )
                    continue
                landmark_buffer.clear()
                word_capture_active = True
                word_capture_request_id = request_id
                await websocket.send_json(
                    _ws_envelope(
                        "progress",
                        request_id,
                        reason="capture_started",
                        mode="words",
                        captured=0,
                        total=WORD_SEQUENCE_LENGTH,
                        awaiting_reset=False,
                    )
                )
                continue

            if message_type == "capture_end":
                if word_awaiting_reset:
                    await websocket.send_json(
                        _ws_envelope(
                            "status",
                            request_id,
                            reason="capture_complete",
                            mode="words",
                            status="awaiting_reset",
                            captured=WORD_SEQUENCE_LENGTH,
                            total=WORD_SEQUENCE_LENGTH,
                            awaiting_reset=True,
                        )
                    )
                elif len(landmark_buffer) != WORD_SEQUENCE_LENGTH:
                    await _send_ws_error(
                        websocket,
                        request_id,
                        "incomplete_sequence",
                        f"Captured {len(landmark_buffer)} of {WORD_SEQUENCE_LENGTH} frames.",
                    )
                continue

            if message_type != "frame":
                await _send_ws_error(websocket, request_id, "invalid_message_type", "Words mode expects capture control or frame messages.")
                continue
            if word_model is None:
                await _send_ws_error(websocket, request_id, "word_unavailable", "Word recognition is unavailable.")
                continue
            if word_awaiting_reset:
                await _send_ws_error(websocket, request_id, "reset_required", "Reset before sending another word sequence.")
                continue
            if not word_capture_active:
                # Preserve compatibility with older clients that send frames directly.
                word_capture_active = True
                word_capture_request_id = request_id
                landmark_buffer.clear()
            if word_capture_request_id and request_id != word_capture_request_id:
                await _send_ws_error(websocket, request_id, "capture_id_mismatch", "Frame request_id does not match the active capture.")
                continue
            frame_index = message.get("frame_index")
            if frame_index is not None and (isinstance(frame_index, bool) or frame_index != len(landmark_buffer)):
                await _send_ws_error(
                    websocket,
                    request_id,
                    "frame_out_of_order",
                    f"Expected frame_index {len(landmark_buffer)}.",
                )
                continue
            try:
                landmarks = _validate_landmarks(message.get("landmarks"))
            except ValueError as exc:
                await _send_ws_error(websocket, request_id, "invalid_landmarks", str(exc))
                continue

            landmark_buffer.append(landmarks)
            captured = len(landmark_buffer)
            await websocket.send_json(
                _ws_envelope(
                    "progress",
                    request_id,
                    reason="capturing" if captured < WORD_SEQUENCE_LENGTH else "capture_complete",
                    mode="words",
                    captured=captured,
                    total=WORD_SEQUENCE_LENGTH,
                    awaiting_reset=captured == WORD_SEQUENCE_LENGTH,
                )
            )
            if captured < WORD_SEQUENCE_LENGTH:
                continue

            word_capture_active = False
            word_awaiting_reset = True
            sequence = list(landmark_buffer)
            started = time.perf_counter()
            try:
                result = await asyncio.to_thread(word_model.predict_word, sequence)
            except Exception:
                logger.exception("Word WebSocket inference failed")
                await _send_ws_error(
                    websocket,
                    request_id,
                    "word_inference_failed",
                    "Word inference failed; reset before retrying.",
                    server_latency_ms=(time.perf_counter() - started) * 1000,
                )
                continue
            elapsed_ms = (time.perf_counter() - started) * 1000
            await websocket.send_json(_prediction_response(dict(result), "words", request_id, elapsed_ms))

    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("Unexpected WebSocket failure")


FRONTEND_DIR = PROJECT_ROOT / "frontend"
if WLASL_SIGN_VIDEOS_DIR.exists():
    app.mount(
        "/sign-assets",
        StaticFiles(directory=str(WLASL_SIGN_VIDEOS_DIR), check_dir=True),
        name="sign-assets",
    )
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
async def serve_frontend():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return JSONResponse({"message": "Frontend not found"}, status_code=404)


@app.get("/health/live")
async def liveness_check():
    return {"status": "alive"}


@app.get("/health")
async def health_check():
    capabilities = _capabilities()
    alphabet_ready = bool(capabilities["alphabet"]["ready"])
    words_ready = bool(capabilities["words"]["ready"])
    status = "not_ready" if not alphabet_ready else ("ready" if words_ready else "degraded")
    response = {
        "status": status,
        "ready": alphabet_ready,
        "capabilities": capabilities,
        "modes": {
            "alphabet": alphabet_ready,
            "words": words_ready,
            "word_suggestions": bool(capabilities["word_suggestions"]["ready"]),
            "speech_to_sign": bool(capabilities["speech_to_sign"]["ready"]),
        },
        # Compatibility fields for older clients.
        "alphabet_model": alphabet_ready,
        "word_model": words_ready,
    }
    return JSONResponse(response, status_code=200 if alphabet_ready else 503)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
        log_level="info",
    )
