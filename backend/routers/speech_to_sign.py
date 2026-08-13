"""Bounded text/voice-to-sign endpoints with explicit visual fallbacks."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from typing import Any, List, Literal, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field, field_validator

try:
    from ..config import (
        ALLOWED_AUDIO_CONTENT_TYPES,
        ALLOWED_AUDIO_SUFFIXES,
        MAX_AUDIO_BYTES,
        MAX_AUDIO_DURATION_SECONDS,
        MAX_TEXT_CHARS,
    )
    from ..services.asr_service import (
        ASRAssetInvalidError,
        ASRAssetMissingError,
        ASRAudioInvalidError,
        ASRAudioTooLongError,
        ASRRuntimeMissingError,
        ASRService,
    )
    from ..services.nlp_processor import NLPProcessor
    from ..services.sign_lookup import SignLookupService
    from ..utils.latency_logger import LatencyLogger
except ImportError:  # pragma: no cover - direct backend-path execution
    from config import (
        ALLOWED_AUDIO_CONTENT_TYPES,
        ALLOWED_AUDIO_SUFFIXES,
        MAX_AUDIO_BYTES,
        MAX_AUDIO_DURATION_SECONDS,
        MAX_TEXT_CHARS,
    )
    from services.asr_service import (
        ASRAssetInvalidError,
        ASRAssetMissingError,
        ASRAudioInvalidError,
        ASRAudioTooLongError,
        ASRRuntimeMissingError,
        ASRService,
    )
    from services.nlp_processor import NLPProcessor
    from services.sign_lookup import SignLookupService
    from utils.latency_logger import LatencyLogger

logger = logging.getLogger(__name__)
router = APIRouter()

asr_service: ASRService | None = None
nlp_processor: NLPProcessor | None = None
sign_lookup: SignLookupService | None = None
latency_logger: LatencyLogger | None = None
service_error: str | None = None
_init_lock = threading.Lock()
_asr_capacity = threading.BoundedSemaphore(1)


class NormalizeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Text must not be blank.")
        return value.strip()


class TextToSignRequest(NormalizeRequest):
    pass


class TranscriptionResponse(BaseModel):
    text: str
    latency_ms: float


class Concept(BaseModel):
    gloss: str
    source_tokens: List[str]
    source_start: int
    source_end: int
    char_start: int
    char_end: int
    match: str
    in_vocabulary: bool


class LetterStep(BaseModel):
    index: int
    letter: str
    representation: str
    path: Optional[str] = None
    mime_type: Optional[str] = None
    renderable: bool = False
    motion_required: bool = False


class SignClip(BaseModel):
    sequence_index: int
    word: str
    gloss: str
    source_tokens: List[str]
    type: Literal["video", "fingerspell", "unavailable"]
    representation: str
    path: Optional[str] = None
    mime_type: Optional[str] = None
    letters: Optional[List[str]] = None
    letter_steps: Optional[List[LetterStep]] = None
    available: bool
    renderable: bool
    native_sign_available: bool
    fallback_reason: Optional[str] = None
    source: str
    duration_seconds: Optional[float] = None
    asset_id: Optional[str] = None
    sha256: Optional[str] = None
    license: Optional[str] = None
    attribution: Optional[str] = None
    use_scope: Optional[str] = None


class TranslationCoverage(BaseModel):
    total_concepts: int
    native_signs: int
    fingerspelled: int
    unavailable: int
    native_ratio: float
    fully_renderable: bool
    local_clip_count: int
    vocabulary_size: int


class NormalizeResponse(BaseModel):
    original_text: str
    analysis_text: str
    input_tokens: List[str]
    normalized_tokens: List[str]
    all_tokens: List[str]
    concepts: List[Concept]
    translation_status: dict


class TranslationResponse(BaseModel):
    mode: Literal["text", "voice"]
    transcript: str
    original_text: str
    analysis_text: str
    input_tokens: List[str]
    normalized_tokens: List[str]
    all_tokens: List[str]
    concepts: List[Concept]
    sign_clips: List[SignClip]
    coverage: TranslationCoverage
    latency: dict
    translation_status: dict


# Backward-compatible import name used by earlier integrations.
SpeechToSignResponse = TranslationResponse


def _public_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: service initialization failed; see server logs"


def initialize_services() -> None:
    """Initialize text/lookup services; the Whisper weights remain unloaded."""

    global asr_service, nlp_processor, sign_lookup, latency_logger, service_error
    if all(service is not None for service in (asr_service, nlp_processor, sign_lookup, latency_logger)):
        return
    with _init_lock:
        if all(service is not None for service in (asr_service, nlp_processor, sign_lookup, latency_logger)):
            return
        try:
            new_asr = ASRService()
            new_nlp = NLPProcessor()
            new_nlp.load()
            new_lookup = SignLookupService()
            new_lookup.load_mappings()
            new_nlp.set_vocabulary(new_lookup.get_vocabulary())
            new_latency = LatencyLogger()
        except Exception as exc:
            service_error = _public_error(exc)
            raise
        asr_service = new_asr
        nlp_processor = new_nlp
        sign_lookup = new_lookup
        latency_logger = new_latency
        service_error = None
        logger.info("Text-to-sign services initialized; speech model remains explicit and lazy")


async def _ensure_services() -> None:
    try:
        await asyncio.to_thread(initialize_services)
    except Exception:
        logger.exception("Text/voice-to-sign initialization failed")
        raise HTTPException(
            status_code=503,
            detail={
                "code": "translation_services_unavailable",
                "message": "Text and voice translation services are unavailable.",
            },
        ) from None


def capability_status() -> dict:
    initialized = all(service is not None for service in (asr_service, nlp_processor, sign_lookup, latency_logger))
    asr_loaded = bool(asr_service is not None and asr_service.is_loaded())
    asr_asset = ASRService.asset_status()
    text_ready = bool(initialized and not service_error)
    voice_ready = bool(text_ready and (asr_loaded or asr_asset["ready_to_load"]))
    if service_error:
        status = "unavailable"
    elif not initialized:
        status = "initializing"
    elif voice_ready:
        status = "ready"
    else:
        status = "text_ready_voice_unavailable"
    lookup_status = sign_lookup.status() if sign_lookup is not None else {
        "vocabulary_size": 0,
        "local_clip_count": 0,
        "fingerspell_fallback": True,
    }
    nlp_status = nlp_processor.language_model_status if nlp_processor else None
    nlp_remote = nlp_processor.remote_processing_status() if nlp_processor else {
        "enabled": False,
        "providers": [],
    }
    lookup_remote = sign_lookup.remote_processing_status() if sign_lookup else {
        "enabled": False,
        "providers": [],
    }
    remote_providers = list(dict.fromkeys([
        *nlp_remote.get("providers", []),
        *lookup_remote.get("providers", []),
    ]))
    return {
        # Overall readiness means typed text can be translated. Voice has its
        # own truthful flag and never blocks the text feature.
        "ready": text_ready,
        "text_ready": text_ready,
        "voice_ready": voice_ready,
        "status": status,
        "lazy": not initialized,
        "asr_loaded": asr_loaded,
        "error": service_error,
        "metadata": {
            "asr_asset": asr_asset,
            "microphone_permission": "requested_by_browser_only",
            "lookup": lookup_status,
            "remote_processing": {
                "enabled": bool(nlp_remote.get("enabled") or lookup_remote.get("enabled")),
                "providers": remote_providers,
                "user_text_may_leave_server": bool(nlp_remote.get("enabled") or lookup_remote.get("enabled")),
            },
            **({"nlp": nlp_status} if nlp_status else {}),
        },
        "mode": "text_voice_to_sign",
    }


async def _read_audio(audio: UploadFile) -> tuple[bytes, str]:
    content_type = (audio.content_type or "application/octet-stream").split(";", 1)[0].lower()
    if content_type not in ALLOWED_AUDIO_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail={"code": "unsupported_audio_type", "message": "Upload WAV, MP3, M4A, MP4, WebM, or OGG audio."},
        )
    data = await audio.read(MAX_AUDIO_BYTES + 1)
    if not data:
        raise HTTPException(status_code=422, detail={"code": "empty_audio", "message": "The audio recording is empty."})
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=413,
            detail={"code": "audio_too_large", "message": f"Audio limit is {MAX_AUDIO_BYTES} bytes."},
        )
    suffix = os.path.splitext(audio.filename or "")[1].lower()
    if suffix not in ALLOWED_AUDIO_SUFFIXES:
        suffix = {
            "audio/wav": ".wav", "audio/x-wav": ".wav", "audio/mpeg": ".mp3",
            "audio/mp3": ".mp3", "audio/mp4": ".m4a", "video/webm": ".webm",
            "audio/webm": ".webm", "audio/ogg": ".ogg",
        }.get(content_type, ".wav")
    return data, suffix


def _speech_preflight() -> None:
    status = ASRService.asset_status()
    if not status["package_available"]:
        raise HTTPException(
            status_code=503,
            detail={"code": "speech_runtime_missing", "message": "The local Whisper package is not installed."},
        )
    if not status["ffmpeg_available"]:
        raise HTTPException(
            status_code=503,
            detail={"code": "speech_decoder_missing", "message": "FFmpeg is required to decode microphone recordings."},
        )
    if not status["cached"]:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "speech_model_missing",
                "message": "The speech model is not installed; no download was started.",
                "action": status["fetch_command"],
            },
        )


def _transcribe(data: bytes, suffix: str) -> tuple[str, float]:
    assert asr_service is not None
    return asr_service.transcribe_bytes(data, suffix=suffix)


async def _bounded_transcribe(data: bytes, suffix: str) -> tuple[str, float]:
    if not _asr_capacity.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail={"code": "speech_busy", "message": "Speech recognition is busy; try again shortly."},
        )
    try:
        await asyncio.to_thread(ASRService.probe_audio_bytes, data, suffix)
        return await asyncio.to_thread(_transcribe, data, suffix)
    except ASRAudioTooLongError:
        raise HTTPException(
            status_code=413,
            detail={
                "code": "audio_too_long",
                "message": f"Audio limit is {MAX_AUDIO_DURATION_SECONDS:g} seconds.",
            },
        ) from None
    except ASRAudioInvalidError:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_audio_duration", "message": "Audio duration could not be validated."},
        ) from None
    finally:
        _asr_capacity.release()


def _translate(text: str) -> tuple[dict, list[dict], dict, dict]:
    assert nlp_processor is not None and sign_lookup is not None
    timings: dict[str, float] = {}
    started = time.perf_counter()
    analysis = nlp_processor.analyze(text)
    timings["nlp_ms"] = round((time.perf_counter() - started) * 1000, 1)
    lookup_started = time.perf_counter()
    clips = sign_lookup.get_sign_clips(analysis["concepts"])
    coverage = sign_lookup.coverage(clips)
    timings["lookup_ms"] = round((time.perf_counter() - lookup_started) * 1000, 1)
    status = dict(analysis["translation_status"])
    status.update(
        {
            "visual_strategy": "verified_native_clip_then_landmark_guides",
            "native_coverage": coverage["native_ratio"],
            "disclaimer": (
                "This output preserves source-linked concepts and uses verified native clips when available; "
                "the schematic fingerspelling guides are not native sign media, and this is not certified ASL grammar."
            ),
        }
    )
    analysis["translation_status"] = status
    return analysis, clips, coverage, timings


def _translation_response(
    *, mode: Literal["text", "voice"], text: str, analysis: dict,
    clips: list[dict], coverage: dict, timings: dict,
) -> TranslationResponse:
    return TranslationResponse(
        mode=mode,
        transcript=text,
        original_text=text,
        analysis_text=analysis["analysis_text"],
        input_tokens=analysis["input_tokens"],
        normalized_tokens=analysis["normalized_tokens"],
        all_tokens=analysis["all_tokens"],
        concepts=[Concept(**item) for item in analysis["concepts"]],
        sign_clips=[SignClip(**item) for item in clips],
        coverage=TranslationCoverage(**coverage),
        latency=timings,
        translation_status=analysis["translation_status"],
    )


async def _log_translation(operation: str, timings: dict, analysis: dict, coverage: dict) -> None:
    assert latency_logger is not None
    await asyncio.to_thread(
        latency_logger.log,
        operation,
        timings,
        {
            "token_count": len(analysis["normalized_tokens"]),
            "clip_count": coverage["total_concepts"],
            "native_clip_count": coverage["native_signs"],
        },
    )


@router.post("/api/speech-to-sign", response_model=TranslationResponse)
async def speech_to_sign(audio: UploadFile = File(...)):
    await _ensure_services()
    audio_bytes, suffix = await _read_audio(audio)
    _speech_preflight()
    started = time.perf_counter()
    try:
        transcript, asr_ms = await _bounded_transcribe(audio_bytes, suffix)
    except ASRAssetMissingError:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "speech_model_missing",
                "message": "The speech model is not installed; no download was started.",
                "action": "python scripts/fetch_speech_model.py",
            },
        ) from None
    except ASRAssetInvalidError:
        raise HTTPException(
            status_code=503,
            detail={"code": "speech_model_invalid", "message": "The cached speech model failed integrity verification."},
        ) from None
    except ASRRuntimeMissingError as exc:
        raise HTTPException(
            status_code=503,
            detail={"code": "speech_runtime_missing", "message": str(exc)},
        ) from None
    except HTTPException:
        raise
    except (ValueError, RuntimeError):
        logger.info("Uploaded audio could not be decoded or transcribed")
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_audio", "message": "The uploaded audio could not be decoded."},
        ) from None
    except Exception:
        logger.exception("Speech transcription failed")
        raise HTTPException(
            status_code=503,
            detail={"code": "speech_processing_failed", "message": "Speech processing is unavailable."},
        ) from None

    transcript = transcript.strip()[:MAX_TEXT_CHARS]
    if not transcript:
        raise HTTPException(
            status_code=422,
            detail={"code": "no_speech_detected", "message": "No intelligible speech was detected."},
        )
    analysis, clips, coverage, timings = await asyncio.to_thread(_translate, transcript)
    timings = {"asr_ms": round(float(asr_ms), 1), **timings}
    timings["total_ms"] = round((time.perf_counter() - started) * 1000, 1)
    await _log_translation("speech_to_sign", timings, analysis, coverage)
    return _translation_response(
        mode="voice", text=transcript, analysis=analysis, clips=clips, coverage=coverage, timings=timings
    )


@router.post("/api/text-to-sign", response_model=TranslationResponse)
async def text_to_sign(request: TextToSignRequest):
    await _ensure_services()
    started = time.perf_counter()
    try:
        analysis, clips, coverage, timings = await asyncio.to_thread(_translate, request.text)
    except Exception:
        logger.exception("Text-to-sign processing failed")
        raise HTTPException(
            status_code=503,
            detail={"code": "text_processing_failed", "message": "Text processing is unavailable."},
        ) from None
    timings["total_ms"] = round((time.perf_counter() - started) * 1000, 1)
    await _log_translation("text_to_sign", timings, analysis, coverage)
    return _translation_response(
        mode="text", text=request.text, analysis=analysis, clips=clips, coverage=coverage, timings=timings
    )


@router.post("/api/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(audio: UploadFile = File(...)):
    await _ensure_services()
    audio_bytes, suffix = await _read_audio(audio)
    _speech_preflight()
    try:
        transcript, latency_ms = await _bounded_transcribe(audio_bytes, suffix)
    except ASRAssetMissingError:
        raise HTTPException(
            status_code=503,
            detail={"code": "speech_model_missing", "message": "The speech model is not installed; no download was started.", "action": "python scripts/fetch_speech_model.py"},
        ) from None
    except ASRAssetInvalidError:
        raise HTTPException(
            status_code=503,
            detail={"code": "speech_model_invalid", "message": "The cached speech model failed integrity verification."},
        ) from None
    except ASRRuntimeMissingError as exc:
        raise HTTPException(status_code=503, detail={"code": "speech_runtime_missing", "message": str(exc)}) from None
    except HTTPException:
        raise
    except Exception:
        logger.info("Transcription failed", exc_info=True)
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_audio", "message": "The uploaded audio could not be transcribed."},
        ) from None
    transcript = transcript.strip()[:MAX_TEXT_CHARS]
    if not transcript:
        raise HTTPException(status_code=422, detail={"code": "no_speech_detected", "message": "No intelligible speech was detected."})
    return TranscriptionResponse(text=transcript, latency_ms=round(latency_ms, 1))


@router.post("/api/normalize", response_model=NormalizeResponse)
async def normalize_text(request: NormalizeRequest):
    await _ensure_services()
    try:
        assert nlp_processor is not None
        analysis = await asyncio.to_thread(nlp_processor.analyze, request.text)
    except Exception:
        logger.exception("Normalization failed")
        raise HTTPException(
            status_code=503,
            detail={"code": "normalization_failed", "message": "Text normalization is unavailable."},
        ) from None
    return NormalizeResponse(
        original_text=request.text,
        analysis_text=analysis["analysis_text"],
        input_tokens=analysis["input_tokens"],
        normalized_tokens=analysis["normalized_tokens"],
        all_tokens=analysis["all_tokens"],
        concepts=[Concept(**item) for item in analysis["concepts"]],
        translation_status=analysis["translation_status"],
    )


@router.get("/api/vocabulary")
async def get_vocabulary():
    await _ensure_services()
    assert sign_lookup is not None
    vocabulary = await asyncio.to_thread(sign_lookup.get_vocabulary)
    status = sign_lookup.status()
    return {
        "vocabulary": vocabulary,
        "available_clips": status["local_clips"],
        "total_signs": len(vocabulary),
        "clips_available": status["local_clip_count"],
        "fingerspell_fallback": status["fingerspell_fallback"],
        "coverage": {
            "native_clip_count": status["local_clip_count"],
            "vocabulary_size": status["vocabulary_size"],
            "native_ratio": round(status["local_clip_count"] / len(vocabulary), 4) if vocabulary else 0.0,
        },
        "fingerspell_guides": {
            "verified_static_letters": status.get("fingerspell_guide_count", 0),
            "dynamic_letters_requiring_motion": status.get("dynamic_fingerspell_letters", ["J", "Z"]),
            "representation": "schematic_landmark_guide_not_native_media",
        },
    }


@router.get("/api/latency-stats")
async def get_latency_stats():
    if latency_logger is None:
        return {"message": "No latency data available"}
    return await asyncio.to_thread(latency_logger.get_statistics)
