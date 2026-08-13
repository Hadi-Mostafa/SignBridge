"""
Automatic Speech Recognition (ASR) Service using OpenAI Whisper.

Provides speech-to-text transcription for the Speech → Sign pipeline.
Uses the local Whisper model (no API key required) for offline operation.

Whisper model sizes and approximate performance:
    | Model  | Parameters | VRAM  | Speed  | English WER |
    |--------|-----------|-------|--------|-------------|
    | tiny   | 39M       | ~1GB  | ~32x   | ~7.6%       |
    | base   | 74M       | ~1GB  | ~16x   | ~5.0%       |
    | small  | 244M      | ~2GB  | ~6x    | ~3.4%       |
    | medium | 769M      | ~5GB  | ~2x    | ~2.7%       |

For MVP, we default to 'base' which balances accuracy and latency well.
On CPU, expect ~2-4 seconds for a 5-second audio clip.

Reference:
    Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision"
    https://arxiv.org/abs/2212.04356
"""

import hashlib
import importlib.util
import logging
import math
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Tuple

try:
    from ..config import (
        WHISPER_CACHE_DIR,
        AUDIO_PROBE_TIMEOUT_SECONDS,
        MAX_AUDIO_DURATION_SECONDS,
        WHISPER_LANGUAGE,
        WHISPER_MODEL_PATH,
        WHISPER_MODEL_SIZE,
        WHISPER_DEVICE,
    )
except ImportError:  # pragma: no cover - direct backend-path execution
    from config import (
        WHISPER_CACHE_DIR,
        AUDIO_PROBE_TIMEOUT_SECONDS,
        MAX_AUDIO_DURATION_SECONDS,
        WHISPER_LANGUAGE,
        WHISPER_MODEL_PATH,
        WHISPER_MODEL_SIZE,
        WHISPER_DEVICE,
    )

logger = logging.getLogger(__name__)


class ASRServiceError(RuntimeError):
    """Base class for explicit speech capability failures."""


class ASRAssetMissingError(ASRServiceError):
    """The explicitly provisioned Whisper checkpoint is absent."""


class ASRAssetInvalidError(ASRServiceError):
    """The cached checkpoint does not match Whisper's published checksum."""


class ASRRuntimeMissingError(ASRServiceError):
    """A required local runtime dependency is absent."""


class ASRAudioInvalidError(ASRServiceError):
    """The uploaded media has no trustworthy positive duration."""


class ASRAudioTooLongError(ASRServiceError):
    """The uploaded media exceeds the configured duration limit."""


class ASRService:
    """
    Speech-to-text service using OpenAI Whisper.

    Usage:
        asr = ASRService()
        asr.load_model()
        text = asr.transcribe_file("audio.wav")
        text = asr.transcribe_bytes(audio_bytes)
    """

    _integrity_lock = threading.Lock()
    _integrity_cache: dict[tuple[str, int, int, str], bool] = {}

    def __init__(self):
        """Initialize ASR service (model loaded lazily)."""
        self.model = None
        self.model_size = WHISPER_MODEL_SIZE
        self.language = WHISPER_LANGUAGE
        self.device = WHISPER_DEVICE
        self.last_transcription_time: float = 0.0

    def load_model(self, *, allow_download: bool = False):
        """
        Load a pre-provisioned Whisper model without hidden network activity.

        Model acquisition is intentionally separate: run
        ``scripts/fetch_speech_model.py`` while online before serving voice
        requests.  This method never turns a user request into a large download.
        """
        status = self.asset_status()
        if not status["package_available"]:
            raise ASRRuntimeMissingError("The openai-whisper package is not installed.")
        if not status["ffmpeg_available"]:
            raise ASRRuntimeMissingError("FFmpeg is required to decode uploaded audio.")
        if not status["cached"] and not allow_download:
            raise ASRAssetMissingError(
                "The Whisper checkpoint is not provisioned. Run scripts/fetch_speech_model.py explicitly."
            )

        import whisper

        if allow_download and not status["cached_valid"]:
            # This is the explicit acquisition boundary used only by the fetch
            # script. Serving paths always use the default False value.
            downloaded_model = whisper.load_model(
                self.model_size,
                device=self.device,
                download_root=str(WHISPER_CACHE_DIR),
            )
            status = self.asset_status(force_verify=True)
            if not status["cached_valid"]:
                raise ASRAssetInvalidError("The downloaded Whisper checkpoint failed integrity verification.")
            self.model = downloaded_model
            return self.model
        if not status["cached_valid"]:
            raise ASRAssetInvalidError(
                "The cached Whisper checkpoint failed checksum verification; fetch it again explicitly."
            )
        logger.info("Loading Whisper model size=%s device=%s", self.model_size, self.device)
        verified_path = str(WHISPER_MODEL_PATH.resolve())
        self.model = whisper.load_model(
            verified_path,
            device=self.device,
            download_root=str(WHISPER_CACHE_DIR),
        )
        logger.info("Whisper model loaded")
        return self.model

    def transcribe_file(self, audio_path: str) -> Tuple[str, float]:
        """
        Transcribe an audio file to text.

        Args:
            audio_path: Path to the audio file (WAV, MP3, etc.).

        Returns:
            Tuple of (transcribed_text, latency_ms).
        """
        if self.model is None:
            self.load_model()

        start_time = time.time()

        result = self.model.transcribe(
            audio_path,
            language=self.language,
            fp16=False,  # CPU doesn't support fp16
        )

        self.last_transcription_time = (time.time() - start_time) * 1000
        transcript = result["text"].strip()

        # Never log transcript content; spoken input can contain sensitive data.
        logger.info(
            "Transcription completed duration_ms=%.0f characters=%d",
            self.last_transcription_time,
            len(transcript),
        )
        return transcript, self.last_transcription_time

    def transcribe_bytes(self, audio_bytes: bytes, suffix: str = ".wav") -> Tuple[str, float]:
        """
        Transcribe audio from raw bytes (e.g. from a WebSocket or HTTP upload).

        Args:
            audio_bytes: Raw audio file bytes.
            suffix: File extension hint for the temp file.

        Returns:
            Tuple of (transcribed_text, latency_ms).
        """
        # Write bytes to a temp file (Whisper requires a file path)
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            return self.transcribe_file(tmp_path)
        finally:
            # Clean up temp file
            os.unlink(tmp_path)

    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready."""
        return self.model is not None

    @staticmethod
    def probe_audio_bytes(audio_bytes: bytes, suffix: str = ".wav") -> float:
        """Return decoded duration using bounded local ffprobe validation."""

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        try:
            try:
                completed = subprocess.run(
                    [
                        "ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=noprint_wrappers=1:nokey=1", tmp_path,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=AUDIO_PROBE_TIMEOUT_SECONDS,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise ASRAudioInvalidError("Audio duration could not be inspected.") from exc
            if completed.returncode != 0:
                raise ASRAudioInvalidError("Audio duration could not be inspected.")
            try:
                duration = float(completed.stdout.strip())
            except ValueError as exc:
                raise ASRAudioInvalidError("Audio duration is invalid.") from exc
            if not math.isfinite(duration) or duration <= 0:
                raise ASRAudioInvalidError("Audio duration is invalid.")
            if duration > MAX_AUDIO_DURATION_SECONDS:
                raise ASRAudioTooLongError(
                    f"Audio exceeds the {MAX_AUDIO_DURATION_SECONDS:g}-second limit."
                )
            return duration
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @classmethod
    def asset_status(cls, *, force_verify: bool = False) -> dict:
        package_available = importlib.util.find_spec("whisper") is not None
        ffmpeg_available = shutil.which("ffmpeg") is not None
        cached = WHISPER_MODEL_PATH.is_file()
        cached_valid = False
        expected_hash = ""
        integrity_error = None
        if package_available:
            try:
                import whisper

                model_url = getattr(whisper, "_MODELS", {}).get(WHISPER_MODEL_SIZE, "")
                expected_hash = model_url.split("/")[-2] if model_url else ""
            except Exception:
                integrity_error = "model_registry_unavailable"
        if cached and len(expected_hash) == 64:
            try:
                stat = WHISPER_MODEL_PATH.stat()
                key = (str(WHISPER_MODEL_PATH.resolve()), stat.st_size, stat.st_mtime_ns, expected_hash)
                with cls._integrity_lock:
                    if force_verify:
                        cls._integrity_cache.pop(key, None)
                    cached_result = cls._integrity_cache.get(key)
                if cached_result is None:
                    digest = hashlib.sha256()
                    with WHISPER_MODEL_PATH.open("rb") as stream:
                        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(chunk)
                    cached_result = digest.hexdigest().lower() == expected_hash.lower()
                    with cls._integrity_lock:
                        cls._integrity_cache[key] = cached_result
                cached_valid = bool(cached_result)
                if not cached_valid:
                    integrity_error = "checksum_mismatch"
            except OSError:
                integrity_error = "asset_unreadable"
        elif cached and not integrity_error:
            integrity_error = "expected_checksum_unavailable"
        return {
            "model_size": WHISPER_MODEL_SIZE,
            "cached": cached,
            "cached_valid": cached_valid,
            "integrity_verified": cached_valid,
            "integrity_error": integrity_error,
            "package_available": package_available,
            "ffmpeg_available": ffmpeg_available,
            "ready_to_load": bool(cached_valid and package_available and ffmpeg_available),
            "runtime_downloads": False,
            # Kept for old clients; requests never initiate downloads now.
            "download_allowed": False,
            "explicit_fetch_required": not cached_valid,
            "fetch_command": "python scripts/fetch_speech_model.py",
        }
