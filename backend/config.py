"""Runtime and training configuration for the sign-language application.

Runtime defaults are deliberately local and bounded. Environment variables may
override deployment-specific values without changing source code.
"""

from __future__ import annotations

import os
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return max(minimum, value)


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        return default
    return max(minimum, value)


def _env_list(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name, "")
    values = tuple(item.strip().rstrip("/") for item in raw.split(",") if item.strip())
    return values or default


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
DATA_DIR = PROJECT_ROOT / "data"
WLASL_DIR = DATA_DIR / "wlasl"
KEYPOINTS_DIR = DATA_DIR / "keypoints"
MODEL_CACHE_DIR = BACKEND_DIR / "checkpoints"
CHECKPOINTS_DIR = MODEL_CACHE_DIR
LEGACY_CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
SIGN_VIDEOS_DIR = PROJECT_ROOT / "frontend" / "assets" / "sign_videos"
WLASL_SIGN_VIDEOS_DIR = DATA_DIR / "wlasl_words20" / "videos"
SIGN_ASSET_MANIFEST = DATA_DIR / "sign_assets.json"
FINGERSPELL_GUIDES_DIR = PROJECT_ROOT / "frontend" / "assets" / "fingerspell_guides"
FINGERSPELL_GUIDE_MANIFEST = FINGERSPELL_GUIDES_DIR / "MANIFEST.json"
LOGS_DIR = PROJECT_ROOT / "logs"
VOCABULARY_FILE = DATA_DIR / "vocabulary.json"
WORD_SUGGESTION_MODEL_DIR = MODEL_CACHE_DIR / "word_suggestions"
WORD_SUGGESTION_ONNX_PATH = (
    WORD_SUGGESTION_MODEL_DIR / "onnx" / "decoder_model_merged_quantized.onnx"
)
WORD_SUGGESTION_TOKENIZER_PATH = WORD_SUGGESTION_MODEL_DIR / "tokenizer.json"
WORD_SUGGESTION_BIGRAM_PATH = WORD_SUGGESTION_MODEL_DIR / "count_2w.txt"
WORD_SUGGESTION_UNIGRAM_PATH = WORD_SUGGESTION_MODEL_DIR / "count_1w.txt"
LATENCY_LOG_FILE = LOGS_DIR / "latency.jsonl"
WHISPER_CACHE_DIR = MODEL_CACHE_DIR / "whisper"

for directory in (DATA_DIR, WLASL_DIR, KEYPOINTS_DIR, MODEL_CACHE_DIR, WHISPER_CACHE_DIR, SIGN_VIDEOS_DIR, LOGS_DIR):
    directory.mkdir(parents=True, exist_ok=True)


# Model artifacts and acceptance policy
ALPHABET_CHECKPOINT = MODEL_CACHE_DIR / "alphabet_mobilenetv3.pt"
ALPHABET_ONNX_PATH = MODEL_CACHE_DIR / "alphabet_mobilenetv3.onnx"
ALPHABET_MIN_CONFIDENCE = _env_float("ALPHABET_MIN_CONFIDENCE", 0.65)

WORD_MODEL_REPO_ID = os.getenv("WORD_MODEL_REPO_ID", "sharonn18/tgcn-wlasl")
WORD_MODEL_REVISION = os.getenv(
    "WORD_MODEL_REVISION",
    "dacb4568719caa03c44764034f599a9f8a0f63f4",
)
WORD_MODEL_SIZE = os.getenv("WORD_MODEL_SIZE", "asl100")
WORD_MIN_CONFIDENCE = _env_float("WORD_MIN_CONFIDENCE", 0.70)
WORD_MIN_MARGIN = _env_float("WORD_MIN_MARGIN", 0.15)
WORD_MIN_HAND_FRAME_RATIO = _env_float("WORD_MIN_HAND_FRAME_RATIO", 0.80)
WORD_MIN_MOTION_P90 = _env_float("WORD_MIN_MOTION_P90", 0.01)
WORD_SEQUENCE_LENGTH = _env_int("WORD_SEQUENCE_LENGTH", 50)
MODEL_DOWNLOAD_ENABLED = _env_bool("MODEL_DOWNLOAD_ENABLED", True)
MODEL_DOWNLOAD_TIMEOUT_SECONDS = _env_int("MODEL_DOWNLOAD_TIMEOUT_SECONDS", 90)


# API and transport limits
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = _env_int("API_PORT", 8000)
DEFAULT_ORIGINS = (
    f"http://127.0.0.1:{API_PORT}",
    f"http://localhost:{API_PORT}",
)
CORS_ALLOWED_ORIGINS = _env_list("CORS_ALLOWED_ORIGINS", DEFAULT_ORIGINS)
WS_ALLOWED_ORIGINS = _env_list("WS_ALLOWED_ORIGINS", CORS_ALLOWED_ORIGINS)

MAX_IMAGE_BYTES = _env_int("MAX_IMAGE_BYTES", 1_000_000)
MAX_IMAGE_PIXELS = _env_int("MAX_IMAGE_PIXELS", 1_500_000)
MAX_BASE64_CHARS = _env_int("MAX_BASE64_CHARS", ((MAX_IMAGE_BYTES + 2) // 3) * 4 + 128)
WS_MAX_MESSAGE_BYTES = _env_int("WS_MAX_MESSAGE_BYTES", MAX_BASE64_CHARS + 4_096)
WS_MAX_FRAME_SIZE = WS_MAX_MESSAGE_BYTES  # Backward-compatible name.
MAX_AUDIO_BYTES = _env_int("MAX_AUDIO_BYTES", 15_000_000)
MAX_AUDIO_DURATION_SECONDS = _env_float("MAX_AUDIO_DURATION_SECONDS", 30.0, minimum=1.0)
AUDIO_PROBE_TIMEOUT_SECONDS = _env_float("AUDIO_PROBE_TIMEOUT_SECONDS", 5.0, minimum=0.5)
MAX_TEXT_CHARS = _env_int("MAX_TEXT_CHARS", 2_000)
MAX_SUGGESTION_CONTEXT_CHARS = _env_int("MAX_SUGGESTION_CONTEXT_CHARS", 500)
MAX_SUGGESTION_PREFIX_CHARS = _env_int("MAX_SUGGESTION_PREFIX_CHARS", 48)
MAX_WORD_SUGGESTIONS = _env_int("MAX_WORD_SUGGESTIONS", 12)
WORD_SUGGESTION_CONTEXT_TOKENS = _env_int("WORD_SUGGESTION_CONTEXT_TOKENS", 48)
LANDMARK_VALUES_PER_FRAME = 110
LANDMARK_MIN_VALUE = float(os.getenv("LANDMARK_MIN_VALUE", "-1.0"))
LANDMARK_MAX_VALUE = float(os.getenv("LANDMARK_MAX_VALUE", "1.0"))

ALLOWED_IMAGE_CONTENT_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})
ALLOWED_AUDIO_CONTENT_TYPES = frozenset(
    {
        "audio/wav",
        "audio/x-wav",
        "audio/mpeg",
        "audio/mp3",
        "audio/mp4",
        "audio/webm",
        "audio/ogg",
        "video/webm",  # MediaRecorder commonly labels audio-only WebM this way.
        "application/octet-stream",
    }
)
ALLOWED_AUDIO_SUFFIXES = frozenset({".wav", ".mp3", ".m4a", ".mp4", ".webm", ".ogg"})


# Optional online integrations are off unless explicitly configured.
ENABLE_ONLINE_SIGN_LOOKUP = _env_bool("ENABLE_ONLINE_SIGN_LOOKUP", False)
SIGN_LOOKUP_TIMEOUT_SECONDS = _env_float("SIGN_LOOKUP_TIMEOUT_SECONDS", 3.0, minimum=0.1)


# Legacy training/service constants retained for existing scripts.
MEDIAPIPE_MODEL_COMPLEXITY = 1
MEDIAPIPE_MIN_DETECTION_CONF = 0.5
MEDIAPIPE_MIN_TRACKING_CONF = 0.5
NUM_POSE_LANDMARKS = 33
NUM_HAND_LANDMARKS = 21
NUM_FACE_LANDMARKS = 468
FEATURES_PER_FRAME = (NUM_POSE_LANDMARKS + 2 * NUM_HAND_LANDMARKS) * 3

YOLO_MODEL_NAME = "yolo11n.pt"
YOLO_CHECKPOINT = MODEL_CACHE_DIR / "yolo_best.pt"
HAND_YOLO_CHECKPOINT = MODEL_CACHE_DIR / "hand_yolo11n.pt"
YOLO_CONFIDENCE_THRESHOLD = 0.5
YOLO_NMS_IOU = 0.45
TEMPORAL_VOTING_WINDOW = 5
TEMPORAL_COOLDOWN = 1.0

NUM_EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 0.01

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
WHISPER_MODEL_PATH = WHISPER_CACHE_DIR / f"{WHISPER_MODEL_SIZE}.pt"
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "en")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
TTS_ENGINE = os.getenv("TTS_ENGINE", "pyttsx3")
TTS_RATE = _env_int("TTS_RATE", 150)
SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_sm")

WLASL_JSON_URL = "https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json"
WLASL_SUBSET_SIZE = 100
