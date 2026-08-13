"""Stable API for local context-aware next-word suggestions."""

from __future__ import annotations

import asyncio
import threading
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

try:
    from ..config import (
        MAX_SUGGESTION_CONTEXT_CHARS,
        MAX_SUGGESTION_PREFIX_CHARS,
        MAX_WORD_SUGGESTIONS,
        VOCABULARY_FILE,
    )
    from ..services.context_word_predictor import ContextWordPredictor
except ImportError:  # pragma: no cover - direct backend-path execution
    from config import (
        MAX_SUGGESTION_CONTEXT_CHARS,
        MAX_SUGGESTION_PREFIX_CHARS,
        MAX_WORD_SUGGESTIONS,
        VOCABULARY_FILE,
    )
    from services.context_word_predictor import ContextWordPredictor


router = APIRouter()
predictor: ContextWordPredictor | None = None
service_error: str | None = None
_lock = threading.Lock()


class WordSuggestionRequest(BaseModel):
    context: List[str] = Field(default_factory=list, max_length=64)
    sentence: str = Field(default="", max_length=MAX_SUGGESTION_CONTEXT_CHARS)
    prefix: str = Field(default="", max_length=MAX_SUGGESTION_PREFIX_CHARS)
    limit: int = Field(default=8, ge=1, le=MAX_WORD_SUGGESTIONS)

    @field_validator("context")
    @classmethod
    def bounded_context(cls, value: List[str]) -> List[str]:
        normalized = [str(item).strip() for item in value if str(item).strip()]
        if sum(len(item) for item in normalized) > MAX_SUGGESTION_CONTEXT_CHARS:
            raise ValueError("Context is too long.")
        return normalized

    @field_validator("prefix")
    @classmethod
    def letters_only_prefix(cls, value: str) -> str:
        clean = value.strip().lower()
        if clean and not clean.isascii():
            raise ValueError("Prefix must use ASCII letters.")
        if clean and not clean.isalpha():
            raise ValueError("Prefix must contain only letters.")
        return clean


class WordSuggestion(BaseModel):
    word: str
    score: float
    rank: int


class WordSuggestionResponse(BaseModel):
    suggestions: List[WordSuggestion]
    context: List[str]
    prefix: str
    limit: int
    engine: str
    offline: bool
    latency_ms: float


def _load_vocabulary() -> list[str]:
    import json

    try:
        payload = json.loads(VOCABULARY_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    values = payload.get("labels", []) if isinstance(payload, dict) else payload
    return [str(item) for item in values] if isinstance(values, list) else []


def initialize_service() -> None:
    global predictor, service_error
    if predictor is not None:
        return
    with _lock:
        if predictor is not None:
            return
        try:
            candidate = ContextWordPredictor(_load_vocabulary())
            candidate.load()
            if not candidate.status()["ready"]:
                raise RuntimeError("No verified local context model is available.")
            predictor = candidate
            service_error = None
        except Exception as exc:
            service_error = f"{type(exc).__name__}: suggestion service unavailable"
            raise


def capability_status() -> dict:
    if predictor is None:
        assets = ContextWordPredictor.asset_status(verify=False)
        return {
            "ready": False,
            "status": "initializing" if assets["ready"] else "unavailable",
            "error": service_error,
            "metadata": assets,
            "mode": "context_word_suggestions",
        }
    status = predictor.status()
    return {
        "ready": status["ready"],
        "status": status["status"],
        "error": status["error"],
        "metadata": {key: value for key, value in status.items() if key not in {"ready", "status", "error"}},
        "mode": "context_word_suggestions",
    }


@router.post("/api/word-suggestions", response_model=WordSuggestionResponse)
async def word_suggestions(request: WordSuggestionRequest):
    try:
        await asyncio.to_thread(initialize_service)
        assert predictor is not None
        result = await asyncio.to_thread(
            predictor.suggest,
            context=request.context,
            sentence=request.sentence,
            prefix=request.prefix,
            limit=request.limit,
        )
        return WordSuggestionResponse(**result)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "word_suggestions_unavailable",
                "message": "Local context-aware suggestions are unavailable.",
                "action": "python scripts/fetch_word_suggestion_model.py",
            },
        ) from None

