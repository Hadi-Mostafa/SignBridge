"""Bounded image encoding and decoding helpers."""

from __future__ import annotations

import base64
import binascii
import io
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

try:
    from ..config import MAX_BASE64_CHARS, MAX_IMAGE_BYTES, MAX_IMAGE_PIXELS
except ImportError:  # pragma: no cover - direct backend-path execution
    from config import MAX_BASE64_CHARS, MAX_IMAGE_BYTES, MAX_IMAGE_PIXELS


class ImageValidationError(ValueError):
    """Raised when an uploaded or WebSocket image is invalid or unsafe."""


def frame_to_base64(frame: np.ndarray, quality: int = 80) -> str:
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
        raise ImageValidationError("Expected a three-channel image frame.")
    quality = max(0, min(100, int(quality)))
    encoded, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not encoded:
        raise ImageValidationError("The image could not be encoded.")
    return base64.b64encode(buffer).decode("ascii")


def _validate_image_header(data: bytes, max_pixels: int) -> None:
    try:
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
            image_format = (image.format or "").upper()
            if image_format not in {"JPEG", "PNG", "WEBP"}:
                raise ImageValidationError("Only JPEG, PNG, and WebP images are supported.")
            if width <= 0 or height <= 0 or width * height > max_pixels:
                raise ImageValidationError(f"Image exceeds the {max_pixels}-pixel limit.")
            image.verify()
    except ImageValidationError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageValidationError("Expected a valid JPEG, PNG, or WebP image.") from exc


def decode_image_bytes(
    data: bytes,
    *,
    max_bytes: int = MAX_IMAGE_BYTES,
    max_pixels: int = MAX_IMAGE_PIXELS,
) -> np.ndarray:
    if not data:
        raise ImageValidationError("The image is empty.")
    if len(data) > max_bytes:
        raise ImageValidationError(f"Image exceeds the {max_bytes}-byte limit.")
    _validate_image_header(data, max_pixels)
    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise ImageValidationError("The image could not be decoded.")
    height, width = frame.shape[:2]
    if height * width > max_pixels:
        raise ImageValidationError(f"Image exceeds the {max_pixels}-pixel limit.")
    return frame


def decode_base64_image(
    value: str,
    *,
    max_chars: int = MAX_BASE64_CHARS,
    max_bytes: int = MAX_IMAGE_BYTES,
    max_pixels: int = MAX_IMAGE_PIXELS,
) -> np.ndarray:
    if not isinstance(value, str) or not value:
        raise ImageValidationError("Missing base64 image data.")
    if len(value) > max_chars:
        raise ImageValidationError(f"Base64 image exceeds the {max_chars}-character limit.")

    encoded = value
    if value.startswith("data:"):
        try:
            header, encoded = value.split(",", 1)
        except ValueError as exc:
            raise ImageValidationError("Malformed image data URI.") from exc
        media_type = header[5:].split(";", 1)[0].lower()
        if ";base64" not in header.lower() or media_type not in {"image/jpeg", "image/png", "image/webp"}:
            raise ImageValidationError("Unsupported image data URI.")

    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ImageValidationError("Malformed base64 image data.") from exc
    return decode_image_bytes(data, max_bytes=max_bytes, max_pixels=max_pixels)


def base64_to_frame(b64_string: str) -> Optional[np.ndarray]:
    """Backward-compatible decoder returning ``None`` on invalid input."""
    try:
        return decode_base64_image(b64_string)
    except ImageValidationError:
        return None


def resize_frame(frame: np.ndarray, max_width: int = 640, max_height: int = 480) -> np.ndarray:
    if not isinstance(frame, np.ndarray) or frame.ndim < 2:
        raise ImageValidationError("Expected an image frame.")
    height, width = frame.shape[:2]
    if height <= 0 or width <= 0:
        raise ImageValidationError("Image dimensions must be positive.")
    scale = min(max_width / width, max_height / height, 1.0)
    if scale < 1.0:
        return cv2.resize(
            frame,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return frame


def add_text_overlay(
    frame: np.ndarray,
    text: str,
    position: Tuple[int, int] = (10, 30),
    font_scale: float = 1.0,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    bg_color: Optional[Tuple[int, int, int]] = (0, 0, 0),
) -> np.ndarray:
    font = cv2.FONT_HERSHEY_SIMPLEX
    if bg_color is not None:
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        x_pos, y_pos = position
        cv2.rectangle(
            frame,
            (x_pos - 2, y_pos - text_height - 5),
            (x_pos + text_width + 2, y_pos + baseline + 5),
            bg_color,
            cv2.FILLED,
        )
    cv2.putText(frame, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
    return frame
