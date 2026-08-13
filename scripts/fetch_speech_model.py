"""Explicitly populate and verify the optional project-local Whisper cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from config import WHISPER_MODEL_PATH, WHISPER_MODEL_SIZE  # noqa: E402
from services.asr_service import ASRService  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download the optional Whisper model into the project cache and load-test it."
    )
    parser.parse_args()
    service = ASRService()
    # This command is the single explicit acquisition boundary. Runtime API
    # requests always call ``load_model()`` with downloads disabled.
    service.load_model(allow_download=True)
    if not WHISPER_MODEL_PATH.is_file():
        raise RuntimeError(f"Whisper did not create its expected cache file: {WHISPER_MODEL_PATH}")
    print(
        json.dumps(
            {
                "model_size": WHISPER_MODEL_SIZE,
                "bytes": WHISPER_MODEL_PATH.stat().st_size,
                "sha256": sha256(WHISPER_MODEL_PATH),
                "load_test": "passed",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
