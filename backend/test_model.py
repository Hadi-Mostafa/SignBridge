"""Manual one-image smoke test for the canonical alphabet pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

try:
    from services.realtime_pipeline import RealtimeASLPipeline
except ModuleNotFoundError:
    from backend.services.realtime_pipeline import RealtimeASLPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one alphabet image through the live pipeline.")
    parser.add_argument("image", type=Path, help="JPEG/PNG/WebP containing one visible hand")
    args = parser.parse_args()
    frame = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if frame is None:
        parser.error(f"cannot decode image: {args.image}")
    with RealtimeASLPipeline() as pipeline:
        print(pipeline.predict(frame, smooth=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
