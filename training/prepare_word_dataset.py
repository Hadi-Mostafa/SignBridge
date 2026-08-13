"""Download a small, real WLASL dataset for the Words & Feelings mode.

This intentionally uses real labelled WLASL clips rather than the project's
old synthetic keypoints.  It writes a manifest so every downloaded training
example remains traceable to its WLASL video ID and URL.

Example:
    python training/prepare_word_dataset.py --max-per-class 12
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WLASL_JSON = PROJECT_ROOT / "data" / "WLASL_v0.3.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "wlasl_words20" / "videos"
MANIFEST_PATH = PROJECT_ROOT / "data" / "wlasl_words20" / "manifest.json"

# These are deliberately common, practical signs, including the requested
# feelings.  The list contains 20 classes and has 355 known WLASL instances.
LABELS = [
    "help", "yes", "no", "eat", "drink", "want", "need", "go", "stop", "thank you",
    "please", "happy", "sad", "angry", "tired", "love", "like", "good", "bad", "pain",
]


def safe_name(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def video_is_valid(path: Path) -> bool:
    capture = cv2.VideoCapture(str(path))
    valid = capture.isOpened() and capture.get(cv2.CAP_PROP_FRAME_COUNT) > 1
    capture.release()
    return valid


def download(instance: dict, destination: Path, timeout_seconds: int) -> bool:
    url = instance.get("url")
    if not url:
        return False
    command = [
        sys.executable, "-m", "yt_dlp", "--no-warnings", "--no-playlist",
        "--socket-timeout", "10", "--retries", "1", "--output", str(destination), url,
    ]
    try:
        completed = subprocess.run(command, timeout=timeout_seconds, capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        return False
    return completed.returncode == 0 and destination.exists() and video_is_valid(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare real WLASL clips for word recognition")
    parser.add_argument("--max-per-class", type=int, default=12)
    parser.add_argument("--download", action="store_true", help="Download clips; otherwise only write the manifest")
    parser.add_argument("--timeout", type=int, default=30, help="Maximum seconds per source clip")
    args = parser.parse_args()

    if not WLASL_JSON.exists():
        raise FileNotFoundError(f"Missing {WLASL_JSON}. Download WLASL metadata first.")
    if args.max_per_class < 1:
        raise ValueError("--max-per-class must be positive")

    with WLASL_JSON.open(encoding="utf-8") as file:
        entries = {entry["gloss"].lower(): entry for entry in json.load(file)}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {"dataset": "WLASL", "labels": LABELS, "samples": []}
    for label in LABELS:
        entry = entries.get(label)
        if not entry:
            print(f"[WARN] No WLASL entry for {label}")
            continue
        class_dir = OUTPUT_DIR / safe_name(label)
        class_dir.mkdir(exist_ok=True)
        for instance in entry.get("instances", [])[:args.max_per_class]:
            video_id = str(instance.get("video_id", "unknown"))
            destination = class_dir / f"{video_id}.mp4"
            available = destination.exists() and video_is_valid(destination)
            if args.download and not available:
                available = download(instance, destination, args.timeout)
            manifest["samples"].append({
                "label": label,
                "video_id": video_id,
                "url": instance.get("url"),
                "path": str(destination.relative_to(PROJECT_ROOT)),
                "available": available,
            })
            print(f"[{label}] {video_id}: {'ready' if available else 'unavailable'}")

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    ready = sum(sample["available"] for sample in manifest["samples"])
    print(f"Prepared {ready}/{len(manifest['samples'])} real clips. Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
