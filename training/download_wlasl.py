"""
WLASL Dataset Download Script.

Downloads the WLASL (Word-Level American Sign Language) dataset:
    1. Fetches the WLASL JSON glossary from GitHub
    2. Downloads video clips from YouTube using yt-dlp
    3. Generates a vocabulary.json mapping class indices to sign labels
    4. Reports download statistics (available vs. missing videos)

The WLASL dataset is organized as:
    {gloss: word, instances: [{video_id, url, bbox, fps, frame_start, frame_end, signer_id}]}

For WLASL-100, we take the first 100 glosses (sorted by frequency/availability).

Usage:
    python download_wlasl.py                    # Download WLASL-100
    python download_wlasl.py --subset_size 300  # Download WLASL-300

Reference:
    Li et al., "Word-level Deep Sign Language Recognition from Video"
    https://github.com/dxli94/WLASL
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from backend.config import (
    WLASL_DIR,
    WLASL_JSON_URL,
    WLASL_SUBSET_SIZE,
    DATA_DIR,
    VOCABULARY_FILE,
)


def download_wlasl_json(output_dir: Path) -> Path:
    """
    Download the WLASL glossary JSON file.

    Returns:
        Path to the downloaded JSON file.
    """
    json_path = output_dir / "WLASL_v0.3.json"

    if json_path.exists():
        print(f"[Download] WLASL JSON already exists: {json_path}")
        return json_path

    print(f"[Download] Fetching WLASL glossary from GitHub...")
    import urllib.request
    urllib.request.urlretrieve(WLASL_JSON_URL, str(json_path))
    print(f"[Download] Saved to {json_path}")

    return json_path


def parse_wlasl_json(json_path: Path, subset_size: int = 100) -> List[Dict]:
    """
    Parse the WLASL JSON and extract the first N glosses.

    Args:
        json_path: Path to WLASL_v0.3.json.
        subset_size: Number of sign classes to include.

    Returns:
        List of gloss entries (each with 'gloss' and 'instances').
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    # Take the first subset_size entries (sorted by availability in WLASL)
    subset = data[:subset_size]

    total_instances = sum(len(entry.get("instances", [])) for entry in subset)
    print(f"[Parse] WLASL-{subset_size}: {len(subset)} signs, {total_instances} total instances")

    return subset


def download_videos(glosses: List[Dict], output_dir: Path, max_per_class: int = 20) -> Dict:
    """
    Download video clips from YouTube using yt-dlp.

    Args:
        glosses: Parsed WLASL gloss entries.
        output_dir: Directory to save downloaded videos.
        max_per_class: Maximum videos to download per sign class.

    Returns:
        Statistics dict with download results.
    """
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    stats = {"total": 0, "downloaded": 0, "failed": 0, "skipped": 0}

    for gloss_entry in glosses:
        gloss = gloss_entry["gloss"]
        instances = gloss_entry.get("instances", [])[:max_per_class]

        print(f"\n[Download] Sign: '{gloss}' ({len(instances)} instances)")

        for i, instance in enumerate(instances):
            stats["total"] += 1
            video_id = instance.get("video_id", f"{gloss}_{i}")
            url = instance.get("url", "")

            output_path = videos_dir / f"{video_id}.mp4"

            # Skip if already downloaded
            if output_path.exists():
                stats["skipped"] += 1
                continue

            if not url:
                stats["failed"] += 1
                continue

            try:
                # Use yt-dlp to download the video
                cmd = [
                    "yt-dlp",
                    "--quiet",
                    "--no-warnings",
                    "-f", "mp4",
                    "--output", str(output_path),
                    "--socket-timeout", "10",
                    url,
                ]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                if result.returncode == 0 and output_path.exists():
                    stats["downloaded"] += 1
                    print(f"  [OK] Downloaded {video_id}")
                else:
                    stats["failed"] += 1

            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                stats["failed"] += 1

    return stats


def generate_vocabulary(glosses: List[Dict], output_path: Path):
    """
    Generate vocabulary.json mapping class indices to sign labels.

    Args:
        glosses: Parsed WLASL gloss entries.
        output_path: Path to save vocabulary JSON.
    """
    labels = [entry["gloss"].lower() for entry in glosses]

    vocab = {
        "labels": labels,
        "label_to_index": {label: idx for idx, label in enumerate(labels)},
        "index_to_label": {idx: label for idx, label in enumerate(labels)},
        "num_classes": len(labels),
        "dataset": "WLASL",
        "subset_size": len(labels),
    }

    with open(output_path, "w") as f:
        json.dump(vocab, f, indent=2)

    print(f"\n[Vocabulary] Generated {len(labels)} labels → {output_path}")
    print(f"[Vocabulary] Sample labels: {labels[:10]}")


def generate_demo_vocabulary():
    """
    Generate a demo vocabulary for testing without downloading WLASL.
    Uses common ASL signs that cover basic communication needs.
    """
    demo_labels = [
        "hello", "thank you", "please", "sorry", "yes", "no",
        "help", "stop", "go", "come", "want", "need",
        "eat", "drink", "water", "food", "hungry", "thirsty",
        "happy", "sad", "angry", "tired", "sick", "pain",
        "good", "bad", "big", "small", "hot", "cold",
        "love", "like", "friend", "family", "mother", "father",
        "brother", "sister", "baby", "child", "man", "woman",
        "work", "school", "home", "car", "money", "time",
        "today", "tomorrow", "yesterday", "morning", "night", "week",
        "name", "what", "where", "when", "why", "how",
        "who", "can", "more", "finish", "again", "wait",
        "sit", "stand", "walk", "run", "sleep", "wake",
        "open", "close", "give", "take", "know", "understand",
        "think", "feel", "see", "hear", "say", "tell",
        "read", "write", "learn", "teach", "play", "practice",
        "right", "wrong", "same", "different", "new", "old",
        "color", "red", "blue", "green", "white", "black",
    ]

    # Trim to 100
    labels = demo_labels[:100]

    vocab = {
        "labels": labels,
        "label_to_index": {label: idx for idx, label in enumerate(labels)},
        "index_to_label": {idx: label for idx, label in enumerate(labels)},
        "num_classes": len(labels),
        "dataset": "demo",
        "subset_size": len(labels),
    }

    with open(VOCABULARY_FILE, "w") as f:
        json.dump(vocab, f, indent=2)

    print(f"[Vocabulary] Generated DEMO vocabulary: {len(labels)} common ASL signs")
    return labels


def main():
    parser = argparse.ArgumentParser(description="Download WLASL dataset")
    parser.add_argument(
        "--subset_size", type=int, default=WLASL_SUBSET_SIZE,
        help=f"Number of sign classes to download (default: {WLASL_SUBSET_SIZE})"
    )
    parser.add_argument(
        "--max_per_class", type=int, default=20,
        help="Maximum videos per sign class (default: 20)"
    )
    parser.add_argument(
        "--demo_only", action="store_true",
        help="Generate demo vocabulary without downloading videos"
    )
    args = parser.parse_args()

    if args.demo_only:
        generate_demo_vocabulary()
        return

    # Step 1: Download WLASL JSON glossary
    json_path = download_wlasl_json(WLASL_DIR)

    # Step 2: Parse glossary
    glosses = parse_wlasl_json(json_path, args.subset_size)

    # Step 3: Generate vocabulary mapping
    generate_vocabulary(glosses, VOCABULARY_FILE)

    # Step 4: Download videos
    print("\n" + "=" * 60)
    print("Starting video downloads (this may take a while)...")
    print("=" * 60)

    stats = download_videos(glosses, WLASL_DIR, args.max_per_class)

    # Report
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    print(f"  Total instances:  {stats['total']}")
    print(f"  Downloaded:       {stats['downloaded']}")
    print(f"  Already cached:   {stats['skipped']}")
    print(f"  Failed/missing:   {stats['failed']}")
    print(f"  Success rate:     {(stats['downloaded'] + stats['skipped']) / max(stats['total'], 1) * 100:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
