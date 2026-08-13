"""Train a one-class hand detector experiment for alphabet ROI localization.

The runtime expects a detector whose class name is exactly ``hand``. It does
not use an A-Z YOLO classifier, and a generic COCO model is never a valid output.
This command requires an existing real dataset and never fabricates examples.
It writes to an experiment directory; promotion is a separate reviewed action.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


def validate_dataset(path: Path) -> None:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot read YOLO dataset YAML: {exc}") from exc
    names = document.get("names") if isinstance(document, dict) else None
    values = list(names.values()) if isinstance(names, dict) else names
    if not isinstance(values, list) or [str(value).strip().lower() for value in values] != ["hand"]:
        raise ValueError("Dataset must be a one-class hand detector with names: {0: hand}.")
    lowered = path.read_text(encoding="utf-8").lower()
    if "synthetic" in lowered:
        raise ValueError("Synthetic YOLO data is forbidden by this training command.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train an experimental one-class hand detector.")
    parser.add_argument("--data", required=True, type=Path, help="real one-class YOLO data.yaml")
    parser.add_argument("--base", default="yolo11n.pt", help="Ultralytics base checkpoint")
    parser.add_argument("--output-dir", required=True, type=Path, help="experiment directory (not runtime cache)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=416)
    args = parser.parse_args()
    try:
        validate_dataset(args.data)
    except ValueError as exc:
        parser.error(str(exc))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.base)
    results = model.train(
        data=str(args.data),
        epochs=max(1, args.epochs),
        batch=max(1, args.batch),
        imgsz=max(64, args.image_size),
        project=str(args.output_dir),
        name="hand_detector_experiment",
        exist_ok=False,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
    )
    print(f"Experiment saved to {results.save_dir}.")
    print("Do not copy it into runtime until detector recall and false positives pass the frozen test set.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
