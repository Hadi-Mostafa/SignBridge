"""Train an *experimental* 42-feature alphabet Random Forest safely.

The historical version overwrote the runtime asset after a row-random split.
That leaks signer/background identity and bypasses the model lock hash. This
replacement requires an explicit grouped manifest and writes only to an
experiment directory. Promotion into ``models.lock.json`` is a separate,
reviewed step after camera-domain evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score


EXPECTED_FEATURES = 42
EXPECTED_CLASSES = set(range(26))
ALLOWED_SPLITS = {"train", "validation", "test"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset(csv_path: Path, manifest_path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict]:
    frame = pd.read_csv(csv_path, header=None)
    if frame.shape[1] != EXPECTED_FEATURES + 1:
        raise ValueError(f"Expected label + {EXPECTED_FEATURES} features, got {frame.shape[1]} columns.")
    values = frame.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Dataset contains NaN or infinity.")
    labels = values[:, 0]
    if not np.equal(labels, labels.astype(int)).all() or set(labels.astype(int)) != EXPECTED_CLASSES:
        raise ValueError("Dataset labels must cover the exact integer class IDs 0..25.")
    features = values[:, 1:].astype(np.float32)
    if np.abs(features).max() > 1.00001:
        raise ValueError("Features must follow wrist-relative max-absolute normalization in [-1,1].")

    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = document.get("samples") if isinstance(document, dict) else None
    if not isinstance(samples, list) or len(samples) != len(frame):
        raise ValueError("Manifest samples must have exactly one row per CSV sample.")

    groups_by_split: dict[str, set[str]] = {name: set() for name in ALLOWED_SPLITS}
    indices_by_split: dict[str, list[int]] = {name: [] for name in ALLOWED_SPLITS}
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"Manifest sample {index} must be an object.")
        split = sample.get("split")
        group = str(sample.get("signer_id") or sample.get("source_id") or "").strip()
        if split not in ALLOWED_SPLITS or not group:
            raise ValueError(f"Sample {index} requires split and signer_id/source_id.")
        if bool(sample.get("synthetic")) or "synthetic" in str(sample.get("path", "")).lower():
            raise ValueError("Synthetic samples are forbidden by the production training command.")
        indices_by_split[split].append(index)
        groups_by_split[split].add(group)

    if any(not indices for indices in indices_by_split.values()):
        raise ValueError("Train, validation, and test splits must all be non-empty.")
    if any(groups_by_split[a] & groups_by_split[b] for a in ALLOWED_SPLITS for b in ALLOWED_SPLITS if a < b):
        raise ValueError("Signer/source groups leak across splits.")
    return features, labels.astype(np.int64), {
        name: np.asarray(indices, dtype=np.int64) for name, indices in indices_by_split.items()
    }, document


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a grouped, non-production alphabet RF experiment.")
    parser.add_argument("--csv", required=True, type=Path, help="label + 42 normalized feature CSV")
    parser.add_argument("--manifest", required=True, type=Path, help="grouped split manifest JSON")
    parser.add_argument("--output-dir", required=True, type=Path, help="new experiment output directory")
    parser.add_argument("--trees", type=int, default=300)
    args = parser.parse_args()
    if args.output_dir.resolve() == (Path(__file__).resolve().parents[1] / "checkpoints").resolve():
        parser.error("output-dir must not be backend/checkpoints; promote a validated model separately")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        features, labels, splits, manifest = load_dataset(args.csv, args.manifest)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))

    model = RandomForestClassifier(
        n_estimators=max(1, args.trees), n_jobs=-1, random_state=42, class_weight="balanced_subsample"
    )
    model.fit(features[splits["train"]], labels[splits["train"]])
    metrics = {}
    for split in ("validation", "test"):
        truth = labels[splits[split]]
        predicted = model.predict(features[splits[split]])
        metrics[split] = {
            "samples": int(len(truth)),
            "accuracy": float(accuracy_score(truth, predicted)),
            "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
            "classification_report": classification_report(
                truth, predicted, labels=list(range(26)), output_dict=True, zero_division=0
            ),
        }

    model_path = args.output_dir / "alphabet_keypoint_rf.joblib"
    joblib.dump(model, model_path)
    metadata = {
        "architecture": "sklearn_random_forest",
        "input_schema": "mediapipe_hand21_wrist_relative_xy_maxabs_v1",
        "features": EXPECTED_FEATURES,
        "labels": [chr(ord("A") + index) for index in range(26)],
        "csv_sha256": sha256(args.csv),
        "manifest_sha256": sha256(args.manifest),
        "model_sha256": sha256(model_path),
        "metrics": metrics,
        "promotion_status": "experimental_not_runtime",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(model_path), "metrics": metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
