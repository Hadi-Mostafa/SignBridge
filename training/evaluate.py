"""Evaluate a training Pose-TGCN checkpoint on one explicit held-out split."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.extract_keypoints import (  # noqa: E402
    ContractError,
    canonical_json_sha256,
    ensure_non_runtime_output,
    schema_metadata,
    sha256_file,
    validate_dataset_manifest,
)
from training.train import (  # noqa: E402
    DEFAULT_CONFIG,
    PRODUCTION_ARCHITECTURE,
    PoseSequenceDataset,
    _select_device,
    build_model,
    load_config,
)


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def compute_metrics(
    truth: Sequence[int],
    top5_predictions: Sequence[Sequence[int]],
    labels: Sequence[str],
) -> dict[str, Any]:
    if not truth or len(truth) != len(top5_predictions):
        raise ContractError("Evaluation predictions and ground truth must be non-empty and aligned.")
    top1 = [int(row[0]) for row in top5_predictions]
    top1_correct = sum(prediction == target for prediction, target in zip(top1, truth))
    top5_correct = sum(target in predictions for target, predictions in zip(truth, top5_predictions))
    per_class: dict[str, Any] = {}
    macro_f1_values: list[float] = []
    for class_id, label in enumerate(labels):
        tp = sum(t == class_id and p == class_id for t, p in zip(truth, top1))
        fp = sum(t != class_id and p == class_id for t, p in zip(truth, top1))
        fn = sum(t == class_id and p != class_id for t, p in zip(truth, top1))
        support = sum(t == class_id for t in truth)
        precision = _safe_rate(tp, tp + fp)
        recall = _safe_rate(tp, tp + fn)
        f1 = _safe_rate(2 * precision * recall, precision + recall)
        macro_f1_values.append(f1)
        per_class[label] = {
            "class_id": class_id,
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    matrix = [[0 for _ in labels] for _ in labels]
    for target, prediction in zip(truth, top1):
        matrix[target][prediction] += 1
    return {
        "samples": len(truth),
        "top1_accuracy": top1_correct / len(truth),
        "top5_accuracy": top5_correct / len(truth),
        "macro_f1_all_labels": sum(macro_f1_values) / len(labels),
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


def _validate_checkpoint_metadata(
    metadata: Mapping[str, Any],
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    if metadata.get("architecture") != PRODUCTION_ARCHITECTURE:
        raise ContractError("Checkpoint architecture metadata does not match production Pose-TGCN.")
    if metadata.get("architecture_sha256") != canonical_json_sha256(PRODUCTION_ARCHITECTURE):
        raise ContractError("Checkpoint architecture hash is invalid.")
    if metadata.get("schema") != schema_metadata():
        raise ContractError("Checkpoint preprocessing schema does not match production.")
    if metadata.get("schema_sha256") != canonical_json_sha256(schema_metadata()):
        raise ContractError("Checkpoint schema hash is invalid.")
    if metadata.get("labels") != manifest.get("labels"):
        raise ContractError("Checkpoint label list/order does not match the evaluation dataset.")
    if metadata.get("labels_sha256") != manifest.get("labels_sha256"):
        raise ContractError("Checkpoint label hash does not match the evaluation dataset.")
    dataset = metadata.get("dataset")
    if not isinstance(dataset, Mapping) or dataset.get("content_sha256") != manifest.get("content_sha256"):
        raise ContractError(
            "Checkpoint was trained on a different dataset manifest. Evaluate the same signed-off manifest "
            "or retrain; silent dataset substitution is prohibited."
        )
    training_config = metadata.get("training_config")
    if not isinstance(training_config, Mapping) or metadata.get(
        "training_config_content_sha256"
    ) != canonical_json_sha256(training_config):
        raise ContractError("Checkpoint effective training config metadata/hash is invalid.")
    if metadata.get("test_split_used_during_training") is not False:
        raise ContractError("Checkpoint does not prove that the test split remained held out.")


def evaluate_checkpoint(
    checkpoint_path: Path,
    config: dict[str, Any],
    split: str,
    output_dir: Path,
    device_name: str = "auto",
) -> Path:
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise ContractError("PyTorch is required to evaluate Pose-TGCN.") from exc
    if split not in ("val", "test"):
        raise ContractError("Only explicit held-out 'val' or 'test' splits may be evaluated.")
    ensure_non_runtime_output(output_dir)
    if output_dir.exists():
        raise ContractError(f"Output directory already exists: {output_dir}. Evaluation never overwrites results.")
    if not checkpoint_path.is_file():
        raise ContractError(f"Checkpoint not found: {checkpoint_path}. Run training/train.py on captured data first.")
    manifest, samples = validate_dataset_manifest(
        config["dataset"]["manifest"],
        config["dataset"]["labels"],
        minimum_per_label_split=config["dataset"]["minimum_samples_per_label"],
        verify_arrays=True,
    )
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ContractError(f"Cannot safely load checkpoint {checkpoint_path}: {exc}") from exc
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("state_dict"), dict):
        raise ContractError("Checkpoint must contain a state_dict and complete metadata.")
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ContractError("Checkpoint has no verifiable metadata; legacy checkpoints are not evaluable.")
    _validate_checkpoint_metadata(metadata, manifest, config)
    device = _select_device(device_name, torch)
    model = build_model(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    selected = [sample for sample in samples if sample["split"] == split]
    loader = DataLoader(
        PoseSequenceDataset(selected),
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
    )
    truth: list[int] = []
    top5_predictions: list[list[int]] = []
    predictions: list[dict[str, Any]] = []
    latencies: list[float] = []
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    with torch.inference_mode():
        for inputs, targets, sample_ids in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            logits = model(inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            per_sample_latency = elapsed_ms / max(1, inputs.shape[0])
            total_loss += float(criterion(logits, targets))
            probabilities = torch.softmax(logits, dim=1)
            values, indices = probabilities.topk(k=5, dim=1)
            for sample_id, target, class_ids, confidences in zip(
                sample_ids, targets.cpu().tolist(), indices.cpu().tolist(), values.cpu().tolist()
            ):
                truth.append(int(target))
                top5_predictions.append([int(value) for value in class_ids])
                latencies.append(per_sample_latency)
                predictions.append(
                    {
                        "sample_id": sample_id,
                        "truth": manifest["labels"][int(target)],
                        "truth_class_id": int(target),
                        "top5": [
                            {
                                "label": manifest["labels"][int(class_id)],
                                "class_id": int(class_id),
                                "confidence": float(confidence),
                            }
                            for class_id, confidence in zip(class_ids, confidences)
                        ],
                    }
                )
    metrics = compute_metrics(truth, top5_predictions, manifest["labels"])
    metrics["cross_entropy_loss"] = total_loss / len(truth)
    metrics["latency_ms_per_sample"] = {
        "mean": statistics.fmean(latencies),
        "median": statistics.median(latencies),
        "p95": _percentile(latencies, 0.95),
        "measure": "batched model forward time divided by batch size",
    }
    result = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "held_out_evaluation",
        "split": split,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "dataset_manifest": str(config["dataset"]["manifest"].resolve()),
        "dataset_manifest_sha256": sha256_file(config["dataset"]["manifest"]),
        "dataset_content_sha256": manifest["content_sha256"],
        "schema": schema_metadata(),
        "schema_sha256": canonical_json_sha256(schema_metadata()),
        "labels": manifest["labels"],
        "labels_sha256": manifest["labels_sha256"],
        "split_provenance": {
            "signers": sorted({str(sample["signer_id"]) for sample in selected}),
            "sources": sorted({str(sample["source_id"]) for sample in selected}),
            "label_counts": dict(sorted(Counter(str(sample["label"]) for sample in selected).items())),
        },
        "metrics": metrics,
        "predictions": predictions,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    result_path = output_dir / f"{split}_evaluation.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the canonical Pose-TGCN checkpoint on an explicit held-out manifest split."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        checkpoint = args.checkpoint if args.checkpoint.is_absolute() else Path.cwd() / args.checkpoint
        output_dir = args.output_dir
        if output_dir is None:
            stamp = datetime.now(timezone.utc).strftime("eval-%Y%m%dT%H%M%SZ")
            output_dir = config["artifacts"]["output_root"] / stamp
        elif not output_dir.is_absolute():
            output_dir = Path.cwd() / output_dir
        result_path = evaluate_checkpoint(
            checkpoint.resolve(), config, args.split, output_dir.resolve(), args.device
        )
    except (ContractError, ImportError, OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")
    print(f"Wrote held-out evaluation: {result_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
