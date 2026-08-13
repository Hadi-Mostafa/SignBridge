"""Train a production-compatible Pose-TGCN model on a verified Pose-55 manifest.

The default command intentionally fails until captured, signer/source-disjoint
OpenPose data has been extracted.  Legacy ``data/keypoints`` arrays (including
the repository's synthetic arrays) are never scanned or accepted.
"""

from __future__ import annotations

import argparse
import json
import random
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
    JOINT_COUNT,
    SCHEMA_NAME,
    SEQUENCE_LENGTH,
    canonical_json_sha256,
    ensure_non_runtime_output,
    schema_metadata,
    sha256_file,
    validate_dataset_manifest,
)


DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")
PRODUCTION_ARCHITECTURE = {
    "name": "pose-tgcn-wlasl100",
    "input_feature": SEQUENCE_LENGTH * 2,
    "hidden_feature": 64,
    "num_classes": 100,
    "dropout": 0.3,
    "num_stages": 20,
    "residual": True,
}


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"Training config '{name}' must be a mapping.")
    return value


def _project_path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"Training config '{field}' must be a non-empty path string.")
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ContractError("PyYAML is required to read training/config.yaml.") from exc
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ContractError(f"Cannot read training config {path}: {exc}") from exc
    root = _mapping(document, "root")
    contract = _mapping(root.get("contract"), "contract")
    dataset = _mapping(root.get("dataset"), "dataset")
    model = _mapping(root.get("model"), "model")
    training = _mapping(root.get("training"), "training")
    artifacts = _mapping(root.get("artifacts"), "artifacts")

    expected_contract = {
        "schema": SCHEMA_NAME,
        "sequence_length": SEQUENCE_LENGTH,
        "joint_count": JOINT_COUNT,
        "coordinates": 2,
        "normalization": "2 * (coordinate / 256 - 0.5)",
    }
    observed_contract = {key: contract.get(key) for key in expected_contract}
    if observed_contract != expected_contract:
        raise ContractError(
            f"config.yaml contract must exactly match production: {expected_contract}; got {observed_contract}."
        )
    observed_model = {key: model.get(key) for key in PRODUCTION_ARCHITECTURE}
    if observed_model != PRODUCTION_ARCHITECTURE:
        raise ContractError(
            "config.yaml model must exactly match the deployed Pose-TGCN architecture: "
            f"{PRODUCTION_ARCHITECTURE}; got {observed_model}."
        )

    minimums = _mapping(dataset.get("minimum_samples_per_label"), "dataset.minimum_samples_per_label")
    parsed_minimums = {split: int(minimums.get(split, 0)) for split in ("train", "val", "test")}
    if any(value < 1 for value in parsed_minimums.values()):
        raise ContractError(f"Every dataset minimum must be >= 1, got {parsed_minimums}.")
    parsed_training = {
        "batch_size": int(training.get("batch_size", 0)),
        "learning_rate": float(training.get("learning_rate", 0)),
        "weight_decay": float(training.get("weight_decay", -1)),
        "epochs": int(training.get("epochs", 0)),
        "early_stopping_patience": int(training.get("early_stopping_patience", 0)),
        "random_seed": int(training.get("random_seed", 42)),
        "num_workers": int(training.get("num_workers", 0)),
    }
    if parsed_training["batch_size"] < 1 or parsed_training["epochs"] < 1:
        raise ContractError("training.batch_size and training.epochs must be positive integers.")
    if parsed_training["learning_rate"] <= 0 or parsed_training["weight_decay"] < 0:
        raise ContractError("training.learning_rate must be > 0 and weight_decay must be >= 0.")
    if parsed_training["early_stopping_patience"] < 1 or parsed_training["num_workers"] < 0:
        raise ContractError("early_stopping_patience must be >= 1 and num_workers must be >= 0.")
    return {
        "contract": dict(contract),
        "dataset": {
            "manifest": _project_path(dataset.get("manifest"), "dataset.manifest"),
            "labels": _project_path(dataset.get("labels"), "dataset.labels"),
            "minimum_samples_per_label": parsed_minimums,
        },
        "model": dict(model),
        "training": parsed_training,
        "artifacts": {"output_root": _project_path(artifacts.get("output_root"), "artifacts.output_root")},
        "source": document,
        "path": path.resolve(),
    }


def sequence_to_model_input(sequence: Any):
    """Convert stored (50,55,2) data to the deployed model's (55,100) layout."""
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise ContractError("NumPy is required for training.") from exc
    array = np.asarray(sequence)
    if array.shape != (SEQUENCE_LENGTH, JOINT_COUNT, 2):
        raise ContractError(f"Expected sequence shape (50, 55, 2), got {array.shape}.")
    if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
        raise ContractError("Sequence must contain finite floating-point coordinates.")
    if float(array.min()) < -1.00001 or float(array.max()) > 1.00001:
        raise ContractError("Sequence coordinates must be normalized to [-1, 1].")
    return np.ascontiguousarray(array.transpose(1, 0, 2).reshape(JOINT_COUNT, SEQUENCE_LENGTH * 2), dtype=np.float32)


class PoseSequenceDataset:
    def __init__(self, samples: Sequence[Mapping[str, Any]]) -> None:
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        try:
            import numpy as np
            import torch
        except ImportError as exc:  # pragma: no cover
            raise ContractError("NumPy and PyTorch are required for training.") from exc
        sample = self.samples[index]
        sequence = np.load(sample["path"], allow_pickle=False)
        tensor = torch.from_numpy(sequence_to_model_input(sequence))
        return tensor, int(sample["label_index"]), str(sample["sample_id"])


def build_model(device: Any):
    try:
        from backend.models.word_model import GCN_muti_att
    except ImportError as exc:
        raise ContractError(f"Cannot import the production Pose-TGCN architecture: {exc}") from exc
    model = GCN_muti_att(
        input_feature=PRODUCTION_ARCHITECTURE["input_feature"],
        hidden_feature=PRODUCTION_ARCHITECTURE["hidden_feature"],
        num_class=PRODUCTION_ARCHITECTURE["num_classes"],
        p_dropout=PRODUCTION_ARCHITECTURE["dropout"],
        num_stage=PRODUCTION_ARCHITECTURE["num_stages"],
        is_resi=PRODUCTION_ARCHITECTURE["residual"],
    )
    return model.to(device)


def _run_epoch(model: Any, loader: Any, criterion: Any, device: Any, optimizer: Any = None) -> dict[str, float]:
    import torch

    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    top1_correct = 0
    top5_correct = 0
    count = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for inputs, targets, _sample_ids in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            if logits.ndim != 2 or logits.shape[1] != PRODUCTION_ARCHITECTURE["num_classes"]:
                raise ContractError(f"Pose-TGCN returned invalid logits shape {tuple(logits.shape)}.")
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()
            batch = targets.numel()
            total_loss += float(loss.detach()) * batch
            predictions = logits.topk(k=5, dim=1).indices
            top1_correct += int((predictions[:, 0] == targets).sum())
            top5_correct += int((predictions == targets.unsqueeze(1)).any(dim=1).sum())
            count += batch
    if count == 0:
        raise ContractError("A requested dataset split has no samples.")
    return {
        "loss": total_loss / count,
        "top1_accuracy": top1_correct / count,
        "top5_accuracy": top5_correct / count,
        "samples": count,
    }


def _select_device(requested: str, torch: Any):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ContractError("--device cuda was requested, but CUDA is unavailable. Use --device cpu or auto.")
    return torch.device(requested)


def _split_counts(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    totals = Counter(str(sample["split"]) for sample in samples)
    by_label = Counter((str(sample["label"]), str(sample["split"])) for sample in samples)
    return {
        "totals": {split: totals[split] for split in ("train", "val", "test")},
        "per_label": {
            label: {split: by_label[(label, split)] for split in ("train", "val", "test")}
            for label in sorted({str(sample["label"]) for sample in samples})
        },
    }


def train_pose_tgcn(config: dict[str, Any], output_dir: Path, device_name: str = "auto") -> Path:
    try:
        import numpy as np
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise ContractError("NumPy and PyTorch are required to train Pose-TGCN.") from exc

    ensure_non_runtime_output(output_dir)
    if output_dir.exists():
        raise ContractError(f"Output directory already exists: {output_dir}. A training run never overwrites artifacts.")
    manifest, samples = validate_dataset_manifest(
        config["dataset"]["manifest"],
        config["dataset"]["labels"],
        minimum_per_label_split=config["dataset"]["minimum_samples_per_label"],
        verify_arrays=True,
    )
    settings = config["training"]
    seed = settings["random_seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:  # older supported torch
        torch.use_deterministic_algorithms(True)
    device = _select_device(device_name, torch)

    split_samples = {
        split: [sample for sample in samples if sample["split"] == split]
        for split in ("train", "val", "test")
    }
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        PoseSequenceDataset(split_samples["train"]),
        batch_size=settings["batch_size"],
        shuffle=True,
        num_workers=settings["num_workers"],
        generator=generator,
    )
    val_loader = DataLoader(
        PoseSequenceDataset(split_samples["val"]),
        batch_size=settings["batch_size"],
        shuffle=False,
        num_workers=settings["num_workers"],
    )
    model = build_model(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings["epochs"])

    history: list[dict[str, Any]] = []
    best_state: dict[str, Any] | None = None
    best_epoch = 0
    best_top1 = -1.0
    best_loss = float("inf")
    stale_epochs = 0
    for epoch in range(1, settings["epochs"] + 1):
        started = time.perf_counter()
        train_metrics = _run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = _run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        entry = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": val_metrics,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "duration_seconds": time.perf_counter() - started,
        }
        history.append(entry)
        improved = val_metrics["top1_accuracy"] > best_top1 or (
            val_metrics["top1_accuracy"] == best_top1 and val_metrics["loss"] < best_loss
        )
        if improved:
            best_epoch = epoch
            best_top1 = val_metrics["top1_accuracy"]
            best_loss = val_metrics["loss"]
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.5f} "
            f"val_loss={val_metrics['loss']:.5f} val_top1={val_metrics['top1_accuracy']:.4f} "
            f"val_top5={val_metrics['top5_accuracy']:.4f}"
        )
        if stale_epochs >= settings["early_stopping_patience"]:
            break
    if best_state is None:
        raise ContractError("Training completed without a valid checkpoint state.")

    output_dir.mkdir(parents=True, exist_ok=False)
    labels = list(manifest["labels"])
    effective_config = json.loads(json.dumps(config["source"]))
    effective_config["training"] = dict(settings)
    metadata = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "training_artifact_not_runtime_asset",
        "architecture": PRODUCTION_ARCHITECTURE,
        "architecture_sha256": canonical_json_sha256(PRODUCTION_ARCHITECTURE),
        "schema": schema_metadata(),
        "schema_sha256": canonical_json_sha256(schema_metadata()),
        "labels": labels,
        "labels_sha256": manifest["labels_sha256"],
        "dataset": {
            "name": manifest["dataset"]["name"],
            "version": manifest["dataset"].get("version", "unspecified"),
            "manifest_path": str(config["dataset"]["manifest"].resolve()),
            "manifest_sha256": sha256_file(config["dataset"]["manifest"]),
            "content_sha256": manifest["content_sha256"],
            "split_counts": _split_counts(samples),
        },
        "training_config": effective_config,
        "training_config_sha256": sha256_file(config["path"]),
        "training_config_content_sha256": canonical_json_sha256(effective_config),
        "random_seed": seed,
        "device": str(device),
        "best_epoch": best_epoch,
        "best_validation_metrics": history[best_epoch - 1]["validation"],
        "history": history,
        "test_split_used_during_training": False,
    }
    checkpoint_path = output_dir / "pose_tgcn_best.pt"
    torch.save({"state_dict": best_state, "metadata": metadata}, checkpoint_path)
    (output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return checkpoint_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train production Pose-TGCN from a verified signer/source-disjoint Pose-55 manifest."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs for a real-data experiment.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.epochs is not None:
            if args.epochs < 1:
                raise ContractError("--epochs must be >= 1.")
            config["training"]["epochs"] = args.epochs
        if args.batch_size is not None:
            if args.batch_size < 1:
                raise ContractError("--batch-size must be >= 1.")
            config["training"]["batch_size"] = args.batch_size
        if args.learning_rate is not None:
            if args.learning_rate <= 0:
                raise ContractError("--learning-rate must be > 0.")
            config["training"]["learning_rate"] = args.learning_rate
        output_dir = args.output_dir
        if output_dir is None:
            stamp = datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ")
            output_dir = config["artifacts"]["output_root"] / stamp
        elif not output_dir.is_absolute():
            output_dir = Path.cwd() / output_dir
        checkpoint = train_pose_tgcn(config, output_dir.resolve(), args.device)
    except (ContractError, ImportError, OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")
    print(f"Wrote isolated training checkpoint: {checkpoint}")
    print("The test split was not used. Run training/evaluate.py for final held-out metrics.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
