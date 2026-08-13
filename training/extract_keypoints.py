"""Build the production Pose-TGCN input dataset from OpenPose JSON frames.

This module is deliberately a data-contract tool, not a video pose detector.  It
expects OpenPose JSON produced from 256x256 frames and emits one ``.npy`` array
per sample with shape ``(50, 55, 2)`` plus a provenance-rich manifest.

The source manifest must give every sample an explicit train/val/test split,
signer, and source id.  Signers and sources may not cross splits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_NAME = "openpose_body25_filtered_plus_hands_xy_v1"
SCHEMA_VERSION = 1
FRAME_SIZE = 256
SEQUENCE_LENGTH = 50
BODY_25_INDICES = tuple(range(9)) + tuple(range(15, 19))
HAND_JOINTS = 21
JOINT_COUNT = len(BODY_25_INDICES) + 2 * HAND_JOINTS
COORDINATES = ("x", "y")
SPLITS = ("train", "val", "test")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = PROJECT_ROOT / "backend" / "checkpoints" / "wlasl100.txt"
DEFAULT_SOURCE_MANIFEST = PROJECT_ROOT / "data" / "openpose" / "source_manifest.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "pose55"
RUNTIME_ASSET_DIR = PROJECT_ROOT / "backend" / "checkpoints"


class ContractError(ValueError):
    """Raised when data cannot honestly satisfy the production contract."""


def schema_metadata() -> dict[str, Any]:
    return {
        "name": SCHEMA_NAME,
        "version": SCHEMA_VERSION,
        "sequence_length": SEQUENCE_LENGTH,
        "joint_count": JOINT_COUNT,
        "coordinates": list(COORDINATES),
        "body_25_indices": list(BODY_25_INDICES),
        "left_hand_indices": list(range(HAND_JOINTS)),
        "right_hand_indices": list(range(HAND_JOINTS)),
        "joint_order": "BODY_25[0..8,15..18], left hand[0..20], right hand[0..20]",
        "normalization": "2 * (coordinate / 256 - 0.5)",
        "input_frame_size": [FRAME_SIZE, FRAME_SIZE],
        "stored_shape": [SEQUENCE_LENGTH, JOINT_COUNT, len(COORDINATES)],
        "model_shape": [JOINT_COUNT, SEQUENCE_LENGTH * len(COORDINATES)],
        "temporal_resampling": "nearest indices from linspace(first,last,50)",
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


def load_labels(path: Path, expected_classes: int = 100) -> list[str]:
    try:
        labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    except OSError as exc:
        raise ContractError(f"Cannot read label file {path}: {exc}") from exc
    if len(labels) != expected_classes:
        raise ContractError(
            f"Expected exactly {expected_classes} labels in {path}, found {len(labels)}."
        )
    if any(not label for label in labels) or len(set(labels)) != len(labels):
        raise ContractError(f"Labels in {path} must be non-empty and unique.")
    if labels != sorted(labels):
        raise ContractError(
            f"Labels in {path} must use the alphabetically sorted WLASL LabelEncoder order."
        )
    return labels


def _require_text(sample: Mapping[str, Any], field: str, sample_number: int) -> str:
    value = sample.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"Source sample {sample_number} needs a non-empty '{field}'.")
    return value.strip()


def _looks_synthetic(sample: Mapping[str, Any]) -> bool:
    fields = (
        str(sample.get("sample_id", "")),
        str(sample.get("source_id", "")),
        str(sample.get("openpose_dir", "")),
    )
    return bool(sample.get("synthetic")) or any("synthetic" in value.lower() for value in fields)


def validate_source_manifest(
    document: Mapping[str, Any],
    labels: Sequence[str],
    *,
    require_complete: bool = True,
) -> list[dict[str, Any]]:
    """Validate explicit provenance and return normalized sample dictionaries."""
    if not isinstance(document, Mapping):
        raise ContractError("Source manifest root must be a JSON object.")
    dataset = document.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ContractError("Source manifest needs a 'dataset' object with provenance metadata.")
    if bool(dataset.get("synthetic")):
        raise ContractError("Synthetic datasets are rejected; provide captured OpenPose data.")
    dataset_name = dataset.get("name")
    if not isinstance(dataset_name, str) or not dataset_name.strip():
        raise ContractError("Source manifest dataset.name must identify the captured dataset.")

    raw_samples = document.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        raise ContractError("Source manifest must contain a non-empty 'samples' list.")

    known_labels = set(labels)
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    signer_splits: dict[str, set[str]] = defaultdict(set)
    source_splits: dict[str, set[str]] = defaultdict(set)
    counts: Counter[tuple[str, str]] = Counter()

    for number, raw in enumerate(raw_samples, start=1):
        if not isinstance(raw, Mapping):
            raise ContractError(f"Source sample {number} must be a JSON object.")
        sample_id = _require_text(raw, "sample_id", number)
        label = _require_text(raw, "label", number)
        split = _require_text(raw, "split", number).lower()
        signer_id = _require_text(raw, "signer_id", number)
        source_id = _require_text(raw, "source_id", number)
        openpose_dir = _require_text(raw, "openpose_dir", number)
        if sample_id in seen_ids:
            raise ContractError(f"Duplicate sample_id '{sample_id}' in source manifest.")
        if label not in known_labels:
            raise ContractError(f"Sample '{sample_id}' has unknown label '{label}'.")
        if split not in SPLITS:
            raise ContractError(
                f"Sample '{sample_id}' split must be one of {SPLITS}, got '{split}'."
            )
        if _looks_synthetic(raw):
            raise ContractError(
                f"Sample '{sample_id}' appears synthetic. Synthetic samples are contract tests only, "
                "never training or performance data."
            )
        seen_ids.add(sample_id)
        signer_splits[signer_id].add(split)
        source_splits[source_id].add(split)
        counts[(label, split)] += 1
        normalized.append(
            {
                "sample_id": sample_id,
                "label": label,
                "split": split,
                "signer_id": signer_id,
                "source_id": source_id,
                "openpose_dir": openpose_dir,
            }
        )

    signer_leaks = sorted(key for key, values in signer_splits.items() if len(values) > 1)
    source_leaks = sorted(key for key, values in source_splits.items() if len(values) > 1)
    if signer_leaks:
        raise ContractError(
            "Signer-disjoint split violation for signer_id(s): " + ", ".join(signer_leaks[:10])
        )
    if source_leaks:
        raise ContractError(
            "Source-disjoint split violation for source_id(s): " + ", ".join(source_leaks[:10])
        )

    if require_complete:
        missing = [f"{label}:{split}" for label in labels for split in SPLITS if counts[(label, split)] < 1]
        if missing:
            preview = ", ".join(missing[:12])
            raise ContractError(
                "Dataset is incomplete: every label needs at least one captured sample in each "
                f"explicit split. Missing {len(missing)} label/split cells (first: {preview})."
            )
    return normalized


def validate_dataset_manifest(
    manifest_path: Path,
    labels_path: Path,
    *,
    minimum_per_label_split: int | Mapping[str, int] = 1,
    verify_arrays: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate an extracted manifest and every referenced array.

    No paths are discovered by scanning: the signed-off manifest is the complete
    dataset.  This prevents accidentally training on stale or synthetic files
    that happen to share a directory.
    """
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise ContractError("NumPy is required. Install the project's training dependencies.") from exc
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(
            f"Cannot read extracted dataset manifest {manifest_path}: {exc}. Run "
            "training/extract_keypoints.py on captured OpenPose JSON first."
        ) from exc
    if not isinstance(document, dict) or document.get("format_version") != 1:
        raise ContractError("Extracted manifest must be a format_version 1 JSON object.")
    dataset = document.get("dataset")
    if not isinstance(dataset, Mapping) or bool(dataset.get("synthetic")):
        raise ContractError("Extracted dataset must declare non-synthetic captured provenance.")
    if document.get("schema") != schema_metadata():
        raise ContractError(
            f"Dataset schema does not exactly match production '{SCHEMA_NAME}' version {SCHEMA_VERSION}."
        )
    expected_labels = load_labels(labels_path)
    if document.get("labels") != expected_labels:
        raise ContractError("Dataset labels/order do not exactly match the production WLASL-100 label file.")
    expected_labels_hash = sha256_file(labels_path)
    if document.get("labels_sha256") != expected_labels_hash:
        raise ContractError("Dataset labels_sha256 does not match the production label asset.")
    declared_content_hash = document.get("content_sha256")
    computed_content_hash = canonical_json_sha256(
        {key: value for key, value in document.items() if key not in {"created_at", "content_sha256"}}
    )
    if declared_content_hash != computed_content_hash:
        raise ContractError(
            "Dataset manifest content hash is missing or invalid; regenerate it instead of hand-editing it."
        )

    minimums = (
        {split: int(minimum_per_label_split) for split in SPLITS}
        if isinstance(minimum_per_label_split, int)
        else {split: int(minimum_per_label_split.get(split, 0)) for split in SPLITS}
    )
    if any(value < 1 for value in minimums.values()):
        raise ContractError(f"Minimum samples per label/split must be >= 1 for all splits, got {minimums}.")

    raw_samples = document.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        raise ContractError("Extracted manifest has no samples.")
    root = manifest_path.resolve().parent
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    signer_splits: dict[str, set[str]] = defaultdict(set)
    source_splits: dict[str, set[str]] = defaultdict(set)
    counts: Counter[tuple[str, str]] = Counter()
    label_to_index = {label: index for index, label in enumerate(expected_labels)}

    for number, raw in enumerate(raw_samples, start=1):
        if not isinstance(raw, Mapping):
            raise ContractError(f"Extracted sample {number} must be a JSON object.")
        sample_id = _require_text(raw, "sample_id", number)
        label = _require_text(raw, "label", number)
        split = _require_text(raw, "split", number).lower()
        signer_id = _require_text(raw, "signer_id", number)
        source_id = _require_text(raw, "source_id", number)
        relative_text = _require_text(raw, "path", number)
        if _looks_synthetic({**raw, "openpose_dir": relative_text}):
            raise ContractError(f"Extracted sample '{sample_id}' appears synthetic and is rejected.")
        if sample_id in seen_ids:
            raise ContractError(f"Duplicate extracted sample_id '{sample_id}'.")
        if label not in label_to_index or raw.get("label_index") != label_to_index.get(label):
            raise ContractError(f"Sample '{sample_id}' label/index does not match production label order.")
        if split not in SPLITS:
            raise ContractError(f"Sample '{sample_id}' has invalid split '{split}'.")
        relative_path = Path(relative_text)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ContractError(f"Sample '{sample_id}' path must be relative and may not traverse directories.")
        array_path = (root / relative_path).resolve()
        if root != array_path and root not in array_path.parents:
            raise ContractError(f"Sample '{sample_id}' resolves outside the dataset directory.")
        if not array_path.is_file():
            raise ContractError(f"Sample '{sample_id}' array is missing: {array_path}")
        declared_hash = raw.get("sha256")
        actual_hash = sha256_file(array_path)
        if declared_hash != actual_hash:
            raise ContractError(f"Sample '{sample_id}' SHA256 mismatch for {array_path}.")
        if verify_arrays:
            try:
                sequence = np.load(array_path, allow_pickle=False)
            except (OSError, ValueError) as exc:
                raise ContractError(f"Cannot load sample '{sample_id}' array {array_path}: {exc}") from exc
            if sequence.shape != (SEQUENCE_LENGTH, JOINT_COUNT, 2):
                raise ContractError(
                    f"Sample '{sample_id}' shape is {sequence.shape}; expected (50, 55, 2)."
                )
            if not np.issubdtype(sequence.dtype, np.floating) or not np.isfinite(sequence).all():
                raise ContractError(f"Sample '{sample_id}' must contain finite floating-point coordinates.")
            if float(sequence.min()) < -1.00001 or float(sequence.max()) > 1.00001:
                raise ContractError(f"Sample '{sample_id}' is outside normalized [-1, 1] coordinates.")
        seen_ids.add(sample_id)
        signer_splits[signer_id].add(split)
        source_splits[source_id].add(split)
        counts[(label, split)] += 1
        normalized.append({**dict(raw), "path": array_path})

    signer_leaks = sorted(key for key, values in signer_splits.items() if len(values) > 1)
    source_leaks = sorted(key for key, values in source_splits.items() if len(values) > 1)
    if signer_leaks:
        raise ContractError("Signer-disjoint split violation: " + ", ".join(signer_leaks[:10]))
    if source_leaks:
        raise ContractError("Source-disjoint split violation: " + ", ".join(source_leaks[:10]))
    missing = [
        f"{label}:{split}({counts[(label, split)]}/{minimums[split]})"
        for label in expected_labels
        for split in SPLITS
        if counts[(label, split)] < minimums[split]
    ]
    if missing:
        raise ContractError(
            "Dataset is insufficient for training/evaluation. Required captured samples per label are "
            f"{minimums}; {len(missing)} cells are short (first: {', '.join(missing[:12])})."
        )
    return document, normalized


def _triples(values: Any, expected_joints: int, field: str, frame_path: Path):
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise ContractError("NumPy is required. Install the project's training dependencies.") from exc
    if not isinstance(values, list) or len(values) < expected_joints * 3:
        size = len(values) if isinstance(values, list) else "non-list"
        raise ContractError(
            f"{frame_path}: '{field}' needs at least {expected_joints * 3} values, got {size}."
        )
    try:
        array = np.asarray(values[: expected_joints * 3], dtype=np.float32).reshape(expected_joints, 3)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{frame_path}: '{field}' contains invalid numeric values.") from exc
    if not np.isfinite(array).all():
        raise ContractError(f"{frame_path}: '{field}' contains NaN or infinity.")
    return array


def _person_score(person: Mapping[str, Any], frame_path: Path) -> float:
    body = _triples(person.get("pose_keypoints_2d"), 25, "pose_keypoints_2d", frame_path)
    left = _triples(person.get("hand_left_keypoints_2d"), HAND_JOINTS, "hand_left_keypoints_2d", frame_path)
    right = _triples(person.get("hand_right_keypoints_2d"), HAND_JOINTS, "hand_right_keypoints_2d", frame_path)
    return float(body[:, 2].sum() + left[:, 2].sum() + right[:, 2].sum())


def parse_openpose_frame(frame_path: Path):
    """Return one frame as 55 xy points normalized to the production [-1, 1] range."""
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise ContractError("NumPy is required. Install the project's training dependencies.") from exc
    try:
        document = json.loads(frame_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"Cannot parse OpenPose frame {frame_path}: {exc}") from exc
    people = document.get("people") if isinstance(document, Mapping) else None
    if not isinstance(people, list) or not people:
        raise ContractError(f"{frame_path}: no OpenPose person was detected.")
    valid_people = [person for person in people if isinstance(person, Mapping)]
    if not valid_people:
        raise ContractError(f"{frame_path}: OpenPose 'people' contains no valid person object.")
    person = max(valid_people, key=lambda item: _person_score(item, frame_path))
    body = _triples(person.get("pose_keypoints_2d"), 25, "pose_keypoints_2d", frame_path)
    left = _triples(person.get("hand_left_keypoints_2d"), HAND_JOINTS, "hand_left_keypoints_2d", frame_path)
    right = _triples(person.get("hand_right_keypoints_2d"), HAND_JOINTS, "hand_right_keypoints_2d", frame_path)
    xy = np.concatenate((body[list(BODY_25_INDICES), :2], left[:, :2], right[:, :2]), axis=0)
    if xy.shape != (JOINT_COUNT, 2):
        raise ContractError(f"{frame_path}: internal joint selection produced {xy.shape}, expected (55, 2).")
    minimum, maximum = float(xy.min()), float(xy.max())
    if minimum < 0.0 or maximum > float(FRAME_SIZE):
        raise ContractError(
            f"{frame_path}: xy coordinates [{minimum:.4g}, {maximum:.4g}] are outside 256x256. "
            "Run OpenPose on frames resized to exactly 256x256; coordinates are not silently clipped."
        )
    normalized = 2.0 * (xy / float(FRAME_SIZE) - 0.5)
    return np.ascontiguousarray(normalized, dtype=np.float32)


def resample_sequence(frames: Sequence[Any], target_frames: int = SEQUENCE_LENGTH):
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise ContractError("NumPy is required. Install the project's training dependencies.") from exc
    if target_frames != SEQUENCE_LENGTH:
        raise ContractError(f"Production contract requires exactly {SEQUENCE_LENGTH} frames.")
    if not hasattr(frames, "__len__") or len(frames) == 0:
        raise ContractError("Cannot resample an empty OpenPose sequence.")
    sequence = np.asarray(frames, dtype=np.float32)
    if sequence.ndim != 3 or sequence.shape[1:] != (JOINT_COUNT, 2):
        raise ContractError(f"Expected raw sequence shaped (frames, 55, 2), got {sequence.shape}.")
    indices = np.rint(np.linspace(0, len(sequence) - 1, target_frames)).astype(np.int64)
    output = np.ascontiguousarray(sequence[indices], dtype=np.float32)
    if output.shape != (SEQUENCE_LENGTH, JOINT_COUNT, 2):
        raise ContractError(f"Resampling produced invalid shape {output.shape}.")
    if not np.isfinite(output).all() or float(output.min()) < -1.00001 or float(output.max()) > 1.00001:
        raise ContractError("Resampled sequence violates finite normalized [-1, 1] contract.")
    return output


def _natural_key(path: Path) -> list[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def load_openpose_sequence(frames_dir: Path):
    frame_paths = sorted(frames_dir.glob("*_keypoints.json"), key=_natural_key)
    if not frame_paths:
        frame_paths = sorted(frames_dir.glob("*.json"), key=_natural_key)
    if not frame_paths:
        raise ContractError(f"No OpenPose JSON frames found in {frames_dir}.")
    return resample_sequence([parse_openpose_frame(path) for path in frame_paths]), len(frame_paths)


def ensure_non_runtime_output(path: Path) -> None:
    resolved = path.resolve()
    runtime = RUNTIME_ASSET_DIR.resolve()
    if resolved == runtime or runtime in resolved.parents:
        raise ContractError(
            f"Refusing output under runtime asset directory {runtime}. Use data/ or artifacts/ instead."
        )


def _safe_filename(sample_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", sample_id).strip("._") or "sample"
    suffix = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:10]
    return f"{slug[:80]}-{suffix}.npy"


def extract_dataset(
    source_manifest_path: Path,
    output_dir: Path,
    labels_path: Path,
    *,
    require_complete: bool = True,
) -> Path:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise ContractError("NumPy is required. Install the project's training dependencies.") from exc
    ensure_non_runtime_output(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ContractError(
            f"Output directory {output_dir} is not empty. Choose a new directory; extraction never overwrites data."
        )
    labels = load_labels(labels_path)
    try:
        source_document = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(
            f"Cannot read source manifest {source_manifest_path}: {exc}. Create it with explicit "
            "sample_id, label, split, signer_id, source_id, and openpose_dir fields."
        ) from exc
    samples = validate_source_manifest(source_document, labels, require_complete=require_complete)
    dataset = source_document["dataset"]
    manifest_samples: list[dict[str, Any]] = []
    for sample in samples:
        frames_dir = Path(sample["openpose_dir"])
        if not frames_dir.is_absolute():
            frames_dir = source_manifest_path.parent / frames_dir
        sequence, source_frame_count = load_openpose_sequence(frames_dir.resolve())
        relative_path = Path("sequences") / sample["split"] / _safe_filename(sample["sample_id"])
        destination = output_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.save(destination, sequence, allow_pickle=False)
        manifest_samples.append(
            {
                "sample_id": sample["sample_id"],
                "label": sample["label"],
                "label_index": labels.index(sample["label"]),
                "split": sample["split"],
                "signer_id": sample["signer_id"],
                "source_id": sample["source_id"],
                "path": relative_path.as_posix(),
                "source_frame_count": source_frame_count,
                "sha256": sha256_file(destination),
            }
        )
    output_manifest = {
        "format_version": 1,
        "dataset": {
            "name": str(dataset["name"]),
            "version": str(dataset.get("version", "unspecified")),
            "synthetic": False,
            "source_manifest": str(source_manifest_path.resolve()),
            "source_manifest_sha256": sha256_file(source_manifest_path),
        },
        "schema": schema_metadata(),
        "labels": labels,
        "labels_sha256": sha256_file(labels_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "samples": manifest_samples,
    }
    output_manifest["content_sha256"] = canonical_json_sha256(
        {key: value for key, value in output_manifest.items() if key not in {"created_at", "content_sha256"}}
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    destination_manifest = output_dir / "manifest.json"
    destination_manifest.write_text(json.dumps(output_manifest, indent=2) + "\n", encoding="utf-8")
    return destination_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract production-compatible 50x55x2 Pose-TGCN sequences from OpenPose JSON."
    )
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Permit missing label/split cells for preprocessing diagnostics only; train/eval still reject them.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        manifest = extract_dataset(
            args.source_manifest,
            args.output_dir,
            args.labels,
            require_complete=not args.allow_incomplete,
        )
    except ContractError as exc:
        parser.exit(2, f"ERROR: {exc}\n")
    print(f"Wrote verified Pose-TGCN dataset manifest: {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
