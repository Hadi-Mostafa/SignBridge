"""Offline-first diagnostic utility for the SignBridge runtime.

This script is intentionally independent of the web server.  It verifies the
installed Python packages, locked model bytes, label assets, frontend files and
known dataset limitations.  With ``--load-models`` it also constructs both
recognizers and executes their deterministic startup self-tests.

Examples from the project root::

    python scripts/doctor.py
    python scripts/doctor.py --load-models
    python scripts/doctor.py --json --load-models
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import sys
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


@dataclass
class Check:
    name: str
    status: str
    message: str
    details: dict[str, Any] | None = None


def _package_check(distribution: str, import_name: str | None = None) -> Check:
    try:
        version = importlib.metadata.version(distribution)
        if import_name:
            __import__(import_name)
        return Check(f"dependency:{distribution}", "pass", f"installed {version}")
    except (importlib.metadata.PackageNotFoundError, ImportError) as exc:
        return Check(
            f"dependency:{distribution}",
            "fail",
            f"missing or not importable: {type(exc).__name__}",
        )


def _dependency_checks() -> list[Check]:
    required = [
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("numpy", "numpy"),
        ("opencv-python", "cv2"),
        ("mediapipe", "mediapipe"),
        ("torch", "torch"),
        ("scikit-learn", "sklearn"),
        ("joblib", "joblib"),
        ("huggingface-hub", "huggingface_hub"),
        ("python-multipart", "multipart"),
        ("Pillow", "PIL"),
        ("openai-whisper", "whisper"),
    ]
    return [_package_check(distribution, module) for distribution, module in required]


def _asset_checks(allow_download: bool) -> list[Check]:
    try:
        from services.model_assets import (
            get_asset,
            list_assets,
            resolve_asset,
            sha256_file,
        )
    except Exception as exc:
        return [Check("model-assets", "fail", f"asset resolver import failed: {type(exc).__name__}: {exc}")]

    checks: list[Check] = []
    for name in list_assets():
        try:
            asset = get_asset(name)
            path = resolve_asset(name, allow_download=allow_download)
            checks.append(
                Check(
                    f"model-asset:{name}",
                    "pass",
                    "present and SHA-256 verified",
                    {
                        "path": str(path),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                        "revision": asset.revision,
                    },
                )
            )
        except Exception as exc:
            checks.append(Check(f"model-asset:{name}", "fail", f"{type(exc).__name__}: {exc}"))
    return checks


def _frontend_checks() -> list[Check]:
    checks: list[Check] = []
    for relative in ("frontend/index.html", "frontend/app.js", "frontend/style.css"):
        path = PROJECT_ROOT / relative
        if path.is_file() and path.stat().st_size > 0:
            checks.append(Check(f"frontend:{relative}", "pass", f"present ({path.stat().st_size} bytes)"))
        else:
            checks.append(Check(f"frontend:{relative}", "fail", "missing or empty"))
    return checks


def _dataset_checks() -> list[Check]:
    checks: list[Check] = []

    manifest_path = PROJECT_ROOT / "data" / "wlasl_words20" / "manifest.json"
    if manifest_path.is_file():
        try:
            document = json.loads(manifest_path.read_text(encoding="utf-8"))
            samples = document.get("samples", [])
            available = [sample for sample in samples if sample.get("available")]
            status = "warn" if len(available) < len(samples) else "pass"
            checks.append(
                Check(
                    "dataset:wlasl_words20",
                    status,
                    f"{len(available)}/{len(samples)} manifest videos available; insufficient for training",
                )
            )
        except (OSError, json.JSONDecodeError, AttributeError) as exc:
            checks.append(Check("dataset:wlasl_words20", "warn", f"manifest unreadable: {exc}"))
    else:
        checks.append(Check("dataset:wlasl_words20", "warn", "manifest absent"))

    synthetic_dir = PROJECT_ROOT / "data" / "keypoints"
    synthetic_count = sum(1 for _ in synthetic_dir.rglob("*.npy")) if synthetic_dir.is_dir() else 0
    checks.append(
        Check(
            "dataset:legacy_keypoints",
            "warn",
            f"{synthetic_count} legacy synthetic arrays; excluded from production evaluation",
        )
    )

    archive = PROJECT_ROOT / "test_dataset" / "roboflow.zip"
    if archive.is_file():
        try:
            with zipfile.ZipFile(archive) as source:
                bad_member = source.testzip()
            if bad_member:
                checks.append(Check("dataset:roboflow_archive", "warn", f"corrupt member: {bad_member}"))
            else:
                checks.append(Check("dataset:roboflow_archive", "pass", "ZIP integrity test passed"))
        except (OSError, zipfile.BadZipFile) as exc:
            checks.append(
                Check(
                    "dataset:roboflow_archive",
                    "warn",
                    f"archive is truncated/corrupt and cannot be a benchmark source: {type(exc).__name__}",
                )
            )
    else:
        checks.append(Check("dataset:roboflow_archive", "warn", "archive absent"))

    camera_data = PROJECT_ROOT / "data" / "asl_alphabet"
    checks.append(
        Check(
            "dataset:camera_alphabet",
            "pass" if camera_data.is_dir() else "warn",
            "camera-style alphabet directory present" if camera_data.is_dir() else "camera-style training directory absent",
        )
    )
    return checks


def _speech_checks() -> list[Check]:
    ffmpeg = shutil.which("ffmpeg")
    checks = [
        Check(
            "speech:ffmpeg",
            "pass" if ffmpeg else "warn",
            "FFmpeg executable available" if ffmpeg else "FFmpeg is absent; audio transcription cannot decode uploads",
        )
    ]
    try:
        from services.asr_service import ASRService

        status = ASRService.asset_status()
        checks.append(
            Check(
                "speech:whisper_asset",
                "pass" if status["ready_to_load"] else "warn",
                (
                    f"Whisper {status['model_size']} cache integrity is verified"
                    if status["ready_to_load"]
                    else "Whisper speech model is missing or invalid; run scripts/fetch_speech_model.py while online"
                ),
                status,
            )
        )
    except Exception as exc:
        checks.append(Check("speech:whisper_asset", "warn", f"cannot inspect speech model: {exc}"))
    try:
        from services.sign_lookup import SignLookupService

        lookup = SignLookupService()
        lookup.load_mappings()
        status = lookup.status()
        checks.append(
            Check(
                "sign-assets:manifest",
                "pass" if status["manifest_valid"] else "warn",
                f"{status['local_clip_count']} verified local sign clips",
                status,
            )
        )
        guide_count = int(status.get("fingerspell_guide_count", 0))
        checks.append(
            Check(
                "sign-assets:fingerspell_guides",
                "pass" if status.get("fingerspell_guides_verified") else "warn",
                f"{guide_count}/24 static landmark guides are SHA-256 verified; J/Z require motion",
                {
                    "representation": "schematic_landmark_guide_not_native_media",
                    "verified_static_letters": guide_count,
                    "motion_required": status.get("dynamic_fingerspell_letters", ["J", "Z"]),
                },
            )
        )
    except Exception as exc:
        checks.append(Check("sign-assets:manifest", "warn", f"cannot inspect sign assets: {exc}"))
    return checks


def _model_load_checks() -> list[Check]:
    checks: list[Check] = []
    pipeline = None
    try:
        from services.realtime_pipeline import RealtimeASLPipeline

        pipeline = RealtimeASLPipeline()
        pipeline.load()
        metadata = pipeline.metadata()
        checks.append(Check("model-load:alphabet", "pass", "startup self-test passed", metadata))
    except Exception as exc:
        checks.append(Check("model-load:alphabet", "fail", f"{type(exc).__name__}: {exc}"))
    finally:
        if pipeline is not None:
            pipeline.close()

    try:
        from models.word_model import WordModel

        model = WordModel()
        self_test = model.self_test()
        metadata = model.metadata()
        checks.append(
            Check(
                "model-load:words",
                "pass",
                "startup self-test passed",
                {
                    "self_test": self_test,
                    "name": metadata.get("name"),
                    "revision": metadata.get("revision"),
                    "classes": metadata.get("classes"),
                    "input_shape": metadata.get("input_shape"),
                },
            )
        )
    except Exception as exc:
        checks.append(Check("model-load:words", "fail", f"{type(exc).__name__}: {exc}"))
    return checks


def run(*, allow_download: bool, load_models: bool) -> list[Check]:
    if not allow_download:
        os.environ.setdefault("SIGNBRIDGE_OFFLINE", "1")
    checks = [
        Check(
            "runtime:python",
            "pass" if sys.version_info >= (3, 10) else "fail",
            sys.version.split()[0],
        )
    ]
    checks.extend(_dependency_checks())
    checks.extend(_asset_checks(allow_download))
    checks.extend(_frontend_checks())
    checks.extend(_dataset_checks())
    checks.extend(_speech_checks())
    if load_models:
        checks.extend(_model_load_checks())
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose SignBridge locally without starting the API.")
    parser.add_argument("--load-models", action="store_true", help="construct both models and run self-tests")
    parser.add_argument("--allow-download", action="store_true", help="allow pinned missing assets to download")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()

    checks = run(allow_download=args.allow_download, load_models=args.load_models)
    counts = {status: sum(check.status == status for check in checks) for status in ("pass", "warn", "fail")}
    document = {
        "project": str(PROJECT_ROOT),
        "offline": not args.allow_download,
        "model_self_tests_requested": args.load_models,
        "summary": counts,
        "checks": [asdict(check) for check in checks],
    }

    if args.json:
        print(json.dumps(document, indent=2))
    else:
        for check in checks:
            print(f"[{check.status.upper():4}] {check.name}: {check.message}")
        print(
            f"Summary: {counts['pass']} passed, {counts['warn']} warnings, "
            f"{counts['fail']} failed."
        )
        if not args.load_models:
            print("Tip: add --load-models to execute both inference startup self-tests.")
    return 1 if counts["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
