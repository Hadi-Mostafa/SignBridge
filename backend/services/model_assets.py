"""Deterministic, checksum-verified model asset resolution.

The application may run from a source checkout, a pre-populated project cache,
or Hugging Face's normal user cache.  This module checks those locations in a
fixed order and never searches recursively for a conveniently named model.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    from .. import config as app_config
except ImportError:  # pragma: no cover - direct backend-path execution
    import config as app_config


PROJECT_ROOT = Path(app_config.PROJECT_ROOT).resolve()
MODEL_CACHE_DIR = Path(
    getattr(app_config, "MODEL_CACHE_DIR", PROJECT_ROOT / "backend" / "checkpoints")
).resolve()
LOCK_FILE = Path(__file__).resolve().parent.parent / "checkpoints" / "models.lock.json"
_TRUE_VALUES = {"1", "true", "yes", "on"}


class ModelAssetError(RuntimeError):
    """Raised when a required model asset cannot be resolved safely."""


@dataclass(frozen=True)
class ModelAsset:
    name: str
    filename: str
    sha256: str
    canonical_path: str | None = None
    repo_id: str | None = None
    revision: str | None = None
    distribution: str | None = None
    source_note: str | None = None

    @classmethod
    def from_dict(cls, name: str, value: dict[str, Any]) -> "ModelAsset":
        asset = cls(
            name=name,
            filename=str(value["filename"]),
            sha256=str(value["sha256"]).lower(),
            canonical_path=value.get("canonical_path"),
            repo_id=value.get("repo_id"),
            revision=value.get("revision"),
            distribution=value.get("distribution"),
            source_note=value.get("source_note"),
        )
        if len(asset.sha256) != 64 or any(char not in "0123456789abcdef" for char in asset.sha256):
            raise ModelAssetError(f"Asset {name!r} has an invalid SHA256 in {LOCK_FILE}.")
        if bool(asset.repo_id) != bool(asset.revision):
            raise ModelAssetError(
                f"Asset {name!r} must specify both repo_id and immutable revision, or neither."
            )
        return asset


def _load_lock_file() -> dict[str, ModelAsset]:
    try:
        document = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ModelAssetError(f"Model lock file is missing: {LOCK_FILE}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelAssetError(f"Cannot read model lock file {LOCK_FILE}: {exc}") from exc

    if document.get("version") != 1 or not isinstance(document.get("assets"), dict):
        raise ModelAssetError(f"Unsupported or malformed model lock file: {LOCK_FILE}")
    return {
        name: ModelAsset.from_dict(name, value)
        for name, value in document["assets"].items()
    }


def get_asset(name: str) -> ModelAsset:
    assets = _load_lock_file()
    try:
        return assets[name]
    except KeyError as exc:
        raise ModelAssetError(
            f"Unknown model asset {name!r}; available assets: {', '.join(sorted(assets))}."
        ) from exc


def list_assets() -> list[str]:
    return sorted(_load_lock_file())


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: str | Path, expected_sha256: str) -> str:
    candidate = Path(path)
    if not candidate.is_file():
        raise ModelAssetError(f"Model asset is not a readable file: {candidate}")
    actual = sha256_file(candidate)
    expected = expected_sha256.lower()
    if actual != expected:
        raise ModelAssetError(
            f"Checksum mismatch for {candidate}: expected {expected}, got {actual}. "
            "Delete the corrupt file and run scripts/fetch_models.py again."
        )
    return actual


def _hf_cache_path(cache_root: Path, asset: ModelAsset) -> Path:
    repo_folder = f"models--{asset.repo_id.replace('/', '--')}"
    return cache_root / repo_folder / "snapshots" / str(asset.revision) / asset.filename


def _canonical_path(asset: ModelAsset) -> Path | None:
    if not asset.canonical_path:
        return None
    path = (PROJECT_ROOT / asset.canonical_path).resolve()
    try:
        path.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ModelAssetError(
            f"Asset {asset.name!r} canonical_path escapes the project root: {path}"
        ) from exc
    return path


def _candidate_paths(asset: ModelAsset) -> Iterable[Path]:
    canonical = _canonical_path(asset)
    if canonical is not None:
        yield canonical

    if asset.repo_id:
        # Configured project cache first.  LOCK_FILE.parent is retained as a
        # deterministic compatibility cache when configuration moves later.
        cache_roots = [MODEL_CACHE_DIR, LOCK_FILE.parent.resolve()]
        seen_roots: set[Path] = set()
        for cache_root in cache_roots:
            if cache_root not in seen_roots:
                seen_roots.add(cache_root)
                yield _hf_cache_path(cache_root, asset)

        try:
            from huggingface_hub import try_to_load_from_cache

            cached = try_to_load_from_cache(
                repo_id=str(asset.repo_id),
                filename=asset.filename,
                revision=str(asset.revision),
            )
            if isinstance(cached, str):
                yield Path(cached)
        except (ImportError, OSError, ValueError):
            # A missing/broken optional default cache is reported by the final
            # actionable resolver error; it must not mask valid project files.
            pass


def offline_mode() -> bool:
    return os.environ.get("SIGNBRIDGE_OFFLINE", "").strip().lower() in _TRUE_VALUES


def downloads_enabled() -> bool:
    """Return the explicit runtime download policy from configuration.

    ``allow_download`` is a per-call request; this setting is the deployment
    ceiling.  Both must permit network access, and offline mode always wins.
    """

    return bool(getattr(app_config, "MODEL_DOWNLOAD_ENABLED", True)) and not offline_mode()


def resolve_asset(name: str, *, allow_download: bool = True) -> Path:
    """Resolve and verify an asset declared in ``models.lock.json``.

    Resolution order is canonical project path, configured project cache,
    deterministic compatibility project cache, default Hugging Face cache, and
    finally an optional pinned download into the project cache.
    """

    asset = get_asset(name)
    checked: list[str] = []
    seen: set[Path] = set()
    for raw_path in _candidate_paths(asset):
        path = raw_path.resolve()
        if path in seen:
            continue
        seen.add(path)
        if not path.is_file():
            checked.append(f"missing: {path}")
            continue
        try:
            verify_sha256(path, asset.sha256)
        except ModelAssetError as exc:
            checked.append(str(exc))
            continue
        return path

    if allow_download and downloads_enabled() and asset.repo_id:
        try:
            from huggingface_hub import hf_hub_download

            downloaded = Path(
                hf_hub_download(
                    repo_id=str(asset.repo_id),
                    filename=asset.filename,
                    revision=str(asset.revision),
                    cache_dir=str(MODEL_CACHE_DIR),
                    etag_timeout=float(
                        getattr(app_config, "MODEL_DOWNLOAD_TIMEOUT_SECONDS", 90)
                    ),
                )
            ).resolve()
            verify_sha256(downloaded, asset.sha256)
            return downloaded
        except Exception as exc:
            checked.append(f"pinned download failed: {type(exc).__name__}: {exc}")

    if offline_mode():
        mode_hint = "SIGNBRIDGE_OFFLINE is enabled, so downloading was not attempted."
    elif not bool(getattr(app_config, "MODEL_DOWNLOAD_ENABLED", True)):
        mode_hint = "MODEL_DOWNLOAD_ENABLED is false, so downloading was not attempted."
    else:
        mode_hint = "Run `python scripts/fetch_models.py` while online to populate the verified cache."
    details = "\n  - ".join(checked) if checked else "no candidate paths were available"
    raise ModelAssetError(
        f"Required model asset {name!r} could not be resolved and verified.\n"
        f"  - {details}\n{mode_hint}"
    )


def asset_metadata(name: str, path: str | Path | None = None) -> dict[str, Any]:
    asset = get_asset(name)
    metadata: dict[str, Any] = {
        "asset": asset.name,
        "filename": asset.filename,
        "sha256": asset.sha256,
        "repo_id": asset.repo_id,
        "revision": asset.revision,
        "distribution": asset.distribution or ("pinned_remote" if asset.repo_id else "bundled_local"),
    }
    if asset.source_note:
        metadata["source_note"] = asset.source_note
    if path is not None:
        metadata["path"] = str(Path(path).resolve())
    return metadata
