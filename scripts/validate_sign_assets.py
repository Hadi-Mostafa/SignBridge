"""Validate the local native-sign manifest without downloading media."""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "sign_assets.json"
DEFAULT_ROOT = PROJECT_ROOT / "data" / "wlasl_words20" / "videos"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe(path: Path) -> dict:
    command = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration,format_name,size:stream=codec_name,codec_type,width,height",
        "-of", "json", str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
    if completed.returncode:
        raise ValueError(f"FFprobe rejected {path.name}: {completed.stderr.strip()}")
    document = json.loads(completed.stdout)
    video = next((item for item in document.get("streams", []) if item.get("codec_type") == "video"), None)
    if not video:
        raise ValueError(f"{path.name} has no decodable video stream.")
    return document


def validate(manifest_path: Path, root: Path) -> list[dict]:
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if document.get("schema_version") not in {1, 2} or not isinstance(document.get("assets"), list):
        raise ValueError("Unsupported sign asset manifest schema.")
    root = root.resolve()
    seen: set[str] = set()
    results: list[dict] = []
    for item in document["assets"]:
        gloss = " ".join(str(item.get("gloss", "")).lower().split())
        if not gloss or gloss in seen:
            raise ValueError(f"Missing or duplicate gloss: {gloss!r}.")
        seen.add(gloss)
        path = (root / str(item.get("relative_path", ""))).resolve()
        path.relative_to(root)
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_hash = sha256(path)
        if actual_hash != str(item.get("sha256", "")).lower():
            raise ValueError(f"SHA-256 mismatch for {gloss}.")
        declared_mime = str(item.get("mime_type", ""))
        guessed_mime = mimetypes.guess_type(path.name)[0]
        if declared_mime != guessed_mime or not declared_mime.startswith(("video/", "image/")):
            raise ValueError(f"MIME mismatch for {gloss}: {declared_mime!r} vs {guessed_mime!r}.")
        metadata = probe(path) if declared_mime.startswith("video/") else {}
        results.append({"gloss": gloss, "path": str(path), "bytes": path.stat().st_size, "sha256": actual_hash, "probe": metadata})
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate declared native sign assets.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    results = validate(args.manifest.resolve(), args.root.resolve())
    print(json.dumps({"status": "ok", "assets": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
