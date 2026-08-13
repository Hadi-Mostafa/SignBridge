"""Explicitly fetch and verify the optional local next-word model assets."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import WORD_SUGGESTION_MODEL_DIR  # noqa: E402
from backend.services.context_word_predictor import (  # noqa: E402
    MODEL_REPO,
    MODEL_REVISION,
    MODEL_SHA256,
    BIGRAM_SHA256,
    BIGRAM_URL,
    UNIGRAM_SHA256,
    UNIGRAM_URL,
    ContextWordPredictor,
)


FILES = (
    "config.json",
    "generation_config.json",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "onnx/decoder_model_merged_quantized.onnx",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Install huggingface-hub before fetching model assets.") from exc

    WORD_SUGGESTION_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for filename in FILES:
        cached = Path(
            hf_hub_download(MODEL_REPO, filename=filename, revision=MODEL_REVISION)
        )
        destination = WORD_SUGGESTION_MODEL_DIR / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, destination)

    for filename, url, expected_hash in (
        ("count_2w.txt", BIGRAM_URL, BIGRAM_SHA256),
        ("count_1w.txt", UNIGRAM_URL, UNIGRAM_SHA256),
    ):
        destination = WORD_SUGGESTION_MODEL_DIR / filename
        temporary = destination.with_suffix(destination.suffix + ".download")
        urllib.request.urlretrieve(url, temporary)
        if sha256(temporary) != expected_hash:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"Downloaded {filename} failed SHA-256 verification.")
        temporary.replace(destination)

    model_path = WORD_SUGGESTION_MODEL_DIR / FILES[-1]
    actual_hash = sha256(model_path)
    if actual_hash != MODEL_SHA256:
        raise RuntimeError("Downloaded context model failed SHA-256 verification.")

    status = ContextWordPredictor.asset_status(verify=True)
    print(
        json.dumps(
            {
                "repo_id": MODEL_REPO,
                "revision": MODEL_REVISION,
                "bytes": model_path.stat().st_size,
                "sha256": actual_hash,
                "neural_asset_ready": status["neural_asset_ready"],
                "note": "Runtime endpoints never invoke this downloader.",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
