"""Resolve concept glosses to verified local clips or honest letter fallbacks."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

try:
    from ..config import (
        ENABLE_ONLINE_SIGN_LOOKUP,
        FINGERSPELL_GUIDE_MANIFEST,
        FINGERSPELL_GUIDES_DIR,
        SIGN_ASSET_MANIFEST,
        SIGN_LOOKUP_TIMEOUT_SECONDS,
        SIGN_VIDEOS_DIR,
        VOCABULARY_FILE,
        WLASL_SIGN_VIDEOS_DIR,
    )
except ImportError:  # pragma: no cover - direct backend-path execution
    from config import (
        ENABLE_ONLINE_SIGN_LOOKUP,
        FINGERSPELL_GUIDE_MANIFEST,
        FINGERSPELL_GUIDES_DIR,
        SIGN_ASSET_MANIFEST,
        SIGN_LOOKUP_TIMEOUT_SECONDS,
        SIGN_VIDEOS_DIR,
        VOCABULARY_FILE,
        WLASL_SIGN_VIDEOS_DIR,
    )

logger = logging.getLogger(__name__)
ASCII_LETTERS = frozenset("abcdefghijklmnopqrstuvwxyz")
VIDEO_EXTENSIONS = frozenset({".mp4", ".webm", ".mov", ".gif"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SignLookupService:
    """Maintain a deterministic, auditable mapping of glosses to visual output."""

    def __init__(self) -> None:
        self.word_to_clip: Dict[str, dict] = {}
        self.vocabulary: List[str] = []
        self.videos_dir = Path(SIGN_VIDEOS_DIR)
        self.dataset_videos_dir = Path(WLASL_SIGN_VIDEOS_DIR)
        self.manifest_path = Path(SIGN_ASSET_MANIFEST)
        self.guide_manifest_path = Path(FINGERSPELL_GUIDE_MANIFEST)
        self.guides_dir = Path(FINGERSPELL_GUIDES_DIR)
        self.letter_guides: Dict[str, dict] = {}
        self._online_cache: dict[str, Optional[str]] = {}
        self._cache_lock = threading.Lock()
        self.manifest_errors: list[str] = []

    def load_mappings(self) -> None:
        """Load vocabulary and only serve clips declared by trusted local sources."""

        self.vocabulary = self._load_vocabulary()
        self.word_to_clip.clear()
        self.letter_guides.clear()
        self.manifest_errors.clear()
        self._load_manifest_assets()
        self._load_fingerspell_guides()
        self._scan_frontend_video_directory()
        logger.info(
            "Loaded %d vocabulary concepts and %d verified local sign clips",
            len(self.vocabulary),
            len(self.get_available_signs()),
        )

    @staticmethod
    def _load_vocabulary() -> List[str]:
        if not VOCABULARY_FILE.is_file():
            return []
        with VOCABULARY_FILE.open("r", encoding="utf-8") as file:
            value = json.load(file)
        labels = value.get("labels", []) if isinstance(value, dict) else []
        return list(dict.fromkeys(str(label).strip().lower() for label in labels if str(label).strip()))

    def _load_manifest_assets(self) -> None:
        if not self.manifest_path.is_file():
            self.manifest_errors.append("manifest_missing")
            return
        try:
            document = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.exception("Sign asset manifest could not be read")
            self.manifest_errors.append("manifest_invalid")
            return
        if document.get("schema_version") not in {1, 2} or not isinstance(document.get("assets"), list):
            self.manifest_errors.append("manifest_schema_invalid")
            return

        root = self.dataset_videos_dir.resolve()
        for item in document["assets"]:
            if not isinstance(item, dict):
                self.manifest_errors.append("manifest_item_invalid")
                continue
            gloss = str(item.get("gloss", "")).strip().lower()
            relative = str(item.get("relative_path", "")).strip().replace("\\", "/")
            expected_hash = str(item.get("sha256", "")).strip().lower()
            try:
                path = (root / relative).resolve()
                path.relative_to(root)
            except (OSError, ValueError):
                self.manifest_errors.append(f"unsafe_path:{gloss or 'unknown'}")
                continue
            if (
                not gloss
                or not path.is_file()
                or path.suffix.lower() not in VIDEO_EXTENSIONS
                or not re.fullmatch(r"[0-9a-f]{64}", expected_hash)
                or _sha256(path) != expected_hash
            ):
                self.manifest_errors.append(f"asset_invalid:{gloss or 'unknown'}")
                continue
            url_path = urllib.parse.quote(Path(relative).as_posix(), safe="/")
            self.word_to_clip[gloss] = {
                "word": gloss,
                "gloss": gloss,
                "type": "video",
                "representation": "native_sign_video",
                "path": f"/sign-assets/{url_path}",
                "mime_type": str(item.get("mime_type") or "video/mp4"),
                "letters": None,
                "letter_steps": None,
                "available": True,
                "renderable": True,
                "native_sign_available": True,
                "fallback_reason": None,
                "source": str(item.get("source") or "local_manifest"),
                "duration_seconds": float(item["duration_seconds"]) if item.get("duration_seconds") else None,
                "asset_id": str(item.get("asset_id") or f"wlasl:{gloss}:{path.stem}"),
                "sha256": expected_hash,
                "license": item.get("license"),
                "attribution": item.get("attribution"),
                "use_scope": str(item.get("use_scope") or "research_only_unknown_redistribution"),
            }

    def _load_fingerspell_guides(self) -> None:
        """Load only hash-declared local guides; never infer missing poses."""

        if not self.guide_manifest_path.is_file():
            self.manifest_errors.append("fingerspell_guide_manifest_missing")
            return
        try:
            document = json.loads(self.guide_manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.manifest_errors.append("fingerspell_guide_manifest_invalid")
            return
        if (
            document.get("schema_version") != 1
            or document.get("not_native_media") is not True
            or not isinstance(document.get("assets"), list)
        ):
            self.manifest_errors.append("fingerspell_guide_schema_invalid")
            return
        root = self.guides_dir.resolve()
        for item in document["assets"]:
            if not isinstance(item, dict):
                self.manifest_errors.append("fingerspell_guide_item_invalid")
                continue
            letter = str(item.get("letter", "")).strip().upper()
            relative = str(item.get("relative_path", "")).strip().replace("\\", "/")
            expected_hash = str(item.get("sha256", "")).strip().lower()
            try:
                path = (root / relative).resolve()
                path.relative_to(root)
            except (OSError, ValueError):
                self.manifest_errors.append(f"fingerspell_guide_unsafe:{letter or 'unknown'}")
                continue
            if (
                len(letter) != 1
                or letter.lower() not in ASCII_LETTERS
                or letter in {"J", "Z"}
                or not path.is_file()
                or path.suffix.lower() != ".svg"
                or not re.fullmatch(r"[0-9a-f]{64}", expected_hash)
                or _sha256(path) != expected_hash
            ):
                self.manifest_errors.append(f"fingerspell_guide_invalid:{letter or 'unknown'}")
                continue
            self.letter_guides[letter] = {
                "path": f"/static/assets/fingerspell_guides/{urllib.parse.quote(path.name)}",
                "mime_type": "image/svg+xml",
                "sha256": expected_hash,
            }

    def _scan_frontend_video_directory(self) -> None:
        """Keep compatibility with curated frontend clips, without overriding manifest assets."""

        if not self.videos_dir.is_dir():
            return
        for video_file in sorted(self.videos_dir.iterdir(), key=lambda item: item.name.lower()):
            if not video_file.is_file() or video_file.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            gloss = video_file.stem.strip().replace("_", " ").lower()
            if not gloss:
                continue
            self.word_to_clip.setdefault(
                gloss,
                {
                    "word": gloss,
                    "gloss": gloss,
                    "type": "video",
                    "representation": "native_sign_video",
                    "path": f"/static/assets/sign_videos/{urllib.parse.quote(video_file.name)}",
                    "mime_type": "video/mp4" if video_file.suffix.lower() == ".mp4" else None,
                    "letters": None,
                    "letter_steps": None,
                    "available": True,
                    "renderable": True,
                    "native_sign_available": True,
                    "fallback_reason": None,
                    "source": "curated_frontend_asset",
                    "duration_seconds": None,
                    "asset_id": f"curated:{gloss}:{video_file.name}",
                    "sha256": _sha256(video_file),
                    "license": None,
                    "attribution": None,
                    "use_scope": "locally_curated_terms_require_review",
                },
            )

    def _letter_fallback(self, gloss: str) -> dict:
        compact = "".join(character for character in gloss.lower() if not character.isspace())
        letters = [character for character in compact if character in ASCII_LETTERS]
        fully_supported = bool(letters) and len(letters) == len(compact)
        steps: list[dict] = []
        for index, letter in enumerate(letters):
            upper = letter.upper()
            guide = self.letter_guides.get(upper)
            motion_required = upper in {"J", "Z"}
            steps.append(
                {
                    "index": index,
                    "letter": upper,
                    "representation": "motion_required" if motion_required else "landmark_guide",
                    "path": guide["path"] if guide else None,
                    "mime_type": guide["mime_type"] if guide else None,
                    "renderable": bool(guide),
                    "motion_required": motion_required,
                }
            )
        guides_complete = fully_supported and all(step["renderable"] for step in steps)
        has_motion_letter = any(step["motion_required"] for step in steps)
        return {
            "word": gloss,
            "gloss": gloss,
            "type": "fingerspell" if fully_supported else "unavailable",
            "representation": "fingerspell_landmark_guides" if fully_supported else "no_visual_representation",
            "path": None,
            "mime_type": None,
            "letters": letters,
            "letter_steps": steps,
            # ``available`` retains the old meaning: a native media asset exists.
            "available": False,
            "renderable": guides_complete,
            "native_sign_available": False,
            "fallback_reason": (
                "dynamic_fingerspelling_requires_motion"
                if has_motion_letter
                else ("no_native_sign_clip" if fully_supported else "unsupported_characters")
            ),
            "source": "bundled_model_landmark_guides",
            "duration_seconds": None,
            "asset_id": None,
            "sha256": None,
            "license": None,
            "attribution": "Derived from the bundled classifier CSV; authoritative source/license unknown.",
            "use_scope": "local_research_guide",
        }

    def get_sign_clips(self, concepts: List[dict] | List[str]) -> List[dict]:
        clips: list[dict] = []
        for sequence_index, concept in enumerate(concepts):
            if isinstance(concept, dict):
                gloss = str(concept.get("gloss", "")).strip().lower()
                source_tokens = [str(item) for item in concept.get("source_tokens", [])]
            else:
                gloss = str(concept).strip().lower()
                source_tokens = [str(concept)]
            if not gloss:
                continue

            local = self.word_to_clip.get(gloss)
            if local:
                clip = dict(local)
            else:
                online_url = self._scrape_online_video(gloss) if ENABLE_ONLINE_SIGN_LOOKUP else None
                if online_url:
                    clip = {
                        "word": gloss,
                        "gloss": gloss,
                        "type": "video",
                        "representation": "native_sign_video",
                        "path": online_url,
                        "mime_type": "video/mp4",
                        "letters": None,
                        "letter_steps": None,
                        "available": True,
                        "renderable": True,
                        "native_sign_available": True,
                        "fallback_reason": None,
                        "source": "configured_online_lookup",
                        "duration_seconds": None,
                        "asset_id": None,
                        "sha256": None,
                        "license": None,
                        "attribution": "External media supplied by the explicitly configured online lookup.",
                        "use_scope": "remote_provider_terms_apply",
                    }
                else:
                    clip = self._letter_fallback(gloss)
            clip.update(
                {
                    "sequence_index": sequence_index,
                    "source_tokens": source_tokens,
                }
            )
            clips.append(clip)
        return clips

    def coverage(self, clips: List[dict]) -> dict:
        native = sum(bool(item.get("native_sign_available")) for item in clips)
        fingerspelled = sum(item.get("type") == "fingerspell" for item in clips)
        unavailable = sum(not bool(item.get("renderable")) for item in clips)
        total = len(clips)
        return {
            "total_concepts": total,
            "native_signs": native,
            "fingerspelled": fingerspelled,
            "unavailable": unavailable,
            "native_ratio": round(native / total, 4) if total else 0.0,
            "fully_renderable": unavailable == 0,
            "local_clip_count": len(self.get_available_signs()),
            "vocabulary_size": len(self.vocabulary),
        }

    def status(self) -> dict:
        return {
            "vocabulary_size": len(self.vocabulary),
            "local_clip_count": len(self.get_available_signs()),
            "local_clips": self.get_available_signs(),
            "fingerspell_fallback": True,
            "online_lookup_enabled": bool(ENABLE_ONLINE_SIGN_LOOKUP),
            "manifest_valid": not self.manifest_errors,
            "manifest_errors": list(self.manifest_errors),
            "fingerspell_guide_count": len(self.letter_guides),
            "fingerspell_guides_verified": len(self.letter_guides) == 24,
            "dynamic_fingerspell_letters": ["J", "Z"],
        }

    @staticmethod
    def remote_processing_status() -> dict:
        return {
            "enabled": bool(ENABLE_ONLINE_SIGN_LOOKUP),
            "providers": ["signasl.org"] if ENABLE_ONLINE_SIGN_LOOKUP else [],
            "user_text_may_leave_server": bool(ENABLE_ONLINE_SIGN_LOOKUP),
        }

    def _scrape_online_video(self, word: str) -> Optional[str]:
        with self._cache_lock:
            if word in self._online_cache:
                return self._online_cache[word]
        result: Optional[str] = None
        try:
            url = "https://www.signasl.org/sign/" + urllib.parse.quote(word, safe="")
            request = urllib.request.Request(url, headers={"User-Agent": "SignBridge/2.0"})
            with urllib.request.urlopen(request, timeout=SIGN_LOOKUP_TIMEOUT_SECONDS) as response:
                html = response.read(1_000_000).decode("utf-8", errors="replace")
            match = re.search(r'<video[^>]+src="([^"]+)"', html) or re.search(
                r'<source[^>]+src="([^"]+)"', html
            )
            if match:
                candidate = urllib.parse.urljoin(url, match.group(1))
                parsed = urllib.parse.urlparse(candidate)
                if parsed.scheme == "https" and parsed.hostname:
                    result = candidate
        except Exception as exc:
            logger.info("Online sign lookup failed error_type=%s", type(exc).__name__)
        with self._cache_lock:
            self._online_cache[word] = result
        return result

    def get_vocabulary(self) -> List[str]:
        # Manifest assets are valid translation concepts even when they are not
        # classifier labels. This is what makes a declared multiword asset take
        # part in the NLP processor's longest-phrase matching.
        return sorted(set(self.vocabulary) | set(self.word_to_clip))

    def get_available_signs(self) -> List[str]:
        return sorted(self.word_to_clip)

    def add_clip(self, word: str, video_filename: str) -> None:
        """Compatibility helper for trusted, already-curated frontend assets."""

        safe_name = Path(video_filename).name
        normalized = word.strip().lower()
        if not normalized or Path(safe_name).suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError("A safe supported video filename and non-empty word are required.")
        self.word_to_clip[normalized] = {
            "word": normalized,
            "gloss": normalized,
            "type": "video",
            "representation": "native_sign_video",
            "path": f"/static/assets/sign_videos/{urllib.parse.quote(safe_name)}",
            "mime_type": "video/mp4" if Path(safe_name).suffix.lower() == ".mp4" else None,
            "letters": None,
            "letter_steps": None,
            "available": True,
            "renderable": True,
            "native_sign_available": True,
            "fallback_reason": None,
            "source": "curated_frontend_asset",
            "duration_seconds": None,
            "asset_id": f"curated:{normalized}:{safe_name}",
            "sha256": None,
            "license": None,
            "attribution": None,
            "use_scope": "locally_curated_terms_require_review",
        }
