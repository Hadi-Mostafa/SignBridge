"""Offline contracts for Page 2 text/voice-to-sign translation."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import backend.main as main
from backend.routers import speech_to_sign
from backend.services.asr_service import ASRAssetMissingError, ASRAudioTooLongError, ASRService
from backend.services.nlp_processor import NLPProcessor
from backend.services.sign_lookup import SignLookupService


class ConceptPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lookup = SignLookupService()
        cls.lookup.load_mappings()
        cls.nlp = NLPProcessor()
        cls.nlp.load()
        cls.nlp.set_vocabulary(cls.lookup.get_vocabulary())

    def test_sentence_preserves_unknown_content_and_matches_phrase(self) -> None:
        analysis = self.nlp.analyze("I am going to the hospital. Thank you!")
        self.assertEqual(
            analysis["normalized_tokens"],
            ["i", "am", "go", "hospital", "thank you"],
        )
        thank_you = analysis["concepts"][-1]
        self.assertEqual(thank_you["match"], "vocabulary_phrase")
        self.assertEqual(thank_you["source_tokens"], ["Thank", "you"])
        self.assertFalse(analysis["translation_status"]["semantic_translation"])

    def test_synonym_can_target_one_multiword_concept(self) -> None:
        analysis = self.nlp.analyze("Thanks for helping")
        self.assertEqual(analysis["normalized_tokens"], ["thank you", "for", "help"])
        self.assertEqual(analysis["concepts"][0]["source_tokens"], ["Thanks"])

    def test_manifest_clips_and_fingerspell_fallback_are_honest(self) -> None:
        concepts = [
            {"gloss": "drink", "source_tokens": ["drink"]},
            {"gloss": "like", "source_tokens": ["like"]},
            {"gloss": "hospital", "source_tokens": ["hospital"]},
        ]
        clips = self.lookup.get_sign_clips(concepts)
        self.assertEqual(clips[0]["path"], "/sign-assets/drink/69302.mp4")
        self.assertEqual(clips[1]["path"], "/sign-assets/like/69389.mp4")
        self.assertTrue(clips[0]["native_sign_available"])
        self.assertEqual(clips[2]["type"], "fingerspell")
        self.assertFalse(clips[2]["available"])
        self.assertTrue(clips[2]["renderable"])
        self.assertEqual("".join(clips[2]["letters"]), "hospital")
        self.assertEqual(clips[2]["representation"], "fingerspell_landmark_guides")
        self.assertTrue(all(step["path"].endswith(".svg") for step in clips[2]["letter_steps"]))
        self.assertTrue(all(step["representation"] == "landmark_guide" for step in clips[2]["letter_steps"]))
        coverage = self.lookup.coverage(clips)
        self.assertEqual(coverage["native_signs"], 2)
        self.assertEqual(coverage["fingerspelled"], 1)

    def test_dynamic_fingerspelling_is_not_misrepresented_as_static(self) -> None:
        clip = self.lookup.get_sign_clips([{"gloss": "jazz", "source_tokens": ["jazz"]}])[0]
        dynamic = [step for step in clip["letter_steps"] if step["letter"] in {"J", "Z"}]
        self.assertEqual(len(dynamic), 3)
        self.assertTrue(all(step["motion_required"] for step in dynamic))
        self.assertTrue(all(step["path"] is None for step in dynamic))
        self.assertFalse(clip["renderable"])
        self.assertEqual(clip["fallback_reason"], "dynamic_fingerspelling_requires_motion")

    def test_manifest_phrase_is_added_to_longest_match_vocabulary(self) -> None:
        # ``get_vocabulary`` deliberately unions classifier labels and manifest
        # glosses, so future validated phrase assets automatically participate
        # in longest-match extraction.
        original = dict(self.lookup.word_to_clip)
        try:
            self.lookup.word_to_clip["good morning"] = {"gloss": "good morning"}
            nlp = NLPProcessor()
            nlp.load()
            nlp.set_vocabulary(self.lookup.get_vocabulary())
            analysis = nlp.analyze("Good morning friend")
        finally:
            self.lookup.word_to_clip = original
        self.assertEqual(analysis["normalized_tokens"][:2], ["good morning", "friend"])
        self.assertEqual(analysis["concepts"][0]["match"], "vocabulary_phrase")

    def test_asr_loader_fails_before_import_or_download_when_asset_missing(self) -> None:
        status = {
            "package_available": True,
            "ffmpeg_available": True,
            "cached": False,
        }
        with patch.object(ASRService, "asset_status", return_value=status):
            with self.assertRaises(ASRAssetMissingError):
                ASRService().load_model()

    def test_corrupt_whisper_asset_is_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corrupt = Path(directory) / "base.pt"
            corrupt.write_bytes(b"not a whisper checkpoint")
            ASRService._integrity_cache.clear()
            with patch("backend.services.asr_service.WHISPER_MODEL_PATH", corrupt):
                status = ASRService.asset_status(force_verify=True)
        self.assertTrue(status["cached"])
        self.assertFalse(status["cached_valid"])
        self.assertFalse(status["integrity_verified"])
        self.assertFalse(status["ready_to_load"])

    def test_runtime_load_uses_verified_local_path(self) -> None:
        service = ASRService()
        ready = {"package_available": True, "ffmpeg_available": True, "cached": True, "cached_valid": True}
        fake_model = object()
        with patch.object(ASRService, "asset_status", return_value=ready), patch(
            "whisper.load_model", return_value=fake_model
        ) as loader:
            self.assertIs(service.load_model(), fake_model)
        self.assertEqual(Path(loader.call_args.args[0]), Path("backend/checkpoints/whisper/base.pt").resolve())


class PageTwoAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        speech_to_sign.initialize_services()
        cls.client = TestClient(main.app)

    def test_text_to_sign_structured_sequence_and_coverage(self) -> None:
        response = self.client.post(
            "/api/text-to-sign",
            json={"text": "I am going to the hospital. Thank you. I like a drink."},
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["mode"], "text")
        self.assertIn("thank you", payload["normalized_tokens"])
        clips = {item["gloss"]: item for item in payload["sign_clips"]}
        self.assertEqual(clips["like"]["path"], "/sign-assets/like/69389.mp4")
        self.assertEqual(clips["drink"]["path"], "/sign-assets/drink/69302.mp4")
        self.assertEqual(clips["hospital"]["representation"], "fingerspell_landmark_guides")
        self.assertGreaterEqual(payload["coverage"]["native_signs"], 2)
        self.assertTrue(payload["coverage"]["fully_renderable"])
        self.assertIn("not certified ASL grammar", payload["translation_status"]["disclaimer"])

    def test_i_am_happy_preserves_source_order_and_serves_visual_guides(self) -> None:
        response = self.client.post("/api/text-to-sign", json={"text": "I am happy"})
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["normalized_tokens"], ["i", "am", "happy"])
        self.assertEqual([clip["sequence_index"] for clip in payload["sign_clips"]], [0, 1, 2])
        for clip in payload["sign_clips"]:
            self.assertTrue(clip["renderable"])
            self.assertTrue(all(step["path"] for step in clip["letter_steps"]))
        guide = self.client.get(payload["sign_clips"][2]["letter_steps"][0]["path"])
        self.assertEqual(guide.status_code, 200)
        self.assertTrue(guide.headers["content-type"].startswith("image/svg+xml"))
        self.assertIn(b"landmark guide", guide.content)

    def test_local_sign_assets_are_served_and_traversal_is_denied(self) -> None:
        for path in ("/sign-assets/drink/69302.mp4", "/sign-assets/like/69389.mp4"):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.headers["content-type"], "video/mp4")
                self.assertGreater(len(response.content), 1_000_000)
        traversal = self.client.get("/sign-assets/%2e%2e/%2e%2e/backend/config.py")
        self.assertEqual(traversal.status_code, 404)

    def test_voice_accepts_media_recorder_content_type(self) -> None:
        ready = {
            "package_available": True,
            "ffmpeg_available": True,
            "cached": True,
            "cached_valid": True,
            "ready_to_load": True,
            "fetch_command": "python scripts/fetch_speech_model.py",
        }
        assert speech_to_sign.asr_service is not None
        with patch.object(ASRService, "asset_status", return_value=ready), patch.object(
            ASRService, "probe_audio_bytes", return_value=1.0,
        ), patch.object(
            speech_to_sign.asr_service,
            "transcribe_bytes",
            return_value=("I like a drink", 12.5),
        ):
            response = self.client.post(
                "/api/speech-to-sign",
                files={"audio": ("recording.webm", b"valid-webm-placeholder", "audio/webm;codecs=opus")},
            )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["mode"], "voice")
        self.assertEqual(payload["transcript"], "I like a drink")
        self.assertEqual(payload["normalized_tokens"], ["i", "like", "drink"])

    def test_speech_capacity_rejects_without_queueing(self) -> None:
        acquired = speech_to_sign._asr_capacity.acquire(blocking=False)
        self.assertTrue(acquired)
        try:
            response = self.client.post(
                "/api/speech-to-sign",
                files={"audio": ("recording.webm", b"data", "audio/webm;codecs=opus")},
            )
        finally:
            speech_to_sign._asr_capacity.release()
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.json()["detail"]["code"], "speech_busy")

    def test_audio_duration_errors_are_structured(self) -> None:
        ready = {"package_available": True, "ffmpeg_available": True, "cached": True, "cached_valid": True, "ready_to_load": True}
        with patch.object(ASRService, "asset_status", return_value=ready), patch.object(
            ASRService, "probe_audio_bytes", side_effect=ASRAudioTooLongError("too long")
        ):
            response = self.client.post(
                "/api/speech-to-sign",
                files={"audio": ("recording.webm", b"data", "audio/webm;codecs=opus")},
            )
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["detail"]["code"], "audio_too_long")

    def test_package_modules_import_in_fresh_processes(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        modules = (
            "backend.routers.sign_to_text", "backend.routers.speech_to_sign",
            "backend.utils.video_utils", "backend.utils.latency_logger",
        )
        for module in modules:
            with self.subTest(module=module):
                completed = subprocess.run(
                    [sys.executable, "-c", f"import {module}"], cwd=project_root,
                    capture_output=True, text=True, timeout=30,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_voice_missing_asset_is_explicit_and_never_downloads(self) -> None:
        missing = {
            "package_available": True,
            "ffmpeg_available": True,
            "cached": False,
            "cached_valid": False,
            "ready_to_load": False,
            "fetch_command": "python scripts/fetch_speech_model.py",
        }
        with patch.object(ASRService, "asset_status", return_value=missing):
            response = self.client.post(
                "/api/speech-to-sign",
                files={"audio": ("recording.webm", b"data", "audio/webm;codecs=opus")},
            )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"]["code"], "speech_model_missing")
        self.assertIn("fetch_speech_model.py", response.json()["detail"]["action"])

    def test_audio_and_text_bounds_use_structured_validation(self) -> None:
        blank = self.client.post("/api/text-to-sign", json={"text": "   "})
        self.assertEqual(blank.status_code, 422)
        unsupported = self.client.post(
            "/api/speech-to-sign",
            files={"audio": ("audio.txt", b"data", "text/plain")},
        )
        self.assertEqual(unsupported.status_code, 415)
        self.assertEqual(unsupported.json()["detail"]["code"], "unsupported_audio_type")

    def test_vocabulary_reports_real_native_coverage(self) -> None:
        response = self.client.get("/api/vocabulary")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["available_clips"], ["drink", "like"])
        self.assertEqual(payload["clips_available"], 2)
        self.assertTrue(payload["fingerspell_fallback"])
        self.assertEqual(payload["fingerspell_guides"]["verified_static_letters"], 24)
        self.assertEqual(payload["fingerspell_guides"]["dynamic_letters_requiring_motion"], ["J", "Z"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
