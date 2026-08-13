"""
Text-to-Speech (TTS) Service.

Converts recognized sign language text to spoken audio so hearing
users can understand what the deaf user is signing.

Supports two engines:
    - pyttsx3: Offline, zero-latency, works without internet
    - gTTS: Online (Google), better voice quality but requires internet

For MVP, we default to pyttsx3 for reliability and offline operation.
"""

import io
import time
import base64
import tempfile
from pathlib import Path
from typing import Optional, Tuple

try:
    from ..config import TTS_ENGINE, TTS_RATE
except ImportError:  # pragma: no cover - direct backend-path execution
    from config import TTS_ENGINE, TTS_RATE


class TTSService:
    """
    Text-to-speech service for converting recognized signs to audio.

    Usage:
        tts = TTSService()
        tts.initialize()

        # Generate audio file
        tts.speak_to_file("hello", "output.wav")

        # Generate base64 audio for web playback
        audio_b64 = tts.speak_to_base64("hello")
    """

    def __init__(self, engine: str = TTS_ENGINE):
        """
        Initialize TTS service.

        Args:
            engine: TTS engine to use ("pyttsx3" or "gtts").
        """
        self.engine_name = engine
        self.engine = None
        self.last_synthesis_time: float = 0.0

    def initialize(self):
        """Initialize the TTS engine."""
        if self.engine_name == "pyttsx3":
            self._init_pyttsx3()
        elif self.engine_name == "gtts":
            # gTTS doesn't need initialization — it's stateless
            print("[TTSService] Using gTTS (Google Text-to-Speech)")
        else:
            raise ValueError(f"Unknown TTS engine: {self.engine_name}")

    def _init_pyttsx3(self):
        """Initialize pyttsx3 offline engine."""
        import pyttsx3
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", TTS_RATE)

        # List available voices and select a clear one
        voices = self.engine.getProperty("voices")
        if voices:
            # Prefer a female voice for clarity (often index 1 on Windows)
            if len(voices) > 1:
                self.engine.setProperty("voice", voices[1].id)
            print(f"[TTSService] pyttsx3 initialized with voice: {voices[0].name}")
        else:
            print("[TTSService] pyttsx3 initialized (default voice)")

    def speak_to_file(self, text: str, output_path: str) -> float:
        """
        Synthesize speech and save to an audio file.

        Args:
            text: Text to convert to speech.
            output_path: Path to save the audio file.

        Returns:
            Synthesis latency in milliseconds.
        """
        start_time = time.time()

        if self.engine_name == "pyttsx3":
            if self.engine is None:
                self._init_pyttsx3()
            self.engine.save_to_file(text, output_path)
            self.engine.runAndWait()

        elif self.engine_name == "gtts":
            from gtts import gTTS
            tts = gTTS(text=text, lang="en", slow=False)
            tts.save(output_path)

        self.last_synthesis_time = (time.time() - start_time) * 1000
        return self.last_synthesis_time

    def speak_to_base64(self, text: str) -> Tuple[str, float]:
        """
        Synthesize speech and return as base64-encoded audio.
        Suitable for sending over WebSocket/HTTP to the browser.

        Args:
            text: Text to convert to speech.

        Returns:
            Tuple of (base64_audio_string, latency_ms).
        """
        # Use a temp file since both engines require file output
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            latency = self.speak_to_file(text, tmp_path)

            # Read and encode as base64
            with open(tmp_path, "rb") as f:
                audio_bytes = f.read()

            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            return audio_b64, latency

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def speak_live(self, text: str):
        """
        Speak text through the system speakers (local playback only).
        Useful for local testing, not for web deployment.

        Args:
            text: Text to speak aloud.
        """
        if self.engine_name == "pyttsx3":
            if self.engine is None:
                self._init_pyttsx3()
            self.engine.say(text)
            self.engine.runAndWait()
        else:
            print(f"[TTSService] Live playback not supported for {self.engine_name}")

    def is_initialized(self) -> bool:
        """Check if the TTS engine is ready."""
        if self.engine_name == "gtts":
            return True  # gTTS is stateless
        return self.engine is not None
