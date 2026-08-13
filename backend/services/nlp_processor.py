"""Deterministic English concept extraction for text/voice-to-sign playback.

This module deliberately does not claim to be a complete English-to-ASL
translator.  It preserves content that cannot be mapped to the small native
clip vocabulary so the renderer can fall back to explicit fingerspelling.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from typing import Any, List, Optional, Set

import requests

try:
    from ..config import SPACY_MODEL
except ImportError:  # pragma: no cover - direct backend-path execution
    from config import SPACY_MODEL

logger = logging.getLogger(__name__)
TOKEN_RE = re.compile(r"[A-Za-z]+(?:['\u2019][A-Za-z]+)?|\d+")


class NLPProcessor:
    """Extract ordered concepts, including longest matching vocabulary phrases."""

    def __init__(self) -> None:
        self.nlp = None
        self.vocabulary: Set[str] = set()
        self._phrase_index: dict[str, list[tuple[tuple[str, ...], str]]] = {}
        self._translation_state = threading.local()
        self._default_translation_status = {
            "provider": "deterministic",
            "mode": "concept_mapping",
            "semantic_translation": False,
            "message": (
                "Deterministic concept mapping is active; output is a visual aid, "
                "not a complete grammatical ASL translation."
            ),
        }
        self.language_model_status = {"ready": False, "mode": "not_loaded", "model": SPACY_MODEL}
        self._translation_lock = threading.Lock()
        self._last_hf_request_at = 0.0
        self._hf_min_interval_seconds = float(os.getenv("HF_MIN_INTERVAL_SECONDS", "1.25"))
        self._hf_endpoint = os.getenv("HF_TRANSLATION_ENDPOINT", "").strip()
        self._ollama_enabled = os.getenv("OLLAMA_ENABLED", "false").strip().lower() in {
            "1", "true", "yes", "on"
        }

        # Targets may be phrases. Phrase targets remain one playback concept.
        self.synonym_map = {
            "hi": "hello", "hey": "hello", "greetings": "hello",
            "thanks": "thank you", "thankyou": "thank you", "thx": "thank you",
            "wanna": "want", "wish": "want",
            "require": "need", "requires": "need", "required": "need",
            "enjoy": "like", "enjoys": "like", "enjoyed": "like",
            "going": "go", "goes": "go", "went": "go", "gone": "go",
            "coming": "come", "comes": "come", "came": "come",
            "eating": "eat", "eats": "eat", "ate": "eat",
            "drinking": "drink", "drinks": "drink", "drank": "drink",
            "helping": "help", "helps": "help", "helped": "help",
            "understood": "understand", "understands": "understand",
            "currently": "now", "great": "good", "nice": "good", "fine": "good",
            "terrible": "bad", "awful": "bad", "large": "big", "huge": "big",
            "little": "small", "tiny": "small", "yeah": "yes", "yep": "yes",
            "nope": "no", "nah": "no", "pls": "please", "apologize": "sorry",
        }
        self.contractions = {
            "i'm": ("i", "am"), "you're": ("you", "are"), "we're": ("we", "are"),
            "they're": ("they", "are"), "i've": ("i", "have"), "we've": ("we", "have"),
            "you've": ("you", "have"), "can't": ("can", "not"),
            "cannot": ("can", "not"), "won't": ("will", "not"),
            "don't": ("do", "not"), "doesn't": ("does", "not"),
            "didn't": ("did", "not"), "isn't": ("is", "not"),
            "aren't": ("are", "not"), "wasn't": ("was", "not"),
            "weren't": ("were", "not"), "it's": ("it", "is"),
            "that's": ("that", "is"), "what's": ("what", "is"),
        }
        # Only omit articles and two high-frequency linkers. Copulas are kept as
        # explicit source-linked concepts: silently removing ``am`` from
        # "I am happy" is neither faithful English playback nor validated ASL
        # translation. A future certified grammar model may make that decision.
        self.structural_words = {"a", "an", "the", "to", "of"}

    @property
    def last_translation_status(self) -> dict:
        return dict(getattr(self._translation_state, "status", self._default_translation_status))

    @last_translation_status.setter
    def last_translation_status(self, value: dict) -> None:
        self._translation_state.status = dict(value)

    def load(self) -> None:
        """Load local linguistic support without downloading packages or models."""

        try:
            import spacy

            try:
                self.nlp = spacy.load(SPACY_MODEL)
                self.language_model_status = {
                    "ready": True,
                    "mode": "lemmatizer",
                    "model": SPACY_MODEL,
                    "downloads_performed": False,
                }
            except (OSError, ImportError, ValueError) as exc:
                self.nlp = spacy.blank("en")
                self.language_model_status = {
                    "ready": True,
                    "mode": "tokenizer_fallback",
                    "model": "spacy.blank(en)",
                    "reason": f"{type(exc).__name__}: configured model unavailable",
                    "downloads_performed": False,
                }
        except ImportError:
            # Regex tokenization is sufficient for deterministic fallback.
            self.nlp = False
            self.language_model_status = {
                "ready": True,
                "mode": "regex_fallback",
                "model": None,
                "reason": "spaCy is not installed",
                "downloads_performed": False,
            }
        logger.info("Text normalization ready in %s mode", self.language_model_status["mode"])

    def set_vocabulary(self, vocab: List[str]) -> None:
        self.vocabulary = {" ".join(str(word).lower().split()) for word in vocab if str(word).strip()}
        phrase_index: dict[str, list[tuple[tuple[str, ...], str]]] = {}
        for gloss in self.vocabulary:
            parts = tuple(gloss.split())
            if len(parts) < 2:
                continue
            phrase_index.setdefault(parts[0], []).append((parts, gloss))
        for first_word in phrase_index:
            phrase_index[first_word].sort(key=lambda item: (-len(item[0]), item[1]))
        self._phrase_index = phrase_index
        logger.info(
            "NLP vocabulary set to %d concepts (%d multiword phrases)",
            len(self.vocabulary),
            sum(len(items) for items in phrase_index.values()),
        )

    def remote_processing_status(self) -> dict:
        providers: list[str] = []
        if self._hf_endpoint:
            providers.append("huggingface_endpoint")
        if self._ollama_enabled:
            providers.append("ollama")
        return {
            "enabled": bool(providers),
            "providers": providers,
            "user_text_may_leave_server": bool(providers),
        }

    @staticmethod
    def _tokenize(text: str) -> list[dict[str, Any]]:
        tokens: list[dict[str, Any]] = []
        for source_index, match in enumerate(TOKEN_RE.finditer(text.replace("\u2019", "'"))):
            raw = match.group(0)
            tokens.append(
                {
                    "raw": raw,
                    "word": raw.lower(),
                    "source_index": source_index,
                    "char_start": match.start(),
                    "char_end": match.end(),
                }
            )
        return tokens

    def _expand_tokens(self, tokens: list[dict[str, Any]]) -> list[dict[str, Any]]:
        expanded: list[dict[str, Any]] = []
        for token in tokens:
            words = self.contractions.get(token["word"], (token["word"],))
            for word in words:
                expanded.append({**token, "word": word})
        return expanded

    def _lemma(self, word: str) -> str:
        """Use a loaded local lemma when it is informative; never invent one."""

        if not self.nlp or self.language_model_status.get("mode") != "lemmatizer":
            return word
        doc = self.nlp(word)
        if not doc:
            return word
        lemma = doc[0].lemma_.lower().strip()
        return word if not lemma or lemma == "-pron-" else lemma

    def _lexical_word(self, word: str) -> tuple[str, bool]:
        mapped = self.synonym_map.get(word)
        if mapped:
            return mapped, True
        # Preserve source copula forms for source-linked playback. Lemmatizing
        # every form to ``be`` loses the exact visible sequence the UI promises.
        if word in {"am", "is", "are", "was", "were", "be", "been", "being"}:
            return word, False
        lemma = self._lemma(word)
        mapped_lemma = self.synonym_map.get(lemma)
        return (mapped_lemma, True) if mapped_lemma else (lemma, lemma != word)

    def analyze(self, text: str) -> dict:
        """Analyze one sentence once and return ordered, source-linked concepts."""

        if self.nlp is None:
            self.load()
        translated_text = self._translate_to_asl_gloss(text)
        translation_status = self.last_translation_status
        raw_tokens = self._tokenize(translated_text)
        expanded = self._expand_tokens(raw_tokens)

        lexical: list[dict[str, Any]] = []
        all_tokens: list[str] = []
        for token in expanded:
            word, changed = self._lexical_word(token["word"])
            lexical.append({**token, "normalized": word, "changed": changed})
            all_tokens.append(word)

        concepts: list[dict[str, Any]] = []
        index = 0
        while index < len(lexical):
            token = lexical[index]
            normalized = token["normalized"]

            # A one-token synonym can intentionally target a multiword concept,
            # for example "thanks" -> "thank you".
            if " " in normalized:
                gloss = " ".join(normalized.split())
                concepts.append(self._concept(gloss, [token], "synonym"))
                index += 1
                continue

            phrase_match: tuple[tuple[str, ...], str] | None = None
            for parts, gloss in self._phrase_index.get(normalized, []):
                candidate = tuple(item["normalized"] for item in lexical[index:index + len(parts)])
                if candidate == parts:
                    phrase_match = (parts, gloss)
                    break
            if phrase_match:
                parts, gloss = phrase_match
                phrase_tokens = lexical[index:index + len(parts)]
                concepts.append(self._concept(gloss, phrase_tokens, "vocabulary_phrase"))
                index += len(parts)
                continue

            if normalized in self.structural_words:
                index += 1
                continue

            match_type = "synonym" if token["changed"] else (
                "vocabulary" if normalized in self.vocabulary else "content_fallback"
            )
            concepts.append(self._concept(normalized, [token], match_type))
            index += 1

        return {
            "original_text": text,
            "analysis_text": translated_text,
            "input_tokens": [item["raw"] for item in raw_tokens],
            "all_tokens": all_tokens,
            "normalized_tokens": [item["gloss"] for item in concepts],
            "concepts": concepts,
            "translation_status": translation_status,
        }

    def _concept(self, gloss: str, tokens: list[dict[str, Any]], match_type: str) -> dict:
        first, last = tokens[0], tokens[-1]
        return {
            "gloss": gloss,
            "source_tokens": [str(item["raw"]) for item in tokens],
            "source_start": int(first["source_index"]),
            "source_end": int(last["source_index"]) + 1,
            "char_start": int(first["char_start"]),
            "char_end": int(last["char_end"]),
            "match": match_type,
            "in_vocabulary": gloss in self.vocabulary,
        }

    def normalize(self, text: str) -> List[str]:
        return list(self.analyze(text)["normalized_tokens"])

    def normalize_without_vocab_filter(self, text: str) -> List[str]:
        tokens = self._expand_tokens(self._tokenize(text))
        return [self._lexical_word(item["word"])[0] for item in tokens]

    def _translate_to_asl_gloss(self, text: str) -> str:
        """Use semantic translation only when an integration is explicitly enabled."""

        if self._hf_endpoint:
            try:
                gloss = self._translate_with_huggingface(text)
                if gloss:
                    return gloss
            except requests.RequestException as exc:
                logger.info("Configured Hugging Face translation failed: %s", type(exc).__name__)

        if self._ollama_enabled:
            try:
                base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
                model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
                response = requests.post(
                    f"{base_url}/api/chat",
                    json={
                        "model": model,
                        "stream": False,
                        "messages": [{
                            "role": "user",
                            "content": (
                                "Convert this English sentence to concise ASL gloss. "
                                "Output gloss only: " + text
                            ),
                        }],
                    },
                    timeout=8.0,
                )
                if response.ok:
                    gloss = str(response.json().get("message", {}).get("content", "")).strip()
                    if gloss:
                        self.last_translation_status = {
                            "provider": "ollama",
                            "mode": "semantic_gloss",
                            "semantic_translation": True,
                            "message": "A configured local model supplied an unverified ASL gloss.",
                        }
                        return gloss
            except (requests.RequestException, ValueError):
                logger.info("Configured Ollama translation unavailable")

        self.last_translation_status = self._default_translation_status
        return text

    def _translate_with_huggingface(self, text: str) -> Optional[str]:
        token = os.getenv("HF_API_TOKEN", "").strip()
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        prompt = (
            "Translate this English sentence into concise ASL gloss. "
            "Output gloss only: " + text
        )
        with self._translation_lock:
            wait_seconds = self._hf_min_interval_seconds - (time.monotonic() - self._last_hf_request_at)
            if wait_seconds > 0:
                time.sleep(wait_seconds)
            response = requests.post(
                self._hf_endpoint,
                headers=headers,
                json={"inputs": prompt, "parameters": {"max_new_tokens": 80, "temperature": 0.1}},
                timeout=20.0,
            )
            self._last_hf_request_at = time.monotonic()
        if not response.ok:
            self.last_translation_status = {
                "provider": "huggingface",
                "mode": "unavailable",
                "semantic_translation": False,
                "message": f"Configured semantic endpoint returned HTTP {response.status_code}; deterministic mapping used.",
            }
            return None
        result = response.json()
        gloss = result[0].get("generated_text", "").strip() if isinstance(result, list) and result else ""
        if gloss:
            self.last_translation_status = {
                "provider": "huggingface",
                "mode": "semantic_gloss",
                "semantic_translation": True,
                "message": "A configured remote model supplied an unverified ASL gloss.",
            }
            return gloss
        return None
