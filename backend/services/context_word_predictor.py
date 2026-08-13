"""Fast, offline, context-aware next-word ranking.

The primary ranker uses a pinned quantized DistilGPT2 causal language model.
It scores complete candidate words conditioned on the preceding sentence.  A
Google Web 1T bigram/unigram table is bundled as a deterministic fallback and
also contributes a small lexical prior.  Runtime requests never access the
network or download model files.
"""

from __future__ import annotations

import hashlib
import math
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:
    from ..config import (
        MAX_WORD_SUGGESTIONS,
        WORD_SUGGESTION_BIGRAM_PATH,
        WORD_SUGGESTION_CONTEXT_TOKENS,
        WORD_SUGGESTION_ONNX_PATH,
        WORD_SUGGESTION_TOKENIZER_PATH,
        WORD_SUGGESTION_UNIGRAM_PATH,
    )
except ImportError:  # pragma: no cover - direct backend-path execution
    from config import (
        MAX_WORD_SUGGESTIONS,
        WORD_SUGGESTION_BIGRAM_PATH,
        WORD_SUGGESTION_CONTEXT_TOKENS,
        WORD_SUGGESTION_ONNX_PATH,
        WORD_SUGGESTION_TOKENIZER_PATH,
        WORD_SUGGESTION_UNIGRAM_PATH,
    )


WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
MODEL_REPO = "Xenova/distilgpt2"
MODEL_REVISION = "a41c10485c18a64b6606729b6a082330cbd8f49e"
MODEL_SHA256 = "dfd02dcbfccb31d289cac235f71cecad357030866fe7019f05a36b1c5692afba"
TOKENIZER_SHA256 = "cda20b8ca044949aa07ac4078420c80d1a57139d5f9f33700e46fb2d891e7c66"
BIGRAM_SHA256 = "781c0596c3eea532d30bef9f3dba1d5137d652f00376260822c761a7584dfb8c"
UNIGRAM_SHA256 = "51df159fd3de12b20e403c108f526e96dbd723d9cabdd5f17955cdc16059e690"
BIGRAM_URL = "https://norvig.com/ngrams/count_2w.txt"
UNIGRAM_URL = "https://norvig.com/ngrams/count_1w.txt"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_words(values: Iterable[object]) -> list[str]:
    words: list[str] = []
    for value in values:
        words.extend(match.group(0).lower() for match in WORD_RE.finditer(str(value)))
    return words


def _softmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    peak = max(values)
    weights = [math.exp(max(-80.0, value - peak)) for value in values]
    total = sum(weights) or 1.0
    return [weight / total for weight in weights]


@dataclass(frozen=True)
class _Candidate:
    word: str
    unigram: int
    bigram: int


class ContextWordPredictor:
    """Rank a bounded vocabulary using sentence context and current prefix."""

    def __init__(self, vocabulary: Iterable[str] | None = None) -> None:
        self._lock = threading.RLock()
        self._session = None
        self._tokenizer = None
        self._load_error: str | None = None
        self._unigrams: dict[str, int] = {}
        self._bigrams: dict[str, dict[str, int]] = {}
        self._vocabulary: tuple[str, ...] = ()
        self._project_vocabulary: tuple[str, ...] = ()
        self._loaded = False
        self._candidate_token_ids: dict[str, tuple[int, ...]] = {}
        self.set_vocabulary(vocabulary or ())

    def set_vocabulary(self, vocabulary: Iterable[str]) -> None:
        words = {word for word in normalize_words(vocabulary) if 1 <= len(word) <= 40}
        with self._lock:
            self._project_vocabulary = tuple(sorted(words))
            self._vocabulary = self._project_vocabulary
            self._candidate_token_ids.clear()

    @staticmethod
    def asset_status(*, verify: bool = False) -> dict:
        files = {
            "model": (WORD_SUGGESTION_ONNX_PATH, MODEL_SHA256),
            "tokenizer": (WORD_SUGGESTION_TOKENIZER_PATH, TOKENIZER_SHA256),
            "bigram": (WORD_SUGGESTION_BIGRAM_PATH, BIGRAM_SHA256),
            "unigram": (WORD_SUGGESTION_UNIGRAM_PATH, UNIGRAM_SHA256),
        }
        present = {name: path.is_file() for name, (path, _) in files.items()}
        hashes_valid: dict[str, bool | None] = {}
        for name, (path, expected) in files.items():
            hashes_valid[name] = (
                _sha256(path) == expected if verify and expected and path.is_file() else None
            )
        statistical_ready = present["bigram"] and present["unigram"]
        neural_ready = present["model"] and present["tokenizer"]
        if verify:
            statistical_ready = statistical_ready and bool(hashes_valid["bigram"] and hashes_valid["unigram"])
            neural_ready = neural_ready and bool(hashes_valid["model"] and hashes_valid["tokenizer"])
        return {
            "ready": bool(statistical_ready),
            "neural_asset_ready": bool(neural_ready),
            "statistical_fallback_ready": bool(statistical_ready),
            "offline": True,
            "downloads_on_request": False,
            "model": "distilgpt2-int8-onnx",
            "model_repo": MODEL_REPO,
            "model_revision": MODEL_REVISION,
            "integrity_verified": bool(verify and statistical_ready),
            "files_present": present,
            "hashes_valid": hashes_valid,
            "fetch_command": "python scripts/fetch_word_suggestion_model.py",
        }

    def load(self) -> None:
        """Load only already-present local artifacts; never fetch anything."""

        with self._lock:
            if self._loaded:
                return
            status = self.asset_status(verify=True)
            if not status["statistical_fallback_ready"]:
                self._load_error = "The pinned statistical language assets are missing or invalid."
                self._loaded = True
                return
            self._load_statistical_tables()
            if status["neural_asset_ready"]:
                try:
                    import onnxruntime as ort
                    from tokenizers import Tokenizer

                    options = ort.SessionOptions()
                    options.intra_op_num_threads = 1
                    options.inter_op_num_threads = 1
                    self._tokenizer = Tokenizer.from_file(str(WORD_SUGGESTION_TOKENIZER_PATH))
                    self._session = ort.InferenceSession(
                        str(WORD_SUGGESTION_ONNX_PATH),
                        sess_options=options,
                        providers=["CPUExecutionProvider"],
                    )
                except Exception as exc:  # statistical model remains usable
                    self._session = None
                    self._tokenizer = None
                    self._load_error = f"{type(exc).__name__}: neural ranker unavailable"
            self._loaded = True

    def _load_statistical_tables(self) -> None:
        project_vocabulary = set(self._project_vocabulary)
        unigrams: dict[str, int] = {}
        with WORD_SUGGESTION_UNIGRAM_PATH.open("r", encoding="utf-8", errors="ignore") as source:
            for line in source:
                try:
                    word, raw_count = line.rstrip("\n").split("\t", 1)
                    normalized = word.lower()
                    if (
                        (normalized in project_vocabulary or len(unigrams) < 20_000)
                        and normalized.isascii()
                        and normalized.isalpha()
                        and 1 <= len(normalized) <= 32
                    ):
                        unigrams[normalized] = unigrams.get(normalized, 0) + int(raw_count)
                except (ValueError, TypeError):
                    continue

        vocabulary = set(unigrams) | project_vocabulary
        for word in project_vocabulary:
            unigrams.setdefault(word, 1)

        bigrams: dict[str, dict[str, int]] = {}
        with WORD_SUGGESTION_BIGRAM_PATH.open("r", encoding="utf-8", errors="ignore") as source:
            for line in source:
                try:
                    phrase, raw_count = line.rstrip("\n").rsplit("\t", 1)
                    previous, following = phrase.lower().split(" ", 1)
                    if following in vocabulary:
                        bucket = bigrams.setdefault(previous, {})
                        bucket[following] = bucket.get(following, 0) + int(raw_count)
                except (ValueError, TypeError):
                    continue
        self._unigrams = unigrams
        self._bigrams = bigrams
        self._vocabulary = tuple(sorted(vocabulary))

    @property
    def neural_ready(self) -> bool:
        return self._session is not None and self._tokenizer is not None

    def status(self) -> dict:
        return {
            "ready": bool(self._loaded and self._unigrams),
            "status": "ready" if self._loaded and self._unigrams else "unavailable",
            "engine": "distilgpt2_int8_hybrid" if self.neural_ready else "google_web_1t_bigram",
            "neural_ready": self.neural_ready,
            "fallback_ready": bool(self._unigrams),
            "offline": True,
            "downloads_on_request": False,
            "vocabulary_size": len(self._vocabulary),
            "error": self._load_error,
            "model_revision": MODEL_REVISION,
        }

    def _candidates(self, prefix: str, context: list[str]) -> list[_Candidate]:
        needle = "".join(normalize_words([prefix]))
        previous = context[-1] if context else ""
        next_counts = self._bigrams.get(previous, {})
        if needle:
            words = [word for word in self._vocabulary if word.startswith(needle)]
            words.sort(key=lambda word: (-next_counts.get(word, 0), -self._unigrams.get(word, 1), word))
            words = words[:256]
        elif next_counts:
            contextual = sorted(next_counts, key=lambda word: (-next_counts[word], word))[:192]
            common = sorted(self._unigrams, key=lambda word: (-self._unigrams[word], word))[:64]
            words = list(dict.fromkeys([*contextual, *common]))
        else:
            words = sorted(self._unigrams, key=lambda word: (-self._unigrams[word], word))[:256]
        return [
            _Candidate(word, self._unigrams.get(word, 1), next_counts.get(word, 0))
            for word in words
        ]

    def _statistical_scores(self, context: list[str], candidates: list[_Candidate]) -> dict[str, float]:
        previous = context[-1] if context else ""
        next_counts = self._bigrams.get(previous, {})
        total_next = sum(next_counts.values())
        vocabulary_total = sum(self._unigrams.values()) or 1
        scores: dict[str, float] = {}
        for candidate in candidates:
            unigram_probability = candidate.unigram / vocabulary_total
            if total_next:
                conditional = (next_counts.get(candidate.word, 0) + 0.04 * unigram_probability * total_next) / (1.04 * total_next)
                if next_counts.get(candidate.word, 0):
                    # Discount ubiquitous function words without deleting
                    # them. This association term lets informative collocates
                    # such as ``happy birthday`` compete with raw-frequency
                    # leaders such as ``happy to``.
                    conditional_log = math.log(max(conditional, 1e-15))
                    scores[candidate.word] = conditional_log - 0.30 * math.log(max(unigram_probability, 1e-15))
                    continue
            else:
                conditional = unigram_probability
            scores[candidate.word] = math.log(max(conditional, 1e-15))
        return scores

    @staticmethod
    def _log_probability(logits: np.ndarray, token_id: int) -> float:
        maximum = float(np.max(logits))
        return float(logits[token_id] - maximum - math.log(float(np.exp(logits - maximum).sum())))

    def _run_neural(self, input_ids: list[int]) -> np.ndarray:
        assert self._session is not None
        token_array = np.asarray([input_ids], dtype=np.int64)
        empty = np.empty((1, 12, 0, 64), dtype=np.float32)
        feeds: dict[str, np.ndarray] = {
            "input_ids": token_array,
            "attention_mask": np.ones_like(token_array),
            "use_cache_branch": np.asarray([False], dtype=np.bool_),
        }
        for layer in range(6):
            feeds[f"past_key_values.{layer}.key"] = empty
            feeds[f"past_key_values.{layer}.value"] = empty
        return self._session.run(["logits"], feeds)[0][0]

    def _neural_scores(self, context: list[str], candidates: list[_Candidate]) -> dict[str, float]:
        if not self.neural_ready or not candidates:
            return {}
        assert self._tokenizer is not None and self._session is not None
        text = " ".join(context[-WORD_SUGGESTION_CONTEXT_TOKENS:]).strip() or "The"
        input_ids = self._tokenizer.encode(text).ids[-WORD_SUGGESTION_CONTEXT_TOKENS:]
        if not input_ids:
            input_ids = [50256]
        first_logits = self._run_neural(input_ids)[-1]
        results: dict[str, float] = {}
        for candidate in candidates:
            ids = self._candidate_token_ids.get(candidate.word)
            if ids is None:
                ids = tuple(self._tokenizer.encode(f" {candidate.word}").ids)
                self._candidate_token_ids[candidate.word] = ids
            if ids:
                score = self._log_probability(first_logits, ids[0])
                if len(ids) > 1:
                    continuation_logits = self._run_neural([*input_ids, *ids[:-1]])
                    start = len(input_ids) - 1
                    for offset, token_id in enumerate(ids):
                        if offset:
                            score += self._log_probability(continuation_logits[start + offset], token_id)
                    score /= len(ids)
                results[candidate.word] = score
        return results

    def suggest(
        self,
        *,
        context: Sequence[object] = (),
        sentence: str = "",
        prefix: str = "",
        limit: int = 8,
    ) -> dict:
        started = time.perf_counter()
        self.load()
        context_words = normalize_words(context) or normalize_words([sentence])
        clean_prefix = "".join(normalize_words([prefix]))
        candidates = self._candidates(clean_prefix, context_words)
        if not candidates:
            return self._result([], context_words, clean_prefix, limit, started, "none")
        statistical = self._statistical_scores(context_words, candidates)
        neural_candidates = sorted(
            candidates,
            key=lambda candidate: (-statistical[candidate.word], len(candidate.word), candidate.word),
        )[:48]
        neural = self._neural_scores(context_words, neural_candidates)
        if neural:
            # Causal probability is primary. The large-corpus bigram prior
            # improves names/phrases that the small neural model underexposes.
            # Large-corpus word statistics are deliberately the stronger
            # signal for short sign-built fragments; the causal LM resolves
            # longer-context ambiguity and supplies grammatical preference.
            scores = {word: 0.10 * neural[word] + 0.90 * statistical[word] for word in neural}
            engine = "distilgpt2_int8_hybrid"
        else:
            scores = statistical
            engine = "google_web_1t_bigram"
        ranked = sorted(scores.items(), key=lambda item: (-item[1], len(item[0]), item[0]))
        bounded_limit = min(MAX_WORD_SUGGESTIONS, max(1, int(limit)))
        ranked = ranked[:bounded_limit]
        probabilities = _softmax([score for _, score in ranked])
        suggestions = [
            {"word": word, "score": round(probability, 6), "rank": index + 1}
            for index, ((word, _), probability) in enumerate(zip(ranked, probabilities))
        ]
        return self._result(suggestions, context_words, clean_prefix, bounded_limit, started, engine)

    def _result(self, suggestions, context, prefix, limit, started, engine) -> dict:
        return {
            "suggestions": suggestions,
            "context": context,
            "prefix": prefix,
            "limit": limit,
            "engine": engine,
            "offline": True,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        }
