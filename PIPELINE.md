# Runtime pipeline contracts

This file documents the behavior implemented by the current application. Model
and frontend code must fail at startup or reject a request when these contracts
do not match; silently reshaping incompatible data is forbidden.

## Alphabet path

```text
Browser camera JPEG
  -> bounded decode and pixel validation
  -> MediaPipe Hand Landmarker (one hand)
  -> 21 wrist-relative (x,y) landmarks, max-absolute normalization
  -> 42-feature Random Forest
  -> per-WebSocket temporal evidence and confidence gate
  -> crossed-finger `R`/`U` cue when those classes compete
  -> guarded fingertip trajectory check for `J` and `Z`
  -> accepted letter OR uncertain candidate OR no_hand
```

`POST /predict` and `/ws/sign-to-text` call the same stateless pipeline. The
service has no cross-client temporal queue. The browser commits one accepted
letter only when its one-second cooldown permits it. A held identical pose cannot
repeat. `no_hand` re-arms the same letter; a genuinely different accepted letter
may proceed after the cooldown. The accepted character updates autocomplete and
sentence-composition state, while rejected candidates never type.

The shared classifier remains single-frame and stateless. Each WebSocket owns a
small, short-lived evidence buffer: three consistent frames are required for a
static acceptance, and `J`/`Z` need their characteristic fingertip path. The
trajectory check deliberately rejects unclear motion and must not be presented
as continuous-sign recognition. Alphabet responses include the 21 detected
landmarks, which the browser renders as a colored hand skeleton over the exact
camera frame sent for inference.

## Experimental isolated-word path

```text
Explicit capture_start
  -> 50 valid frames, at no more than 25 samples/s
  -> 13 approximated OpenPose BODY_25 joints + 21 left hand + 21 right hand
  -> 55 (x,y) points/frame normalized to [-1,1], missing points = (-1,-1)
  -> tensor [1,55,100]
  -> hand-coverage and temporal-activity rejection gate
  -> pinned Pose-TGCN WLASL-100 model
  -> confidence + top-two margin gate
  -> one result, then mandatory reset
```

The 13 body positions correspond to retained BODY_25 indices `0..8,15..18`:
nose, neck, right shoulder/elbow/wrist, left shoulder/elbow/wrist, mid-hip,
right eye, left eye, right ear, and left ear. The browser derives neck and
mid-hip as shoulder/hip midpoints.

The checkpoint was trained with OpenPose, not MediaPipe. The conversion is an
approximation and is why the mode is labeled experimental. It classifies one
isolated gloss; it performs neither motion segmentation nor sentence
translation.

## Context-aware word-suggestion path

```text
Committed sentence + current letter prefix
  -> bounded/debounced POST /api/word-suggestions
  -> hash-verified project-local quantized DistilGPT2 causal scoring
  -> Google Web 1T unigram/bigram association reranking
  -> deterministic statistical fallback when the neural runtime is unavailable
  -> ranked, prefix-filtered candidates
  -> Arrow-key selection and shared Enter/click commit transition
```

The runtime performs no request-time download. Model, tokenizer, unigram, and
bigram artifacts are pinned in `backend/checkpoints/models.lock.json`. The
service accepts at most 64 context tokens / 500 characters and returns at most
12 candidates. Client requests use cancellation and generation IDs so an older
response cannot overwrite suggestions for newer input.

## Text and voice to visual-sign path

```text
Typed text OR explicit browser microphone recording
  -> bounded JSON or multipart validation
  -> decoded-duration validation (30-second default)
  -> SHA-256-verified project-local Whisper transcription for voice
     (lazy load, no request-time download, one active inference)
  -> deterministic tokenization, lemmatization, synonyms, longest phrase match
  -> ordered source-linked concepts
  -> SHA-256-verified native sign clip when available
     (longest declared phrase takes precedence)
  -> otherwise SHA-256-verified schematic landmark guides for static letters
     (J/Z are explicitly motion-required and not rendered as static poses)
  -> coverage + translation-status metadata
```

`POST /api/text-to-sign` and `POST /api/speech-to-sign` return the same
`TranslationResponse` shape. It includes the source text, normalized concepts,
source spans, ordered `sign_clips`, coverage counts, latency, and an honest
translation-status disclaimer. `GET /api/vocabulary` supplies autocomplete and
reports actual native-clip coverage.

The bundled manifest currently verifies native clips for `drink` and `like`.
It records media hash, MIME/container details, WLASL/source provenance,
attribution, and the unresolved redistribution scope. Manifest-only phrases are
added to the NLP vocabulary so phrase assets work without classifier-label
changes. Fingerspelling landmark guides are deterministic visual fallbacks, not
native lexical signs, hand photographs, or a substitute for full ASL grammar.

## WebSocket request contract

Alphabet frame:

```json
{"type":"frame","mode":"alphabet","request_id":"...","sent_at":0,"image":"<base64 JPEG>"}
```

Word capture:

```json
{"type":"capture_start","mode":"words","request_id":"...","total":50}
{"type":"frame","mode":"words","request_id":"...","frame_index":0,"landmarks":[110]}
{"type":"capture_end","mode":"words","request_id":"...","total":50}
```

Reset:

```json
{"type":"reset","mode":"words","request_id":"..."}
```

## WebSocket response contract

Every envelope includes `type`, `request_id`, `accepted`, `reason`,
`confidence`, `top_predictions`, `top5`, and `server_latency_ms`.

- `prediction` adds `mode`, accepted labels or a rejected `candidate_label`,
  optional box/model/detector/margin, the 21 alphabet landmarks when detected,
  and detailed latency.
- `no_hand` is an explicit non-error rejection with null labels.
- `progress` reports the captured frame count.
- `error` contains a stable code and public message; internal tracebacks remain
  in server logs.

## Readiness and model loading

Application lifespan loads alphabet, words, and lightweight text services
independently. Failure of one does not invent readiness for another. Health
reports text and voice readiness separately. Whisper weights remain lazy even
when cached and are loaded only for an explicit voice request. Voice readiness
requires the local checkpoint to pass its expected SHA-256, and runtime loading
uses that verified absolute path. Concurrent speech requests fail with a stable
`speech_busy` response rather than waiting in an unbounded worker queue.

The browser's experimental word tracker is also local: the pinned MediaPipe
Holistic JavaScript, WASM, graph, packed assets, and full pose model are served
from `frontend/assets/mediapipe` and verified against its manifest. The UI keeps
word capture disabled until that lazy client-side tracker initializes.

Locked assets are resolved from deterministic project or Hugging Face cache
locations and verified by SHA-256. Remote downloads use immutable revisions and
are disabled by `SIGNBRIDGE_OFFLINE=1` or `MODEL_DOWNLOAD_ENABLED=false`.

Run:

```powershell
.\venv\Scripts\python.exe scripts\fetch_models.py
.\venv\Scripts\python.exe scripts\doctor.py --load-models
```

## Scientific boundary

Confidence is not the same as correctness. Thresholds control selective
coverage only after calibration; they cannot repair domain mismatch. Read
`DIAGNOSTIC_REPORT.md` for the measured failure of the original image CNN, the
data limitations, and the required signer-disjoint evaluation gate.
