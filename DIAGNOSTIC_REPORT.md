# SignBridge engineering and model diagnostic

Audit date: 2026-08-13  
Scope: browser frontend, FastAPI backend, REST and WebSocket contracts, model
provenance/download/loading, inference preprocessing, datasets, training code,
runtime safety, and scientific validity.

## Executive verdict

The original browser recognition path was not operational. Alphabet mode was
wired to a second, incompatible Hugging Face loader and therefore returned
`Alphabet model not loaded`. Word mode could execute, but it had no class map
and received landmarks in a coordinate system and joint order different from
the model's training data. It consequently returned numeric class IDs for
out-of-distribution inputs and marked them accepted.

These were contract and architecture defects, not a threshold-tuning problem.
The repair establishes one deterministic alphabet service for REST and
WebSocket inference, makes rejection observable end to end, restores the exact
100-word class map, converts browser landmarks to the documented 55-joint
contract, validates pinned assets by SHA-256, and makes health/readiness
truthful. The word model remains experimental because converting MediaPipe
landmarks to an OpenPose-trained model is an approximation.

The project must **not** claim continuous sign-language translation or
near-perfect real-world accuracy. At this stage it is an isolated static-letter
recognizer plus an experimental isolated-word classifier.

## Integrated product follow-up

The repaired runtime is now exposed as one two-page application. The first page
keeps the conservative alphabet recognizer and adds a one-second, re-arm-aware
sign keyboard with autocomplete, sentence composition, and distinct letter/word
tones. The second page accepts typed text or an explicit microphone recording,
maps it into an ordered concept sequence, plays checksum-verified local sign
clips where available, and labels fingerspelling fallback honestly.

Autocomplete now uses a bounded local prediction API rather than only filtering
a fixed list. A quantized DistilGPT2 causal model scores continuations from the
completed sentence, Google Web 1T unigram/bigram statistics broaden and rerank
the candidates, and the browser filters by the currently signed prefix. All
assets are pinned and hash-verified; the statistical path remains available if
the optional neural runtime cannot load, and serving never downloads a model.

The follow-up also vendors the browser word tracker for offline use, verifies the
local Whisper checkpoint before reporting voice readiness, caps decoded audio at
30 seconds, prevents unbounded speech inference queues, and discloses whether an
optional remote NLP or sign-lookup provider is enabled. These product changes do
not alter the scientific boundary below: native visual coverage is still small,
and the word classifier remains an experimental isolated-gloss model.

## Audit method

The review used five independent evidence sources:

1. Static tracing of every user-facing route from browser capture through API
   serialization, preprocessing, model invocation, and UI display.
2. Offline startup and inference reproduction under the installed Windows
   environment, including its `cp1256` console encoding.
3. Checkpoint structure, tensor-key, class-count, label-order, path, cache,
   revision, and SHA-256 inspection.
4. Browser runtime inspection with camera acquisition unavailable, to test
   readiness/error states without granting camera permission.
5. A cross-domain alphabet probe using 72 labeled test images recoverable from
   the repository's truncated Roboflow archive. This is an audit probe, not a
   statistically complete benchmark.

Installed audit runtime: Python 3.11.9, CPU PyTorch 2.13, torchvision 0.28,
Transformers 4.46, FastAPI 0.141, MediaPipe Tasks 1.0, and OpenCV 5.

## Reproduced failures in the original system

| Severity | Area | Reproduction | Root cause |
|---|---|---|---|
| P0 | Alphabet live stream | Every alphabet WebSocket request returned `Alphabet model not loaded`. | `backend/main.py` bypassed the local pipeline and instantiated `AutoImageProcessor` for a repository declaring unsupported `ResNetImageProcessor`. |
| P0 | Alphabet model validity | Bypassing the processor yielded a two-label generic ResNet with many newly initialized weights. | The remote checkpoint is a custom 29-class ASLResNet; `AutoModelForImageClassification` is not its implementation and its config lacks the standard `num_labels`/`id2label` contract. |
| P0 | Windows startup | The caught model error could abort application lifespan with `UnicodeEncodeError`. | Emoji/mojibake startup logs were not encodable by the active Windows console. |
| P0 | Word output | A 50-frame constant input returned class `77`, confidence about 0.615, and `accepted: true`. | `wlasl100.txt` was absent, so indices were emitted; acceptance was hard-coded. |
| P0 | Word semantics | Browser arrays had the right length but the wrong meaning and range. | The browser sent a different 13-body MediaPipe set in `[0,1]`; training used selected OpenPose BODY_25 joints plus hands in `[-1,1]`. |
| P0 | UI rejection | Low-confidence predictions were shown and added to history. | The browser ignored the server's `accepted: false` field. |
| P1 | Health | `/health` returned `status: healthy` while alphabet was absent. | Health tested process existence, not capability readiness or model self-tests. |
| P1 | Word temporal logic | Three almost identical windows appeared to form a stable vote. | Step-one sliding windows shared 49 of 50 frames, so the votes were not independent. |
| P1 | Asset loading | Startup depended on whatever happened to exist in recursive/global caches. | Revisions were not pinned, hashes were not checked, paths contradicted one another, and downloaded config was ignored. |
| P1 | Training | `train.py`, `evaluate.py`, and `extract_keypoints.py` failed before argument parsing. | Missing config symbols/modules and mutually incompatible PoseFormer, CNN-LSTM, and production TGCN schemas. |
| P1 | Browser metrics | Camera failure still displayed roughly 10 FPS and 0 ms latency. | FPS counted a timer, not processed frames; no request timing was correlated. |
| P1 | API safety | Large inputs, blocking compute, permissive CORS, and raw internal errors were possible. | Declared limits were unused, work ran on the event loop, and trust boundaries were not enforced. |

## Quantitative alphabet probe

Baseline/provenance note: these measurements were produced during the audit of
the **pre-repair** image-ResNet path and the already-bundled keypoint model. The
exact archive recovery and scoring commands were executed in the audited local
environment but were not previously versioned as a repository benchmark
script. Treat the table as reproducible forensic evidence for the architectural
decision, not as a maintained benchmark artifact. The new regression suite
tests contracts and rejection; a new integrity-checked camera benchmark must be
created for repeatable performance claims.

The included archive is corrupt/truncated, but its first 72 labeled test images
can be recovered sequentially. They span 24 letters (E and L are absent), with
one to five samples per represented class. Results from this small set are
useful for detecting catastrophic domain shift, but not for estimating final
population accuracy.

| Path | Detection coverage | Top-1 among evaluated samples | Mean confidence | Accepted at 0.65 | Accuracy among accepted |
|---|---:|---:|---:|---:|---:|
| Bundled image ResNet, full frame | 72/72 | 6.9% | 0.852 | 59/72 | 6.8% |
| MediaPipe detection -> padded ROI -> image ResNet | 64/72 | 10.9% | high and miscalibrated | 56/64 | 10.7% |
| MediaPipe landmarks -> bundled Random Forest | 64/72 | 67.2% | not used alone as an accuracy claim | 17/64 | 94.1% |

The original CNN was therefore confidently wrong out of domain. Raising or
lowering its softmax threshold cannot repair the learned representation. The
Random Forest is used as a conservative interim static-handshape recognizer
because its selective result was materially safer on the same probe. This does
not establish production quality; it establishes the least unsafe available
local path.

## Model and preprocessing contracts

### Alphabet

- Task: one static handshape, A-Z class space.
- Runtime input: one decoded camera frame with one detected hand.
- Extractor: pinned MediaPipe Hand Landmarker Tasks asset.
- Features: 21 `(x,y)` points relative to the wrist, flattened to 42 values and
  normalized by the maximum absolute coordinate.
- Classifier: pinned local Random Forest; model feature count and classes are
  checked during load.
- Decision: a result below the provisional conservative minimum is returned as rejected and
  is not translated or placed in history. WebSocket sessions also require three
  short-lived consistent observations before accepting a static letter.
- Dynamic letters: `J` and `Z` remain rejected by the static model. A
  per-connection fingertip-trace check may accept them only after it observes a
  clear hook or Z-shaped path; this is a guarded heuristic, not a temporal model
  accuracy claim.
- Visualization: alphabet responses return the detected 21 landmarks so the
  browser can draw the recognition hand skeleton directly over the camera image.

### Isolated words

- Task: one isolated WLASL-100 gloss, not a sentence and not continuous signing.
- Runtime tensor: 50 frames x 55 joints x 2 coordinates.
- Joint order: OpenPose BODY_25 indices `0..8,15..18`, then 21 left-hand and 21
  right-hand points.
- Normalization: training coordinates expressed on a 256-pixel reference canvas
  and mapped to `[-1,1]`; missing points map to `-1`.
- Label order: the exact sorted WLASL-100 gloss list is bundled and hash-checked;
  for example index 77 maps to `school`.
- Decision: confidence and top-one/top-two margin must both pass. Otherwise the
  response is a rejection, not a guessed word.
- Activity gate: sequences with inadequate hand coverage or effectively no
  temporal motion are rejected before the closed-set softmax can force an idle
  sequence into one of the 100 words. These provisional gates still require
  calibration on real no-sign and signed clips.
- Limitation: the checkpoint was trained on OpenPose. Browser conversion from
  MediaPipe approximates its schema but does not eliminate estimator/domain
  shift. A production version needs training on the exact live extractor.

## Asset provenance and load policy

`backend/checkpoints/models.lock.json` is the source of truth. Each asset has an
expected SHA-256. Remote assets also have an immutable repository revision and
filename. Resolution is deterministic: explicit project path, project cache,
default Hugging Face cache, then an optional pinned download. Offline mode never
silently downloads.

The important pinned revisions are:

- MediaPipe hand task source repository revision:
  `a76ea634b877b67ef5c1bb6bb7850d1d8926109b`.
- WLASL TGCN repository revision:
  `dacb4568719caa03c44764034f599a9f8a0f63f4`.

The runtime no longer scans arbitrary cache directories for a coincidentally
named checkpoint. PyTorch weights use restricted `weights_only=True` loading
where applicable, tensor keys and output shapes are checked, and a missing or
corrupt capability fails closed while the rest of the application can start.

The Random Forest is a hash-verified bundled-local artifact and has no declared
authoritative remote source. The bootstrap tool can validate but not recreate
it. Release packaging must include the exact locked file, or replace it only
through the grouped-data training and promotion gate. Because `joblib` is a
pickle-based format, the SHA-256 trust boundary must be established before
deserialization; a future release should prefer a safer interchange format.

## Frontend findings and repair

The original frontend coupled camera, WebSocket, and backend state; started the
camera immediately; trusted every server guess; generated misleading FPS and
latency; retained stale predictions after no-hand frames; and sent a word
feature layout different from training.

The repaired client:

- requests camera access only after an explicit button press;
- fetches capability readiness before enabling a recognition mode;
- displays backend, stream, camera, alphabet, and word status separately;
- keeps at most one alphabet frame in flight and correlates `request_id` and
  send time for real round-trip latency;
- counts processed frames rather than timer ticks;
- renders rejected/no-hand states without appending translation history;
- uses text nodes instead of injecting prediction strings with `innerHTML`;
- captures a deliberate, isolated 50-frame word clip at a bounded rate instead
  of pseudo-voting over 49/50-overlapping windows;
- cancels stale suggestion requests, ranks results from sentence context plus
  the current prefix, and gives Enter and click one duplicate-safe commit path;
- clears stale output, provides keyboard/accessibility state, and uses a single
  external stylesheet.

## Backend findings and repair

The backend previously maintained two alphabet stacks with different behavior,
performed blocking work in async handlers, exposed misleading health, accepted
unbounded payloads, returned internal exception text, and mixed static URL
prefixes.

The repaired backend:

- initializes one shared alphabet pipeline used by both REST and WebSocket;
- loads capabilities independently and reports degraded readiness honestly;
- performs model inference and blocking auxiliary work off the event loop;
- validates message type, mode, base64 size, decoded dimensions, landmark count,
  numeric finiteness/range, and word frame count;
- uses per-connection word buffers and resets them after one prediction;
- propagates `accepted`, rejection reason, candidates, request ID, and measured
  inference timing through one structured response contract;
- binds locally by default, restricts origins, and avoids raw exception details;
- removes non-ASCII startup symbols so caught failures stay caught on Windows;
- bounds upload sizes and avoids recording raw transcripts in latency logs.

## Training and dataset assessment

The repository cannot currently support a scientifically valid retraining
claim:

- `data/keypoints` contains 2,000 synthetic arrays. The saved experimental
  checkpoint reached about 0.00667 validation/test accuracy, below the 1% random
  baseline for 100 balanced classes.
- The local WLASL-20 manifest has 20 requested samples but only two videos
  available (`drink` and `like`).
- The camera-style alphabet training directory is absent.
- The Roboflow archive is truncated/corrupt and cannot serve as an integrity-
  checked training or benchmark release.
- Legacy training, evaluation, and extraction files describe incompatible
  feature dimensions and architectures, and validation/test subsets inherited
  augmentation through one shared dataset object.

For that reason, the repair does not present a synthetic retraining result as a
fix. `training/README.md` records the production contracts and promotion gate.
The replacement word training tools use one Pose-TGCN/OpenPose-55 schema and
reject synthetic, incomplete, or split-leaking manifests by default. The legacy
experiment remains historical evidence only and is not a source of runtime
checkpoints or headline metrics.

## Scientific promotion gate

A credible next model release should meet all of these conditions:

1. Real multi-signer webcam data, including varied devices, lighting,
   backgrounds, distance, orientation, skin tones, handedness, occlusion, idle
   hands, transitions, no-hand frames, and unknown signs.
2. Signer/source-disjoint train, validation, calibration, and frozen test sets.
3. One extractor/schema used identically for data preparation and production.
4. Checkpoint metadata containing architecture, input schema version, FPS,
   sequence length, normalization, label hash, sample manifest hash, data/code
   revision, and dependency versions.
5. Macro F1, per-class recall, confusion matrix, calibration error, detector
   recall, no-sign false-accept rate, selective accuracy-versus-coverage, and
   p50/p95 end-to-end latency.
6. Separate evaluation for static letters and dynamic J/Z, plus isolated signs
   versus continuous signing. These tasks must not be conflated.
7. A declared acceptance operating point chosen on calibration data and tested
   once on the frozen set. Do not tune against the test set.

## Residual risks

1. **Word domain shift remains high.** The OpenPose-trained TGCN receives a
   MediaPipe-derived approximation. Keep the mode visibly experimental until a
   same-extractor model passes the promotion gate.
2. **Alphabet coverage is intentionally conservative.** Rejection improves
   trustworthiness but reduces coverage, and the 72-image audit set is too small
   and imbalanced to set a final production threshold.
3. **Static letters J/Z remain scientifically mismatched.** Add a short temporal
   trajectory classifier or declare those two letters unsupported.
4. **No continuous-language model exists.** Sentence segmentation, coarticulation,
   non-manual markers, grammar, and signer adaptation are outside this model.
5. **Text/speech-to-sign is a token/video lookup prototype.** Its optional
   semantic service is not a validated ASL translation model, and third-party
   online clip lookup is disabled by default because availability, terms, and
   content can change.
6. **Camera testing is still environment-dependent.** Automated tests cover
   contracts and failure states; representative physical camera validation must
   be repeated on deployment hardware.
7. **Word capture still has a lazy CDN dependency.** Alphabet mode is local,
   but experimental MediaPipe Holistic browser assets are version-pinned and
   loaded from jsDelivr on demand. Vendor them for a fully offline deployment.
8. **Speech recognition is an optional uncached capability on this machine.**
   FFmpeg and the Whisper package are installed, but the `base` model is not.
   The backend now reports this honestly and refuses an implicit download when
   offline; `scripts/fetch_speech_model.py` performs the explicit project-cache
   download and load test when speech APIs are required.

## Priority roadmap

1. Collect and freeze a signer-disjoint camera validation set; calibrate the
   static alphabet rejection threshold and publish selective metrics.
2. Add temporal J/Z recognition and an explicit unknown/no-sign class.
3. Retrain isolated-word recognition on the exact 55-point MediaPipe-derived
   schema, fixed FPS, and real clips; then repeat calibration and OOD testing.
4. Add motion/activity segmentation only after isolated-word quality is sound.
5. If continuous translation is the research goal, build and evaluate a
   sequence-to-sequence or CTC-style pipeline with non-manual features and an
   appropriate continuous ASL corpus; do not extend the isolated classifier by
   accumulating overlapping guesses.
6. Package the verified model cache and frontend dependencies for a fully
   offline deployment, and add CI on Windows plus at least one Linux target.

## Conclusion

The immediate failures were repaired at their interfaces: loader, model
contract, label map, readiness, rejection, temporal capture, and UI state. The
project is now structured to fail closed and explain what is unavailable. Its
remaining weaknesses are principally data and task-definition problems; those
require real, properly split observations and cannot be solved honestly by a
new confidence number or another synthetic training run.

## External model documentation consulted

- [Official WLASL repository](https://github.com/dxli94/WLASL) and its
  [Pose-TGCN dataset preprocessing code](https://github.com/dxli94/WLASL/blob/master/code/TGCN/sign_dataset.py).
- [Pinned WLASL-100 Pose-TGCN model repository](https://huggingface.co/sharonn18/tgcn-wlasl).
- [Bundled ResNet/hand-landmarker source repository](https://huggingface.co/huzaifanasirrr/realtime-sign-language-translator).
- [Custom 29-class ASLResNet repository used by the broken original loader](https://huggingface.co/Abuzaid01/asl-sign-language-classifier).

Repository cards and self-reported metrics were treated as provenance and
limitations, not as independent validation of this application.
