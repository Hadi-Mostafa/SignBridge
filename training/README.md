# Training status and model contracts

The runtime and the legacy training scripts in this directory do **not** share
one interchangeable checkpoint format. Treating them as if they did was one of
the reasons the application appeared to train successfully while inference did
not improve.

## Models used by the application

### Alphabet mode

The repaired runtime uses the bundled Random Forest keypoint classifier for
static A-Z handshapes. Its input is exactly 42 values: the `(x, y)` coordinates
of 21 MediaPipe hand landmarks, translated relative to the wrist and divided by
the largest absolute coordinate. The runtime validates that feature count and
the model's 26 classes at startup.

This is a static classifier. The ASL letters **J** and **Z** contain motion and
cannot be recognized scientifically from one frame. They require a temporal
model or must be declared unsupported in a static-only evaluation.

`train_alphabet.py`, `evaluate_alphabet.py`, and `export_alphabet_onnx.py`
describe a separate image-classification experiment. Their MobileNet
checkpoint is not the current production classifier. Do not copy a resulting
file into production unless the runtime is deliberately migrated and the full
camera-domain test gate below passes.

### Isolated-word mode

The runtime uses the pinned `sharonn18/tgcn-wlasl` ASL-100 checkpoint. Its
contract is:

- 50 frames per isolated clip;
- 55 joints per frame;
- 2 coordinates per joint;
- OpenPose BODY_25 joints `0..8, 15..18`, followed by all 21 left-hand and all
  21 right-hand joints;
- coordinates transformed from the original 256-pixel reference system to
  `[-1, 1]`;
- the exact 100-label ordering in `backend/checkpoints/wlasl100.txt`.

The browser converts its MediaPipe landmarks into this contract as an
approximation. MediaPipe and OpenPose are different estimators, so word mode is
still an **experimental isolated-sign classifier**, not continuous sign-language
translation. A robust production model should be retrained end to end on the
same MediaPipe extractor that is used at inference.

## Canonical word training tools

`extract_keypoints.py`, `train.py`, and `evaluate.py` now use the same
50-frame, 55-joint Pose-TGCN/OpenPose contract as production. They require
explicit source and extracted manifests, signer/source-disjoint splits, real
data, and isolated experiment output paths. Synthetic or incomplete datasets
fail closed. The historical 30x225 PoseFormer/CNN-LSTM experiment and its
near-random checkpoint remain data-provenance evidence only; the replacement
tools do not load or overwrite them.

Run each command with `--help` for the exact manifest schema and arguments.

## Minimum data and evaluation gate

Before promoting a newly trained model, require all of the following:

1. Record real webcam samples from multiple signers, devices, backgrounds,
   skin tones, distances, orientations, and lighting conditions.
2. Split by signer and source video before any augmentation. No signer or clip
   may appear in more than one split.
3. Include explicit no-hand, idle-hand, transition, unknown-sign, and
   out-of-vocabulary negatives.
4. Freeze the test manifest before tuning. Save the sample IDs and a SHA-256
   digest with the checkpoint.
5. Serialize the architecture, input schema version, FPS, sequence length,
   normalization, label list and hash, dependency versions, data revision, and
   code revision with every checkpoint.
6. Report macro F1, per-class recall, confusion matrix, calibration error,
   accepted coverage versus selective accuracy, no-sign false-accept rate,
   detector recall, and end-to-end p50/p95 latency. Accuracy alone is not
   sufficient.
7. Run a blind camera-domain acceptance set. A useful starting gate is at least
   90% selective accuracy at a declared coverage, with the false-accept target
   chosen for the intended risk level.

The repository currently does not contain enough real word videos to meet this
gate. That is a data limitation, not something threshold tuning can repair.
