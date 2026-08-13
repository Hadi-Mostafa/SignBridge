"""Contract-only tests; temporary synthetic values are never performance data."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.extract_keypoints import (  # noqa: E402
    BODY_25_INDICES,
    ContractError,
    JOINT_COUNT,
    parse_openpose_frame,
    resample_sequence,
    validate_source_manifest,
)
from training.train import load_config, sequence_to_model_input  # noqa: E402


class TrainingContractTests(unittest.TestCase):
    def _labels(self) -> list[str]:
        return [f"label-{index:03d}" for index in range(100)]

    def _source_sample(self, split: str, suffix: str, label: str = "label-000") -> dict[str, str]:
        return {
            "sample_id": f"captured-{suffix}",
            "label": label,
            "split": split,
            "signer_id": f"signer-{suffix}",
            "source_id": f"source-{suffix}",
            "openpose_dir": f"frames/{suffix}",
        }

    def test_openpose_joint_order_normalization_and_shape(self) -> None:
        body = []
        for index in range(25):
            body.extend([float(index), float(index + 100), 0.9])
        left = []
        right = []
        for index in range(21):
            left.extend([float(index + 20), float(index + 40), 0.8])
            right.extend([float(index + 60), float(index + 80), 0.7])
        document = {
            "people": [
                {
                    "pose_keypoints_2d": body,
                    "hand_left_keypoints_2d": left,
                    "hand_right_keypoints_2d": right,
                }
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            frame_path = Path(directory) / "frame_keypoints.json"
            frame_path.write_text(json.dumps(document), encoding="utf-8")
            frame = parse_openpose_frame(frame_path)
        self.assertEqual(frame.shape, (55, 2))
        expected_body_x = 2.0 * (np.asarray(BODY_25_INDICES, dtype=np.float32) / 256.0 - 0.5)
        np.testing.assert_allclose(frame[:13, 0], expected_body_x)
        self.assertAlmostEqual(float(frame[13, 0]), 2.0 * (20.0 / 256.0 - 0.5))
        self.assertAlmostEqual(float(frame[34, 0]), 2.0 * (60.0 / 256.0 - 0.5))

    def test_resample_exactly_50_and_model_layout(self) -> None:
        frames = np.zeros((3, JOINT_COUNT, 2), dtype=np.float32)
        frames[0, :, :] = -1.0
        frames[1, :, :] = 0.0
        frames[2, :, :] = 1.0
        sequence = resample_sequence(frames)
        model_input = sequence_to_model_input(sequence)
        self.assertEqual(sequence.shape, (50, 55, 2))
        self.assertEqual(model_input.shape, (55, 100))
        np.testing.assert_allclose(model_input[0], sequence[:, 0, :].reshape(100))

    def test_source_manifest_rejects_synthetic_and_split_leakage(self) -> None:
        labels = self._labels()
        base = {"dataset": {"name": "captured-fixture", "synthetic": False}}
        synthetic = {**base, "samples": [{**self._source_sample("train", "one"), "synthetic": True}]}
        with self.assertRaisesRegex(ContractError, "synthetic"):
            validate_source_manifest(synthetic, labels, require_complete=False)
        leaky = {
            **base,
            "samples": [
                self._source_sample("train", "one"),
                {**self._source_sample("test", "two"), "signer_id": "signer-one"},
            ],
        }
        with self.assertRaisesRegex(ContractError, "Signer-disjoint"):
            validate_source_manifest(leaky, labels, require_complete=False)

    def test_complete_gate_requires_every_label_in_every_split(self) -> None:
        document = {
            "dataset": {"name": "captured-fixture", "synthetic": False},
            "samples": [self._source_sample("train", "only")],
        }
        with self.assertRaisesRegex(ContractError, "incomplete"):
            validate_source_manifest(document, self._labels(), require_complete=True)

    def test_production_config_matches_runtime_architecture(self) -> None:
        config_path = PROJECT_ROOT / "training" / "config.yaml"
        config = load_config(config_path)
        self.assertEqual(config["contract"]["sequence_length"], 50)
        self.assertEqual(config["contract"]["joint_count"], 55)
        self.assertEqual(config["model"]["input_feature"], 100)
        self.assertEqual(config["model"]["num_classes"], 100)
        self.assertEqual(config["model"]["num_stages"], 20)
        self.assertNotIn("backend/checkpoints", config["artifacts"]["output_root"].as_posix())

    def test_bad_coordinate_and_bad_shape_fail(self) -> None:
        frames = np.zeros((1, 55, 2), dtype=np.float32)
        frames[0, 0, 0] = 2.0
        with self.assertRaisesRegex(ContractError, "normalized"):
            sequence_to_model_input(np.repeat(frames, 50, axis=0))
        with self.assertRaisesRegex(ContractError, "shape"):
            sequence_to_model_input(np.zeros((30, 55, 2), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
