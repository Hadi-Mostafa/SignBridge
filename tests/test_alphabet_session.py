"""Deterministic tests for per-client alphabet temporal decoding."""

from __future__ import annotations

import unittest

import numpy as np

from backend.services.alphabet_session import AlphabetSessionDecoder, LABEL_TO_INDEX


def _landmarks(*, index_tip: tuple[float, float] = (0.48, 0.24), pinky_tip: tuple[float, float] = (0.72, 0.38), crossed: bool = False) -> list[list[float]]:
    points = np.array([[0.5, 0.7, 0.0]] * 21, dtype=np.float64)
    points[5] = [0.40, 0.56, 0.0]
    points[9] = [0.48, 0.54, 0.0]
    points[17] = [0.64, 0.56, 0.0]
    points[8] = [index_tip[0], index_tip[1], 0.0]
    points[12] = [0.58, 0.22, 0.0]
    points[20] = [pinky_tip[0], pinky_tip[1], 0.0]
    if crossed:
        points[8] = [0.61, 0.24, 0.0]
        points[12] = [0.44, 0.22, 0.0]
    return points.tolist()


def _result(label: str, landmarks: list[list[float]], *, alternate: str | None = None) -> dict:
    scores = np.full(26, 0.002, dtype=np.float64)
    scores[LABEL_TO_INDEX[label]] = 0.90
    if alternate:
        scores[LABEL_TO_INDEX[label]] = 0.72
        scores[LABEL_TO_INDEX[alternate]] = 0.18
    scores /= scores.sum()
    return {
        "label": label,
        "sign": label,
        "class_id": LABEL_TO_INDEX[label],
        "confidence": float(scores[LABEL_TO_INDEX[label]]),
        "accepted": label not in {"J", "Z"},
        "reason": "accepted",
        "probabilities": scores.tolist(),
        "landmarks": landmarks,
        "top5": [],
    }


class AlphabetSessionDecoderTests(unittest.TestCase):
    def test_static_letter_waits_for_consistent_evidence(self) -> None:
        decoder = AlphabetSessionDecoder(0.65)
        frames = [decoder.update(_result("K", _landmarks()), timestamp=index * 0.1) for index in range(3)]
        self.assertFalse(frames[0]["accepted"])
        self.assertEqual(frames[0]["reason"], "temporal_stabilizing")
        self.assertTrue(frames[-1]["accepted"])
        self.assertEqual(frames[-1]["label"], "K")

    def test_crossed_index_and_middle_fingers_can_resolve_r_over_u(self) -> None:
        decoder = AlphabetSessionDecoder(0.65)
        frames = [
            decoder.update(_result("U", _landmarks(crossed=True), alternate="R"), timestamp=index * 0.1)
            for index in range(3)
        ]
        self.assertTrue(frames[-1]["accepted"])
        self.assertEqual(frames[-1]["label"], "R")

    def test_static_z_remains_rejected_without_a_trace(self) -> None:
        decoder = AlphabetSessionDecoder(0.65)
        frame = _result("Z", _landmarks())
        for index in range(4):
            outcome = decoder.update(frame, timestamp=index * 0.1)
        self.assertFalse(outcome["accepted"])
        self.assertEqual(outcome["reason"], "dynamic_letter_requires_motion")

    def test_z_requires_a_z_shaped_index_trace(self) -> None:
        decoder = AlphabetSessionDecoder(0.65)
        trace = [
            (0.28, 0.25), (0.39, 0.25), (0.50, 0.25), (0.43, 0.34),
            (0.36, 0.43), (0.46, 0.43), (0.57, 0.43),
        ]
        for index, point in enumerate(trace):
            outcome = decoder.update(_result("Z", _landmarks(index_tip=point)), timestamp=index * 0.1)
        self.assertTrue(outcome["accepted"])
        self.assertEqual(outcome["label"], "Z")
        self.assertEqual(outcome["reason"], "motion_confirmed_z")

    def test_j_requires_a_hooked_pinky_trace(self) -> None:
        decoder = AlphabetSessionDecoder(0.65)
        trace = [
            (0.68, 0.18), (0.68, 0.31), (0.68, 0.44), (0.68, 0.55),
            (0.76, 0.61), (0.86, 0.58),
        ]
        for index, point in enumerate(trace):
            outcome = decoder.update(_result("J", _landmarks(pinky_tip=point)), timestamp=index * 0.1)
        self.assertTrue(outcome["accepted"])
        self.assertEqual(outcome["label"], "J")
        self.assertEqual(outcome["reason"], "motion_confirmed_j")


if __name__ == "__main__":
    unittest.main(verbosity=2)
