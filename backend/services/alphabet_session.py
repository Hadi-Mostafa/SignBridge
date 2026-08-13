"""Per-WebSocket temporal decoding for live ASL alphabet recognition."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time
from typing import Any

import numpy as np


LABELS = tuple(chr(ord("A") + index) for index in range(26))
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}
DYNAMIC_TIPS = {"J": 20, "Z": 8}


@dataclass(frozen=True)
class _Observation:
    timestamp: float
    probabilities: np.ndarray
    landmarks: np.ndarray | None


class AlphabetSessionDecoder:
    """Keep temporal evidence scoped to one browser connection."""

    static_window_seconds = 0.55
    motion_window_seconds = 1.7
    minimum_static_samples = 3

    def __init__(self, confidence_threshold: float) -> None:
        self.confidence_threshold = float(confidence_threshold)
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1].")
        self._observations: deque[_Observation] = deque()

    def reset(self) -> None:
        self._observations.clear()

    def update(self, result: dict[str, Any], *, timestamp: float | None = None) -> dict[str, Any]:
        """Return a stabilized result without mutating the shared pipeline output."""

        probabilities = self._probabilities(result)
        if probabilities is None:
            return dict(result)

        now = time.monotonic() if timestamp is None else float(timestamp)
        landmarks = self._landmarks(result)
        self._observations.append(_Observation(now, probabilities, landmarks))
        self._discard_stale(now)

        aggregate, sample_count, consensus = self._static_evidence()
        if aggregate is None:
            return dict(result)

        dynamic = self._dynamic_decision()
        if dynamic is not None:
            label, confidence, motion_score = dynamic
            output = self._with_prediction(
                result,
                label=label,
                confidence=confidence,
                accepted=True,
                reason=f"motion_confirmed_{label.lower()}",
                scores=self._promote_dynamic_label(aggregate, label),
            )
            output["motion"] = {"label": label, "score": round(motion_score, 4)}
            self.reset()
            return output

        class_index = int(np.argmax(aggregate))
        label = LABELS[class_index]
        confidence = float(aggregate[class_index])
        if label in DYNAMIC_TIPS:
            accepted, reason = False, "dynamic_letter_requires_motion"
        elif confidence < self.confidence_threshold:
            accepted, reason = False, "low_confidence"
        elif sample_count < self.minimum_static_samples or consensus < 2:
            accepted, reason = False, "temporal_stabilizing"
        else:
            accepted, reason = True, "accepted"

        return self._with_prediction(
            result,
            label=label,
            confidence=confidence,
            accepted=accepted,
            reason=reason,
            scores=aggregate,
        )

    def _discard_stale(self, now: float) -> None:
        cutoff = now - self.motion_window_seconds
        while self._observations and self._observations[0].timestamp < cutoff:
            self._observations.popleft()

    @staticmethod
    def _probabilities(result: dict[str, Any]) -> np.ndarray | None:
        values = result.get("probabilities")
        try:
            probabilities = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if probabilities.shape != (len(LABELS),) or not np.isfinite(probabilities).all():
            return None
        if np.any(probabilities < 0):
            return None
        total = float(probabilities.sum())
        if total <= np.finfo(np.float64).eps:
            return None
        return probabilities / total

    @staticmethod
    def _landmarks(result: dict[str, Any]) -> np.ndarray | None:
        values = result.get("landmarks")
        try:
            landmarks = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if landmarks.ndim != 2 or landmarks.shape[0] != 21 or landmarks.shape[1] < 2:
            return None
        points = landmarks[:, :2]
        if not np.isfinite(points).all():
            return None
        return points

    def _static_evidence(self) -> tuple[np.ndarray | None, int, int]:
        if not self._observations:
            return None, 0, 0
        latest = self._observations[-1].timestamp
        recent = [
            observation
            for observation in self._observations
            if observation.timestamp >= latest - self.static_window_seconds
        ]
        if not recent:
            return None, 0, 0
        adjusted = np.stack(
            [self._apply_static_geometry(item.probabilities, item.landmarks) for item in recent]
        )
        scores = adjusted.mean(axis=0)
        label_index = int(np.argmax(scores))
        consensus = int(np.count_nonzero(np.argmax(adjusted, axis=1) == label_index))
        return scores, len(recent), consensus

    @staticmethod
    def _apply_static_geometry(probabilities: np.ndarray, landmarks: np.ndarray | None) -> np.ndarray:
        scores = probabilities.copy()
        if landmarks is None:
            return scores

        # R and U heuristic
        u_index = LABEL_TO_INDEX["U"]
        r_index = LABEL_TO_INDEX["R"]
        ur_total = float(scores[u_index] + scores[r_index])
        if ur_total >= 0.25:
            lateral = landmarks[17] - landmarks[5]
            palm_width = float(np.linalg.norm(lateral))
            if palm_width > np.finfo(np.float64).eps:
                axis = lateral / palm_width
                base_order = float(np.dot(landmarks[9] - landmarks[5], axis))
                tip_order = float(np.dot(landmarks[12] - landmarks[8], axis))
                crossing = base_order * tip_order / (palm_width * palm_width)
                if crossing < -0.05:
                    strength = float(np.clip((-crossing - 0.05) / 0.1, 0.0, 1.0))
                    target_share = 0.68 + 0.22 * strength
                    r_score = max(float(scores[r_index]), ur_total * target_share)
                    scores[r_index] = r_score
                    scores[u_index] = ur_total - r_score
                else:
                    u_score = max(float(scores[u_index]), ur_total * 0.9)
                    scores[u_index] = u_score
                    scores[r_index] = ur_total - u_score

        # K and P heuristic
        k_index = LABEL_TO_INDEX["K"]
        p_index = LABEL_TO_INDEX["P"]
        kp_total = float(scores[k_index] + scores[p_index])
        if kp_total >= 0.25:
            # Check vertical orientation: y goes down in images
            index_y_diff = landmarks[8][1] - landmarks[5][1]
            if index_y_diff > 0:
                p_score = max(float(scores[p_index]), kp_total * 0.9)
                scores[p_index] = p_score
                scores[k_index] = kp_total - p_score
            else:
                k_score = max(float(scores[k_index]), kp_total * 0.9)
                scores[k_index] = k_score
                scores[p_index] = kp_total - k_score

        return scores

    def _dynamic_decision(self) -> tuple[str, float, float] | None:
        observations = [item for item in self._observations if item.landmarks is not None]
        if len(observations) < 6:
            return None
        if observations[-1].timestamp - observations[0].timestamp < 0.35:
            return None

        scale = self._motion_scale(observations)
        candidates: list[tuple[str, float, float]] = []
        for label, tip_index in DYNAMIC_TIPS.items():
            class_index = LABEL_TO_INDEX[label]
            support = max(float(item.probabilities[class_index]) for item in observations)
            if support < 0.18:
                continue
            trace = np.stack([item.landmarks[tip_index] for item in observations if item.landmarks is not None])
            if label == "Z":
                motion_score = self._z_motion_score(trace, scale)
            else:
                motion_score = self._j_motion_score(trace, scale)
            if motion_score < 0.72:
                continue
            confidence = min(0.99, 0.46 + 0.30 * support + 0.30 * motion_score)
            if confidence >= self.confidence_threshold:
                candidates.append((label, confidence, motion_score))
        return max(candidates, key=lambda item: item[1], default=None)

    @staticmethod
    def _motion_scale(observations: list[_Observation]) -> float:
        spans = [
            float(np.linalg.norm(item.landmarks[17] - item.landmarks[5]))
            for item in observations
            if item.landmarks is not None
        ]
        median_span = float(np.median(spans)) if spans else 0.1
        return float(np.clip(median_span * 0.55, 0.04, 0.14))

    @staticmethod
    def _resample_trace(trace: np.ndarray, samples: int = 4) -> np.ndarray | None:
        if trace.ndim != 2 or trace.shape[0] < 2 or trace.shape[1] != 2:
            return None
        segment_lengths = np.linalg.norm(np.diff(trace, axis=0), axis=1)
        total_length = float(segment_lengths.sum())
        if total_length <= np.finfo(np.float64).eps:
            return None
        cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        targets = np.linspace(0.0, total_length, samples)
        output = []
        for target in targets:
            index = min(int(np.searchsorted(cumulative, target, side="right")) - 1, len(trace) - 2)
            start, end = cumulative[index], cumulative[index + 1]
            fraction = 0.0 if end <= start else (target - start) / (end - start)
            output.append(trace[index] + (trace[index + 1] - trace[index]) * fraction)
        return np.asarray(output, dtype=np.float64)

    @classmethod
    def _z_motion_score(cls, trace: np.ndarray, scale: float) -> float:
        samples = cls._resample_trace(trace)
        if samples is None:
            return 0.0
        first, second, third, fourth = samples
        first_leg, diagonal, final_leg = second - first, third - second, fourth - third
        if min(abs(first_leg[0]), abs(final_leg[0])) < 0.7 * scale:
            return 0.0
        if abs(diagonal[0]) < 0.45 * scale or diagonal[1] < 0.3 * scale:
            return 0.0
        first_direction = np.sign(first_leg[0])
        if first_direction == 0 or np.sign(diagonal[0]) != -first_direction:
            return 0.0
        if np.sign(final_leg[0]) != first_direction:
            return 0.0
        if abs(first_leg[1]) > abs(first_leg[0]) * 1.2:
            return 0.0
        if abs(final_leg[1]) > abs(final_leg[0]) * 1.2:
            return 0.0
        return float(
            np.clip(
                min(
                    abs(first_leg[0]) / (0.7 * scale),
                    abs(final_leg[0]) / (0.7 * scale),
                    abs(diagonal[0]) / (0.45 * scale),
                    diagonal[1] / (0.3 * scale),
                ),
                0.0,
                1.0,
            )
        )

    @classmethod
    def _j_motion_score(cls, trace: np.ndarray, scale: float) -> float:
        samples = cls._resample_trace(trace)
        if samples is None:
            return 0.0
        first, second, third, fourth = samples
        first_leg, second_leg, hook = second - first, third - second, fourth - third
        downward = float(first_leg[1] + second_leg[1])
        sideways = abs(float(hook[0]))
        vertical_drift = abs(float(first_leg[0])) + abs(float(second_leg[0]))
        if downward < 2.0 * scale or sideways < 0.6 * scale:
            return 0.0
        if vertical_drift > downward * 0.75 or hook[1] > 0.8 * scale:
            return 0.0
        return float(np.clip(min(downward / (2.0 * scale), sideways / (0.6 * scale)), 0.0, 1.0))

    @staticmethod
    def _top5(scores: np.ndarray) -> list[dict[str, Any]]:
        indices = np.argsort(scores)[::-1][:5]
        return [
            {
                "label": LABELS[int(index)],
                "class_id": int(index),
                "confidence": round(float(scores[int(index)]), 6),
            }
            for index in indices
        ]

    @classmethod
    def _promote_dynamic_label(cls, scores: np.ndarray, label: str) -> np.ndarray:
        promoted = scores.copy()
        index = LABEL_TO_INDEX[label]
        promoted[index] = max(float(promoted[index]), float(promoted.max()) + 0.001)
        return promoted / promoted.sum()

    @classmethod
    def _with_prediction(
        cls,
        result: dict[str, Any],
        *,
        label: str,
        confidence: float,
        accepted: bool,
        reason: str,
        scores: np.ndarray,
    ) -> dict[str, Any]:
        output = dict(result)
        output.update(
            {
                "label": label,
                "sign": label,
                "class_id": LABEL_TO_INDEX[label],
                "confidence": round(float(confidence), 4),
                "accepted": bool(accepted),
                "reason": reason,
                "top5": cls._top5(scores),
            }
        )
        return output
