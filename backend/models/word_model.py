import configparser
import json
import math
import threading
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

try:
    from .. import config as app_config
    from ..services.model_assets import asset_metadata, resolve_asset
except ImportError:  # pragma: no cover - direct backend-path execution
    import config as app_config
    from services.model_assets import asset_metadata, resolve_asset

class GraphConvolution_att(nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_A=0):
        super(GraphConvolution_att, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.att = Parameter(torch.FloatTensor(55, 55))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(self.att, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GC_Block(nn.Module):
    def __init__(self, in_features, p_dropout, bias=True, is_resi=True):
        super(GC_Block, self).__init__()
        self.in_features = in_features
        self.out_features = in_features
        self.is_resi = is_resi

        self.gc1 = GraphConvolution_att(in_features, in_features)
        self.bn1 = nn.BatchNorm1d(55 * in_features)

        self.gc2 = GraphConvolution_att(in_features, in_features)
        self.bn2 = nn.BatchNorm1d(55 * in_features)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gc2(y)
        b, n, f = y.shape
        y = self.bn2(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)
        if self.is_resi:
            return y + x
        else:
            return y


class GCN_muti_att(nn.Module):
    def __init__(self, input_feature, hidden_feature, num_class, p_dropout, num_stage=1, is_resi=True):
        super(GCN_muti_att, self).__init__()
        self.num_stage = num_stage

        self.gc1 = GraphConvolution_att(input_feature, hidden_feature)
        self.bn1 = nn.BatchNorm1d(55 * hidden_feature)

        self.gcbs = nn.ModuleList([
            GC_Block(hidden_feature, p_dropout=p_dropout, is_resi=is_resi)
            for _ in range(num_stage)
        ])

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()
        self.fc_out = nn.Linear(hidden_feature, num_class)

    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        for i in range(self.num_stage):
            y = self.gcbs[i](y)

        out = torch.mean(y, dim=1)
        out = self.fc_out(out)
        return out


def validate_and_reshape_55(flat_landmarks: Sequence[float]) -> np.ndarray:
    """
    Takes a flat list of 110 floats (55 keypoints * x, y) from the frontend
    and reshapes it to (55, 2) for one frame.
    """
    if not hasattr(flat_landmarks, "__len__") or len(flat_landmarks) != 110:
        length = len(flat_landmarks) if hasattr(flat_landmarks, "__len__") else "unknown"
        raise ValueError(f"Expected 110 float values (55 keypoints), got {length}")

    arr = np.asarray(flat_landmarks, dtype=np.float32).reshape(55, 2)
    if not np.isfinite(arr).all():
        raise ValueError("Word landmarks contain NaN or infinity.")
    minimum, maximum = float(arr.min()), float(arr.max())
    if minimum < -1.00001 or maximum > 1.00001:
        raise ValueError(
            "Word landmarks must use the checkpoint's normalized [-1, 1] coordinate range; "
            f"observed [{minimum:.6g}, {maximum:.6g}]."
        )
    return arr


class WordModel:
    """Pinned WLASL-100 Pose-TGCN with strict input and label contracts."""

    REPO_ID = "sharonn18/tgcn-wlasl"
    REVISION = "dacb4568719caa03c44764034f599a9f8a0f63f4"
    NUM_CLASSES = 100

    def __init__(self, model_size: str = "asl100") -> None:
        if model_size != "asl100":
            raise ValueError("Only the pinned and validated asl100 model is supported.")
        self.repo_id = self.REPO_ID
        self.revision = self.REVISION
        self.model_size = model_size
        self.confidence_threshold = float(getattr(app_config, "WORD_MIN_CONFIDENCE", 0.70))
        self.margin_threshold = float(getattr(app_config, "WORD_MIN_MARGIN", 0.15))
        self.min_hand_frame_ratio = float(
            getattr(app_config, "WORD_MIN_HAND_FRAME_RATIO", 0.80)
        )
        self.min_motion_p90 = float(getattr(app_config, "WORD_MIN_MOTION_P90", 0.01))
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("WORD_MIN_CONFIDENCE must be in [0, 1].")
        if not 0.0 <= self.margin_threshold <= 1.0:
            raise ValueError("WORD_MIN_MARGIN must be in [0, 1].")
        if not 0.0 <= self.min_hand_frame_ratio <= 1.0:
            raise ValueError("WORD_MIN_HAND_FRAME_RATIO must be in [0, 1].")
        if not 0.0 <= self.min_motion_p90 <= 2.8285:
            raise ValueError("WORD_MIN_MOTION_P90 must be in [0, sqrt(8)].")

        self._lock = threading.RLock()
        self.checkpoint_path = resolve_asset("word_asl100_weights")
        self.config_path = resolve_asset("word_asl100_config")
        self.labels_path = resolve_asset("wlasl100_labels", allow_download=False)

        settings = self._read_config(self.config_path)
        self.num_samples = settings["num_samples"]
        self.hidden_feature = settings["hidden_feature"]
        self.num_stages = settings["num_stages"]
        self.dropout = settings["dropout"]
        if self.num_samples != 50:
            raise ValueError(f"Pinned WLASL model must use 50 samples; config declares {self.num_samples}.")

        self.vocab = self._read_labels(self.labels_path)
        self._validate_against_local_wlasl(self.vocab)
        self.model = GCN_muti_att(
            input_feature=self.num_samples * 2,
            hidden_feature=self.hidden_feature,
            num_class=self.NUM_CLASSES,
            p_dropout=self.dropout,
            num_stage=self.num_stages,
        )
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("state_dict", checkpoint)
        if not isinstance(state_dict, dict):
            raise ValueError("Pinned WLASL checkpoint does not contain a state dictionary.")
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        if self.model.fc_out.out_features != len(self.vocab):
            raise ValueError(
                f"Word output size {self.model.fc_out.out_features} does not match "
                f"label count {len(self.vocab)}."
            )
        self.self_test()

    @staticmethod
    def _read_config(path: Path) -> dict[str, Any]:
        parser = configparser.ConfigParser()
        try:
            with path.open("r", encoding="utf-8") as source:
                parser.read_file(source)
            return {
                "num_samples": parser.getint("TRAIN", "NUM_SAMPLES"),
                "dropout": parser.getfloat("TRAIN", "DROP_P"),
                "hidden_feature": parser.getint("GCN", "HIDDEN_SIZE"),
                "num_stages": parser.getint("GCN", "NUM_STAGES"),
            }
        except (OSError, configparser.Error, ValueError) as exc:
            raise ValueError(f"Invalid pinned WLASL config {path}: {exc}") from exc

    @classmethod
    def _read_labels(cls, path: Path) -> list[str]:
        try:
            labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        except OSError as exc:
            raise ValueError(f"Cannot read WLASL label file {path}: {exc}") from exc
        if len(labels) != cls.NUM_CLASSES:
            raise ValueError(f"Expected exactly 100 WLASL labels, got {len(labels)} in {path}.")
        if any(not label for label in labels) or len(set(labels)) != cls.NUM_CLASSES:
            raise ValueError("WLASL labels must be non-empty and unique.")
        if labels != sorted(labels):
            raise ValueError("WLASL labels must use the original alphabetically sorted LabelEncoder order.")
        return labels

    @staticmethod
    def _validate_against_local_wlasl(labels: list[str]) -> None:
        glossary = Path(app_config.PROJECT_ROOT) / "data" / "WLASL_v0.3.json"
        if not glossary.is_file():
            return
        try:
            entries = json.loads(glossary.read_text(encoding="utf-8"))
            expected = sorted(str(entry["gloss"]) for entry in entries[:100])
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"Cannot validate labels against local WLASL glossary {glossary}: {exc}") from exc
        if labels != expected:
            raise ValueError(
                "wlasl100.txt does not match sorted glosses from the first 100 local WLASL entries."
            )

    def _prepare_input(self, frames_landmarks: Sequence[Sequence[float]]) -> torch.Tensor:
        """
        frames_landmarks: List of 50 frames, each frame is a flat list of 110 floats.
        """
        if not hasattr(frames_landmarks, "__len__") or len(frames_landmarks) != self.num_samples:
            length = len(frames_landmarks) if hasattr(frames_landmarks, "__len__") else "unknown"
            raise ValueError(f"Expected {self.num_samples} frames, got {length}")
        reshaped_frames = [validate_and_reshape_55(frame) for frame in frames_landmarks]
        frames_array = np.stack(reshaped_frames, axis=0)
        input_data = np.ascontiguousarray(
            frames_array.transpose(1, 0, 2).reshape(55, self.num_samples * 2),
            dtype=np.float32,
        )
        if input_data.shape != (55, 100) or not np.isfinite(input_data).all():
            raise ValueError(f"Invalid prepared WLASL tensor shape/content: {input_data.shape}.")
        return torch.from_numpy(input_data).unsqueeze(0)

    def _infer(self, tensor: torch.Tensor) -> torch.Tensor:
        with self._lock, torch.inference_mode():
            logits = self.model(tensor)
            if logits.shape != (1, self.NUM_CLASSES) or not torch.isfinite(logits).all():
                raise ValueError(f"Word model returned invalid logits shaped {tuple(logits.shape)}.")
            probabilities = torch.softmax(logits[0], dim=0).cpu()
        if not torch.isfinite(probabilities).all() or not torch.isclose(
            probabilities.sum(), torch.tensor(1.0), atol=1e-5
        ):
            raise ValueError("Word model returned invalid probabilities.")
        return probabilities

    def _sequence_quality(self, frames_landmarks: Sequence[Sequence[float]]) -> dict[str, Any]:
        frames = np.stack(
            [validate_and_reshape_55(frame) for frame in frames_landmarks], axis=0
        )
        hands = frames[:, 13:, :]
        # OpenPose missing coordinates are encoded as (-1,-1). A point on one
        # image boundary may have a single -1 coordinate and remains valid.
        valid = ~np.all(np.isclose(hands, -1.0, atol=1e-6), axis=2)
        hand_frame_ratio = float(np.mean(valid.sum(axis=1) >= 15))

        transition_motion: list[float] = []
        for index in range(1, len(hands)):
            paired = valid[index - 1] & valid[index]
            if np.any(paired):
                displacement = np.linalg.norm(
                    hands[index, paired] - hands[index - 1, paired], axis=1
                )
                transition_motion.append(float(np.median(displacement)))
        motion_p90 = float(np.percentile(transition_motion, 90)) if transition_motion else 0.0

        if hand_frame_ratio < self.min_hand_frame_ratio:
            reason = "insufficient_hand_presence"
        elif motion_p90 < self.min_motion_p90:
            reason = "insufficient_motion"
        else:
            reason = "quality_passed"
        return {
            "passed": reason == "quality_passed",
            "reason": reason,
            "hand_frame_ratio": round(hand_frame_ratio, 4),
            "motion_p90": round(motion_p90, 6),
        }

    def predict_word(self, frames_landmarks: Sequence[Sequence[float]]) -> dict[str, Any]:
        started = time.perf_counter()
        # Validate sign activity before asking a closed-set softmax model to
        # classify idle/missing landmarks as one of its 100 known words.
        if not hasattr(frames_landmarks, "__len__") or len(frames_landmarks) != self.num_samples:
            length = len(frames_landmarks) if hasattr(frames_landmarks, "__len__") else "unknown"
            raise ValueError(f"Expected {self.num_samples} frames, got {length}")
        quality = self._sequence_quality(frames_landmarks)
        if not quality["passed"]:
            return {
                "label": None,
                "class_id": None,
                "confidence": 0.0,
                "margin": 0.0,
                "accepted": False,
                "reason": quality["reason"],
                "quality": quality,
                "top5": [],
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
                "model": "pose-tgcn-wlasl100",
            }
        probabilities = self._infer(self._prepare_input(frames_landmarks))
        values, indices = torch.topk(probabilities, k=5)
        top5 = [
            {
                "label": self.vocab[int(index)],
                "class_id": int(index),
                "confidence": float(value),
            }
            for value, index in zip(values.tolist(), indices.tolist())
        ]
        confidence = top5[0]["confidence"]
        margin = confidence - top5[1]["confidence"]
        accepted = confidence >= self.confidence_threshold and margin >= self.margin_threshold
        if confidence < self.confidence_threshold:
            reason = "low_confidence"
        elif margin < self.margin_threshold:
            reason = "low_margin"
        else:
            reason = "accepted"
        return {
            "label": top5[0]["label"],
            "class_id": top5[0]["class_id"],
            "confidence": confidence,
            "margin": margin,
            "accepted": accepted,
            "reason": reason,
            "quality": quality,
            "top5": top5,
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
            "model": "pose-tgcn-wlasl100",
        }

    def self_test(self) -> dict[str, Any]:
        # Deterministic, in-range signal checks the complete tensor/model/output
        # contract without pretending to be a meaningful sign prediction.
        frame = np.linspace(-0.25, 0.25, 110, dtype=np.float32)
        tensor = self._prepare_input([frame.tolist() for _ in range(self.num_samples)])
        probabilities = self._infer(tensor)
        if probabilities.numel() != len(self.vocab):
            raise ValueError("Word self-test output/label cardinality mismatch.")
        return {"status": "ok", "input_shape": list(tensor.shape), "classes": len(self.vocab)}

    def metadata(self) -> dict[str, Any]:
        return {
            "name": "pose-tgcn-wlasl100",
            "repo_id": self.repo_id,
            "revision": self.revision,
            "classes": len(self.vocab),
            "labels": list(self.vocab),
            "num_samples": self.num_samples,
            "input_shape": [1, 55, self.num_samples * 2],
            "input_range": [-1.0, 1.0],
            "preprocessing": "openpose_body25_filtered_plus_hands_xy_v1",
            "confidence_threshold": self.confidence_threshold,
            "margin_threshold": self.margin_threshold,
            "minimum_hand_frame_ratio": self.min_hand_frame_ratio,
            "minimum_motion_p90": self.min_motion_p90,
            "threshold_status": "provisional_unvalidated",
            "weights": asset_metadata("word_asl100_weights", self.checkpoint_path),
            "config": asset_metadata("word_asl100_config", self.config_path),
            "label_asset": asset_metadata("wlasl100_labels", self.labels_path),
        }
