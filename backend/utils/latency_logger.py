"""Thread-safe JSONL latency logging with non-duplicated totals."""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

try:
    from ..config import LATENCY_LOG_FILE
except ImportError:  # pragma: no cover - direct backend-path execution
    from config import LATENCY_LOG_FILE


class LatencyLogger:
    _locks_guard = threading.Lock()
    _file_locks: dict[str, threading.RLock] = {}
    _metadata_allowlist = {
        "accepted",
        "clip_count",
        "confidence",
        "label",
        "model_version",
        "token_count",
    }

    def __init__(self, log_file: Optional[Path] = None):
        self.log_file = Path(log_file or LATENCY_LOG_FILE)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        key = str(self.log_file.resolve())
        with self._locks_guard:
            self._file_lock = self._file_locks.setdefault(key, threading.RLock())
        self._request_state = threading.local()

    @staticmethod
    def _total_from(timings: Dict[str, float]) -> float:
        if "total_ms" in timings:
            return float(timings["total_ms"])
        if "total_wall_ms" in timings:
            return float(timings["total_wall_ms"])
        return float(
            sum(
                float(value)
                for name, value in timings.items()
                if name.endswith("_ms") and name not in {"total_ms", "total_wall_ms"}
            )
        )

    def log(self, pipeline: str, timings: Dict[str, float], metadata: Optional[Dict] = None):
        clean_timings = {str(key): round(float(value), 3) for key, value in timings.items()}
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pipeline": str(pipeline),
            "timings": clean_timings,
            "total_ms": round(self._total_from(clean_timings), 3),
        }
        if metadata:
            safe_metadata = {
                str(key): value for key, value in metadata.items() if key in self._metadata_allowlist
            }
            if safe_metadata:
                entry["metadata"] = safe_metadata

        with self._file_lock:
            with self.log_file.open("a", encoding="utf-8") as file:
                file.write(json.dumps(entry, ensure_ascii=True, separators=(",", ":")) + "\n")
        return entry

    def start_request(self):
        self._request_state.started_at = time.perf_counter()
        self._request_state.timings = {}

    def record_stage(self, stage_name: str, duration_ms: float):
        timings = getattr(self._request_state, "timings", None)
        if timings is None:
            timings = {}
            self._request_state.timings = timings
        timings[str(stage_name)] = float(duration_ms)

    def end_request(self, pipeline: str, metadata: Optional[Dict] = None):
        timings = dict(getattr(self._request_state, "timings", {}))
        started_at = getattr(self._request_state, "started_at", None)
        if started_at is not None:
            timings["total_ms"] = (time.perf_counter() - started_at) * 1000
        entry = self.log(pipeline, timings, metadata)
        self._request_state.started_at = None
        self._request_state.timings = {}
        return entry

    def timer(self, pipeline: str, stage_name: str):
        return _TimerContext(self, pipeline, stage_name)

    def get_statistics(self, last_n: int = 100) -> Dict:
        import numpy as np

        last_n = max(1, min(int(last_n), 10_000))
        entries = []
        with self._file_lock:
            if self.log_file.exists():
                with self.log_file.open("r", encoding="utf-8", errors="replace") as file:
                    for line in file:
                        try:
                            entry = json.loads(line)
                            if isinstance(entry, dict) and "pipeline" in entry and "total_ms" in entry:
                                entries.append(entry)
                        except (json.JSONDecodeError, TypeError, ValueError):
                            continue
        entries = entries[-last_n:]
        if not entries:
            return {"message": "No latency data available"}

        stats = {}
        for pipeline in sorted({str(entry["pipeline"]) for entry in entries}):
            totals = [
                float(entry["total_ms"])
                for entry in entries
                if str(entry["pipeline"]) == pipeline
            ]
            stats[pipeline] = {
                "count": len(totals),
                "mean_ms": float(np.mean(totals)),
                "median_ms": float(np.median(totals)),
                "p95_ms": float(np.percentile(totals, 95)),
                "p99_ms": float(np.percentile(totals, 99)),
                "min_ms": float(np.min(totals)),
                "max_ms": float(np.max(totals)),
            }
        return stats


class _TimerContext:
    def __init__(self, logger: LatencyLogger, pipeline: str, stage_name: str):
        self.logger = logger
        self.pipeline = pipeline
        self.stage_name = stage_name
        self.start_time = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *_args):
        self.logger.record_stage(self.stage_name, (time.perf_counter() - self.start_time) * 1000)
