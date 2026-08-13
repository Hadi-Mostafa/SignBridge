"""Build deterministic landmark-guide SVGs from the bundled alphabet CSV.

The output is deliberately a schematic MediaPipe landmark guide, not a hand
photograph and not a newly invented ASL pose.  Each static class guide uses the
geometric medoid of the exact 42-feature rows used by the bundled classifier.
Dynamic J and Z are omitted because a static point cloud cannot represent their
required motion trajectory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = PROJECT_ROOT / "backend" / "checkpoints" / "asl_keypoints.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "frontend" / "assets" / "fingerspell_guides"
EXPECTED_FEATURES = 42
EXPECTED_CLASSES = tuple(range(26))
DYNAMIC_CLASSES = frozenset({9, 25})  # J and Z require motion.
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path) -> dict[int, list[list[float]]]:
    grouped: dict[int, list[list[float]]] = {label: [] for label in EXPECTED_CLASSES}
    malformed_rows: list[int] = []
    with path.open(newline="", encoding="utf-8-sig") as source:
        for row_number, row in enumerate(csv.reader(source), 1):
            if len(row) != EXPECTED_FEATURES + 1:
                # The historical bundled CSV contains one concatenated line.
                # It cannot be reconstructed unambiguously, so exclude it and
                # record the exclusion in the generated manifest.
                malformed_rows.append(row_number)
                continue
            label = int(row[0].strip())
            values = [float(item.strip()) for item in row[1:]]
            if label not in grouped or len(values) != EXPECTED_FEATURES:
                raise ValueError(f"Row {row_number} violates the 26-class/42-feature contract.")
            if not all(math.isfinite(item) and abs(item) <= 1.00001 for item in values):
                raise ValueError(f"Row {row_number} contains invalid normalized landmarks.")
            grouped[label].append(values)
    missing = [label for label, rows in grouped.items() if not rows]
    if missing:
        raise ValueError(f"CSV is missing alphabet classes: {missing}.")
    load_rows.malformed_rows = malformed_rows
    return grouped


load_rows.malformed_rows = []


def geometric_medoid(rows: list[list[float]]) -> list[float]:
    """Return the observed row nearest the coordinate-wise mean."""

    centroid = [sum(row[index] for row in rows) / len(rows) for index in range(EXPECTED_FEATURES)]
    return min(rows, key=lambda row: sum((value - centroid[index]) ** 2 for index, value in enumerate(row)))


def render_svg(letter: str, features: list[float], sample_count: int) -> str:
    points = [(features[index], features[index + 1]) for index in range(0, EXPECTED_FEATURES, 2)]
    xs, ys = [item[0] for item in points], [item[1] for item in points]
    span_x = max(max(xs) - min(xs), 0.1)
    span_y = max(max(ys) - min(ys), 0.1)
    scale = min(150.0 / span_x, 150.0 / span_y)
    center_x, center_y = (max(xs) + min(xs)) / 2.0, (max(ys) + min(ys)) / 2.0
    mapped = [(100.0 + (x - center_x) * scale, 99.0 + (y - center_y) * scale) for x, y in points]
    lines = "\n".join(
        f'    <line x1="{mapped[a][0]:.2f}" y1="{mapped[a][1]:.2f}" '
        f'x2="{mapped[b][0]:.2f}" y2="{mapped[b][1]:.2f}" />'
        for a, b in HAND_CONNECTIONS
    )
    circles = "\n".join(
        f'    <circle cx="{x:.2f}" cy="{y:.2f}" r="3.1" />' for x, y in mapped
    )
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 232" role="img" aria-labelledby="title desc">
  <title id="title">{letter} fingerspelling landmark guide</title>
  <desc id="desc">Schematic landmark guide derived from an observed class medoid; not a hand photograph.</desc>
  <rect width="200" height="232" rx="18" fill="#071923" />
  <g stroke="#39d9c6" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" opacity="0.92">
{lines}
  </g>
  <g fill="#d8fff9" stroke="#071923" stroke-width="1.4">
{circles}
  </g>
  <text x="100" y="207" text-anchor="middle" fill="#ffffff" font-family="system-ui, sans-serif" font-size="30" font-weight="750">{letter}</text>
  <text x="100" y="224" text-anchor="middle" fill="#87a9b5" font-family="system-ui, sans-serif" font-size="8">LANDMARK GUIDE · MEDOID OF {sample_count} SAMPLES</text>
</svg>
'''


def build(csv_path: Path, output_dir: Path) -> dict:
    grouped = load_rows(csv_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    assets: list[dict] = []
    for label in EXPECTED_CLASSES:
        if label in DYNAMIC_CLASSES:
            continue
        letter = chr(ord("A") + label)
        destination = output_dir / f"{letter.lower()}.svg"
        destination.write_text(
            render_svg(letter, geometric_medoid(grouped[label]), len(grouped[label])),
            encoding="utf-8",
            newline="\n",
        )
        assets.append(
            {
                "letter": letter,
                "relative_path": destination.name,
                "mime_type": "image/svg+xml",
                "sha256": sha256(destination),
                "sample_count": len(grouped[label]),
            }
        )
    manifest = {
        "schema_version": 1,
        "representation": "mediapipe_landmark_guide",
        "not_native_media": True,
        "dynamic_letters_omitted": ["J", "Z"],
        "source_csv": str(csv_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "source_csv_sha256": sha256(csv_path),
        "source_provenance": "bundled classifier training CSV; authoritative origin/license not declared",
        "excluded_malformed_source_rows": list(load_rows.malformed_rows),
        "generation": "observed row nearest coordinate-wise class mean",
        "assets": assets,
    }
    manifest_path = output_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build static ASL landmark-guide SVGs.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = build(args.csv.resolve(), args.output.resolve())
    print(f"Built {len(manifest['assets'])} static guides in {args.output.resolve()}.")
    print("J and Z were intentionally omitted because they require motion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
