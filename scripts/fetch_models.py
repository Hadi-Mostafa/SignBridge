"""Fetch and verify every model required by SignBridge.

Usage from the project root::

    python scripts/fetch_models.py
    python scripts/fetch_models.py word_asl100_weights word_asl100_config
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.model_assets import (  # noqa: E402
    ModelAssetError,
    asset_metadata,
    list_assets,
    resolve_asset,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch checksum-verified SignBridge model assets.")
    parser.add_argument("assets", nargs="*", help="Asset names (default: all locked assets).")
    args = parser.parse_args()
    names = args.assets or list_assets()
    unknown = sorted(set(names) - set(list_assets()))
    if unknown:
        parser.error(f"unknown assets: {', '.join(unknown)}")

    resolved = []
    failures = []
    for name in names:
        try:
            path = resolve_asset(name, allow_download=True)
            resolved.append(asset_metadata(name, path))
        except ModelAssetError as exc:
            failures.append({"asset": name, "error": str(exc)})

    print(json.dumps({"resolved": resolved, "failures": failures}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
