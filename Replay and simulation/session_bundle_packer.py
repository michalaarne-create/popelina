from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from replay_lib import ROOT
from replay_lib import build_session_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack one saved session into a portable replay bundle.")
    parser.add_argument("--run-dir", required=True, help="Path to auto_main run dir.")
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "Replay and simulation" / "bundles"),
        help="Target directory for generated bundle.",
    )
    args = parser.parse_args()
    report = build_session_bundle(args.run_dir, Path(args.output_root))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
