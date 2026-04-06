from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from replay_lib import _load_json
from replay_lib import diff_decisions


def main() -> None:
    parser = argparse.ArgumentParser(description="Show high-signal diff between two decision artifacts.")
    parser.add_argument("--left", required=True, help="Path to first decision.json.")
    parser.add_argument("--right", required=True, help="Path to second decision.json.")
    args = parser.parse_args()
    report = diff_decisions(_load_json(Path(args.left)), _load_json(Path(args.right)))
    report["left"] = args.left
    report["right"] = args.right
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
