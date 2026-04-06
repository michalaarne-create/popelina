from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from replay_lib import build_counterfactual_candidates
from replay_lib import diff_decisions
from replay_lib import simulate_counterfactuals
from replay_lib import _load_json
from replay_lib import _load_qa_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare alternative plans against one saved screen_state.")
    parser.add_argument("--decision-json", required=True, help="Path to decision.json artifact.")
    parser.add_argument("--screen-state-json", help="Optional explicit screen_state.json path.")
    args = parser.parse_args()

    decision_path = Path(args.decision_json)
    screen_state_path = Path(args.screen_state_json) if args.screen_state_json else decision_path.with_name("screen_state.json")
    decision = _load_json(decision_path)
    screen_state = _load_json(screen_state_path)
    qa_cache = _load_qa_cache()
    candidates = build_counterfactual_candidates(screen_state, decision, qa_cache)
    report = simulate_counterfactuals(screen_state, decision.get("trace"), candidates)
    report["decision_path"] = str(decision_path)
    report["screen_state_path"] = str(screen_state_path)
    report["baseline"] = diff_decisions(decision, decision)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
