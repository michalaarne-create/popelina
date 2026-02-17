from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from ...runtime.pipeline_state_builder import build_brain_state

ROOT = Path(__file__).resolve().parents[1]
DATA_SCREEN_DIR = ROOT / "data" / "screen"
RATE_SUMMARY_DIR = DATA_SCREEN_DIR / "rate" / "rate_summary"
DOM_LIVE_DIR = ROOT / "scripts" / "dom" / "dom_live"
DEFAULT_BRAIN_STATE = DATA_SCREEN_DIR / "brain_state.json"


def load_json(path: Optional[Path]) -> Optional[dict]:
    if path is None:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def latest_summary_file() -> Optional[Path]:
    if not RATE_SUMMARY_DIR.is_dir():
        return None
    candidates = sorted(RATE_SUMMARY_DIR.glob("*_summary.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Pipeline brain aggregator.")
    ap.add_argument("--question-json", type=str, default=str(DOM_LIVE_DIR / "current_question.json"))
    ap.add_argument("--summary-json", type=str, default="")
    ap.add_argument("--state-json", type=str, default=str(DEFAULT_BRAIN_STATE))
    ap.add_argument("--mark-answer-clicked", action="store_true")
    ap.add_argument("--mark-next-clicked", action="store_true")
    args = ap.parse_args()

    question_path = Path(args.question_json)
    summary_path = Path(args.summary_json) if args.summary_json else latest_summary_file()
    state_path = Path(args.state_json)
    question_data = load_json(question_path)
    if question_data is not None:
        question_data["_source"] = str(question_path)
    summary_data = load_json(summary_path)
    if summary_data is not None:
        summary_data["_source"] = str(summary_path) if summary_path else None
    prev_state = load_json(state_path) or {}

    brain = build_brain_state(
        question_data,
        summary_data,
        prev_state,
        mark_answer_clicked=args.mark_answer_clicked,
        mark_next_clicked=args.mark_next_clicked,
    )
    ensure_dir(state_path.parent)
    state_path.write_text(json.dumps(brain, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[brain] Updated brain state -> {state_path}")


if __name__ == "__main__":
    main()


