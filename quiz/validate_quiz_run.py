from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

try:
    from .test_quiz_server import LOG_FILE, build_bank
except Exception:
    from test_quiz_server import LOG_FILE, build_bank


def _norm(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _load_events(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    events: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def _filter_questions(type_ids: Optional[Sequence[int]]) -> List[Any]:
    bank = build_bank()
    if not type_ids:
        return [question for quiz_type in bank for question in quiz_type.questions]
    allowed: Set[int] = set()
    for item in type_ids:
        try:
            idx = int(item)
        except Exception:
            continue
        if 1 <= idx <= len(bank):
            allowed.add(idx)
    return [question for quiz_type in bank if quiz_type.idx in allowed for question in quiz_type.questions]


def validate_log(log_path: Path, *, type_ids: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    questions = _filter_questions(type_ids)
    events = _load_events(log_path)
    by_qid: Dict[str, List[Dict[str, Any]]] = {}
    for event in events:
        qid = str(event.get("qid") or "").strip()
        if not qid:
            continue
        by_qid.setdefault(qid, []).append(event)

    failures: List[Dict[str, Any]] = []
    summary_by_type: Dict[str, Dict[str, int]] = {}
    for question in questions:
        summary_by_type.setdefault(question.qtype, {"total": 0, "passed": 0, "failed": 0})
        summary_by_type[question.qtype]["total"] += 1
        q_events = by_qid.get(question.key, [])
        correct_seen = False
        next_before_correct = False
        for event in q_events:
            selected = event.get("selected") or []
            if not isinstance(selected, list):
                selected = [selected]
            normalized = [_norm(value) for value in selected if _norm(value)]
            if "next_click" in normalized:
                if not correct_seen:
                    next_before_correct = True
                continue
            expected = {_norm(value) for value in question.correct if _norm(value)}
            got = {value for value in normalized if value != "next_click"}
            if question.qtype == "text":
                correct_seen = any(value in expected for value in got)
            elif question.qtype == "multi":
                correct_seen = bool(expected) and expected.issubset(got)
            else:
                correct_seen = got == expected
            if correct_seen:
                break
        if correct_seen and not next_before_correct:
            summary_by_type[question.qtype]["passed"] += 1
            continue
        summary_by_type[question.qtype]["failed"] += 1
        failures.append(
            {
                "qid": question.key,
                "qtype": question.qtype,
                "prompt": question.prompt,
                "reason": "next_before_correct" if next_before_correct else "missing_correct_answer",
                "events": q_events,
            }
        )

    return {
        "log_path": str(log_path),
        "total_questions": len(questions),
        "failures": failures,
        "summary_by_type": summary_by_type,
        "passed": not failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate quiz_submissions.log against test_quiz_server BANK.")
    parser.add_argument("--log", type=str, default=str(LOG_FILE), help="Path to quiz_submissions.log")
    parser.add_argument("--type-id", type=int, nargs="*", default=None, help="Only validate selected quiz type ids (1..13).")
    args = parser.parse_args()
    result = validate_log(Path(args.log), type_ids=args.type_id)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result.get("passed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
