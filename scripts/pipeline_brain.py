#!/usr/bin/env python3
"""
Pipeline brain: łączy dane z current_question.json oraz rate_summary,
pilnuje stanu cyklu (kliknięto odpowiedź / kliknięto next) i zapisuje
ładny JSON z podziałem co jest czym.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
DATA_SCREEN_DIR = ROOT / "data" / "screen"
RATE_SUMMARY_DIR = DATA_SCREEN_DIR / "rate" / "rate_summary"
DOM_LIVE_DIR = ROOT / "dom_live"
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


def question_hash(text: Optional[str]) -> str:
    base = (text or "").strip().encode("utf-8", errors="ignore")
    return hashlib.sha1(base).hexdigest()


def build_brain_state(
    question_data: Optional[dict],
    summary_data: Optional[dict],
    prev_state: dict,
    mark_answer_clicked: bool,
    mark_next_clicked: bool,
) -> dict:
    question_text = ""
    main_question = None
    if question_data:
        question_text = question_data.get("main_question", {}).get("text") or question_data.get("full_text") or ""
        main_question = question_data.get("main_question")

    q_hash = question_hash(question_text)
    changed = q_hash != prev_state.get("question_hash")

    answer_clicked = prev_state.get("answer_clicked", False)
    next_clicked = prev_state.get("next_clicked", False)
    if changed:
        answer_clicked = False
        next_clicked = False

    if mark_answer_clicked:
        answer_clicked = True
    if mark_next_clicked:
        next_clicked = True

    top_labels = (summary_data or {}).get("top_labels") or {}
    background_layout = (summary_data or {}).get("background_layout") or {}
    answers = []
    for key in ("answer_single", "answer_multi"):
        if key in top_labels:
            answers.append(top_labels[key])
    next_candidate = None
    for key in ("next_active", "next_inactive"):
        if key in top_labels:
            next_candidate = top_labels[key]
            break

    # Preferuj wyłącznie przycisk akceptacji cookies – nie klikamy „reject”.
    cookies_candidate = top_labels.get("cookie_accept")

    has_answers = bool(answers)
    has_next = next_candidate is not None
    need_scroll = not has_answers and not has_next

    # Priorytet: najpierw cookies -> potem odpowiedź -> Next -> scroll.
    if cookies_candidate is not None:
        recommended = "click_cookies_accept"
    elif has_answers and not answer_clicked:
        recommended = "click_answer"
    elif has_next and answer_clicked and not next_clicked:
        recommended = "click_next"
    elif need_scroll:
        # Brak odpowiedzi i przycisku Next w aktualnym summary:
        # spróbuj przewinąć stronę niżej, żeby odsłonić więcej UI.
        recommended = "scroll_page_down"
    else:
        recommended = "idle"

    # Szkic podpowiedzi czytania (layout tła -> priorytet regionów)
    reading_hints: Dict[str, Any] = {}
    if background_layout:
        reading_hints["layout"] = background_layout
        label_bg: Dict[str, Any] = {}
        for name, obj in top_labels.items():
            if not isinstance(obj, dict):
                continue
            info: Dict[str, Any] = {}
            if "bg_cluster_id" in obj:
                info["bg_cluster_id"] = obj.get("bg_cluster_id")
            if "bg_is_main_like" in obj:
                info["bg_is_main_like"] = bool(obj.get("bg_is_main_like"))
            if "bg_mean_rgb" in obj:
                info["bg_mean_rgb"] = obj.get("bg_mean_rgb")
            if "bg_dist_to_global" in obj:
                info["bg_dist_to_global"] = obj.get("bg_dist_to_global")
            if info:
                label_bg[name] = info
        if label_bg:
            reading_hints["labels_by_bg"] = label_bg

        main_cluster_id = background_layout.get("main_cluster_id")
        if main_cluster_id is not None:
            reading_hints["primary_bg_cluster_id"] = main_cluster_id
            preferred_labels = [
                name
                for name, info in label_bg.items()
                if info.get("bg_cluster_id") == main_cluster_id
            ]
            reading_hints["preferred_labels_main_bg"] = preferred_labels

    brain = {
        "timestamp": time.time(),
        "question_text": question_text,
        "question_hash": q_hash,
        "question_changed": bool(changed),
        "answer_clicked": bool(answer_clicked),
        "next_clicked": bool(next_clicked),
        "has_answers": has_answers,
        "has_next": has_next,
        "recommended_action": recommended,
        "sources": {
            "question_json": str(question_data.get("_source")) if question_data else None,
            "summary_json": str(summary_data.get("_source")) if summary_data else None,
        },
        "objects": {
            "question": main_question,
            "answers": answers,
            "next": next_candidate,
            "cookies": cookies_candidate,
        },
        "background_layout": background_layout,
        "reading_hints": reading_hints,
    }
    return brain


def main() -> None:
    ap = argparse.ArgumentParser(description="Pipeline brain aggregator.")
    ap.add_argument("--question-json", type=str, default=str(DOM_LIVE_DIR / "current_question.json"))
    ap.add_argument("--summary-json", type=str, default="")
    ap.add_argument("--state-json", type=str, default=str(DEFAULT_BRAIN_STATE))
    ap.add_argument("--mark-answer-clicked", action="store_true", help="Oznacz, że odpowiedź została kliknięta.")
    ap.add_argument("--mark-next-clicked", action="store_true", help="Oznacz, że przycisk Next został kliknięty.")
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

    if state_path.exists():
        try:
            prev_state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            prev_state = {}
    else:
        prev_state = {}

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


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
