from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, Optional


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
    answer_clicked = False if changed else prev_state.get("answer_clicked", False)
    next_clicked = False if changed else prev_state.get("next_clicked", False)
    if mark_answer_clicked:
        answer_clicked = True
    if mark_next_clicked:
        next_clicked = True

    top_labels = (summary_data or {}).get("top_labels") or {}
    background_layout = (summary_data or {}).get("background_layout") or {}
    answers = [top_labels[k] for k in ("answer_single", "answer_multi") if k in top_labels]

    next_candidate = None
    for key in ("next_active", "next_inactive"):
        if key in top_labels:
            next_candidate = top_labels[key]
            break

    cookies_candidate = top_labels.get("cookie_accept")
    has_answers = bool(answers)
    has_next = next_candidate is not None
    need_scroll = not has_answers and not has_next

    if cookies_candidate is not None:
        recommended = "click_cookies_accept"
    elif has_answers and not answer_clicked:
        recommended = "click_answer"
    elif has_next and answer_clicked and not next_clicked:
        recommended = "click_next"
    elif need_scroll:
        recommended = "scroll_page_down"
    else:
        recommended = "idle"

    reading_hints: Dict[str, Any] = {}
    if background_layout:
        reading_hints["layout"] = background_layout
        label_bg: Dict[str, Any] = {}
        for name, obj in top_labels.items():
            if not isinstance(obj, dict):
                continue
            info: Dict[str, Any] = {}
            for key in ("bg_cluster_id", "bg_mean_rgb", "bg_dist_to_global"):
                if key in obj:
                    info[key] = obj.get(key)
            if "bg_is_main_like" in obj:
                info["bg_is_main_like"] = bool(obj.get("bg_is_main_like"))
            if info:
                label_bg[name] = info
        if label_bg:
            reading_hints["labels_by_bg"] = label_bg
        main_cluster_id = background_layout.get("main_cluster_id")
        if main_cluster_id is not None:
            reading_hints["primary_bg_cluster_id"] = main_cluster_id
            reading_hints["preferred_labels_main_bg"] = [
                name for name, info in label_bg.items() if info.get("bg_cluster_id") == main_cluster_id
            ]

    return {
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
        "objects": {"question": main_question, "answers": answers, "next": next_candidate, "cookies": cookies_candidate},
        "background_layout": background_layout,
        "reading_hints": reading_hints,
    }

