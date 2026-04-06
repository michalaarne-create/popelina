from __future__ import annotations

from typing import Any, Dict, List, Optional

from scripts.pipeline.contracts import SCHEMA_VERSION


def build_back_navigation_confirmation_state(
    *,
    previous_screen_state: Optional[Dict[str, Any]] = None,
    current_screen_state: Optional[Dict[str, Any]] = None,
    current_page_data: Optional[Dict[str, Any]] = None,
    previous_url: str = "",
    current_url: str = "",
    action_kind: str = "",
    action_reason: str = "",
) -> Dict[str, Any]:
    previous_state = previous_screen_state or {}
    current_state = current_screen_state or {}
    previous_question = str(previous_state.get("question_text") or "").strip()
    current_question = str(current_state.get("question_text") or "").strip()
    previous_signature = str(previous_state.get("active_question_signature") or previous_state.get("screen_signature") or "").strip()
    current_signature = str(current_state.get("active_question_signature") or current_state.get("screen_signature") or "").strip()
    url_changed = bool(previous_url and current_url and previous_url.rstrip("/") != current_url.rstrip("/"))
    question_changed = bool(previous_question and current_question and previous_question != current_question)
    signature_changed = bool(previous_signature and current_signature and previous_signature != current_signature)
    kind = str(action_kind or "").strip().lower()
    reason = str(action_reason or "").strip().lower()
    action_is_back = bool(kind == "navigate_back" or reason == "click_back")
    readback_values: List[str] = []
    if isinstance(current_page_data, dict):
        for row in current_page_data.get("options") or []:
            if isinstance(row, dict) and bool(row.get("checked")):
                text = str(row.get("text") or "").strip()
                if text:
                    readback_values.append(text)
        for row in current_page_data.get("selectOptions") or []:
            if isinstance(row, dict) and bool(row.get("selected")):
                text = str(row.get("text") or row.get("value") or "").strip()
                if text:
                    readback_values.append(text)
        textbox = str(current_page_data.get("textboxValue") or "").strip()
        if textbox:
            readback_values.append(textbox)
    deduped_values: List[str] = []
    seen = set()
    for value in readback_values:
        key = value.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped_values.append(value)
    readback_preserved = bool(deduped_values)
    is_confirmed = bool(action_is_back and current_question and (question_changed or signature_changed or url_changed))
    return {
        "schema_version": SCHEMA_VERSION,
        "is_confirmed": is_confirmed,
        "signals": {
            "action_is_back": bool(action_is_back),
            "url_changed": bool(url_changed),
            "question_changed": bool(question_changed),
            "signature_changed": bool(signature_changed),
            "current_question_present": bool(current_question),
            "readback_preserved": bool(readback_preserved),
            "readback_values": deduped_values,
        },
    }
