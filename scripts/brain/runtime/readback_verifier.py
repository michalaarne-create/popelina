from __future__ import annotations

from typing import Any, Dict, List, Optional

from .quiz_utils import normalize_match_text, text_similarity


def _selected_values(controls_data: Optional[Dict[str, Any]]) -> List[str]:
    values: List[str] = []
    if not isinstance(controls_data, dict):
        return values
    for control in controls_data.get("controls") or []:
        if not isinstance(control, dict):
            continue
        kind = str(control.get("kind") or "")
        text = str(control.get("text") or control.get("value") or "")
        if kind in {"radio", "checkbox", "option"} and bool(control.get("checked") or control.get("selected")):
            if text:
                values.append(text)
        if kind in {"select", "textbox"}:
            value = str(control.get("value") or "")
            if value:
                values.append(value)
    return values


def _values_match(expected: List[str], actual: List[str]) -> bool:
    norm_actual = [normalize_match_text(value) for value in actual if normalize_match_text(value)]
    for expected_value in expected:
        target = normalize_match_text(expected_value)
        if not target:
            continue
        if any(text_similarity(target, candidate) >= 0.88 for candidate in norm_actual):
            return True
    return False


def _qa_text_from_prev_state(prev_state: Dict[str, Any]) -> str:
    if not isinstance(prev_state, dict):
        return ""
    parts: List[str] = []
    q = str(prev_state.get("question_text") or "").strip()
    if q:
        parts.append(q)
    objs = prev_state.get("objects") if isinstance(prev_state.get("objects"), dict) else {}
    answers = objs.get("answers") if isinstance(objs.get("answers"), list) else []
    for row in answers:
        if not isinstance(row, dict):
            continue
        t = str(row.get("text") or "").strip()
        if t:
            parts.append(t)
    return "\n".join(parts).strip()


def _qa_text_from_screen_state(screen_state: Dict[str, Any]) -> str:
    if not isinstance(screen_state, dict):
        return ""
    parts: List[str] = []
    q = str(screen_state.get("question_text") or "").strip()
    if q:
        parts.append(q)
    options = screen_state.get("options") if isinstance(screen_state.get("options"), list) else []
    for row in options:
        if not isinstance(row, dict):
            continue
        t = str(row.get("text") or "").strip()
        if t:
            parts.append(t)
    return "\n".join(parts).strip()


def _qa_change_ratio(prev_text: str, cur_text: str) -> float:
    a = normalize_match_text(prev_text or "")
    b = normalize_match_text(cur_text or "")
    if (not a) and (not b):
        return 0.0
    if not a or not b:
        return 1.0
    sim = text_similarity(a, b)
    try:
        return max(0.0, min(1.0, 1.0 - float(sim)))
    except Exception:
        return 0.0


def evaluate_transition(
    *,
    prev_state: Dict[str, Any],
    current_screen_state: Dict[str, Any],
    controls_data: Optional[Dict[str, Any]] = None,
    page_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    prev_action = (prev_state.get("last_action") or {}) if isinstance(prev_state, dict) else {}
    prev_kind = str(prev_action.get("kind") or "")
    current_question_sig = str(current_screen_state.get("active_question_signature") or "")
    prev_question_sig = str(prev_action.get("question_signature") or "")
    current_page_sig = str((page_data or {}).get("page_signature") or current_screen_state.get("page_signature") or "")
    prev_page_sig = str(prev_action.get("page_signature") or "")
    current_values = _selected_values(controls_data)
    expected_values = [str(v) for v in prev_action.get("expected_values") or [] if str(v)]
    values_match = _values_match(expected_values, current_values) if expected_values else False
    question_changed = bool(prev_question_sig and current_question_sig and current_question_sig != prev_question_sig)
    page_changed = bool(prev_page_sig and current_page_sig and current_page_sig != prev_page_sig)
    same_screen = str(prev_state.get("last_screen_signature") or "") == str(current_screen_state.get("screen_signature") or "")
    prev_qa_text = _qa_text_from_prev_state(prev_state)
    cur_qa_text = _qa_text_from_screen_state(current_screen_state)
    qa_change_ratio = _qa_change_ratio(prev_qa_text, cur_qa_text)
    qa_changed_30 = bool(qa_change_ratio >= 0.30)

    success = False
    if prev_kind == "answer":
        success = values_match or question_changed or page_changed or qa_changed_30
    elif prev_kind == "next":
        success = question_changed or page_changed
    elif prev_kind in {"type", "dropdown"}:
        success = values_match or question_changed or page_changed

    failure = False
    if prev_kind in {"answer", "next", "type", "dropdown"} and not success:
        failure = bool(prev_question_sig and current_question_sig == prev_question_sig and same_screen)

    return {
        "previous_action_kind": prev_kind,
        "expected_values": expected_values,
        "current_values": current_values,
        "values_match": values_match,
        "question_changed": question_changed,
        "page_changed": page_changed,
        "qa_change_ratio": round(float(qa_change_ratio), 4),
        "qa_changed_30": qa_changed_30,
        "same_screen": same_screen,
        "success": success,
        "failure": failure,
    }
