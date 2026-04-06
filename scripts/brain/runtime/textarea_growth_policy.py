from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from scripts.pipeline.contracts import SCHEMA_VERSION


def _bbox4(value: Any) -> Optional[list[float]]:
    if not isinstance(value, Sequence) or len(value) != 4:
        return None
    try:
        return [float(value[0]), float(value[1]), float(value[2]), float(value[3])]
    except Exception:
        return None


def _same_question(previous_page_data: Optional[Dict[str, Any]], current_page_data: Optional[Dict[str, Any]]) -> bool:
    prev = previous_page_data if isinstance(previous_page_data, dict) else {}
    cur = current_page_data if isinstance(current_page_data, dict) else {}
    prev_qid = str(prev.get("qid") or "").strip()
    cur_qid = str(cur.get("qid") or "").strip()
    if prev_qid and cur_qid:
        return prev_qid == cur_qid
    prev_question = " ".join(str(prev.get("question") or "").strip().lower().split())
    cur_question = " ".join(str(cur.get("question") or "").strip().lower().split())
    return bool(prev_question and cur_question and prev_question == cur_question)


def build_textarea_growth_state(
    *,
    previous_page_data: Optional[Dict[str, Any]] = None,
    current_page_data: Optional[Dict[str, Any]] = None,
    action_kind: str = "",
    action_reason: str = "",
    expected_value: str = "",
) -> Dict[str, Any]:
    prev = previous_page_data if isinstance(previous_page_data, dict) else {}
    cur = current_page_data if isinstance(current_page_data, dict) else {}
    reason = str(action_reason or "").strip().lower()
    kind = str(action_kind or "").strip().lower()
    expected = str(expected_value or "").strip()

    prev_multiline = bool(prev.get("textboxMultiline"))
    cur_multiline = bool(cur.get("textboxMultiline"))
    monitor_active = bool((prev_multiline or cur_multiline) and (kind in {"type_text", "key_press"} or reason in {"type_answer", "type_text_answer", "blur_textbox"}))

    prev_bbox = _bbox4(prev.get("textboxBbox"))
    cur_bbox = _bbox4(cur.get("textboxBbox"))
    textbox_present_after = bool(cur.get("hasTextbox")) and cur_bbox is not None
    current_value = str(cur.get("textboxValue") or "").strip()
    value_preserved = bool(expected and current_value == expected)
    focus_released = not bool(cur.get("textboxFocused"))
    same_question = _same_question(prev, cur)

    height_growth = 0.0
    width_change = 0.0
    rerender_suspected = False
    if prev_bbox is not None and cur_bbox is not None:
        prev_height = max(0.0, prev_bbox[3] - prev_bbox[1])
        cur_height = max(0.0, cur_bbox[3] - cur_bbox[1])
        prev_width = max(0.0, prev_bbox[2] - prev_bbox[0])
        cur_width = max(0.0, cur_bbox[2] - cur_bbox[0])
        height_growth = round(cur_height - prev_height, 4)
        width_change = round(cur_width - prev_width, 4)
        rerender_suspected = bool(same_question and abs(width_change) > 12.0)
    elif prev_multiline and same_question and not textbox_present_after:
        rerender_suspected = True

    overlay_or_loss_suspected = bool(monitor_active and same_question and not textbox_present_after and not value_preserved)
    is_healthy = bool(not monitor_active or value_preserved or not same_question)

    return {
        "schema_version": SCHEMA_VERSION,
        "monitor_active": monitor_active,
        "is_healthy": is_healthy,
        "signals": {
            "previous_multiline": prev_multiline,
            "current_multiline": cur_multiline,
            "same_question": same_question,
            "textbox_present_after": textbox_present_after,
            "value_preserved": value_preserved,
            "focus_released": focus_released,
            "height_growth": height_growth,
            "width_change": width_change,
            "rerender_suspected": rerender_suspected,
            "overlay_or_loss_suspected": overlay_or_loss_suspected,
            "current_value": current_value,
            "expected_value": expected,
        },
    }
