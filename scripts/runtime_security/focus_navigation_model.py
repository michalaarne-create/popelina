from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from scripts.pipeline.contracts import SCHEMA_VERSION


def build_focus_navigation_model(
    *,
    previous_page_state: Optional[Dict[str, Any]] = None,
    current_page_state: Optional[Dict[str, Any]] = None,
    actions_plan: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    prev = previous_page_state or {}
    cur = current_page_state or {}
    first_action = ((list(actions_plan or []) or [{}])[0]) if actions_plan else {}
    action_kind = str((first_action or {}).get("kind") or "").strip().lower()
    action_reason = str((first_action or {}).get("reason") or "").strip().lower()

    expected_focus = "unchanged"
    if action_reason in {"focus_textbox", "focus_autocomplete", "focus_masked_input"}:
        expected_focus = "textbox"
    elif action_reason in {"focus_select", "select_to_first", "move_to_option", "move_to_option_tail", "scroll_select_page", "confirm_select"}:
        expected_focus = "select"
    elif action_reason == "blur_textbox":
        expected_focus = "none"

    current_focus = "textbox" if bool(cur.get("textboxFocused")) else ("select" if bool(cur.get("hasSelect")) else "none")
    previous_focus = "textbox" if bool(prev.get("textboxFocused")) else ("select" if bool(prev.get("hasSelect")) else "none")

    aligned = True
    reason = "no_focus_transition_expected"
    if expected_focus == "textbox":
        aligned = current_focus == "textbox"
        reason = "focus_to_textbox" if aligned else "textbox_focus_not_reached"
    elif expected_focus == "select":
        aligned = bool(cur.get("hasSelect")) and not bool(cur.get("textboxFocused"))
        reason = "focus_to_select" if aligned else "select_focus_not_reached"
    elif expected_focus == "none":
        aligned = not bool(cur.get("textboxFocused"))
        reason = "focus_released" if aligned else "focus_not_released"

    return {
        "schema_version": SCHEMA_VERSION,
        "action_kind": action_kind,
        "action_reason": action_reason,
        "expected_focus": expected_focus,
        "previous_focus": previous_focus,
        "current_focus": current_focus,
        "is_aligned": aligned,
        "reason": reason,
    }
