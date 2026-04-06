from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from scripts.pipeline.contracts import SCHEMA_VERSION


def build_focus_trap_detector(
    *,
    previous_page_state: Optional[Dict[str, Any]] = None,
    current_page_state: Optional[Dict[str, Any]] = None,
    actions_plan: Optional[Sequence[Dict[str, Any]]] = None,
    anti_modal_guard: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    prev = previous_page_state or {}
    cur = current_page_state or {}
    first_action = ((list(actions_plan or []) or [{}])[0]) if actions_plan else {}
    action_kind = str((first_action or {}).get("kind") or "").strip().lower()
    action_reason = str((first_action or {}).get("reason") or "").strip().lower()
    prev_textbox_focused = bool(prev.get("textboxFocused"))
    cur_textbox_focused = bool(cur.get("textboxFocused"))
    anti_modal = anti_modal_guard or {}

    reason = "clear"
    trapped = False
    if bool(anti_modal.get("is_blocking")):
        trapped = True
        reason = "modal_focus_trap"
    elif action_kind == "key_press" and action_reason == "blur_textbox" and prev_textbox_focused and cur_textbox_focused:
        trapped = True
        reason = "textbox_focus_not_released"

    active_focus = "textbox" if cur_textbox_focused else "none"
    return {
        "schema_version": SCHEMA_VERSION,
        "is_trapped": trapped,
        "reason": reason,
        "requires_review": trapped,
        "signals": {
            "action_kind": action_kind,
            "action_reason": action_reason,
            "previous_textbox_focused": prev_textbox_focused,
            "current_textbox_focused": cur_textbox_focused,
            "active_focus": active_focus,
            "modal_blocking": bool(anti_modal.get("is_blocking")),
        },
    }
