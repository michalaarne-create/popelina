from __future__ import annotations

from typing import Any, Dict, List, Sequence

from scripts.pipeline.contracts import SCHEMA_VERSION


_SAFE_KEY_RULES = {
    ("key_press", "tab", "blur_textbox"),
    ("key_press", "ctrl+a", "select_all_text"),
    ("key_press", "ctrl+a", "clear_textbox"),
    ("key_press", "ctrl+a", "clear_autocomplete"),
    ("key_press", "backspace", "clear_textbox"),
    ("key_press", "backspace", "clear_autocomplete"),
    ("key_press", "home", "select_to_first"),
    ("key_press", "enter", "confirm_select"),
    ("key_repeat", "down", "move_to_option"),
    ("key_repeat", "down", "move_to_option_tail"),
    ("key_repeat", "pagedown", "scroll_select_page"),
}


def build_keyboard_action_policy(
    *,
    actions_plan: Sequence[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    actions = list(actions_plan or [])
    per_action: List[Dict[str, Any]] = []
    decision = "allow"
    reason = "keyboard_actions_safe_enough"
    blocked_action = None

    for index, action in enumerate(actions):
        kind = str((action or {}).get("kind") or "").strip().lower()
        if kind not in {"key_press", "key_repeat"}:
            per_action.append(
                {
                    "index": index,
                    "kind": kind,
                    "decision": "not_applicable",
                    "reason": "non_keyboard_action",
                }
            )
            continue

        combo = str((action or {}).get("combo") or "").strip().lower()
        action_reason = str((action or {}).get("reason") or "").strip().lower()
        allowed = (kind, combo, action_reason) in _SAFE_KEY_RULES
        action_decision = "allow" if allowed else "block"
        action_reason_out = "known_safe_keyboard_path" if allowed else "unsafe_or_unknown_keyboard_combo"
        per_action.append(
            {
                "index": index,
                "kind": kind,
                "combo": combo,
                "reason": action_reason,
                "decision": action_decision,
                "policy_reason": action_reason_out,
            }
        )
        if not allowed and blocked_action is None:
            decision = "block"
            reason = action_reason_out
            blocked_action = {"index": index, "kind": kind, "combo": combo, "reason": action_reason}

    has_keyboard_actions = any(row.get("kind") in {"key_press", "key_repeat"} for row in per_action)
    return {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "reason": reason,
        "is_allowed": decision == "allow",
        "requires_review": decision != "allow",
        "has_keyboard_actions": has_keyboard_actions,
        "blocked_action": blocked_action,
        "per_action": per_action,
    }
