from __future__ import annotations

from typing import Any, Dict, Optional

from scripts.pipeline.contracts import SCHEMA_VERSION


def _primary_action(actions_plan: Any) -> Dict[str, Any]:
    if isinstance(actions_plan, list) and actions_plan:
        first = actions_plan[0]
        if isinstance(first, dict):
            return first
    return {}


def build_accessibility_target_regression_suite(
    *,
    accessibility_hint_packet: Optional[Dict[str, Any]] = None,
    actions_plan: Any = None,
) -> Dict[str, Any]:
    packet = accessibility_hint_packet or {}
    hint = packet.get("hint") if isinstance(packet.get("hint"), dict) else {}
    action = _primary_action(actions_plan)
    action_reason = str(action.get("reason") or "")
    is_target_action = action_reason.startswith(("click_answer", "click_next", "focus_textbox", "focus_select"))
    role = str(hint.get("role") or "").strip()
    name = str(hint.get("name") or "").strip()

    passed = bool(packet.get("has_hint")) and (bool(role) or bool(name) or not is_target_action)
    reason = "semantic_target_preserved"
    if not packet.get("has_hint"):
        reason = "missing_accessibility_hint"
    elif is_target_action and not (role or name):
        reason = "target_lost_semantic_identity"

    return {
        "schema_version": SCHEMA_VERSION,
        "passed": passed,
        "reason": reason,
        "is_target_action": is_target_action,
        "target_hint": {
            "role": role,
            "name": name,
            "name_source": str(hint.get("name_source") or ""),
        },
        "action_reason": action_reason,
    }
