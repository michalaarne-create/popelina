from __future__ import annotations

from typing import Any, Dict, Sequence

from scripts.pipeline.contracts import SCHEMA_VERSION


_KEYBOARD_KINDS = {"key_press", "key_repeat", "type_text", "wait", "noop"}


def build_fallback_transition_policy(
    *,
    plan_source: str,
    actions_plan: Sequence[Dict[str, Any]] | None = None,
    screen_actions: Sequence[Dict[str, Any]] | None = None,
    has_dom_fallback: bool = False,
    has_execution_override: bool = False,
    has_navigation_override: bool = False,
    has_page_state_navigation: bool = False,
) -> Dict[str, Any]:
    source = str(plan_source or "").strip().lower()
    actions = list(actions_plan or [])
    screen_plan = list(screen_actions or [])
    first_kind = str(((actions[0] or {}) if actions else {}).get("kind") or "").strip().lower()
    first_screen_kind = str(((screen_plan[0] or {}) if screen_plan else {}).get("kind") or "").strip().lower()

    decision = "allow"
    reason = "screen_plan_kept"
    transition_kind = "none"

    if source == "screen_only":
        decision = "allow"
        reason = "screen_plan_kept"
        transition_kind = "screen_only"
    elif source in {"dom_fallback", "runtime_recovery_dom_fallback"}:
        decision = "allow" if has_dom_fallback else "block"
        reason = "model_to_heuristic_dom_fallback" if has_dom_fallback else "missing_dom_fallback_evidence"
        transition_kind = "model_to_heuristic"
    elif source == "execution_override":
        decision = "allow" if has_execution_override else "block"
        reason = "model_to_execution_override" if has_execution_override else "missing_execution_override_evidence"
        transition_kind = "model_to_heuristic"
    elif source == "navigation_override":
        decision = "allow" if has_navigation_override else "block"
        reason = "screen_to_navigation_override" if has_navigation_override else "missing_navigation_override_evidence"
        transition_kind = "model_to_heuristic"
    elif source == "page_state_navigation":
        decision = "allow" if has_page_state_navigation else "block"
        reason = "screen_to_page_state_navigation" if has_page_state_navigation else "missing_page_state_navigation_evidence"
        transition_kind = "model_to_heuristic"
    elif source == "runtime_recovery_retry_same_action":
        decision = "allow"
        reason = "action_to_observe_retry_same_action"
        transition_kind = "action_to_observe"

    if decision == "allow" and first_kind in _KEYBOARD_KINDS and first_screen_kind not in _KEYBOARD_KINDS:
        transition_kind = "click_to_keyboard"
        reason = "click_to_keyboard_fallback"

    return {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "reason": reason,
        "is_allowed": decision == "allow",
        "requires_review": decision != "allow",
        "transition_kind": transition_kind,
        "plan_source": source,
        "signals": {
            "has_dom_fallback": bool(has_dom_fallback),
            "has_execution_override": bool(has_execution_override),
            "has_navigation_override": bool(has_navigation_override),
            "has_page_state_navigation": bool(has_page_state_navigation),
            "first_action_kind": first_kind,
            "first_screen_action_kind": first_screen_kind,
        },
    }
