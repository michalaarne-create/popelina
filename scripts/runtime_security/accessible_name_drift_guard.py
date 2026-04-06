from __future__ import annotations

from typing import Any, Dict, Optional

from scripts.pipeline.contracts import SCHEMA_VERSION


def _primary_name(contract: Optional[Dict[str, Any]]) -> str:
    payload = contract or {}
    primary = payload.get("primary_name") if isinstance(payload.get("primary_name"), dict) else {}
    return str(primary.get("chosen_name") or "").strip()


def build_accessible_name_drift_guard(
    *,
    previous_accessible_name_precedence_contract: Optional[Dict[str, Any]] = None,
    current_accessible_name_precedence_contract: Optional[Dict[str, Any]] = None,
    action_result_packet: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    previous_name = _primary_name(previous_accessible_name_precedence_contract)
    current_name = _primary_name(current_accessible_name_precedence_contract)
    transition = action_result_packet or {}
    state_changed = bool(transition.get("question_changed")) or bool(transition.get("page_changed"))
    transition_kind = str(transition.get("transition_kind") or "")

    drift_detected = False
    reason = "accessible_name_stable"
    if previous_name and current_name and previous_name != current_name and not state_changed:
        drift_detected = True
        reason = "name_changed_without_state_transition"
    elif not previous_name or not current_name:
        reason = "insufficient_accessible_name_signal"

    return {
        "schema_version": SCHEMA_VERSION,
        "drift_detected": drift_detected,
        "reason": reason,
        "signals": {
            "previous_name": previous_name,
            "current_name": current_name,
            "transition_kind": transition_kind,
            "state_changed": state_changed,
        },
    }
