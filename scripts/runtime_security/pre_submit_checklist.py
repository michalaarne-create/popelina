from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from scripts.pipeline.contracts import SCHEMA_VERSION


_SUBMIT_REASON_TOKENS = (
    "submit",
    "final_submit",
    "finish",
    "send",
    "done",
    "confirm",
    "potwierdz",
    "wyslij",
    "zakoncz",
)


def _action_requests_submit(action: Dict[str, Any]) -> bool:
    kind = str(action.get("kind") or "").strip().lower()
    reason = str(action.get("reason") or "").strip().lower()
    combo = str(action.get("combo") or "").strip().lower()
    if kind == "key_press" and combo == "enter" and "submit" in reason:
        return True
    if kind in {"screen_click", "dom_click", "key_press"} and any(token in reason for token in _SUBMIT_REASON_TOKENS):
        return True
    return False


def _first_submit_action(actions_plan: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    for row in actions_plan:
        if isinstance(row, dict) and _action_requests_submit(row):
            return row
    return {}


def _resolve_site_consistency_score(
    site_consistency_score: Optional[float],
    allowlist_state: Optional[Dict[str, Any]],
) -> float:
    if site_consistency_score is not None:
        try:
            return max(0.0, min(1.0, float(site_consistency_score)))
        except Exception:
            pass
    return 1.0 if bool((allowlist_state or {}).get("is_allowed")) else 0.0


def build_pre_submit_checklist(
    *,
    actions_plan: Iterable[Dict[str, Any]],
    allowlist_state: Optional[Dict[str, Any]] = None,
    anti_modal_guard: Optional[Dict[str, Any]] = None,
    validation_state: Optional[Dict[str, Any]] = None,
    terminal_state: Optional[Dict[str, Any]] = None,
    submission_confirmation: Optional[Dict[str, Any]] = None,
    submit_guard_state: Optional[Dict[str, Any]] = None,
    site_consistency_score: Optional[float] = None,
    min_site_consistency_score: float = 0.72,
) -> Dict[str, Any]:
    submit_action = _first_submit_action(actions_plan)
    requires_checklist = bool(submit_action)
    allowlist = allowlist_state or {}
    anti_modal = anti_modal_guard or {}
    validation = validation_state or {}
    terminal = terminal_state or {}
    confirmation = submission_confirmation or {}
    submit_guard = submit_guard_state or {}
    site_score = _resolve_site_consistency_score(site_consistency_score, allowlist)
    terminal_confidence = float(terminal.get("terminal_confidence_score") or 0.0)

    decision = "skip"
    reason = "not_submit_action"
    if requires_checklist:
        decision = "allow"
        reason = "ok"
        if not bool(allowlist.get("is_allowed")):
            decision = "block"
            reason = "domain_not_allowlisted"
        elif bool(anti_modal.get("is_blocking")):
            decision = "block"
            reason = "anti_modal_blocking"
        elif bool(submit_guard.get("is_hard_stop")):
            decision = "block"
            reason = str(submit_guard.get("stop_reason") or "submit_guard_hard_stop")
        elif bool(validation.get("is_blocking")):
            decision = "block"
            reason = "blocking_validation_state"
        elif bool(confirmation.get("is_confirmed")) or bool(terminal.get("is_terminal")):
            decision = "block"
            reason = "already_terminal_or_confirmed"
        elif site_score < float(min_site_consistency_score):
            decision = "block"
            reason = "low_site_consistency"

    return {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "reason": reason,
        "is_required": requires_checklist,
        "is_allowed": decision != "block",
        "submit_action_kind": str(submit_action.get("kind") or ""),
        "submit_action_reason": str(submit_action.get("reason") or ""),
        "site_consistency_score": round(site_score, 4),
        "min_site_consistency_score": float(min_site_consistency_score),
        "terminal_confidence_score": round(terminal_confidence, 4),
        "signals": {
            "allowlist_allowed": bool(allowlist.get("is_allowed")),
            "anti_modal_blocking": bool(anti_modal.get("is_blocking")),
            "validation_blocking": bool(validation.get("is_blocking")),
            "submit_guard_hard_stop": bool(submit_guard.get("is_hard_stop")),
            "submission_confirmed": bool(confirmation.get("is_confirmed")),
            "terminal_detected": bool(terminal.get("is_terminal")),
        },
    }
