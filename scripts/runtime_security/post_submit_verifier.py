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


def build_post_submit_verifier(
    *,
    actions_plan: Iterable[Dict[str, Any]],
    terminal_state: Optional[Dict[str, Any]] = None,
    submission_confirmation: Optional[Dict[str, Any]] = None,
    validation_outcome: Optional[Dict[str, Any]] = None,
    previous_url: str = "",
    current_url: str = "",
) -> Dict[str, Any]:
    submit_action = _first_submit_action(actions_plan)
    required = bool(submit_action)
    terminal = terminal_state or {}
    confirmation = submission_confirmation or {}
    outcome = validation_outcome or {}
    screen = (terminal.get("terminal_screen") or {}) if isinstance(terminal.get("terminal_screen"), dict) else {}
    terminal_kind = str(screen.get("screen_kind") or "")
    url_changed = bool(previous_url and current_url and previous_url.rstrip("/") != current_url.rstrip("/"))
    terminal_detected = bool(terminal.get("is_terminal"))
    confirmation_detected = bool(confirmation.get("is_confirmed"))
    accepted = str(outcome.get("outcome") or "") == "accepted"

    decision = "skip"
    reason = "not_submit_action"
    if required:
        decision = "unverified"
        reason = "missing_success_signal"
        if confirmation_detected:
            decision = "verified"
            reason = "submission_confirmation"
        elif terminal_detected and terminal_kind in {"thank_you", "confirmation", "complete", "failure", "expired_session"}:
            decision = "verified"
            reason = f"terminal:{terminal_kind}"
        elif accepted and url_changed and terminal_kind in {"review", "summary"}:
            decision = "tentative"
            reason = "review_redirect_only"

    return {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "reason": reason,
        "is_required": required,
        "is_verified": decision == "verified",
        "is_tentative": decision == "tentative",
        "submit_action_kind": str(submit_action.get("kind") or ""),
        "submit_action_reason": str(submit_action.get("reason") or ""),
        "terminal_kind": terminal_kind,
        "submission_confirmed": confirmation_detected,
        "terminal_detected": terminal_detected,
        "url_changed": url_changed,
    }
