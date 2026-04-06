from __future__ import annotations

from typing import Any, Dict, Optional

from scripts.pipeline.contracts import SCHEMA_VERSION


def build_validation_outcome_state(
    *,
    validation_state: Optional[Dict[str, Any]] = None,
    terminal_state: Optional[Dict[str, Any]] = None,
    submission_confirmation: Optional[Dict[str, Any]] = None,
    draft_confirmation: Optional[Dict[str, Any]] = None,
    back_navigation_confirmation: Optional[Dict[str, Any]] = None,
    action_progressed: Optional[bool] = None,
    previous_url: str = "",
    current_url: str = "",
) -> Dict[str, Any]:
    validation = validation_state or {}
    terminal = terminal_state or {}
    confirmation = submission_confirmation or {}
    draft = draft_confirmation or {}
    back_nav = back_navigation_confirmation or {}
    category = str(validation.get("category") or "none")
    is_blocking = bool(validation.get("is_blocking"))
    progressed = None if action_progressed is None else bool(action_progressed)
    terminal_kind = str(((terminal.get("terminal_screen") or {}) if isinstance(terminal.get("terminal_screen"), dict) else {}).get("screen_kind") or "")
    confirmed = bool(confirmation.get("is_confirmed"))
    draft_confirmed = bool(draft.get("is_confirmed"))
    back_confirmed = bool(back_nav.get("is_confirmed"))
    url_changed = bool(previous_url and current_url and previous_url.rstrip("/") != current_url.rstrip("/"))

    outcome = "accepted"
    reason = "no_validation_signals"
    if is_blocking:
        outcome = "blocked"
        reason = category or "blocking_validation"
    elif back_confirmed:
        outcome = "accepted"
        reason = "back_navigated"
    elif draft_confirmed:
        outcome = "accepted"
        reason = "draft_saved"
    elif confirmed or bool(terminal.get("is_terminal")):
        outcome = "accepted"
        reason = "terminal_or_confirmation"
    elif terminal_kind in {"review", "summary"} and (progressed is True or url_changed):
        outcome = "corrected"
        reason = "review_or_summary_transition"
    elif progressed is False and url_changed:
        outcome = "auto_reverted"
        reason = "url_changed_without_acceptance"
    elif progressed is False:
        outcome = "silently_ignored"
        reason = "no_progress_after_action"

    return {
        "schema_version": SCHEMA_VERSION,
        "outcome": outcome,
        "reason": reason,
        "validation_category": category,
        "terminal_kind": terminal_kind,
        "submission_confirmed": confirmed,
        "draft_confirmed": draft_confirmed,
        "back_navigation_confirmed": back_confirmed,
        "action_progressed": progressed,
        "url_changed": url_changed,
    }
