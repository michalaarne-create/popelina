from __future__ import annotations

import time
from typing import Any, Dict, Iterable, Optional

from scripts.pipeline.contracts import SCHEMA_VERSION
from popelina_github.scripts.runtime_security.submit_confirmation_guard import (
    build_submit_confirmation_guard_state,
)


def _requests_submit(actions_plan: Iterable[Dict[str, Any]]) -> bool:
    for row in actions_plan:
        if not isinstance(row, dict):
            continue
        reason = str(row.get("reason") or "").strip().lower()
        kind = str(row.get("kind") or "").strip().lower()
        if "submit" in reason or (kind == "screen_click" and "final" in reason and "form" in reason):
            return True
    return False


def build_submit_two_phase_guard(
    *,
    actions_plan: Iterable[Dict[str, Any]],
    runtime_state: Optional[Dict[str, Any]] = None,
    terminal_state: Optional[Dict[str, Any]] = None,
    validation_state: Optional[Dict[str, Any]] = None,
    submit_guard_state: Optional[Dict[str, Any]] = None,
    mission_mode: str = "live",
    operator_confirmation: Optional[Dict[str, Any]] = None,
    commit_requested: bool = False,
    now_ts: Optional[float] = None,
) -> Dict[str, Any]:
    now = float(now_ts if now_ts is not None else time.time())
    confirmation_guard = build_submit_confirmation_guard_state(
        actions_plan=actions_plan,
        runtime_state=runtime_state,
        terminal_state=terminal_state,
        validation_state=validation_state,
        submit_guard_state=submit_guard_state,
        mission_mode=mission_mode,
        operator_confirmation=operator_confirmation,
        now_ts=now,
    )
    submit_requested = _requests_submit(actions_plan)
    phase = "none"
    decision = "allow"
    reason = "not_submit"
    if submit_requested:
        phase = "ready_to_submit"
        reason = "ready_to_submit"
        if not bool(confirmation_guard.get("is_allowed")):
            decision = "block"
            reason = str(confirmation_guard.get("reason") or "missing_operator_confirmation")
        elif not bool(commit_requested):
            decision = "block"
            reason = "commit_submit_required"
        else:
            phase = "commit_submit"
            decision = "allow"
            reason = "commit_submit"
    return {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "reason": reason,
        "phase": phase,
        "is_allowed": decision == "allow",
        "submit_requested": bool(submit_requested),
        "commit_requested": bool(commit_requested),
        "requires_commit": bool(submit_requested),
        "confirmation_guard": confirmation_guard,
        "signals": {
            "operator_confirmed": bool((operator_confirmation or {}).get("confirmed")),
            "confirmation_allowed": bool(confirmation_guard.get("is_allowed")),
            "mission_mode": str(mission_mode or "live").strip().lower(),
        },
    }
