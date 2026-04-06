from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from scripts.pipeline.contracts import SCHEMA_VERSION


def _is_risky_step(step: Dict[str, Any]) -> bool:
    validation_outcome = step.get("validation_outcome") if isinstance(step.get("validation_outcome"), dict) else {}
    watchdog = step.get("operational_watchdog") if isinstance(step.get("operational_watchdog"), dict) else {}
    post_submit = step.get("post_submit_verifier") if isinstance(step.get("post_submit_verifier"), dict) else {}
    live_safety = step.get("live_site_safety_policy") if isinstance(step.get("live_site_safety_policy"), dict) else {}
    outcome = str(validation_outcome.get("outcome") or "")
    watchdog_decision = str(watchdog.get("decision") or "")
    if outcome in {"blocked", "silently_ignored", "auto_reverted"}:
        return True
    if watchdog_decision in {"abort", "fallback"}:
        return True
    if bool(post_submit.get("is_required")) and not bool(post_submit.get("is_verified")):
        return True
    if bool(live_safety.get("is_blocking")):
        return True
    return False


def build_escalation_threshold_policy(
    *,
    previous_steps: Optional[Iterable[Dict[str, Any]]] = None,
    current_step: Optional[Dict[str, Any]] = None,
    consecutive_threshold: int = 2,
    rolling_threshold: int = 3,
    rolling_window: int = 4,
) -> Dict[str, Any]:
    history = [row for row in (previous_steps or []) if isinstance(row, dict)]
    current = current_step or {}
    recent = (history + [current])[-max(1, int(rolling_window or 1)) :]

    consecutive_risky = 0
    for row in reversed(history + [current]):
        if _is_risky_step(row):
            consecutive_risky += 1
        else:
            break
    rolling_risky = sum(1 for row in recent if _is_risky_step(row))

    decision = "continue"
    reason = "risk_below_threshold"
    if consecutive_risky >= int(consecutive_threshold):
        decision = "escalate"
        reason = "consecutive_risky_steps"
    elif rolling_risky >= int(rolling_threshold):
        decision = "escalate"
        reason = "rolling_risky_steps"

    return {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "reason": reason,
        "is_blocking": decision == "escalate",
        "requires_operator": decision == "escalate",
        "signals": {
            "consecutive_risky_steps": int(consecutive_risky),
            "rolling_risky_steps": int(rolling_risky),
            "rolling_window": int(rolling_window),
            "consecutive_threshold": int(consecutive_threshold),
            "rolling_threshold": int(rolling_threshold),
        },
    }
