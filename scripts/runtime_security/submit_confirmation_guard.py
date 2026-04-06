from __future__ import annotations

import hashlib
import json
import time
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


def _action_fingerprint(action: Optional[Dict[str, Any]]) -> str:
    payload = {
        "kind": str((action or {}).get("kind") or ""),
        "reason": str((action or {}).get("reason") or ""),
        "combo": str((action or {}).get("combo") or ""),
        "bbox": list((action or {}).get("bbox") or []),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _first_action(actions_plan: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    for row in actions_plan:
        if isinstance(row, dict):
            return row
    return {}


def _first_submit_action(actions_plan: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    first_action: Dict[str, Any] = {}
    for row in actions_plan:
        if not isinstance(row, dict):
            continue
        if not first_action:
            first_action = row
        if _action_requests_submit(row):
            return row
    return first_action


def _action_requests_submit(action: Dict[str, Any]) -> bool:
    kind = str(action.get("kind") or "").strip().lower()
    reason = str(action.get("reason") or "").strip().lower()
    combo = str(action.get("combo") or "").strip().lower()
    if kind == "key_press" and combo == "enter" and "submit" in reason:
        return True
    if kind in {"screen_click", "dom_click", "key_press"} and any(token in reason for token in _SUBMIT_REASON_TOKENS):
        return True
    return False


def _screen_submit_context(
    *,
    runtime_state: Optional[Dict[str, Any]],
    terminal_state: Optional[Dict[str, Any]],
) -> bool:
    runtime = runtime_state or {}
    terminal = terminal_state or {}
    state_name = str(runtime.get("state") or "")
    terminal_kind = str(((terminal.get("terminal_screen") or {}) if isinstance(terminal.get("terminal_screen"), dict) else {}).get("screen_kind") or "")
    return state_name in {"submit", "review"} or terminal_kind in {"review", "summary"}


def build_submit_confirmation_guard_state(
    *,
    actions_plan: Iterable[Dict[str, Any]],
    runtime_state: Optional[Dict[str, Any]] = None,
    terminal_state: Optional[Dict[str, Any]] = None,
    validation_state: Optional[Dict[str, Any]] = None,
    submit_guard_state: Optional[Dict[str, Any]] = None,
    mission_mode: str = "live",
    operator_confirmation: Optional[Dict[str, Any]] = None,
    confirmation_ttl_s: float = 120.0,
    now_ts: Optional[float] = None,
) -> Dict[str, Any]:
    now = float(now_ts if now_ts is not None else time.time())
    first = _first_action(actions_plan)
    submit_action = _first_submit_action(actions_plan)
    first_kind = str(submit_action.get("kind") or first.get("kind") or "")
    fingerprint = _action_fingerprint(submit_action)
    action_requests_submit = _action_requests_submit(submit_action)
    screen_submit_context = _screen_submit_context(runtime_state=runtime_state, terminal_state=terminal_state)
    mode = str(mission_mode or "live").strip().lower()
    strict_mode = mode in {"live", "prod", "production"}
    requires_confirmation = bool(action_requests_submit and (strict_mode or screen_submit_context))

    validation_block = bool((validation_state or {}).get("is_blocking"))
    hard_stop = bool((submit_guard_state or {}).get("is_hard_stop"))
    ack = operator_confirmation or {}
    ack_confirmed = bool(ack.get("confirmed"))
    ack_fingerprint = str(ack.get("action_fingerprint") or "")
    ack_ts = float(ack.get("confirmed_ts") or 0.0)
    ack_age_s = max(0.0, now - ack_ts) if ack_ts else float("inf")
    ack_fingerprint_ok = bool(not ack_fingerprint or ack_fingerprint == fingerprint)
    ack_fresh = bool(ack_confirmed and ack_age_s <= float(confirmation_ttl_s))
    is_confirmed = bool(ack_fresh and ack_fingerprint_ok)

    decision = "allow"
    reason = "ok"
    if hard_stop:
        decision = "block"
        reason = "submit_guard_hard_stop"
    elif validation_block:
        decision = "block"
        reason = "blocking_validation_state"
    elif requires_confirmation and not is_confirmed:
        decision = "block"
        reason = "missing_operator_confirmation"

    return {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "reason": reason,
        "is_allowed": decision == "allow",
        "requires_confirmation": bool(requires_confirmation),
        "is_confirmed": bool(is_confirmed),
        "first_action_kind": first_kind,
        "action_fingerprint": fingerprint,
        "signals": {
            "action_requests_submit": bool(action_requests_submit),
            "screen_submit_context": bool(screen_submit_context),
            "validation_block": bool(validation_block),
            "submit_hard_stop": bool(hard_stop),
            "ack_confirmed": bool(ack_confirmed),
            "ack_fingerprint_ok": bool(ack_fingerprint_ok),
            "ack_age_s": None if ack_age_s == float("inf") else round(ack_age_s, 4),
            "mission_mode": mode,
        },
        "confirmation_request": {
            "action_fingerprint": fingerprint,
            "ttl_s": float(confirmation_ttl_s),
            "issued_ts": now,
        },
    }
