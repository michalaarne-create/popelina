from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, Optional

from scripts.pipeline.contracts import SCHEMA_VERSION


_SUBMIT_TOKENS = (
    "submit",
    "finish",
    "send",
    "confirm",
    "done",
    "potwierdz",
    "wyslij",
    "zakoncz",
)


def _normalize_event(row: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(row, dict):
        return None
    ts = row.get("ts")
    if ts is None:
        return None
    try:
        ts_value = float(ts)
    except Exception:
        return None
    return {
        "ts": ts_value,
        "family": str(row.get("family") or "unknown"),
        "kind": str(row.get("kind") or ""),
        "reason": str(row.get("reason") or ""),
    }


def classify_action_family(*, action_kind: str, reason: str = "") -> str:
    kind = str(action_kind or "").strip().lower()
    why = str(reason or "").strip().lower()
    submit_like = any(token in why for token in _SUBMIT_TOKENS)
    if submit_like:
        return "submit"
    if kind in {"screen_click", "dom_click", "key_press", "key_repeat"}:
        return "interaction"
    if kind in {"type_text", "dom_fill"}:
        return "typing"
    if kind in {"wait", "noop"}:
        return "passive"
    return "unknown"


def apply_action_rate_limit(
    *,
    action_kind: str,
    reason: str = "",
    history: Optional[Iterable[Dict[str, Any]]] = None,
    now_ts: Optional[float] = None,
    min_interval_s: float = 0.22,
    burst_window_s: float = 2.0,
    burst_max: int = 6,
    submit_cooldown_s: float = 8.0,
) -> Dict[str, Any]:
    now = float(now_ts if now_ts is not None else time.time())
    family = classify_action_family(action_kind=action_kind, reason=reason)
    max_window = max(float(burst_window_s), float(submit_cooldown_s), float(min_interval_s))
    normalized_history: List[Dict[str, Any]] = []
    for row in history or []:
        parsed = _normalize_event(row)
        if parsed is None:
            continue
        if now - float(parsed["ts"]) <= max_window + 0.001:
            normalized_history.append(parsed)
    normalized_history.sort(key=lambda row: float(row["ts"]))

    allowed = True
    rule = "ok"
    wait_s = 0.0
    last_event = normalized_history[-1] if normalized_history else None
    if last_event is not None:
        since_last = now - float(last_event["ts"])
        if since_last < float(min_interval_s):
            allowed = False
            rule = "min_interval"
            wait_s = max(wait_s, float(min_interval_s) - since_last)

    if allowed and family in {"interaction", "typing"}:
        interaction_in_window = [
            row for row in normalized_history
            if row.get("family") in {"interaction", "typing"} and (now - float(row["ts"])) <= float(burst_window_s)
        ]
        if len(interaction_in_window) >= int(burst_max):
            oldest = float(interaction_in_window[0]["ts"])
            allowed = False
            rule = "burst_limit"
            wait_s = max(wait_s, (oldest + float(burst_window_s)) - now)

    if allowed and family == "submit":
        submit_events = [row for row in normalized_history if row.get("family") == "submit"]
        if submit_events:
            last_submit_ts = float(submit_events[-1]["ts"])
            since_submit = now - last_submit_ts
            if since_submit < float(submit_cooldown_s):
                allowed = False
                rule = "submit_cooldown"
                wait_s = max(wait_s, float(submit_cooldown_s) - since_submit)

    updated_history = list(normalized_history)
    if allowed:
        updated_history.append(
            {
                "ts": now,
                "family": family,
                "kind": str(action_kind or ""),
                "reason": str(reason or ""),
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "allowed": bool(allowed),
        "rule": rule,
        "wait_ms": int(max(0.0, wait_s) * 1000.0),
        "action_family": family,
        "history": updated_history,
        "window": {
            "min_interval_s": float(min_interval_s),
            "burst_window_s": float(burst_window_s),
            "burst_max": int(burst_max),
            "submit_cooldown_s": float(submit_cooldown_s),
        },
    }
