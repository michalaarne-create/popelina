from __future__ import annotations

from typing import Any, Dict, Optional

from scripts.pipeline.contracts import SCHEMA_VERSION


def build_accessibility_confidence_packet(
    *,
    accessibility_hint_packet: Optional[Dict[str, Any]] = None,
    accessibility_identity_resolver: Optional[Dict[str, Any]] = None,
    aria_signal_registry: Optional[Dict[str, Any]] = None,
    decision_trace: Optional[Dict[str, Any]] = None,
    screen_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    hint_packet = accessibility_hint_packet or {}
    identity = accessibility_identity_resolver or {}
    aria_registry = aria_signal_registry or {}
    trace = decision_trace or {}
    state = screen_state or {}

    has_hint = bool(hint_packet.get("has_hint"))
    has_identity = bool(identity.get("has_semantic_identity"))
    has_aria = bool(aria_registry.get("has_aria_signals"))
    screen_site_consistency = float(trace.get("screen_site_consistency") or 0.0)
    fused_confidence = float(((state.get("confidence_fusion_policy") or {}) if isinstance(state.get("confidence_fusion_policy"), dict) else {}).get("fused_confidence") or 0.0)

    evidence_count = sum((has_hint, has_identity, has_aria))
    aria_confidence = min(1.0, 0.2 + 0.2 * evidence_count + 0.2 * screen_site_consistency + 0.2 * fused_confidence)
    reason = "strong_accessibility_signal"
    if aria_confidence < 0.45:
        reason = "weak_accessibility_signal"
    elif aria_confidence < 0.75:
        reason = "moderate_accessibility_signal"

    return {
        "schema_version": SCHEMA_VERSION,
        "aria_confidence": round(aria_confidence, 4),
        "reason": reason,
        "signals": {
            "has_hint": has_hint,
            "has_identity": has_identity,
            "has_aria": has_aria,
            "screen_site_consistency": screen_site_consistency,
            "fused_confidence": fused_confidence,
        },
    }
