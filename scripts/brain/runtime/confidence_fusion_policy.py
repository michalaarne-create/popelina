from __future__ import annotations

from typing import Any, Dict

from scripts.pipeline.contracts import SCHEMA_VERSION


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def build_confidence_fusion_policy(
    *,
    screen_confidence: float,
    type_confidence: float,
    decision_margin: float,
    control_model: Dict[str, Any] | None,
    has_dom_context: bool,
) -> Dict[str, Any]:
    control_conf = float((control_model or {}).get("conf") or 0.0)
    margin_conf = _clamp(float(decision_margin or 0.0))
    dom_conf = 0.75 if has_dom_context else 0.0
    fused = _clamp(
        (float(screen_confidence or 0.0) * 0.35)
        + (float(type_confidence or 0.0) * 0.35)
        + (control_conf * 0.2)
        + (margin_conf * 0.1)
    )
    if has_dom_context:
        fused = _clamp(fused + 0.05)
    mode = "screen_primary"
    if has_dom_context and fused < 0.72:
        mode = "screen_with_dom_fallback"
    elif fused < 0.5:
        mode = "low_confidence_review"
    return {
        "schema_version": SCHEMA_VERSION,
        "weights": {
            "screen": 0.35,
            "type": 0.35,
            "control_model": 0.2,
            "margin": 0.1,
            "dom_bonus": 0.05 if has_dom_context else 0.0,
        },
        "inputs": {
            "screen_confidence": round(float(screen_confidence or 0.0), 4),
            "type_confidence": round(float(type_confidence or 0.0), 4),
            "decision_margin": round(float(decision_margin or 0.0), 4),
            "control_model_confidence": round(control_conf, 4),
            "dom_confidence": round(dom_conf, 4),
        },
        "fused_confidence": round(fused, 4),
        "mode": mode,
    }
