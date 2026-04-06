from __future__ import annotations

from typing import Any, Dict, Optional

from scripts.pipeline.contracts import SCHEMA_VERSION


def build_autonomy_drop_rule(
    *,
    screen_site_consistency: float = 1.0,
    confidence_fusion_policy: Optional[Dict[str, Any]] = None,
    broken_screen_shape_guard: Optional[Dict[str, Any]] = None,
    qid_alignment_contract: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    consistency = float(screen_site_consistency or 0.0)
    confidence_policy = confidence_fusion_policy or {}
    raw_fused_confidence = confidence_policy.get("fused_confidence")
    fused_confidence = float(raw_fused_confidence) if raw_fused_confidence is not None else 1.0
    broken_guard = broken_screen_shape_guard or {}
    qid_contract = qid_alignment_contract or {}
    broken = bool(broken_guard.get("is_broken"))
    qid_aligned = bool(qid_contract.get("aligned", True))

    decision = "allow"
    reason = "sources_consistent_enough"
    if (consistency < 0.45 and fused_confidence < 0.55) or (broken and fused_confidence < 0.6):
        decision = "drop_to_operator"
        reason = "high_source_conflict"
    elif not qid_aligned and fused_confidence < 0.7:
        decision = "drop_to_operator"
        reason = "qid_alignment_conflict"
    elif consistency < 0.62 or fused_confidence < 0.62 or broken or not qid_aligned:
        decision = "warn"
        reason = "source_conflict_review"

    return {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "reason": reason,
        "is_blocking": decision == "drop_to_operator",
        "requires_review": decision in {"warn", "drop_to_operator"},
        "signals": {
            "screen_site_consistency": round(consistency, 4),
            "fused_confidence": round(fused_confidence, 4),
            "broken_screen_shape": bool(broken),
            "qid_aligned": bool(qid_aligned),
        },
    }
