from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from scripts.pipeline.screen_contracts import build_hidden_state_explainer


def build_llm_ui_explainer(
    *,
    screen_state: Mapping[str, Any] | None = None,
    decision: Mapping[str, Any] | None = None,
    hidden_state_diff: Mapping[str, Any] | None = None,
    client_state_snapshot: Mapping[str, Any] | None = None,
    storage_dependency_registry: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    screen = dict(screen_state or {})
    decision_row = dict(decision or {})
    hidden = build_hidden_state_explainer(
        state_side_effect_diff=hidden_state_diff,
        client_state_snapshot=client_state_snapshot,
        storage_dependency_registry=storage_dependency_registry,
    )

    question = str(screen.get("question_text") or screen.get("prompt_text") or screen.get("question") or "").strip()
    detected_type = str(screen.get("detected_quiz_type") or screen.get("control_kind") or "").strip()
    action_kind = str((decision_row.get("action") or {}).get("kind") or decision_row.get("action_kind") or "").strip()
    reason = str(decision_row.get("reason") or decision_row.get("plan_reason") or "").strip()
    screen_signature = str(screen.get("screen_signature") or screen.get("signature") or "").strip()

    brief_parts = []
    if question:
        brief_parts.append(f"question={question}")
    if detected_type:
        brief_parts.append(f"type={detected_type}")
    if action_kind:
        brief_parts.append(f"action={action_kind}")
    if reason:
        brief_parts.append(f"reason={reason}")
    if hidden.get("has_hidden_state_shift"):
        brief_parts.append(f"hidden_state={hidden.get('source')}")
    if screen_signature:
        brief_parts.append(f"signature={screen_signature}")

    if not brief_parts:
        brief_parts.append("screen_explainer_unavailable")

    return {
        "brief": " | ".join(brief_parts),
        "screen": screen,
        "decision": decision_row,
        "hidden_state_explainer": hidden,
        "confidence": float(hidden.get("confidence") or 0.0),
        "recommended_action_kind": action_kind or str(screen.get("recommended_action_kind") or ""),
        "source": "llm_ui_explainer",
    }
