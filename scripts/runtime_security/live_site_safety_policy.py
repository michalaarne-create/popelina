from __future__ import annotations

from typing import Any, Dict, Optional

from scripts.pipeline.contracts import SCHEMA_VERSION


def build_live_site_safety_policy(
    *,
    domain_allowlist: Optional[Dict[str, Any]] = None,
    allowed_context_policy: Optional[Dict[str, Any]] = None,
    allowed_action_policy: Optional[Dict[str, Any]] = None,
    frame_context_guard: Optional[Dict[str, Any]] = None,
    hard_stop_guard: Optional[Dict[str, Any]] = None,
    anti_modal_guard: Optional[Dict[str, Any]] = None,
    consent_barrier_policy: Optional[Dict[str, Any]] = None,
    secret_source_policy: Optional[Dict[str, Any]] = None,
    pre_submit_checklist: Optional[Dict[str, Any]] = None,
    submit_confirmation_guard: Optional[Dict[str, Any]] = None,
    submit_two_phase_guard: Optional[Dict[str, Any]] = None,
    post_submit_verifier: Optional[Dict[str, Any]] = None,
    side_effect_watchdog: Optional[Dict[str, Any]] = None,
    operational_watchdog: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    allowlist = domain_allowlist or {}
    allowed_context = allowed_context_policy or {}
    allowed_action = allowed_action_policy or {}
    frame_context = frame_context_guard or {}
    hard_stop = hard_stop_guard or {}
    anti_modal = anti_modal_guard or {}
    consent_barrier = consent_barrier_policy or {}
    secret_source = secret_source_policy or {}
    pre_submit = pre_submit_checklist or {}
    submit_confirmation = submit_confirmation_guard or {}
    submit_two_phase = submit_two_phase_guard or {}
    post_submit = post_submit_verifier or {}
    side_effects = side_effect_watchdog or {}
    watchdog = operational_watchdog or {}

    decision = "allow"
    reason = "clear"
    if bool(hard_stop.get("is_blocking")):
        decision = "block"
        reason = f"hard_stop:{str(hard_stop.get('source') or 'unknown')}"
    elif bool(anti_modal.get("is_blocking")):
        decision = "block"
        reason = f"anti_modal:{str(anti_modal.get('reason') or 'blocked')}"
    elif bool(consent_barrier.get("is_blocking")):
        decision = "block"
        reason = f"consent_barrier:{str(consent_barrier.get('reason') or 'blocked')}"
    elif bool(secret_source.get("is_blocking")):
        decision = "block"
        reason = f"secret_source:{str(secret_source.get('reason') or 'blocked')}"
    elif allowlist and not bool(allowlist.get("is_allowed", True)):
        decision = "block"
        reason = f"domain_allowlist:{str(allowlist.get('reason') or 'blocked')}"
    elif frame_context and not bool(frame_context.get("is_allowed", True)):
        decision = "block"
        reason = f"frame_context:{str(frame_context.get('reason') or 'blocked')}"
    elif allowed_context and not bool(allowed_context.get("is_allowed")) and not bool(allowed_context.get("requires_review")):
        decision = "block"
        reason = f"allowed_context:{str(allowed_context.get('reason') or 'blocked')}"
    elif allowed_action and not bool(allowed_action.get("is_allowed", True)):
        decision = "block"
        reason = f"allowed_action:{str(allowed_action.get('reason') or 'blocked')}"
    elif bool(pre_submit.get("is_required")) and not bool(pre_submit.get("is_allowed")):
        decision = "block"
        reason = f"pre_submit:{str(pre_submit.get('reason') or 'blocked')}"
    elif submit_confirmation and not bool(submit_confirmation.get("is_allowed", True)):
        decision = "block"
        reason = f"submit_confirmation:{str(submit_confirmation.get('reason') or 'blocked')}"
    elif submit_two_phase and not bool(submit_two_phase.get("is_allowed", True)):
        decision = "block"
        reason = f"submit_two_phase:{str(submit_two_phase.get('reason') or 'blocked')}"
    elif bool(side_effects.get("is_blocking")):
        decision = "block"
        reason = f"side_effects:{str(side_effects.get('reason') or 'blocked')}"
    elif bool(post_submit.get("is_required")) and not bool(post_submit.get("is_verified")):
        if bool(post_submit.get("is_tentative")):
            decision = "warn"
            reason = f"post_submit:{str(post_submit.get('reason') or 'tentative')}"
        else:
            decision = "warn"
            reason = f"post_submit:{str(post_submit.get('reason') or 'unverified')}"
    elif bool(secret_source.get("requires_review")):
        decision = "warn"
        reason = f"secret_source:{str(secret_source.get('reason') or 'review')}"
    elif bool(allowed_context.get("requires_review")):
        decision = "warn"
        reason = f"allowed_context:{str(allowed_context.get('reason') or 'review')}"
    elif str(watchdog.get("decision") or "") in {"retry", "fallback", "watch"}:
        decision = "warn"
        reason = f"operational_watchdog:{str(watchdog.get('reason') or watchdog.get('decision') or 'review')}"
    elif str(watchdog.get("decision") or "") in {"abort", "stop"}:
        decision = "block"
        reason = f"operational_watchdog:{str(watchdog.get('reason') or watchdog.get('decision') or 'blocked')}"

    return {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "reason": reason,
        "is_blocking": decision == "block",
        "requires_review": decision == "warn",
        "signals": {
            "domain_allowlist_blocking": allowlist and not bool(allowlist.get("is_allowed", True)),
            "frame_context_blocking": frame_context and not bool(frame_context.get("is_allowed", True)),
            "allowed_context_blocking": allowed_context and not bool(allowed_context.get("is_allowed")) and not bool(allowed_context.get("requires_review")),
            "allowed_context_review": bool(allowed_context.get("requires_review")),
            "allowed_action_blocking": allowed_action and not bool(allowed_action.get("is_allowed", True)),
            "hard_stop_blocking": bool(hard_stop.get("is_blocking")),
            "anti_modal_blocking": bool(anti_modal.get("is_blocking")),
            "consent_barrier_blocking": bool(consent_barrier.get("is_blocking")),
            "secret_source_blocking": bool(secret_source.get("is_blocking")),
            "secret_source_review": bool(secret_source.get("requires_review")),
            "pre_submit_blocking": bool(pre_submit.get("is_required")) and not bool(pre_submit.get("is_allowed")),
            "submit_confirmation_blocking": submit_confirmation and not bool(submit_confirmation.get("is_allowed", True)),
            "submit_two_phase_blocking": submit_two_phase and not bool(submit_two_phase.get("is_allowed", True)),
            "post_submit_unverified": bool(post_submit.get("is_required")) and not bool(post_submit.get("is_verified")),
            "side_effect_blocking": bool(side_effects.get("is_blocking")),
            "watchdog_requires_review": str(watchdog.get("decision") or "") in {"retry", "fallback", "watch"},
            "watchdog_blocking": str(watchdog.get("decision") or "") in {"abort", "stop"},
        },
        "components": {
            "domain_allowlist": allowlist,
            "allowed_context_policy": allowed_context,
            "allowed_action_policy": allowed_action,
            "frame_context_guard": frame_context,
            "hard_stop_guard": hard_stop,
            "anti_modal_guard": anti_modal,
            "consent_barrier_policy": consent_barrier,
            "secret_source_policy": secret_source,
            "pre_submit_checklist": pre_submit,
            "submit_confirmation_guard": submit_confirmation,
            "submit_two_phase_guard": submit_two_phase,
            "post_submit_verifier": post_submit,
            "side_effect_watchdog": side_effects,
            "operational_watchdog": watchdog,
        },
    }
