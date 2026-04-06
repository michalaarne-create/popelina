from __future__ import annotations

from typing import Any, Dict, Optional

from scripts.pipeline.contracts import SCHEMA_VERSION


def build_operational_watchdog_state(
    *,
    runtime_state: Optional[Dict[str, Any]] = None,
    iteration: int = 0,
    max_steps: int = 0,
) -> Dict[str, Any]:
    runtime = runtime_state or {}
    stuck = runtime.get("stuck_detection") if isinstance(runtime.get("stuck_detection"), dict) else {}
    recovery = runtime.get("recovery") if isinstance(runtime.get("recovery"), dict) else {}

    no_progress = bool(stuck.get("no_progress"))
    is_stuck = bool(stuck.get("is_stuck"))
    should_abort = bool(recovery.get("should_abort"))
    should_fallback = bool(recovery.get("should_use_dom_fallback"))
    should_retry = bool(recovery.get("should_retry_same_action"))
    remaining_steps = max(0, int(max_steps or 0) - int(iteration or 0))

    decision = "continue"
    state = "healthy"
    reason = "progress_ok"
    if should_abort:
        decision = "abort"
        state = "aborting"
        reason = str(recovery.get("reason") or "abort_iteration")
    elif should_fallback:
        decision = "fallback"
        state = "recovering"
        reason = str(recovery.get("reason") or "dom_fallback")
    elif should_retry:
        decision = "retry"
        state = "recovering"
        reason = str(recovery.get("reason") or "retry_same_action")
    elif no_progress or is_stuck:
        decision = "watch"
        state = "watching"
        reason = "no_progress_watchdog"
    elif remaining_steps <= 0 and int(max_steps or 0) > 0:
        decision = "stop"
        state = "exhausted"
        reason = "max_steps_exhausted"

    return {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "state": state,
        "reason": reason,
        "iteration": int(iteration or 0),
        "max_steps": int(max_steps or 0),
        "remaining_steps": int(remaining_steps),
        "signals": {
            "no_progress": no_progress,
            "is_stuck": is_stuck,
            "should_retry_same_action": should_retry,
            "should_use_dom_fallback": should_fallback,
            "should_abort": should_abort,
            "attempt_count": int(stuck.get("attempt_count") or 0),
            "consecutive_no_progress_count": int(stuck.get("consecutive_no_progress_count") or 0),
            "recovery_level": int(recovery.get("level") or 0),
            "recovery_action": str(recovery.get("action") or "none"),
        },
    }


def build_failure_summary_contract(
    *,
    error: str = "",
    stop_reason: str = "",
    operational_watchdog: Optional[Dict[str, Any]] = None,
    runtime_state: Optional[Dict[str, Any]] = None,
    domain_allowlist: Optional[Dict[str, Any]] = None,
    secret_source_policy: Optional[Dict[str, Any]] = None,
    submit_guard: Optional[Dict[str, Any]] = None,
    anti_modal_guard: Optional[Dict[str, Any]] = None,
    hard_stop_guard: Optional[Dict[str, Any]] = None,
    pre_submit_checklist: Optional[Dict[str, Any]] = None,
    submit_confirmation_guard: Optional[Dict[str, Any]] = None,
    dom_has_quiz: Optional[bool] = None,
    screen_has_quiz: Optional[bool] = None,
) -> Dict[str, Any]:
    watchdog = operational_watchdog or {}
    runtime = runtime_state or {}
    stuck = runtime.get("stuck_detection") if isinstance(runtime.get("stuck_detection"), dict) else {}
    recovery = runtime.get("recovery") if isinstance(runtime.get("recovery"), dict) else {}
    allowlist = domain_allowlist or {}
    secret_source = secret_source_policy or {}
    submit = submit_guard or {}
    anti_modal = anti_modal_guard or {}
    hard_stop = hard_stop_guard or {}
    checklist = pre_submit_checklist or {}
    submit_confirm = submit_confirmation_guard or {}

    what_failed = "runtime_stopped"
    why_failed = str(error or stop_reason or watchdog.get("reason") or "unknown")
    next_best_action = "inspect_latest_trace"
    confidence = "medium"

    if str(stop_reason or "").strip():
        what_failed = str(stop_reason).strip()
    elif bool(hard_stop.get("is_blocking")):
        what_failed = str(hard_stop.get("source") or "hard_stop_guard")
    elif bool(secret_source.get("is_blocking")):
        what_failed = "secret_source_block"
    elif not bool(allowlist.get("is_allowed", True)) and allowlist:
        what_failed = "domain_allowlist_block"
    elif not bool(checklist.get("is_allowed", True)) and checklist:
        what_failed = "pre_submit_check_block"
    elif not bool(submit_confirm.get("is_allowed", True)) and submit_confirm:
        what_failed = "submit_confirmation_block"
    elif bool((watchdog.get("signals") or {}).get("should_abort")):
        what_failed = "watchdog_abort"
    elif bool((watchdog.get("signals") or {}).get("should_use_dom_fallback")):
        what_failed = "watchdog_recovery"
    elif bool((watchdog.get("signals") or {}).get("no_progress")):
        what_failed = "no_progress_watchdog"
    elif bool(dom_has_quiz) and not bool(screen_has_quiz):
        what_failed = "screen_projection_mismatch"

    if not bool(allowlist.get("is_allowed", True)) and allowlist:
        why_failed = f"host_not_allowed:{allowlist.get('current_host') or 'unknown_host'}"
        next_best_action = "return_to_allowed_host_or_stop"
        confidence = "high"
    elif bool(secret_source.get("is_blocking")):
        why_failed = str(secret_source.get("reason") or error or "secret_source_blocked")
        next_best_action = "review_secret_source_policy"
        confidence = "high"
    elif not bool(checklist.get("is_allowed", True)) and checklist:
        why_failed = str(checklist.get("reason") or error or "pre_submit_check_failed")
        next_best_action = "review_pre_submit_checklist"
        confidence = "high"
    elif not bool(submit_confirm.get("is_allowed", True)) and submit_confirm:
        why_failed = str(submit_confirm.get("reason") or error or "submit_confirmation_required")
        next_best_action = "request_operator_confirmation"
        confidence = "high"
    elif bool(hard_stop.get("is_blocking")):
        why_failed = str(hard_stop.get("error") or hard_stop.get("reason") or error or "hard_stop_guard")
        next_best_action = "inspect_runtime_guards"
        confidence = "high"
    elif bool(anti_modal.get("is_blocking")):
        why_failed = str(anti_modal.get("reason") or error or "anti_modal_guard")
        next_best_action = "pause_and_review_modal"
        confidence = "high"
    elif bool(submit.get("is_blocking")):
        why_failed = str(submit.get("stop_reason") or error or "submit_guard")
        next_best_action = "avoid_submit_and_review_payload"
        confidence = "high"
    elif bool((watchdog.get("signals") or {}).get("should_abort")):
        why_failed = str(watchdog.get("reason") or recovery.get("reason") or error or "watchdog_abort")
        next_best_action = "escalate_to_operator_or_lead"
        confidence = "high"
    elif bool((watchdog.get("signals") or {}).get("should_use_dom_fallback")):
        why_failed = str(watchdog.get("reason") or recovery.get("reason") or "dom_fallback_requested")
        next_best_action = "retry_with_fallback_or_capture_more_evidence"
    elif bool((watchdog.get("signals") or {}).get("should_retry_same_action")):
        why_failed = str(watchdog.get("reason") or recovery.get("reason") or "retry_same_action")
        next_best_action = "retry_current_action_once"
    elif bool((watchdog.get("signals") or {}).get("no_progress")) or bool((watchdog.get("signals") or {}).get("is_stuck")):
        why_failed = str(watchdog.get("reason") or "no_progress_watchdog")
        next_best_action = "inspect_click_target_and_readback"
    elif str(stop_reason or "") == "screen_wrong_tab":
        why_failed = "dom_points_to_quiz_but_visible_screen_does_not_match"
        next_best_action = "refocus_expected_tab"
        confidence = "high"
    elif str(stop_reason or "") == "brain_no_action":
        why_failed = "screen_and_dom_are_present_but_runtime_produced_no_action"
        next_best_action = "inspect_brain_state_and_action_planner"
    elif str(stop_reason or "") == "recorder_not_ready":
        why_failed = "controls_projection_missing_for_current_session"
        next_best_action = "check_recorder_health"
        confidence = "high"
    elif str(stop_reason or "") == "region_grow_empty":
        why_failed = "screen_is_fresh_but_region_grow_returned_no_results"
        next_best_action = "inspect_ocr_pipeline"
        confidence = "high"

    evidence = {
        "error": str(error or ""),
        "stop_reason": str(stop_reason or ""),
        "watchdog_decision": str(watchdog.get("decision") or ""),
        "watchdog_reason": str(watchdog.get("reason") or ""),
        "secret_source_decision": str(secret_source.get("decision") or ""),
        "secret_source_reason": str(secret_source.get("reason") or ""),
        "recovery_action": str((watchdog.get("signals") or {}).get("recovery_action") or ""),
        "consecutive_no_progress_count": int((watchdog.get("signals") or {}).get("consecutive_no_progress_count") or 0),
        "attempt_count": int((watchdog.get("signals") or {}).get("attempt_count") or 0),
        "current_host": str(allowlist.get("current_host") or ""),
        "submit_guard_reason": str(submit.get("stop_reason") or ""),
        "hard_stop_source": str(hard_stop.get("source") or ""),
        "hard_stop_error": str(hard_stop.get("error") or ""),
        "dom_has_quiz": None if dom_has_quiz is None else bool(dom_has_quiz),
        "screen_has_quiz": None if screen_has_quiz is None else bool(screen_has_quiz),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "what_failed": what_failed,
        "why_failed": why_failed,
        "next_best_action": next_best_action,
        "confidence": confidence,
        "evidence": evidence,
    }
