from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .action_planner import plan_actions
from .answer_resolver import build_answer_source_policy, resolve_answer
from .quiz_types import QuizAction, ResolvedQuizAnswer
from .readback_verifier import evaluate_transition


def first_action_to_legacy(actions: List[Dict[str, Any]]) -> str:
    if not actions:
        return "idle"
    first = actions[0] if isinstance(actions[0], dict) else {}
    kind = str(first.get("kind") or "")
    reason = str(first.get("reason") or "")
    if kind == "screen_click":
        return "click_next" if "next" in reason else "click_answer"
    if kind == "screen_scroll":
        return "scroll_page_down"
    return "idle"


@dataclass
class DecisionCoreResult:
    transition: Dict[str, Any]
    resolved: ResolvedQuizAnswer
    actions_objs: List[QuizAction]
    actions: List[Dict[str, Any]]
    trace: Dict[str, Any]
    fallback_used: bool
    legacy_action: str


def build_decision_core(
    *,
    cache_path: Path,
    screen_state: Dict[str, Any],
    prev_state: Optional[Dict[str, Any]] = None,
    controls_data: Optional[Dict[str, Any]] = None,
    page_data: Optional[Dict[str, Any]] = None,
) -> DecisionCoreResult:
    prev_state = prev_state or {}
    transition = evaluate_transition(
        prev_state=prev_state,
        current_screen_state=screen_state,
        controls_data=controls_data,
        page_data=page_data,
    )
    resolved = resolve_answer(
        cache_path=cache_path,
        screen_state=screen_state,
        controls_data=controls_data,
        page_url=str((page_data or {}).get("url") or ""),
        stable_site_key=str((page_data or {}).get("stable_site_key") or ""),
        canonical_site_id=str((page_data or {}).get("canonical_site_id") or ""),
    )
    actions_objs, trace, fallback_used = plan_actions(
        screen_state=screen_state,
        resolved_answer=resolved,
        brain_state=prev_state,
        controls_data=controls_data,
        transition=transition,
    )
    source_policy = build_answer_source_policy(
        page_url=str((page_data or {}).get("url") or ""),
        stable_site_key=str((page_data or {}).get("stable_site_key") or ""),
        canonical_site_id=str((page_data or {}).get("canonical_site_id") or ""),
    )
    trace["answer_source_trace"] = {
        "host": str(source_policy.get("host") or ""),
        "is_synthetic_host": bool(source_policy.get("is_synthetic_host")),
        "preferred_source": str(source_policy.get("preferred_source") or ""),
        "allowed_sources": list(source_policy.get("allowed_sources") or []),
        "domain_registry_ready": bool(source_policy.get("domain_registry_ready")),
        "domain_registry_quality_gate": dict(source_policy.get("domain_registry_quality_gate") or {}),
        "answer_source_release_manifest": dict(source_policy.get("answer_source_release_manifest") or {}),
        "effective_source": str(resolved.cache_item.get("answer_source_kind") or "unresolved"),
        "resolved_source": str(resolved.source or ""),
    }
    actions = [action.to_dict() for action in actions_objs]
    return DecisionCoreResult(
        transition=transition,
        resolved=resolved,
        actions_objs=actions_objs,
        actions=actions,
        trace=trace,
        fallback_used=bool(fallback_used),
        legacy_action=first_action_to_legacy(actions),
    )
