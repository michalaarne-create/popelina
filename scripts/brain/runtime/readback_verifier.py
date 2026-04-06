from __future__ import annotations

from typing import Any, Dict, List, Optional

from scripts.pipeline.contracts import (
    SCHEMA_VERSION,
    build_before_after_pair_id,
    classify_transition_kind,
)

from .quiz_utils import normalize_match_text, text_similarity


def _selected_values(controls_data: Optional[Dict[str, Any]]) -> List[str]:
    values: List[str] = []
    if not isinstance(controls_data, dict):
        return values
    for control in controls_data.get("controls") or []:
        if not isinstance(control, dict):
            continue
        kind = str(control.get("kind") or "")
        text = str(control.get("text") or control.get("value") or "")
        if kind in {"radio", "checkbox", "option"} and bool(control.get("checked") or control.get("selected")):
            if text:
                values.append(text)
        if kind in {"select", "textbox"}:
            value = str(control.get("value") or "")
            if value:
                values.append(value)
    return values


def _values_match(expected: List[str], actual: List[str]) -> bool:
    remaining_actual = [normalize_match_text(value) for value in actual if normalize_match_text(value)]
    normalized_expected = [normalize_match_text(value) for value in expected if normalize_match_text(value)]
    if not normalized_expected:
        return False
    if len(normalized_expected) != len(remaining_actual):
        return False
    for target in normalized_expected:
        match_index = next(
            (idx for idx, candidate in enumerate(remaining_actual) if text_similarity(target, candidate) >= 0.88),
            None,
        )
        if match_index is None:
            return False
        remaining_actual.pop(match_index)
    return not remaining_actual


def _qa_text_from_prev_state(prev_state: Dict[str, Any]) -> str:
    if not isinstance(prev_state, dict):
        return ""
    parts: List[str] = []
    q = str(prev_state.get("question_text") or "").strip()
    if q:
        parts.append(q)
    objs = prev_state.get("objects") if isinstance(prev_state.get("objects"), dict) else {}
    answers = objs.get("answers") if isinstance(objs.get("answers"), list) else []
    for row in answers:
        if not isinstance(row, dict):
            continue
        t = str(row.get("text") or "").strip()
        if t:
            parts.append(t)
    return "\n".join(parts).strip()


def _qa_text_from_screen_state(screen_state: Dict[str, Any]) -> str:
    if not isinstance(screen_state, dict):
        return ""
    parts: List[str] = []
    q = str(screen_state.get("question_text") or "").strip()
    if q:
        parts.append(q)
    options = screen_state.get("options") if isinstance(screen_state.get("options"), list) else []
    for row in options:
        if not isinstance(row, dict):
            continue
        t = str(row.get("text") or "").strip()
        if t:
            parts.append(t)
    return "\n".join(parts).strip()


def _qa_change_ratio(prev_text: str, cur_text: str) -> float:
    a = normalize_match_text(prev_text or "")
    b = normalize_match_text(cur_text or "")
    if (not a) and (not b):
        return 0.0
    if not a or not b:
        return 1.0
    sim = text_similarity(a, b)
    try:
        return max(0.0, min(1.0, 1.0 - float(sim)))
    except Exception:
        return 0.0


def _transition_state_snapshot(
    *,
    screen_signature: str,
    question_signature: str,
    page_signature: str,
    question_text: str,
    qa_text: str,
    selected_values: List[str],
) -> Dict[str, Any]:
    return {
        "screen_signature": str(screen_signature or ""),
        "question_signature": str(question_signature or ""),
        "page_signature": str(page_signature or ""),
        "question_text": str(question_text or ""),
        "qa_text": str(qa_text or ""),
        "selected_values": [str(value) for value in selected_values if str(value or "").strip()],
    }


def _bbox_center(bbox: List[Any]) -> Optional[tuple[float, float]]:
    if not isinstance(bbox, list) or len(bbox) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    except Exception:
        return None
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _build_success_semantics_policy(
    *,
    previous_action_kind: str,
    chain_type: str,
    values_match: bool,
    question_changed: bool,
    page_changed: bool,
    failure: bool,
) -> Dict[str, Any]:
    prev_kind = str(previous_action_kind or "").strip().lower()
    chain = str(chain_type or "").strip().lower()
    compound_chain = chain == "triple_block" or chain.startswith("mixed_block:")
    selection_persisted = bool(values_match and prev_kind in {"answer", "dropdown", "type"})
    compound_block_completed = bool(compound_chain and prev_kind in {"answer", "dropdown", "type"} and values_match)
    question_advanced = bool(question_changed or page_changed)
    submit_confirmed = bool(prev_kind == "submit" and question_advanced)
    outcome_type = "no_progress"
    if submit_confirmed:
        outcome_type = "submit_confirmed"
    elif compound_block_completed:
        outcome_type = "compound_block_completed"
    elif question_advanced:
        outcome_type = "question_advanced"
    elif selection_persisted:
        outcome_type = "selection_persisted"
    elif failure:
        outcome_type = "failed_progress"
    return {
        "schema_version": SCHEMA_VERSION,
        "selection_persisted": selection_persisted,
        "compound_block_completed": compound_block_completed,
        "question_advanced": question_advanced,
        "submit_confirmed": submit_confirmed,
        "outcome_type": outcome_type,
        "semantic_success": bool(compound_block_completed or question_advanced or submit_confirmed),
        "scope": (
            "compound_block"
            if compound_block_completed
            else ("question_completion" if (question_advanced or submit_confirmed) else ("selection_only" if selection_persisted else "none"))
        ),
    }


def _build_target_scope_guard(
    *,
    target_bbox: List[Any],
    active_block_bbox: List[Any],
) -> Dict[str, Any]:
    target_center = _bbox_center(target_bbox)
    block_center = _bbox_center(active_block_bbox)
    if target_center is None or block_center is None:
        return {
            "schema_version": SCHEMA_VERSION,
            "has_scope": False,
            "in_scope": True,
            "reason": "missing_scope_evidence",
        }
    x, y = target_center
    x1, y1, x2, y2 = [float(v) for v in active_block_bbox[:4]]
    in_scope = bool(x1 <= x <= x2 and y1 <= y <= y2)
    return {
        "schema_version": SCHEMA_VERSION,
        "has_scope": True,
        "in_scope": in_scope,
        "reason": "target_within_active_block" if in_scope else "target_outside_active_block",
        "target_center": [round(x, 3), round(y, 3)],
        "active_block_bbox": [round(float(v), 3) for v in active_block_bbox[:4]],
    }


def _build_visual_progress_guard(
    *,
    previous_action_kind: str,
    values_match: bool,
    question_changed: bool,
    page_changed: bool,
    qa_changed_30: bool,
) -> Dict[str, Any]:
    prev_kind = str(previous_action_kind or "").strip().lower()
    answer_like_action = prev_kind in {"answer", "dropdown", "type", "next"}
    visual_only_progress = bool(
        answer_like_action
        and qa_changed_30
        and not question_changed
        and not page_changed
        and not values_match
    )
    reason = "no_visual_progress_issue"
    if visual_only_progress:
        reason = "rerender_without_logical_form_progress"
    return {
        "schema_version": SCHEMA_VERSION,
        "is_visual_only_progress": visual_only_progress,
        "reason": reason,
        "signals": {
            "answer_like_action": answer_like_action,
            "values_match": bool(values_match),
            "question_changed": bool(question_changed),
            "page_changed": bool(page_changed),
            "qa_changed_30": bool(qa_changed_30),
        },
    }


def _build_commit_verifier(
    *,
    previous_action_kind: str,
    values_match: bool,
    question_changed: bool,
    page_changed: bool,
    qa_changed_30: bool,
) -> Dict[str, Any]:
    prev_kind = str(previous_action_kind or "").strip().lower()
    applies = prev_kind == "dropdown"
    committed = False
    reason = "not_applicable"
    if applies:
        committed = bool(values_match or question_changed or page_changed)
        if committed:
            reason = "readback_or_progress_confirmed"
        elif qa_changed_30:
            reason = "dropdown_label_changed_without_commit"
        else:
            reason = "missing_dropdown_commit_signal"
    return {
        "schema_version": SCHEMA_VERSION,
        "applies": applies,
        "committed": committed,
        "reason": reason,
        "signals": {
            "values_match": bool(values_match),
            "question_changed": bool(question_changed),
            "page_changed": bool(page_changed),
            "qa_changed_30": bool(qa_changed_30),
        },
    }


def evaluate_transition(
    *,
    prev_state: Dict[str, Any],
    current_screen_state: Dict[str, Any],
    controls_data: Optional[Dict[str, Any]] = None,
    page_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    prev_action = (prev_state.get("last_action") or {}) if isinstance(prev_state, dict) else {}
    prev_kind = str(prev_action.get("kind") or "")
    current_question_sig = str(current_screen_state.get("active_question_signature") or "")
    prev_question_sig = str(prev_action.get("question_signature") or "")
    current_page_sig = str((page_data or {}).get("page_signature") or current_screen_state.get("page_signature") or "")
    prev_page_sig = str(prev_action.get("page_signature") or "")
    current_values = _selected_values(controls_data)
    expected_values = [str(v) for v in prev_action.get("expected_values") or [] if str(v)]
    values_match = _values_match(expected_values, current_values) if expected_values else False
    question_changed = bool(prev_question_sig and current_question_sig and current_question_sig != prev_question_sig)
    page_changed = bool(prev_page_sig and current_page_sig and current_page_sig != prev_page_sig)
    same_screen = str(prev_state.get("last_screen_signature") or "") == str(current_screen_state.get("screen_signature") or "")
    prev_qa_text = _qa_text_from_prev_state(prev_state)
    cur_qa_text = _qa_text_from_screen_state(current_screen_state)
    qa_change_ratio = _qa_change_ratio(prev_qa_text, cur_qa_text)
    qa_changed_30 = bool(qa_change_ratio >= 0.30)

    success = False
    if prev_kind == "answer":
        success = values_match or question_changed or page_changed or qa_changed_30
    elif prev_kind == "next":
        success = question_changed or page_changed
    elif prev_kind in {"type", "dropdown"}:
        success = values_match or question_changed or page_changed

    failure = False
    if prev_kind in {"answer", "next", "type", "dropdown"} and not success:
        failure = bool(prev_question_sig and current_question_sig == prev_question_sig and same_screen)
    action_id = str(prev_action.get("action_id") or "")
    chain_type = str((prev_action.get("post_action_expectation") or {}).get("chain_type") or "")
    previous_screen_signature = str(prev_state.get("last_screen_signature") or "")
    current_screen_signature = str(current_screen_state.get("screen_signature") or "")
    target_bbox = list(prev_action.get("target_bbox") or [])
    active_block_bbox = list(prev_action.get("active_block_bbox") or [])
    transition_kind = classify_transition_kind(
        previous_action_kind=prev_kind,
        question_changed=question_changed,
        page_changed=page_changed,
        values_match=values_match,
        qa_changed_30=qa_changed_30,
        same_screen=same_screen,
        failure=failure,
    )
    before_state = _transition_state_snapshot(
        screen_signature=previous_screen_signature,
        question_signature=prev_question_sig,
        page_signature=prev_page_sig,
        question_text=str(prev_state.get("question_text") or ""),
        qa_text=prev_qa_text,
        selected_values=expected_values,
    )
    after_state = _transition_state_snapshot(
        screen_signature=current_screen_signature,
        question_signature=current_question_sig,
        page_signature=current_page_sig,
        question_text=str(current_screen_state.get("question_text") or ""),
        qa_text=cur_qa_text,
        selected_values=current_values,
    )
    success_semantics_policy = _build_success_semantics_policy(
        previous_action_kind=prev_kind,
        chain_type=chain_type,
        values_match=values_match,
        question_changed=question_changed,
        page_changed=page_changed,
        failure=failure,
    )
    visual_progress_guard = _build_visual_progress_guard(
        previous_action_kind=prev_kind,
        values_match=values_match,
        question_changed=question_changed,
        page_changed=page_changed,
        qa_changed_30=qa_changed_30,
    )
    target_scope_guard = _build_target_scope_guard(
        target_bbox=target_bbox,
        active_block_bbox=active_block_bbox,
    )
    commit_verifier = _build_commit_verifier(
        previous_action_kind=prev_kind,
        values_match=values_match,
        question_changed=question_changed,
        page_changed=page_changed,
        qa_changed_30=qa_changed_30,
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "transition_kind": transition_kind,
        "action_id": action_id,
        "target_instance_id": str(prev_action.get("target_instance_id") or ""),
        "expectation_id": str((prev_action.get("post_action_expectation") or {}).get("expectation_id") or ""),
        "chain_type": chain_type,
        "before_after_pair_id": build_before_after_pair_id(
            action_id=action_id,
            previous_screen_signature=previous_screen_signature,
            current_screen_signature=current_screen_signature,
            current_page_signature=current_page_sig,
        ),
        "previous_action_kind": prev_kind,
        "expected_values": expected_values,
        "current_values": current_values,
        "values_match": values_match,
        "question_changed": question_changed,
        "page_changed": page_changed,
        "qa_change_ratio": round(float(qa_change_ratio), 4),
        "qa_changed_30": qa_changed_30,
        "same_screen": same_screen,
        "before_state": before_state,
        "after_state": after_state,
        "target_text": str(prev_action.get("target_text") or ""),
        "target_bbox": target_bbox,
        "target_scope_guard": target_scope_guard,
        "success": success,
        "outcome_type": str(success_semantics_policy.get("outcome_type") or ""),
        "semantic_success": bool(success_semantics_policy.get("semantic_success")),
        "semantic_scope": str(success_semantics_policy.get("scope") or ""),
        "success_semantics_policy": success_semantics_policy,
        "visual_progress_guard": visual_progress_guard,
        "commit_verifier": commit_verifier,
        "failure": failure,
    }
