from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .quiz_types import QuizAction, ResolvedQuizAnswer
from .quiz_utils import normalize_match_text, text_similarity


def _attempt_count(state: Dict[str, Any], key: str) -> int:
    attempts = state.get("attempt_counts") or {}
    try:
        return int(attempts.get(key) or 0)
    except Exception:
        return 0


def _control_state_matches(answer: ResolvedQuizAnswer, controls_data: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(controls_data, dict):
        return False
    expected = [normalize_match_text(value) for value in answer.correct_answers if normalize_match_text(value)]
    if not expected:
        return False
    for control in controls_data.get("controls") or []:
        if not isinstance(control, dict):
            continue
        current = normalize_match_text(control.get("text") or control.get("value") or "")
        if not current:
            continue
        if bool(control.get("checked") or control.get("selected")) and any(text_similarity(current, exp) >= 0.88 for exp in expected):
            return True
        if str(control.get("kind") or "") in {"textbox", "select"} and str(control.get("value") or "").strip():
            if any(text_similarity(normalize_match_text(control.get("value") or ""), exp) >= 0.88 for exp in expected):
                return True
    return False


def _best_option_bbox(screen_state: Dict[str, Any], target_text: str) -> Optional[List[float]]:
    best_bbox = None
    best_score = 0.0
    for option in screen_state.get("options") or []:
        if not isinstance(option, dict):
            continue
        score = text_similarity(target_text, option.get("text") or "")
        if score > best_score:
            best_score = score
            best_bbox = option.get("bbox")
    if best_score >= 0.82 and isinstance(best_bbox, list) and len(best_bbox) == 4:
        return [float(v) for v in best_bbox]
    return None


def _controls_bbox_for_answer(controls_data: Optional[Dict[str, Any]], target_text: str) -> Optional[List[float]]:
    if not isinstance(controls_data, dict):
        return None
    best_bbox = None
    best_score = 0.0
    for control in controls_data.get("controls") or []:
        if not isinstance(control, dict):
            continue
        current = control.get("text") or control.get("value") or ""
        score = text_similarity(target_text, current)
        bbox = control.get("bbox")
        if score > best_score and isinstance(bbox, list) and len(bbox) == 4:
            best_score = score
            best_bbox = bbox
    if best_score >= 0.82:
        return [float(v) for v in best_bbox]
    return None


def _controls_next_bbox(controls_data: Optional[Dict[str, Any]]) -> Optional[List[float]]:
    if not isinstance(controls_data, dict):
        return None
    next_id = str(((controls_data.get("next_control_id") or "") or "")).strip()
    for control in controls_data.get("controls") or []:
        if not isinstance(control, dict):
            continue
        if next_id and str(control.get("id") or "") == next_id:
            bbox = control.get("bbox")
            if isinstance(bbox, list) and len(bbox) == 4:
                return [float(v) for v in bbox]
        text = str(control.get("text") or "")
        if any(token in text.lower() for token in ("nast", "next", "dalej")):
            bbox = control.get("bbox")
            if isinstance(bbox, list) and len(bbox) == 4:
                return [float(v) for v in bbox]
    return None


def plan_actions(
    *,
    screen_state: Dict[str, Any],
    resolved_answer: ResolvedQuizAnswer,
    brain_state: Dict[str, Any],
    controls_data: Optional[Dict[str, Any]] = None,
    transition: Optional[Dict[str, Any]] = None,
) -> Tuple[List[QuizAction], Dict[str, Any], bool]:
    active_sig = str(screen_state.get("active_question_signature") or "")
    control_kind = str(resolved_answer.question_type or screen_state.get("control_kind") or "single")
    next_bbox = screen_state.get("next_bbox")
    if next_bbox is None:
        next_bbox = _controls_next_bbox(controls_data)
    answer_ready = _control_state_matches(resolved_answer, controls_data)
    if transition and transition.get("success") and transition.get("previous_action_kind") in {"answer", "dropdown", "type"}:
        answer_ready = True

    trace: Dict[str, Any] = {
        "active_question_signature": active_sig,
        "control_kind": control_kind,
        "answer_ready": answer_ready,
        "screen_confidence": ((screen_state.get("confidence") or {}).get("screen") or 0.0),
        "fallback_used": False,
    }
    actions: List[QuizAction] = []

    if not resolved_answer.matched:
        if screen_state.get("scroll_needed"):
            bbox = screen_state.get("content_column")
            actions.append(QuizAction(kind="screen_scroll", bbox=bbox, direction="down", amount=4, reason="screen_scroll_for_unresolved"))
        else:
            actions.append(QuizAction(kind="noop", reason="unresolved_question"))
        return actions, trace, False

    if next_bbox and answer_ready:
        actions.append(QuizAction(kind="screen_click", bbox=[float(v) for v in next_bbox], reason="click_next"))
        trace["stage"] = "next"
        return actions, trace, False

    if control_kind == "text":
        input_bbox = screen_state.get("input_bbox")
        if not input_bbox and isinstance(controls_data, dict):
            for control in controls_data.get("controls") or []:
                if str(control.get("kind") or "") == "textbox" and isinstance(control.get("bbox"), list):
                    input_bbox = control.get("bbox")
                    trace["fallback_used"] = True
                    break
        if input_bbox:
            answer_text = resolved_answer.correct_answers[0] if resolved_answer.correct_answers else ""
            actions.extend(
                [
                    QuizAction(kind="screen_click", bbox=[float(v) for v in input_bbox], reason="focus_textbox"),
                    QuizAction(kind="key_press", combo="ctrl+a", reason="select_all_text"),
                    QuizAction(kind="key_press", combo="backspace", reason="clear_textbox"),
                    QuizAction(kind="type_text", text=answer_text, reason="type_answer"),
                    QuizAction(kind="key_press", combo="enter", reason="submit_text_answer"),
                ]
            )
            trace["stage"] = "answer"
            return actions, trace, bool(trace["fallback_used"])
        actions.append(QuizAction(kind="noop", reason="missing_textbox_bbox"))
        trace["stage"] = "blocked"
        return actions, trace, True

    if control_kind in {"dropdown", "dropdown_scroll"}:
        select_bbox = screen_state.get("select_bbox")
        if not select_bbox and isinstance(controls_data, dict):
            for control in controls_data.get("controls") or []:
                if str(control.get("kind") or "") == "select" and isinstance(control.get("bbox"), list):
                    select_bbox = control.get("bbox")
                    trace["fallback_used"] = True
                    break
        option_index = resolved_answer.option_indexes[0] if resolved_answer.option_indexes else 0
        if select_bbox:
            actions.append(QuizAction(kind="screen_click", bbox=[float(v) for v in select_bbox], reason="focus_select"))
            actions.append(QuizAction(kind="key_press", combo="home", reason="select_to_first"))
            if option_index > 0:
                actions.append(QuizAction(kind="key_repeat", combo="down", repeat=int(option_index), reason="move_to_option"))
            actions.append(QuizAction(kind="key_press", combo="enter", reason="confirm_select"))
            trace["stage"] = "answer"
            return actions, trace, bool(trace["fallback_used"])
        actions.append(QuizAction(kind="noop", reason="missing_select_bbox"))
        trace["stage"] = "blocked"
        return actions, trace, True

    target_answers = resolved_answer.correct_answers or []
    if not target_answers:
        actions.append(QuizAction(kind="noop", reason="no_target_answers"))
        return actions, trace, False
    for target_text in target_answers:
        bbox = _best_option_bbox(screen_state, target_text)
        used_dom = False
        if bbox is None:
            bbox = _controls_bbox_for_answer(controls_data, target_text)
            used_dom = bbox is not None
        if bbox is None:
            attempt_key = f"{active_sig}:{control_kind}:{normalize_match_text(target_text)}:scroll"
            if _attempt_count(brain_state, attempt_key) < 2 and screen_state.get("scroll_needed"):
                actions.append(
                    QuizAction(
                        kind="screen_scroll",
                        bbox=screen_state.get("content_column"),
                        direction="down",
                        amount=4,
                        reason=f"scroll_for_{target_text}",
                    )
                )
                trace["stage"] = "scroll"
                return actions, trace, bool(trace["fallback_used"])
            actions.append(QuizAction(kind="noop", reason=f"missing_option_bbox:{target_text}"))
            trace["stage"] = "blocked"
            return actions, trace, True
        trace["fallback_used"] = trace["fallback_used"] or used_dom
        actions.append(QuizAction(kind="screen_click", bbox=bbox, reason=f"click_answer:{target_text}"))
    trace["stage"] = "answer"
    return actions, trace, bool(trace["fallback_used"])
