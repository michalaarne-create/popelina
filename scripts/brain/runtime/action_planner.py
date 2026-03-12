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


def _append_next_after_answer(actions: List[QuizAction], next_bbox: Any, *, reason: str = "click_next_after_answer") -> bool:
    if isinstance(next_bbox, list) and len(next_bbox) == 4:
        actions.append(QuizAction(kind="wait", amount=120, reason="wait_before_next"))
        actions.append(QuizAction(kind="screen_click", bbox=[float(v) for v in next_bbox], reason=reason))
        return True
    return False


def _to_operational_type(qtype: str) -> str:
    t = str(qtype or "").strip().lower()
    if t == "text":
        return "text"
    if t in {"dropdown", "dropdown_scroll"}:
        return t
    return "choice"


def plan_actions(
    *,
    screen_state: Dict[str, Any],
    resolved_answer: ResolvedQuizAnswer,
    brain_state: Dict[str, Any],
    controls_data: Optional[Dict[str, Any]] = None,
    transition: Optional[Dict[str, Any]] = None,
) -> Tuple[List[QuizAction], Dict[str, Any], bool]:
    active_sig = str(screen_state.get("active_question_signature") or "")
    control_kind = _to_operational_type(
        str(
            screen_state.get("detected_operational_type")
            or resolved_answer.question_type
            or screen_state.get("control_kind")
            or "single"
        )
    )
    next_bbox = screen_state.get("next_bbox")
    answer_ready = _control_state_matches(resolved_answer, controls_data)
    if transition and transition.get("success") and transition.get("previous_action_kind") in {"answer", "dropdown", "type"}:
        answer_ready = True

    trace: Dict[str, Any] = {
        "active_question_signature": active_sig,
        "control_kind": control_kind,
        "answer_ready": answer_ready,
        "screen_confidence": ((screen_state.get("confidence") or {}).get("screen") or 0.0),
        "fallback_used": False,
        "screen_only_targets": True,
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

    # Chain policy (single/multi/text/dropdown):
    # after answer-like action, if Next is missing and QA text changed enough,
    # assume auto-next and do nothing else in this iteration.
    if (not next_bbox) and isinstance(transition, dict):
        prev_kind = str(transition.get("previous_action_kind") or "")
        quiz_type = str(screen_state.get("detected_quiz_type") or "")
        answer_like_prev = prev_kind in {"answer", "dropdown", "type"}
        supported_type = quiz_type in {"single", "multi", "dropdown", "dropdown_scroll", "text"}
        if answer_like_prev and supported_type and bool(transition.get("qa_changed_30")):
            ratio = float(transition.get("qa_change_ratio") or 0.0)
            actions.append(
                QuizAction(
                    kind="noop",
                    reason=f"auto_next_detected_no_next:{quiz_type}:qa_delta={ratio:.2f}",
                )
            )
            trace["stage"] = "auto_next"
            trace["auto_next"] = True
            trace["qa_change_ratio"] = ratio
            trace["auto_next_quiz_type"] = quiz_type
            return actions, trace, False

    if control_kind == "text":
        input_bbox = screen_state.get("input_bbox")
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
            _append_next_after_answer(actions, next_bbox)
            trace["stage"] = "answer"
            return actions, trace, bool(trace["fallback_used"])
        if screen_state.get("scroll_needed"):
            actions.append(QuizAction(kind="screen_scroll", bbox=screen_state.get("content_column"), direction="down", amount=3, reason="screen_scroll_for_textbox"))
            trace["stage"] = "scroll"
            return actions, trace, False
        actions.append(QuizAction(kind="noop", reason="missing_screen_textbox_bbox"))
        trace["stage"] = "blocked"
        return actions, trace, False

    if control_kind in {"dropdown", "dropdown_scroll"}:
        select_bbox = screen_state.get("select_bbox")
        option_index = resolved_answer.option_indexes[0] if resolved_answer.option_indexes else 0
        if select_bbox:
            actions.append(QuizAction(kind="screen_click", bbox=[float(v) for v in select_bbox], reason="focus_select"))
            actions.append(QuizAction(kind="key_press", combo="home", reason="select_to_first"))
            if option_index > 0:
                actions.append(QuizAction(kind="key_repeat", combo="down", repeat=int(option_index), reason="move_to_option"))
            actions.append(QuizAction(kind="key_press", combo="enter", reason="confirm_select"))
            _append_next_after_answer(actions, next_bbox)
            trace["stage"] = "answer"
            return actions, trace, bool(trace["fallback_used"])
        if screen_state.get("scroll_needed"):
            actions.append(QuizAction(kind="screen_scroll", bbox=screen_state.get("content_column"), direction="down", amount=3, reason="screen_scroll_for_select"))
            trace["stage"] = "scroll"
            return actions, trace, False
        actions.append(QuizAction(kind="noop", reason="missing_screen_select_bbox"))
        trace["stage"] = "blocked"
        return actions, trace, False

    target_answers = resolved_answer.correct_answers or []
    if not target_answers:
        actions.append(QuizAction(kind="noop", reason="no_target_answers"))
        return actions, trace, False
    for target_text in target_answers:
        bbox = _best_option_bbox(screen_state, target_text)
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
            actions.append(QuizAction(kind="noop", reason=f"missing_screen_option_bbox:{target_text}"))
            trace["stage"] = "blocked"
            return actions, trace, False
        actions.append(QuizAction(kind="screen_click", bbox=bbox, reason=f"click_answer:{target_text}"))
    _append_next_after_answer(actions, next_bbox)
    trace["stage"] = "answer"
    return actions, trace, bool(trace["fallback_used"])
