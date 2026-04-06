from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from scripts.pipeline.contracts import SCHEMA_VERSION, build_expectation_id

from .canonical_type_contract import canonical_operational_type
from .quiz_types import QuizAction, ResolvedQuizAnswer
from .quiz_utils import normalize_match_text, text_similarity


def _attempt_count(state: Dict[str, Any], key: str) -> int:
    attempts = state.get("attempt_counts") or {}
    try:
        return int(attempts.get(key) or 0)
    except Exception:
        return 0


def _to_operational_type(qtype: str) -> str:
    return canonical_operational_type(qtype)


def _matched_expected_values(answer: ResolvedQuizAnswer, controls_data: Optional[Dict[str, Any]]) -> set[str]:
    if not isinstance(controls_data, dict):
        return set()
    expected = [normalize_match_text(value) for value in answer.correct_answers if normalize_match_text(value)]
    if not expected:
        return set()
    matched_expected = set()
    for control in controls_data.get("controls") or []:
        if not isinstance(control, dict):
            continue
        current = normalize_match_text(control.get("text") or control.get("value") or "")
        if not current:
            continue
        if bool(control.get("checked") or control.get("selected")):
            for exp in expected:
                if text_similarity(current, exp) >= 0.88:
                    matched_expected.add(exp)
        if str(control.get("kind") or "") in {"textbox", "select"} and str(control.get("value") or "").strip():
            value_norm = normalize_match_text(control.get("value") or "")
            for exp in expected:
                if text_similarity(value_norm, exp) >= 0.88:
                    matched_expected.add(exp)
    return matched_expected


def _control_state_matches(answer: ResolvedQuizAnswer, controls_data: Optional[Dict[str, Any]]) -> bool:
    expected = [normalize_match_text(value) for value in answer.correct_answers if normalize_match_text(value)]
    if not expected:
        return False
    matched_expected = _matched_expected_values(answer, controls_data)
    return len(matched_expected) == len(set(expected))


def _idempotency_guard_blocks_next(brain_state: Dict[str, Any], screen_state: Dict[str, Any], *, answer_ready: bool, next_bbox: Any) -> bool:
    if not answer_ready or not (isinstance(next_bbox, list) and len(next_bbox) == 4):
        return False
    if not isinstance(brain_state, dict):
        return False
    last_action = brain_state.get("last_action") if isinstance(brain_state.get("last_action"), dict) else {}
    if str(last_action.get("kind") or "").strip().lower() != "next":
        return False
    expectation = last_action.get("post_action_expectation") if isinstance(last_action.get("post_action_expectation"), dict) else {}
    if not bool(expectation.get("expect_next_click")):
        return False
    previous_screen_signature = str(brain_state.get("last_screen_signature") or "")
    current_screen_signature = str(screen_state.get("screen_signature") or "")
    if not previous_screen_signature or previous_screen_signature != current_screen_signature:
        return False
    previous_question_signature = str(last_action.get("question_signature") or "")
    current_question_signature = str(screen_state.get("active_question_signature") or "")
    return bool(previous_question_signature and previous_question_signature == current_question_signature)


def _build_action_confidence(screen_state: Dict[str, Any], *, actions: List[QuizAction], control_kind: str) -> Dict[str, Any]:
    confidence = screen_state.get("confidence") if isinstance(screen_state.get("confidence"), dict) else {}
    merged_conf = float(confidence.get("merged") or confidence.get("screen") or 0.0)
    type_conf = float(screen_state.get("type_confidence") or 0.0)
    first_kind = actions[0].kind if actions else "noop"
    bbox_bonus = 0.08 if actions and isinstance(actions[0].bbox, list) and len(actions[0].bbox) == 4 else 0.0
    fused = max(0.0, min(1.0, (merged_conf * 0.65) + (type_conf * 0.25) + bbox_bonus))
    prefer_dom = fused < 0.58 and first_kind in {"screen_click", "type_text"} and control_kind in {"choice", "dropdown", "dropdown_scroll", "text"}
    return {
        "fused_action_confidence": round(fused, 4),
        "low_confidence_gate": {
            "threshold": 0.58,
            "prefer_dom_fallback": bool(prefer_dom),
            "mode": "dom_fallback_if_available" if prefer_dom else "screen_primary",
        },
    }


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


def _synthetic_choice_bbox(screen_state: Dict[str, Any], answer: ResolvedQuizAnswer, target_text: str) -> Optional[List[float]]:
    """
    Fallback for screen-only cases where OCR misses very short options (e.g. 3/7/8/9),
    but the overall vertical answer layout is still obvious from prompt -> next spacing.
    """
    option_index = -1
    if answer.option_indexes:
        option_index = int(answer.option_indexes[0])
    elif target_text:
        options_map = answer.cache_item.get("options_text") if isinstance(answer.cache_item, dict) else {}
        if isinstance(options_map, dict):
            values = [normalize_match_text(v) for v in options_map.values()]
            target_norm = normalize_match_text(target_text)
            for idx, value in enumerate(values):
                if value and value == target_norm:
                    option_index = idx
                    break
    if option_index < 0:
        return None

    cache_item = answer.cache_item if isinstance(answer.cache_item, dict) else {}
    options_map = cache_item.get("options_text") if isinstance(cache_item.get("options_text"), dict) else {}
    option_count = len(options_map) if options_map else 0
    if option_count <= 0:
        option_count = max(option_index + 1, len(screen_state.get("options") or []), 4)

    prompt_bbox = screen_state.get("active_block", {}).get("bbox") or screen_state.get("active_block", {}).get("cluster_bbox") or screen_state.get("active_block", {}).get("prompt_bbox") or []
    if not (isinstance(prompt_bbox, list) and len(prompt_bbox) == 4):
        prompt_bbox = (screen_state.get("active_block") or {}).get("bbox") or []
    next_bbox = screen_state.get("next_bbox") or []
    content_column = screen_state.get("content_column") or []
    if not (isinstance(prompt_bbox, list) and len(prompt_bbox) == 4 and isinstance(content_column, list) and len(content_column) == 4):
        return None

    top_y = int(prompt_bbox[3]) + 8
    if isinstance(next_bbox, list) and len(next_bbox) == 4:
        bottom_y = int(next_bbox[1]) - 8
    else:
        screen_h = 1080
        try:
            screen_h = int(((screen_state.get("parser_debug") or {}).get("screen_size") or {}).get("height") or 1080)
        except Exception:
            screen_h = 1080
        estimated_block_h = max(140, int(option_count * 52))
        bottom_y = min(int(screen_h * 0.88), top_y + estimated_block_h)
    if (bottom_y - top_y) < max(80, option_count * 20):
        return None

    slot_h = max(20.0, float(bottom_y - top_y) / float(max(1, option_count)))
    center_y = top_y + (slot_h * option_index) + (slot_h / 2.0)
    y1 = max(float(top_y), center_y - min(18.0, slot_h * 0.35))
    y2 = min(float(bottom_y), center_y + min(18.0, slot_h * 0.35))
    x1 = float(content_column[0]) + 18.0
    x2 = min(float(content_column[2]) - 18.0, x1 + 260.0)
    if y2 <= y1 or x2 <= x1:
        return None
    return [x1, y1, x2, y2]


def _append_next_after_answer(actions: List[QuizAction], next_bbox: Any, *, reason: str = "click_next_after_answer") -> bool:
    if isinstance(next_bbox, list) and len(next_bbox) == 4:
        actions.append(QuizAction(kind="wait", amount=120, reason="wait_before_next"))
        actions.append(QuizAction(kind="screen_click", bbox=[float(v) for v in next_bbox], reason=reason))
        return True
    return False


def _append_dropdown_actions(
    actions: List[QuizAction],
    *,
    select_bbox: List[float],
    option_index: int,
    control_kind: str,
) -> None:
    # Native/faux quiz dropdowns often keep a placeholder row at index 0.
    # Cache option_indexes are based on real options, so shift by one step
    # from the list start after resetting to HOME.
    effective_index = max(0, int(option_index)) + 1
    actions.append(QuizAction(kind="screen_click", bbox=[float(v) for v in select_bbox], reason="focus_select"))
    actions.append(QuizAction(kind="wait", amount=80, reason="wait_after_focus_select"))
    actions.append(QuizAction(kind="key_press", combo="home", reason="select_to_first"))
    if control_kind == "dropdown_scroll":
        page_size = 8
        page_jumps = int(effective_index // page_size)
        tail_steps = int(effective_index % page_size)
        if page_jumps > 0:
            actions.append(QuizAction(kind="key_repeat", combo="pagedown", repeat=page_jumps, reason="scroll_select_page"))
        if tail_steps > 0:
            actions.append(QuizAction(kind="key_repeat", combo="down", repeat=tail_steps, reason="move_to_option_tail"))
        return
    if effective_index > 0:
        actions.append(QuizAction(kind="key_repeat", combo="down", repeat=int(effective_index), reason="move_to_option"))


def _part_option_bbox(part: Dict[str, Any], target_text: str) -> Optional[List[float]]:
    best_bbox = None
    best_score = 0.0
    for option in part.get("options") or []:
        if not isinstance(option, dict):
            continue
        score = text_similarity(target_text, option.get("text") or "")
        if score > best_score:
            best_score = score
            best_bbox = option.get("bbox")
    if best_score >= 0.82 and isinstance(best_bbox, list) and len(best_bbox) == 4:
        return [float(v) for v in best_bbox]
    return None


def _part_option_index(part: Dict[str, Any], target_text: str) -> int:
    best_index = -1
    best_score = 0.0
    for idx, option in enumerate(part.get("options") or []):
        if not isinstance(option, dict):
            continue
        score = text_similarity(target_text, option.get("text") or "")
        if score > best_score:
            best_score = score
            best_index = idx
    return best_index if best_score >= 0.82 else -1


def _plan_compound_actions(
    *,
    screen_state: Dict[str, Any],
    resolved_answer: ResolvedQuizAnswer,
    next_bbox: Any,
) -> Optional[Tuple[List[QuizAction], Dict[str, Any], bool]]:
    detected = str(screen_state.get("detected_quiz_type") or screen_state.get("active_block_type") or "").strip().lower()
    question_split = screen_state.get("question_split") if isinstance(screen_state.get("question_split"), dict) else {}
    parts = question_split.get("parts") if isinstance(question_split.get("parts"), list) else []
    answers = [str(value or "").strip() for value in resolved_answer.correct_answers if str(value or "").strip()]
    if detected not in {"mixed", "triple"} or len(parts) <= 1 or len(answers) < len(parts):
        return None

    actions: List[QuizAction] = []
    fallback_used = False
    fallback_reasons: List[str] = []
    for idx, part in enumerate(parts):
        if not isinstance(part, dict):
            return None
        control_kind = canonical_operational_type(str(part.get("control_kind") or "choice"))
        answer_text = answers[idx] if idx < len(answers) else ""
        if control_kind == "text":
            input_bbox = part.get("input_bbox")
            if not (isinstance(input_bbox, list) and len(input_bbox) == 4 and answer_text):
                return None
            actions.extend(
                [
                    QuizAction(kind="screen_click", bbox=[float(v) for v in input_bbox], reason=f"focus_textbox_part_{idx+1}"),
                    QuizAction(kind="key_press", combo="ctrl+a", reason=f"select_all_text_part_{idx+1}"),
                    QuizAction(kind="key_press", combo="backspace", reason=f"clear_textbox_part_{idx+1}"),
                    QuizAction(kind="type_text", text=answer_text, reason=f"type_answer_part_{idx+1}"),
                    QuizAction(kind="key_press", combo="tab", reason=f"blur_textbox_part_{idx+1}"),
                ]
            )
            continue
        if control_kind in {"dropdown", "dropdown_scroll"}:
            select_bbox = part.get("select_bbox") or part.get("input_bbox")
            if not (isinstance(select_bbox, list) and len(select_bbox) == 4 and answer_text):
                return None
            option_index = _part_option_index(part, answer_text)
            if option_index < 0:
                return None
            _append_dropdown_actions(
                actions,
                select_bbox=[float(v) for v in select_bbox],
                option_index=option_index,
                control_kind=control_kind,
            )
            actions.append(QuizAction(kind="key_press", combo="enter", reason=f"confirm_select_part_{idx+1}"))
            continue

        bbox = _part_option_bbox(part, answer_text)
        if bbox is None:
            return None
        actions.append(QuizAction(kind="screen_click", bbox=bbox, reason=f"click_answer_part_{idx+1}:{answer_text}"))

    _append_next_after_answer(actions, next_bbox, reason="click_next_after_compound_block")
    trace = {
        "stage": "answer",
        "compound_block": True,
        "compound_part_count": len(parts),
        "compound_answers": answers[: len(parts)],
        "fallback_used": fallback_used,
    }
    if fallback_reasons:
        trace["fallback_reasons"] = fallback_reasons
    return actions, trace, fallback_used


def _quiz_chain_type(screen_state: Dict[str, Any], control_kind: str) -> str:
    detected = str(screen_state.get("detected_quiz_type") or "")
    active_block_type = str(screen_state.get("active_block_type") or "")
    if detected == "triple" or active_block_type == "triple":
        return "triple_block"
    if detected == "mixed" or active_block_type == "mixed":
        return f"mixed_block:{control_kind}"
    if control_kind == "text":
        return "text_input"
    if control_kind in {"dropdown", "dropdown_scroll"}:
        return control_kind
    if detected == "multi":
        return "multi_choice"
    return "single_choice"


def _expectation(
    *,
    screen_state: Dict[str, Any],
    stage: str,
    control_kind: str,
    answer_ready: bool,
    actions: List[QuizAction],
    expected_values: Optional[List[str]] = None,
    auto_next: bool = False,
) -> Dict[str, Any]:
    first_kind = actions[0].kind if actions else "noop"
    expects_next_click = any(action.kind == "screen_click" and "next" in str(action.reason or "") for action in actions)
    expected_control_state = "selected"
    if control_kind == "text":
        expected_control_state = "typed"
    elif control_kind in {"dropdown", "dropdown_scroll"}:
        expected_control_state = "chosen"
    normalized_expected = [str(value or "").strip() for value in (expected_values or []) if str(value or "").strip()]
    expectation = {
        "schema_version": SCHEMA_VERSION,
        "expectation_id": build_expectation_id(
            question_signature=str(screen_state.get("active_question_signature") or ""),
            stage=stage,
            first_action_kind=first_kind,
            expected_values=normalized_expected,
        ),
        "stage": stage,
        "chain_type": _quiz_chain_type(screen_state, control_kind),
        "expected_first_action_kind": first_kind,
        "expected_control_state": expected_control_state,
        "expected_values": normalized_expected,
        "expect_answer_state_change": stage == "answer" and not answer_ready,
        "expect_next_click": expects_next_click,
        "expect_question_change": auto_next or expects_next_click,
        "expect_auto_next": auto_next,
        "has_next_target": bool(screen_state.get("next_bbox")),
        "has_select_target": bool(screen_state.get("select_bbox")),
        "has_input_target": bool(screen_state.get("input_bbox")),
    }
    return expectation


def plan_actions(
    *,
    screen_state: Dict[str, Any],
    resolved_answer: ResolvedQuizAnswer,
    brain_state: Dict[str, Any],
    controls_data: Optional[Dict[str, Any]] = None,
    transition: Optional[Dict[str, Any]] = None,
) -> Tuple[List[QuizAction], Dict[str, Any], bool]:
    expected_values = [str(value or "").strip() for value in resolved_answer.correct_answers if str(value or "").strip()]
    active_sig = str(screen_state.get("active_question_signature") or "")
    raw_kind = str(
        screen_state.get("detected_operational_type")
        or resolved_answer.question_type
        or screen_state.get("control_kind")
        or "single"
    )
    control_kind = canonical_operational_type(
        raw_kind
    )
    if (
        control_kind == "text"
        and resolved_answer.question_type in {"single", "multi", "triple", "mixed", "choice"}
        and resolved_answer.option_indexes
        and screen_state.get("next_bbox")
    ):
        control_kind = "choice"
        trace_override_kind = "choice_fallback_from_text"
    else:
        trace_override_kind = ""
    if (
        control_kind == "text"
        and resolved_answer.question_type in {"dropdown", "dropdown_scroll"}
        and (screen_state.get("input_bbox") or screen_state.get("next_bbox"))
    ):
        control_kind = str(resolved_answer.question_type)
        trace_override_kind = "dropdown_fallback_from_text"
    next_bbox = screen_state.get("next_bbox")
    answer_ready = _control_state_matches(resolved_answer, controls_data)
    if transition and transition.get("success") and transition.get("previous_action_kind") in {"answer", "dropdown", "type"}:
        answer_ready = True

    trace: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "active_question_signature": active_sig,
        "control_kind": control_kind,
        "chain_type": _quiz_chain_type(screen_state, control_kind),
        "answer_ready": answer_ready,
        "screen_confidence": ((screen_state.get("confidence") or {}).get("merged") or (screen_state.get("confidence") or {}).get("screen") or 0.0),
        "fallback_used": False,
        "screen_only_targets": True,
    }
    if trace_override_kind:
        trace["control_kind_override"] = trace_override_kind
    actions: List[QuizAction] = []

    def _finalize_trace() -> None:
        trace.update(_build_action_confidence(screen_state, actions=actions, control_kind=control_kind))

    compound_plan = _plan_compound_actions(
        screen_state=screen_state,
        resolved_answer=resolved_answer,
        next_bbox=next_bbox,
    )
    if compound_plan is not None:
        actions, compound_trace, compound_fallback_used = compound_plan
        trace["stage"] = str(compound_trace.get("stage") or "answer")
        trace["compound_block"] = bool(compound_trace.get("compound_block"))
        trace["compound_part_count"] = int(compound_trace.get("compound_part_count") or 0)
        trace["compound_answers"] = list(compound_trace.get("compound_answers") or [])
        trace["fallback_used"] = bool(compound_fallback_used)
        if compound_trace.get("fallback_reasons"):
            trace["fallback_reasons"] = list(compound_trace.get("fallback_reasons") or [])
        trace["post_action_expectation"] = _expectation(
            screen_state=screen_state,
            stage=str(compound_trace.get("stage") or "answer"),
            control_kind=control_kind,
            answer_ready=answer_ready,
            actions=actions,
            expected_values=expected_values,
        )
        _finalize_trace()
        return actions, trace, bool(trace["fallback_used"])

    if not resolved_answer.matched:
        if screen_state.get("scroll_needed"):
            bbox = screen_state.get("content_column")
            actions.append(QuizAction(kind="screen_scroll", bbox=bbox, direction="down", amount=4, reason="screen_scroll_for_unresolved"))
        else:
            actions.append(QuizAction(kind="noop", reason="unresolved_question"))
        trace["stage"] = "blocked"
        trace["post_action_expectation"] = _expectation(
            screen_state=screen_state,
            stage="blocked",
            control_kind=control_kind,
            answer_ready=answer_ready,
            actions=actions,
            expected_values=expected_values,
        )
        _finalize_trace()
        return actions, trace, False

    if next_bbox and answer_ready:
        if _idempotency_guard_blocks_next(brain_state, screen_state, answer_ready=answer_ready, next_bbox=next_bbox):
            actions.append(QuizAction(kind="noop", reason="idempotency_guard_next_same_state"))
            trace["stage"] = "blocked"
            trace["idempotency_guard"] = "next_same_state"
            trace["post_action_expectation"] = _expectation(
                screen_state=screen_state,
                stage="blocked",
                control_kind=control_kind,
                answer_ready=answer_ready,
                actions=actions,
                expected_values=expected_values,
            )
            _finalize_trace()
            return actions, trace, False
        actions.append(QuizAction(kind="screen_click", bbox=[float(v) for v in next_bbox], reason="click_next"))
        trace["stage"] = "next"
        trace["post_action_expectation"] = _expectation(
            screen_state=screen_state,
            stage="next",
            control_kind=control_kind,
            answer_ready=answer_ready,
            actions=actions,
            expected_values=expected_values,
        )
        _finalize_trace()
        return actions, trace, False

    # Chain policy (single/multi/text/dropdown):
    # after answer-like action, if Next is missing and QA text changed enough,
    # assume auto-next and do nothing else in this iteration.
    if (not next_bbox) and isinstance(transition, dict):
        prev_kind = str(transition.get("previous_action_kind") or "")
        quiz_type = str(screen_state.get("active_block_type") or screen_state.get("detected_quiz_type") or "")
        if quiz_type in {"triple", "mixed"}:
            quiz_type = str(screen_state.get("control_kind") or "single")
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
            trace["post_action_expectation"] = _expectation(
                screen_state=screen_state,
                stage="auto_next",
                control_kind=control_kind,
                answer_ready=answer_ready,
                actions=actions,
                expected_values=expected_values,
                auto_next=True,
            )
            _finalize_trace()
            return actions, trace, False

    if control_kind == "text":
        input_bbox = screen_state.get("input_bbox")
        if input_bbox:
            answer_text = resolved_answer.correct_answers[0] if resolved_answer.correct_answers else ""
            if not str(answer_text or "").strip():
                actions.append(QuizAction(kind="noop", reason="no_text_answer"))
                trace["stage"] = "blocked"
                trace["post_action_expectation"] = _expectation(
                    screen_state=screen_state,
                    stage="blocked",
                    control_kind=control_kind,
                    answer_ready=answer_ready,
                    actions=actions,
                    expected_values=expected_values,
                )
                _finalize_trace()
                return actions, trace, False
            if answer_ready:
                actions.append(QuizAction(kind="noop", reason="textbox_already_matches"))
                trace["stage"] = "blocked"
                trace["post_action_expectation"] = _expectation(
                    screen_state=screen_state,
                    stage="blocked",
                    control_kind=control_kind,
                    answer_ready=answer_ready,
                    actions=actions,
                    expected_values=expected_values,
                )
                _finalize_trace()
                return actions, trace, False
            actions.extend(
                [
                    QuizAction(kind="screen_click", bbox=[float(v) for v in input_bbox], reason="focus_textbox"),
                    QuizAction(kind="key_press", combo="ctrl+a", reason="select_all_text"),
                    QuizAction(kind="key_press", combo="backspace", reason="clear_textbox"),
                    QuizAction(kind="type_text", text=answer_text, reason="type_answer"),
                    QuizAction(kind="key_press", combo="tab", reason="blur_textbox"),
                ]
            )
            _append_next_after_answer(actions, next_bbox)
            trace["stage"] = "answer"
            trace["post_action_expectation"] = _expectation(
                screen_state=screen_state,
                stage="answer",
                control_kind=control_kind,
                answer_ready=answer_ready,
                actions=actions,
                expected_values=expected_values,
            )
            _finalize_trace()
            return actions, trace, bool(trace["fallback_used"])
        if screen_state.get("scroll_needed"):
            actions.append(QuizAction(kind="screen_scroll", bbox=screen_state.get("content_column"), direction="down", amount=3, reason="screen_scroll_for_textbox"))
            trace["stage"] = "scroll"
            trace["post_action_expectation"] = _expectation(
                screen_state=screen_state,
                stage="scroll",
                control_kind=control_kind,
                answer_ready=answer_ready,
                actions=actions,
                expected_values=expected_values,
            )
            _finalize_trace()
            return actions, trace, False
        actions.append(QuizAction(kind="noop", reason="missing_screen_textbox_bbox"))
        trace["stage"] = "blocked"
        trace["post_action_expectation"] = _expectation(
            screen_state=screen_state,
            stage="blocked",
            control_kind=control_kind,
            answer_ready=answer_ready,
            actions=actions,
            expected_values=expected_values,
        )
        _finalize_trace()
        return actions, trace, False

    if control_kind in {"dropdown", "dropdown_scroll"}:
        select_bbox = screen_state.get("select_bbox")
        if not select_bbox:
            input_bbox = screen_state.get("input_bbox")
            if isinstance(input_bbox, list) and len(input_bbox) == 4:
                select_bbox = input_bbox
                trace["fallback_used"] = True
                trace.setdefault("fallback_reasons", []).append("input_bbox_as_select_bbox")
        visible_screen_options = screen_state.get("options") if isinstance(screen_state.get("options"), list) else []
        target_answers = resolved_answer.correct_answers or []
        if not target_answers and not resolved_answer.option_indexes:
            actions.append(QuizAction(kind="noop", reason="no_dropdown_target_answers"))
            trace["stage"] = "blocked"
            trace["post_action_expectation"] = _expectation(
                screen_state=screen_state,
                stage="blocked",
                control_kind=control_kind,
                answer_ready=answer_ready,
                actions=actions,
                expected_values=expected_values,
            )
            _finalize_trace()
            return actions, trace, bool(trace["fallback_used"])
        visible_option_actions: List[QuizAction] = []
        if visible_screen_options and target_answers:
            for target_text in target_answers:
                bbox = _best_option_bbox(screen_state, target_text)
                if bbox is None:
                    continue
                visible_option_actions.append(
                    QuizAction(kind="screen_click", bbox=[float(v) for v in bbox], reason=f"click_visible_dropdown_option:{target_text}")
                )
            if visible_option_actions:
                actions.extend(visible_option_actions)
                _append_next_after_answer(actions, next_bbox)
                trace["fallback_used"] = True
                trace.setdefault("fallback_reasons", []).append("visible_dropdown_options_as_choice_targets")
                trace["stage"] = "answer"
                trace["post_action_expectation"] = _expectation(
                    screen_state=screen_state,
                    stage="answer",
                    control_kind=control_kind,
                    answer_ready=answer_ready,
                    actions=actions,
                    expected_values=expected_values,
                    auto_next=not bool(next_bbox),
                )
                _finalize_trace()
                return actions, trace, True
        if (not visible_screen_options) and target_answers:
            synthetic_actions: List[QuizAction] = []
            for target_text in target_answers:
                bbox = _synthetic_choice_bbox(screen_state, resolved_answer, target_text)
                if bbox is None:
                    synthetic_actions = []
                    break
                synthetic_actions.append(
                    QuizAction(kind="screen_click", bbox=[float(v) for v in bbox], reason=f"click_synthetic_dropdown_option:{target_text}")
                )
            if synthetic_actions:
                actions.extend(synthetic_actions)
                _append_next_after_answer(actions, next_bbox)
                trace["fallback_used"] = True
                trace.setdefault("fallback_reasons", []).append("synthetic_dropdown_option_targets")
                trace["stage"] = "answer"
                trace["post_action_expectation"] = _expectation(
                    screen_state=screen_state,
                    stage="answer",
                    control_kind=control_kind,
                    answer_ready=answer_ready,
                    actions=actions,
                    expected_values=expected_values,
                    auto_next=not bool(next_bbox),
                )
                _finalize_trace()
                return actions, trace, True
        option_index = resolved_answer.option_indexes[0] if resolved_answer.option_indexes else 0
        if select_bbox:
            _append_dropdown_actions(
                actions,
                select_bbox=[float(v) for v in select_bbox],
                option_index=int(option_index),
                control_kind=control_kind,
            )
            actions.append(QuizAction(kind="key_press", combo="enter", reason="confirm_select"))
            _append_next_after_answer(actions, next_bbox)
            trace["stage"] = "answer"
            trace["post_action_expectation"] = _expectation(
                screen_state=screen_state,
                stage="answer",
                control_kind=control_kind,
                answer_ready=answer_ready,
                actions=actions,
                expected_values=expected_values,
            )
            _finalize_trace()
            return actions, trace, bool(trace["fallback_used"])
        if screen_state.get("scroll_needed"):
            actions.append(QuizAction(kind="screen_scroll", bbox=screen_state.get("content_column"), direction="down", amount=3, reason="screen_scroll_for_select"))
            trace["stage"] = "scroll"
            trace["post_action_expectation"] = _expectation(
                screen_state=screen_state,
                stage="scroll",
                control_kind=control_kind,
                answer_ready=answer_ready,
                actions=actions,
                expected_values=expected_values,
            )
            _finalize_trace()
            return actions, trace, False
        actions.append(QuizAction(kind="noop", reason="missing_screen_select_bbox"))
        trace["stage"] = "blocked"
        trace["post_action_expectation"] = _expectation(
            screen_state=screen_state,
            stage="blocked",
            control_kind=control_kind,
            answer_ready=answer_ready,
            actions=actions,
            expected_values=expected_values,
        )
        _finalize_trace()
        return actions, trace, False

    if control_kind == "autocomplete":
        input_bbox = screen_state.get("input_bbox")
        if input_bbox:
            answer_text = resolved_answer.correct_answers[0] if resolved_answer.correct_answers else ""
            if not str(answer_text or "").strip():
                actions.append(QuizAction(kind="noop", reason="no_autocomplete_answer"))
                trace["stage"] = "blocked"
                trace["post_action_expectation"] = _expectation(
                    screen_state=screen_state,
                    stage="blocked",
                    control_kind=control_kind,
                    answer_ready=answer_ready,
                    actions=actions,
                    expected_values=expected_values,
                )
                _finalize_trace()
                return actions, trace, False
            if answer_ready:
                actions.append(QuizAction(kind="noop", reason="autocomplete_already_matches"))
                trace["stage"] = "blocked"
                trace["post_action_expectation"] = _expectation(
                    screen_state=screen_state,
                    stage="blocked",
                    control_kind=control_kind,
                    answer_ready=answer_ready,
                    actions=actions,
                    expected_values=expected_values,
                )
                _finalize_trace()
                return actions, trace, False
            actions.extend(
                [
                    QuizAction(kind="screen_click", bbox=[float(v) for v in input_bbox], reason="focus_autocomplete"),
                    QuizAction(kind="key_press", combo="ctrl+a", reason="select_all_text"),
                    QuizAction(kind="key_press", combo="backspace", reason="clear_autocomplete"),
                    QuizAction(kind="type_text", text=answer_text, reason="type_autocomplete_query"),
                    QuizAction(kind="wait", amount=120, reason="wait_for_autocomplete_options"),
                    QuizAction(kind="dom_pick_autocomplete_option", text=answer_text, reason="autocomplete_pick_answer"),
                    QuizAction(kind="key_press", combo="tab", reason="blur_textbox"),
                ]
            )
            _append_next_after_answer(actions, next_bbox)
            trace["stage"] = "answer"
            trace["post_action_expectation"] = _expectation(
                screen_state=screen_state,
                stage="answer",
                control_kind=control_kind,
                answer_ready=answer_ready,
                actions=actions,
                expected_values=expected_values,
            )
            _finalize_trace()
            return actions, trace, bool(trace["fallback_used"])
        if screen_state.get("scroll_needed"):
            actions.append(QuizAction(kind="screen_scroll", bbox=screen_state.get("content_column"), direction="down", amount=3, reason="screen_scroll_for_autocomplete"))
            trace["stage"] = "scroll"
            trace["post_action_expectation"] = _expectation(
                screen_state=screen_state,
                stage="scroll",
                control_kind=control_kind,
                answer_ready=answer_ready,
                actions=actions,
                expected_values=expected_values,
            )
            _finalize_trace()
            return actions, trace, False
        actions.append(QuizAction(kind="noop", reason="missing_screen_autocomplete_bbox"))
        trace["stage"] = "blocked"
        trace["post_action_expectation"] = _expectation(
            screen_state=screen_state,
            stage="blocked",
            control_kind=control_kind,
            answer_ready=answer_ready,
            actions=actions,
            expected_values=expected_values,
        )
        _finalize_trace()
        return actions, trace, False

    if control_kind == "masked_input":
        input_bbox = screen_state.get("input_bbox")
        if input_bbox:
            answer_text = resolved_answer.correct_answers[0] if resolved_answer.correct_answers else ""
            if not str(answer_text or "").strip():
                actions.append(QuizAction(kind="noop", reason="no_masked_input_answer"))
                trace["stage"] = "blocked"
                trace["post_action_expectation"] = _expectation(
                    screen_state=screen_state,
                    stage="blocked",
                    control_kind=control_kind,
                    answer_ready=answer_ready,
                    actions=actions,
                    expected_values=expected_values,
                )
                _finalize_trace()
                return actions, trace, False
            if answer_ready:
                actions.append(QuizAction(kind="noop", reason="masked_input_already_matches"))
                trace["stage"] = "blocked"
                trace["post_action_expectation"] = _expectation(
                    screen_state=screen_state,
                    stage="blocked",
                    control_kind=control_kind,
                    answer_ready=answer_ready,
                    actions=actions,
                    expected_values=expected_values,
                )
                _finalize_trace()
                return actions, trace, False
            actions.extend(
                [
                    QuizAction(kind="screen_click", bbox=[float(v) for v in input_bbox], reason="focus_masked_input"),
                    QuizAction(kind="dom_fill_masked_input", text=answer_text, reason="fill_masked_input_answer"),
                    QuizAction(kind="key_press", combo="tab", reason="blur_textbox"),
                ]
            )
            _append_next_after_answer(actions, next_bbox)
            trace["stage"] = "answer"
            trace["post_action_expectation"] = _expectation(
                screen_state=screen_state,
                stage="answer",
                control_kind=control_kind,
                answer_ready=answer_ready,
                actions=actions,
                expected_values=expected_values,
            )
            _finalize_trace()
            return actions, trace, bool(trace["fallback_used"])
        if screen_state.get("scroll_needed"):
            actions.append(QuizAction(kind="screen_scroll", bbox=screen_state.get("content_column"), direction="down", amount=3, reason="screen_scroll_for_masked_input"))
            trace["stage"] = "scroll"
            trace["post_action_expectation"] = _expectation(
                screen_state=screen_state,
                stage="scroll",
                control_kind=control_kind,
                answer_ready=answer_ready,
                actions=actions,
                expected_values=expected_values,
            )
            _finalize_trace()
            return actions, trace, False
        actions.append(QuizAction(kind="noop", reason="missing_screen_masked_input_bbox"))
        trace["stage"] = "blocked"
        trace["post_action_expectation"] = _expectation(
            screen_state=screen_state,
            stage="blocked",
            control_kind=control_kind,
            answer_ready=answer_ready,
            actions=actions,
            expected_values=expected_values,
        )
        _finalize_trace()
        return actions, trace, False

    if control_kind in {"autocomplete", "masked_input"}:
        reason = {
            "autocomplete": "unsupported_autocomplete_control",
            "masked_input": "unsupported_masked_input_control",
        }.get(control_kind, "unsupported_control")
        actions.append(QuizAction(kind="noop", reason=reason))
        trace["stage"] = "blocked"
        trace["post_action_expectation"] = _expectation(
            screen_state=screen_state,
            stage="blocked",
            control_kind=control_kind,
            answer_ready=answer_ready,
            actions=actions,
            expected_values=expected_values,
        )
        _finalize_trace()
        return actions, trace, False

    target_answers = resolved_answer.correct_answers or []
    if not target_answers:
        actions.append(QuizAction(kind="noop", reason="no_target_answers"))
        trace["stage"] = "blocked"
        trace["post_action_expectation"] = _expectation(
            screen_state=screen_state,
            stage="blocked",
            control_kind=control_kind,
            answer_ready=answer_ready,
            actions=actions,
            expected_values=expected_values,
        )
        _finalize_trace()
        return actions, trace, False
    matched_expected = _matched_expected_values(resolved_answer, controls_data)
    pending_target_answers = [
        target_text
        for target_text in target_answers
        if normalize_match_text(target_text) not in matched_expected
    ]
    for target_text in pending_target_answers:
        bbox = _best_option_bbox(screen_state, target_text)
        if bbox is None:
            bbox = _synthetic_choice_bbox(screen_state, resolved_answer, target_text)
            if bbox is not None:
                trace["fallback_used"] = True
                trace.setdefault("fallback_reasons", []).append(f"synthetic_choice_bbox:{normalize_match_text(target_text)}")
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
                trace["post_action_expectation"] = _expectation(
                    screen_state=screen_state,
                    stage="scroll",
                    control_kind=control_kind,
                    answer_ready=answer_ready,
                    actions=actions,
                    expected_values=expected_values,
                )
                _finalize_trace()
                return actions, trace, bool(trace["fallback_used"])
            actions.append(QuizAction(kind="noop", reason=f"missing_screen_option_bbox:{target_text}"))
            trace["stage"] = "blocked"
            trace["post_action_expectation"] = _expectation(
                screen_state=screen_state,
                stage="blocked",
                control_kind=control_kind,
                answer_ready=answer_ready,
                actions=actions,
                expected_values=expected_values,
            )
            _finalize_trace()
            return actions, trace, False
        actions.append(QuizAction(kind="screen_click", bbox=bbox, reason=f"click_answer:{target_text}"))
    if not actions and next_bbox and matched_expected:
        actions.append(QuizAction(kind="screen_click", bbox=[float(v) for v in next_bbox], reason="click_next_after_answer"))
        trace["stage"] = "next"
        trace["post_action_expectation"] = _expectation(
            screen_state=screen_state,
            stage="next",
            control_kind=control_kind,
            answer_ready=answer_ready,
            actions=actions,
            expected_values=expected_values,
        )
        _finalize_trace()
        return actions, trace, False
    _append_next_after_answer(actions, next_bbox)
    trace["stage"] = "answer"
    trace["post_action_expectation"] = _expectation(
        screen_state=screen_state,
        stage="answer",
        control_kind=control_kind,
        answer_ready=answer_ready,
        actions=actions,
        expected_values=expected_values,
    )
    _finalize_trace()
    return actions, trace, bool(trace["fallback_used"])
