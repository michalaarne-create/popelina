from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

from scripts.pipeline.contracts import SCHEMA_VERSION


_REVIEW_TOKENS = (
    "review",
    "summary",
    "podsumowanie",
    "sprawd",
    "sprawdz",
    "confirm your answers",
)

_SUBMIT_TOKENS = (
    "submit",
    "send",
    "finish",
    "done",
    "complete",
    "confirm",
    "wyślij",
    "wyslij",
    "zakończ",
    "zakoncz",
    "potwierd",
)

_STEP_PATTERNS = (
    re.compile(r"\b(?:question|step|krok|pytanie)\s+(\d{1,3})\s*(?:/|z|of)\s*(\d{1,3})\b", re.IGNORECASE),
    re.compile(r"\b(\d{1,3})\s*(?:/|z|of)\s*(\d{1,3})\b", re.IGNORECASE),
)

_PERCENT_PATTERN = re.compile(r"\b(\d{1,3})\s*%")
_SECTION_PATTERNS = (
    re.compile(r"\b(?:section|sekcja)\s*[:\-]?\s*([A-Za-z0-9ĄąĆćĘęŁłŃńÓóŚśŹźŻż _\-/]{2,80})", re.IGNORECASE),
    re.compile(r"\b(?:part|część|czesc)\s*[:\-]?\s*([A-Za-z0-9ĄąĆćĘęŁłŃńÓóŚśŹźŻż _\-/]{2,80})", re.IGNORECASE),
)


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split()).lower()


def _iter_screen_texts(screen_state: Dict[str, Any], page_data: Optional[Dict[str, Any]]) -> Iterable[str]:
    if isinstance(screen_state, dict):
        yield str(screen_state.get("question_text") or "")
        for row in screen_state.get("options") or []:
            if isinstance(row, dict):
                yield str(row.get("text") or "")
    if isinstance(page_data, dict):
        yield str(page_data.get("pageText") or page_data.get("page_text") or "")
        yield str(page_data.get("title") or "")
        for row in page_data.get("textBlocks") or []:
            if isinstance(row, dict):
                yield str(row.get("text") or "")


def _extract_readback_values(page_data: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(page_data, dict):
        return []
    values: List[str] = []
    for row in page_data.get("options") or []:
        if isinstance(row, dict) and bool(row.get("checked")):
            text = _normalize_text(row.get("text") or "")
            if text:
                values.append(text)
    for row in page_data.get("selectOptions") or []:
        if isinstance(row, dict) and bool(row.get("selected")):
            text = _normalize_text(row.get("text") or row.get("value") or "")
            if text:
                values.append(text)
    textbox = _normalize_text(page_data.get("textboxValue") or "")
    if textbox:
        values.append(textbox)
    deduped: List[str] = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _values_match(expected_values: Iterable[Any], readback_values: Iterable[str]) -> bool:
    expected = [_normalize_text(value) for value in expected_values if _normalize_text(value)]
    actual = {_normalize_text(value) for value in readback_values if _normalize_text(value)}
    if not expected or not actual:
        return False
    return all(value in actual for value in expected)


def _extract_progress_semantics(screen_state: Dict[str, Any], page_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    texts = [str(text or "").strip() for text in _iter_screen_texts(screen_state, page_data) if str(text or "").strip()]
    lines: List[str] = []
    for text in texts:
        for line in str(text).splitlines():
            line = " ".join(line.split()).strip()
            if line:
                lines.append(line)
    current_step = None
    total_steps = None
    progress_percent = None
    section_label = ""

    for text in lines:
        compact = " ".join(text.split())
        if current_step is None or total_steps is None:
            for pattern in _STEP_PATTERNS:
                match = pattern.search(compact)
                if not match:
                    continue
                try:
                    cur = int(match.group(1))
                    total = int(match.group(2))
                except Exception:
                    continue
                if cur >= 1 and total >= cur:
                    current_step = cur
                    total_steps = total
                    break
        if progress_percent is None:
            percent_match = _PERCENT_PATTERN.search(compact)
            if percent_match:
                try:
                    pct = int(percent_match.group(1))
                except Exception:
                    pct = None
                if pct is not None and 0 <= pct <= 100:
                    progress_percent = pct
        if not section_label:
            for pattern in _SECTION_PATTERNS:
                match = pattern.search(compact)
                if not match:
                    continue
                section_label = " ".join(str(match.group(1) or "").split("•")[0].split()).strip(" -:")
                if section_label:
                    break

    return {
        "current_step": current_step,
        "total_steps": total_steps,
        "progress_percent": progress_percent,
        "section_label": section_label,
        "has_progress": bool(
            current_step is not None
            or total_steps is not None
            or progress_percent is not None
            or section_label
        ),
    }


def _classify_state(
    *,
    screen_state: Dict[str, Any],
    page_data: Optional[Dict[str, Any]],
    terminal_state: Optional[Dict[str, Any]],
    validation_state: Optional[Dict[str, Any]],
    submission_confirmation: Optional[Dict[str, Any]],
    previous_state: str,
    home_screen_detected: bool,
) -> Dict[str, Any]:
    current_url = str((page_data or {}).get("url") or "")
    question_text = str(screen_state.get("question_text") or "").strip()
    options = screen_state.get("options") if isinstance(screen_state.get("options"), list) else []
    has_question = bool(question_text)
    has_options = bool(options)
    next_visible = bool((page_data or {}).get("nextVisible")) or bool(screen_state.get("next_bbox"))
    has_input = bool((page_data or {}).get("hasTextbox")) or bool(screen_state.get("input_bbox"))
    has_select = bool((page_data or {}).get("hasSelect")) or bool(screen_state.get("select_bbox"))
    joined = "\n".join(_normalize_text(text) for text in _iter_screen_texts(screen_state, page_data) if _normalize_text(text))
    submit_visible = any(token in joined for token in _SUBMIT_TOKENS)
    review_visible = any(token in joined for token in _REVIEW_TOKENS)
    terminal_screen = (terminal_state or {}).get("terminal_screen") if isinstance((terminal_state or {}).get("terminal_screen"), dict) else {}
    terminal_kind = str((terminal_screen or {}).get("screen_kind") or "")
    if terminal_kind in {"review", "summary"}:
        review_visible = True
    if terminal_kind == "summary":
        submit_visible = True
    is_blocked = bool((validation_state or {}).get("is_blocking"))
    is_complete = (
        not is_blocked
        and (bool((terminal_state or {}).get("is_terminal")) or bool((submission_confirmation or {}).get("is_confirmed")))
    )
    if terminal_kind in {"home", "thank_you", "confirmation", "complete"} and not is_blocked:
        is_complete = True

    state = "start"
    reason = "no_strong_signals"
    confidence = 0.55
    if is_blocked and not is_complete:
        if not (has_question or has_options or has_input or has_select):
            state = "blocked"
            reason = "blocking_validation_without_actionable_controls"
            confidence = 0.9
        elif terminal_kind in {"review", "summary"} and not has_options:
            state = "review"
            reason = "terminal_review_or_summary"
            confidence = 0.84
        elif review_visible and next_visible and not has_options:
            state = "review"
            reason = "review_tokens_plus_next"
            confidence = 0.82
        elif has_question or has_options or has_input or has_select:
            state = "question"
            reason = "active_question_controls_present"
            confidence = 0.88 if has_options or has_input or has_select else 0.76
        elif current_url:
            state = "start"
            reason = "url_present_without_question"
            confidence = 0.62
    elif is_complete:
        state = "complete"
        reason = "terminal_or_submission_confirmation"
        confidence = 0.98
    elif home_screen_detected:
        state = "complete"
        reason = "home_screen_detected"
        confidence = 0.8 if previous_state in {"question", "review", "submit"} else 0.74
    elif is_blocked and not (has_question or has_options or has_input or has_select):
        state = "blocked"
        reason = "blocking_validation_without_actionable_controls"
        confidence = 0.9
    elif terminal_kind in {"review", "summary"} and not has_options:
        state = "review"
        reason = "terminal_review_or_summary"
        confidence = 0.84
    elif submit_visible and not has_options:
        state = "submit"
        reason = "submit_tokens_visible"
        confidence = 0.86
    elif review_visible and next_visible and not has_options:
        state = "review"
        reason = "review_tokens_plus_next"
        confidence = 0.82
    elif has_question or has_options or has_input or has_select:
        state = "question"
        reason = "active_question_controls_present"
        confidence = 0.93 if has_options or has_input or has_select else 0.79
    elif current_url:
        state = "start"
        reason = "url_present_without_question"
        confidence = 0.62
    return {
        "state": state,
        "reason": reason,
        "confidence": round(confidence, 4),
        "signals": {
            "has_question": bool(has_question),
            "has_options": bool(has_options),
            "next_visible": bool(next_visible),
            "has_input": bool(has_input),
            "has_select": bool(has_select),
            "submit_visible": bool(submit_visible),
            "review_visible": bool(review_visible),
            "terminal_kind": terminal_kind,
            "current_url": current_url,
        },
    }


def build_quiz_runtime_fsm(
    *,
    screen_state: Dict[str, Any],
    page_data: Optional[Dict[str, Any]] = None,
    terminal_state: Optional[Dict[str, Any]] = None,
    validation_state: Optional[Dict[str, Any]] = None,
    submission_confirmation: Optional[Dict[str, Any]] = None,
    previous_runtime_state: Optional[Dict[str, Any]] = None,
    previous_signature: str = "",
    previous_url: str = "",
    previous_action: Optional[Dict[str, Any]] = None,
    expected_values: Iterable[Any] = (),
    attempt_count: int = 0,
    action_progressed: Optional[bool] = None,
    home_screen_detected: bool = False,
) -> Dict[str, Any]:
    current_signature = str(screen_state.get("screen_signature") or "")
    current_url = str((page_data or {}).get("url") or "")
    previous_state = str((previous_runtime_state or {}).get("state") or "")
    state_info = _classify_state(
        screen_state=screen_state,
        page_data=page_data,
        terminal_state=terminal_state,
        validation_state=validation_state,
        submission_confirmation=submission_confirmation,
        previous_state=previous_state,
        home_screen_detected=home_screen_detected,
    )
    readback_values = _extract_readback_values(page_data)
    readback_match = _values_match(expected_values, readback_values)
    same_signature = bool(previous_signature) and current_signature == previous_signature
    same_url = bool(previous_url) and current_url.rstrip("/") == previous_url.rstrip("/")
    inferred_no_progress = bool(same_signature and same_url and previous_signature)
    if action_progressed is None:
        no_progress = inferred_no_progress
    else:
        no_progress = not bool(action_progressed)
    previous_stuck = int(((previous_runtime_state or {}).get("stuck_detection") or {}).get("consecutive_no_progress_count") or 0)
    consecutive_no_progress_count = (previous_stuck + 1) if no_progress else 0
    repeated_attempts = int(attempt_count or 0)
    is_stuck = bool(
        state_info["state"] not in {"complete", "blocked"}
        and (
            consecutive_no_progress_count >= 2
            or (no_progress and repeated_attempts >= 2)
            or (no_progress and readback_match and repeated_attempts >= 1)
        )
    )

    recovery_level = 0
    recovery_action = "none"
    recovery_reason = "progress_ok"
    if state_info["state"] == "blocked":
        recovery_reason = "blocked_validation_state"
    elif state_info["state"] == "complete":
        recovery_reason = "terminal_state"
    elif no_progress and repeated_attempts >= 3:
        recovery_level = 3
        recovery_action = "abort_iteration"
        recovery_reason = "max_retries_without_progress"
    elif no_progress and repeated_attempts >= 2:
        recovery_level = 2
        recovery_action = "dom_fallback"
        recovery_reason = "repeat_no_progress"
    elif no_progress:
        recovery_level = 1
        recovery_action = "retry_same_action"
        recovery_reason = "single_no_progress"

    progress_semantics = _extract_progress_semantics(screen_state, page_data)

    return {
        "schema_version": SCHEMA_VERSION,
        "state": state_info["state"],
        "state_reason": state_info["reason"],
        "state_confidence": state_info["confidence"],
        "state_changed": bool(previous_state and previous_state != state_info["state"]),
        "previous_state": previous_state or None,
        "signals": state_info["signals"],
        "progress": progress_semantics,
        "stuck_detection": {
            "same_signature": bool(same_signature),
            "same_url": bool(same_url),
            "readback_match": bool(readback_match),
            "readback_values": readback_values,
            "expected_values": [_normalize_text(value) for value in expected_values if _normalize_text(value)],
            "action_progressed": None if action_progressed is None else bool(action_progressed),
            "no_progress": bool(no_progress),
            "consecutive_no_progress_count": int(consecutive_no_progress_count),
            "attempt_count": int(repeated_attempts),
            "is_stuck": bool(is_stuck),
        },
        "recovery": {
            "level": int(recovery_level),
            "action": recovery_action,
            "reason": recovery_reason,
            "should_retry_same_action": recovery_action == "retry_same_action",
            "should_use_dom_fallback": recovery_action == "dom_fallback",
            "should_abort": recovery_action == "abort_iteration",
        },
        "context": {
            "current_signature": current_signature,
            "current_url": current_url,
            "previous_signature": previous_signature,
            "previous_url": previous_url,
            "previous_action": dict(previous_action or {}),
        },
    }
