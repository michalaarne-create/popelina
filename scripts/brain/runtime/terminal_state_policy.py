from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from scripts.pipeline.contracts import SCHEMA_VERSION
from .screen_terminal_parser import parse_terminal_screen


_TERMINAL_TOKENS = (
    "thank you",
    "thanks",
    "dziekujemy",
    "dziękujemy",
    "submitted",
    "submission",
    "confirmation",
    "confirmed",
    "success",
    "completed",
    "zakoncz",
    "zakończ",
    "gotowe",
    "wyslano",
    "wysłano",
)


def _iter_texts(screen_state: Dict[str, Any], page_data: Optional[Dict[str, Any]], region_payload: Optional[Dict[str, Any]]) -> Iterable[str]:
    if isinstance(screen_state, dict):
        yield str(screen_state.get("question_text") or "")
        for row in screen_state.get("options") or []:
            if isinstance(row, dict):
                yield str(row.get("text") or "")
    if isinstance(page_data, dict):
        yield str(page_data.get("pageText") or page_data.get("page_text") or "")
        yield str(page_data.get("title") or "")
    if isinstance(region_payload, dict):
        for row in region_payload.get("results") or []:
            if isinstance(row, dict):
                yield str(row.get("text") or row.get("box_text") or "")


def build_terminal_state_score(
    *,
    screen_state: Dict[str, Any],
    page_data: Optional[Dict[str, Any]] = None,
    region_payload: Optional[Dict[str, Any]] = None,
    page_changed: bool = False,
    home_screen_detected: bool = False,
) -> Dict[str, Any]:
    parsed = parse_terminal_screen(
        screen_state=screen_state,
        page_data=page_data,
        region_payload=region_payload,
        previous_url="",
        current_url=str((page_data or {}).get("url") or ""),
        home_screen_detected=home_screen_detected,
    )
    signals = parsed.get("signals") if isinstance(parsed.get("signals"), dict) else {}
    confirmation_text = bool(signals.get("thank_you_text") or signals.get("confirmation_text") or signals.get("complete_text"))
    failure_terminal = str(parsed.get("screen_kind") or "") in {"failure", "expired_session"}
    no_question_left = bool(signals.get("no_question_left"))
    confirmation_number_present = bool(str(parsed.get("confirmation_number") or "").strip())
    complete_like_terminal = bool(parsed.get("is_complete_like"))
    score = 0.0
    if home_screen_detected:
        score += 0.4
    if bool(parsed.get("is_terminal")):
        score += 0.45
    if complete_like_terminal:
        score += 0.2
    if confirmation_text:
        score += 0.5
    if confirmation_number_present:
        score += 0.35
    if failure_terminal:
        score += 0.35
    if confirmation_text and no_question_left:
        score += 0.15
    if confirmation_number_present and no_question_left:
        score += 0.1
    if page_changed:
        score += 0.2
    if no_question_left:
        score += 0.1
    score = max(0.0, min(1.0, score))
    return {
        "schema_version": SCHEMA_VERSION,
        "terminal_confidence_score": round(score, 4),
        "is_terminal": bool(score >= 0.65),
        "terminal_screen": parsed,
        "confirmation_number": str(parsed.get("confirmation_number") or ""),
        "signals": {
            "home_screen_detected": bool(home_screen_detected),
            "confirmation_text": bool(confirmation_text),
            "confirmation_number_present": bool(confirmation_number_present),
            "complete_like_terminal": bool(complete_like_terminal),
            "failure_terminal": bool(failure_terminal),
            "page_changed": bool(page_changed),
            "no_question_left": bool(no_question_left),
            "screen_kind": str(parsed.get("screen_kind") or ""),
        },
    }


def build_terminal_confidence_report(
    *,
    terminal_state: Dict[str, Any],
    calibration_threshold: float = 0.65,
) -> Dict[str, Any]:
    score = float((terminal_state or {}).get("terminal_confidence_score") or 0.0)
    is_terminal = bool((terminal_state or {}).get("is_terminal"))
    expected_terminal = score >= float(calibration_threshold)
    return {
        "schema_version": SCHEMA_VERSION,
        "terminal_confidence_score": round(score, 4),
        "calibration_threshold": float(calibration_threshold),
        "is_terminal": is_terminal,
        "expected_terminal": expected_terminal,
        "is_calibrated": is_terminal == expected_terminal,
        "signals": {
            "has_terminal_state": bool(terminal_state),
        },
    }
