from __future__ import annotations

from typing import Any, Dict, Iterable, Optional
from urllib.parse import urlparse

from scripts.pipeline.contracts import SCHEMA_VERSION
from .screen_terminal_parser import parse_terminal_screen


_CONFIRMATION_TOKENS = (
    "thank you",
    "thanks",
    "submitted",
    "submission",
    "confirmation",
    "confirmed",
    "success",
    "completed",
    "gotowe",
    "wyslano",
    "wysłano",
    "dziekujemy",
    "dziękujemy",
)

_CONFIRMATION_URL_TOKENS = (
    "thank",
    "thanks",
    "confirm",
    "confirmed",
    "complete",
    "completed",
    "submitted",
    "success",
    "done",
    "finish",
)


def _iter_texts(
    screen_state: Dict[str, Any],
    page_data: Optional[Dict[str, Any]],
    region_payload: Optional[Dict[str, Any]],
) -> Iterable[str]:
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


def build_submission_confirmation_state(
    *,
    screen_state: Dict[str, Any],
    page_data: Optional[Dict[str, Any]] = None,
    region_payload: Optional[Dict[str, Any]] = None,
    previous_url: str = "",
    current_url: str = "",
) -> Dict[str, Any]:
    joined = "\n".join(text.strip().lower() for text in _iter_texts(screen_state, page_data, region_payload) if str(text or "").strip())
    parsed = parse_terminal_screen(
        screen_state=screen_state,
        page_data=page_data,
        region_payload=region_payload,
        previous_url=previous_url,
        current_url=current_url,
        home_screen_detected=False,
    )
    parsed_signals = parsed.get("signals") if isinstance(parsed.get("signals"), dict) else {}
    confirmation_text = bool(parsed_signals.get("thank_you_text") or parsed_signals.get("confirmation_text") or parsed_signals.get("complete_text"))
    current = str(current_url or "")
    previous = str(previous_url or "")
    parsed_url = urlparse(current)
    path_and_query = f"{parsed_url.path} {parsed_url.query}".lower()
    confirmation_url = any(token in path_and_query for token in _CONFIRMATION_URL_TOKENS)
    url_changed = bool(previous and current and previous.rstrip("/") != current.rstrip("/"))
    has_question = bool(str(screen_state.get("question_text") or "").strip())
    has_options = bool(screen_state.get("options"))
    no_question_left = not has_question and not has_options
    is_confirmed = bool(parsed.get("is_complete_like") or confirmation_url or (url_changed and no_question_left))
    confirmation_kind = "none"
    if confirmation_text:
        confirmation_kind = "text"
    elif confirmation_url:
        confirmation_kind = "url"
    elif url_changed and no_question_left:
        confirmation_kind = "url_change_no_question"
    return {
        "schema_version": SCHEMA_VERSION,
        "is_confirmed": is_confirmed,
        "confirmation_kind": confirmation_kind,
        "confirmation_number": str(parsed.get("confirmation_number") or ""),
        "terminal_screen": parsed,
        "signals": {
            "confirmation_text": bool(confirmation_text),
            "confirmation_url": bool(confirmation_url),
            "url_changed": bool(url_changed),
            "no_question_left": bool(no_question_left),
        },
    }
