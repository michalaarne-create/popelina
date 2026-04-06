from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse

from scripts.pipeline.contracts import SCHEMA_VERSION


_REVIEW_TOKENS = (
    "review",
    "check your answers",
    "verify answers",
    "sprawdz",
    "podsumowanie odpowiedzi",
)

_SUMMARY_TOKENS = (
    "summary",
    "podsumowanie",
    "overview",
    "recap",
)

_THANK_YOU_TOKENS = (
    "thank you",
    "thanks",
    "dziekujemy",
    "dziekujemy za",
    "dziekujemy za udzial",
)

_CONFIRMATION_TOKENS = (
    "confirmation",
    "confirmed",
    "confirmation number",
    "reference number",
    "submission id",
    "numer potwierdzenia",
    "id zgloszenia",
)

_COMPLETE_TOKENS = (
    "submitted",
    "submission complete",
    "completed",
    "success",
    "gotowe",
    "wyslano",
    "zakonczono",
)

_FAILURE_TOKENS = (
    "something went wrong",
    "try again later",
    "submission failed",
    "failed to submit",
    "could not submit",
    "error occurred",
    "wystapil blad",
    "wystąpił błąd",
    "nie udalo sie",
    "nie udało się",
)

_EXPIRED_SESSION_TOKENS = (
    "session expired",
    "your session has expired",
    "expired session",
    "time expired",
    "sesja wygasla",
    "sesja wygasła",
    "czas sesji wygasl",
    "czas sesji wygasł",
)

_HOME_TOKENS = (
    "start",
    "reset",
    "test quiz server",
    "podzial na typy pytan",
)

_URL_TERMINAL_TOKENS = (
    "thank",
    "confirm",
    "complete",
    "submitted",
    "success",
    "done",
    "finish",
)

_CONFIRMATION_NUMBER_PATTERNS = (
    re.compile(
        r"\b(?:confirmation|reference|submission|order|ticket|numer|nr|id)\b\s*"
        r"(?:number|no\.?|#|id)\s*[:#-]?\s*([A-Z0-9][A-Z0-9\-]{4,})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:confirmation|reference|submission|order|ticket|numer|nr|id)\b\s*[:#-]\s*"
        r"([A-Z0-9][A-Z0-9\-]{4,})",
        re.IGNORECASE,
    ),
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
        yield str(page_data.get("title") or "")
        yield str(page_data.get("pageText") or page_data.get("page_text") or "")
        for row in page_data.get("textBlocks") or []:
            if isinstance(row, dict):
                yield str(row.get("text") or "")
    if isinstance(region_payload, dict):
        for row in region_payload.get("results") or []:
            if isinstance(row, dict):
                yield str(row.get("text") or row.get("box_text") or "")


def _extract_confirmation_number(texts: List[str]) -> str:
    for text in texts:
        compact = " ".join(str(text or "").strip().split())
        if not compact:
            continue
        for pattern in _CONFIRMATION_NUMBER_PATTERNS:
            match = pattern.search(compact)
            if match:
                return str(match.group(1) or "").strip().upper()
    return ""


def parse_terminal_screen(
    *,
    screen_state: Dict[str, Any],
    page_data: Optional[Dict[str, Any]] = None,
    region_payload: Optional[Dict[str, Any]] = None,
    previous_url: str = "",
    current_url: str = "",
    home_screen_detected: bool = False,
) -> Dict[str, Any]:
    texts = [str(v or "").strip() for v in _iter_texts(screen_state, page_data, region_payload) if str(v or "").strip()]
    joined = "\n".join(text.lower() for text in texts)

    has_question = bool(str(screen_state.get("question_text") or "").strip())
    has_options = bool(screen_state.get("options"))
    no_question_left = not has_question and not has_options
    next_visible = bool((page_data or {}).get("nextVisible")) or bool(screen_state.get("next_bbox"))
    submit_visible = any(token in joined for token in ("submit", "send", "finish", "done", "wyslij", "zakoncz", "potwierd"))

    review_text = any(token in joined for token in _REVIEW_TOKENS)
    summary_text = any(token in joined for token in _SUMMARY_TOKENS)
    thank_you_text = any(token in joined for token in _THANK_YOU_TOKENS)
    confirmation_text = any(token in joined for token in _CONFIRMATION_TOKENS)
    complete_text = any(token in joined for token in _COMPLETE_TOKENS)
    failure_text = any(token in joined for token in _FAILURE_TOKENS)
    expired_session_text = any(token in joined for token in _EXPIRED_SESSION_TOKENS)
    home_text = any(token in joined for token in _HOME_TOKENS)

    current = str(current_url or "")
    previous = str(previous_url or "")
    parsed_url = urlparse(current)
    path_and_query = f"{parsed_url.path} {parsed_url.query}".lower()
    url_confirmation_token = any(token in path_and_query for token in _URL_TERMINAL_TOKENS)
    url_changed = bool(previous and current and previous.rstrip("/") != current.rstrip("/"))

    confirmation_number = _extract_confirmation_number(texts)

    screen_kind = "question"
    if expired_session_text:
        screen_kind = "expired_session"
    elif home_screen_detected or home_text:
        screen_kind = "home"
    elif failure_text and not has_options:
        screen_kind = "failure"
    elif confirmation_number:
        screen_kind = "confirmation"
    elif thank_you_text:
        screen_kind = "thank_you"
    elif confirmation_text:
        screen_kind = "confirmation"
    elif complete_text or (url_changed and no_question_left and not submit_visible and not review_text and not summary_text):
        screen_kind = "complete"
    elif summary_text and not has_options:
        screen_kind = "summary"
    elif review_text or (submit_visible and not has_options):
        screen_kind = "review"
    elif no_question_left:
        screen_kind = "unknown"

    is_terminal = screen_kind in {"home", "thank_you", "confirmation", "complete", "failure", "expired_session"}
    return {
        "schema_version": SCHEMA_VERSION,
        "screen_kind": screen_kind,
        "is_terminal": bool(is_terminal),
        "is_review": bool(screen_kind == "review"),
        "is_summary": bool(screen_kind == "summary"),
        "is_complete_like": bool(screen_kind in {"thank_you", "confirmation", "complete", "home"}),
        "confirmation_number": confirmation_number,
        "signals": {
            "has_question": bool(has_question),
            "has_options": bool(has_options),
            "no_question_left": bool(no_question_left),
            "next_visible": bool(next_visible),
            "submit_visible": bool(submit_visible),
            "review_text": bool(review_text),
            "summary_text": bool(summary_text),
            "thank_you_text": bool(thank_you_text),
            "confirmation_text": bool(confirmation_text),
            "complete_text": bool(complete_text),
            "failure_text": bool(failure_text),
            "expired_session_text": bool(expired_session_text),
            "home_text": bool(home_text),
            "home_screen_detected": bool(home_screen_detected),
            "url_changed": bool(url_changed),
            "url_confirmation_token": bool(url_confirmation_token),
        },
    }
