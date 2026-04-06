from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Optional

from scripts.pipeline.contracts import SCHEMA_VERSION


_REQUIRED_TOKENS = (
    "required",
    "is required",
    "please fill",
    "please select",
    "wymagane",
    "wymagany",
    "uzupelnij",
    "uzupelnic",
)

_INLINE_ERROR_TOKENS = (
    "invalid",
    "incorrect",
    "must be",
    "error",
    "blad",
    "nieprawid",
)

_EMAIL_TOKENS = (
    "invalid email",
    "e-mail",
    "must be a valid email",
    "valid email address",
)

_NUMBER_TOKENS = (
    "must be a number",
    "invalid number",
    "digits only",
    "numeric",
    "liczba",
    "cyfry",
)

_REGEX_TOKENS = (
    "invalid format",
    "expected format",
    "format should be",
    "pattern",
    "regex",
    "format",
)

_MIN_MAX_TOKENS = (
    "minimum",
    "maximum",
    "at least",
    "at most",
    "too short",
    "too long",
    "between",
    "characters",
    "znak",
    "minimum",
    "maksimum",
)

_SELECTION_LIMIT_TOKENS = (
    "select up to",
    "select at least",
    "too many selected",
    "selection limit",
    "choose up to",
    "wybierz maksymalnie",
    "wybierz co najmniej",
)

_DISABLED_TOKENS = (
    "disabled",
    "inactive",
    "unavailable",
    "not available",
    "blocked",
    "zablok",
    "nieaktyw",
)

_SUBMIT_TOKENS = (
    "submit",
    "send",
    "finish",
    "done",
    "continue",
    "dalej",
    "wyslij",
    "zakoncz",
)

_TOAST_TOKENS = (
    "toast",
    "snackbar",
    "notification",
    "powiadomienie",
    "komunikat",
)

_BANNER_TOKENS = (
    "banner",
    "alert",
    "warning",
    "ostrzezenie",
    "uwaga",
)

_MODAL_TOKENS = (
    "dialog",
    "modal",
    "overlay",
    "close",
    "zamknij",
    "cancel",
    "anuluj",
    "are you sure",
    "czy na pewno",
    "confirm",
    "potwierd",
)

_VALIDATION_TOKENS = tuple(set(_REQUIRED_TOKENS + _INLINE_ERROR_TOKENS + ("validation",)))

_FIELD_LABEL_PATTERNS = (
    re.compile(r"^\s*([A-Za-z0-9 _\-/]{2,48})\s+(?:is required|is invalid|must be)\b", re.IGNORECASE),
    re.compile(r"(?:field|pole|pytanie)\s*[:\-]\s*([A-Za-z0-9 _\-/]{2,48})", re.IGNORECASE),
)


def _iter_text_rows(
    screen_state: Dict[str, Any],
    page_data: Optional[Dict[str, Any]],
    region_payload: Optional[Dict[str, Any]],
) -> Iterable[Dict[str, str]]:
    if isinstance(screen_state, dict):
        question = str(screen_state.get("question_text") or "").strip()
        if question:
            yield {"text": question, "source": "screen_state.question_text"}
        for idx, row in enumerate(screen_state.get("options") or []):
            if not isinstance(row, dict):
                continue
            text = str(row.get("text") or "").strip()
            if text:
                yield {"text": text, "source": f"screen_state.options[{idx}]"}
    if isinstance(page_data, dict):
        title = str(page_data.get("title") or "").strip()
        if title:
            yield {"text": title, "source": "page_data.title"}
        page_text = str(page_data.get("pageText") or page_data.get("page_text") or "").strip()
        if page_text:
            yield {"text": page_text, "source": "page_data.page_text"}
        for idx, row in enumerate(page_data.get("textBlocks") or []):
            if not isinstance(row, dict):
                continue
            text = str(row.get("text") or "").strip()
            if text:
                yield {"text": text, "source": f"page_data.textBlocks[{idx}]"}
    if isinstance(region_payload, dict):
        for idx, row in enumerate(region_payload.get("results") or []):
            if not isinstance(row, dict):
                continue
            text = str(row.get("text") or row.get("box_text") or "").strip()
            if text:
                yield {"text": text, "source": f"region_payload.results[{idx}]"}


def _extract_field_label(text: str) -> str:
    for pattern in _FIELD_LABEL_PATTERNS:
        match = pattern.search(text or "")
        if match:
            return " ".join(str(match.group(1) or "").split())
    return ""


def _infer_field_kind(field_label: str, text: str, page_data: Optional[Dict[str, Any]]) -> str:
    joined = f"{field_label} {text}".lower()
    if "email" in joined or "e-mail" in joined:
        return "email"
    if "phone" in joined or "telefon" in joined or "tel" in joined:
        return "phone"
    if "date" in joined or "data" in joined:
        return "date"
    if isinstance(page_data, dict):
        if bool(page_data.get("hasSelect")):
            return "select"
        if bool(page_data.get("hasTextbox")):
            return "text"
    return "unknown"


def parse_form_validation(
    *,
    screen_state: Dict[str, Any],
    page_data: Optional[Dict[str, Any]] = None,
    region_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    rows = list(_iter_text_rows(screen_state, page_data, region_payload))
    joined = "\n".join(str((row or {}).get("text") or "").lower() for row in rows)

    required = any(token in joined for token in _REQUIRED_TOKENS)
    inline = any(token in joined for token in _INLINE_ERROR_TOKENS)
    validation = any(token in joined for token in _VALIDATION_TOKENS)
    email_error = any(token in joined for token in _EMAIL_TOKENS)
    number_error = any(token in joined for token in _NUMBER_TOKENS)
    regex_error = any(token in joined for token in _REGEX_TOKENS)
    min_max_error = any(token in joined for token in _MIN_MAX_TOKENS)
    selection_limit_error = any(token in joined for token in _SELECTION_LIMIT_TOKENS)
    modal = any(token in joined for token in _MODAL_TOKENS)
    toast = any(token in joined for token in _TOAST_TOKENS) and inline
    banner = any(token in joined for token in _BANNER_TOKENS) and inline
    submit_disabled = any(token in joined for token in _DISABLED_TOKENS) and any(token in joined for token in _SUBMIT_TOKENS)

    category = "none"
    if modal:
        category = "modal_overlay"
    elif toast:
        category = "toast_error"
    elif banner:
        category = "banner_error"
    elif submit_disabled:
        category = "disabled_submit"
    elif selection_limit_error:
        category = "selection_limit_error"
    elif email_error:
        category = "email_error"
    elif number_error:
        category = "number_error"
    elif regex_error:
        category = "regex_error"
    elif min_max_error:
        category = "min_max_error"
    elif required:
        category = "required_error"
    elif inline:
        category = "inline_error"
    elif validation:
        category = "validation_error"

    message_text = ""
    source = "none"
    if category != "none":
        matched_tokens = ()
        if category == "required_error":
            matched_tokens = _REQUIRED_TOKENS
        elif category == "inline_error":
            matched_tokens = _INLINE_ERROR_TOKENS
        elif category == "disabled_submit":
            matched_tokens = _DISABLED_TOKENS
        elif category == "selection_limit_error":
            matched_tokens = _SELECTION_LIMIT_TOKENS
        elif category == "email_error":
            matched_tokens = _EMAIL_TOKENS
        elif category == "number_error":
            matched_tokens = _NUMBER_TOKENS
        elif category == "regex_error":
            matched_tokens = _REGEX_TOKENS
        elif category == "min_max_error":
            matched_tokens = _MIN_MAX_TOKENS
        elif category == "toast_error":
            matched_tokens = _TOAST_TOKENS
        elif category == "banner_error":
            matched_tokens = _BANNER_TOKENS
        elif category == "modal_overlay":
            matched_tokens = _MODAL_TOKENS
        else:
            matched_tokens = _VALIDATION_TOKENS
        for row in rows:
            text = str((row or {}).get("text") or "").strip()
            text_norm = text.lower()
            if any(token in text_norm for token in matched_tokens):
                message_text = text
                source = str((row or {}).get("source") or "none")
                break

    field_label = _extract_field_label(message_text)
    field_kind = _infer_field_kind(field_label, message_text, page_data)

    return {
        "schema_version": SCHEMA_VERSION,
        "category": category,
        "validation_family": category.replace("_error", "") if category.endswith("_error") else category,
        "message_text": message_text,
        "field_label": field_label,
        "field_kind": field_kind,
        "submit_disabled": bool(submit_disabled),
        "constraints": {
            "email": bool(email_error),
            "number": bool(number_error),
            "regex": bool(regex_error),
            "min_max": bool(min_max_error),
            "selection_limit": bool(selection_limit_error),
        },
        "source": source,
        "is_blocking": bool(category != "none"),
        "matched_text_count": len(rows),
    }
