from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from scripts.pipeline.contracts import SCHEMA_VERSION
from .form_validation_parser import parse_form_validation


_VALIDATION_ERROR_TOKENS = (
    "required",
    "is required",
    "please fill",
    "please select",
    "invalid",
    "incorrect",
    "must be",
    "error",
    "błąd",
    "blad",
    "wymagane",
    "wymagany",
    "uzupełnij",
    "uzupelnij",
    "nieprawidł",
    "nieprawidlow",
)

_REQUIRED_TOKENS = (
    "required",
    "is required",
    "please fill",
    "please select",
    "wymagane",
    "wymagany",
    "uzupełnij",
    "uzupelnij",
)

_INLINE_ERROR_TOKENS = (
    "invalid",
    "incorrect",
    "must be",
    "error",
    "błąd",
    "blad",
    "nieprawidł",
    "nieprawidlow",
)

_SUBMIT_TOKENS = (
    "submit",
    "send",
    "finish",
    "done",
    "continue",
    "dalej",
    "wyślij",
    "wyslij",
    "zakończ",
    "zakoncz",
)

_DISABLED_TOKENS = (
    "disabled",
    "inactive",
    "unavailable",
    "not available",
    "blocked",
    "zablok",
    "nieaktyw",
    "disabled submit",
)

_MODAL_OVERLAY_TOKENS = (
    "dialog",
    "modal",
    "overlay",
    "close",
    "zamknij",
    "cancel",
    "anuluj",
    "are you sure",
    "czy na pewno",
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


def build_validation_signal_state(
    *,
    screen_state: Dict[str, Any],
    page_data: Optional[Dict[str, Any]] = None,
    region_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    structured = parse_form_validation(
        screen_state=screen_state,
        page_data=page_data,
        region_payload=region_payload,
    )
    texts = [str(text or "").strip() for text in _iter_texts(screen_state, page_data, region_payload) if str(text or "").strip()]
    joined = "\n".join(text.lower() for text in texts)
    validation_error = any(token in joined for token in _VALIDATION_ERROR_TOKENS)
    required_error = any(token in joined for token in _REQUIRED_TOKENS)
    inline_error = any(token in joined for token in _INLINE_ERROR_TOKENS)
    modal_overlay = any(token in joined for token in _MODAL_OVERLAY_TOKENS)
    toast_error = any(token in joined for token in _TOAST_TOKENS) and any(token in joined for token in _INLINE_ERROR_TOKENS)
    banner_error = any(token in joined for token in _BANNER_TOKENS) and any(token in joined for token in _INLINE_ERROR_TOKENS)
    disabled_submit = any(submit in joined for submit in _SUBMIT_TOKENS) and any(token in joined for token in _DISABLED_TOKENS)
    raw_structured_category = str(structured.get("category") or "none")
    specific_inline_categories = {"email_error", "number_error", "regex_error", "min_max_error", "selection_limit_error"}
    category = raw_structured_category
    if category == "none":
        if modal_overlay and validation_error:
            category = "modal_overlay"
        elif modal_overlay:
            category = "modal_overlay"
        elif toast_error:
            category = "toast_error"
        elif banner_error:
            category = "banner_error"
        elif disabled_submit:
            category = "disabled_submit"
        elif required_error:
            category = "required_error"
        elif inline_error:
            category = "inline_error"
        elif validation_error:
            category = "validation_error"
    blocking = bool(
        structured.get("is_blocking")
        or validation_error
        or modal_overlay
        or toast_error
        or banner_error
        or disabled_submit
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "category": category,
        "is_validation_error": bool(validation_error),
        "is_required_error": bool(required_error),
        "is_inline_error": bool(inline_error or raw_structured_category in specific_inline_categories),
        "is_email_error": raw_structured_category == "email_error" or bool((structured.get("constraints") or {}).get("email")),
        "is_number_error": raw_structured_category == "number_error" or bool((structured.get("constraints") or {}).get("number")),
        "is_regex_error": raw_structured_category == "regex_error" or bool((structured.get("constraints") or {}).get("regex")),
        "is_min_max_error": raw_structured_category == "min_max_error" or bool((structured.get("constraints") or {}).get("min_max")),
        "is_selection_limit_error": raw_structured_category == "selection_limit_error" or bool((structured.get("constraints") or {}).get("selection_limit")),
        "is_modal_overlay": bool(modal_overlay),
        "is_toast_error": bool(toast_error),
        "is_banner_error": bool(banner_error),
        "is_disabled_submit": bool(disabled_submit),
        "is_blocking": blocking,
        "message_text": str(structured.get("message_text") or ""),
        "field_label": str(structured.get("field_label") or ""),
        "field_kind": str(structured.get("field_kind") or ""),
        "submit_disabled": bool(structured.get("submit_disabled")),
        "constraints": dict(structured.get("constraints") or {}),
        "source": str(structured.get("source") or "none"),
        "structured": structured,
        "matched_text_count": len(texts),
    }
