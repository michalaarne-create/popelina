from __future__ import annotations

from typing import Any, Dict, Optional

from scripts.pipeline.contracts import SCHEMA_VERSION


_CONFIRMATION_TOKENS = (
    "thank you",
    "submitted",
    "submission complete",
    "saved",
    "success",
    "confirmed",
    "dziekujemy",
    "dziękujemy",
    "wyslano",
    "wysłano",
    "zapisano",
    "gotowe",
    "ukonczono",
    "ukończono",
)

_VALIDATION_TOKENS = (
    "required",
    "invalid",
    "error",
    "incorrect",
    "please fill",
    "please select",
    "wymagane",
    "blad",
    "błąd",
    "nieprawid",
)

_STATUS_TOKENS = (
    "loading",
    "processing",
    "saving",
    "updated",
    "selected",
    "step",
    "question",
    "ladowanie",
    "ładowanie",
    "zapisywanie",
    "zaktualizowano",
    "wybrano",
    "krok",
    "pytanie",
)


def _normalize_live_region_texts(state: Optional[Dict[str, Any]]) -> list[str]:
    texts = []
    for value in (state or {}).get("liveRegionTexts") or []:
        text = str(value or "").strip()
        if text:
            texts.append(text)
    return texts


def _classify_live_region_text(joined_text: str) -> tuple[str, str]:
    if any(token in joined_text for token in _CONFIRMATION_TOKENS):
        return "confirmation", "aria_live_confirmation_message"
    if any(token in joined_text for token in _VALIDATION_TOKENS):
        return "validation", "aria_live_validation_message"
    if any(token in joined_text for token in _STATUS_TOKENS):
        return "status", "aria_live_status_message"
    return "none", "no_live_region_event"


def build_live_region_event_parser(
    *,
    previous_page_state: Optional[Dict[str, Any]] = None,
    current_page_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    previous_texts = _normalize_live_region_texts(previous_page_state)
    current_texts = _normalize_live_region_texts(current_page_state)
    previous_set = {text.lower() for text in previous_texts}
    new_texts = [text for text in current_texts if text.lower() not in previous_set]
    effective_texts = new_texts or current_texts
    joined = "\n".join(effective_texts).lower()
    category, reason = _classify_live_region_text(joined)

    return {
        "schema_version": SCHEMA_VERSION,
        "category": category,
        "reason": reason,
        "is_eventful": category != "none",
        "live_region_texts": current_texts,
        "new_live_region_texts": new_texts,
        "signals": {
            "current_count": len(current_texts),
            "new_count": len(new_texts),
            "changed": bool(new_texts),
        },
    }
