from __future__ import annotations

from typing import Any, Dict, Optional

from scripts.pipeline.contracts import SCHEMA_VERSION


_LIVE_ERROR_TOKENS = (
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


def build_accessibility_validation_reader(
    *,
    current_page_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    state = current_page_state or {}
    aria_invalid = bool(state.get("textboxAriaInvalid"))
    live_texts = [str(text or "").strip() for text in (state.get("liveRegionTexts") or []) if str(text or "").strip()]
    joined = "\n".join(live_texts).lower()
    live_validation = any(token in joined for token in _LIVE_ERROR_TOKENS)

    category = "none"
    reason = "no_accessibility_validation_signal"
    if aria_invalid:
        category = "aria_invalid"
        reason = "textbox_marked_invalid"
    elif live_validation:
        category = "live_region_validation"
        reason = "aria_live_validation_message"

    return {
        "schema_version": SCHEMA_VERSION,
        "category": category,
        "reason": reason,
        "is_blocking": category != "none",
        "signals": {
            "textbox_aria_invalid": aria_invalid,
            "live_region_count": len(live_texts),
        },
        "live_region_texts": live_texts,
    }
