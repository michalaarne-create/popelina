from __future__ import annotations

from typing import Any, Dict, Iterable, Optional


_MODAL_TOKENS = (
    "paywall",
    "subscribe",
    "sign in",
    "log in",
    "cookie consent",
    "accept cookies",
    "dialog",
    "modal",
    "overlay",
)


def _iter_texts(screen_state: Dict[str, Any], page_data: Optional[Dict[str, Any]], region_payload: Optional[Dict[str, Any]]) -> Iterable[str]:
    yield str(screen_state.get("question_text") or "")
    if isinstance(page_data, dict):
        yield str(page_data.get("title") or "")
        yield str(page_data.get("pageText") or page_data.get("page_text") or "")
    if isinstance(region_payload, dict):
        for row in region_payload.get("results") or []:
            if isinstance(row, dict):
                yield str(row.get("text") or row.get("box_text") or "")


def build_anti_modal_guard_state(
    *,
    screen_state: Dict[str, Any],
    page_data: Optional[Dict[str, Any]] = None,
    region_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    joined = "\n".join(str(text or "").lower() for text in _iter_texts(screen_state, page_data, region_payload) if str(text or "").strip())
    matched = sorted({token for token in _MODAL_TOKENS if token in joined})
    return {
        "is_blocking": bool(matched),
        "reason": "unexpected_modal_overlay" if matched else "clear",
        "matched_tokens": matched,
    }
