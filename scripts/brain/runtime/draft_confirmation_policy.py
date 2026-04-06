from __future__ import annotations

from typing import Any, Dict, Iterable, Optional
from urllib.parse import urlparse

from scripts.pipeline.contracts import SCHEMA_VERSION


_DRAFT_TEXT_TOKENS = (
    "draft saved",
    "saved as draft",
    "saved for later",
    "progress saved",
    "responses saved",
    "your progress has been saved",
    "szkic zapisany",
    "zapisano szkic",
    "postep zapisany",
    "postęp zapisany",
)

_DRAFT_URL_TOKENS = (
    "draft",
    "saved",
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


def build_draft_confirmation_state(
    *,
    screen_state: Dict[str, Any],
    page_data: Optional[Dict[str, Any]] = None,
    region_payload: Optional[Dict[str, Any]] = None,
    previous_url: str = "",
    current_url: str = "",
) -> Dict[str, Any]:
    joined = "\n".join(
        text.strip().lower() for text in _iter_texts(screen_state, page_data, region_payload) if str(text or "").strip()
    )
    text_confirmed = any(token in joined for token in _DRAFT_TEXT_TOKENS)
    current = str(current_url or "")
    previous = str(previous_url or "")
    parsed_url = urlparse(current)
    path_and_query = f"{parsed_url.path} {parsed_url.query}".lower()
    url_confirmed = any(token in path_and_query for token in _DRAFT_URL_TOKENS) and bool(previous and current and previous.rstrip("/") != current.rstrip("/"))
    is_confirmed = bool(text_confirmed or url_confirmed)
    confirmation_kind = "none"
    if text_confirmed:
        confirmation_kind = "text"
    elif url_confirmed:
        confirmation_kind = "url"
    return {
        "schema_version": SCHEMA_VERSION,
        "is_confirmed": is_confirmed,
        "confirmation_kind": confirmation_kind,
        "signals": {
            "draft_text": bool(text_confirmed),
            "draft_url": bool(url_confirmed),
        },
    }
