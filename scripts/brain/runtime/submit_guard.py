from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from scripts.pipeline.contracts import SCHEMA_VERSION


_CAPTCHA_TOKENS = (
    "captcha",
    "recaptcha",
    "hcaptcha",
    "cloudflare",
    "turnstile",
    "verify you are human",
    "i am human",
    "robot",
    "anti-bot",
    "security check",
    "challenge",
)


def _iter_texts(
    screen_state: Dict[str, Any],
    page_data: Optional[Dict[str, Any]],
    region_payload: Optional[Dict[str, Any]],
) -> Iterable[str]:
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
        for row in page_data.get("frames") or []:
            if isinstance(row, dict):
                yield str(row.get("url") or "")
                yield str(row.get("name") or "")
    if isinstance(region_payload, dict):
        for row in region_payload.get("results") or []:
            if isinstance(row, dict):
                yield str(row.get("text") or row.get("box_text") or "")


def build_submit_guard_state(
    *,
    screen_state: Dict[str, Any],
    page_data: Optional[Dict[str, Any]] = None,
    region_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    texts = [str(text or "").strip() for text in _iter_texts(screen_state, page_data, region_payload) if str(text or "").strip()]
    joined = "\n".join(text.lower() for text in texts)
    tokens = sorted({token for token in _CAPTCHA_TOKENS if token in joined})
    has_captcha = bool(tokens)
    iframe_hint = "recaptcha" in joined or "hcaptcha" in joined or "turnstile" in joined
    challenge_kind = "captcha" if has_captcha else "none"
    if "cloudflare" in tokens or "turnstile" in tokens:
        challenge_kind = "turnstile"
    elif "hcaptcha" in tokens:
        challenge_kind = "hcaptcha"
    elif "recaptcha" in tokens:
        challenge_kind = "recaptcha"

    return {
        "schema_version": SCHEMA_VERSION,
        "is_hard_stop": bool(has_captcha or iframe_hint),
        "stop_reason": challenge_kind if (has_captcha or iframe_hint) else "none",
        "challenge_kind": challenge_kind,
        "matched_tokens": tokens,
        "signals": {
            "captcha_text": bool(has_captcha),
            "iframe_hint": bool(iframe_hint),
        },
    }
