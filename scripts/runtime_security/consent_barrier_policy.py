from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from scripts.pipeline.contracts import SCHEMA_VERSION


_CONSENT_TOKENS = (
    "cookie consent",
    "accept cookies",
    "cookies",
    "privacy",
    "consent",
    "gdpr",
    "rodo",
    "ciasteczk",
)


def _iter_texts(
    screen_state: Dict[str, Any],
    page_data: Optional[Dict[str, Any]],
    region_payload: Optional[Dict[str, Any]],
) -> Iterable[str]:
    yield str(screen_state.get("question_text") or "")
    if isinstance(page_data, dict):
        yield str(page_data.get("title") or "")
        yield str(page_data.get("pageText") or page_data.get("page_text") or "")
    if isinstance(region_payload, dict):
        for row in region_payload.get("results") or []:
            if isinstance(row, dict):
                yield str(row.get("text") or row.get("box_text") or "")


def build_consent_barrier_policy(
    *,
    screen_state: Dict[str, Any],
    page_data: Optional[Dict[str, Any]] = None,
    region_payload: Optional[Dict[str, Any]] = None,
    anti_modal_guard: Optional[Dict[str, Any]] = None,
    submit_guard_state: Optional[Dict[str, Any]] = None,
    recommended_action: str = "",
) -> Dict[str, Any]:
    submit_guard = submit_guard_state or {}
    anti_modal = anti_modal_guard or {}
    joined = "\n".join(
        str(text or "").lower() for text in _iter_texts(screen_state, page_data, region_payload) if str(text or "").strip()
    )
    matched = sorted({token for token in _CONSENT_TOKENS if token in joined})
    challenge_kind = str(submit_guard.get("challenge_kind") or "none")
    action = str(recommended_action or "").strip().lower()

    decision = "allow"
    reason = "clear"
    barrier_kind = "none"
    if challenge_kind not in {"", "none"}:
        decision = "block"
        reason = f"challenge:{challenge_kind}"
        barrier_kind = "captcha"
    elif matched or any(
        consent_token in str(token or "").lower()
        for token in (anti_modal.get("matched_tokens") or [])
        for consent_token in _CONSENT_TOKENS
    ):
        barrier_kind = "consent"
        if action == "click_cookies_accept":
            decision = "block"
            reason = "consent_accept_not_allowed"
        else:
            decision = "block"
            reason = "consent_requires_operator_review"

    return {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "reason": reason,
        "barrier_kind": barrier_kind,
        "is_blocking": decision == "block",
        "requires_review": barrier_kind == "consent",
        "matched_tokens": matched,
        "signals": {
            "challenge_kind": challenge_kind,
            "recommended_action": action,
            "anti_modal_blocking": bool(anti_modal.get("is_blocking")),
        },
    }
