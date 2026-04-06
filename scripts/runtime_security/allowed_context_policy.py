from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from scripts.pipeline.contracts import SCHEMA_VERSION


_SURVEY_TOKENS = ("question", "survey", "quiz", "ankieta", "pytanie", "wybierz", "odpowiedz", "next")
_LOGIN_TOKENS = ("sign in", "log in", "login", "zaloguj", "konto")
_PAYMENT_TOKENS = ("checkout", "payment", "card", "credit card", "billing", "platn", "kup", "buy now")
_NEWSLETTER_TOKENS = ("newsletter", "subscribe", "subskryb", "join our list")
_LEGAL_TOKENS = ("terms", "privacy", "regulamin", "polityka prywatnosci", "cookies")


def _iter_texts(screen_state: Dict[str, Any], page_data: Optional[Dict[str, Any]]) -> Iterable[str]:
    yield str(screen_state.get("question_text") or "")
    for row in screen_state.get("options") or []:
        if isinstance(row, dict):
            yield str(row.get("text") or "")
    if isinstance(page_data, dict):
        yield str(page_data.get("title") or "")
        yield str(page_data.get("pageText") or page_data.get("page_text") or "")


def _detect_page_intent(screen_state: Dict[str, Any], page_data: Optional[Dict[str, Any]]) -> str:
    joined = "\n".join(str(text or "").lower() for text in _iter_texts(screen_state, page_data) if str(text or "").strip())
    if any(token in joined for token in _LOGIN_TOKENS):
        return "login"
    if any(token in joined for token in _PAYMENT_TOKENS):
        return "payment"
    if any(token in joined for token in _NEWSLETTER_TOKENS):
        return "newsletter"
    if any(token in joined for token in _LEGAL_TOKENS):
        return "legal"
    if any(token in joined for token in _SURVEY_TOKENS):
        return "survey"
    return "unknown"


def build_allowed_context_policy(
    *,
    screen_state: Dict[str, Any],
    page_data: Optional[Dict[str, Any]] = None,
    allowlist_state: Optional[Dict[str, Any]] = None,
    wrong_context_recovery_policy: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    allowlist = allowlist_state or {}
    wrong_context = wrong_context_recovery_policy or {}
    page_intent = _detect_page_intent(screen_state, page_data)
    frames = (page_data or {}).get("frames") if isinstance((page_data or {}).get("frames"), list) else []
    current_host = str(allowlist.get("current_host") or "").strip().lower()
    foreign_frame_count = 0
    for row in frames:
        if not isinstance(row, dict):
            continue
        frame_url = str(row.get("url") or "").strip().lower()
        if current_host and frame_url and current_host not in frame_url:
            foreign_frame_count += 1

    decision = "allow"
    reason = "survey_context_ok"
    if not bool(allowlist.get("is_allowed", True)) and allowlist:
        decision = "block"
        reason = "domain_not_allowlisted"
    elif str(wrong_context.get("recommended_action") or "") in {"pause_session", "refocus_tab"}:
        decision = "block"
        reason = str(wrong_context.get("recommended_action") or "wrong_context")
    elif page_intent in {"login", "payment", "newsletter"}:
        decision = "block"
        reason = f"page_intent:{page_intent}"
    elif foreign_frame_count > 0:
        decision = "warn"
        reason = "foreign_frame_context"
    elif page_intent == "legal":
        decision = "warn"
        reason = "page_intent:legal"

    return {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "reason": reason,
        "is_allowed": decision == "allow",
        "requires_review": decision == "warn",
        "page_intent": page_intent,
        "foreign_frame_count": int(foreign_frame_count),
        "current_host": current_host,
        "signals": {
            "allowlist_allowed": bool(allowlist.get("is_allowed", True)),
            "wrong_context_action": str(wrong_context.get("recommended_action") or "continue"),
        },
    }
