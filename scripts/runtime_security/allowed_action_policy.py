from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

from scripts.pipeline.contracts import SCHEMA_VERSION
from .harmless_test_data_registry import build_harmless_test_data_registry, evaluate_harmless_test_payload


_FORBIDDEN_TOKENS = (
    "buy",
    "pay",
    "payment",
    "checkout",
    "billing",
    "credit card",
    "delete",
    "remove account",
    "subscribe",
    "newsletter",
    "install",
    "download",
    "login",
    "log in",
    "sign in",
    "share",
)

_SAFE_TEXT_RE = re.compile(r"^[A-Za-zÀ-ÿ0-9 .,'/_()#-]{1,80}$")
_SAFE_PHONE_RE = re.compile(r"^[0-9 +()/-]{5,24}$")
_SAFE_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@(example\.(com|org|net)|test\.local)$", re.IGNORECASE)
_SAFE_URL_RE = re.compile(r"^https?://", re.IGNORECASE)
_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z]{2,8}-[A-Za-z0-9]{2,12}$")


def _extract_target_texts(action: Dict[str, Any]) -> List[str]:
    texts: List[str] = []
    reason = str(action.get("reason") or "").strip()
    if ":" in reason:
        suffix = reason.split(":", 1)[1].strip()
        if suffix:
            texts.append(suffix)
    text = str(action.get("text") or "").strip()
    if text:
        texts.append(text)
    for value in action.get("texts") or []:
        candidate = str(value or "").strip()
        if candidate:
            texts.append(candidate)
    return texts


def classify_action_risk_tier(action: Dict[str, Any]) -> str:
    kind = str(action.get("kind") or "").strip().lower()
    reason = str(action.get("reason") or "").strip().lower()
    if "submit" in reason:
        return "critical"
    if kind == "navigate_back":
        return "medium"
    if kind == "dom_click_button":
        if "save_draft" in reason:
            return "high"
        return "critical"
    if kind in {"type_text", "dom_fill_masked_input", "dom_pick_autocomplete_option"}:
        return "medium"
    if kind == "screen_click" and "next" in reason:
        return "high"
    if kind in {"screen_click", "dom_select_option"}:
        return "low"
    if kind in {"key_press", "key_repeat"}:
        return "low"
    return "medium"


def build_forbidden_action_catalog(action: Dict[str, Any]) -> Dict[str, Any]:
    matched_token = ""
    matched_text = ""
    for text in _extract_target_texts(action):
        lowered = text.lower()
        for token in _FORBIDDEN_TOKENS:
            if token in lowered:
                matched_token = token
                matched_text = text
                break
        if matched_token:
            break
    return {
        "schema_version": SCHEMA_VERSION,
        "decision": "block" if matched_token else "allow",
        "reason": f"forbidden_token:{matched_token}" if matched_token else "clear",
        "matched_token": matched_token,
        "matched_text": matched_text,
        "is_allowed": not bool(matched_token),
    }


def evaluate_input_content_policy(action: Dict[str, Any]) -> Dict[str, Any]:
    kind = str(action.get("kind") or "").strip().lower()
    value = str(action.get("text") or "").strip()
    registry = build_harmless_test_data_registry()
    if kind not in {"type_text", "dom_fill_masked_input", "dom_pick_autocomplete_option"}:
        return {
            "schema_version": SCHEMA_VERSION,
            "decision": "allow",
            "reason": "not_textual",
            "payload_class": "n/a",
            "harmless_test_data_registry": {"decision": "skip", "reason": "not_textual", "is_allowed": True},
            "is_allowed": True,
        }
    if not value:
        return {
            "schema_version": SCHEMA_VERSION,
            "decision": "block",
            "reason": "empty_text_payload",
            "payload_class": "empty",
            "harmless_test_data_registry": {"decision": "skip", "reason": "empty_payload", "is_allowed": False},
            "is_allowed": False,
        }
    if _SAFE_URL_RE.match(value):
        return {
            "schema_version": SCHEMA_VERSION,
            "decision": "block",
            "reason": "url_payload_not_allowed",
            "payload_class": "url",
            "harmless_test_data_registry": {"decision": "skip", "reason": "url_payload_not_allowed", "is_allowed": False},
            "is_allowed": False,
        }
    if _SAFE_EMAIL_RE.match(value):
        payload_class = "safe_email"
    elif _SAFE_PHONE_RE.match(value):
        payload_class = "numeric_masked"
    elif _SAFE_IDENTIFIER_RE.match(value):
        payload_class = "safe_identifier"
    elif _SAFE_TEXT_RE.match(value):
        payload_class = "safe_text"
    else:
        return {
            "schema_version": SCHEMA_VERSION,
            "decision": "block",
            "reason": "payload_class_not_approved",
            "payload_class": "unknown",
            "harmless_test_data_registry": {"decision": "skip", "reason": "payload_class_not_approved", "is_allowed": False},
            "is_allowed": False,
        }
    harmless_registry = evaluate_harmless_test_payload(value=value, payload_class=payload_class, registry=registry)
    if not harmless_registry.get("is_allowed"):
        return {
            "schema_version": SCHEMA_VERSION,
            "decision": "block",
            "reason": str(harmless_registry.get("reason") or "payload_not_in_harmless_registry"),
            "payload_class": payload_class,
            "harmless_test_data_registry": harmless_registry,
            "is_allowed": False,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "decision": "allow",
        "reason": "approved_test_payload",
        "payload_class": payload_class,
        "harmless_test_data_registry": harmless_registry,
        "is_allowed": True,
    }


def _max_risk_tier(items: Iterable[str]) -> str:
    order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    best = "low"
    best_rank = -1
    for item in items:
        rank = order.get(str(item or "").strip().lower(), -1)
        if rank > best_rank:
            best = str(item or "low")
            best_rank = rank
    return best


def build_allowed_action_policy(
    *,
    actions_plan: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    per_action: List[Dict[str, Any]] = []
    blocked_reason = ""
    for index, action in enumerate(actions_plan or []):
        risk_tier = classify_action_risk_tier(action)
        forbidden = build_forbidden_action_catalog(action)
        input_policy = evaluate_input_content_policy(action)
        option_risk = "review" if risk_tier in {"high", "critical"} and any(_extract_target_texts(action)) else "low"
        row = {
            "index": int(index),
            "kind": str(action.get("kind") or ""),
            "reason": str(action.get("reason") or ""),
            "risk_tier": risk_tier,
            "forbidden_action_catalog": forbidden,
            "input_content_policy": input_policy,
            "option_risk_classifier": {
                "decision": option_risk,
                "reason": "elevated_target_text" if option_risk == "review" else "clear",
            },
            "harmless_test_data_registry": input_policy.get("harmless_test_data_registry") if isinstance(input_policy, dict) else {},
        }
        per_action.append(row)
        if not forbidden.get("is_allowed"):
            blocked_reason = str(forbidden.get("reason") or "forbidden_action")
            break
        if not input_policy.get("is_allowed"):
            blocked_reason = str(input_policy.get("reason") or "disallowed_input_payload")
            break

    decision = "allow" if not blocked_reason else "block"
    return {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "reason": blocked_reason or "allowed_actions_ok",
        "is_allowed": not bool(blocked_reason),
        "requires_review": False,
        "max_risk_tier": _max_risk_tier(row.get("risk_tier") for row in per_action),
        "action_risk_tier": {
            "decision": "classified",
            "max_risk_tier": _max_risk_tier(row.get("risk_tier") for row in per_action),
        },
        "per_action": per_action,
        "signals": {
            "blocked_actions": int(sum(1 for row in per_action if not row["forbidden_action_catalog"].get("is_allowed", True))),
            "blocked_payloads": int(sum(1 for row in per_action if not row["input_content_policy"].get("is_allowed", True))),
        },
    }
