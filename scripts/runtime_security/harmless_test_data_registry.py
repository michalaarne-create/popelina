from __future__ import annotations

import re
from typing import Any, Dict, Optional

from scripts.pipeline.contracts import SCHEMA_VERSION

from .test_data_policy import map_environment_profile

_SAFE_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@([A-Za-z0-9.-]+)$", re.IGNORECASE)
_DIGIT_RE = re.compile(r"\D+")

_DEFAULT_PROFILES: Dict[str, Dict[str, Any]] = {
    "default": {
        "safe_text_values": [
            "Warszawa",
            "Krakow",
            "Gdansk",
            "Wroclaw",
            "Poznan",
            "Test User",
            "QA User",
            "Sample City",
            "Lorem ipsum",
        ],
        "safe_text_prefixes": ["Test ", "QA ", "Sample ", "Demo "],
        "safe_email_domains": ["example.com", "example.org", "example.net", "test.local"],
        "safe_phone_digits": ["123456789", "111222333", "000000000"],
        "safe_postal_values": ["00-001", "11-111", "99-999"],
        "safe_date_values": ["2000-01-01", "2000/01/01", "01/01/2000"],
        "safe_identifier_prefixes": ["QA-", "TEST-", "DEMO-"],
        "safe_identifier_values": ["ABC-12345", "ID-0001"],
    },
    "staging": {
        "safe_text_values": ["Staging City"],
        "safe_identifier_prefixes": ["STAGE-"],
    },
}


def _merge_profile_list(defaults: Dict[str, Any], profile: Dict[str, Any], field: str) -> list[str]:
    merged: list[str] = []
    for source in (defaults.get(field) or []):
        item = str(source or "").strip()
        if item and item not in merged:
            merged.append(item)
    for source in (profile.get(field) or []):
        item = str(source or "").strip()
        if item and item not in merged:
            merged.append(item)
    return merged


def build_harmless_test_data_registry(
    *,
    environment: str = "default",
    profiles: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    source_profiles = dict(_DEFAULT_PROFILES)
    for key, value in (profiles or {}).items():
        if isinstance(value, dict):
            source_profiles[str(key)] = dict(value)
    defaults = dict(source_profiles.get("default") or {})
    profile = map_environment_profile(environment=environment, profiles=source_profiles)
    merged = {
        "safe_text_values": _merge_profile_list(defaults, profile, "safe_text_values"),
        "safe_text_prefixes": _merge_profile_list(defaults, profile, "safe_text_prefixes"),
        "safe_email_domains": _merge_profile_list(defaults, profile, "safe_email_domains"),
        "safe_phone_digits": _merge_profile_list(defaults, profile, "safe_phone_digits"),
        "safe_postal_values": _merge_profile_list(defaults, profile, "safe_postal_values"),
        "safe_date_values": _merge_profile_list(defaults, profile, "safe_date_values"),
        "safe_identifier_prefixes": _merge_profile_list(defaults, profile, "safe_identifier_prefixes"),
        "safe_identifier_values": _merge_profile_list(defaults, profile, "safe_identifier_values"),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "environment": str(environment or "default").strip().lower() or "default",
        "profile": merged,
    }


def evaluate_harmless_test_payload(
    *,
    value: str,
    payload_class: str,
    registry: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    text = str(value or "").strip()
    active_registry = registry or build_harmless_test_data_registry()
    profile = dict(active_registry.get("profile") or {})
    allowed = False
    match = ""
    if payload_class == "safe_email":
        email_match = _SAFE_EMAIL_RE.match(text)
        domain = str(email_match.group(1) if email_match else "").lower()
        if domain and domain in {item.lower() for item in profile.get("safe_email_domains") or []}:
            allowed = True
            match = domain
    elif payload_class == "numeric_masked":
        digits = _DIGIT_RE.sub("", text)
        if digits in {item.lower() for item in profile.get("safe_phone_digits") or []}:
            allowed = True
            match = digits
        elif text.lower() in {item.lower() for item in profile.get("safe_postal_values") or []}:
            allowed = True
            match = text
        elif text.lower() in {item.lower() for item in profile.get("safe_date_values") or []}:
            allowed = True
            match = text
    elif payload_class == "safe_text":
        if text.lower() in {item.lower() for item in profile.get("safe_text_values") or []}:
            allowed = True
            match = text
        else:
            for prefix in profile.get("safe_text_prefixes") or []:
                if text.startswith(str(prefix)):
                    allowed = True
                    match = str(prefix)
                    break
    elif payload_class == "safe_identifier":
        if text.lower() in {item.lower() for item in profile.get("safe_identifier_values") or []}:
            allowed = True
            match = text
        else:
            for prefix in profile.get("safe_identifier_prefixes") or []:
                if text.startswith(str(prefix)):
                    allowed = True
                    match = str(prefix)
                    break
    return {
        "schema_version": SCHEMA_VERSION,
        "decision": "allow" if allowed else "block",
        "reason": "registry_match" if allowed else "payload_not_in_harmless_registry",
        "payload_class": payload_class,
        "registry_match": match,
        "is_allowed": allowed,
    }
