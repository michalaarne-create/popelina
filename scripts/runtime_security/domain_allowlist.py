from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse

DEFAULT_ALLOWLIST_ENV = "FULLBOT_DOMAIN_ALLOWLIST"
DEFAULT_ALLOWED_SCHEMES = ("http", "https")
_SEPARATOR_RE = re.compile(r"[,;\s]+")


def _normalize_hosts(hosts: Iterable[str]) -> List[str]:
    normalized: List[str] = []
    seen = set()
    for value in hosts:
        host = str(value or "").strip().lower()
        if not host:
            continue
        if host.startswith("*."):
            host = host[2:]
        if "://" in host or "/" in host:
            parsed = urlparse(host if "://" in host else f"https://{host}")
            host = str(parsed.hostname or "").lower()
        host = host.split(":")[0].strip().strip(".")
        if not host or host in seen:
            continue
        seen.add(host)
        normalized.append(host)
    return normalized


def parse_allowlist_hosts(value: Optional[str | Iterable[str]]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw = [token for token in _SEPARATOR_RE.split(value) if token]
    else:
        raw = [str(token or "") for token in value]
    return _normalize_hosts(raw)


def _normalize_current_url(current_url: str) -> tuple[str, str, bool]:
    text = str(current_url or "").strip()
    if not text:
        return "", "", False
    parsed = urlparse(text if "://" in text else f"https://{text}")
    scheme = str(parsed.scheme or "").lower()
    host = str(parsed.hostname or "").lower().strip(".")
    return scheme, host, bool(host)


def _match_allowed_host(host: str, allowed_hosts: Iterable[str]) -> str:
    for allowed in allowed_hosts:
        rule = str(allowed or "").strip().lower()
        if not rule:
            continue
        if host == rule or host.endswith(f".{rule}"):
            return rule
    return ""


def evaluate_domain_allowlist(
    *,
    current_url: str,
    allowed_hosts: Optional[Iterable[str]] = None,
    env: Optional[Dict[str, str]] = None,
    allowlist_env: str = DEFAULT_ALLOWLIST_ENV,
    allowed_schemes: Iterable[str] = DEFAULT_ALLOWED_SCHEMES,
) -> Dict[str, Any]:
    source_env = os.environ if env is None else env
    allowlist_source = "explicit"
    if allowed_hosts is None:
        allowlist_source = "env"
        allowed_hosts = parse_allowlist_hosts(str(source_env.get(allowlist_env) or ""))
    allowed = _normalize_hosts(allowed_hosts)
    normalized_schemes = {str(item or "").strip().lower() for item in allowed_schemes if str(item or "").strip()}
    scheme, host, has_host = _normalize_current_url(current_url)
    matched_host = _match_allowed_host(host, allowed) if has_host else ""

    reason = "allowed"
    is_allowed = bool(has_host and matched_host)
    if not has_host:
        reason = "invalid_url"
        is_allowed = False
    elif normalized_schemes and scheme and scheme not in normalized_schemes:
        reason = "disallowed_scheme"
        is_allowed = False
    elif not allowed:
        reason = "allowlist_empty"
        is_allowed = False
    elif not matched_host:
        reason = "host_not_allowlisted"
        is_allowed = False

    return {
        "current_url": str(current_url or ""),
        "current_scheme": scheme,
        "current_host": host,
        "allowed_hosts": allowed,
        "allowlist_source": allowlist_source,
        "allowed_schemes": sorted(normalized_schemes),
        "matched_host": matched_host,
        "is_allowed": bool(is_allowed),
        "reason": reason,
    }
