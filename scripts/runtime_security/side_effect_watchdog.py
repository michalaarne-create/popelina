from __future__ import annotations

from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from scripts.pipeline.contracts import SCHEMA_VERSION


_DOWNLOAD_SUFFIXES = (
    ".pdf",
    ".csv",
    ".zip",
    ".xlsx",
    ".xls",
    ".doc",
    ".docx",
    ".ppt",
    ".pptx",
)


def _host(url: str) -> str:
    try:
        return str(urlparse(str(url or "")).hostname or "").strip().lower()
    except Exception:
        return ""


def _download_like_url(url: str) -> bool:
    path = str(urlparse(str(url or "")).path or "").strip().lower()
    return any(path.endswith(suffix) for suffix in _DOWNLOAD_SUFFIXES)


def build_expected_side_effect_map(*, actions_plan: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    allow_request_spike = False
    allow_external_navigation = False
    for action in actions_plan or []:
        kind = str((action or {}).get("kind") or "").strip().lower()
        reason = str((action or {}).get("reason") or "").strip().lower()
        if "submit" in reason or kind == "dom_click_button":
            allow_request_spike = True
        if "submit" in reason:
            allow_external_navigation = True
    return {
        "allow_request_spike": allow_request_spike,
        "allow_external_navigation": allow_external_navigation,
        "allow_popup": False,
        "allow_download": False,
    }


def build_side_effect_watchdog(
    *,
    actions_plan: Optional[List[Dict[str, Any]]] = None,
    previous_url: str = "",
    current_url: str = "",
    domain_allowlist: Optional[Dict[str, Any]] = None,
    context_pages_before: int = 1,
    context_pages_after: int = 1,
    request_count_before: int = 0,
    request_count_after: int = 0,
    popup_events_before: int = 0,
    popup_events_after: int = 0,
    download_events_before: int = 0,
    download_events_after: int = 0,
) -> Dict[str, Any]:
    expected_side_effects = build_expected_side_effect_map(actions_plan=actions_plan)
    previous_host = _host(previous_url)
    current_host = _host(current_url)
    pages_before = max(0, int(context_pages_before or 0))
    pages_after = max(0, int(context_pages_after or 0))
    popup_opened = pages_after > pages_before
    external_navigation = bool(current_host and previous_host and current_host != previous_host)
    allowlist_block = False
    if external_navigation and isinstance(domain_allowlist, dict):
        allowlist_block = not bool(domain_allowlist.get("is_allowed"))
    download_like = _download_like_url(current_url) and str(current_url or "").strip() != str(previous_url or "").strip()
    requests_before = max(0, int(request_count_before or 0))
    requests_after = max(0, int(request_count_after or 0))
    request_delta = max(0, requests_after - requests_before)
    request_spike = request_delta >= 25 or (requests_before > 0 and requests_after >= requests_before * 3 and request_delta >= 10)
    popup_event_delta = max(0, int(popup_events_after or 0) - int(popup_events_before or 0))
    download_event_delta = max(0, int(download_events_after or 0) - int(download_events_before or 0))

    decision = "allow"
    reason = "no_unsafe_side_effects"
    if (popup_opened or popup_event_delta > 0) and not bool(expected_side_effects.get("allow_popup")):
        decision = "block"
        reason = "popup_or_new_window_detected"
    elif allowlist_block and not bool(expected_side_effects.get("allow_external_navigation")):
        decision = "block"
        reason = "external_navigation_detected"
    elif (download_event_delta > 0 or download_like) and not bool(expected_side_effects.get("allow_download")):
        decision = "block"
        reason = "download_like_navigation_detected"
    elif request_spike and not bool(expected_side_effects.get("allow_request_spike")):
        decision = "block"
        reason = "request_spike_detected"

    return {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "reason": reason,
        "is_blocking": decision == "block",
        "requires_review": decision == "block",
        "expected_side_effects": expected_side_effects,
        "signals": {
            "action_count": len(actions_plan or []),
            "previous_host": previous_host,
            "current_host": current_host,
            "context_pages_before": pages_before,
            "context_pages_after": pages_after,
            "popup_opened": popup_opened,
            "external_navigation": external_navigation,
            "download_like_navigation": download_like,
            "request_count_before": requests_before,
            "request_count_after": requests_after,
            "request_delta": request_delta,
            "request_spike": request_spike,
            "popup_event_delta": popup_event_delta,
            "download_event_delta": download_event_delta,
        },
    }
