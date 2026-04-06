from __future__ import annotations

from typing import Any, Dict
from urllib.parse import urlparse

from scripts.pipeline.contracts import SCHEMA_VERSION


def _origin(url: str) -> str:
    parsed = urlparse(str(url or ""))
    if not parsed.scheme or not parsed.netloc:
        return ""
    return f"{parsed.scheme}://{parsed.netloc}"


def apply_privacy_reset_contract(*, page: Any, start_url: str) -> Dict[str, Any]:
    reset_cookies = False
    reset_storage = False
    reset_reload = False
    errors: list[str] = []
    origin = _origin(start_url)

    try:
        context_attr = getattr(page, "context", None)
        if context_attr is not None and hasattr(context_attr, "clear_cookies"):
            context = context_attr
        elif callable(context_attr):
            context = context_attr()
        else:
            context = None
        clear_cookies = getattr(context, "clear_cookies", None)
        if callable(clear_cookies):
            clear_cookies()
            reset_cookies = True
    except Exception as exc:
        errors.append(f"clear_cookies:{exc.__class__.__name__}")

    if origin:
        try:
            page.evaluate(
                """(targetOrigin) => {
                    try {
                        if (window.location.origin !== targetOrigin) {
                            return { cleared: false, reason: "origin_mismatch" };
                        }
                        window.localStorage.clear();
                        window.sessionStorage.clear();
                        return { cleared: true, reason: "cleared" };
                    } catch (err) {
                        return { cleared: false, reason: String(err && err.message || err) };
                    }
                }""",
                origin,
            )
            reset_storage = True
        except Exception as exc:
            errors.append(f"clear_storage:{exc.__class__.__name__}")

    try:
        page.goto(start_url, wait_until="domcontentloaded")
        page.wait_for_load_state("networkidle")
        reset_reload = True
    except Exception as exc:
        errors.append(f"reload:{exc.__class__.__name__}")

    is_applied = reset_cookies and reset_storage and reset_reload
    return {
        "schema_version": SCHEMA_VERSION,
        "decision": "applied" if is_applied else "partial",
        "reason": "privacy_reset_applied" if is_applied else "privacy_reset_incomplete",
        "is_applied": is_applied,
        "requires_review": not is_applied,
        "signals": {
            "origin": origin,
            "cookies_cleared": reset_cookies,
            "storage_cleared": reset_storage,
            "reloaded": reset_reload,
            "errors": errors,
        },
    }
