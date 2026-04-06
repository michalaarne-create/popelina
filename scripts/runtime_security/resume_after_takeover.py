from __future__ import annotations

from typing import Any, Dict, Optional


def build_resume_state(*, previous_runtime_state: Optional[Dict[str, Any]] = None, stable_signature: str = "", current_url: str = "") -> Dict[str, Any]:
    runtime = previous_runtime_state or {}
    context = runtime.get("context") if isinstance(runtime.get("context"), dict) else {}
    return {
        "can_resume": bool(stable_signature or context.get("current_signature")),
        "resume_signature": str(stable_signature or context.get("current_signature") or ""),
        "resume_url": str(current_url or context.get("current_url") or ""),
    }
