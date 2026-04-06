from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


def build_voice_control_guard_state(
    *,
    requested: bool,
    voice_enabled: bool = False,
    reason: str = "",
    session_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    marker_path = ""
    if session_dir:
        marker = Path(session_dir) / "voice_control_guard.flag"
        marker_path = str(marker)
        if requested and voice_enabled:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(reason or "voice control enabled", encoding="utf-8")

    allowed = bool(requested and voice_enabled)
    guard_reason = "ok" if allowed else ("voice_control_disabled" if requested else "not_requested")
    return {
        "requested": bool(requested),
        "voice_enabled": bool(voice_enabled),
        "is_allowed": allowed,
        "decision": "allow" if allowed else "block",
        "reason": guard_reason,
        "marker_path": marker_path,
        "detail": str(reason or ""),
    }
