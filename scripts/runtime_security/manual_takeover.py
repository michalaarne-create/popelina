from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


def build_manual_takeover_state(*, requested: bool, reason: str = "", session_dir: Optional[str | Path] = None) -> Dict[str, Any]:
    marker_path = ""
    if session_dir:
        marker = Path(session_dir) / "manual_takeover.flag"
        marker_path = str(marker)
        if requested:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(reason or "manual takeover", encoding="utf-8")
    return {
        "requested": bool(requested),
        "reason": str(reason or ""),
        "marker_path": marker_path,
    }
