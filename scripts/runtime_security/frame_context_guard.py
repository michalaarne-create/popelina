from __future__ import annotations

from typing import Any, Dict, Optional
from urllib.parse import urlparse

from scripts.pipeline.contracts import SCHEMA_VERSION


def build_frame_context_guard(
    *,
    page_data: Optional[Dict[str, Any]] = None,
    current_url: str = "",
) -> Dict[str, Any]:
    page = page_data or {}
    frames = page.get("frames") if isinstance(page.get("frames"), list) else []
    current_host = str(urlparse(str(current_url or "")).hostname or "").strip().lower()
    foreign_frames = []
    for row in frames:
        if not isinstance(row, dict):
            continue
        frame_url = str(row.get("url") or "").strip()
        frame_host = str(urlparse(frame_url).hostname or "").strip().lower()
        if frame_host and current_host and frame_host != current_host and not frame_host.endswith(f".{current_host}") and not current_host.endswith(f".{frame_host}"):
            foreign_frames.append(
                {
                    "url": frame_url,
                    "host": frame_host,
                    "name": str(row.get("name") or ""),
                }
            )

    decision = "allow"
    reason = "frame_context_clear"
    if foreign_frames:
        decision = "block"
        reason = "foreign_frame_requires_confirmation"

    return {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "reason": reason,
        "is_allowed": decision == "allow",
        "current_host": current_host,
        "foreign_frame_count": len(foreign_frames),
        "foreign_frames": foreign_frames,
    }
