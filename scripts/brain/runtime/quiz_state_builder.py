from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from .screen_quiz_parser import parse_screen_quiz_state
from .quiz_utils import md5_text


def _load_json(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _page_signature(page_data: Optional[Dict[str, Any]], question_data: Optional[Dict[str, Any]]) -> Optional[str]:
    page = page_data or {}
    if page.get("page_signature"):
        return str(page.get("page_signature"))
    url = str(page.get("url") or (question_data or {}).get("url") or "")
    title = str(page.get("title") or (question_data or {}).get("title") or "")
    viewport = page.get("viewport") or (question_data or {}).get("viewport") or {}
    if not any((url, title, viewport)):
        return None
    return md5_text(f"{url}|{title}|{viewport}")


def build_quiz_state(
    *,
    summary_path: Path,
    region_json_path: Optional[Path],
    screenshot_path: Optional[Path],
    question_path: Optional[Path],
    controls_path: Optional[Path],
    page_path: Optional[Path],
) -> Dict[str, Any]:
    summary_data = _load_json(summary_path)
    if isinstance(summary_data, dict):
        summary_data["_source"] = str(summary_path)
    region_payload = _load_json(region_json_path)
    if isinstance(region_payload, dict) and region_json_path is not None:
        region_payload["_source"] = str(region_json_path)
    question_data = _load_json(question_path)
    if isinstance(question_data, dict) and question_path is not None:
        question_data["_source"] = str(question_path)
    controls_data = _load_json(controls_path)
    if isinstance(controls_data, dict) and controls_path is not None:
        controls_data["_source"] = str(controls_path)
    page_data = _load_json(page_path)
    if isinstance(page_data, dict) and page_path is not None:
        page_data["_source"] = str(page_path)
    screen_state = parse_screen_quiz_state(
        region_payload=region_payload,
        summary_data=summary_data,
        page_data=page_data,
    )
    page_sig = _page_signature(page_data, question_data)
    if page_sig:
        screen_state["page_signature"] = page_sig
    if screenshot_path is not None:
        screen_state["screenshot_path"] = str(screenshot_path)
    return {
        "summary_data": summary_data,
        "region_payload": region_payload,
        "question_data": question_data,
        "controls_data": controls_data,
        "page_data": page_data,
        "screen_state": screen_state,
    }
