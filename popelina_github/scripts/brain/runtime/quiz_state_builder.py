from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def _resolve_rated_path(summary_path: Optional[Path], screenshot_path: Optional[Path]) -> Optional[Path]:
    cand: List[Path] = []
    if isinstance(summary_path, Path):
        try:
            stem = summary_path.stem
            if stem.endswith("_summary"):
                base = stem[: -len("_summary")]
                cand.append(summary_path.parent.parent / "rate_results" / f"{base}_rated.json")
        except Exception:
            pass
    if isinstance(screenshot_path, Path):
        try:
            cand.append(screenshot_path.parent.parent / "rate" / "rate_results" / f"{screenshot_path.stem}_rated.json")
        except Exception:
            pass
    for p in cand:
        if isinstance(p, Path) and p.exists():
            return p
    return None


def build_quiz_state(
    *,
    summary_path: Path,
    region_json_path: Optional[Path],
    screenshot_path: Optional[Path],
    question_path: Optional[Path],
    controls_path: Optional[Path],
    page_path: Optional[Path],
) -> Dict[str, Any]:
    dom_fallback_active = str(os.environ.get("FULLBOT_DOM_FALLBACK_ACTIVE", "0") or "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    summary_data = _load_json(summary_path)
    if isinstance(summary_data, dict):
        summary_data["_source"] = str(summary_path)
    rated_path = _resolve_rated_path(summary_path, screenshot_path)
    rated_data = _load_json(rated_path)
    if isinstance(rated_data, dict) and rated_path is not None:
        rated_data["_source"] = str(rated_path)
    region_payload = _load_json(region_json_path)
    if isinstance(region_payload, dict) and region_json_path is not None:
        region_payload["_source"] = str(region_json_path)
    question_data = _load_json(question_path) if dom_fallback_active else None
    if isinstance(question_data, dict) and question_path is not None:
        question_data["_source"] = str(question_path)
    controls_data = _load_json(controls_path) if dom_fallback_active else None
    if isinstance(controls_data, dict) and controls_path is not None:
        controls_data["_source"] = str(controls_path)
    page_data = _load_json(page_path) if dom_fallback_active else None
    if isinstance(page_data, dict) and page_path is not None:
        page_data["_source"] = str(page_path)
    screen_state = parse_screen_quiz_state(
        region_payload=region_payload,
        summary_data=summary_data,
        page_data=page_data,
        rated_data=rated_data,
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
        "rated_data": rated_data,
    }
