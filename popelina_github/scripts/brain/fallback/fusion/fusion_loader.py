from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from ...shared.types import BBox


def load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def iter_rated_items(rate_summary: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    if not rate_summary:
        return []
    for key in ("elements", "items", "results", "list"):
        if isinstance(rate_summary.get(key), list):
            return rate_summary[key]
    return []


def iter_dom_nodes(dom_clickables: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    if not dom_clickables:
        return []
    if isinstance(dom_clickables.get("clickables"), list):
        return dom_clickables["clickables"]
    if isinstance(dom_clickables.get("items"), list):
        return dom_clickables["items"]
    return []


def bbox_from_item(obj: Dict[str, Any]) -> Optional[BBox]:
    bbox = obj.get("bbox") or obj.get("text_box") or obj.get("dropdown_box")
    if not bbox or len(bbox) < 4:
        return None
    x1, y1, x2, y2 = bbox[:4]
    return int(x1), int(y1), int(x2), int(y2)


def extract_text(node: Dict[str, Any]) -> str:
    if not node:
        return ""
    for key in ("text", "inner_text", "value", "placeholder", "aria-label", "title"):
        val = node.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


