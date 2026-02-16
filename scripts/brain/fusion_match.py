from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

from .fusion_loader import bbox_from_item
from .types import BBox


def iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1)
    ub = (bx2 - bx1) * (by2 - by1)
    return inter / float(ua + ub - inter + 1e-9)


def text_similarity(a: Optional[str], b: Optional[str]) -> float:
    if not a or not b:
        return 0.0
    a_norm = a.strip().lower()
    b_norm = b.strip().lower()
    if not a_norm or not b_norm:
        return 0.0
    if a_norm == b_norm:
        return 1.0
    smaller, larger = sorted([a_norm, b_norm], key=len)
    if smaller in larger:
        return len(smaller) / max(1, len(larger))
    return 0.0


def match_dom_node(
    rated_item: Dict[str, Any],
    dom_nodes: Iterable[Dict[str, Any]],
    overlap_iou_threshold: float,
) -> Tuple[Optional[Dict[str, Any]], float]:
    bbox_rated = bbox_from_item(rated_item)
    text_rated = (rated_item.get("text") or "").strip()
    best = None
    best_score = 0.0
    for node in dom_nodes:
        bbox_dom = bbox_from_item(node)
        overlap = iou(bbox_rated, bbox_dom) if bbox_rated and bbox_dom else 0.0
        ts = text_similarity(text_rated, (node.get("text") or node.get("inner_text") or ""))
        score = (overlap >= overlap_iou_threshold) * 0.6 + ts * 0.4
        if score > best_score:
            best = node
            best_score = score
    return best, best_score

