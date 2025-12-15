"""
Fusion of DOM clickables and rated screen boxes into unified Element objects.

Inputs:
- rate_summary.json (from rating.py)
- dom_live/current_clickables.json (from ai_recorder_live.py)

Output:
- list[Element] with merged DOM + OCR + rating fields and fused scores.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .types import BBox, Element


DEFAULT_LABELS: Tuple[str, ...] = (
    "next_active",
    "next_inactive",
    "cookie_accept",
    "cookie_reject",
    "answer_single",
    "answer_multi",
    "dropdown",
    "option",
)


@dataclass
class FusionConfig:
    overlap_iou_threshold: float = 0.25
    text_similarity_threshold: float = 0.50
    fuse_labels: Tuple[str, ...] = DEFAULT_LABELS
    disabled_keys: Tuple[str, ...] = ("disabled", "aria-disabled")
    dom_weight: float = 1.0
    rated_weight: float = 1.0
    noisy_or_floor: float = 0.0
    meta_fields: Tuple[str, ...] = ("role", "tag", "aria-label", "title")
    include_dom_orphans: bool = True
    dom_orphan_min_score: float = 0.25
    max_dom_orphans: int = 50


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
    # Fallback: top_labels entries may still be useful
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


def extract_text(node: Dict[str, Any]) -> str:
    if not node:
        return ""
    for key in ("text", "inner_text", "value", "placeholder", "aria-label", "title"):
        val = node.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def noisy_or(score_dom: float, score_rated: float, floor: float = 0.0) -> float:
    dom = max(floor, min(1.0, score_dom))
    rated = max(floor, min(1.0, score_rated))
    return 1.0 - (1.0 - dom) * (1.0 - rated)


def score_from_dom(node: Dict[str, Any], labels: Tuple[str, ...], cfg: FusionConfig) -> Dict[str, float]:
    """
    Lightweight heuristics based on DOM attributes.
    This is intentionally simple; extend with better rules or nano model later.
    """
    scores: Dict[str, float] = {label: 0.0 for label in labels}
    if not node:
        return scores

    role = (node.get("role") or node.get("aria-role") or "").lower()
    text = (node.get("text") or node.get("inner_text") or "").strip().lower()
    aria_label = (node.get("aria-label") or node.get("label") or "").strip().lower()
    combined_text = f"{text} {aria_label}".strip()

    # Basic signals
    if role in {"button", "submit"}:
        scores["next_active"] = 0.2
        scores["cookie_accept"] = 0.1
    if role in {"checkbox", "radio"}:
        scores["option"] = 0.4
    if "next" in combined_text or "dalej" in combined_text:
        scores["next_active"] = max(scores["next_active"], 0.6)
    if "accept" in combined_text or "zgoda" in combined_text:
        scores["cookie_accept"] = max(scores["cookie_accept"], 0.6)
    if "reject" in combined_text or "odrzuc" in combined_text:
        scores["cookie_reject"] = max(scores["cookie_reject"], 0.6)
    if "select" in combined_text or "wybierz" in combined_text:
        scores["dropdown"] = max(scores["dropdown"], 0.3)

    # Disabled guard
    disabled = any(str(node.get(k)).lower() in {"true", "1"} for k in cfg.disabled_keys)
    if disabled:
        scores["next_active"] = 0.0
    return scores


def match_dom_node(
    rated_item: Dict[str, Any],
    dom_nodes: Iterable[Dict[str, Any]],
    cfg: FusionConfig,
) -> Tuple[Optional[Dict[str, Any]], float]:
    bbox_rated = bbox_from_item(rated_item)
    text_rated = (rated_item.get("text") or "").strip()
    best = None
    best_score = 0.0
    for node in dom_nodes:
        bbox_dom = bbox_from_item(node)
        if bbox_rated and bbox_dom:
            overlap = iou(bbox_rated, bbox_dom)
        else:
            overlap = 0.0
        ts = text_similarity(text_rated, (node.get("text") or node.get("inner_text") or ""))
        score = (overlap >= cfg.overlap_iou_threshold) * 0.6 + ts * 0.4
        if score > best_score:
            best = node
            best_score = score
    return best, best_score


def build_ui_hints(dom_node: Optional[Dict[str, Any]], rated_scores: Dict[str, float], fused_scores: Dict[str, float]) -> Dict[str, Any]:
    node = dom_node or {}
    role = str(node.get("role") or node.get("aria-role") or "").lower()
    tag = str(node.get("tag") or node.get("nodeName") or "").lower()
    input_type = str(node.get("type") or node.get("input_type") or "").lower()
    aria_expanded = node.get("aria-expanded")
    aria_haspopup = node.get("aria-haspopup")
    text = extract_text(node).lower()

    dropdown_dom = role in {"combobox", "listbox"} or tag in {"select", "option"} or input_type in {
        "select-one",
        "select-multiple",
    }
    dropdown_text = any(token in text for token in ("dropdown", "wybierz", "select", "rozwin")) if text else False
    dropdown_score = fused_scores.get("dropdown", 0.0)
    dropdown_like = dropdown_dom or dropdown_text or dropdown_score >= 0.35 or rated_scores.get("dropdown", 0.0) >= 0.45

    interactive_roles = {"button", "submit", "link", "option", "checkbox", "radio", "combobox", "listbox"}
    interactive_tags = {"button", "a", "input", "select", "option", "label"}
    interactive = role in interactive_roles or tag in interactive_tags

    return {
        "dom_role": role,
        "dom_tag": tag,
        "input_type": input_type,
        "aria_expanded": aria_expanded,
        "aria_haspopup": aria_haspopup,
        "dropdown_like": dropdown_like,
        "interactive": interactive,
    }


def fuse_scores(
    rated_scores: Dict[str, float],
    dom_scores: Dict[str, float],
    cfg: FusionConfig,
) -> Tuple[Dict[str, float], Dict[str, str]]:
    fused = {}
    source_priority: Dict[str, str] = {}
    for label in cfg.fuse_labels:
        dom_val = dom_scores.get(label, 0.0) * cfg.dom_weight
        rated_val = rated_scores.get(label, 0.0) * cfg.rated_weight
        if label in {"dropdown", "option"}:
            rated_val *= 1.15  # vision often more reliable for dropdown hits
        if label in {"next_active", "next_inactive", "cookie_accept", "cookie_reject"}:
            dom_val *= 1.15  # DOM semantics more reliable for navigation/cookies
        fused[label] = noisy_or(dom_val, rated_val, cfg.noisy_or_floor)
        if dom_val or rated_val:
            source_priority[label] = "dom" if dom_val >= rated_val else "rated"
    return fused, source_priority


def build_dom_orphans(dom_nodes: List[Dict[str, Any]], used_ids: Set[str], cfg: FusionConfig) -> List[Element]:
    orphans: List[Element] = []
    for idx, node in enumerate(dom_nodes):
        node_id = str(node.get("id") or node.get("uid") or node.get("el_id") or f"dom_{idx}")
        if node_id in used_ids:
            continue
        bbox = bbox_from_item(node)
        if bbox is None:
            continue
        dom_scores = score_from_dom(node, cfg.fuse_labels, cfg)
        top_score = max(dom_scores.values()) if dom_scores else 0.0
        if top_score < cfg.dom_orphan_min_score:
            continue
        fused_scores, source_priority = fuse_scores({}, dom_scores, cfg)
        ui_hints = build_ui_hints(node, {}, fused_scores)
        orphans.append(
            Element(
                id=node_id,
                bbox=bbox,
                text_dom=node.get("text") or node.get("inner_text"),
                dom=node,
                rated={},
                scores_fused=fused_scores,
                metadata={
                    "matched_dom": True,
                    "dom_match_score": 1.0,
                    "source_priority": source_priority,
                    "ui_hints": ui_hints,
                    "origin": "dom_only",
                },
            )
        )
    orphans.sort(key=lambda el: (el.bbox[1], el.bbox[0]))
    return orphans[: cfg.max_dom_orphans]

def fuse(rate_summary: Dict[str, Any], dom_clickables: Dict[str, Any], cfg: Optional[FusionConfig] = None) -> List[Element]:
    """
    Merge rated items with DOM nodes into Element objects.

    The function is deliberately conservative: it keeps data as-is and records
    fused scores while leaving complex heuristics to be refined later.
    """
    cfg = cfg or FusionConfig()
    dom_nodes = list(iter_dom_nodes(dom_clickables))
    elements: List[Element] = []
    used_dom_ids: Set[str] = set()
    for rated in iter_rated_items(rate_summary):
        bbox = bbox_from_item(rated)
        if bbox is None:
            continue
        dom_node, match_score = match_dom_node(rated, dom_nodes, cfg)
        if dom_node:
            dom_id = str(dom_node.get("id") or dom_node.get("uid") or dom_node.get("el_id") or len(used_dom_ids))
            used_dom_ids.add(dom_id)
        rated_scores = rated.get("scores") or rated.get("score") or {}
        dom_scores = score_from_dom(dom_node or {}, cfg.fuse_labels, cfg)
        fused_scores, source_priority = fuse_scores(rated_scores, dom_scores, cfg)
        ui_hints = build_ui_hints(dom_node, rated_scores, fused_scores)

        elem = Element(
            id=str(rated.get("id") or rated.get("uid") or len(elements)),
            bbox=bbox,
            text_dom=(dom_node or {}).get("text") or (dom_node or {}).get("inner_text"),
            text_ocr=rated.get("text"),
            dom=dom_node or {},
            rated=rated,
            scores_fused=fused_scores,
            metadata={
                "matched_dom": bool(dom_node),
                "dom_match_score": match_score,
                "source_priority": source_priority,
                "source_scores": {"dom": dom_scores, "rated": rated_scores},
                "ui_hints": ui_hints,
                "origin": "rated_first",
            },
        )
        elements.append(elem)

    if cfg.include_dom_orphans:
        elements.extend(build_dom_orphans(dom_nodes, used_dom_ids, cfg))

    return elements


def load_and_fuse(
    rate_summary_path: Path,
    dom_clickables_path: Path,
    cfg: Optional[FusionConfig] = None,
) -> List[Element]:
    rate_summary = load_json(rate_summary_path)
    dom_clickables = load_json(dom_clickables_path)
    return fuse(rate_summary, dom_clickables, cfg)
