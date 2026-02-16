from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .fusion_loader import bbox_from_item, extract_text, iter_dom_nodes, iter_rated_items, load_json
from .fusion_match import match_dom_node
from .fusion_score import FusionConfig, fuse_scores, score_from_dom
from .types import Element


def build_ui_hints(dom_node: Optional[Dict[str, Any]], rated_scores: Dict[str, float], fused_scores: Dict[str, float]) -> Dict[str, Any]:
    node = dom_node or {}
    role = str(node.get("role") or node.get("aria-role") or "").lower()
    tag = str(node.get("tag") or node.get("nodeName") or "").lower()
    input_type = str(node.get("type") or node.get("input_type") or "").lower()
    text = extract_text(node).lower()
    dropdown_dom = role in {"combobox", "listbox"} or tag in {"select", "option"} or input_type in {"select-one", "select-multiple"}
    dropdown_text = any(token in text for token in ("dropdown", "wybierz", "select", "rozwin")) if text else False
    dropdown_like = dropdown_dom or dropdown_text or fused_scores.get("dropdown", 0.0) >= 0.35 or rated_scores.get("dropdown", 0.0) >= 0.45
    interactive = role in {"button", "submit", "link", "option", "checkbox", "radio", "combobox", "listbox"} or tag in {"button", "a", "input", "select", "option", "label"}
    return {"dom_role": role, "dom_tag": tag, "input_type": input_type, "aria_expanded": node.get("aria-expanded"), "aria_haspopup": node.get("aria-haspopup"), "dropdown_like": dropdown_like, "interactive": interactive}


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
        if (max(dom_scores.values()) if dom_scores else 0.0) < cfg.dom_orphan_min_score:
            continue
        fused, src = fuse_scores({}, dom_scores, cfg)
        orphans.append(Element(id=node_id, bbox=bbox, text_dom=node.get("text") or node.get("inner_text"), dom=node, rated={}, scores_fused=fused, metadata={"matched_dom": True, "dom_match_score": 1.0, "source_priority": src, "ui_hints": build_ui_hints(node, {}, fused), "origin": "dom_only"}))
    orphans.sort(key=lambda el: (el.bbox[1], el.bbox[0]))
    return orphans[: cfg.max_dom_orphans]


def fuse(rate_summary: Dict[str, Any], dom_clickables: Dict[str, Any], cfg: Optional[FusionConfig] = None) -> List[Element]:
    cfg = cfg or FusionConfig()
    dom_nodes = list(iter_dom_nodes(dom_clickables))
    elements: List[Element] = []
    used_dom_ids: Set[str] = set()
    for rated in iter_rated_items(rate_summary):
        bbox = bbox_from_item(rated)
        if bbox is None:
            continue
        dom_node, match_score = match_dom_node(rated, dom_nodes, cfg.overlap_iou_threshold)
        if dom_node:
            used_dom_ids.add(str(dom_node.get("id") or dom_node.get("uid") or dom_node.get("el_id") or len(used_dom_ids)))
        rated_scores = rated.get("scores") or rated.get("score") or {}
        dom_scores = score_from_dom(dom_node or {}, cfg.fuse_labels, cfg)
        fused_scores, source_priority = fuse_scores(rated_scores, dom_scores, cfg)
        elements.append(
            Element(
                id=str(rated.get("id") or rated.get("uid") or len(elements)),
                bbox=bbox,
                text_dom=(dom_node or {}).get("text") or (dom_node or {}).get("inner_text"),
                text_ocr=rated.get("text"),
                dom=dom_node or {},
                rated=rated,
                scores_fused=fused_scores,
                metadata={"matched_dom": bool(dom_node), "dom_match_score": match_score, "source_priority": source_priority, "source_scores": {"dom": dom_scores, "rated": rated_scores}, "ui_hints": build_ui_hints(dom_node, rated_scores, fused_scores), "origin": "rated_first"},
            )
        )
    if cfg.include_dom_orphans:
        elements.extend(build_dom_orphans(dom_nodes, used_dom_ids, cfg))
    return elements


def load_and_fuse(rate_summary_path: Path, dom_clickables_path: Path, cfg: Optional[FusionConfig] = None) -> List[Element]:
    return fuse(load_json(rate_summary_path), load_json(dom_clickables_path), cfg)

