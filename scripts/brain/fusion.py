from __future__ import annotations

from .fusion_build import build_dom_orphans, build_ui_hints, fuse, load_and_fuse
from .fusion_loader import bbox_from_item, extract_text, iter_dom_nodes, iter_rated_items, load_json
from .fusion_match import iou, match_dom_node, text_similarity
from .fusion_score import DEFAULT_LABELS, FusionConfig, fuse_scores, noisy_or, score_from_dom

__all__ = [
    "DEFAULT_LABELS",
    "FusionConfig",
    "load_json",
    "iter_rated_items",
    "iter_dom_nodes",
    "bbox_from_item",
    "iou",
    "text_similarity",
    "extract_text",
    "noisy_or",
    "score_from_dom",
    "match_dom_node",
    "build_ui_hints",
    "fuse_scores",
    "build_dom_orphans",
    "fuse",
    "load_and_fuse",
]
