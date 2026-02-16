from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple


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


def noisy_or(score_dom: float, score_rated: float, floor: float = 0.0) -> float:
    dom = max(floor, min(1.0, score_dom))
    rated = max(floor, min(1.0, score_rated))
    return 1.0 - (1.0 - dom) * (1.0 - rated)


def score_from_dom(node: Dict[str, Any], labels: Tuple[str, ...], cfg: FusionConfig) -> Dict[str, float]:
    scores: Dict[str, float] = {label: 0.0 for label in labels}
    if not node:
        return scores
    role = (node.get("role") or node.get("aria-role") or "").lower()
    text = (node.get("text") or node.get("inner_text") or "").strip().lower()
    aria_label = (node.get("aria-label") or node.get("label") or "").strip().lower()
    combined_text = f"{text} {aria_label}".strip()
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
    disabled = any(str(node.get(k)).lower() in {"true", "1"} for k in cfg.disabled_keys)
    if disabled:
        scores["next_active"] = 0.0
    return scores


def fuse_scores(
    rated_scores: Dict[str, float],
    dom_scores: Dict[str, float],
    cfg: FusionConfig,
) -> Tuple[Dict[str, float], Dict[str, str]]:
    fused: Dict[str, float] = {}
    source_priority: Dict[str, str] = {}
    for label in cfg.fuse_labels:
        dom_val = dom_scores.get(label, 0.0) * cfg.dom_weight
        rated_val = rated_scores.get(label, 0.0) * cfg.rated_weight
        if label in {"dropdown", "option"}:
            rated_val *= 1.15
        if label in {"next_active", "next_inactive", "cookie_accept", "cookie_reject"}:
            dom_val *= 1.15
        fused[label] = noisy_or(dom_val, rated_val, cfg.noisy_or_floor)
        if dom_val or rated_val:
            source_priority[label] = "dom" if dom_val >= rated_val else "rated"
    return fused, source_priority

