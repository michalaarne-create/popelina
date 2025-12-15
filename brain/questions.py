"""
Question cluster builder.

Takes fused elements (DOM + rated) and groups them into QuestionCluster objects
ready for the QA Gateway. The heuristics are intentionally lightweight and
documented inline so they can be swapped for better models later.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .types import BBox, Element, QuestionCluster, QuestionOption


@dataclass
class QuestionBuilderConfig:
    answer_score_threshold: float = 0.45
    option_score_threshold: float = 0.25
    dropdown_score_threshold: float = 0.35
    vertical_gap_px: int = 35
    width_tolerance: float = 0.35
    max_option_distance_px: int = 300
    question_min_chars: int = 12
    topic_keywords: Dict[str, Tuple[str, ...]] = None  # set in __post_init__

    def __post_init__(self):
        if self.topic_keywords is None:
            self.topic_keywords = {
                "car_brand": ("toyota", "vw", "volkswagen", "ford", "bmw", "audi"),
                "car_model": ("corolla", "golf", "focus", "astra", "passat"),
                "job": ("praca", "stanowisko", "zawod", "firma", "pracujesz"),
                "address": ("adres", "ulica", "miasto", "kod pocztowy"),
                "contact": ("telefon", "email", "e-mail", "mail"),
            }


def build_question_clusters(
    elements: List[Element],
    cfg: Optional[QuestionBuilderConfig] = None,
) -> Tuple[List[QuestionCluster], Dict[str, Optional[Element]]]:
    """
    Return (clusters, specials) where specials collects next/cookie elements.
    """
    cfg = cfg or QuestionBuilderConfig()
    specials = detect_special_elements(elements)
    candidates = detect_question_candidates(elements, cfg)
    clusters: List[QuestionCluster] = []

    for candidate in candidates:
        options = detect_options(candidate, elements, cfg)
        if not options:
            continue
        q_type = infer_question_type(candidate)
        ui_mode = infer_ui_mode(candidate, options)
        canonical_key = make_canonical_key(candidate, options, q_type)
        topic_tags = tag_question(candidate, cfg)
        cluster = QuestionCluster(
            id=f"q_{len(clusters)}",
            type=q_type,
            ui_mode=ui_mode,
            question_bbox=candidate.bbox,
            question_text=candidate.text_dom or candidate.text_ocr or "",
            options=options,
            topic_tags=topic_tags,
            canonical_key=canonical_key,
            source_element_id=candidate.id,
            metadata={
                "source_scores": candidate.scores_fused,
                "ui_hints": candidate.metadata.get("ui_hints") or {},
            },
        )
        clusters.append(cluster)

    clusters = deduplicate_clusters(clusters)
    return clusters, specials


def detect_special_elements(elements: List[Element]) -> Dict[str, Optional[Element]]:
    def best_for_label(label: str) -> Optional[Element]:
        best: Optional[Element] = None
        best_score = 0.0
        for el in elements:
            score = float(el.scores_fused.get(label, 0.0))
            if score > best_score:
                best = el
                best_score = score
        return best

    return {
        "next": best_for_label("next_active"),
        "cookie_accept": best_for_label("cookie_accept"),
        "cookie_reject": best_for_label("cookie_reject"),
    }


def detect_question_candidates(elements: List[Element], cfg: QuestionBuilderConfig) -> List[Element]:
    candidates = []
    for el in elements:
        score_single = float(el.scores_fused.get("answer_single", 0.0))
        score_multi = float(el.scores_fused.get("answer_multi", 0.0))
        text = (el.text_dom or el.text_ocr or "").strip()
        ui_hints = el.metadata.get("ui_hints", {})
        dropdown_like = bool(ui_hints.get("dropdown_like"))
        # Skip pure dropdown toggles unless they look like labeled questions themselves
        if dropdown_like and max(score_single, score_multi) < cfg.answer_score_threshold:
            continue
        text_hint = ("?" in text) or len(text) >= cfg.question_min_chars
        if max(score_single, score_multi) >= cfg.answer_score_threshold or text_hint:
            candidates.append(el)
    candidates.sort(key=lambda e: (e.bbox[1], -(e.bbox[3] - e.bbox[1])))  # top of screen first
    return candidates


def detect_options(question: Element, elements: List[Element], cfg: QuestionBuilderConfig) -> List[QuestionOption]:
    opts: List[QuestionOption] = []
    q_x1, q_y1, q_x2, q_y2 = question.bbox
    q_w = max(1, q_x2 - q_x1)

    for el in elements:
        if el.id == question.id:
            continue
        bbox = el.bbox
        # Geometric checks
        below = bbox[1] >= (q_y2 - cfg.vertical_gap_px)
        vertically_close = (bbox[1] - q_y2) <= cfg.max_option_distance_px
        similar_width = abs((bbox[2] - bbox[0]) - q_w) / float(q_w) <= cfg.width_tolerance
        ui_hints = el.metadata.get("ui_hints", {})
        dropdown_like = bool(ui_hints.get("dropdown_like"))
        if dropdown_like:
            # Dropdown toggles can be narrower/wider; relax width constraint
            if not (below and vertically_close):
                continue
        else:
            if not (below and vertically_close and similar_width):
                continue

        score_option = float(el.scores_fused.get("option", 0.0))
        score_single = float(el.scores_fused.get("answer_single", 0.0))
        score_multi = float(el.scores_fused.get("answer_multi", 0.0))
        score_dropdown = float(el.scores_fused.get("dropdown", 0.0))
        score = max(score_option, score_single, score_multi, score_dropdown)
        if score < cfg.option_score_threshold:
            continue
        if dropdown_like and score_dropdown < cfg.dropdown_score_threshold:
            continue

        selected = str(el.dom.get("aria-checked") or el.dom.get("checked") or "").lower() in {
            "true",
            "1",
            "checked",
        }

        text = el.text_dom or el.text_ocr or ""
        opts.append(
            QuestionOption(
                id=str(el.id),
                bbox=el.bbox,
                text=text.strip(),
                selected=selected,
                ranked_score=score,
                dom=el.dom,
                rated=el.rated,
                metadata={
                    "source_scores": el.scores_fused,
                    "ui_hints": ui_hints,
                },
            )
        )

    opts.sort(key=lambda o: (o.bbox[1] if o.bbox else 0, -o.ranked_score))
    return opts


def infer_question_type(question: Element) -> str:
    score_single = float(question.scores_fused.get("answer_single", 0.0))
    score_multi = float(question.scores_fused.get("answer_multi", 0.0))
    return "multi" if score_multi > score_single else "single"


def infer_ui_mode(question: Element, options: List[QuestionOption]) -> str:
    dropdown_opts = [opt for opt in options if (opt.metadata.get("ui_hints") or {}).get("dropdown_like")]
    q_hint = (question.metadata.get("ui_hints") or {}).get("dropdown_like")
    if dropdown_opts or q_hint:
        return "dropdown"
    return "standard"


def make_canonical_key(question: Element, options: List[QuestionOption], q_type: str) -> str:
    q_text = (question.text_dom or question.text_ocr or "").strip().lower()
    opt_texts = [opt.text.strip().lower() for opt in options]
    payload = f"{q_type}\n{q_text}\n" + "\n".join(sorted(opt_texts))
    return hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()


def tag_question(question: Element, cfg: QuestionBuilderConfig) -> List[str]:
    text = (question.text_dom or question.text_ocr or "").lower()
    tags: List[str] = []
    for tag, keywords in cfg.topic_keywords.items():
        if any(re.search(rf"\b{kw}\b", text) for kw in keywords):
            tags.append(tag)
    return tags


def deduplicate_clusters(clusters: List[QuestionCluster]) -> List[QuestionCluster]:
    seen = set()
    unique: List[QuestionCluster] = []
    for cl in clusters:
        key = cl.canonical_key or cl.id
        if key in seen:
            continue
        seen.add(key)
        unique.append(cl)
    return unique
