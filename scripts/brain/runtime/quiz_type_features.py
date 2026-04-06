from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .quiz_utils import clamp_float, normalize_match_text, question_like


def prompt_flags(
    prompt: str,
    *,
    prompt_multi: Sequence[str],
    prompt_scroll: Sequence[str],
    prompt_triple: Sequence[str],
    prompt_mix: Sequence[str],
    prompt_text: Sequence[str],
    prompt_slider: Sequence[str],
) -> Dict[str, bool]:
    raw = str(prompt or "")
    norm = normalize_match_text(raw)
    raw_norm = " ".join(raw.strip().lower().split())
    return {
        "question_like": bool(question_like(raw)),
        "has_colon": raw.strip().endswith(":"),
        "has_qmark": "?" in raw,
        "multi_hint": any(tok in norm for tok in prompt_multi),
        "scroll_hint": any(tok in norm for tok in prompt_scroll),
        "triple_hint": any(tok in raw_norm for tok in prompt_triple),
        "mix_hint": any(tok in raw_norm for tok in prompt_mix),
        "text_hint": any(tok in norm for tok in prompt_text),
        "slider_hint": any(tok in norm for tok in prompt_slider),
    }


def prompt_prefers_single(prompt: str) -> bool:
    norm = normalize_match_text(prompt)
    if not norm:
        return False
    return any(tok in norm for tok in ("choose one", "one option", "jedna odpowiedz", "jeden wybor", "wybierz jedna"))


def prompt_prefers_multi(prompt: str, *, prompt_multi: Sequence[str]) -> bool:
    norm = normalize_match_text(prompt)
    if not norm:
        return False
    return any(tok in norm for tok in prompt_multi)


def calc_vertical_regularity(options: Sequence[Dict[str, Any]]) -> float:
    ys: List[float] = []
    for opt in options or []:
        bb = opt.get("bbox") if isinstance(opt, dict) else None
        if isinstance(bb, list) and len(bb) == 4:
            ys.append(float(bb[1]))
    if len(ys) < 3:
        return 0.0
    ys = sorted(ys)
    gaps = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
    if not gaps:
        return 0.0
    mean_gap = sum(gaps) / max(1.0, float(len(gaps)))
    if mean_gap <= 1.0:
        return 0.0
    var = sum((g - mean_gap) ** 2 for g in gaps) / max(1.0, float(len(gaps)))
    std = math.sqrt(max(0.0, var))
    return float(clamp_float(1.0 - (std / max(8.0, mean_gap * 0.75)), 0.0, 1.0))


def extract_marker_stats(
    *,
    rated_data: Optional[Dict[str, Any]],
    active_bbox: Optional[Sequence[float]],
    next_bbox: Optional[Sequence[float]],
) -> Dict[str, float]:
    out = {
        "marker_total": 0.0,
        "marker_circle": 0.0,
        "marker_square": 0.0,
        "marker_unknown": 0.0,
        "marker_mean_conf": 0.0,
    }
    if not isinstance(rated_data, dict):
        return out
    elems = rated_data.get("elements")
    if not isinstance(elems, list):
        return out
    q_bottom = float(active_bbox[3]) if isinstance(active_bbox, (list, tuple)) and len(active_bbox) == 4 else 0.0
    next_top = float(next_bbox[1]) if isinstance(next_bbox, (list, tuple)) and len(next_bbox) == 4 else 10e9
    confs: List[float] = []
    for el in elems:
        if not isinstance(el, dict):
            continue
        bb = el.get("bbox")
        if not (isinstance(bb, list) and len(bb) == 4):
            continue
        if float(bb[1]) <= q_bottom:
            continue
        if float(bb[3]) >= next_top:
            continue
        txt = normalize_match_text(str(el.get("text") or ""))
        if not txt or txt in {"home", "next", "nastepne"}:
            continue
        m = el.get("marker") if isinstance(el.get("marker"), dict) else {}
        shape = str(m.get("shape") or "none").strip().lower()
        kind = str(m.get("kind") or "none").strip().lower()
        try:
            mc = float(m.get("conf") or 0.0)
        except Exception:
            mc = 0.0
        out["marker_total"] += 1.0
        confs.append(mc)
        if shape == "circle" or kind == "radio":
            out["marker_circle"] += 1.0
        elif shape == "square" or kind == "checkbox":
            out["marker_square"] += 1.0
        else:
            out["marker_unknown"] += 1.0
    if confs:
        out["marker_mean_conf"] = float(sum(confs) / max(1, len(confs)))
    return out


def infer_block_type(
    *,
    block: Dict[str, Any],
    global_marker_stats: Dict[str, float],
    prompt_multi: Sequence[str],
    prompt_scroll: Sequence[str],
    prompt_triple: Sequence[str],
    prompt_mix: Sequence[str],
    prompt_text: Sequence[str],
    prompt_slider: Sequence[str],
) -> str:
    control_kind = str(block.get("control_kind") or "choice")
    prompt = str(block.get("prompt_text") or "")
    flags = prompt_flags(
        prompt,
        prompt_multi=prompt_multi,
        prompt_scroll=prompt_scroll,
        prompt_triple=prompt_triple,
        prompt_mix=prompt_mix,
        prompt_text=prompt_text,
        prompt_slider=prompt_slider,
    )
    options = block.get("options") if isinstance(block.get("options"), list) else []
    if control_kind == "text" or bool(block.get("input_bbox")) or flags["text_hint"]:
        return "text"
    if control_kind == "slider" or flags["slider_hint"]:
        return "slider"
    if control_kind == "dropdown" or bool(block.get("select_bbox")):
        if flags["scroll_hint"] or len(options) >= 10:
            return "dropdown_scroll"
        return "dropdown"
    if flags["multi_hint"]:
        return "multi"
    if global_marker_stats.get("marker_square", 0.0) > global_marker_stats.get("marker_circle", 0.0):
        return "multi"
    return "single"


def build_heuristic_quiz_features(
    *,
    active: Optional[Dict[str, Any]],
    questions: Sequence[Dict[str, Any]],
    marker_stats: Dict[str, float],
    quiz_types: Sequence[str],
    prompt_multi: Sequence[str],
    prompt_scroll: Sequence[str],
    prompt_triple: Sequence[str],
    prompt_mix: Sequence[str],
    prompt_text: Sequence[str],
    prompt_slider: Sequence[str],
) -> Tuple[Dict[str, float], Dict[str, Any], Dict[str, float], List[str]]:
    logits: Dict[str, float] = {k: -0.35 for k in quiz_types}
    evidence: Dict[str, Any] = {}
    feature_values: Dict[str, float] = {}
    block_types: List[str] = []
    aggregate_flags = {
        "multi_hint": False,
        "scroll_hint": False,
        "triple_hint": False,
        "mix_hint": False,
        "text_hint": False,
        "slider_hint": False,
    }
    block_triple_markers = 0
    block_mix_markers = 0

    q_count = int(len(questions or []))
    feature_values["question_count"] = float(q_count)
    evidence["question_count"] = q_count
    if active is None:
        logits["single"] += 0.2
        return logits, evidence, feature_values, block_types

    prompt = str(active.get("prompt_text") or active.get("question_text") or "")
    flags = prompt_flags(
        prompt,
        prompt_multi=prompt_multi,
        prompt_scroll=prompt_scroll,
        prompt_triple=prompt_triple,
        prompt_mix=prompt_mix,
        prompt_text=prompt_text,
        prompt_slider=prompt_slider,
    )
    options = active.get("options") if isinstance(active.get("options"), list) else []
    next_bbox = active.get("next_bbox")
    has_next = bool(isinstance(next_bbox, list) and len(next_bbox) == 4)
    has_select = bool(isinstance(active.get("select_bbox"), list) and len(active.get("select_bbox")) == 4)
    has_input = bool(isinstance(active.get("input_bbox"), list) and len(active.get("input_bbox")) == 4)
    control_kind = str(active.get("control_kind") or "choice")
    scroll_needed = bool(active.get("scroll_needed"))
    opt_n = len(options)
    reg = calc_vertical_regularity(options)
    question_count = max(1, q_count)

    feature_values.update(
        {
            "has_next": 1.0 if has_next else 0.0,
            "has_select": 1.0 if has_select else 0.0,
            "has_input": 1.0 if has_input else 0.0,
            "option_count": float(opt_n),
            "vertical_regularity": float(reg),
            "scroll_needed": 1.0 if scroll_needed else 0.0,
            "multi_hint": 1.0 if flags["multi_hint"] else 0.0,
            "scroll_hint": 1.0 if flags["scroll_hint"] else 0.0,
            "triple_hint": 1.0 if flags["triple_hint"] else 0.0,
            "mix_hint": 1.0 if flags["mix_hint"] else 0.0,
            "text_hint": 1.0 if flags["text_hint"] else 0.0,
            "slider_hint": 1.0 if flags["slider_hint"] else 0.0,
            "marker_total": float(marker_stats.get("marker_total", 0.0)),
            "marker_circle": float(marker_stats.get("marker_circle", 0.0)),
            "marker_square": float(marker_stats.get("marker_square", 0.0)),
            "marker_unknown": float(marker_stats.get("marker_unknown", 0.0)),
            "marker_mean_conf": float(marker_stats.get("marker_mean_conf", 0.0)),
        }
    )
    evidence.update(
        {
            "control_kind": control_kind,
            "has_next": has_next,
            "has_select": has_select,
            "has_input": has_input,
            "option_count": opt_n,
            "vertical_regularity": round(reg, 4),
            "scroll_needed": scroll_needed,
            "prompt_flags": {k: bool(v) for k, v in flags.items()},
            "marker": {k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in marker_stats.items()},
        }
    )

    if has_input or control_kind == "text" or flags["text_hint"]:
        logits["text"] += 3.0
    if control_kind == "slider" or flags["slider_hint"]:
        logits["slider"] += 3.1
        logits["dropdown"] -= 0.4
        logits["text"] -= 0.3
    if has_select or control_kind == "dropdown":
        logits["dropdown"] += 2.7
        if flags["scroll_hint"] or opt_n >= 10 or scroll_needed:
            logits["dropdown_scroll"] += 3.0
        else:
            logits["dropdown"] += 0.9

    if flags["multi_hint"]:
        logits["multi"] += 2.2
    if flags["scroll_hint"]:
        logits["dropdown_scroll"] += 1.4

    marker_total = float(marker_stats.get("marker_total", 0.0))
    marker_circle = float(marker_stats.get("marker_circle", 0.0))
    marker_square = float(marker_stats.get("marker_square", 0.0))
    marker_conf = float(marker_stats.get("marker_mean_conf", 0.0))
    if marker_total >= 1.0:
        if marker_circle > marker_square:
            logits["single"] += 1.6 + 0.6 * marker_conf
        elif marker_square > marker_circle:
            logits["multi"] += 1.6 + 0.6 * marker_conf
        else:
            logits["single"] += 0.5
            logits["multi"] += 0.5

    if opt_n >= 2:
        choice_boost = 0.8 + 0.8 * reg
        logits["single"] += choice_boost * 0.9
        logits["multi"] += choice_boost * 0.9
        logits["dropdown"] -= 0.5
        logits["text"] -= 0.7

    for q in questions:
        if not isinstance(q, dict):
            continue
        q_flags = prompt_flags(
            str(q.get("prompt_text") or q.get("question_text") or ""),
            prompt_multi=prompt_multi,
            prompt_scroll=prompt_scroll,
            prompt_triple=prompt_triple,
            prompt_mix=prompt_mix,
            prompt_text=prompt_text,
            prompt_slider=prompt_slider,
        )
        for key, value in q_flags.items():
            if key in aggregate_flags and bool(value):
                aggregate_flags[key] = True
        if q_flags.get("triple_hint"):
            block_triple_markers += 1
        if q_flags.get("mix_hint"):
            block_mix_markers += 1
        block_types.append(
            infer_block_type(
                block=q,
                global_marker_stats=marker_stats,
                prompt_multi=prompt_multi,
                prompt_scroll=prompt_scroll,
                prompt_triple=prompt_triple,
                prompt_mix=prompt_mix,
                prompt_text=prompt_text,
                prompt_slider=prompt_slider,
            )
        )
    if not block_types:
        block_types = [
            infer_block_type(
                block=active,
                global_marker_stats=marker_stats,
                prompt_multi=prompt_multi,
                prompt_scroll=prompt_scroll,
                prompt_triple=prompt_triple,
                prompt_mix=prompt_mix,
                prompt_text=prompt_text,
                prompt_slider=prompt_slider,
            )
        ]
    bt_set = set(block_types)
    single_blocks = sum(1 for t in block_types if t == "single")
    multi_blocks = sum(1 for t in block_types if t == "multi")
    dropdown_blocks = sum(1 for t in block_types if t in {"dropdown", "dropdown_scroll"})
    text_blocks = sum(1 for t in block_types if t == "text")
    slider_blocks = sum(1 for t in block_types if t == "slider")
    per_block_option_counts: List[float] = []
    if questions:
        for q in questions:
            if not isinstance(q, dict):
                continue
            q_opts = q.get("options") if isinstance(q.get("options"), list) else []
            per_block_option_counts.append(float(len(q_opts)))
    else:
        per_block_option_counts.append(float(opt_n))
    feature_values.update(
        {
            "is_multi_block_layout": 1.0 if question_count >= 2 else 0.0,
            "is_exact_triple_count": 1.0 if question_count == 3 else 0.0,
            "distinct_block_type_count": float(len(bt_set)),
            "all_blocks_same_type": 1.0 if len(bt_set) <= 1 and len(block_types) >= 1 else 0.0,
            "has_mixed_controls": 1.0 if len(bt_set) >= 2 else 0.0,
            "avg_options_per_block": float(sum(per_block_option_counts)) / max(1.0, float(len(per_block_option_counts))),
            "min_options_per_block": float(min(per_block_option_counts)) if per_block_option_counts else 0.0,
            "max_options_per_block": float(max(per_block_option_counts)) if per_block_option_counts else 0.0,
            "single_block_ratio": float(single_blocks) / float(question_count),
            "multi_block_ratio": float(multi_blocks) / float(question_count),
            "dropdown_block_ratio": float(dropdown_blocks) / float(question_count),
            "text_block_ratio": float(text_blocks) / float(question_count),
            "slider_block_ratio": float(slider_blocks) / float(question_count),
            "has_slider": 1.0 if slider_blocks > 0 else 0.0,
            "all_blocks_have_triple_marker": 1.0 if block_triple_markers == question_count and question_count >= 1 else 0.0,
            "all_blocks_have_mix_marker": 1.0 if block_mix_markers == question_count and question_count >= 1 else 0.0,
        }
    )
    evidence["block_types"] = block_types[:12]
    evidence["block_type_counts"] = {k: int(block_types.count(k)) for k in sorted(bt_set)}
    evidence["aggregate_prompt_flags"] = {k: bool(v) for k, v in aggregate_flags.items()}
    if aggregate_flags["triple_hint"]:
        logits["triple"] += 3.8
        logits["mixed"] -= 0.6
    if aggregate_flags["mix_hint"]:
        logits["mixed"] += 3.6
        logits["triple"] -= 0.8
    if q_count >= 2:
        logits["mixed"] += 0.25 if len(bt_set) >= 2 else 0.1
        logits["triple"] += 1.2 if q_count == 3 else (0.2 if q_count >= 3 else 0.0)
    if q_count == 3:
        logits["triple"] += 0.45
        logits["single"] -= 0.25
        logits["multi"] -= 0.25
        logits["dropdown"] -= 0.15
        logits["text"] -= 0.15

    if not has_next:
        logits["single"] += 0.15
        logits["multi"] += 0.15
        logits["dropdown"] += 0.1
        logits["dropdown_scroll"] += 0.1
        logits["text"] += 0.1

    if control_kind == "choice" and opt_n >= 2 and marker_total <= 0.0:
        logits["single"] += 0.35

    return logits, evidence, feature_values, block_types
