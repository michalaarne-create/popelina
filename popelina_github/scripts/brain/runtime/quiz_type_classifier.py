from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .quiz_utils import (
    box_center,
    box_height,
    box_width,
    clamp_float,
    md5_text,
    normalize_match_text,
    question_like,
    text_similarity,
)


QUIZ_TYPES: List[str] = [
    "single",
    "multi",
    "dropdown",
    "dropdown_scroll",
    "text",
    "triple",
    "mixed",
]

_PROMPT_MULTI = (
    "zaznacz",
    "wybierz wszystkie",
    "wielokrot",
    "all that apply",
    "select all",
)
_PROMPT_SCROLL = ("scroll", "przewi", "duzo opcji")
_PROMPT_TRIPLE = ("(1/3)", "(2/3)", "(3/3)", "1/3", "2/3", "3/3")
_PROMPT_MIX = ("(mix)", "mix")
_PROMPT_TEXT = ("wpisz", "podaj", "type", "enter")

_MODEL_CACHE: Dict[str, Dict[str, Any]] = {}


def _env_flag(name: str, default: str = "0") -> bool:
    raw = str(os.environ.get(name, default) or default).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)) or default)
    except Exception:
        return float(default)


def _softmax(logits: Dict[str, float]) -> Dict[str, float]:
    if not logits:
        return {}
    m = max(float(v) for v in logits.values())
    exps: Dict[str, float] = {}
    total = 0.0
    for k, v in logits.items():
        e = math.exp(float(v) - m)
        exps[k] = e
        total += e
    if total <= 0:
        n = float(len(logits))
        return {k: 1.0 / n for k in logits}
    return {k: float(v) / total for k, v in exps.items()}


def _model_default_path() -> Path:
    root = Path(__file__).resolve().parents[3]
    return root / "data" / "models" / "quiz_type_classifier_v1.json"


def _load_model() -> Optional[Dict[str, Any]]:
    enabled = _env_flag("FULLBOT_QUIZ_TYPE_MODEL_ENABLED", "1")
    if not enabled:
        return None
    path = Path(str(os.environ.get("FULLBOT_QUIZ_TYPE_MODEL_PATH", "") or "").strip() or str(_model_default_path()))
    key = str(path).lower()
    if key in _MODEL_CACHE:
        payload = _MODEL_CACHE[key]
        return payload if payload else None
    if not path.exists():
        _MODEL_CACHE[key] = {}
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        _MODEL_CACHE[key] = {}
        return None
    if not isinstance(payload, dict):
        _MODEL_CACHE[key] = {}
        return None
    _MODEL_CACHE[key] = payload
    return payload


def _prompt_flags(prompt: str) -> Dict[str, bool]:
    raw = str(prompt or "")
    norm = normalize_match_text(raw)
    return {
        "question_like": bool(question_like(raw)),
        "has_colon": raw.strip().endswith(":"),
        "has_qmark": "?" in raw,
        "multi_hint": any(tok in norm for tok in _PROMPT_MULTI),
        "scroll_hint": any(tok in norm for tok in _PROMPT_SCROLL),
        "triple_hint": any(tok in norm for tok in _PROMPT_TRIPLE),
        "mix_hint": any(tok in norm for tok in _PROMPT_MIX),
        "text_hint": any(tok in norm for tok in _PROMPT_TEXT),
    }


def _calc_vertical_regularity(options: Sequence[Dict[str, Any]]) -> float:
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
    # 1.0 => very regular, 0.0 => chaotic
    return float(clamp_float(1.0 - (std / max(8.0, mean_gap * 0.75)), 0.0, 1.0))


def _extract_marker_stats(
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


def _infer_block_type(
    *,
    block: Dict[str, Any],
    global_marker_stats: Dict[str, float],
) -> str:
    control_kind = str(block.get("control_kind") or "choice")
    prompt = str(block.get("prompt_text") or "")
    flags = _prompt_flags(prompt)
    options = block.get("options") if isinstance(block.get("options"), list) else []
    if control_kind == "text" or bool(block.get("input_bbox")) or flags["text_hint"]:
        return "text"
    if control_kind == "dropdown" or bool(block.get("select_bbox")):
        if flags["scroll_hint"] or len(options) >= 10:
            return "dropdown_scroll"
        return "dropdown"
    if flags["multi_hint"]:
        return "multi"
    if global_marker_stats.get("marker_square", 0.0) > global_marker_stats.get("marker_circle", 0.0):
        return "multi"
    return "single"


def _heuristic_logits(
    *,
    active: Optional[Dict[str, Any]],
    questions: Sequence[Dict[str, Any]],
    marker_stats: Dict[str, float],
) -> Tuple[Dict[str, float], Dict[str, Any], Dict[str, float], List[str]]:
    logits: Dict[str, float] = {k: -0.35 for k in QUIZ_TYPES}
    evidence: Dict[str, Any] = {}
    feature_values: Dict[str, float] = {}
    block_types: List[str] = []

    q_count = int(len(questions or []))
    feature_values["question_count"] = float(q_count)
    evidence["question_count"] = q_count
    if active is None:
        logits["single"] += 0.2
        return logits, evidence, feature_values, block_types

    prompt = str(active.get("prompt_text") or active.get("question_text") or "")
    flags = _prompt_flags(prompt)
    options = active.get("options") if isinstance(active.get("options"), list) else []
    next_bbox = active.get("next_bbox")
    has_next = bool(isinstance(next_bbox, list) and len(next_bbox) == 4)
    has_select = bool(isinstance(active.get("select_bbox"), list) and len(active.get("select_bbox")) == 4)
    has_input = bool(isinstance(active.get("input_bbox"), list) and len(active.get("input_bbox")) == 4)
    control_kind = str(active.get("control_kind") or "choice")
    scroll_needed = bool(active.get("scroll_needed"))
    opt_n = len(options)
    reg = _calc_vertical_regularity(options)

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

    # Hard-stop signals
    if has_input or control_kind == "text" or flags["text_hint"]:
        logits["text"] += 3.0
    if has_select or control_kind == "dropdown":
        logits["dropdown"] += 2.7
        if flags["scroll_hint"] or opt_n >= 10 or scroll_needed:
            logits["dropdown_scroll"] += 3.0
        else:
            logits["dropdown"] += 0.9

    # Prompt signals
    if flags["multi_hint"]:
        logits["multi"] += 2.2
    if flags["scroll_hint"]:
        logits["dropdown_scroll"] += 1.4

    # Marker signals (when options visible)
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

    # Option-column regularity helps choice family.
    if opt_n >= 2:
        choice_boost = 0.8 + 0.8 * reg
        logits["single"] += choice_boost * 0.9
        logits["multi"] += choice_boost * 0.9
        logits["dropdown"] -= 0.5
        logits["text"] -= 0.7

    # Per-block decomposition (block types are always answer-control types).
    for q in questions:
        if not isinstance(q, dict):
            continue
        block_types.append(_infer_block_type(block=q, global_marker_stats=marker_stats))
    if not block_types:
        block_types = [_infer_block_type(block=active, global_marker_stats=marker_stats)]
    bt_set = set(block_types)
    evidence["block_types"] = block_types[:12]
    evidence["block_type_counts"] = {k: int(block_types.count(k)) for k in sorted(bt_set)}
    if q_count >= 2:
        # Multi-block layout should not override control-kind classification.
        logits["mixed"] += 0.25 if len(bt_set) >= 2 else 0.1
        logits["triple"] += 0.1 if q_count >= 3 else 0.0

    # Next/no-next is only weak evidence.
    if not has_next:
        logits["single"] += 0.15
        logits["multi"] += 0.15
        logits["dropdown"] += 0.1
        logits["dropdown_scroll"] += 0.1
        logits["text"] += 0.1

    # Stabilize defaults.
    if control_kind == "choice" and opt_n >= 2 and marker_total <= 0.0:
        logits["single"] += 0.35

    return logits, evidence, feature_values, block_types


def _apply_model_logits(feature_values: Dict[str, float], heur_logits: Dict[str, float]) -> Tuple[Dict[str, float], str]:
    model = _load_model()
    if not model:
        return heur_logits, "heuristic"
    weights = model.get("weights") if isinstance(model.get("weights"), dict) else {}
    bias = model.get("bias") if isinstance(model.get("bias"), dict) else {}
    if not isinstance(weights, dict):
        return heur_logits, "heuristic"

    norm = model.get("normalization") if isinstance(model.get("normalization"), dict) else {}
    norm_mean = norm.get("mean") if isinstance(norm.get("mean"), dict) else {}
    norm_std = norm.get("std") if isinstance(norm.get("std"), dict) else {}

    model_logits: Dict[str, float] = {c: float(bias.get(c, 0.0) or 0.0) for c in QUIZ_TYPES}
    for feat_name, feat_val in feature_values.items():
        row = weights.get(feat_name)
        if not isinstance(row, dict):
            continue
        fv = float(feat_val)
        try:
            m = float(norm_mean.get(feat_name, 0.0) or 0.0)
            s = float(norm_std.get(feat_name, 1.0) or 1.0)
        except Exception:
            m, s = 0.0, 1.0
        if abs(s) < 1e-9:
            s = 1.0
        fv = (fv - m) / s
        for cls_name, w in row.items():
            if cls_name in model_logits:
                try:
                    model_logits[cls_name] += fv * float(w)
                except Exception:
                    continue
    alpha = clamp_float(_env_float("FULLBOT_QUIZ_TYPE_MODEL_ALPHA", 0.65), 0.0, 1.0)
    out: Dict[str, float] = {}
    for c in QUIZ_TYPES:
        out[c] = (1.0 - alpha) * float(heur_logits.get(c, 0.0)) + alpha * float(model_logits.get(c, 0.0))
    return out, "hybrid_model"


def classify_quiz_type(
    *,
    region_payload: Optional[Dict[str, Any]],
    summary_data: Optional[Dict[str, Any]],
    rated_data: Optional[Dict[str, Any]],
    questions: Sequence[Dict[str, Any]],
    active: Optional[Dict[str, Any]],
    screen_w: int,
    screen_h: int,
) -> Dict[str, Any]:
    marker_stats = _extract_marker_stats(
        rated_data=rated_data,
        active_bbox=(active or {}).get("bbox") if isinstance(active, dict) else None,
        next_bbox=(active or {}).get("next_bbox") if isinstance(active, dict) else None,
    )
    heur_logits, evidence, feature_values, block_types = _heuristic_logits(
        active=active,
        questions=questions,
        marker_stats=marker_stats,
    )
    logits, source = _apply_model_logits(feature_values, heur_logits)
    probs = _softmax(logits)
    ordered = sorted(probs.items(), key=lambda kv: float(kv[1]), reverse=True)
    top_label = ordered[0][0] if ordered else "single"
    top_prob = float(ordered[0][1]) if ordered else 0.0
    second_prob = float(ordered[1][1]) if len(ordered) > 1 else 0.0
    margin = max(0.0, top_prob - second_prob)

    layout_type = "multi_block" if int(len(questions or [])) >= 2 else "single_block"
    active_block_type = str(block_types[0] if block_types else "single")
    if active_block_type not in {"single", "multi", "dropdown", "dropdown_scroll", "text"}:
        active_block_type = "single"

    answer_types = ["single", "multi", "dropdown", "dropdown_scroll", "text"]
    block_prob = float(probs.get(active_block_type, 0.0))
    other_block_prob = 0.0
    for t in answer_types:
        if t == active_block_type:
            continue
        other_block_prob = max(other_block_prob, float(probs.get(t, 0.0)))
    block_margin = max(0.0, block_prob - other_block_prob)

    unknown_enabled = _env_flag("FULLBOT_QUIZ_TYPE_UNKNOWN_ENABLED", "1")
    min_conf = clamp_float(_env_float("FULLBOT_QUIZ_TYPE_MIN_CONF", 0.45), 0.0, 1.0)
    detected = active_block_type
    detected_conf = block_prob
    detected_margin = block_margin
    unknown_prob = 0.0
    reason = "active_block_type"
    if unknown_enabled and detected_conf < min_conf:
        detected = "unknown"
        unknown_prob = max(0.0, min(1.0, 1.0 - detected_conf))
        reason = "low_conf_unknown"

    type_probs: Dict[str, float] = {c: round(float(probs.get(c, 0.0)), 6) for c in QUIZ_TYPES}
    type_probs["unknown"] = round(float(unknown_prob), 6)

    detected_op = "choice"
    if detected == "text":
        detected_op = "text"
    elif detected in {"dropdown", "dropdown_scroll"}:
        detected_op = "dropdown"
    elif detected == "unknown":
        detected_op = "unknown"

    prompt_text = str((active or {}).get("prompt_text") or "")
    options = (active or {}).get("options") if isinstance((active or {}).get("options"), list) else []
    question_split = {
        "question": prompt_text,
        "answers": [str((o or {}).get("text") or "") for o in options[:16] if isinstance(o, dict)],
    }
    parse_sig = md5_text(
        f"{normalize_match_text(prompt_text)}|{len(options)}|{screen_w}x{screen_h}|{detected}|{layout_type}|{round(top_prob,4)}"
    )
    return {
        "detected_quiz_type": detected,
        "detected_operational_type": detected_op,
        "layout_type": layout_type,
        "active_block_type": active_block_type,
        "top_global_type": top_label,
        "type_confidence": round(float(detected_conf if detected != "unknown" else top_prob), 6),
        "decision_margin": round(float(detected_margin if detected != "unknown" else margin), 6),
        "type_probs": type_probs,
        "type_source": source,
        "type_reason": reason,
        "type_signals": {
            **evidence,
            "rule": reason,
            "screen_size": [int(screen_w), int(screen_h)],
        },
        "block_types": block_types,
        "quiz_type_features": {k: round(float(v), 6) for k, v in feature_values.items()},
        "question_split": question_split,
        "parse_signature_v2": parse_sig,
    }
