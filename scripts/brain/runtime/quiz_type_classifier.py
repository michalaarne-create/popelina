from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .quiz_ai_inputs import build_prompt_artifact_signals
from .quiz_type_features import (
    build_heuristic_quiz_features,
    extract_marker_stats,
    prompt_flags as build_prompt_flags,
    prompt_prefers_multi,
    prompt_prefers_single,
)
from .quiz_utils import (
    clamp_float,
    md5_text,
    normalize_match_text,
)


QUIZ_TYPES: List[str] = [
    "single",
    "multi",
    "dropdown",
    "dropdown_scroll",
    "text",
    "slider",
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
_PROMPT_SLIDER = (
    "suwak",
    "slider",
    "przesun",
    "move slider",
    "ocen",
    "rating",
    "przedzial wieku",
    "przedzia?? wieku",
    "intensity",
    "level",
    "progress",
    "temperature",
)

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
    return root / "data" / "models" / "quiz_type_robust_v3.json"


def _model_rollout_contract_path() -> Path:
    root = Path(__file__).resolve().parents[3]
    return root / "data" / "models" / "quiz_type_rollout.json"


def _resolve_rollout_model_path() -> Path:
    explicit_contract = str(os.environ.get("FULLBOT_QUIZ_TYPE_MODEL_ROLLOUT_PATH", "") or "").strip()
    contract_path = Path(explicit_contract) if explicit_contract else _model_rollout_contract_path()
    try:
        payload = json.loads(contract_path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return _model_default_path()
    if not isinstance(payload, dict):
        return _model_default_path()
    selected_path = str(payload.get("selected_model_path") or "").strip()
    return Path(selected_path) if selected_path else _model_default_path()


def _load_model() -> Optional[Dict[str, Any]]:
    enabled = _env_flag("FULLBOT_QUIZ_TYPE_MODEL_ENABLED", "1")
    if not enabled:
        return None
    selector = str(os.environ.get("FULLBOT_QUIZ_TYPE_MODEL_SELECTOR", "") or "").strip().lower()
    if selector in {"rollout", "latest"}:
        path = _resolve_rollout_model_path()
    else:
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


def _apply_model_logits(feature_values: Dict[str, float], heur_logits: Dict[str, float]) -> Tuple[Dict[str, float], str]:
    model = _load_model()
    if not model:
        return heur_logits, "heuristic"
    model_type = str(model.get("model_type") or "linear").strip().lower()
    weights = model.get("weights") if isinstance(model.get("weights"), dict) else {}
    bias = model.get("bias") if isinstance(model.get("bias"), dict) else {}
    if model_type == "linear" and not isinstance(weights, dict):
        return heur_logits, "heuristic"

    norm = model.get("normalization") if isinstance(model.get("normalization"), dict) else {}
    norm_mean = norm.get("mean") if isinstance(norm.get("mean"), dict) else {}
    norm_std = norm.get("std") if isinstance(norm.get("std"), dict) else {}
    feature_names = model.get("feature_names") if isinstance(model.get("feature_names"), list) else []
    feature_order = feature_names if feature_names else list(feature_values.keys())
    has_structural_features = bool(
        isinstance(feature_names, list)
        and "is_exact_triple_count" in feature_names
        and "distinct_block_type_count" in feature_names
    )

    x_vec: List[float] = []
    for feat_name in feature_order:
        fv = float(feature_values.get(feat_name, 0.0))
        try:
            m = float(norm_mean.get(feat_name, 0.0) or 0.0)
            s = float(norm_std.get(feat_name, 1.0) or 1.0)
        except Exception:
            m, s = 0.0, 1.0
        if abs(s) < 1e-9:
            s = 1.0
        x_vec.append((fv - m) / s)
    model_logits: Dict[str, float] = {c: 0.0 for c in QUIZ_TYPES}
    if model_type == "mlp":
        layers = model.get("layers") if isinstance(model.get("layers"), dict) else {}
        W1 = layers.get("W1")
        b1 = layers.get("b1")
        has_h2 = isinstance(layers.get("W2h"), list) and isinstance(layers.get("b2h"), list)
        if has_h2:
            W2h = layers.get("W2h")
            b2h = layers.get("b2h")
            W3 = layers.get("W3")
            b3 = layers.get("b3")
            if not all(isinstance(v, list) for v in (W1, b1, W2h, b2h, W3, b3)):
                return heur_logits, "heuristic"
        else:
            W2 = layers.get("W2")
            b2 = layers.get("b2")
            if not all(isinstance(v, list) for v in (W1, b1, W2, b2)):
                return heur_logits, "heuristic"
        # Hidden layer 1
        h_vals: List[float] = []
        for col_idx in range(len(b1)):
            acc = float(b1[col_idx] or 0.0)
            for row_idx, x in enumerate(x_vec):
                try:
                    acc += x * float(W1[row_idx][col_idx])
                except Exception:
                    continue
            h_vals.append(max(0.0, acc))
        
        if has_h2:
            # Hidden layer 2
            h2_vals: List[float] = []
            for col_idx in range(len(b2h)):
                acc = float(b2h[col_idx] or 0.0)
                for h_idx, h in enumerate(h_vals):
                    try:
                        acc += h * float(W2h[h_idx][col_idx])
                    except Exception:
                        continue
                h2_vals.append(max(0.0, acc))
            # Output layer (3-layer)
            for cls_idx, cls_name in enumerate(QUIZ_TYPES):
                acc = float(b3[cls_idx] or 0.0) if cls_idx < len(b3) else 0.0
                for h_idx, h in enumerate(h2_vals):
                    try:
                        acc += h * float(W3[h_idx][cls_idx])
                    except Exception:
                        continue
                model_logits[cls_name] = acc
        else:
            # Output layer (2-layer, backward compat)
            for cls_idx, cls_name in enumerate(QUIZ_TYPES):
                acc = float(b2[cls_idx] or 0.0) if cls_idx < len(b2) else 0.0
                for h_idx, h in enumerate(h_vals):
                    try:
                        acc += h * float(W2[h_idx][cls_idx])
                    except Exception:
                        continue
                model_logits[cls_name] = acc
    else:
        model_logits = {c: float(bias.get(c, 0.0) or 0.0) for c in QUIZ_TYPES}
        for feat_name, fv in zip(feature_order, x_vec):
            row = weights.get(feat_name)
            if not isinstance(row, dict):
                continue
            for cls_name, w in row.items():
                if cls_name in model_logits:
                    try:
                        model_logits[cls_name] += fv * float(w)
                    except Exception:
                        continue
    alpha = clamp_float(_env_float("FULLBOT_QUIZ_TYPE_MODEL_ALPHA", 0.65), 0.0, 1.0)
    if float(feature_values.get("question_count", 1.0)) >= 2.0 and not has_structural_features:
        # Legacy flat models are useful for control type, but they hurt multi-block structure.
        alpha = min(alpha, 0.15)
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
    q_count = int(len(questions or []))
    guardrails_enabled = _env_flag("FULLBOT_QUIZ_TYPE_GUARDRAILS", "1")
    prompt_text = str((active or {}).get("prompt_text") or (active or {}).get("question_text") or "")
    prompt_flags = build_prompt_flags(
        prompt_text,
        prompt_multi=_PROMPT_MULTI,
        prompt_scroll=_PROMPT_SCROLL,
        prompt_triple=_PROMPT_TRIPLE,
        prompt_mix=_PROMPT_MIX,
        prompt_text=_PROMPT_TEXT,
        prompt_slider=_PROMPT_SLIDER,
    )
    artifact_signals = build_prompt_artifact_signals(
        region_payload=region_payload,
        summary_data=summary_data,
        rated_data=rated_data,
        questions=questions,
        active=active,
        prompt_triple=_PROMPT_TRIPLE,
        prompt_mix=_PROMPT_MIX,
        prompt_slider=_PROMPT_SLIDER,
        prompt_scroll=_PROMPT_SCROLL,
        prompt_multi=_PROMPT_MULTI,
    )
    marker_stats = extract_marker_stats(
        rated_data=rated_data,
        active_bbox=(active or {}).get("bbox") if isinstance(active, dict) else None,
        next_bbox=(active or {}).get("next_bbox") if isinstance(active, dict) else None,
    )
    heur_logits, evidence, feature_values, block_types = build_heuristic_quiz_features(
        active=active,
        questions=questions,
        marker_stats=marker_stats,
        quiz_types=QUIZ_TYPES,
        prompt_multi=_PROMPT_MULTI,
        prompt_scroll=_PROMPT_SCROLL,
        prompt_triple=_PROMPT_TRIPLE,
        prompt_mix=_PROMPT_MIX,
        prompt_text=_PROMPT_TEXT,
        prompt_slider=_PROMPT_SLIDER,
    )
    logits, source = _apply_model_logits(feature_values, heur_logits)
    probs = _softmax(logits)
    ordered = sorted(probs.items(), key=lambda kv: float(kv[1]), reverse=True)
    top_label = ordered[0][0] if ordered else "single"
    top_prob = float(ordered[0][1]) if ordered else 0.0
    second_prob = float(ordered[1][1]) if len(ordered) > 1 else 0.0
    margin = max(0.0, top_prob - second_prob)

    layout_type = "multi_block" if q_count >= 2 else "single_block"
    active_block_type = "single"
    if isinstance(active, dict) and q_count >= 1:
        active_id = str(active.get("id") or "")
        active_index = None
        if active_id:
            for idx, question in enumerate(questions or []):
                if not isinstance(question, dict):
                    continue
                if str(question.get("id") or "") == active_id:
                    active_index = idx
                    break
        if active_index is not None and active_index < len(block_types):
            active_block_type = str(block_types[active_index] or "single")
        elif block_types:
            active_block_type = str(block_types[0] or "single")
    elif block_types:
        active_block_type = str(block_types[0] or "single")
    if active_block_type not in {"single", "multi", "dropdown", "dropdown_scroll", "text", "slider"}:
        active_block_type = "single"

    answer_types = ["single", "multi", "dropdown", "dropdown_scroll", "text", "slider"]
    block_prob = float(probs.get(active_block_type, 0.0))
    other_block_prob = 0.0
    for t in answer_types:
        if t == active_block_type:
            continue
        other_block_prob = max(other_block_prob, float(probs.get(t, 0.0)))
    block_margin = max(0.0, block_prob - other_block_prob)

    unknown_enabled = _env_flag("FULLBOT_QUIZ_TYPE_UNKNOWN_ENABLED", "1")
    min_conf = clamp_float(_env_float("FULLBOT_QUIZ_TYPE_MIN_CONF", 0.45), 0.0, 1.0)
    distinct_block_type_count = int(round(float(feature_values.get("distinct_block_type_count", 0.0) or 0.0)))
    aggregate_prompt_flags = evidence.get("aggregate_prompt_flags") if isinstance(evidence.get("aggregate_prompt_flags"), dict) else {}
    has_triple_prompt = bool(aggregate_prompt_flags.get("triple_hint")) or bool(artifact_signals.get("triple_hint"))
    has_mix_prompt = bool(aggregate_prompt_flags.get("mix_hint")) or bool(artifact_signals.get("mix_hint"))
    artifact_triple_token_count = int(artifact_signals.get("triple_token_count") or 0)
    derived_structural = ""
    structural_reason = ""
    if guardrails_enabled:
        if has_mix_prompt and q_count >= 1:
            derived_structural = "mixed"
            structural_reason = "derived_mix_prompt"
        elif has_triple_prompt and q_count == 3:
            derived_structural = "triple"
            structural_reason = "derived_triple_prompt"
        elif has_triple_prompt and artifact_triple_token_count >= 3 and q_count in {3, 4} and not has_mix_prompt:
            derived_structural = "triple"
            structural_reason = "derived_triple_artifact_count"
        elif has_triple_prompt and (q_count == 1 or q_count == 2) and artifact_triple_token_count >= 2:
            derived_structural = "triple"
            structural_reason = "derived_triple_artifact_markers"
    structural_types = {"triple", "mixed"}
    model_structural = top_label if top_label in structural_types else ""
    structural_label = derived_structural or model_structural
    structural_prob = float(probs.get(structural_label, 0.0)) if structural_label else 0.0
    use_structural = False
    structural_conf = structural_prob
    structural_margin = margin
    reason = "active_block_type"
    if structural_label and q_count >= 2:
        if derived_structural:
            use_structural = True
            structural_conf = max(structural_prob, 0.86 if structural_label == "triple" else 0.74)
            structural_margin = max(margin, structural_conf - block_prob)
            reason = structural_reason
        elif structural_prob >= max(min_conf, 0.40) or structural_prob >= (block_prob + 0.08):
            use_structural = True
            reason = "global_structure"
    elif structural_label == "triple" and derived_structural:
        use_structural = True
        structural_conf = max(structural_prob, 0.84)
        structural_margin = max(margin, structural_conf - block_prob)
        reason = structural_reason
    elif structural_label == "mixed" and derived_structural:
        use_structural = True
        structural_conf = max(structural_prob, 0.78)
        structural_margin = max(margin, structural_conf - block_prob)
        reason = structural_reason
    # Trust a strong model winner for base answer types before applying prompt/layout overrides.
    # This prevents heuristics from flipping confident predictions such as single<->multi.
    answer_type_set = {"single", "multi", "dropdown", "dropdown_scroll", "text", "slider"}
    trust_model_base = False
    trusted_base_label = ""
    trusted_base_conf = 0.0
    trusted_base_margin = 0.0
    model_trust_min_conf = clamp_float(_env_float("FULLBOT_QUIZ_TYPE_MODEL_TRUST_MIN_CONF", 0.68), 0.0, 1.0)
    model_trust_min_margin = clamp_float(_env_float("FULLBOT_QUIZ_TYPE_MODEL_TRUST_MIN_MARGIN", 0.12), 0.0, 1.0)
    if (
        not use_structural
        and top_label in answer_type_set
        and source == "hybrid_model"
        and top_prob >= model_trust_min_conf
        and margin >= model_trust_min_margin
    ):
        trust_model_base = True
        trusted_base_label = top_label
        trusted_base_conf = top_prob
        trusted_base_margin = margin
        reason = "trusted_model_top"
    no_structure_markers = (not has_triple_prompt) and (not has_mix_prompt)
    stable_dropdown_family = bool(block_types) and all(bt in {"dropdown", "dropdown_scroll", "text"} for bt in block_types)
    stable_choice_family = bool(block_types) and all(bt in {"single", "multi", "text"} for bt in block_types)
    active_options = (active or {}).get("options") if isinstance((active or {}).get("options"), list) else []
    option_norms = {
        normalize_match_text(str((opt or {}).get("text") or ""))
        for opt in active_options
        if isinstance(opt, dict)
    }
    dropdown_trigger_like = any(
        token in option_norms
        for token in {"szukaj", "szukaj.", "expand", "rozwin", "rozwin."}
    )
    base_override = ""
    base_override_conf = 0.0
    base_override_margin = 0.0
    if not use_structural and not trust_model_base and no_structure_markers:
        if (
            prompt_prefers_multi(prompt_text, prompt_multi=_PROMPT_MULTI)
            and active_block_type in {"single"}
            and not bool(artifact_signals.get("scroll_hint"))
        ):
            base_override = "multi"
            base_override_conf = max(float(probs.get("multi", 0.0)), 0.76)
            base_override_margin = max(margin, base_override_conf - max(block_prob, float(probs.get("dropdown", 0.0))))
            reason = "prompt_multi_override"
        elif prompt_prefers_single(prompt_text) and active_block_type in {"multi"} and not bool(artifact_signals.get("scroll_hint")):
            base_override = "single"
            base_override_conf = max(float(probs.get("single", 0.0)), 0.76)
            base_override_margin = max(margin, base_override_conf - max(block_prob, float(probs.get("multi", 0.0))))
            reason = "prompt_single_override"
        elif (
            active_block_type != "text"
            and (not bool(feature_values.get("text_hint", 0.0)))
            and (bool(prompt_flags.get("slider_hint")) or bool(artifact_signals.get("slider_hint")))
        ):
            base_override = "slider"
            base_override_conf = max(float(probs.get("slider", 0.0)), 0.82)
            base_override_margin = max(margin, base_override_conf - max(block_prob, float(probs.get("dropdown", 0.0))))
            reason = "artifact_slider_prompt"
        elif bool(artifact_signals.get("scroll_hint")) and q_count <= 2:
            base_override = "dropdown_scroll"
            base_override_conf = max(float(probs.get("dropdown_scroll", 0.0)), 0.80)
            base_override_margin = max(margin, base_override_conf - max(block_prob, float(probs.get("dropdown", 0.0))))
            reason = "artifact_scroll_prompt"
        elif active_block_type in {"dropdown", "dropdown_scroll"} and q_count <= 4 and stable_dropdown_family:
            prefer_scroll_family = (
                active_block_type == "dropdown_scroll"
                or (q_count >= 3 and float(probs.get("dropdown_scroll", 0.0)) >= float(probs.get("dropdown", 0.0)) + 0.08)
                or (
                    bool(feature_values.get("scroll_needed", 0.0))
                    and float(probs.get("dropdown_scroll", 0.0)) >= float(probs.get("dropdown", 0.0))
                )
            )
            base_override = "dropdown_scroll" if prefer_scroll_family else "dropdown"
            base_override_conf = max(block_prob, 0.78 if base_override == "dropdown_scroll" else 0.72)
            base_override_margin = max(block_margin, base_override_conf - other_block_prob)
            reason = "dropdown_family_override"
        elif active_block_type in {"single", "multi", "text"} and q_count <= 4 and stable_choice_family:
            base_override = active_block_type
            base_override_conf = max(block_prob, 0.72 if active_block_type != "text" else 0.70)
            base_override_margin = max(block_margin, base_override_conf - other_block_prob)
            reason = "single_block_override"
    detected = structural_label if use_structural else (trusted_base_label if trust_model_base else (base_override or active_block_type))
    detected_conf = structural_conf if use_structural else (trusted_base_conf if trust_model_base else (base_override_conf if base_override else block_prob))
    detected_margin = structural_margin if use_structural else (trusted_base_margin if trust_model_base else (base_override_margin if base_override else block_margin))
    if active_block_type == "slider" and detected == "slider":
        detected_conf = max(detected_conf, float(probs.get("slider", 0.0)), 0.74)
        detected_margin = max(detected_margin, margin)
        reason = "slider_family_floor"
    if active_block_type == "slider" and detected == "unknown":
        detected = "slider"
        detected_conf = max(float(probs.get("slider", 0.0)), 0.74)
        detected_margin = max(margin, detected_conf - other_block_prob)
        reason = "active_slider_floor"
    unknown_prob = 0.0
    if unknown_enabled and (not base_override) and detected_conf < min_conf and not (q_count == 1 and top_label == active_block_type):
        detected = "unknown"
        unknown_prob = max(0.0, min(1.0, 1.0 - detected_conf))
        reason = "low_conf_unknown"

    type_probs: Dict[str, float] = {c: round(float(probs.get(c, 0.0)), 6) for c in QUIZ_TYPES}
    type_probs["unknown"] = round(float(unknown_prob), 6)

    detected_op = "choice"
    if detected in {"triple", "mixed"}:
        if active_block_type == "text":
            detected_op = "text"
        elif active_block_type in {"dropdown", "dropdown_scroll"}:
            detected_op = "dropdown"
        else:
            detected_op = "choice"
    elif detected == "text":
        detected_op = "text"
    elif detected == "slider":
        detected_op = "slider"
    elif detected in {"dropdown", "dropdown_scroll"}:
        detected_op = "dropdown"
    elif detected == "unknown":
        detected_op = "unknown"

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
            "artifact_prompt_flags": {
                "triple_hint": bool(artifact_signals.get("triple_hint")),
                "triple_token_count": int(artifact_signals.get("triple_token_count") or 0),
                "mix_hint": bool(artifact_signals.get("mix_hint")),
                "slider_hint": bool(artifact_signals.get("slider_hint")),
                "scroll_hint": bool(artifact_signals.get("scroll_hint")),
                "multi_hint": bool(artifact_signals.get("multi_hint")),
            },
            "rule": reason,
            "guardrails_enabled": bool(guardrails_enabled),
            "screen_size": [int(screen_w), int(screen_h)],
        },
        "block_types": block_types,
        "quiz_type_features": {k: round(float(v), 6) for k, v in feature_values.items()},
        "question_split": question_split,
        "parse_signature_v2": parse_sig,
    }
