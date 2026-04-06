from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .quiz_utils import normalize_match_text, next_like, question_like


ROLES: List[str] = ["question", "answer", "next", "noise"]
FEATURE_NAMES: List[str] = [
    "x_center",
    "y_center",
    "w_ratio",
    "h_ratio",
    "area_ratio",
    "text_len",
    "word_count",
    "has_question_mark",
    "has_colon",
    "is_empty_text",
    "next_keyword",
    "question_keyword",
    "answer_keyword",
    "has_digit",
    "is_top_quarter",
    "is_bottom_quarter",
    "is_first_in_view",
    "is_last_in_view",
    "y_rank_norm",
    "prev_gap_norm",
    "next_gap_norm",
    "wide_box",
    "short_text",
    "long_text",
]
FEATURE_NAMES_V2: List[str] = FEATURE_NAMES + [
    "prev_text_len",
    "next_text_len",
    "prev_has_digit",
    "next_has_digit",
    "prev_has_question_mark",
    "next_has_question_mark",
    "y_cluster_rank_norm",
    "is_cta_keyword",
    "cta_keyword_count",
    "alpha_ratio",
    "digit_ratio",
]

_MODEL_CACHE: Dict[str, Dict[str, Any]] = {}


def _env_flag(name: str, default: str = "1") -> bool:
    raw = str(os.environ.get(name, default) or default).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _model_default_path() -> Path:
    root = Path(__file__).resolve().parents[3]
    return root / "data" / "models" / "element_role_classifier_v1.json"


def _model_v2_default_path() -> Path:
    root = Path(__file__).resolve().parents[3]
    return root / "data" / "models" / "element_role_classifier_v2.json"


def _model_rollout_contract_path() -> Path:
    root = Path(__file__).resolve().parents[3]
    return root / "data" / "models" / "element_role_classifier_rollout.json"


def _resolve_rollout_model_path() -> Path:
    explicit_contract = str(os.environ.get("FULLBOT_ELEMENT_ROLE_MODEL_ROLLOUT_PATH", "") or "").strip()
    contract_path = Path(explicit_contract) if explicit_contract else _model_rollout_contract_path()
    try:
        payload = json.loads(contract_path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return _model_v2_default_path()
    if not isinstance(payload, dict):
        return _model_v2_default_path()
    selected_path = str(payload.get("selected_model_path") or "").strip()
    return Path(selected_path) if selected_path else _model_v2_default_path()


def build_explicit_model_selector() -> Dict[str, Any]:
    selector = str(os.environ.get("FULLBOT_ELEMENT_ROLE_MODEL_SELECTOR", "") or "").strip().lower()
    explicit_path = str(os.environ.get("FULLBOT_ELEMENT_ROLE_MODEL_PATH", "") or "").strip()
    selected_variant = "v2"
    selected_path = _model_v2_default_path()
    source = "default_selector"
    if selector in {"v1", "legacy"}:
        selected_variant = "v1"
        selected_path = _model_default_path()
        source = "selector_env"
    elif selector in {"v2", "mlp_v2"}:
        selected_variant = "v2"
        selected_path = _model_v2_default_path()
        source = "selector_env"
    elif selector in {"rollout", "latest"}:
        selected_variant = "rollout"
        selected_path = _resolve_rollout_model_path()
        source = "selector_env"
    elif selector in {"path", "custom"}:
        selected_variant = "custom_path"
        selected_path = Path(explicit_path) if explicit_path else _model_v2_default_path()
        source = "selector_env"
    elif explicit_path:
        selected_variant = "legacy_path"
        selected_path = Path(explicit_path)
        source = "legacy_path_env"
    return {
        "selected_variant": selected_variant,
        "selected_path": str(selected_path),
        "source": source,
        "selector": selector or "default_v2",
        "is_explicit": source == "selector_env",
    }


def _load_model() -> Optional[Dict[str, Any]]:
    if not _env_flag("FULLBOT_ELEMENT_ROLE_MODEL_ENABLED", "1"):
        return None
    selector = build_explicit_model_selector()
    path = Path(str(selector.get("selected_path") or "").strip() or str(_model_v2_default_path()))
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


def _relu_vec(values: List[float]) -> List[float]:
    return [v if v > 0.0 else 0.0 for v in values]


def _matvec(inp: Sequence[float], W: Sequence[Sequence[float]], b: Sequence[float]) -> List[float]:
    out: List[float] = []
    for j in range(len(b)):
        s = float(b[j])
        for i, x in enumerate(inp):
            s += float(x) * float(W[i][j])
        out.append(s)
    return out


def _softmax(vals: Sequence[float]) -> List[float]:
    if not vals:
        return []
    m = max(float(v) for v in vals)
    exps = [math.exp(float(v) - m) for v in vals]
    total = sum(exps)
    if total <= 0.0:
        return [1.0 / float(len(vals)) for _ in vals]
    return [v / total for v in exps]


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _apply_activation(values: List[float], name: str) -> List[float]:
    act = str(name or "").strip().lower()
    if act in {"relu", ""}:
        return _relu_vec(values)
    if act == "tanh":
        return [math.tanh(v) for v in values]
    if act == "sigmoid":
        return [_sigmoid(v) for v in values]
    return values


def _heuristic_logits(item: Dict[str, Any], screen_h: int) -> Dict[str, float]:
    text = str(item.get("text") or "")
    norm = normalize_match_text(text)
    bbox = item.get("bbox") if isinstance(item.get("bbox"), list) else [0, 0, 0, 0]
    y_mid = float(bbox[1] + bbox[3]) * 0.5
    h = max(1.0, float(bbox[3] - bbox[1]))
    w = max(1.0, float(bbox[2] - bbox[0]))
    candidate_type = str(item.get("candidate_type") or "")
    out = {r: 0.0 for r in ROLES}
    if candidate_type == "page_header" or norm in {"home", "docs", "help"}:
        out["noise"] += 2.2
    if next_like(text) or candidate_type == "next_button":
        out["next"] += 2.6
    if question_like(text) or ("?" in text) or text.strip().endswith(":") or candidate_type == "question_prompt":
        out["question"] += 2.2
    if candidate_type == "answer_option":
        out["answer"] += 1.4
    if y_mid >= float(screen_h) * 0.72 and len(norm.split()) <= 3 and h <= 80:
        out["next"] += 0.8
    if len(norm.split()) <= 4 and ("?" not in text) and (":" not in text) and y_mid > float(screen_h) * 0.18:
        out["answer"] += 0.7
    if len(norm.split()) >= 6 and y_mid <= float(screen_h) * 0.55:
        out["question"] += 0.6
    if w < 80 and len(norm) <= 2:
        out["noise"] += 0.9
    return out


def _sort_key(item: Dict[str, Any]) -> Tuple[float, float]:
    bb = item.get("bbox") if isinstance(item.get("bbox"), list) else [0, 0, 0, 0]
    return (float(bb[1]), float(bb[0]))


def _extract_features(item: Dict[str, Any], ordered: List[Dict[str, Any]], idx: int, screen_w: int, screen_h: int) -> Dict[str, float]:
    bbox = item.get("bbox") if isinstance(item.get("bbox"), list) else [0, 0, 1, 1]
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    vw = max(1.0, float(screen_w))
    vh = max(1.0, float(screen_h))
    text = str(item.get("text") or "").strip()
    norm = normalize_match_text(text)
    words = [w for w in norm.split(" ") if w]
    next_keywords = ("next", "dalej", "nastep", "continue", "submit", "send", "finish")
    question_keywords = (
        "wybierz",
        "zaznacz",
        "wpisz",
        "choose",
        "select",
        "type",
        "copy",
        "przepisz",
        "jaki",
        "jaka",
        "jakie",
        "ile",
        "ktory",
        "ktora",
        "ktore",
    )
    answer_keywords = ("tak", "nie", "true", "false", "yes", "no")
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    n = max(1, len(ordered))
    prev_gap = 0.0
    next_gap = 0.0
    if idx > 0:
        pb = ordered[idx - 1].get("bbox") if isinstance(ordered[idx - 1].get("bbox"), list) else None
        if isinstance(pb, list) and len(pb) == 4:
            prev_gap = max(0.0, y1 - float(pb[3]))
    if idx + 1 < n:
        nb = ordered[idx + 1].get("bbox") if isinstance(ordered[idx + 1].get("bbox"), list) else None
        if isinstance(nb, list) and len(nb) == 4:
            next_gap = max(0.0, float(nb[1]) - y2)
    prev_text = str(ordered[idx - 1].get("text") or "").strip() if idx > 0 else ""
    next_text = str(ordered[idx + 1].get("text") or "").strip() if idx + 1 < n else ""
    cta_keywords = ("next", "dalej", "continue", "submit", "finish", "send", "done", "start", "go")
    cta_count = sum(1 for k in cta_keywords if k in norm)
    alpha_count = sum(1 for ch in text if ch.isalpha())
    digit_count = sum(1 for ch in text if ch.isdigit())
    text_len_raw = max(1, len(text))
    cluster_prev = 1.0 if prev_gap <= max(18.0, vh * 0.018) and idx > 0 else 0.0
    cluster_next = 1.0 if next_gap <= max(18.0, vh * 0.018) and idx + 1 < n else 0.0
    y_cluster_rank_norm = cluster_prev / max(1.0, cluster_prev + cluster_next)
    return {
        "x_center": ((x1 + x2) * 0.5) / vw,
        "y_center": ((y1 + y2) * 0.5) / vh,
        "w_ratio": w / vw,
        "h_ratio": h / vh,
        "area_ratio": (w * h) / max(1.0, vw * vh),
        "text_len": min(200.0, float(len(norm))) / 200.0,
        "word_count": min(32.0, float(len(words))) / 32.0,
        "has_question_mark": 1.0 if "?" in text else 0.0,
        "has_colon": 1.0 if ":" in text else 0.0,
        "is_empty_text": 1.0 if not norm else 0.0,
        "next_keyword": 1.0 if (next_like(text) or any(k in norm for k in next_keywords)) else 0.0,
        "question_keyword": 1.0 if (question_like(text) or any(k in norm for k in question_keywords)) else 0.0,
        "answer_keyword": 1.0 if any(k in norm for k in answer_keywords) else 0.0,
        "has_digit": 1.0 if any(ch.isdigit() for ch in text) else 0.0,
        "is_top_quarter": 1.0 if ((y1 + y2) * 0.5) <= (vh * 0.25) else 0.0,
        "is_bottom_quarter": 1.0 if ((y1 + y2) * 0.5) >= (vh * 0.75) else 0.0,
        "is_first_in_view": 1.0 if idx == 0 else 0.0,
        "is_last_in_view": 1.0 if idx == n - 1 else 0.0,
        "y_rank_norm": float(idx) / max(1.0, float(n - 1)) if n > 1 else 0.0,
        "prev_gap_norm": min(1.0, prev_gap / max(24.0, vh * 0.1)),
        "next_gap_norm": min(1.0, next_gap / max(24.0, vh * 0.1)),
        "wide_box": 1.0 if (w / vw) >= 0.28 else 0.0,
        "short_text": 1.0 if len(words) <= 3 else 0.0,
        "long_text": 1.0 if len(words) >= 7 else 0.0,
        "prev_text_len": min(200.0, float(len(normalize_match_text(prev_text)))) / 200.0,
        "next_text_len": min(200.0, float(len(normalize_match_text(next_text)))) / 200.0,
        "prev_has_digit": 1.0 if any(ch.isdigit() for ch in prev_text) else 0.0,
        "next_has_digit": 1.0 if any(ch.isdigit() for ch in next_text) else 0.0,
        "prev_has_question_mark": 1.0 if "?" in prev_text else 0.0,
        "next_has_question_mark": 1.0 if "?" in next_text else 0.0,
        "y_cluster_rank_norm": y_cluster_rank_norm,
        "is_cta_keyword": 1.0 if cta_count > 0 else 0.0,
        "cta_keyword_count": min(4.0, float(cta_count)) / 4.0,
        "alpha_ratio": float(alpha_count) / float(text_len_raw),
        "digit_ratio": float(digit_count) / float(text_len_raw),
    }


def _apply_guardrails(
    *,
    item: Dict[str, Any],
    screen_h: int,
    role_probs: Dict[str, float],
) -> Dict[str, float]:
    if not _env_flag("FULLBOT_ELEMENT_ROLE_GUARDRAILS", "1"):
        return role_probs
    out = dict(role_probs)
    text = str(item.get("text") or "")
    norm = normalize_match_text(text)
    bbox = item.get("bbox") if isinstance(item.get("bbox"), list) else [0, 0, 0, 0]
    y_mid = (float(bbox[1]) + float(bbox[3])) * 0.5
    cta = any(k in norm for k in ("next", "dalej", "continue", "submit", "finish", "send", "done"))
    if cta and y_mid >= float(screen_h) * 0.68:
        out["next"] = float(out.get("next", 0.0)) + 0.25
    if (("?" in text) or text.strip().endswith(":")) and len(norm.split()) >= 4:
        out["question"] = float(out.get("question", 0.0)) + 0.15
    s = sum(max(0.0, float(v)) for v in out.values())
    if s <= 1e-9:
        return role_probs
    for k in list(out.keys()):
        out[k] = max(0.0, float(out[k])) / s
    return out


def _infer_logits_mlp_v2(xn: Sequence[float], model: Dict[str, Any]) -> Optional[List[float]]:
    layers = model.get("layers")
    if not isinstance(layers, list) or not layers:
        return None
    cur: List[float] = [float(v) for v in xn]
    for layer in layers:
        if not isinstance(layer, dict):
            return None
        W = layer.get("W")
        b = layer.get("b")
        if not (isinstance(W, list) and isinstance(b, list)):
            return None
        out = _matvec(cur, W, b)
        act = str(layer.get("activation") or "").strip().lower()
        if act and act != "linear":
            out = _apply_activation(out, act)
        cur = out
    return cur


def classify_element_roles(items: Sequence[Dict[str, Any]], screen_w: int, screen_h: int) -> List[Dict[str, Any]]:
    model = _load_model()
    ordered = [dict(item) for item in sorted(items, key=_sort_key)]
    if not model:
        return ordered
    try:
        model_type = str(model.get("model_type") or "linear_softmax")
        mu = [float(v) for v in model.get("mu") or []]
        sigma = [float(v) for v in model.get("sigma") or []]
        W = model.get("W") or []
        b = [float(v) for v in model.get("b") or []]
        W1 = model.get("W1") or []
        b1 = [float(v) for v in model.get("b1") or []]
        W2 = model.get("W2") or []
        b2 = [float(v) for v in model.get("b2") or []]
        W3 = model.get("W3") or []
        b3 = [float(v) for v in model.get("b3") or []]
        roles = model.get("roles") if isinstance(model.get("roles"), list) else ROLES
        feature_names = model.get("feature_names") if isinstance(model.get("feature_names"), list) else (
            FEATURE_NAMES_V2 if str(model.get("model_type") or "").strip().lower() == "mlp_v2" else FEATURE_NAMES
        )
        calibration = model.get("calibration") if isinstance(model.get("calibration"), dict) else {}
        temperature = float(calibration.get("temperature", 1.0) or 1.0)
        fallback_conf_threshold = float(calibration.get("fallback_conf_threshold", 0.60) or 0.60)
    except Exception:
        return ordered
    if not (mu and sigma and roles and feature_names):
        return ordered

    for idx, item in enumerate(ordered):
        feats = _extract_features(item, ordered, idx, screen_w, screen_h)
        vec = [float(feats.get(str(name), 0.0)) for name in feature_names]
        xn = [((vec[i] - mu[i]) / (sigma[i] if abs(sigma[i]) > 1e-8 else 1.0)) for i in range(min(len(vec), len(mu)))]
        if model_type == "linear_softmax" and W and b:
            logits = _matvec(xn, W, b)
        elif model_type == "mlp_v2":
            logits = _infer_logits_mlp_v2(xn, model)
            if not isinstance(logits, list):
                continue
        elif W1 and W2 and W3:
            z1 = _matvec(xn, W1, b1)
            a1 = _relu_vec(z1)
            z2 = _matvec(a1, W2, b2)
            a2 = _relu_vec(z2)
            logits = _matvec(a2, W3, b3)
        else:
            continue
        heur = _heuristic_logits(item, screen_h)
        for ridx, role in enumerate(roles):
            logits[ridx] = float(logits[ridx]) + float(heur.get(str(role), 0.0))
        t = max(0.2, min(5.0, temperature))
        probs = _softmax([float(v) / t for v in logits])
        role_probs = {str(roles[i]): float(probs[i]) for i in range(min(len(roles), len(probs)))}
        role_probs = _apply_guardrails(item=item, screen_h=screen_h, role_probs=role_probs)
        top_role = max(role_probs.items(), key=lambda kv: kv[1])[0] if role_probs else "noise"
        top_conf = float(role_probs.get(top_role, 0.0))
        if top_conf < fallback_conf_threshold:
            fallback = _heuristic_logits(item, screen_h)
            for r in roles:
                role_probs[str(r)] = 0.65 * float(role_probs.get(str(r), 0.0)) + 0.35 * _sigmoid(float(fallback.get(str(r), 0.0)))
            s = sum(max(0.0, float(v)) for v in role_probs.values())
            if s > 1e-9:
                for r in list(role_probs.keys()):
                    role_probs[r] = max(0.0, float(role_probs[r])) / s
            top_role = max(role_probs.items(), key=lambda kv: kv[1])[0] if role_probs else "noise"
            top_conf = float(role_probs.get(top_role, 0.0))
        item["role_pred"] = top_role
        item["role_conf"] = top_conf
        item["role_probs"] = role_probs
        item["role_features"] = feats
    return ordered
