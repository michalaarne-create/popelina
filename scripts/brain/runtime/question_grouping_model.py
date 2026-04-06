from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .quiz_utils import box_center, box_height, box_iou, box_width, normalize_match_text, text_similarity


_MODEL_CACHE: Dict[str, Dict[str, Any]] = {}


def _env_flag(name: str, default: str = "1") -> bool:
    raw = str(os.environ.get(name, default) or default).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _default_model_path() -> Path:
    root = Path(__file__).resolve().parents[3]
    return root / "data" / "models" / "question_grouping_classifier_v2.json"


def _load_model() -> Optional[Dict[str, Any]]:
    if not _env_flag("FULLBOT_QUESTION_GROUPING_MODEL_ENABLED", "1"):
        return None
    path = Path(str(os.environ.get("FULLBOT_QUESTION_GROUPING_MODEL_PATH", "") or "").strip() or str(_default_model_path()))
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


def _as_float(v: Any) -> float:
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    try:
        return float(v)
    except Exception:
        return 0.0


def _norm_token(v: Any) -> str:
    return str(v or "").strip().lower() or "unknown"


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _relu(x: float) -> float:
    return x if x > 0.0 else 0.0


def _feature_dict(prompt: Dict[str, Any], item: Dict[str, Any], screen_h: int) -> Dict[str, float]:
    pbox = prompt.get("bbox") or [0, 0, 1, 1]
    ibox = item.get("bbox") or [0, 0, 1, 1]
    pcx, _ = box_center(pbox)
    icx, _ = box_center(ibox)
    dy = float(ibox[1] - pbox[3])
    dx = float(icx - pcx)
    overlap_x = max(0.0, min(float(pbox[2]), float(ibox[2])) - max(float(pbox[0]), float(ibox[0])))
    overlap_x_ratio = overlap_x / float(max(1.0, min(box_width(pbox), box_width(ibox))))
    text = str(item.get("text") or "")
    prompt_text = str(prompt.get("text") or prompt.get("prompt_text") or "")
    norm = normalize_match_text(text)
    pnorm = normalize_match_text(prompt_text)
    out = {
        "prompt_y": float(pbox[1]),
        "item_y": float(ibox[1]),
        "dy": dy,
        "dx": dx,
        "vertical_gap_norm": dy / float(max(1, screen_h)),
        "center_dx_norm": dx / float(max(1.0, box_width(pbox))),
        "x_overlap_ratio": overlap_x_ratio,
        "iou_prompt": box_iou(pbox, ibox),
        "item_width_norm": box_width(ibox) / float(max(1.0, box_width(pbox))),
        "item_height_norm": box_height(ibox) / float(max(1.0, box_height(pbox))),
        "prompt_item_text_sim": text_similarity(pnorm, norm),
        "item_has_frame": 1.0 if item.get("has_frame") else 0.0,
        "item_conf": float(item.get("conf") or 0.0),
        "item_text_len": float(len(norm)),
        "item_word_count": float(len([w for w in norm.split() if w])),
        "item_is_below_prompt": 1.0 if ibox[1] >= pbox[3] else 0.0,
        "item_is_far_below": 1.0 if dy > max(90, box_height(pbox) * 2.2) else 0.0,
        "item_has_digit": 1.0 if any(ch.isdigit() for ch in norm) else 0.0,
        "item_has_scroll_hint": 1.0 if any(tok in norm for tok in ("scroll", "przewin", "lista")) else 0.0,
        "item_has_expand_hint": 1.0 if "expand" in norm or "rozwin" in norm or "rozwiń" in norm else 0.0,
        "item_has_input_hint": 1.0 if any(tok in norm for tok in ("wpisz", "type", "input", "odpowiedz", "odpowiedź")) else 0.0,
    }
    for field, cats in {
        "prompt_control_kind": ["choice", "dropdown", "text", "slider", "unknown"],
        "item_candidate_type": ["question_prompt", "answer_option", "dropdown_trigger", "next_button", "page_header", "unknown"],
        "item_role_pred": ["question", "answer", "next", "noise", "unknown"],
    }.items():
        token = _norm_token(prompt.get("control_kind")) if field == "prompt_control_kind" else _norm_token(item.get(field.replace("item_", "")))
        for cat in cats:
            out[f"{field}__{cat}"] = 1.0 if token == cat else 0.0
    return out


def score_item_for_prompt(prompt: Dict[str, Any], item: Dict[str, Any], screen_h: int) -> Optional[float]:
    model = _load_model()
    if not model:
        return None
    feature_names = model.get("feature_names") if isinstance(model.get("feature_names"), list) else []
    if not feature_names:
        return None
    f = _feature_dict(prompt, item, screen_h)
    mu = model.get("norm_mean") if isinstance(model.get("norm_mean"), list) else []
    sigma = model.get("norm_std") if isinstance(model.get("norm_std"), list) else []
    x: List[float] = []
    for idx, name in enumerate(feature_names):
        val = float(f.get(name, 0.0))
        m = float(mu[idx]) if idx < len(mu) else 0.0
        s = float(sigma[idx]) if idx < len(sigma) else 1.0
        if abs(s) < 1e-9:
            s = 1.0
        x.append((val - m) / s)
    W1 = model.get("W1") if isinstance(model.get("W1"), list) else []
    b1 = model.get("b1") if isinstance(model.get("b1"), list) else []
    W2 = model.get("W2") if isinstance(model.get("W2"), list) else []
    b2 = model.get("b2") if isinstance(model.get("b2"), list) else []
    h: List[float] = []
    for j in range(len(b1)):
        acc = float(b1[j] or 0.0)
        for i, xv in enumerate(x):
            try:
                acc += xv * float(W1[i][j])
            except Exception:
                continue
        h.append(_relu(acc))
    logit = float(b2[0] or 0.0) if b2 else 0.0
    for j, hv in enumerate(h):
        try:
            logit += hv * float(W2[j][0])
        except Exception:
            continue
    return _sigmoid(logit)
