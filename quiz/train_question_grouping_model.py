from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


FEATURE_NAMES: List[str] = [
    "prompt_y",
    "item_y",
    "dy",
    "dx",
    "vertical_gap_norm",
    "center_dx_norm",
    "x_overlap_ratio",
    "iou_prompt",
    "item_width_norm",
    "item_height_norm",
    "prompt_item_text_sim",
    "item_has_frame",
    "item_conf",
    "item_text_len",
    "item_word_count",
    "item_is_below_prompt",
    "item_is_far_below",
    "item_has_digit",
    "item_has_scroll_hint",
    "item_has_expand_hint",
    "item_has_input_hint",
]

CATEGORICAL_FEATURES: Dict[str, List[str]] = {
    "prompt_control_kind": ["choice", "dropdown", "text", "slider", "unknown"],
    "item_candidate_type": ["question_prompt", "answer_option", "dropdown_trigger", "next_button", "page_header", "unknown"],
    "item_role_pred": ["question", "answer", "next", "noise", "unknown"],
}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        raw = line.strip()
        if not raw:
            continue
        obj = json.loads(raw)
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _as_float(v: Any) -> float:
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    try:
        return float(v)
    except Exception:
        return 0.0


def _norm_token(v: Any) -> str:
    return str(v or "").strip().lower() or "unknown"


def _build_feature_names() -> List[str]:
    out = list(FEATURE_NAMES)
    for field, cats in CATEGORICAL_FEATURES.items():
        for cat in cats:
            out.append(f"{field}__{cat}")
    return out


def _extract_features(row: Dict[str, Any], feature_names: Sequence[str]) -> Dict[str, float]:
    out = {name: _as_float(row.get(name)) for name in FEATURE_NAMES}
    for field, cats in CATEGORICAL_FEATURES.items():
        token = _norm_token(row.get(field))
        for cat in cats:
            out[f"{field}__{cat}"] = 1.0 if token == cat else 0.0
    for name in feature_names:
        out.setdefault(name, 0.0)
    return out


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _split(rows: Sequence[Dict[str, Any]], val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    pos = [i for i, r in enumerate(rows) if int(r.get("label") or 0) == 1]
    neg = [i for i, r in enumerate(rows) if int(r.get("label") or 0) == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    def split_one(xs: List[int]) -> Tuple[List[int], List[int]]:
        if len(xs) <= 1:
            return xs, []
        n_val = max(1, int(round(len(xs) * val_ratio)))
        if n_val >= len(xs):
            n_val = len(xs) - 1
        return xs[n_val:], xs[:n_val]
    train_p, val_p = split_one(pos)
    train_n, val_n = split_one(neg)
    train = sorted(train_p + train_n)
    val = sorted(val_p + val_n)
    return train, val


def train_mlp(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, hidden: int, epochs: int, lr: float, seed: int, pos_weight: float) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    rng = np.random.default_rng(seed)
    in_dim = X_train.shape[1]
    params = {
        "W1": rng.normal(0.0, 0.12, size=(in_dim, hidden)),
        "b1": np.zeros((1, hidden), dtype=np.float64),
        "W2": rng.normal(0.0, 0.12, size=(hidden, 1)),
        "b2": np.zeros((1, 1), dtype=np.float64),
    }
    best = {k: v.copy() for k, v in params.items()}
    best_acc = -1.0
    wait = 0
    patience = 60
    sample_w = np.where(y_train.reshape(-1, 1) > 0.5, pos_weight, 1.0)
    for _ in range(max(1, epochs)):
        z1 = X_train @ params["W1"] + params["b1"]
        h1 = _relu(z1)
        logits = h1 @ params["W2"] + params["b2"]
        probs = _sigmoid(logits)
        dlogits = (probs - y_train.reshape(-1, 1)) * sample_w / max(1, X_train.shape[0])
        dW2 = h1.T @ dlogits
        db2 = np.sum(dlogits, axis=0, keepdims=True)
        dh1 = dlogits @ params["W2"].T
        dz1 = dh1 * (z1 > 0.0)
        dW1 = X_train.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)
        params["W1"] -= lr * dW1
        params["b1"] -= lr * db1
        params["W2"] -= lr * dW2
        params["b2"] -= lr * db2
        val_probs = _sigmoid(_relu(X_val @ params["W1"] + params["b1"]) @ params["W2"] + params["b2"]).reshape(-1)
        val_pred = (val_probs >= 0.5).astype(np.int64)
        acc = float(np.mean(val_pred == y_val)) if y_val.size else 0.0
        if acc > best_acc + 1e-9:
            best = {k: v.copy() for k, v in params.items()}
            best_acc = acc
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    return best, {"val_acc": float(best_acc)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Train question grouping model from prompt-item pairs.")
    ap.add_argument("--data-jsonl", required=True)
    ap.add_argument("--out-model", required=True)
    ap.add_argument("--out-report", required=True)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--val-ratio", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    rows = _read_jsonl(Path(args.data_jsonl).resolve())
    if len(rows) < 10:
        print("[ERROR] not enough rows")
        return 2
    feature_names = _build_feature_names()
    feats = [_extract_features(r, feature_names) for r in rows]
    X = np.asarray([[float(f.get(name, 0.0)) for name in feature_names] for f in feats], dtype=np.float64)
    y = np.asarray([int(r.get("label") or 0) for r in rows], dtype=np.int64)
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma = np.where(sigma < 1e-6, 1.0, sigma)
    Xn = (X - mu) / sigma
    train_idx, val_idx = _split(rows, args.val_ratio, args.seed)
    X_train, y_train = Xn[train_idx], y[train_idx]
    X_val, y_val = Xn[val_idx], y[val_idx]
    pos = float(max(1, int(np.sum(y_train == 1))))
    neg = float(max(1, int(np.sum(y_train == 0))))
    pos_weight = neg / pos
    candidates = [
        (int(args.hidden), float(args.lr)),
        (max(16, int(args.hidden * 2)), float(args.lr * 0.5)),
        (max(24, int(args.hidden * 3)), float(args.lr * 0.35)),
    ]
    best_params = None
    best = None
    sweep: List[Dict[str, float]] = []
    for hidden, lr in candidates:
        params, metrics = train_mlp(X_train, y_train, X_val, y_val, hidden, args.epochs, lr, args.seed, pos_weight)
        probs = _sigmoid(_relu(X_val @ params["W1"] + params["b1"]) @ params["W2"] + params["b2"]).reshape(-1)
        pred = (probs >= 0.5).astype(np.int64)
        acc = float(np.mean(pred == y_val))
        tp = int(np.sum((pred == 1) & (y_val == 1)))
        fp = int(np.sum((pred == 1) & (y_val == 0)))
        fn = int(np.sum((pred == 0) & (y_val == 1)))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        sweep.append({"hidden": hidden, "lr": lr, "val_accuracy": acc, "val_precision": prec, "val_recall": rec, "val_f1": f1})
        if best is None or f1 > best["val_f1"] + 1e-9 or (abs(f1 - best["val_f1"]) <= 1e-9 and acc > best["val_accuracy"]):
            best = {"val_accuracy": acc, "val_precision": prec, "val_recall": rec, "val_f1": f1, "hidden": hidden, "lr": lr}
            best_params = params
    assert best_params is not None and best is not None
    model = {
        "model_type": "mlp_binary",
        "classes": ["not_same_question", "same_question"],
        "feature_names": feature_names,
        "hidden": int(best["hidden"]),
        "norm_mean": mu.tolist(),
        "norm_std": sigma.tolist(),
        "W1": best_params["W1"].tolist(),
        "b1": best_params["b1"].reshape(-1).tolist(),
        "W2": best_params["W2"].tolist(),
        "b2": best_params["b2"].reshape(-1).tolist(),
        "training_meta": {
            "samples": len(rows),
            "train_samples": int(len(train_idx)),
            "val_samples": int(len(val_idx)),
            "val_accuracy": best["val_accuracy"],
            "val_precision": best["val_precision"],
            "val_recall": best["val_recall"],
            "val_f1": best["val_f1"],
            "seed": int(args.seed),
            "hidden": int(best["hidden"]),
            "epochs": int(args.epochs),
            "lr": float(best["lr"]),
        },
    }
    report = {
        "data_jsonl": str(Path(args.data_jsonl).resolve()),
        "samples": len(rows),
        "train_samples": len(train_idx),
        "val_samples": len(val_idx),
        "feature_names": feature_names,
        "metrics": best,
        "sweep": sweep,
    }
    out_model = Path(args.out_model).resolve()
    out_report = Path(args.out_report).resolve()
    out_model.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_model.write_text(json.dumps(model, ensure_ascii=False), encoding="utf-8")
    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out_model": str(out_model), "out_report": str(out_report), "metrics": best}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
