from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ROLES: List[str] = ["question", "answer", "next", "noise"]
ROLE_TO_IDX = {r: i for i, r in enumerate(ROLES)}
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


def _try_tqdm(total: int, desc: str):
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm(total=total, desc=desc, ncols=100)
    except Exception:
        return None


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        raw = line.strip()
        if not raw:
            continue
        obj = json.loads(raw)
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _norm_text(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.clip(np.sum(e, axis=1, keepdims=True), 1e-12, None)


def _one_hot(y: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros((y.shape[0], k), dtype=np.float64)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def _acc(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(pred == y)) if y.size else 0.0


def _row_sort_key(row: Dict[str, Any]) -> Tuple[float, float]:
    bbox = row.get("bbox") if isinstance(row.get("bbox"), list) else [0, 0, 0, 0]
    y1 = float(bbox[1]) if len(bbox) >= 2 else 0.0
    x1 = float(bbox[0]) if len(bbox) >= 1 else 0.0
    return (y1, x1)


def _extract_features(row: Dict[str, Any], view_rows_sorted: List[Dict[str, Any]], idx: int) -> Dict[str, float]:
    bbox = row.get("bbox") if isinstance(row.get("bbox"), list) else [0, 0, 1, 1]
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    viewport = row.get("viewport") if isinstance(row.get("viewport"), dict) else {}
    vw = max(1.0, float(viewport.get("width") or 1.0))
    vh = max(1.0, float(viewport.get("height") or 1.0))
    text = _norm_text(str(row.get("text") or ""))
    words = [w for w in text.split(" ") if w]
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
    n = max(1, len(view_rows_sorted))
    prev_gap = 0.0
    next_gap = 0.0
    if idx > 0:
        pb = view_rows_sorted[idx - 1].get("bbox") if isinstance(view_rows_sorted[idx - 1].get("bbox"), list) else None
        if isinstance(pb, list) and len(pb) == 4:
            prev_gap = max(0.0, y1 - float(pb[3]))
    if idx + 1 < n:
        nb = view_rows_sorted[idx + 1].get("bbox") if isinstance(view_rows_sorted[idx + 1].get("bbox"), list) else None
        if isinstance(nb, list) and len(nb) == 4:
            next_gap = max(0.0, float(nb[1]) - y2)

    return {
        "x_center": ((x1 + x2) * 0.5) / vw,
        "y_center": ((y1 + y2) * 0.5) / vh,
        "w_ratio": w / vw,
        "h_ratio": h / vh,
        "area_ratio": (w * h) / max(1.0, vw * vh),
        "text_len": min(200.0, float(len(text))) / 200.0,
        "word_count": min(32.0, float(len(words))) / 32.0,
        "has_question_mark": 1.0 if "?" in text else 0.0,
        "has_colon": 1.0 if ":" in text else 0.0,
        "is_empty_text": 1.0 if not text else 0.0,
        "next_keyword": 1.0 if any(k in text for k in next_keywords) else 0.0,
        "question_keyword": 1.0 if any(k in text for k in question_keywords) else 0.0,
        "answer_keyword": 1.0 if any(k in text for k in answer_keywords) else 0.0,
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
    }


def _vectorize(rows: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    by_view: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_view.setdefault(str(row.get("view_id") or "__none__"), []).append(row)

    X_rows: List[List[float]] = []
    y_rows: List[int] = []
    for view_rows in by_view.values():
        ordered = sorted(view_rows, key=_row_sort_key)
        for idx, row in enumerate(ordered):
            role = str(row.get("role") or "").strip()
            if role not in ROLE_TO_IDX:
                continue
            feats = _extract_features(row, ordered, idx)
            X_rows.append([float(feats.get(n, 0.0)) for n in FEATURE_NAMES])
            y_rows.append(ROLE_TO_IDX[role])
    return np.asarray(X_rows, dtype=np.float64), np.asarray(y_rows, dtype=np.int64)


def train(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    lr: float,
    l2: float,
    seed: int,
    hidden1: int,
    hidden2: int,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n, d = X_train.shape
    k = len(ROLES)
    W = rng.normal(0.0, 0.02, size=(d, k))
    b = np.zeros((k,), dtype=np.float64)

    counts = np.bincount(y_train, minlength=k).astype(np.float64)
    class_w = np.where(counts > 0.0, np.sqrt(counts.sum() / np.maximum(1.0, counts)), 1.0)
    class_w = class_w / np.mean(class_w)
    sample_w = class_w[y_train].reshape(-1, 1)

    pbar = _try_tqdm(epochs, "train")
    for ep in range(1, epochs + 1):
        logits = X_train @ W + b
        probs = _softmax(logits)
        y_oh = _one_hot(y_train, k)
        diff = (probs - y_oh) * sample_w / max(1, n)
        gW = X_train.T @ diff + l2 * W
        gb = np.sum(diff, axis=0)
        W -= lr * gW
        b -= lr * gb

        tr_pred = np.argmax(probs, axis=1)
        tr_acc = _acc(tr_pred, y_train)
        if X_val.size:
            vprobs = _softmax(X_val @ W + b)
            vpred = np.argmax(vprobs, axis=1)
            val_acc = _acc(vpred, y_val)
        else:
            val_acc = 0.0

        if pbar is not None:
            pbar.set_postfix({"train_acc": f"{tr_acc:.4f}", "val_acc": f"{val_acc:.4f}"})
            pbar.update(1)
        else:
            bar_w = 28
            done = int(math.floor(ep * bar_w / max(1, epochs)))
            bar = "#" * done + "-" * (bar_w - done)
            print(f"[{bar}] {ep}/{epochs} train_acc={tr_acc:.4f} val_acc={val_acc:.4f}")
    if pbar is not None:
        pbar.close()

    return {"W": W, "b": b}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train runtime-ready element role MLP from element-role manifest.")
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(ROOT / "data" / "benchmarks" / "element_role_manifest.jsonl"),
    )
    parser.add_argument(
        "--out-model",
        type=str,
        default=str(ROOT / "data" / "models" / "element_role_classifier_v1.json"),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.18)
    parser.add_argument("--l2", type=float, default=5e-5)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260314)
    parser.add_argument("--hidden1", type=int, default=48)
    parser.add_argument("--hidden2", type=int, default=24)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    rows = _read_jsonl(manifest_path)
    rng = random.Random(int(args.seed))
    rng.shuffle(rows)
    X, y = _vectorize(rows)
    if X.size == 0:
        raise RuntimeError("No training rows after filtering.")

    mu = np.mean(X, axis=0, keepdims=True)
    sigma = np.std(X, axis=0, keepdims=True)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    Xn = (X - mu) / sigma

    n = Xn.shape[0]
    n_val = max(1, int(round(n * float(args.val_ratio)))) if n >= 2 else 0
    n_val = min(max(1, n_val), n - 1) if n >= 2 else 0
    if n_val > 0:
        X_train = Xn[:-n_val]
        y_train = y[:-n_val]
        X_val = Xn[-n_val:]
        y_val = y[-n_val:]
    else:
        X_train, y_train = Xn, y
        X_val = np.zeros((0, Xn.shape[1]), dtype=np.float64)
        y_val = np.zeros((0,), dtype=np.int64)

    print(
        json.dumps(
            {
                "rows_total": int(n),
                "train_rows": int(X_train.shape[0]),
                "val_rows": int(X_val.shape[0]),
                "epochs": int(args.epochs),
                "roles": {role: int(np.sum(y == ROLE_TO_IDX[role])) for role in ROLES},
            },
            ensure_ascii=False,
        )
    )

    params = train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=int(args.epochs),
        lr=float(args.lr),
        l2=float(args.l2),
        seed=int(args.seed),
        hidden1=int(args.hidden1),
        hidden2=int(args.hidden2),
    )

    if X_val.size:
        vpred = np.argmax(_softmax(X_val @ params["W"] + params["b"]), axis=1)
        val_acc = _acc(vpred, y_val)
    else:
        val_acc = 0.0

    out_path = Path(args.out_model).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_type": "linear_softmax",
        "task": "element_role_classifier",
        "roles": ROLES,
        "feature_names": FEATURE_NAMES,
        "mu": mu.reshape(-1).tolist(),
        "sigma": sigma.reshape(-1).tolist(),
        "W": params["W"].tolist(),
        "b": params["b"].tolist(),
        "val_acc": float(val_acc),
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"out_model": str(out_path), "val_acc": float(val_acc)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
