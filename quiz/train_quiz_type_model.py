from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CLASSES: List[str] = [
    "single",
    "multi",
    "dropdown",
    "dropdown_scroll",
    "text",
    "triple",
    "mixed",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
FEATURE_NAMES: List[str] = [
    "question_count",
    "has_next",
    "has_select",
    "has_input",
    "option_count",
    "vertical_regularity",
    "scroll_needed",
    "multi_hint",
    "scroll_hint",
    "triple_hint",
    "mix_hint",
    "text_hint",
    "marker_total",
    "marker_circle",
    "marker_square",
    "marker_unknown",
    "marker_mean_conf",
]


def _try_tqdm(total: int, desc: str):
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm(total=total, desc=desc, ncols=100)
    except Exception:
        return None


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def _read_manifest(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            raw = line.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            if isinstance(obj, dict):
                rows.append(obj)
        return rows
    payload = _load_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        return [r for r in payload["rows"] if isinstance(r, dict)]
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    return []


def _resolve(path: str, manifest_path: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (manifest_path.parent.parent / p).resolve() if "labels/" in path.replace("\\", "/") else (manifest_path.parent / p).resolve()


def _extract_features(label: Dict[str, Any]) -> Dict[str, float]:
    blocks = label.get("blocks") if isinstance(label.get("blocks"), list) else []
    block_types = [str(b.get("type") or "") for b in blocks if isinstance(b, dict)]
    has_next = bool(label.get("has_next"))
    require_scroll = bool(label.get("require_scroll"))
    option_count = 0
    for b in blocks:
        if not isinstance(b, dict):
            continue
        option_count += len(b.get("options") or [])

    has_select = any(t in {"dropdown", "dropdown_scroll"} for t in block_types)
    has_input = any(t == "text" for t in block_types)
    multi_hint = any(t == "multi" for t in block_types)
    scroll_hint = any(t == "dropdown_scroll" for t in block_types) or require_scroll
    triple_hint = bool(label.get("global_type") == "triple")
    mix_hint = bool(label.get("global_type") == "mixed")
    text_hint = any(t == "text" for t in block_types)

    if block_types:
        if all(t == "single" for t in block_types):
            marker_circle = 1.0
            marker_square = 0.0
        elif all(t == "multi" for t in block_types):
            marker_circle = 0.0
            marker_square = 1.0
        else:
            marker_circle = 0.45
            marker_square = 0.45
    else:
        marker_circle = 0.0
        marker_square = 0.0
    marker_total = 1.0 if any(t in {"single", "multi"} for t in block_types) else 0.0
    marker_unknown = max(0.0, 1.0 - marker_circle - marker_square) if marker_total > 0 else 0.0
    marker_mean_conf = 0.9 if marker_total > 0 else 0.0

    f = {
        "question_count": float(len(blocks) if blocks else 1),
        "has_next": 1.0 if has_next else 0.0,
        "has_select": 1.0 if has_select else 0.0,
        "has_input": 1.0 if has_input else 0.0,
        "option_count": float(option_count),
        "vertical_regularity": 1.0 if option_count >= 2 else 0.0,
        "scroll_needed": 1.0 if require_scroll else 0.0,
        "multi_hint": 1.0 if multi_hint else 0.0,
        "scroll_hint": 1.0 if scroll_hint else 0.0,
        "triple_hint": 1.0 if triple_hint else 0.0,
        "mix_hint": 1.0 if mix_hint else 0.0,
        "text_hint": 1.0 if text_hint else 0.0,
        "marker_total": marker_total,
        "marker_circle": marker_circle,
        "marker_square": marker_square,
        "marker_unknown": marker_unknown,
        "marker_mean_conf": marker_mean_conf,
    }
    return f


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
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n, d = X_train.shape
    k = len(CLASSES)
    W = rng.normal(loc=0.0, scale=0.01, size=(d, k))
    b = np.zeros((k,), dtype=np.float64)

    pbar = _try_tqdm(epochs, "train")
    for ep in range(1, epochs + 1):
        logits = X_train @ W + b
        probs = _softmax(logits)
        y_oh = _one_hot(y_train, k)
        diff = (probs - y_oh) / max(1, n)
        grad_W = X_train.T @ diff + l2 * W
        grad_b = np.sum(diff, axis=0)
        W -= lr * grad_W
        b -= lr * grad_b

        tr_pred = np.argmax(probs, axis=1)
        tr_acc = _acc(tr_pred, y_train)
        val_probs = _softmax(X_val @ W + b) if X_val.size else np.zeros((0, k))
        val_pred = np.argmax(val_probs, axis=1) if X_val.size else np.array([], dtype=np.int64)
        val_acc = _acc(val_pred, y_val) if X_val.size else 0.0

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
    return W, b


def main() -> None:
    parser = argparse.ArgumentParser(description="Train quiz_type linear model from generated dataset labels.")
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(ROOT / "data" / "benchmarks" / "random_www_quiz_100k" / "manifests" / "dataset_manifest.jsonl"),
    )
    parser.add_argument(
        "--out-model",
        type=str,
        default=str(ROOT / "data" / "models" / "quiz_type_classifier_v1.json"),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260313)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    rows = _read_manifest(manifest_path)
    rng = random.Random(int(args.seed))
    rng.shuffle(rows)

    X_rows: List[List[float]] = []
    y_rows: List[int] = []
    skipped = 0
    for row in rows:
        gt = str(row.get("expected_global_type") or "").strip()
        if gt not in CLASS_TO_IDX:
            skipped += 1
            continue
        label_path_raw = str(row.get("label_path") or "").strip()
        if not label_path_raw:
            skipped += 1
            continue
        label_path = _resolve(label_path_raw, manifest_path)
        if not label_path.exists():
            skipped += 1
            continue
        try:
            label = _load_json(label_path)
        except Exception:
            skipped += 1
            continue
        feats = _extract_features(label)
        X_rows.append([float(feats.get(n, 0.0)) for n in FEATURE_NAMES])
        y_rows.append(CLASS_TO_IDX[gt])

    if not X_rows:
        raise RuntimeError("No training rows after filtering.")

    X = np.asarray(X_rows, dtype=np.float64)
    y = np.asarray(y_rows, dtype=np.int64)
    n = X.shape[0]
    split = int(max(1, min(n - 1, round(n * (1.0 - float(args.val_ratio))))))
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    mu = np.mean(X_train, axis=0)
    sigma = np.std(X_train, axis=0)
    sigma = np.where(sigma < 1e-6, 1.0, sigma)
    X_train_n = (X_train - mu) / sigma
    X_val_n = (X_val - mu) / sigma if X_val.size else X_val

    print(
        json.dumps(
            {
                "rows_total": len(rows),
                "rows_used": int(n),
                "rows_skipped": int(skipped),
                "train_rows": int(X_train.shape[0]),
                "val_rows": int(X_val.shape[0]),
                "epochs": int(args.epochs),
            },
            ensure_ascii=False,
        )
    )

    W, b = train(
        X_train=X_train_n,
        y_train=y_train,
        X_val=X_val_n,
        y_val=y_val,
        epochs=int(args.epochs),
        lr=float(args.lr),
        l2=float(args.l2),
        seed=int(args.seed),
    )

    probs_val = _softmax(X_val_n @ W + b) if X_val_n.size else np.zeros((0, len(CLASSES)))
    val_pred = np.argmax(probs_val, axis=1) if X_val_n.size else np.array([], dtype=np.int64)
    val_acc = _acc(val_pred, y_val) if X_val_n.size else 0.0

    weights: Dict[str, Dict[str, float]] = {}
    for fi, fn in enumerate(FEATURE_NAMES):
        weights[fn] = {cn: float(W[fi, ci]) for ci, cn in enumerate(CLASSES)}
    bias = {cn: float(b[ci]) for ci, cn in enumerate(CLASSES)}

    out_payload = {
        "version": "v1",
        "description": "Linear softmax model trained from random_www_quiz labels.",
        "classes": CLASSES,
        "feature_names": FEATURE_NAMES,
        "normalization": {
            "mean": {fn: float(mu[i]) for i, fn in enumerate(FEATURE_NAMES)},
            "std": {fn: float(sigma[i]) for i, fn in enumerate(FEATURE_NAMES)},
        },
        "bias": bias,
        "weights": weights,
        "meta": {
            "train_rows": int(X_train.shape[0]),
            "val_rows": int(X_val.shape[0]),
            "val_acc": float(val_acc),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "l2": float(args.l2),
            "seed": int(args.seed),
            "manifest": str(manifest_path),
        },
    }

    out_path = Path(args.out_model).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out_model": str(out_path), "val_acc": float(val_acc)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
