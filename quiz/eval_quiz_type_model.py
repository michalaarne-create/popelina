from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from train_quiz_type_model import CLASSES, FEATURE_NAMES, _extract_features, _read_manifest, _resolve


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.clip(np.sum(e, axis=1, keepdims=True), 1e-12, None)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate quiz_type linear model on manifest.")
    parser.add_argument(
        "--manifest",
        type=str,
        default="",
        required=True,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        required=True,
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default="",
        help="Optional path for full JSON report.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    model_path = Path(args.model).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = _load_json(model_path)
    classes = model.get("classes") if isinstance(model.get("classes"), list) else CLASSES
    feature_names = model.get("feature_names") if isinstance(model.get("feature_names"), list) else FEATURE_NAMES
    cls_to_idx = {str(c): i for i, c in enumerate(classes)}
    weights = model.get("weights") if isinstance(model.get("weights"), dict) else {}
    bias = model.get("bias") if isinstance(model.get("bias"), dict) else {}
    model_type = str(model.get("model_type") or "linear").strip().lower()
    norm = model.get("normalization") if isinstance(model.get("normalization"), dict) else {}
    mean = norm.get("mean") if isinstance(norm.get("mean"), dict) else {}
    std = norm.get("std") if isinstance(norm.get("std"), dict) else {}

    rows = _read_manifest(manifest_path)
    X_rows: List[List[float]] = []
    y_rows: List[int] = []
    skipped = 0

    for row in rows:
        gt = str(row.get("expected_global_type") or "").strip()
        if gt not in cls_to_idx:
            skipped += 1
            continue
        lp = str(row.get("label_path") or "").strip()
        if not lp:
            skipped += 1
            continue
        label_path = _resolve(lp, manifest_path)
        if not label_path.exists():
            skipped += 1
            continue
        try:
            label = _load_json(label_path)
        except Exception:
            skipped += 1
            continue
        feats = _extract_features(label)
        fv: List[float] = []
        for fn in feature_names:
            x = float(feats.get(fn, 0.0))
            m = float(mean.get(fn, 0.0) or 0.0)
            s = float(std.get(fn, 1.0) or 1.0)
            if abs(s) < 1e-9:
                s = 1.0
            fv.append((x - m) / s)
        X_rows.append(fv)
        y_rows.append(cls_to_idx[gt])

    if not X_rows:
        raise RuntimeError("No valid eval rows.")

    X = np.asarray(X_rows, dtype=np.float64)
    y = np.asarray(y_rows, dtype=np.int64)
    k = len(classes)

    if model_type == "mlp":
        layers = model.get("layers") if isinstance(model.get("layers"), dict) else {}
        W1 = np.asarray(layers.get("W1") or [], dtype=np.float64)
        b1 = np.asarray(layers.get("b1") or [], dtype=np.float64)
        if "W2h" in layers:
            W2h = np.asarray(layers.get("W2h") or [], dtype=np.float64)
            b2h = np.asarray(layers.get("b2h") or [], dtype=np.float64)
            W3 = np.asarray(layers.get("W3") or [], dtype=np.float64)
            b3 = np.asarray(layers.get("b3") or [], dtype=np.float64)
            # Hidden layer 1
            a1 = _relu(X @ W1 + b1)
            # Hidden layer 2
            a2 = _relu(a1 @ W2h + b2h)
            # Output layer
            probs = _softmax(a2 @ W3 + b3)
        else:
            W2 = np.asarray(layers.get("W2") or [], dtype=np.float64)
            b2 = np.asarray(layers.get("b2") or [], dtype=np.float64)
            probs = _softmax(_relu(X @ W1 + b1) @ W2 + b2)
    else:
        W = np.zeros((len(feature_names), k), dtype=np.float64)
        b = np.zeros((k,), dtype=np.float64)
        for fi, fn in enumerate(feature_names):
            roww = weights.get(fn) if isinstance(weights.get(fn), dict) else {}
            for cn, idx in cls_to_idx.items():
                W[fi, idx] = float(roww.get(cn, 0.0) or 0.0)
        for cn, idx in cls_to_idx.items():
            b[idx] = float(bias.get(cn, 0.0) or 0.0)
        probs = _softmax(X @ W + b)
    pred = np.argmax(probs, axis=1)
    acc = float(np.mean(pred == y))
    idx_to_cls = {idx: cls for cls, idx in cls_to_idx.items()}

    per_class: Dict[str, Dict[str, float]] = {}
    confusion: Dict[str, Dict[str, int]] = defaultdict(dict)
    confusion_counter: Counter[tuple[str, str]] = Counter()
    for cn, idx in cls_to_idx.items():
        mask = y == idx
        n = int(np.sum(mask))
        if n <= 0:
            per_class[cn] = {"n": 0, "acc": 0.0}
            continue
        per_class[cn] = {"n": n, "acc": float(np.mean(pred[mask] == y[mask]))}
        pred_for_cls = pred[mask]
        row_conf: Dict[str, int] = {}
        for pred_idx, count in Counter(int(v) for v in pred_for_cls).items():
            pred_name = idx_to_cls.get(pred_idx, str(pred_idx))
            row_conf[pred_name] = int(count)
            confusion_counter[(cn, pred_name)] += int(count)
        confusion[cn] = row_conf

    top_errors: List[Dict[str, Any]] = []
    for (expected, predicted), count in confusion_counter.most_common():
        if expected == predicted:
            continue
        top_errors.append({"expected": expected, "predicted": predicted, "count": int(count)})
        if len(top_errors) >= 15:
            break

    out = {
        "rows_total": len(rows),
        "rows_used": int(X.shape[0]),
        "rows_skipped": int(skipped),
        "accuracy": acc,
        "model_type": model_type,
        "feature_count": int(len(feature_names)),
        "per_class": per_class,
        "confusion": confusion,
        "top_errors": top_errors,
    }
    if args.out_json:
        out_path = Path(args.out_json).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
