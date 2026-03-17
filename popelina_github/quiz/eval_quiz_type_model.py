from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from train_quiz_type_model import CLASS_TO_IDX, CLASSES, FEATURE_NAMES, _extract_features, _read_manifest, _resolve


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.clip(np.sum(e, axis=1, keepdims=True), 1e-12, None)


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
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    model_path = Path(args.model).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = _load_json(model_path)
    classes = model.get("classes") if isinstance(model.get("classes"), list) else CLASSES
    cls_to_idx = {str(c): i for i, c in enumerate(classes)}
    weights = model.get("weights") if isinstance(model.get("weights"), dict) else {}
    bias = model.get("bias") if isinstance(model.get("bias"), dict) else {}
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
        for fn in FEATURE_NAMES:
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

    W = np.zeros((len(FEATURE_NAMES), k), dtype=np.float64)
    b = np.zeros((k,), dtype=np.float64)
    for fi, fn in enumerate(FEATURE_NAMES):
        roww = weights.get(fn) if isinstance(weights.get(fn), dict) else {}
        for cn, idx in cls_to_idx.items():
            W[fi, idx] = float(roww.get(cn, 0.0) or 0.0)
    for cn, idx in cls_to_idx.items():
        b[idx] = float(bias.get(cn, 0.0) or 0.0)

    probs = _softmax(X @ W + b)
    pred = np.argmax(probs, axis=1)
    acc = float(np.mean(pred == y))

    per_class: Dict[str, Dict[str, float]] = {}
    for cn, idx in cls_to_idx.items():
        mask = y == idx
        n = int(np.sum(mask))
        if n <= 0:
            per_class[cn] = {"n": 0, "acc": 0.0}
            continue
        per_class[cn] = {"n": n, "acc": float(np.mean(pred[mask] == y[mask]))}

    out = {
        "rows_total": len(rows),
        "rows_used": int(X.shape[0]),
        "rows_skipped": int(skipped),
        "accuracy": acc,
        "per_class": per_class,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
