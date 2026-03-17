from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from train_element_role_model import FEATURE_NAMES, ROLE_TO_IDX, ROLES, _extract_features, _read_jsonl, _row_sort_key


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.clip(np.sum(e, axis=1, keepdims=True), 1e-12, None)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def _vectorize(rows: List[Dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate element role model with per-class precision/recall/F1.")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    model_path = Path(args.model).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    rows = _read_jsonl(manifest_path)
    X, y = _vectorize(rows)
    if X.size == 0:
        raise RuntimeError("No valid eval rows.")

    model = _load_json(model_path)
    mu = np.asarray(model.get("mu") or [], dtype=np.float64).reshape(1, -1)
    sigma = np.asarray(model.get("sigma") or [], dtype=np.float64).reshape(1, -1)
    sigma = np.where(np.abs(sigma) < 1e-8, 1.0, sigma)
    W = np.asarray(model.get("W") or [], dtype=np.float64)
    b = np.asarray(model.get("b") or [], dtype=np.float64)
    Xn = (X - mu) / sigma
    probs = _softmax(Xn @ W + b)
    pred = np.argmax(probs, axis=1)

    confusion = np.zeros((len(ROLES), len(ROLES)), dtype=np.int64)
    for yt, yp in zip(y, pred):
        confusion[int(yt), int(yp)] += 1

    per_class: Dict[str, Dict[str, float]] = {}
    for idx, role in enumerate(ROLES):
        tp = float(confusion[idx, idx])
        fp = float(np.sum(confusion[:, idx]) - tp)
        fn = float(np.sum(confusion[idx, :]) - tp)
        prec = tp / max(1.0, tp + fp)
        rec = tp / max(1.0, tp + fn)
        f1 = 0.0 if (prec + rec) <= 0.0 else (2.0 * prec * rec / (prec + rec))
        per_class[role] = {
            "support": int(np.sum(confusion[idx, :])),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
        }

    out = {
        "rows_used": int(X.shape[0]),
        "accuracy": float(np.mean(pred == y)),
        "per_class": per_class,
        "confusion": confusion.tolist(),
        "roles": ROLES,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
