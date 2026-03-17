from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.brain.runtime.element_role_classifier import FEATURE_NAMES_V2, ROLES, _extract_features


ROLE_TO_ID = {r: i for i, r in enumerate(ROLES)}


@dataclass
class Sample:
    view_id: str
    role: str
    text: str
    bbox: List[int]
    attrs: Dict[str, Any]
    viewport: Dict[str, Any]


class TinyMLP(nn.Module):
    def __init__(self, in_dim: int, hidden1: int, hidden2: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def _load_manifest_rows(paths: Sequence[Path]) -> List[Sample]:
    out: List[Sample] = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            role = str(obj.get("role") or "")
            bbox = obj.get("bbox") if isinstance(obj.get("bbox"), list) else []
            viewport = obj.get("viewport") if isinstance(obj.get("viewport"), dict) else {}
            if role not in ROLE_TO_ID or len(bbox) != 4 or not viewport:
                continue
            view_id = str(obj.get("view_id") or obj.get("sample_id") or "")
            if not view_id:
                continue
            out.append(
                Sample(
                    view_id=view_id,
                    role=role,
                    text=str(obj.get("text") or ""),
                    bbox=[int(v) for v in bbox[:4]],
                    attrs=obj.get("attrs") if isinstance(obj.get("attrs"), dict) else {},
                    viewport=viewport,
                )
            )
    return out


def _dedupe_and_qc(samples: Sequence[Sample]) -> Tuple[List[Sample], Dict[str, int]]:
    dedup: Dict[Tuple[str, str, Tuple[int, int, int, int], str], Sample] = {}
    counters = {"dropped_duplicates": 0, "dropped_outliers": 0}
    for s in samples:
        vw = max(1, int(s.viewport.get("width") or 1))
        vh = max(1, int(s.viewport.get("height") or 1))
        x1, y1, x2, y2 = s.bbox
        if x2 <= x1 or y2 <= y1:
            counters["dropped_outliers"] += 1
            continue
        if x1 < -5 or y1 < -5 or x2 > vw + 5 or y2 > vh + 5:
            counters["dropped_outliers"] += 1
            continue
        key = (s.view_id, s.role, tuple(s.bbox), s.text.strip())
        if key in dedup:
            counters["dropped_duplicates"] += 1
            continue
        dedup[key] = s
    return list(dedup.values()), counters


def _group_by_view(samples: Sequence[Sample]) -> Dict[str, List[Sample]]:
    g: Dict[str, List[Sample]] = defaultdict(list)
    for s in samples:
        g[s.view_id].append(s)
    return g


def _stratified_view_split(view_groups: Dict[str, List[Sample]], holdout_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    rng = random.Random(seed)
    role_bucket: Dict[str, List[str]] = defaultdict(list)
    for view_id, rows in view_groups.items():
        roles = [r.role for r in rows]
        dominant = max(set(roles), key=roles.count)
        role_bucket[dominant].append(view_id)
    train_views: List[str] = []
    holdout_views: List[str] = []
    for _, ids in role_bucket.items():
        ids = ids[:]
        rng.shuffle(ids)
        cut = max(1, int(round(len(ids) * holdout_ratio)))
        holdout_views.extend(ids[:cut])
        train_views.extend(ids[cut:])
    return train_views, holdout_views


def _build_arrays(samples: Sequence[Sample]) -> Tuple[np.ndarray, np.ndarray]:
    by_view = _group_by_view(samples)
    xs: List[List[float]] = []
    ys: List[int] = []
    for _, rows in by_view.items():
        ordered = sorted(rows, key=lambda s: (s.bbox[1], s.bbox[0]))
        items = [{"text": s.text, "bbox": s.bbox, "meta": s.attrs} for s in ordered]
        vw = int(ordered[0].viewport.get("width") or 1920)
        vh = int(ordered[0].viewport.get("height") or 1080)
        for i, s in enumerate(ordered):
            feats = _extract_features(items[i], items, i, vw, vh)
            xs.append([float(feats.get(name, 0.0)) for name in FEATURE_NAMES_V2])
            ys.append(ROLE_TO_ID[s.role])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int64)


def _confusion_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    n_cls = len(ROLES)
    cm = np.zeros((n_cls, n_cls), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    recalls = {}
    precisions = {}
    f1s = {}
    for i, role in enumerate(ROLES):
        tp = float(cm[i, i])
        fn = float(cm[i, :].sum() - tp)
        fp = float(cm[:, i].sum() - tp)
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        pre = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * pre * rec / (pre + rec)) if (pre + rec) > 0 else 0.0
        recalls[role] = rec
        precisions[role] = pre
        f1s[role] = f1
    macro_f1 = float(sum(f1s.values()) / float(max(1, len(f1s))))
    return {
        "accuracy": acc,
        "recall": recalls,
        "precision": precisions,
        "f1": f1s,
        "macro_f1": macro_f1,
        "confusion_matrix": cm.tolist(),
    }


def _fit_one(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    hidden1: int,
    hidden2: int,
    lr: float,
    weight_decay: float,
    epochs: int,
    seed: int,
) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = TinyMLP(x_train.shape[1], hidden1, hidden2, len(ROLES))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    cls_counts = np.bincount(y_train, minlength=len(ROLES)).astype(np.float32)
    cls_weights = (cls_counts.sum() / np.maximum(cls_counts, 1.0)).astype(np.float32)
    cls_weights = cls_weights / np.mean(cls_weights)
    weight_t = torch.tensor(cls_weights, dtype=torch.float32)
    xtr = torch.tensor(x_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.long)
    xva = torch.tensor(x_val, dtype=torch.float32)
    yva = torch.tensor(y_val, dtype=torch.long)

    best = {"score": -1.0, "metrics": None}
    best_state: Dict[str, torch.Tensor] = {}
    patience = 15
    stale = 0
    for _ in range(max(10, epochs)):
        model.train()
        logits = model(xtr)
        loss = F.cross_entropy(logits, ytr, weight=weight_t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            pred = model(xva).argmax(dim=1).cpu().numpy()
        m = _confusion_metrics(yva.cpu().numpy(), pred)
        score = min(m["recall"]["next"], m["accuracy"])
        if score > best["score"]:
            stale = 0
            best["score"] = score
            best["metrics"] = m
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= patience:
            break
    return best, best_state


def _select_temperature(logits: np.ndarray, y_true: np.ndarray) -> float:
    best_t = 1.0
    best_nll = 1e18
    y = y_true.astype(np.int64)
    for t in [0.6, 0.75, 0.9, 1.0, 1.15, 1.3, 1.5]:
        z = logits / float(t)
        z = z - np.max(z, axis=1, keepdims=True)
        p = np.exp(z)
        p /= np.maximum(1e-12, p.sum(axis=1, keepdims=True))
        nll = -np.log(np.maximum(1e-12, p[np.arange(len(y)), y])).mean()
        if nll < best_nll:
            best_nll = float(nll)
            best_t = float(t)
    return best_t


def main() -> int:
    parser = argparse.ArgumentParser(description="Train element role classifier v2 (tiny MLP).")
    parser.add_argument("--manifest", action="append", required=True, help="Input manifest JSONL; can be passed multiple times.")
    parser.add_argument("--holdout-ratio", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--out-model", required=True)
    parser.add_argument("--out-report", default="")
    args = parser.parse_args()

    manifest_paths = [Path(p).resolve() for p in args.manifest]
    raw = _load_manifest_rows(manifest_paths)
    clean, qc = _dedupe_and_qc(raw)
    groups = _group_by_view(clean)
    train_views, holdout_views = _stratified_view_split(groups, holdout_ratio=float(args.holdout_ratio), seed=int(args.seed))
    train_samples = [s for v in train_views for s in groups.get(v, [])]
    hold_samples = [s for v in holdout_views for s in groups.get(v, [])]

    x_train, y_train = _build_arrays(train_samples)
    x_hold, y_hold = _build_arrays(hold_samples)
    mu = x_train.mean(axis=0)
    sigma = x_train.std(axis=0)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    xn_train = (x_train - mu) / sigma
    xn_hold = (x_hold - mu) / sigma

    sweeps = [
        {"hidden1": 32, "hidden2": 16, "lr": 2e-3, "weight_decay": 5e-4},
        {"hidden1": 48, "hidden2": 24, "lr": 2e-3, "weight_decay": 1e-3},
        {"hidden1": 64, "hidden2": 32, "lr": 1.5e-3, "weight_decay": 8e-4},
    ]
    best_cfg = None
    best_metrics = None
    best_state = None
    for cfg in sweeps:
        info, state = _fit_one(
            xn_train,
            y_train,
            xn_hold,
            y_hold,
            hidden1=int(cfg["hidden1"]),
            hidden2=int(cfg["hidden2"]),
            lr=float(cfg["lr"]),
            weight_decay=float(cfg["weight_decay"]),
            epochs=int(args.epochs),
            seed=int(args.seed),
        )
        metrics = info["metrics"]
        if metrics is None:
            continue
        score = min(float(metrics["recall"]["next"]), float(metrics["accuracy"]))
        if (best_metrics is None) or (score > min(float(best_metrics["recall"]["next"]), float(best_metrics["accuracy"]))):
            best_cfg, best_metrics, best_state = cfg, metrics, state
    if best_cfg is None or best_metrics is None or best_state is None:
        print("[ERROR] training failed")
        return 2

    model = TinyMLP(xn_train.shape[1], int(best_cfg["hidden1"]), int(best_cfg["hidden2"]), len(ROLES))
    model.load_state_dict(best_state)
    with torch.no_grad():
        hold_logits = model(torch.tensor(xn_hold, dtype=torch.float32)).cpu().numpy()
    temperature = _select_temperature(hold_logits, y_hold)

    def _arr(v: torch.Tensor) -> List[Any]:
        x = v.detach().cpu().numpy()
        return x.tolist()

    payload = {
        "model_type": "mlp_v2",
        "feature_schema_version": "element_role_v2",
        "feature_names": FEATURE_NAMES_V2,
        "roles": ROLES,
        "mu": mu.astype(float).tolist(),
        "sigma": sigma.astype(float).tolist(),
        "layers": [
            {"W": _arr(best_state["fc1.weight"].T), "b": _arr(best_state["fc1.bias"]), "activation": "relu"},
            {"W": _arr(best_state["fc2.weight"].T), "b": _arr(best_state["fc2.bias"]), "activation": "relu"},
            {"W": _arr(best_state["fc3.weight"].T), "b": _arr(best_state["fc3.bias"]), "activation": "linear"},
        ],
        "calibration": {
            "temperature": float(temperature),
            "fallback_conf_threshold": 0.60,
        },
        "training_meta": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "seed": int(args.seed),
            "dataset_hash": hashlib.sha1(("||".join(str(p) for p in manifest_paths)).encode("utf-8")).hexdigest(),
            "metrics_holdout": best_metrics,
            "sweep_selected": best_cfg,
            "qc": qc,
            "n_train_views": len(train_views),
            "n_holdout_views": len(holdout_views),
            "n_train_items": int(len(y_train)),
            "n_holdout_items": int(len(y_hold)),
        },
    }

    out_model = Path(args.out_model).resolve()
    out_model.parent.mkdir(parents=True, exist_ok=True)
    out_model.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[TRAIN] model={out_model}")
    print(
        f"[TRAIN] holdout acc={best_metrics['accuracy']:.4f} "
        f"macro_f1={best_metrics['macro_f1']:.4f} next_recall={best_metrics['recall']['next']:.4f}"
    )

    if args.out_report:
        out_report = Path(args.out_report).resolve()
        out_report.parent.mkdir(parents=True, exist_ok=True)
        out_report.write_text(json.dumps(payload["training_meta"], ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[TRAIN] report={out_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
