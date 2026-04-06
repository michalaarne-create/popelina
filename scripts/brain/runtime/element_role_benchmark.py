from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.brain.runtime.element_role_classifier import ROLES, classify_element_roles


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    vals = sorted(float(v) for v in values)
    if len(vals) == 1:
        return vals[0]
    rank = max(0.0, min(1.0, q)) * (len(vals) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(vals) - 1)
    frac = rank - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _eval_manifest(path: Path) -> Dict[str, Any]:
    rows = _load_rows(path)
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        view_id = str(r.get("view_id") or r.get("sample_id") or "")
        if not view_id:
            continue
        groups[view_id].append(r)

    conf = Counter()
    lat: List[float] = []
    errors: List[Dict[str, str]] = []
    for view_id, grp in groups.items():
        try:
            vw = int((grp[0].get("viewport") or {}).get("width") or 1920)
            vh = int((grp[0].get("viewport") or {}).get("height") or 1080)
            ordered = sorted(grp, key=lambda x: (int((x.get("bbox") or [0, 0, 0, 0])[1]), int((x.get("bbox") or [0, 0, 0, 0])[0])))
            items = [{"text": str(x.get("text") or ""), "bbox": [int(v) for v in (x.get("bbox") or [0, 0, 0, 0])[:4]], "meta": x.get("attrs") or {}} for x in ordered]
            gt = [str(x.get("role") or "") for x in ordered]
            t0 = time.perf_counter()
            pred = classify_element_roles(items, vw, vh)
            lat.append((time.perf_counter() - t0) * 1000.0)
            for p, g in zip(pred, gt):
                conf[(g, str(p.get("role_pred") or ""))] += 1
        except Exception as exc:
            errors.append({"view_id": view_id, "error": str(exc)})

    total = sum(conf.values())
    correct = sum(n for (g, p), n in conf.items() if g == p)
    accuracy = (float(correct) / float(total)) if total > 0 else 0.0
    recall = {}
    precision = {}
    for role in ROLES:
        tp = float(conf[(role, role)])
        fn = float(sum(n for (g, _), n in conf.items() if g == role and _ != role))
        fp = float(sum(n for (g, p), n in conf.items() if p == role and g != role))
        recall[role] = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        precision[role] = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    confusion = []
    for gt in ROLES:
        row = {}
        for pred in ROLES:
            row[pred] = int(conf[(gt, pred)])
        confusion.append({"gt": gt, "row": row})

    return {
        "manifest": str(path),
        "views": len(groups),
        "items": total,
        "errors": errors,
        "metrics": {
            "accuracy": accuracy,
            "recall": recall,
            "precision": precision,
            "latency_ms_avg": float(statistics.mean(lat)) if lat else 0.0,
            "latency_ms_p95": _percentile(lat, 0.95),
        },
        "confusion": confusion,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Benchmark element role classifier.")
    p.add_argument("--manifest", action="append", required=True, help="Path to JSONL benchmark manifest; can be used multiple times.")
    p.add_argument("--model-path", default="", help="Optional FULLBOT_ELEMENT_ROLE_MODEL_PATH override.")
    p.add_argument("--target-accuracy", type=float, default=0.995)
    p.add_argument("--target-next-recall", type=float, default=0.99)
    p.add_argument("--target-p95-ms", type=float, default=5.0)
    p.add_argument("--enforce", action="store_true")
    p.add_argument("--out-json", default="")
    args = p.parse_args()

    if args.model_path:
        os.environ["FULLBOT_ELEMENT_ROLE_MODEL_PATH"] = str(Path(args.model_path).resolve())

    reports = []
    for manifest in args.manifest:
        path = Path(manifest).resolve()
        if not path.exists():
            print(f"[ERROR] manifest not found: {path}")
            return 2
        rep = _eval_manifest(path)
        reports.append(rep)
        m = rep["metrics"]
        print(
            f"[BENCH] {path.name} items={rep['items']} acc={m['accuracy']:.4f} "
            f"next_recall={m['recall'].get('next', 0.0):.4f} p95={m['latency_ms_p95']:.3f}ms"
        )

    agg_items = sum(int(r["items"]) for r in reports)
    agg_acc = 0.0
    agg_next = 0.0
    agg_p95 = max((float(r["metrics"]["latency_ms_p95"]) for r in reports), default=0.0)
    if agg_items > 0:
        agg_acc = sum(float(r["metrics"]["accuracy"]) * int(r["items"]) for r in reports) / float(agg_items)
        agg_next = sum(float(r["metrics"]["recall"]["next"]) * int(r["items"]) for r in reports) / float(agg_items)

    summary = {
        "reports": reports,
        "aggregate": {
            "items": agg_items,
            "accuracy": agg_acc,
            "next_recall": agg_next,
            "latency_ms_p95_max": agg_p95,
        },
        "targets": {
            "accuracy": float(args.target_accuracy),
            "next_recall": float(args.target_next_recall),
            "latency_ms_p95": float(args.target_p95_ms),
        },
    }
    print(
        f"[BENCH] aggregate items={agg_items} acc={agg_acc:.4f} "
        f"next_recall={agg_next:.4f} p95_max={agg_p95:.3f}ms"
    )

    if args.out_json:
        out = Path(args.out_json).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[BENCH] report={out}")

    failed = (
        agg_acc < float(args.target_accuracy)
        or agg_next < float(args.target_next_recall)
        or agg_p95 > float(args.target_p95_ms)
    )
    if args.enforce and failed:
        print("[BENCH] FAIL target gate not met.")
        return 1
    print("[BENCH] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
